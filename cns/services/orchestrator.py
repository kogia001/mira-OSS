"""
Main continuum orchestrator for CNS.

Coordinates all continuum processing: message handling, LLM interaction,
tool execution, working memory updates, and event publishing.

Optimized to generate embeddings once and propagate them to all services.
"""
import logging
import ast
import json
import os
import re
import contextvars
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from config import config
from cns.core.continuum import Continuum
from cns.core.events import (
    ContinuumEvent,
    TurnCompletedEvent
)
from clients.llm_provider import LLMProvider, ContextOverflowError
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
from cns.services.overflow_logger import get_overflow_logger
from utils.tag_parser import TagParser, match_memory_id

logger = logging.getLogger(__name__)


class ContinuumOrchestrator:
    """
    Main orchestration service for continuum processing.
    
    Coordinates the entire continuum flow from user input to final response,
    managing all system interactions through clean interfaces.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        continuum_repo,
        working_memory,
        tool_repo,
        tag_parser,
        fingerprint_generator,
        event_bus,
        memory_relevance_service,
        memory_evacuator=None
    ):
        """
        Initialize orchestrator with dependencies.

        All parameters are REQUIRED except memory_evacuator. The orchestrator will
        fail immediately if any required dependency is missing or used incorrectly.

        Args:
            llm_provider: LLM provider for generating responses (required)
            continuum_repo: Repository for message persistence (required)
            working_memory: Working memory system for prompt composition (required)
            tool_repo: Tool repository for tool definitions (required)
            tag_parser: Tag parser for response parsing (required)
            fingerprint_generator: Fingerprint generator for retrieval query expansion (required).
                                  Raises RuntimeError on generation failures - no degraded state.
            event_bus: Event bus for publishing/subscribing to events (required)
            memory_relevance_service: Memory relevance service for surfacing long-term memories (required).
                                     Raises exceptions on infrastructure failures - no degraded state.
            memory_evacuator: Memory evacuator for curating pinned memories under pressure (optional).
                             When provided, triggers evacuation when anchor count exceeds threshold.
        """
        self.llm_provider = llm_provider
        self.continuum_repo = continuum_repo
        self.working_memory = working_memory
        self.tool_repo = tool_repo
        self.tag_parser = tag_parser
        self.fingerprint_generator = fingerprint_generator
        self.memory_relevance_service = memory_relevance_service
        self.memory_evacuator = memory_evacuator
        self.event_bus = event_bus

        # Get singleton embeddings provider for generating embeddings once
        self.embeddings_provider = get_hybrid_embeddings_provider()

        # Store composed prompt sections when received via event
        self._cached_content = None
        self._non_cached_content = None
        self._notification_center = None

        # In-memory token tracking for context overflow detection
        # Tracks actual input tokens from previous turn for accurate estimation
        self._last_turn_usage: Dict[str, int] = {}  # {continuum_id: input_tokens}
        # One-shot context trim from async LLM judgment (future extension)
        self._pending_context_trim: Dict[str, int] = {}  # {continuum_id: trim_index}

        # Subscribe to system prompt composed event
        self.event_bus.subscribe('SystemPromptComposedEvent', self._handle_system_prompt_composed)

        logger.info("ContinuumOrchestrator initialized")

    def process_message(
        self,
        continuum: Continuum,
        user_message: Union[str, List[Dict[str, Any]]],
        system_prompt: str,
        stream: bool = False,
        stream_callback=None,
        _tried_loading_all_tools: bool = False,
        _internal_user_message: bool = False,
        unit_of_work=None,
        storage_content: Optional[Union[str, List[Dict[str, Any]]]] = None,
        segment_turn_number: int = 1,
    ) -> tuple[Continuum, str, Dict[str, Any]]:
        """
        Process user message through complete continuum flow.

        Args:
            continuum: Current continuum state
            user_message: User's input message (string or multimodal content array).
                         For images, this should be the inference tier (1200px).
            system_prompt: Base system prompt
            stream: Whether to stream response
            stream_callback: Callback for streaming chunks
            _tried_loading_all_tools: Internal flag to prevent infinite need_tool loops
            _internal_user_message: Internal flag for synthetic orchestration prompts.
                                  Internal user messages are used for inference only and
                                  are not persisted to history.
            unit_of_work: Optional UnitOfWork for batching persistence operations
            storage_content: Content for persistence (optional). For images, this should
                           be the storage tier (512px WebP). If not provided, user_message
                           is used for persistence.
            segment_turn_number: Turn number within current segment (1-indexed).
                               Incremented at API entry point for real user messages.

        Returns:
            Tuple of (updated_continuum, final_response, metadata)
        """
        # Initialize metadata collection
        metadata = {
            "tools_used": [],
            "referenced_memories": []
        }
        
        # Add user message to continuum cache (no persistence yet)
        user_msg_obj, user_events = continuum.add_user_message(user_message)
        self._publish_events(user_events)
        
        # Extract text content for weighted context (bypass for multimodal)
        # For multimodal content, we only use the text portion for embeddings
        text_for_context = user_message
        if isinstance(user_message, list):
            # Extract text from multimodal content array
            text_parts = [item['text'] for item in user_message if item.get('type') == 'text']
            text_for_context = ' '.join(text_parts) if text_parts else 'Image uploaded'

        # Get previous memories from trinket for retention evaluation
        previous_memories = self._get_previous_memories()

        # Evacuation checkpoint: curate pinned memories under pressure
        if self.memory_evacuator:
            anchor_count = len(previous_memories)
            if self.memory_evacuator.should_evacuate(previous_memories):
                logger.debug(
                    f"Memory evacuation triggered: {anchor_count} anchors > "
                    f"threshold {self.memory_evacuator.config.evacuation_trigger_threshold}"
                )
                previous_memories = self.memory_evacuator.evacuate(
                    memories=previous_memories,
                    continuum=continuum,
                    user_message=text_for_context
                )
                logger.debug(
                    f"Evacuation complete: {len(previous_memories)}/{anchor_count} anchors retained "
                    f"(target: {self.memory_evacuator.config.evacuation_target_count})"
                )

                # Update trinket cache with evacuated list
                trinket = self.working_memory.get_trinket('ProactiveMemoryTrinket')
                if trinket:
                    trinket._cached_memories = previous_memories
            else:
                logger.debug(
                    f"Evacuation check: {anchor_count} anchors <= "
                    f"threshold {self.memory_evacuator.config.evacuation_trigger_threshold}, no action"
                )

        # Generate fingerprint and evaluate retention of previous memories
        # The fingerprint expands fragmentary queries into retrieval-optimized specifics.
        # Retention evaluation uses LLM reasoning to decide which previous memories
        # should stay in context based on conversation trajectory.
        #
        # generate_fingerprint() raises RuntimeError on failure - no degraded state
        fingerprint, pinned_ids = self.fingerprint_generator.generate_fingerprint(
            continuum,
            text_for_context,
            previous_memories=previous_memories
        )

        # Apply retention to get pinned memories (filters by 8-char ID match)
        pinned_memories = self._apply_retention(previous_memories, pinned_ids)

        # Generate 768d embedding for the fingerprint (query encoding)
        fingerprint_embedding = self.embeddings_provider.encode_realtime(fingerprint)

        # Fresh retrieval using configured proactive cap
        # Memory service raises exceptions on infrastructure failures - no hedging
        retrieval_limit = config.lt_memory.proactive.max_memories
        fresh_memories = self.memory_relevance_service.get_relevant_memories(
            fingerprint=fingerprint,
            fingerprint_embedding=fingerprint_embedding,
            limit=retrieval_limit
        )

        # Merge pinned + fresh, deduplicating by memory ID
        # Pinned memories take precedence (appear first)
        surfaced_memories = self._merge_memories(pinned_memories, fresh_memories)

        # Log retrieval for quality evaluation
        from cns.services.retrieval_logger import get_retrieval_logger
        get_retrieval_logger().log_retrieval(
            continuum_id=continuum.id,
            raw_query=text_for_context,
            fingerprint=fingerprint,
            surfaced_memories=surfaced_memories
        )

        logger.info(
            f"Memory surfacing: {len(pinned_memories)} pinned + "
            f"{len(fresh_memories)} fresh = {len(surfaced_memories)} total"
        )

        # Send merged memories to ProactiveMemoryTrinket
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=str(continuum.id),
            target_trinket="ProactiveMemoryTrinket",
            context={"memories": surfaced_memories}
        ))

        # Now compose system prompt with all context ready
        cached_content, non_cached_content, notification_center = self._compose_prompt_sections(
            continuum_id=str(continuum.id),
            system_prompt=system_prompt
        )
        
        # Get available tools - only currently enabled tools
        # With invokeother_tool, the LLM can see all available tools in working memory
        # and load what it needs on demand
        available_tools = self.tool_repo.get_all_tool_definitions()
        
        # Build messages from continuum
        messages = continuum.get_messages_for_api()

        # Build structured system content with cache breakpoints
        system_blocks = self._build_system_blocks(cached_content, non_cached_content)

        # Build complete message array with notification center injection
        # Structure: SYSTEM -> CONVERSATION [cached] -> NOTIFICATION CENTER -> CURRENT USER
        if notification_center and messages:
            # Separate current user message from conversation history
            # The current user message was just added to continuum, so it's the last message
            current_user_msg = messages[-1]
            history_messages = messages[:-1]

            complete_messages = [
                {"role": "system", "content": system_blocks},
                *history_messages,
                {"role": "assistant", "content": "═══════════════════════════ HUD ════════════════════════════\n" + notification_center},
                current_user_msg,
            ]
            logger.debug(
                f"Injected notification center: {len(history_messages)} history msgs + "
                f"notification center ({len(notification_center)} chars)"
            )
        else:
            # No notification center or no messages - use original structure
            complete_messages = [{"role": "system", "content": system_blocks}] + messages

        # Initialize messages for LLM (may be modified by overflow remediation)
        messages_for_llm = complete_messages

        # Check for one-shot adjustment from previous async LLM judgment
        one_shot_trim = self._pending_context_trim.pop(str(continuum.id), None)
        if one_shot_trim:
            logger.info(f"Applying one-shot trim from async LLM judgment: {one_shot_trim} messages")
            messages_for_llm = messages_for_llm[:1] + messages_for_llm[one_shot_trim + 1:]

        # Process through streaming events API
        events = []
        response_text = ""
        raw_response = None
        invoked_tool_loader = False  # Track if invokeother_tool was called during this turn
        saw_code_execution = False

        # Apply tier-based model and thinking configuration
        from utils.user_context import get_user_preferences, resolve_tier, LLMProvider
        from clients.vault_client import get_api_key

        llm_kwargs = {}
        prefs = get_user_preferences()
        tier_config = resolve_tier(prefs.llm_tier)

        llm_kwargs['model_preference'] = tier_config.model
        if tier_config.thinking_budget == 0:
            llm_kwargs['thinking_enabled'] = False
        else:
            llm_kwargs['thinking_enabled'] = True
            llm_kwargs['thinking_budget'] = tier_config.thinking_budget

        # Provider routing for generic OpenAI-compatible endpoints (Groq, OpenRouter, etc.)
        if tier_config.provider == LLMProvider.GENERIC:
            llm_kwargs['endpoint_url'] = tier_config.endpoint_url
            llm_kwargs['model_override'] = tier_config.model
            if tier_config.api_key_name:
                llm_kwargs['api_key_override'] = get_api_key(tier_config.api_key_name)

        # Retrieve container_id from Valkey for multi-turn file persistence
        # Only pass container_id if code_execution tool is enabled (Anthropic requirement)
        from clients.valkey_client import get_valkey
        valkey = get_valkey()

        # Check if code_execution is in the tools list
        has_code_execution = any(
            tool.get("type") == "code_execution_20250825"
            for tool in available_tools
        )

        if has_code_execution:
            valkey_key = f"container:{continuum.id}"
            container_id = valkey.get(valkey_key)
            if container_id:
                llm_kwargs['container_id'] = container_id
                logger.info(f"📦 Reusing container from Valkey: {container_id}")
            else:
                logger.info("📦 No existing container - new container will be created")

        # Context overflow remediation loop
        max_overflow_retries = 3
        overflow_attempt = 0

        while overflow_attempt <= max_overflow_retries:
            # === PROACTIVE TOKEN CHECK ===
            last_input = self._last_turn_usage.get(str(continuum.id))
            estimated = self._estimate_request_tokens(messages_for_llm, available_tools, last_input)
            available_for_input = config.api.context_window_tokens - config.api.max_tokens

            if estimated > available_for_input:
                # Proactive overflow - go directly to remediation
                overflow_attempt += 1
                logger.warning(
                    f"Proactive context overflow detected: ~{estimated} tokens > {available_for_input} available "
                    f"(attempt {overflow_attempt}/{max_overflow_retries})"
                )
                if overflow_attempt > max_overflow_retries:
                    raise RuntimeError(
                        f"Request exceeds context window after {max_overflow_retries} remediation attempts. "
                        f"Estimated ~{estimated} tokens vs {available_for_input} available."
                    )
                # Apply remediation and continue loop
                messages_for_llm = self._apply_overflow_remediation(
                    overflow_attempt, messages_for_llm, complete_messages, continuum, text_for_context,
                    system_prompt=system_prompt,
                    estimated_tokens=estimated, event_type='proactive'
                )
                continue

            try:
                # Collect events from generator
                for event in self.llm_provider.stream_events(
                    messages=messages_for_llm,
                    tools=available_tools,
                    **llm_kwargs
                ):
                    from cns.core.stream_events import (
                        TextEvent, ThinkingEvent, CompleteEvent,
                        ToolExecutingEvent, ToolCompletedEvent, ToolErrorEvent,
                        CircuitBreakerEvent
                    )

                    # Detect invokeother_tool execution for auto-continuation
                    # Must check here because final response won't contain intermediate tool calls
                    if isinstance(event, ToolExecutingEvent):
                        # Log ALL tool executions for visibility
                        if event.tool_name == "code_execution":
                            saw_code_execution = True
                            logger.info("=" * 80)
                            logger.info("🐍 CODE_EXECUTION INVOKED")
                            logger.info("=" * 80)
                            # Log the Python code being executed
                            code = event.arguments.get("code", "")
                            logger.info(f"Python code to execute:\n{code}")
                            logger.info("=" * 80)
                        elif event.tool_name == "invokeother_tool":
                            mode = event.arguments.get("mode", "")
                            if mode in ["load", "fallback", "prepare_code_execution"]:
                                invoked_tool_loader = True
                                logger.info(f"Detected invokeother_tool execution with mode={mode}")
                        else:
                            # Log other tool calls for context
                            logger.info(f"Tool executing: {event.tool_name} with args: {event.arguments}")

                    # Log tool completion results
                    if isinstance(event, ToolCompletedEvent):
                        if event.tool_name == "code_execution":
                            logger.info("=" * 80)
                            logger.info("✅ CODE_EXECUTION COMPLETED")
                            logger.info("=" * 80)
                            logger.info(f"Result:\n{event.result}")
                            logger.info("=" * 80)
                        else:
                            logger.info(f"Tool completed: {event.tool_name} -> {event.result[:200]}...")

                    # Log tool errors
                    if isinstance(event, ToolErrorEvent):
                        if event.tool_name == "code_execution":
                            logger.error("=" * 80)
                            logger.error("❌ CODE_EXECUTION FAILED")
                            logger.error("=" * 80)
                            logger.error(f"Error:\n{event.error}")
                            logger.error("=" * 80)
                        else:
                            logger.error(f"Tool error: {event.tool_name} -> {event.error}")

                    # Call stream callback if provided (for compatibility during transition)
                    if stream and stream_callback:
                        if isinstance(event, TextEvent):
                            stream_callback({"type": "text", "content": event.content})
                            response_text += event.content
                        elif isinstance(event, ThinkingEvent):
                            # Filter thinking from generic providers unless config allows
                            is_generic = llm_kwargs.get('endpoint_url') is not None
                            if is_generic and not config.api.show_generic_thinking:
                                pass  # Skip showing to user (still in history)
                            else:
                                stream_callback({"type": "thinking", "content": event.content})
                        elif hasattr(event, 'tool_name'):
                            stream_callback({"type": "tool_event", "event": event.type, "tool": event.tool_name})
                        elif isinstance(event, CircuitBreakerEvent):
                            # Send model error notification for tool validation failures
                            if "failed after correction" in event.reason:
                                stream_callback({
                                    "type": "model_error",
                                    "reason": event.reason
                                })

                    # Store events for websocket
                    events.append(event)

                    # Capture final response
                    if isinstance(event, CompleteEvent):
                        raw_response = event.response
                        response_text = self.llm_provider.extract_text_content(raw_response)

                        # Store container_id in Valkey for reuse (1-hour TTL)
                        if hasattr(raw_response, '_container_id') and raw_response._container_id:
                            valkey_key = f"container:{continuum.id}"
                            valkey.setex(valkey_key, 3600, raw_response._container_id)  # 1-hour TTL
                            logger.info(f"📦 Stored container ID in Valkey: {raw_response._container_id}")

                        # Capture Anthropic code_execution artifact IDs for webapp downloads.
                        artifact_file_ids = self.llm_provider.extract_code_execution_file_ids(raw_response)
                        if artifact_file_ids:
                            metadata["code_execution_file_ids"] = artifact_file_ids

                            artifact_key = f"codeexec_artifacts:{continuum.id}"
                            captured_at = datetime.now(timezone.utc).isoformat()
                            try:
                                raw_cached = valkey.get(artifact_key)
                                cached_entries = json.loads(raw_cached) if raw_cached else []
                                if not isinstance(cached_entries, list):
                                    cached_entries = []
                            except Exception:
                                cached_entries = []

                            seen_ids = {
                                entry.get("file_id")
                                for entry in cached_entries
                                if isinstance(entry, dict)
                            }
                            for file_id in artifact_file_ids:
                                if file_id not in seen_ids:
                                    cached_entries.append({
                                        "file_id": file_id,
                                        "captured_at": captured_at,
                                    })
                                    seen_ids.add(file_id)

                            # Keep recent history and retain for 24h.
                            cached_entries = cached_entries[-200:]
                            try:
                                valkey.setex(artifact_key, 86400, json.dumps(cached_entries))
                                logger.info(
                                    f"🗂️ Cached {len(artifact_file_ids)} code_execution artifacts "
                                    f"for continuum {continuum.id}"
                                )
                            except Exception as cache_error:
                                logger.warning(f"Failed to cache code_execution artifacts: {cache_error}")

                        # Log cache metrics and track for next turn's estimation
                        if hasattr(raw_response, 'usage') and raw_response.usage:
                            usage = raw_response.usage
                            # Track input tokens for next turn's proactive estimation
                            self._last_turn_usage[str(continuum.id)] = usage.input_tokens
                            cache_created = getattr(usage, 'cache_creation_input_tokens', 0)
                            cache_read = getattr(usage, 'cache_read_input_tokens', 0)
                            if cache_created > 0:
                                logger.info(f"Cache created: {cache_created} tokens")
                            if cache_read > 0:
                                logger.debug(f"Cache read: {cache_read} tokens")

                # Success - break out of retry loop
                break

            except ContextOverflowError as e:
                overflow_attempt += 1
                logger.warning(
                    f"Context overflow from API: {e} (attempt {overflow_attempt}/{max_overflow_retries})"
                )
                if overflow_attempt > max_overflow_retries:
                    raise RuntimeError(
                        f"Request exceeds context window after {max_overflow_retries} remediation attempts."
                    ) from e
                # Apply remediation and retry
                messages_for_llm = self._apply_overflow_remediation(
                    overflow_attempt, messages_for_llm, complete_messages, continuum, text_for_context,
                    system_prompt=system_prompt,
                    estimated_tokens=e.estimated_tokens, event_type='reactive'
                )
                # Reset events for retry
                events = []
                response_text = ""
                raw_response = None
                continue

        # Extract tools used from LLM response for metadata
        tool_calls = self.llm_provider.extract_tool_calls(raw_response)
        if tool_calls:
            tools_used_this_turn = [call["tool_name"] for call in tool_calls]
            metadata["tools_used"] = tools_used_this_turn
        if saw_code_execution:
            existing_tools = metadata.get("tools_used", [])
            if "code_execution" not in existing_tools:
                metadata["tools_used"] = [*existing_tools, "code_execution"]

        # Parse tags from final response (preserve emotion tag for frontend extraction)
        parsed_tags = self.tag_parser.parse_response(response_text, preserve_tags=['my_emotion'])
        clean_response_text = parsed_tags['clean_text']

        # Debug: Log emotion tag presence
        logger.info(f"Emotion extracted: {parsed_tags.get('emotion')}")
        logger.info(f"Emotion tag in clean_text: {'<mira:my_emotion>' in clean_response_text}")

        # Check if model tool error caused a blank response - provide user-friendly fallback
        from cns.core.stream_events import CircuitBreakerEvent
        model_tool_error = next(
            (e for e in events if isinstance(e, CircuitBreakerEvent)
             and "failed after correction" in e.reason),
            None
        )
        if model_tool_error and (not clean_response_text or not clean_response_text.strip()):
            logger.warning(f"Model returned blank after tool error: {model_tool_error.reason}")
            clean_response_text = (
                "I encountered an issue with this request. The AI model made an invalid "
                "tool call that couldn't be corrected. This is a limitation of the model, "
                "not MIRA. Please try rephrasing your request."
            )
            metadata["model_error"] = True
            metadata["model_error_reason"] = str(model_tool_error.reason)

        # Strict output-claim guard:
        # Block claims of file creation unless there is verifiable evidence.
        claims_file_output = self._response_claims_file_creation(clean_response_text)
        text_verified_paths = self._collect_verified_local_file_paths(clean_response_text)
        tool_verified_paths = self._collect_verified_local_file_paths_from_tool_events(events)
        verified_paths: List[str] = []
        for path in text_verified_paths + tool_verified_paths:
            if path not in verified_paths:
                verified_paths.append(path)
        metadata["verified_output_paths"] = verified_paths
        metadata["output_claim_detected"] = claims_file_output

        artifact_ids = metadata.get("code_execution_file_ids", []) or []
        if claims_file_output and not verified_paths:
            metadata["output_claim_blocked"] = True
            if artifact_ids:
                sample = ", ".join(artifact_ids[:3])
                suffix = "..." if len(artifact_ids) > 3 else ""
                clean_response_text = (
                    "I cannot verify local filesystem creation for this turn. "
                    "I can only verify Anthropic sandbox artifacts from code execution "
                    f"(file_id: {sample}{suffix}). "
                    "Use the Files tab -> Anthropic sandbox artifacts to download them."
                )
            else:
                clean_response_text = (
                    "I cannot confirm file creation for this turn because no verifiable output evidence "
                    "was returned (no sandbox artifact file_id and no verified local file path). "
                    "Please ask me to return only verified outputs."
                )
        else:
            metadata["output_claim_blocked"] = False

        # Add final assistant response to continuum FIRST (before topic change handling)
        # Validate response is not blank before saving
        if not clean_response_text or not clean_response_text.strip():
            logger.error("Attempted to save blank assistant response - rejecting")
            raise ValueError("Assistant response cannot be blank or empty. This may indicate an API error.")

        # Resolve short memory IDs (8-char) to full UUIDs using surfaced memories
        # LLM outputs mem_XXXXXXXX format, tag parser extracts 8-char portion
        short_refs = parsed_tags.get('referenced_memories', [])
        resolved_refs = []
        for short_id in short_refs:
            for mem in surfaced_memories:
                if match_memory_id(short_id, mem['id']):
                    resolved_refs.append(mem['id'])
                    break

        assistant_metadata = {
            "referenced_memories": resolved_refs,
            "surfaced_memories": [m['id'] for m in surfaced_memories],
            "pinned_memory_ids": list(pinned_ids)  # 8-char IDs for importance boost
        }
        if metadata.get("code_execution_file_ids"):
            assistant_metadata["code_execution_file_ids"] = metadata["code_execution_file_ids"]
        if metadata.get("verified_output_paths"):
            assistant_metadata["verified_output_paths"] = metadata["verified_output_paths"]
        if metadata.get("output_claim_blocked"):
            assistant_metadata["output_claim_blocked"] = True

        # Add emotion if present
        if parsed_tags.get('emotion'):
            assistant_metadata["emotion"] = parsed_tags['emotion']

        assistant_msg_obj, response_events = continuum.add_assistant_message(
            clean_response_text, assistant_metadata
        )
        self._publish_events(response_events)

        # Internal orchestration prompts are inference-only and should not appear in
        # user-visible/history message streams.
        if _internal_user_message:
            continuum.apply_cache([
                msg for msg in continuum.messages
                if msg.id != user_msg_obj.id
            ])
            logger.debug("Removed internal synthetic user message from continuum cache")

        # Publish turn completed event for subscribers (Letta buffering, tool auto-unload, etc.)
        # Pass continuum object so handlers can extract whatever data they need
        # Calculate turn number from message count (each turn = user msg + assistant msg)
        turn_number = (len(continuum.messages) + 1) // 2
        self._publish_events([TurnCompletedEvent.create(
            continuum_id=str(continuum.id),
            turn_number=turn_number,
            segment_turn_number=segment_turn_number,  # Turn within current segment
            continuum=continuum
        )])

        final_response = clean_response_text
        
        # Update metadata with referenced memories and pinned IDs
        metadata["referenced_memories"] = resolved_refs
        metadata["surfaced_memories"] = [m['id'] for m in surfaced_memories]
        metadata["pinned_memory_ids"] = list(pinned_ids)  # 8-char IDs for importance boost

        # Add emotion to metadata for persistence
        if parsed_tags.get('emotion'):
            metadata["emotion"] = parsed_tags['emotion']
        
        # Unit of Work is required for proper persistence
        if not unit_of_work:
            raise ValueError("Unit of Work is required for message persistence")

        if _internal_user_message:
            # Internal synthetic prompts are never persisted as user turns.
            unit_of_work.add_messages(assistant_msg_obj)
        else:
            # Prepare user message for persistence
            # Validate: if user_message contains images, storage_content MUST be provided
            if isinstance(user_msg_obj.content, list):
                has_image = any(item.get('type') == 'image' for item in user_msg_obj.content)
                if has_image and storage_content is None:
                    raise ValueError(
                        "storage_content is required when user_message contains images. "
                        "Callers must provide the 512px WebP storage tier for image persistence."
                    )

            # Use storage_content if provided (e.g., 512px WebP for images), otherwise use original
            persist_content = storage_content if storage_content is not None else user_msg_obj.content

            # Create message with appropriate content for persistence
            from cns.core.message import Message
            persist_user_msg = Message(
                content=persist_content,
                role=user_msg_obj.role,
                id=user_msg_obj.id,
                created_at=user_msg_obj.created_at,
                metadata=user_msg_obj.metadata
            )

            # Add messages to unit of work for batch persistence
            unit_of_work.add_messages(persist_user_msg, assistant_msg_obj)

        # Mark metadata for update
        unit_of_work.mark_metadata_updated()

        # Auto-continuation: If tools were loaded and we haven't already tried,
        # automatically continue with the task
        if invoked_tool_loader and not _tried_loading_all_tools:
            logger.info("Auto-continuing after tool loading...")

            # Create synthetic user message to prompt continuation
            synthetic_message = (
                "Great, the tool is now available. Please proceed with completing "
                "the original task using the newly loaded tool."
            )

            # Recursively process with the synthetic message
            # Pass _tried_loading_all_tools=True to prevent infinite loops
            continuum, final_response, metadata = self.process_message(
                continuum,
                synthetic_message,
                system_prompt,
                stream=stream,
                stream_callback=stream_callback,
                _tried_loading_all_tools=True,  # Prevent infinite loops
                _internal_user_message=True,  # Keep synthetic prompt internal-only
                unit_of_work=unit_of_work,
                segment_turn_number=segment_turn_number
            )
            logger.info("Auto-continuation completed successfully")

        return continuum, final_response, metadata
    
    def _handle_system_prompt_composed(self, event):
        """Handle system prompt composed event."""
        from cns.core.events import SystemPromptComposedEvent
        event: SystemPromptComposedEvent
        self._cached_content = event.cached_content
        self._non_cached_content = event.non_cached_content
        self._notification_center = event.notification_center
        logger.debug(
            f"Received system prompt: cached {len(event.cached_content)} chars, "
            f"non-cached {len(event.non_cached_content)} chars, "
            f"notification center {len(event.notification_center)} chars"
        )

    def _compose_prompt_sections(self, continuum_id: str, system_prompt: str) -> tuple[str, str, str]:
        """
        Compose working-memory prompt sections synchronously via event bus.

        Returns:
            Tuple of (cached_content, non_cached_content, notification_center)
        """
        from cns.core.events import ComposeSystemPromptEvent

        self._cached_content = None
        self._non_cached_content = None
        self._notification_center = None
        self.event_bus.publish(ComposeSystemPromptEvent.create(
            continuum_id=continuum_id,
            base_prompt=system_prompt
        ))
        return (
            self._cached_content or "",
            self._non_cached_content or "",
            self._notification_center or "",
        )

    @staticmethod
    def _build_system_blocks(cached_content: str, non_cached_content: str) -> List[Dict[str, Any]]:
        """Build Anthropic-style structured system blocks."""
        system_blocks: List[Dict[str, Any]] = []

        if cached_content:
            system_blocks.append({
                "type": "text",
                "text": cached_content,
                "cache_control": {"type": "ephemeral"}
            })

        if non_cached_content:
            system_blocks.append({
                "type": "text",
                "text": non_cached_content
            })

        return system_blocks

    def _recompose_messages_for_retry(
        self,
        messages_for_llm: List[Dict[str, Any]],
        cached_content: str,
        non_cached_content: str,
        notification_center: str
    ) -> List[Dict[str, Any]]:
        """
        Update prompt scaffolding in-place for retry while preserving trimmed history.

        This is used by overflow remediation so Tier-1 memory evacuation can affect
        the same turn without rebuilding conversation history from scratch.
        """
        hud_prefix = "═══════════════════════════ HUD ════════════════════════════\n"
        updated_messages = list(messages_for_llm)
        system_blocks = self._build_system_blocks(cached_content, non_cached_content)

        if not updated_messages:
            return [{"role": "system", "content": system_blocks}]

        updated_messages[0] = {"role": "system", "content": system_blocks}

        hud_index = None
        for idx, msg in enumerate(updated_messages[1:], start=1):
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, str) and content.startswith(hud_prefix):
                hud_index = idx
                break

        if notification_center:
            hud_message = {"role": "assistant", "content": hud_prefix + notification_center}
            if hud_index is None:
                insert_idx = len(updated_messages) - 1 if len(updated_messages) > 1 else len(updated_messages)
                updated_messages.insert(insert_idx, hud_message)
            else:
                updated_messages[hud_index] = hud_message
        elif hud_index is not None:
            updated_messages.pop(hud_index)

        return updated_messages


    def _publish_events(self, events: List[ContinuumEvent]):
        """Publish events to event bus."""
        for event in events:
            self.event_bus.publish(event)

    def _response_claims_file_creation(self, response_text: str) -> bool:
        """
        Detect explicit assistant claims that files were created/written.

        This intentionally avoids triggering on:
        - Echoed user transcript lines (e.g., "YOU: create file ...")
        - Requests/instructions ("please create ...")
        - Uncertainty/negation ("cannot confirm file creation")
        """
        if not response_text:
            return False

        # Ignore quoted/user-transcript lines that can appear in model output.
        filtered_lines: List[str] = []
        for line in response_text.splitlines():
            low = line.strip().lower()
            if low.startswith(("you:", "user:", "human:")):
                continue
            filtered_lines.append(line.strip())

        text = " ".join(filtered_lines).strip().lower()
        if not text:
            return False

        # Explicit uncertainty/negation should never count as a success claim.
        uncertainty_or_negation = [
            r"\bcannot confirm file creation\b",
            r"\bno verifiable output evidence\b",
            r"\b(no|without)\b.{0,24}\b(file|files|artifact|artifacts)\b",
            r"\b(cannot|can't|unable|failed|did not|didn't|not)\b.{0,40}\b(create|created|write|wrote|save|saved|generated|file|files)\b",
        ]
        if any(re.search(pattern, text) for pattern in uncertainty_or_negation):
            return False

        mentions_file_target = any(
            token in text
            for token in (" file", " files", "$output_dir", "output_dir", "/tmp/", "/files/")
        )
        mentions_file_target = mentions_file_target or bool(
            re.search(r"\.(txt|md|csv|json|pdf|xlsx|docx)\b", text)
        )
        if not mentions_file_target:
            return False

        # Strong success signals (first-person or explicit completion phrasing).
        first_person_success = bool(
            re.search(
                r"\b(i|we)\s+(?:have\s+|just\s+|successfully\s+)?"
                r"(?:created|wrote|written|saved|generated|produced|exported)\b",
                text,
            )
        )
        passive_success = bool(
            re.search(
                r"\b(file|files|document|documents|report)\s+"
                r"(?:has|have|was|were)\s+(?:been\s+)?"
                r"(?:created|written|saved|generated|produced|exported)\b",
                text,
            )
        )
        bare_success = bool(
            re.search(
                r"(?:^|[.!?]\s+)(?:created|wrote|written|saved|generated|produced|exported)\b",
                text,
            )
        )

        # If it's clearly a request/instruction and no success signal, do not block.
        request_language = bool(
            re.search(r"\b(please|can you|could you|should|need to|ask me to|try to)\b", text)
        )
        if request_language and not (first_person_success or passive_success or bare_success):
            return False

        return first_person_success or passive_success or bare_success

    def _collect_verified_local_file_paths(self, response_text: str) -> List[str]:
        """
        Extract path-like tokens from response text and return those that exist as files.
        """
        if not response_text:
            return []

        path_pattern = re.compile(r"(\$OUTPUT_DIR(?:/[^\s`'\",;:()<>]+)?|/(?:[A-Za-z0-9._-]+/?)+)")
        output_dir = os.getenv("OUTPUT_DIR", "").strip()
        verified: List[str] = []
        seen: set[str] = set()

        for raw_token in path_pattern.findall(response_text):
            token = raw_token.rstrip(".,!?:;)]}'\"")

            if token.startswith("$OUTPUT_DIR"):
                if not output_dir:
                    continue
                token = output_dir + token[len("$OUTPUT_DIR") :]

            try:
                path = Path(token).expanduser().resolve()
            except Exception:
                continue

            candidate = str(path)
            if candidate in seen:
                continue

            if path.exists() and path.is_file():
                seen.add(candidate)
                verified.append(candidate)

        return verified

    def _collect_verified_local_file_paths_from_tool_events(self, events: List[Any]) -> List[str]:
        """
        Extract verifiable file paths from successful tool result payloads.

        This is used to ground file-creation claims in actual tool outputs.
        """
        if not events:
            return []

        from cns.core.stream_events import ToolCompletedEvent

        verified: List[str] = []
        seen: set[str] = set()

        for event in events:
            if not isinstance(event, ToolCompletedEvent):
                continue

            raw_result = getattr(event, "result", None)
            if not isinstance(raw_result, str) or not raw_result.strip():
                continue

            payload: Optional[Dict[str, Any]] = None
            try:
                parsed = json.loads(raw_result)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(raw_result)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = None

            if not payload:
                continue
            if payload.get("success") is False:
                continue

            # For notes_tool, only treat create/overwrite-like payloads as file-output evidence.
            if event.tool_name == "notes_tool":
                if not (
                    payload.get("created") is True
                    or payload.get("overwritten") is True
                    or "bytes_written" in payload
                ):
                    continue

            path_candidates: List[str] = []

            absolute_path = payload.get("absolute_path")
            if isinstance(absolute_path, str) and absolute_path.strip():
                path_candidates.append(absolute_path.strip())

            file_path = payload.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                path_candidates.append(file_path.strip())

            filepath = payload.get("filepath")
            if isinstance(filepath, str) and filepath.strip():
                path_candidates.append(filepath.strip())

            # Some tools return root + relative path.
            root = payload.get("root")
            rel_path = payload.get("path")
            if isinstance(root, str) and root.strip() and isinstance(rel_path, str) and rel_path.strip():
                rel_candidate = Path(rel_path.strip())
                if not rel_candidate.is_absolute():
                    path_candidates.append(str(Path(root.strip()) / rel_candidate))
                else:
                    path_candidates.append(str(rel_candidate))

            for token in path_candidates:
                try:
                    candidate_path = Path(token).expanduser().resolve()
                except Exception:
                    continue

                candidate = str(candidate_path)
                if candidate in seen:
                    continue
                if candidate_path.exists() and candidate_path.is_file():
                    seen.add(candidate)
                    verified.append(candidate)

        return verified

    def _get_previous_memories(self) -> List[Dict[str, Any]]:
        """
        Get previously surfaced memories from the trinket cache.

        Returns:
            List of memory dicts from the previous turn, or empty list if none
        """
        trinket = self.working_memory.get_trinket('ProactiveMemoryTrinket')
        if trinket and hasattr(trinket, 'get_cached_memories'):
            return trinket.get_cached_memories()
        return []

    def _apply_retention(
        self,
        previous_memories: List[Dict[str, Any]],
        pinned_ids: set
    ) -> List[Dict[str, Any]]:
        """
        Filter previous memories to keep only those marked for retention.

        Matches by 8-char ID prefix since the LLM outputs shortened IDs.

        Args:
            previous_memories: All memories from previous turn
            pinned_ids: Set of 8-char memory IDs marked [x] by the LLM

        Returns:
            List of memories that should be pinned (retained)
        """
        if not previous_memories or not pinned_ids:
            return []

        pinned = []
        for memory in previous_memories:
            memory_id = memory.get('id', '')
            if memory_id and any(match_memory_id(memory_id, pid) for pid in pinned_ids):
                pinned.append(memory)

        logger.debug(
            f"Retention: {len(pinned)}/{len(previous_memories)} memories retained"
        )
        return pinned

    def _merge_memories(
        self,
        pinned_memories: List[Dict[str, Any]],
        fresh_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge pinned and fresh memories, deduplicating by ID.

        Pinned memories appear first and take precedence.

        Args:
            pinned_memories: Memories retained from previous turn
            fresh_memories: Newly retrieved memories

        Returns:
            Merged list with pinned first, then fresh (no duplicates)
        """
        # Start with pinned memories
        merged = list(pinned_memories)
        seen_ids = {m.get('id') for m in pinned_memories if m.get('id')}

        # Add fresh memories that aren't already in pinned
        for memory in fresh_memories:
            memory_id = memory.get('id')
            if memory_id and memory_id not in seen_ids:
                merged.append(memory)
                seen_ids.add(memory_id)

        return merged

    # =========================================================================
    # Context Overflow Detection and Remediation
    # =========================================================================

    def _estimate_request_tokens(
        self,
        messages: List[Dict],
        tools: List[Dict],
        last_turn_input_tokens: Optional[int] = None
    ) -> int:
        """
        Estimate tokens for upcoming LLM request.

        Uses actual token count from previous turn when available (most accurate),
        otherwise falls back to conservative character-based estimation.

        Args:
            messages: Messages to send (including system message)
            tools: Tool definitions
            last_turn_input_tokens: Actual input tokens from previous turn

        Returns:
            Estimated token count for the request
        """
        if last_turn_input_tokens is not None:
            # Use actual count from last turn as baseline
            base_tokens = last_turn_input_tokens
        else:
            # Fallback: 4 chars/token (conservative estimate)
            total_chars = 0
            for msg in messages:
                content = msg.get('content', '')
                if isinstance(content, list):
                    # Handle structured content (system blocks, multimodal)
                    for block in content:
                        if isinstance(block, dict):
                            total_chars += len(str(block.get('text', '')))
                else:
                    total_chars += len(str(content))
            base_tokens = total_chars // 4

        # Tool definitions: ~100 tokens per tool baseline
        tool_tokens = len(tools) * 100 if tools else 0

        # 5% overhead buffer for formatting, separators, etc.
        return int((base_tokens + tool_tokens) * 1.05)

    def _prune_by_topic_drift(
        self,
        messages: List[Dict],
        return_details: bool = False
    ):
        """
        Find topic drift boundary using sliding window embedding similarity.
        Drop messages before the boundary to reduce context.

        Algorithm:
        1. Generate embeddings for sliding windows of messages
        2. Compare adjacent windows via cosine similarity
        3. Find largest similarity drop (= topic drift boundary)
        4. If drop below threshold: cut at boundary
        5. Fallback: oldest-first pruning with configurable count

        Args:
            messages: Full message list including system message at [0]
            return_details: If True, return (pruned_messages, drift_details) for logging

        Returns:
            Pruned message list, or tuple of (pruned_list, details_dict) if return_details=True
        """
        import numpy as np

        overflow_logger = get_overflow_logger()

        # Config values
        window_size = config.context.topic_drift_window_size
        drift_threshold = config.context.topic_drift_threshold
        fallback_prune_count = config.context.overflow_fallback_prune_count

        def make_result(pruned: List[Dict], details: Dict):
            """Helper to return consistent format."""
            if return_details:
                return pruned, details
            return pruned

        # Need enough messages to analyze (window_size * 2 + 1 for system msg)
        if len(messages) < window_size * 2 + 1:
            # Too few messages, use oldest-first fallback
            logger.info(f"Too few messages for drift detection ({len(messages)}), using fallback")
            result = messages[:1] + messages[fallback_prune_count + 1:]
            details = overflow_logger.log_topic_drift_analysis(
                continuum_id=None,  # Not available here
                candidate_cuts=[],
                selected_index=None,
                selection_method="too_few_messages",
                window_size=window_size,
                threshold=drift_threshold
            )
            return make_result(result, details)

        # Generate window embeddings (exclude system message at [0])
        content_messages = messages[1:]
        windows = []

        for i in range(len(content_messages) - window_size + 1):
            window_text = " ".join(
                str(m.get('content', ''))[:500]  # Truncate long messages
                for m in content_messages[i:i + window_size]
            )
            # Use fast embeddings for drift detection
            embedding = self.embeddings_provider.embed_query(window_text)
            windows.append((i, embedding))

        # Find candidate cut points (similarity drops)
        candidate_cuts = []

        def cosine_similarity(a, b) -> float:
            """Compute cosine similarity between two vectors."""
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        for i in range(len(windows) - 1, 0, -1):
            similarity = cosine_similarity(windows[i][1], windows[i - 1][1])
            drop = 1.0 - similarity
            if drop > (1.0 - drift_threshold):
                candidate_cuts.append({
                    'index': windows[i][0],
                    'similarity': similarity,
                    'drop': drop
                })

        # Current implementation: select largest drop
        best_cut_idx = None
        selection_method = "no_candidates"
        if candidate_cuts:
            best_cut = max(candidate_cuts, key=lambda c: c['drop'])
            best_cut_idx = best_cut['index']
            selection_method = "largest_drop"

        # Build details for logging
        details = overflow_logger.log_topic_drift_analysis(
            continuum_id=None,  # Filled in by caller
            candidate_cuts=candidate_cuts,
            selected_index=best_cut_idx,
            selection_method=selection_method,
            window_size=window_size,
            threshold=drift_threshold
        )

        if best_cut_idx is not None:
            # Found topic boundary - keep system msg + messages from boundary onward
            logger.info(f"Topic drift detected at message {best_cut_idx}, dropping {best_cut_idx} messages")
            return make_result(messages[:1] + content_messages[best_cut_idx:], details)
        else:
            # No clear boundary - use oldest-first fallback
            logger.info(f"No topic drift found, using oldest-first fallback ({fallback_prune_count} messages)")
            details["selection_method"] = "fallback"
            return make_result(messages[:1] + messages[fallback_prune_count + 1:], details)

    def _llm_judge_cut_point(self, messages: List[Dict]) -> Optional[int]:
        """
        Use LLM to intelligently select the best cut point for context reduction.

        Analyzes conversation for topic boundaries and selects where to cut
        that minimizes loss of relevant context.

        Args:
            messages: Full message list including system message at [0]

        Returns:
            Index to cut at (messages before this index will be dropped), or None if no cut recommended
        """
        import numpy as np

        # Need enough messages to analyze
        if len(messages) < 7:  # System + at least 3 turns
            return None

        content_messages = messages[1:]  # Exclude system message

        # First, find candidate cut points using embedding similarity
        window_size = config.context.topic_drift_window_size
        drift_threshold = config.context.topic_drift_threshold

        if len(content_messages) < window_size * 2:
            return None

        # Generate window embeddings
        windows = []
        for i in range(len(content_messages) - window_size + 1):
            window_text = " ".join(
                str(m.get('content', ''))[:500]
                for m in content_messages[i:i + window_size]
            )
            embedding = self.embeddings_provider.embed_query(window_text)
            windows.append((i, embedding))

        # Find candidate cut points
        def cosine_similarity(a, b) -> float:
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        candidate_cuts = []
        for i in range(len(windows) - 1, 0, -1):
            similarity = cosine_similarity(windows[i][1], windows[i - 1][1])
            drop = 1.0 - similarity
            if drop > (1.0 - drift_threshold):
                candidate_cuts.append({
                    'index': windows[i][0],
                    'similarity': similarity,
                    'drop': drop
                })

        if not candidate_cuts:
            return None

        # Build context for LLM showing candidate boundaries
        boundary_contexts = []
        for i, cut in enumerate(candidate_cuts[:5]):  # Limit to top 5 candidates
            cut_idx = cut['index']
            before_start = max(0, cut_idx - 2)
            after_end = min(len(content_messages), cut_idx + 2)

            before_msgs = []
            for j in range(before_start, cut_idx):
                msg = content_messages[j]
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:200]
                before_msgs.append(f"  [{role}]: {content}...")

            after_msgs = []
            for j in range(cut_idx, after_end):
                msg = content_messages[j]
                role = msg.get('role', 'unknown')
                content = str(msg.get('content', ''))[:200]
                after_msgs.append(f"  [{role}]: {content}...")

            boundary_contexts.append(
                f"BOUNDARY {i + 1} (similarity drop: {cut['drop']:.2f}):\n"
                f"Before boundary:\n" + "\n".join(before_msgs) + "\n"
                f"--- CUT HERE (drop {cut_idx} messages) ---\n"
                f"After boundary:\n" + "\n".join(after_msgs)
            )

        recent_msg = content_messages[-1]
        recent_content = str(recent_msg.get('content', ''))[:300]

        prompt = f"""You are helping manage conversation context. The conversation has grown too large and we need to trim older messages.

Below are candidate cut points where the topic appears to shift. Each boundary shows messages before and after the potential cut point.

MOST RECENT MESSAGE (what we're trying to respond to):
{recent_content}

CANDIDATE BOUNDARIES:
{chr(10).join(boundary_contexts)}

Which boundary is the BEST place to cut? Consider:
1. Which older content is least relevant to the recent message?
2. Where does a clear topic shift occur?
3. We want to preserve context that helps answer the recent message.

Respond with ONLY the boundary number (1-{len(candidate_cuts)}) or "NONE" if no cut is recommended.
"""

        try:
            # Use a cheap, fast model for this judgment
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_preference="claude-3-5-haiku-20241022",
                tools=None
            )

            result_text = self.llm_provider.extract_text_content(response).strip().upper()

            if result_text == "NONE":
                return None

            # Parse boundary number
            try:
                boundary_num = int(result_text.replace("BOUNDARY", "").strip())
                if 1 <= boundary_num <= len(candidate_cuts):
                    selected_cut = candidate_cuts[boundary_num - 1]
                    logger.info(f"LLM selected boundary {boundary_num} at index {selected_cut['index']}")
                    return selected_cut['index']
            except ValueError:
                pass

            # Fallback: if we can't parse, use largest drop
            best_cut = max(candidate_cuts, key=lambda c: c['drop'])
            return best_cut['index']

        except Exception as e:
            logger.warning(f"LLM judgment failed, using embedding fallback: {e}")
            # Fallback to largest drop
            best_cut = max(candidate_cuts, key=lambda c: c['drop'])
            return best_cut['index']

    def _schedule_async_context_judgment(self, continuum_id, messages: List[Dict]) -> None:
        """
        Schedule async LLM judgment to determine optimal cut point.
        Result stored in _pending_context_trim for one-shot application on next request.

        Args:
            continuum_id: ID of the continuum (UUID or string)
            messages: Full message list for analysis
        """
        import concurrent.futures
        judgment_ctx = contextvars.copy_context()

        def _run_judgment_sync():
            """Synchronous wrapper for LLM judgment."""
            try:
                optimal_cut = self._llm_judge_cut_point(messages)
                if optimal_cut is not None:
                    self._pending_context_trim[str(continuum_id)] = optimal_cut
                    logger.info(f"Async LLM judgment complete: trim index {optimal_cut} stored for next request")
            except Exception as e:
                logger.warning(f"Async context judgment failed (non-critical): {e}")

        # Run in thread pool to not block
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            executor.submit(judgment_ctx.run, _run_judgment_sync)
            executor.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Failed to schedule async judgment: {e}")

    def _apply_overflow_remediation(
        self,
        attempt: int,
        messages_for_llm: List[Dict],
        complete_messages: List[Dict],
        continuum,
        text_for_context: str,
        system_prompt: str,
        estimated_tokens: int = 0,
        event_type: str = "proactive"
    ) -> List[Dict]:
        """
        Apply tiered overflow remediation strategy.

        Tier 1: Force memory evacuation (preserves conversation, shrinks system prompt)
        Tier 2: Embedding-based topic drift pruning (fast, no LLM)
        Tier 3: Pure oldest-first fallback (maximum speed)

        Args:
            attempt: Current remediation attempt (1, 2, or 3)
            messages_for_llm: Current message list to reduce
            complete_messages: Original complete message list (for async judgment)
            continuum: Continuum object for ID
            text_for_context: User's text for evacuation context
            system_prompt: Base system prompt used to recompose sections in Tier-1
            estimated_tokens: Token estimate that triggered overflow (for logging)
            event_type: 'proactive' or 'reactive' (for logging)

        Returns:
            Reduced message list
        """
        overflow_logger = get_overflow_logger()
        messages_before = len(messages_for_llm)

        if attempt == 1 and self.memory_evacuator:
            # Remediation 1: Force aggressive memory evacuation
            logger.info("Remediation 1: Forcing aggressive memory evacuation")
            previous_memories = self._get_previous_memories()
            if len(previous_memories) > 3:  # Only evacuate if meaningful reduction possible
                evacuated = self.memory_evacuator.evacuate(
                    memories=previous_memories,
                    continuum=continuum,
                    user_message=text_for_context
                )
                # Update trinket with reduced set
                trinket = self.working_memory.get_trinket('ProactiveMemoryTrinket')
                if trinket and hasattr(trinket, '_cached_memories'):
                    trinket._cached_memories = evacuated
                logger.info(f"Memory evacuation: {len(previous_memories)} -> {len(evacuated)} memories")

            # Recompose prompt sections and update current retry messages in-turn.
            # This makes Tier-1 effective for the current request instead of next turn.
            cached_content, non_cached_content, notification_center = self._compose_prompt_sections(
                continuum_id=str(continuum.id),
                system_prompt=system_prompt
            )
            remediated_messages = self._recompose_messages_for_retry(
                messages_for_llm=messages_for_llm,
                cached_content=cached_content,
                non_cached_content=non_cached_content,
                notification_center=notification_center
            )

            # Log the remediation attempt
            overflow_logger.log_overflow(
                continuum_id=continuum.id,
                event_type=event_type,
                estimated_tokens=estimated_tokens,
                remediation_tier=1,
                messages_before=messages_before,
                messages_after=len(remediated_messages),
                success=True
            )
            return remediated_messages

        elif attempt == 2:
            # Remediation 2: Embedding-based topic drift pruning
            logger.info("Remediation 2: Embedding-based topic drift pruning")
            pruned, drift_result = self._prune_by_topic_drift(messages_for_llm, return_details=True)

            # Log the remediation with topic drift details
            overflow_logger.log_overflow(
                continuum_id=continuum.id,
                event_type=event_type,
                estimated_tokens=estimated_tokens,
                remediation_tier=2,
                messages_before=messages_before,
                messages_after=len(pruned),
                topic_drift_result=drift_result,
                success=True
            )

            # Fire async LLM judgment for next request (one-shot improvement)
            self._schedule_async_context_judgment(continuum.id, complete_messages)
            return pruned

        else:
            # Remediation 3: Pure oldest-first fallback (maximum speed)
            logger.info("Remediation 3: Pure oldest-first fallback")
            prune_count = config.context.overflow_fallback_prune_count
            result = messages_for_llm[:1] + messages_for_llm[prune_count + 1:]

            # Log the fallback remediation
            overflow_logger.log_overflow(
                continuum_id=continuum.id,
                event_type=event_type,
                estimated_tokens=estimated_tokens,
                remediation_tier=3,
                messages_before=messages_before,
                messages_after=len(result),
                success=True
            )
            return result


# Global orchestrator instance (singleton pattern)
_orchestrator_instance = None


def initialize_orchestrator(orchestrator_instance: ContinuumOrchestrator) -> None:
    """
    Initialize the global orchestrator instance.
    
    This should be called once during application startup after creating
    the orchestrator with all its dependencies.
    
    Args:
        orchestrator_instance: The configured ConversationOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        logger.warning("Orchestrator already initialized, replacing existing instance")
    _orchestrator_instance = orchestrator_instance
    logger.info("Global orchestrator instance initialized")


def get_orchestrator() -> ContinuumOrchestrator:
    """
    Get the global orchestrator instance.
    
    Returns:
        The singleton ConversationOrchestrator instance
        
    Raises:
        RuntimeError: If orchestrator has not been initialized
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        raise RuntimeError(
            "Orchestrator not initialized. Ensure initialize_orchestrator() "
            "is called during application startup."
        )
    return _orchestrator_instance
