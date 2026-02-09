"""
Fingerprint generator service for retrieval-optimized query expansion.

Transforms fragmentary user queries into detailed, specific queries
optimized for finding relevant memories via embedding similarity.

Key principle: The fingerprint REPLACES the original query for retrieval,
rather than augmenting it. Research showed this approach outperforms
query augmentation for personal memory search.

Also handles memory retention decisions - evaluating which previously
surfaced memories should remain in context based on conversation trajectory.
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional

from cns.core.continuum import Continuum
from clients.vault_client import get_api_key
from utils.tag_parser import format_memory_id

logger = logging.getLogger(__name__)

# Number of user/assistant pairs to include as context (6 pairs = 12 messages)
CONTEXT_PAIRS = 6


class FingerprintGenerator:
    """
    Generates retrieval-optimized memory fingerprints from user messages.

    Uses a fast model (Groq) to expand fragmentary queries into detailed
    specifics that match stored memory vocabulary for better embedding similarity.
    """

    def __init__(self, analysis_enabled: bool, llm_provider):
        """
        Initialize fingerprint generator.

        Args:
            analysis_enabled: Whether fingerprint generation is enabled
            llm_provider: LLM provider for fingerprint generation calls

        Raises:
            FileNotFoundError: If prompt files not found
            ValueError: If API key not found in Vault
            RuntimeError: If fingerprint generation is disabled
        """
        self.llm_provider = llm_provider

        if not analysis_enabled:
            raise RuntimeError(
                "FingerprintGenerator requires analysis_enabled=True"
            )

        # Load prompt templates
        system_prompt_path = Path("config/prompts/fingerprint_expansion_system.txt")
        user_prompt_path = Path("config/prompts/fingerprint_expansion_user.txt")

        if not system_prompt_path.exists():
            raise FileNotFoundError(
                f"Fingerprint system prompt not found at {system_prompt_path}"
            )

        if not user_prompt_path.exists():
            raise FileNotFoundError(
                f"Fingerprint user prompt not found at {user_prompt_path}"
            )

        with open(system_prompt_path, 'r') as f:
            self.system_prompt = f.read()

        with open(user_prompt_path, 'r') as f:
            self.user_prompt_template = f.read()

        # Get LLM config from database
        from utils.user_context import get_internal_llm
        llm_config = get_internal_llm('analysis')
        self._llm_config = llm_config

        # Get API key for LLM endpoint (None for local providers like Ollama)
        if llm_config.api_key_name:
            self.api_key = get_api_key(llm_config.api_key_name)
            if not self.api_key:
                raise ValueError(
                    f"API key '{llm_config.api_key_name}' not found in Vault"
                )
        else:
            self.api_key = None  # Local provider (Ollama) - no API key needed

        logger.info(
            f"FingerprintGenerator initialized: {llm_config.model} @ "
            f"{llm_config.endpoint_url}"
        )

    def generate_fingerprint(
        self,
        continuum: Continuum,
        current_user_message: str,
        previous_memories: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Set[str]]:
        """
        Generate retrieval-optimized memory fingerprint and evaluate retention.

        Expands fragmentary queries into detailed specifics:
        - Resolves "that", "it", "the one" to concrete references
        - Expands implicit context to explicit names, places, dates
        - Outputs vocabulary that matches stored memories

        Also evaluates which previously surfaced memories should remain in context.

        Args:
            continuum: Current continuum with message history
            current_user_message: User message to expand
            previous_memories: List of memory dicts from previous turn (optional)

        Returns:
            Tuple of (fingerprint, pinned_ids):
            - fingerprint: Expanded query string for embedding and retrieval
            - pinned_ids: Set of 8-char memory IDs marked for retention ([x])

        Raises:
            RuntimeError: On generation failure
        """
        try:
            conversation_turns = self._format_recent_turns(
                continuum,
                current_user_message
            )

            # Format previous memories if provided (now includes 8-char IDs)
            memories_block = self._format_previous_memories(previous_memories)

            user_message = self.user_prompt_template.replace(
                "{conversation_turns}",
                conversation_turns
            ).replace(
                "{user_message}",
                current_user_message
            ).replace(
                "{previous_memories}",
                memories_block
            )

            logger.debug(f"Generating fingerprint for: {current_user_message[:100]}...")
            if previous_memories:
                logger.debug(f"Evaluating retention for {len(previous_memories)} memories")

            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": user_message}],
                stream=False,
                endpoint_url=self._llm_config.endpoint_url,
                model_override=self._llm_config.model,
                api_key_override=self.api_key,
                system_override=self.system_prompt
            )

            response_text = self.llm_provider.extract_text_content(response).strip()

            if not response_text:
                raise RuntimeError("Fingerprint generation returned empty response")

            # Parse fingerprint and pinned IDs from response
            fingerprint, pinned_ids = self._parse_response(
                response_text,
                previous_memories
            )

            logger.info(f"Generated fingerprint: {fingerprint[:150]}...")
            if previous_memories:
                logger.info(
                    f"Retention: {len(pinned_ids)}/{len(previous_memories)} memories pinned"
                )

            return fingerprint, pinned_ids

        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Fingerprint generation failed: {str(e)}") from e

    @staticmethod
    def _importance_to_dots(importance_score: float) -> str:
        """
        Convert importance score (0.0-1.0) to 5-dot visual indicator.

        Scale:
            ●●●●● = 0.8-1.0 (high importance)
            ●●●●○ = 0.6-0.8
            ●●●○○ = 0.4-0.6
            ●●○○○ = 0.2-0.4
            ●○○○○ = 0.0-0.2 (low importance)
        """
        score = max(0.0, min(1.0, importance_score))
        filled = int(score * 5) + (1 if score > 0 else 0)  # At least 1 dot if score > 0
        filled = min(5, max(1, filled)) if score > 0 else 1
        return "●" * filled + "○" * (5 - filled)

    def _format_previous_memories(
        self,
        memories: Optional[List[Dict[str, Any]]]
    ) -> str:
        """
        Format previous memories for the prompt with 8-char IDs and importance indicators.

        Args:
            memories: List of memory dicts with 'id', 'text', and 'importance_score' fields

        Returns:
            Formatted memory block or empty string if no memories
        """
        if not memories:
            return ""

        lines = ["\n<previous_memories>"]
        for memory in memories:
            text = memory.get('text', '')
            memory_id = memory.get('id', '')
            importance = memory.get('importance_score', 0.5)
            formatted_id = format_memory_id(memory_id)
            dots = FingerprintGenerator._importance_to_dots(importance)
            if text and formatted_id:
                lines.append(f"- {formatted_id} [{dots}] - {text}")
            elif text:
                # Fallback if no ID (shouldn't happen but be defensive)
                lines.append(f"- [{dots}] - {text}")
        lines.append("</previous_memories>\n")

        return "\n".join(lines)

    def _parse_response(
        self,
        response_text: str,
        previous_memories: Optional[List[Dict[str, Any]]]
    ) -> Tuple[str, Set[str]]:
        """
        Parse fingerprint and pinned memory IDs from LLM response.

        Args:
            response_text: Raw LLM response
            previous_memories: Original memories for fallback

        Returns:
            Tuple of (fingerprint, pinned_ids):
            - fingerprint: Expanded query string
            - pinned_ids: Set of 8-char memory IDs marked [x]
        """
        # Extract fingerprint from <fingerprint> tags
        fingerprint_match = re.search(
            r'<fingerprint>(.*?)</fingerprint>',
            response_text,
            re.DOTALL
        )
        if fingerprint_match:
            fingerprint = fingerprint_match.group(1).strip()
        else:
            # Fallback: if no tags, treat entire response as fingerprint (backward compat)
            fingerprint = response_text.strip()
            # Remove any retention block if present
            fingerprint = re.sub(
                r'<memory_retention>.*?</memory_retention>',
                '',
                fingerprint,
                flags=re.DOTALL
            ).strip()

        if not fingerprint:
            raise RuntimeError("Failed to extract fingerprint from response")

        # Extract pinned memory IDs from <memory_retention> tags
        pinned_ids: Set[str] = set()

        if not previous_memories:
            return fingerprint, pinned_ids

        # Use ID mode for real UUID-like memory IDs, legacy text mode otherwise.
        use_id_mode = False
        for memory in previous_memories:
            raw_id = str(memory.get('id', '')).strip()
            clean_id = raw_id.replace('-', '')
            if (
                len(clean_id) >= 8 and
                re.fullmatch(r'[a-fA-F0-9]{8,}', clean_id) is not None
            ):
                use_id_mode = True
                break

        retention_match = re.search(
            r'<memory_retention>(.*?)</memory_retention>',
            response_text,
            re.DOTALL
        )

        if retention_match:
            retention_block = retention_match.group(1)
            # Preferred format: [x] - mem_a1B2c3D4 - memory text
            id_matches = re.findall(
                r'\[x\]\s*-\s*mem_([a-fA-F0-9]{8})',
                retention_block,
                re.IGNORECASE
            )
            if id_matches:
                pinned_ids = {match.lower() for match in id_matches}
                logger.debug(f"Parsed {len(pinned_ids)} pinned IDs from response")
            else:
                # Legacy format fallback: [x] Memory text
                retained_texts: Set[str] = set()
                for line in retention_block.splitlines():
                    text_match = re.match(
                        r'^\s*\[x\]\s*(?:-\s*)?(.*?)\s*$',
                        line,
                        re.IGNORECASE,
                    )
                    if not text_match:
                        continue
                    retained_text = text_match.group(1).strip()
                    if not retained_text:
                        continue
                    retained_text = re.sub(
                        r'^mem_[a-fA-F0-9]{8}\s*-\s*',
                        '',
                        retained_text,
                        flags=re.IGNORECASE,
                    ).strip()
                    if retained_text:
                        retained_texts.add(retained_text)

                if use_id_mode:
                    text_to_id: Dict[str, str] = {}
                    for memory in previous_memories:
                        text = str(memory.get('text', '')).strip()
                        formatted_id = format_memory_id(memory.get('id', ''))
                        if text and formatted_id:
                            text_to_id[text] = formatted_id.replace('mem_', '').lower()
                    pinned_ids = {
                        text_to_id[text]
                        for text in retained_texts
                        if text in text_to_id
                    }
                else:
                    pinned_ids = retained_texts
        else:
            # Conservative fallback when retention block is missing:
            # keep all previously surfaced memories.
            logger.warning(
                "No <memory_retention> block found in response, keeping all memories"
            )
            if use_id_mode:
                pinned_ids = {
                    format_memory_id(memory.get('id', '')).replace('mem_', '').lower()
                    for memory in previous_memories
                    if format_memory_id(memory.get('id', ''))
                }
            else:
                pinned_ids = {
                    str(memory.get('text', '')).strip()
                    for memory in previous_memories
                    if str(memory.get('text', '')).strip()
                }

        return fingerprint, pinned_ids

    def _format_recent_turns(
        self,
        continuum: Continuum,
        current_user_message: str
    ) -> str:
        """
        Format recent conversation turns for context.

        Skips collapsed segment summaries to only include actual conversation pairs.

        Args:
            continuum: Continuum with message cache
            current_user_message: Current user message

        Returns:
            Formatted string with last 6 pairs plus current message
        """
        lines = []
        pairs_found = 0
        i = len(continuum.messages) - 1

        # Walk backwards to extract user/assistant pairs
        while i >= 0 and pairs_found < CONTEXT_PAIRS:
            # Find assistant message (skip segment summaries)
            while i >= 0:
                msg = continuum.messages[i]
                if msg.role == "assistant" and not self._is_segment_summary(msg):
                    break
                i -= 1
            if i < 0:
                break
            assistant_msg = continuum.messages[i]
            i -= 1

            # Find preceding user message
            while i >= 0 and continuum.messages[i].role != "user":
                i -= 1
            if i < 0:
                break
            user_msg = continuum.messages[i]
            i -= 1

            # Prepend pair (we're walking backwards)
            user_content = self._extract_text_content(user_msg.content)
            lines.insert(0, f"Assistant: {assistant_msg.content}")
            lines.insert(0, f"User: {user_content}")
            pairs_found += 1

        # Append current user message
        lines.append(f"User: {current_user_message}")

        return "\n".join(lines)

    def _is_segment_summary(self, message) -> bool:
        """Check if message is a collapsed segment summary."""
        metadata = getattr(message, 'metadata', {}) or {}
        return (
            metadata.get('is_segment_boundary', False) and
            metadata.get('status') == 'collapsed'
        )

    def _extract_text_content(self, content) -> str:
        """Extract text from potentially multimodal content."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = [
                item['text'] for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            ]
            return ' '.join(text_parts) if text_parts else '[non-text content]'

        return str(content)
