"""
Continuum aggregate root for CNS.

Immutable continuum entity that encapsulates business logic
and state transitions without external dependencies.
"""
import logging
from typing import Tuple, List, Optional, Union, Dict, Any
from uuid import UUID, uuid4

from .message import Message
from .state import ContinuumState
from .events import ContinuumEvent

logger = logging.getLogger(__name__)


class Continuum:
    """
    Continuum aggregate root.

    Immutable entity that manages continuum state and business rules.
    All state changes return new continuum instances and domain events.
    """

    def __init__(self, state: ContinuumState):
        """Initialize continuum with state."""
        self._state = state
        self._message_cache = []  # Hot cache of recent messages

    @classmethod
    def create_new(cls, user_id: str) -> 'Continuum':
        """Create a new continuum for user."""
        state = ContinuumState(
            id=uuid4(),
            user_id=user_id
        )
        return cls(state)
    
    @property
    def id(self) -> UUID:
        """Get continuum ID."""
        return self._state.id

    @property
    def user_id(self) -> str:
        """Get user ID."""
        return self._state.user_id

    @property
    def messages(self) -> List[Message]:
        """Get cached messages - must be initialized through ContinuumPool."""
        return self._message_cache

    def apply_cache(self, messages: List[Message]) -> None:
        """
        Apply an externally managed cache update.
        
        Used by hot cache manager to update the cache after operations
        like topic-based pruning and summary insertion.
        
        Args:
            messages: New message cache to apply
        """
        self._message_cache = messages
    
    def add_user_message(self, content: Union[str, List[Dict[str, Any]]]) -> tuple[Message, List[ContinuumEvent]]:
        """
        Add user message to continuum.

        Returns:
            Tuple of (created Message, list of domain events)
        """
        # Create message with original content for processing
        message = Message(content=content, role="user")

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        return message, []
    
    def add_assistant_message(self, content: str, metadata: dict = None) -> tuple[Message, List[ContinuumEvent]]:
        """
        Add assistant message to continuum.

        Returns:
            Tuple of (created Message, list of domain events)
        """
        # Validate content is not blank
        if not content or not content.strip():
            raise ValueError("Assistant message content cannot be blank or empty")

        # Create message
        message = Message(content=content, role="assistant", metadata=metadata or {})

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        return message, []
    
    def add_tool_message(self, content: str, tool_call_id: str) -> List[ContinuumEvent]:
        """
        Add tool result message to continuum.

        Returns:
            List of domain events (empty for tool messages)
        """
        # Create message
        message = Message(
            content=content,
            role="tool",
            metadata={"tool_call_id": tool_call_id}
        )

        # Add to cache only - persistence will be handled by orchestrator
        self._message_cache.append(message)

        # Tool messages don't generate events by themselves
        return []

    def get_messages_for_api(self) -> List[dict]:
        """Get messages formatted for LLM API with proper prefixes and cache control."""
        from cns.services.segment_helpers import format_segment_for_display
        from utils.timezone_utils import convert_from_utc
        from utils.user_context import get_user_preferences

        # Get user timezone for timestamp injection - no fallback, skip if unavailable
        try:
            user_tz = get_user_preferences().timezone
        except Exception:
            user_tz = None

        formatted_messages = []

        for message in self.messages:  # This uses the property which handles cache loading
            # Format content based on message type
            content = message.content
            if isinstance(content, list):
                content = [block.copy() for block in content]

            # Apply display formatting for collapsed segments
            if (message.metadata.get('is_segment_boundary') and
                message.metadata.get('status') == 'collapsed'):
                content = format_segment_for_display(message)

            # Inject ephemeral timestamps for user/assistant messages (not persisted)
            elif (user_tz is not None and
                  message.role in ("user", "assistant") and
                  not message.metadata.get('is_segment_boundary') and
                  not message.metadata.get('system_notification')):
                local_dt = convert_from_utc(message.created_at, user_tz)
                timestamp = local_dt.strftime("%-I:%M%p").lower()
                if isinstance(content, str):
                    content = f"[{timestamp}] {content}"
                elif isinstance(content, list):
                    # Multimodal: inject into first text block
                    for block in content:
                        if block.get("type") == "text":
                            block["text"] = f"[{timestamp}] {block['text']}"
                            break

            if message.role == "assistant" and message.metadata.get("has_tool_calls", False):
                # Assistant message with tool calls
                msg_dict = {
                    "role": "assistant",
                    "content": content
                }
                if "tool_calls" in message.metadata:
                    msg_dict["tool_calls"] = message.metadata["tool_calls"]
                formatted_messages.append(msg_dict)
            elif message.role == "tool":
                # Tool result message
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": message.metadata.get("tool_call_id"),
                    "content": content
                })
            elif message.role == "user" and isinstance(message.content, list):
                # User message with content blocks (multimodal)
                formatted_messages.append({
                    "role": "user",
                    "content": content  # Keep original for multimodal
                })
            else:
                # Standard text message
                formatted_messages.append({
                    "role": message.role,
                    "content": content
                })

        # Apply cache_control to last assistant message for conversation history caching
        # Anthropic ignores cache markers on content < 1024 tokens, so always mark
        # and let the API handle threshold logic. This keeps us stateless per-request.
        for i in range(len(formatted_messages) - 1, -1, -1):
            if formatted_messages[i]["role"] == "assistant":
                content = formatted_messages[i]["content"]

                # Ensure content is structured as blocks (required for cache_control)
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                # Apply cache_control to the last content block
                if isinstance(content, list) and len(content) > 0:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                    formatted_messages[i]["content"] = content
                break

        return formatted_messages
    
    def to_dict(self) -> dict:
        """Convert continuum to dictionary for persistence."""
        return self._state.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'Continuum':
        """Create continuum from dictionary."""
        state = ContinuumState.from_dict(data)
        return cls(state)
