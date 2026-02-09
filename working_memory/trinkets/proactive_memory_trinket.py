"""Proactive memory trinket for displaying relevant long-term memories."""
import logging
from html import escape as xml_escape
from typing import List, Dict, Any

from utils.tag_parser import format_memory_id
from .base import EventAwareTrinket

logger = logging.getLogger(__name__)


class ProactiveMemoryTrinket(EventAwareTrinket):
    """
    Displays surfaced memories in the notification center.

    This trinket formats memories passed via context into
    a structured section for the sliding notification center.
    """
    
    def __init__(self, event_bus, working_memory):
        """Initialize with memory cache."""
        super().__init__(event_bus, working_memory)
        self._cached_memories = []  # Store memories between updates

    def _get_variable_name(self) -> str:
        """Proactive memory publishes to 'relevant_memories'."""
        return "relevant_memories"

    def get_cached_memories(self) -> List[Dict[str, Any]]:
        """
        Get the currently cached memories.

        Used by the orchestrator for memory retention evaluation -
        previous turn's surfaced memories are evaluated for continued relevance.

        Returns:
            List of memory dicts from the previous turn
        """
        return self._cached_memories
    
    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate memory content from context.
        
        Args:
            context: Update context containing 'memories' list
            
        Returns:
            Formatted memories section or empty string if no memories
        """
        # Update cache if memories are provided
        if 'memories' in context:
            self._cached_memories = context['memories']
        
        # Use cached memories
        if not self._cached_memories:
            return ""
        
        # Format memories for prompt
        memory_content = self._format_memories_for_prompt(self._cached_memories)
        
        logger.debug(f"Formatted {len(self._cached_memories)} memories for display")
        return memory_content
    
    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories as XML with nested linked_memories elements."""
        if not memories:
            return ""

        parts = ["<surfaced_memories>"]

        for memory in memories:
            parts.append(self._format_primary_memory_xml(memory))

        parts.append("</surfaced_memories>")
        return "\n".join(parts)

    def _format_primary_memory_xml(self, memory: Dict[str, Any]) -> str:
        """Format a primary memory as XML with nested linked_memories."""
        from utils.timezone_utils import format_relative_time, parse_time_string, format_datetime

        raw_id = memory.get('id', '')
        formatted_id = format_memory_id(raw_id) if raw_id else 'unknown'
        text = xml_escape(str(memory.get('text', '')), quote=False)

        # Build attributes
        attrs = [f'id="{xml_escape(formatted_id, quote=True)}"']

        confidence = memory.get('confidence') or memory.get('similarity_score')
        if confidence is not None and confidence > 0.75:
            attrs.append(f'confidence="{int(confidence * 100)}"')

        parts = [f"<memory {' '.join(attrs)}>"]
        parts.append(f"<text>{text}</text>")

        # Created time
        if memory.get('created_at'):
            created_dt = parse_time_string(memory['created_at'])
            relative_time = format_relative_time(created_dt)
            parts.append(f"<created>{relative_time}</created>")

        # Temporal info
        temporal_attrs = []
        if memory.get('expires_at'):
            expires_dt = parse_time_string(memory['expires_at'])
            expiry_date = format_datetime(expires_dt, 'date')
            temporal_attrs.append(f'expires="{xml_escape(expiry_date, quote=True)}"')
        if memory.get('happens_at'):
            happens_dt = parse_time_string(memory['happens_at'])
            event_date = format_datetime(happens_dt, 'date')
            temporal_attrs.append(f'happens="{xml_escape(event_date, quote=True)}"')
        if temporal_attrs:
            parts.append(f"<temporal {' '.join(temporal_attrs)}/>")

        # Linked memories (nested)
        linked_memories = memory.get('linked_memories', [])
        if linked_memories:
            parts.append(self._format_linked_memories_xml(linked_memories, current_depth=1))

        parts.append("</memory>")
        return "\n".join(parts)

    def _format_linked_memories_xml(
        self,
        linked_memories: List[Dict[str, Any]],
        current_depth: int = 1,
        max_display_depth: int = 2
    ) -> str:
        """
        Recursively format linked memories as nested XML elements.

        NOTE: max_display_depth is distinct from traversal depth:
        - Traversal depth (config.max_link_traversal_depth): How deep to walk the graph
        - Display depth (this parameter): How many levels to show in output

        We may traverse 3-4 levels deep to discover important memories,
        but only display Primary + 2 levels to avoid context window bloat.

        Args:
            linked_memories: List of linked memory dicts
            current_depth: Current display depth (1-indexed)
            max_display_depth: Maximum depth to display (default 2 = Primary + 2 levels)
        """
        # Stop display if we've reached max depth
        if current_depth > max_display_depth:
            return ""

        if not linked_memories:
            return ""

        parts = ["<linked_memories>"]

        for linked in linked_memories:
            # Link metadata
            link_meta = linked.get('link_metadata', {})
            link_type = link_meta.get('link_type', 'unknown')
            confidence = link_meta.get('confidence')

            # Build attributes
            raw_id = linked.get('id', '')
            formatted_id = format_memory_id(raw_id) if raw_id else 'unknown'
            attrs = [
                f'id="{xml_escape(formatted_id, quote=True)}"',
                f'link_type="{xml_escape(str(link_type), quote=True)}"',
            ]
            if confidence is not None and confidence > 0.75:
                attrs.append(f'confidence="{int(confidence * 100)}"')

            parts.append(f"<linked_memory {' '.join(attrs)}>")
            linked_text = xml_escape(str(linked.get('text', '')), quote=False)
            parts.append(f"<text>{linked_text}</text>")

            # Nested linked memories (recursive)
            nested_linked = linked.get('linked_memories', [])
            if nested_linked:
                nested_xml = self._format_linked_memories_xml(
                    nested_linked,
                    current_depth=current_depth + 1,
                    max_display_depth=max_display_depth
                )
                if nested_xml:
                    parts.append(nested_xml)

            parts.append("</linked_memory>")

        parts.append("</linked_memories>")
        return "\n".join(parts)
