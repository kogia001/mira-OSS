"""
System Prompt Composer

Handles in-memory composition of system prompts by collecting
sections from trinkets and assembling them in a defined order.
"""
import logging
import re
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Placement constants (match TrinketPlacement enum values)
PLACEMENT_SYSTEM = "system"
PLACEMENT_NOTIFICATION = "notification"


class SectionData(NamedTuple):
    """Data for a single section."""
    content: str
    cache_policy: bool
    placement: str


@dataclass
class ComposerConfig:
    """Configuration for the system prompt composer."""
    # Display order for all sections (placement determines which group they're in)
    section_order: List[str] = field(default_factory=lambda: [
        # System prompt sections
        'base_prompt',
        'domaindoc',
        'tool_guidance',
        'tool_hints',
        # Notification center sections
        'datetime_section',
        'conversation_manifest',
        'active_reminders',
        'context_search_results',
        'relevant_memories',
    ])
    section_separator: str = "\n\n---\n\n"
    strip_empty_sections: bool = True


class SystemPromptComposer:
    """
    Composes system prompts by collecting and ordering sections.

    This composer provides a clean interface for trinkets to contribute
    sections and handles the final assembly in a predictable order.
    """

    def __init__(self, config: Optional[ComposerConfig] = None):
        """
        Initialize the composer with configuration.

        Args:
            config: Composer configuration. If None, uses defaults.
        """
        self.config = config or ComposerConfig()
        self._sections: Dict[str, SectionData] = {}

        logger.info(f"SystemPromptComposer initialized with {len(self.config.section_order)} ordered sections")
    
    def set_base_prompt(self, prompt: str) -> None:
        """
        Set the base system prompt.

        Args:
            prompt: Base system prompt that always appears first
        """
        # Add delimiter after base prompt to visually separate from injected content
        delimiter = "═" * 60
        scaffolding_note = (
            "Everything after this delimiter is part of MIRA's scaffolding, "
            "injected to provide additional context during the reply."
        )
        delimited_prompt = f"{prompt}\n\n{delimiter}\n{scaffolding_note}\n{delimiter}"

        self._sections['base_prompt'] = SectionData(
            content=delimited_prompt,
            cache_policy=True,
            placement=PLACEMENT_SYSTEM
        )
        logger.debug(f"Set base prompt ({len(prompt)} chars)")

    def add_section(
        self,
        name: str,
        content: str,
        cache_policy: bool = False,
        placement: str = PLACEMENT_SYSTEM
    ) -> None:
        """
        Add or update a section.

        Args:
            name: Section name (e.g., 'datetime_section', 'active_reminders')
            content: Section content (can include formatting)
            cache_policy: Whether this section should be cached (default False)
            placement: Where content appears - PLACEMENT_SYSTEM or PLACEMENT_NOTIFICATION
        """
        if not content or not content.strip():
            logger.debug(f"Skipping empty section '{name}'")
            return

        self._sections[name] = SectionData(
            content=content,
            cache_policy=cache_policy,
            placement=placement
        )
        logger.debug(f"Added section '{name}' ({len(content)} chars, placement={placement})")

    def clear_sections(self, preserve_base: bool = True) -> None:
        """
        Clear all sections.

        Args:
            preserve_base: If True, keeps the base_prompt section
        """
        base_data = self._sections.get('base_prompt') if preserve_base else None
        self._sections.clear()
        if base_data:
            self._sections['base_prompt'] = base_data
        logger.debug(f"Cleared sections (preserved_base={preserve_base})")
    
    def compose(self) -> Dict[str, str]:
        """
        Compose system prompt and notification center content.

        Sections are routed based on their placement attribute:
        - PLACEMENT_SYSTEM: Goes in system prompt (cached/non-cached based on cache_policy)
        - PLACEMENT_NOTIFICATION: Goes in notification center (slides forward each turn)

        Returns:
            Dictionary with:
            - 'cached_content': Static system prompt content (base_prompt + domaindoc)
            - 'non_cached_content': Dynamic system prompt content (tool_guidance, tool_hints)
            - 'notification_center': Sliding assistant message content (time, memories, etc.)
        """
        if not self._sections:
            logger.warning("No sections to compose! This means no base prompt.")
            return {"cached_content": "", "non_cached_content": "", "notification_center": ""}

        # Route sections by placement, maintaining configured order
        cached_parts = []
        non_cached_parts = []
        notification_parts = []

        for section_name in self.config.section_order:
            if section_name not in self._sections:
                continue

            section = self._sections[section_name]
            if self.config.strip_empty_sections and not section.content.strip():
                continue

            if section.placement == PLACEMENT_NOTIFICATION:
                notification_parts.append(section.content)
            elif section.cache_policy:
                cached_parts.append(section.content)
            else:
                non_cached_parts.append(section.content)

        # Build notification center from collected parts
        notification_center = self._build_notification_center(notification_parts)

        # Join and clean system prompt content
        cached_content = self._clean_content(self.config.section_separator.join(cached_parts))
        non_cached_content = self._clean_content(self.config.section_separator.join(non_cached_parts))

        logger.info(
            f"Composed: {len(cached_parts)} cached ({len(cached_content)} chars), "
            f"{len(non_cached_parts)} non-cached ({len(non_cached_content)} chars), "
            f"{len(notification_parts)} notification ({len(notification_center)} chars)"
        )

        return {
            "cached_content": cached_content,
            "non_cached_content": non_cached_content,
            "notification_center": notification_center
        }

    def _build_notification_center(self, parts: List[str]) -> str:
        """
        Build notification center content from parts.

        The notification center is an assistant message that slides forward
        each turn, containing dynamic context like time, memories, and reminders.

        Args:
            parts: List of content strings to include

        Returns:
            Formatted notification center content or empty string if no content
        """
        if not parts:
            return ""

        # Build formatted notification center
        # Opening delimiter is provided by the assistant message in orchestrator
        lines = [
            "NOTIFICATION CENTER",
            "Runtime state. Authoritative for current context.",
            "Provides: temporal orientation, conversation structure, pending tasks, relevant memories.",
            "Keep this front-of-mind while generating the reply.",
            "",
        ]

        for content in parts:
            lines.append(content)
            lines.append("")

        lines.append("═" * 60)

        return "\n".join(lines)

    def _clean_content(self, content: str) -> str:
        """Clean up excessive whitespace in content."""
        # Replace 3+ newlines with exactly 2 newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
