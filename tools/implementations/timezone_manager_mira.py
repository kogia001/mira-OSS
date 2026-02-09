"""
MIRA-OSS Timezone Manager Tool

Manages user timezone preferences with support for temporary overrides.
Designed to drive the HUD clock display with proper DST handling.

Follows MIRA patterns from HOW_TO_BUILD_A_TOOL.md:
- Tool Base Class (tools/repo.py:40-143)
- Deferred Initialization (reminder_tool.py:114-145)
- Operation Routing (contacts_tool.py:173-220)
- Response Formatting (reminder_tool.py:340-354)
- Timezone Handling (reminder_tool.py:788-852)
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import (
    utc_now, 
    format_utc_iso, 
    parse_utc_time_string,
    convert_from_utc,
)
from utils.user_context import has_user_context, get_user_preferences

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class TimezoneManagerConfig(BaseModel):
    """Configuration for timezone manager tool."""
    enabled: bool = Field(default=True, description="Whether the tool is enabled")


# Register with MIRA
registry.register("timezone_manager", TimezoneManagerConfig)


# =============================================================================
# Timezone Aliases
# =============================================================================

TIMEZONE_ALIASES = {
    # User's common locations
    "athens": "Europe/Athens",
    "greece": "Europe/Athens",
    "uk": "Europe/London",
    "london": "Europe/London",
    "belfast": "Europe/London",
    "nottingham": "Europe/London",
    # Common shortcuts
    "utc": "UTC",
    "gmt": "Europe/London",
    "est": "America/New_York",
    "pst": "America/Los_Angeles",
    "cet": "Europe/Paris",
}


def resolve_timezone_alias(tz: str) -> str:
    """Resolve a timezone alias to IANA format, or validate existing."""
    if not tz:
        raise ValueError("Timezone is required")
    
    lower = tz.lower().strip()
    if lower in TIMEZONE_ALIASES:
        return TIMEZONE_ALIASES[lower]
    
    # Validate it's a real IANA timezone
    try:
        ZoneInfo(tz)
        return tz
    except KeyError:
        available = ", ".join(sorted(TIMEZONE_ALIASES.keys()))
        raise ValueError(f"Invalid timezone: '{tz}'. Use IANA format (e.g., 'Europe/Athens') or alias ({available})")


# =============================================================================
# Tool Implementation
# =============================================================================

class TimezoneManagerTool(Tool):
    """
    Manage timezone settings for the HUD clock display.
    
    Supports:
    - Getting current timezone state (for HUD)
    - Setting default (home) timezone
    - Setting temporary override (e.g., when traveling)
    - Clearing override
    - Viewing change history
    """
    
    name = "timezone_manager"
    
    simple_description = "Manage timezone settings for clock display"
    
    anthropic_schema = {
        "name": "timezone_manager",
        "description": (
            "Manage user timezone settings for the HUD clock display. "
            "Supports getting current state, setting default timezone, "
            "setting temporary overrides (e.g., when traveling), and viewing history. "
            "Timezone aliases: athens, greece, uk, london, belfast, nottingham, utc, gmt, est, pst, cet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get", "set_default", "set_override", "clear_override", "history"],
                    "description": (
                        "Operation to perform: "
                        "'get' returns current timezone state for HUD; "
                        "'set_default' changes permanent home timezone; "
                        "'set_override' sets temporary working timezone; "
                        "'clear_override' removes temporary override; "
                        "'history' shows recent changes."
                    )
                },
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone (e.g., 'Europe/Athens', 'Europe/London') or alias "
                        "('athens', 'uk', 'belfast'). Required for set_default and set_override."
                    )
                },
                "until": {
                    "type": "string",
                    "description": (
                        "When override expires. Supports: ISO datetime ('2025-02-05T18:00:00'), "
                        "relative ('2h', '3d', '1w'), or weekday ('friday 18:00'). "
                        "Optional for set_override; omit for indefinite."
                    )
                },
                "reason": {
                    "type": "string",
                    "description": "Optional note explaining the change (stored in history)."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of history entries to return (default: 10)."
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Deferred initialization - only create tables when user context exists
        # (prevents startup failures during tool discovery)
        if has_user_context():
            self._ensure_tables()
    
    def _ensure_tables(self):
        """Create tables if they don't exist."""
        # Override columns table - just extends users conceptually
        # In practice, we store override state in a tool-specific table
        # to avoid modifying the core users table
        
        # Current override state (one row per user, managed by tool)
        override_schema = """
            id TEXT PRIMARY KEY,
            encrypted__working_timezone TEXT,
            working_timezone_until TEXT,
            updated_at TEXT NOT NULL
        """
        self.db.create_table('timezone_override', override_schema)
        
        # History table for audit trail
        history_schema = """
            id TEXT PRIMARY KEY,
            encrypted__timezone TEXT NOT NULL,
            kind TEXT NOT NULL,
            starts_at TEXT NOT NULL,
            ends_at TEXT,
            encrypted__reason TEXT,
            changed_by TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        """
        self.db.create_table('timezone_history', history_schema)
        
        # Index for history queries
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_timezone_history_created "
            "ON timezone_history(created_at DESC)"
        )
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a timezone management operation.
        
        Pattern: reminder_tool.py:179-252 (operation routing)
        """
        # Ensure tables exist on first use
        self._ensure_tables()
        
        operation = kwargs.get("operation", "").lower()
        
        if not operation:
            return {
                "success": False,
                "message": "Operation is required. Use: get, set_default, set_override, clear_override, history"
            }
        
        try:
            if operation == "get":
                return self._get_state()
            elif operation == "set_default":
                return self._set_default(
                    timezone=kwargs.get("timezone"),
                    reason=kwargs.get("reason")
                )
            elif operation == "set_override":
                return self._set_override(
                    timezone=kwargs.get("timezone"),
                    until=kwargs.get("until"),
                    reason=kwargs.get("reason")
                )
            elif operation == "clear_override":
                return self._clear_override(reason=kwargs.get("reason"))
            elif operation == "history":
                return self._get_history(limit=kwargs.get("limit", 10))
            else:
                return {
                    "success": False,
                    "message": f"Unknown operation: '{operation}'. Use: get, set_default, set_override, clear_override, history"
                }
        except ValueError as e:
            self.logger.error(f"Validation error in {operation}: {e}")
            return {"success": False, "message": str(e)}
        except Exception as e:
            self.logger.exception(f"Error in timezone_manager.{operation}: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    # =========================================================================
    # Operations
    # =========================================================================
    
    def _get_state(self) -> Dict[str, Any]:
        """
        Get current timezone state for HUD display.
        
        Returns resolved timezone (override if active, else default from user prefs).
        """
        # Get user's default timezone from preferences
        user_prefs = get_user_preferences()
        default_tz = user_prefs.timezone or "UTC"
        
        # Get any active override
        override_rows = self.db.select('timezone_override', 'id = :id', {'id': 'current'})
        
        override_tz = None
        override_until = None
        override_active = False
        
        if override_rows:
            row = override_rows[0]
            working_tz = row.get('encrypted__working_timezone')
            until_str = row.get('working_timezone_until')
            
            if working_tz:
                # Check if override is still valid
                if until_str:
                    until_dt = parse_utc_time_string(until_str)
                    if until_dt and until_dt > utc_now():
                        override_tz = working_tz
                        override_until = until_str
                        override_active = True
                else:
                    # No expiry = indefinite override
                    override_tz = working_tz
                    override_active = True
        
        # Resolve timezone
        resolved_tz = override_tz if override_active else default_tz
        
        # Get current time in resolved timezone
        try:
            tz_info = ZoneInfo(resolved_tz)
            current_local = datetime.now(tz_info)
            utc_offset = current_local.strftime("%z")
            utc_offset_formatted = f"{utc_offset[:3]}:{utc_offset[3:]}" if len(utc_offset) >= 5 else utc_offset
        except Exception:
            current_local = utc_now()
            utc_offset_formatted = "+00:00"
        
        return {
            "success": True,
            "state": {
                "default_timezone": default_tz,
                "override_timezone": override_tz,
                "override_expires_at": override_until,
                "override_active": override_active,
                "resolved_timezone": resolved_tz,
                "current_local_time": current_local.isoformat(),
                "utc_offset": utc_offset_formatted,
            },
            "message": f"Current timezone: {resolved_tz}" + (" (override)" if override_active else "")
        }
    
    def _set_default(self, timezone: Optional[str], reason: Optional[str]) -> Dict[str, Any]:
        """
        Set the user's default (home) timezone.
        
        Note: This updates user preferences, not a tool-specific table.
        The actual preference update would need to go through the user prefs system.
        For now, we log to history and return guidance.
        """
        if not timezone:
            return {"success": False, "message": "timezone is required for set_default"}
        
        try:
            tz = resolve_timezone_alias(timezone)
        except ValueError as e:
            return {"success": False, "message": str(e)}
        
        # Log to history
        self._log_history(tz, "default", reason=reason)
        
        self.logger.info(f"Set default timezone request: {tz}")
        
        return {
            "success": True,
            "timezone": tz,
            "message": (
                f"To set default timezone to '{tz}', update user preferences. "
                f"History logged. Use set_override for temporary changes."
            )
        }
    
    def _set_override(
        self,
        timezone: Optional[str],
        until: Optional[str],
        reason: Optional[str]
    ) -> Dict[str, Any]:
        """Set a temporary timezone override."""
        if not timezone:
            return {"success": False, "message": "timezone is required for set_override"}
        
        try:
            tz = resolve_timezone_alias(timezone)
        except ValueError as e:
            return {"success": False, "message": str(e)}
        
        # Parse expiry
        until_dt = None
        until_str = None
        if until:
            until_dt = self._parse_relative_time(until)
            if not until_dt:
                until_dt = parse_utc_time_string(until)
            if not until_dt:
                return {"success": False, "message": f"Could not parse 'until': {until}"}
            until_str = format_utc_iso(until_dt)
        
        timestamp = format_utc_iso(utc_now())
        
        # Upsert override record
        existing = self.db.select('timezone_override', 'id = :id', {'id': 'current'})
        
        if existing:
            self.db.update(
                'timezone_override',
                {
                    'encrypted__working_timezone': tz,
                    'working_timezone_until': until_str,
                    'updated_at': timestamp
                },
                'id = :id',
                {'id': 'current'}
            )
        else:
            self.db.insert('timezone_override', {
                'id': 'current',
                'encrypted__working_timezone': tz,
                'working_timezone_until': until_str,
                'updated_at': timestamp
            })
        
        # Log to history
        self._log_history(tz, "override", ends_at=until_dt, reason=reason)
        
        expiry_msg = f" until {until}" if until else " (indefinite)"
        self.logger.info(f"Set override timezone: {tz}{expiry_msg}")
        
        return self._get_state()
    
    def _clear_override(self, reason: Optional[str]) -> Dict[str, Any]:
        """Clear any active timezone override."""
        # Check if there's an override to clear
        existing = self.db.select('timezone_override', 'id = :id', {'id': 'current'})
        
        if not existing or not existing[0].get('encrypted__working_timezone'):
            return {
                "success": True,
                "message": "No active override to clear"
            }
        
        old_tz = existing[0].get('encrypted__working_timezone')
        
        # Clear the override
        self.db.update(
            'timezone_override',
            {
                'encrypted__working_timezone': None,
                'working_timezone_until': None,
                'updated_at': format_utc_iso(utc_now())
            },
            'id = :id',
            {'id': 'current'}
        )
        
        # Close out history entry
        self.db.execute(
            "UPDATE timezone_history SET ends_at = :now WHERE kind = 'override' AND ends_at IS NULL",
            {'now': format_utc_iso(utc_now())}
        )
        
        self.logger.info(f"Cleared override timezone (was: {old_tz})")
        
        result = self._get_state()
        result["message"] = f"Override cleared (was: {old_tz}). " + result.get("message", "")
        return result
    
    def _get_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get timezone change history."""
        limit = min(max(1, limit), 50)  # Clamp to 1-50
        
        rows = self.db.execute(
            f"SELECT * FROM timezone_history ORDER BY created_at DESC LIMIT {limit}"
        ).fetchall()
        
        # Convert to dicts (Row objects)
        history = []
        for row in rows:
            entry = {
                "id": row["id"],
                "timezone": row["encrypted__timezone"],
                "kind": row["kind"],
                "starts_at": row["starts_at"],
                "ends_at": row["ends_at"],
                "reason": row["encrypted__reason"],
                "changed_by": row["changed_by"],
                "created_at": row["created_at"],
            }
            history.append(entry)
        
        return {
            "success": True,
            "history": history,
            "count": len(history),
            "message": f"Showing {len(history)} history entries"
        }
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _log_history(
        self,
        timezone: str,
        kind: str,
        ends_at: Optional[datetime] = None,
        reason: Optional[str] = None
    ):
        """Log a timezone change to history."""
        try:
            entry_id = f"tz_{uuid.uuid4().hex[:8]}"
            timestamp = format_utc_iso(utc_now())
            
            self.db.insert('timezone_history', {
                'id': entry_id,
                'encrypted__timezone': timezone,
                'kind': kind,
                'starts_at': timestamp,
                'ends_at': format_utc_iso(ends_at) if ends_at else None,
                'encrypted__reason': reason,
                'changed_by': 'tool',
                'created_at': timestamp
            })
        except Exception as e:
            self.logger.warning(f"Failed to log timezone history: {e}")
    
    def _parse_relative_time(self, value: str) -> Optional[datetime]:
        """
        Parse relative time expressions.
        
        Pattern: reminder_tool.py:788-852 (natural language dates)
        
        Supported:
        - Xh, X hours: X hours from now
        - Xd, X days: X days from now  
        - Xw, X weeks: X weeks from now
        - friday, monday 09:00: Next weekday occurrence
        """
        import re
        
        value = value.strip().lower()
        now = utc_now()
        
        # Hours
        match = re.match(r"^(\d+)\s*(h|hour|hours?)$", value)
        if match:
            return now + timedelta(hours=int(match.group(1)))
        
        # Days
        match = re.match(r"^(\d+)\s*(d|day|days?)$", value)
        if match:
            return now + timedelta(days=int(match.group(1)))
        
        # Weeks
        match = re.match(r"^(\d+)\s*(w|week|weeks?)$", value)
        if match:
            return now + timedelta(weeks=int(match.group(1)))
        
        # Weekdays
        weekdays = {
            "monday": 0, "mon": 0,
            "tuesday": 1, "tue": 1,
            "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3,
            "friday": 4, "fri": 4,
            "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6,
        }
        
        for day_name, day_num in weekdays.items():
            if value.startswith(day_name):
                # Calculate next occurrence
                days_ahead = day_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target = now + timedelta(days=days_ahead)
                
                # Check for time spec
                time_match = re.search(r"(\d{1,2}):(\d{2})", value)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    # Default to end of day
                    target = target.replace(hour=23, minute=59, second=59, microsecond=0)
                
                return target
        
        return None


# =============================================================================
# HUD Helper Functions
# =============================================================================

def get_hud_timezone(user_id: str = None) -> str:
    """
    Quick helper to get resolved timezone for HUD display.
    
    Returns IANA timezone string. HUD can use with Intl.DateTimeFormat
    for automatic DST handling.
    """
    tool = TimezoneManagerTool()
    result = tool.run(operation="get")
    
    if result.get("success") and result.get("state"):
        return result["state"]["resolved_timezone"]
    return "UTC"


def get_hud_time_data(user_id: str = None) -> Dict[str, Any]:
    """
    Get complete time data for HUD display.
    
    Returns dict with timezone, local_time, utc_offset, is_override.
    """
    tool = TimezoneManagerTool()
    result = tool.run(operation="get")
    
    if result.get("success") and result.get("state"):
        state = result["state"]
        return {
            "timezone": state["resolved_timezone"],
            "local_time": state["current_local_time"],
            "utc_offset": state["utc_offset"],
            "is_override": state["override_active"],
            "override_expires": state["override_expires_at"],
        }
    
    # Fallback
    now = utc_now()
    return {
        "timezone": "UTC",
        "local_time": now.isoformat(),
        "utc_offset": "+00:00",
        "is_override": False,
        "override_expires": None,
    }
