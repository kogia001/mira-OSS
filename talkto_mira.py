#!/usr/bin/env python3
"""
MIRA Chat - Rich-based CLI for chatting with MIRA.

Usage:
    python talkto_mira.py              # Interactive chat
    python talkto_mira.py --headless "message"  # One-shot query
    python talkto_mira.py --show-key   # Display API key for curl/API usage
"""

import argparse
import atexit
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
from rich.console import Console
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from clients.vault_client import get_api_key

MIRA_API_URL = os.getenv("MIRA_API_URL", "http://localhost:1993")
REQUEST_TIMEOUT = 120
SERVER_STARTUP_TIMEOUT = 30
UPDATE_CHECK_URL = "https://miraos.org/check_update"
# Update check on startup (modified by deploy script based on user preference)
UPDATE_CHECK_ENABLED = True

_server_process = None
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Version & Update Check
# ─────────────────────────────────────────────────────────────────────────────

def get_current_version() -> str:
    """Read current version from VERSION file."""
    version_file = project_root / "VERSION"
    try:
        return version_file.read_text().strip()
    except Exception:
        return "unknown"


def check_for_update() -> tuple[bool, str | None]:
    """Check if a newer version is available.

    Returns (update_available, latest_version).
    Only sends the installed version to the server - no other information.
    """
    if not UPDATE_CHECK_ENABLED:
        return False, None

    current = get_current_version()
    if current == "unknown":
        return False, None

    try:
        response = requests.get(
            UPDATE_CHECK_URL,
            params={"version": current},
            timeout=3
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("update_available"):
                return True, data.get("latest_version")
    except Exception:
        pass  # Silent failure - update check is optional

    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

def strip_emotion_tag(text: str) -> str:
    pattern = r'\n?<mira:my_emotion>.*?</mira:my_emotion>'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


def send_message(token: str, message: str) -> dict:
    url = f"{MIRA_API_URL}/v0/api/chat"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={"message": message}, timeout=REQUEST_TIMEOUT)
        return response.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": {"message": "Request timed out"}}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": {"message": f"Cannot connect to {MIRA_API_URL}"}}
    except Exception as e:
        return {"success": False, "error": {"message": str(e)}}


def call_action(token: str, domain: str, action: str, data: dict = None) -> dict:
    url = f"{MIRA_API_URL}/v0/api/actions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={"domain": domain, "action": action, "data": data or {}}, timeout=10)
        return response.json()
    except Exception as e:
        return {"success": False, "error": {"message": str(e)}}


def fetch_history(token: str, limit: int = 20) -> list[dict]:
    """Fetch recent message history from API."""
    url = f"{MIRA_API_URL}/v0/api/data"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, params={"type": "history", "limit": limit}, timeout=10)
        result = response.json()
        if result.get("success"):
            return result.get("data", {}).get("messages", [])
        return []
    except Exception:
        return []  # Graceful degradation - empty history is acceptable


def fetch_health(token: str) -> dict:
    """Fetch system health from API."""
    url = f"{MIRA_API_URL}/v0/api/health"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        return response.json()
    except Exception:
        return {"data": {"status": "unknown"}}


def fetch_memory_stats(token: str) -> tuple[int, bool]:
    """Fetch memory count. Returns (count, has_more)."""
    url = f"{MIRA_API_URL}/v0/api/data"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, params={"type": "memories", "limit": 100}, timeout=5)
        result = response.json()
        if result.get("success"):
            meta = result.get("data", {}).get("meta", {})
            return meta.get("total_returned", 0), meta.get("has_more", False)
    except Exception:
        pass
    return 0, False


def fetch_user_info(token: str) -> dict | None:
    """Fetch user profile info."""
    url = f"{MIRA_API_URL}/v0/api/data"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, params={"type": "user"}, timeout=5)
        result = response.json()
        if result.get("success"):
            return result.get("data", {})
    except Exception:
        pass
    return None


def fetch_segment_status(token: str) -> dict | None:
    """Fetch segment timeout status via actions API."""
    resp = call_action(token, "continuum", "get_segment_status", {})
    if resp.get("success"):
        return resp.get("data", {})
    return None


def format_time_remaining(collapse_at_iso: str) -> str:
    """Format time remaining until collapse as human-readable string."""
    from datetime import datetime, timezone
    try:
        collapse_at = datetime.fromisoformat(collapse_at_iso.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        remaining = collapse_at - now

        total_seconds = int(remaining.total_seconds())
        if total_seconds <= 0:
            return "expired"

        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    except Exception:
        return "unknown"


def load_recent_pairs(token: str, pair_count: int = 5) -> list[tuple[str, list[str], str]]:
    """Load recent user/assistant message pairs for history display.

    Fetches extra messages to account for tool messages, then extracts
    the most recent N complete user→assistant pairs. Historical pairs
    don't include tool call info (empty list) since we didn't see them live.
    """
    messages = fetch_history(token, limit=pair_count * 4)  # Newest-first

    # Work backwards from newest: find assistant, then its preceding user message
    pairs = []
    i = 0
    while i < len(messages) and len(pairs) < pair_count:
        msg = messages[i]
        if msg.get("role") == "assistant":
            # Look backwards (older) for the user message that prompted this
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "user":
                    user_content = messages[j].get("content", "")
                    assistant_content = msg.get("content", "")
                    # Handle multimodal content (arrays)
                    if isinstance(user_content, list):
                        user_content = next((b.get("text", "") for b in user_content if b.get("type") == "text"), "[media]")
                    if isinstance(assistant_content, list):
                        assistant_content = next((b.get("text", "") for b in assistant_content if b.get("type") == "text"), "")
                    pairs.append((user_content, [], strip_emotion_tag(assistant_content)))
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1

    # Reverse to chronological order for display (oldest first)
    pairs.reverse()
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Preferences (LLM Tier System)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TierInfo:
    """Tier information from API."""
    name: str
    model: str
    description: str
    accessible: bool
    locked_message: str | None = None


def get_tier_info(token: str) -> tuple[str, list[TierInfo]]:
    """Get current tier and available tiers from API."""
    resp = call_action(token, "continuum", "get_llm_tier")
    if resp.get("success"):
        data = resp.get("data", {})
        current = data.get("tier", "balanced")
        tiers = [
            TierInfo(
                name=t["name"],
                model=t.get("model", t.get("description", t["name"])),
                description=t.get("description", t["name"]),
                accessible=t["accessible"],
                locked_message=t.get("locked_message")
            )
            for t in data.get("available_tiers", [])
        ]
        return current, tiers
    return "balanced", []


def set_tier(token: str, tier: str) -> bool:
    """Set LLM tier preference."""
    resp = call_action(token, "continuum", "set_llm_tier", {"tier": tier})
    return resp.get("success", False)


def get_enabled_domaindocs(token: str) -> list[str]:
    """Get list of enabled domaindoc labels."""
    resp = call_action(token, "domain_knowledge", "list", {})
    if resp.get("success"):
        data = resp.get("data", {})
        return [d["label"] for d in data.get("domaindocs", []) if d.get("enabled")]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

def is_api_running() -> bool:
    try:
        response = requests.get(f"{MIRA_API_URL}/v0/api/health", timeout=2)
        return response.status_code in [200, 503]
    except:
        return False


def start_api_server() -> subprocess.Popen:
    global _server_process
    main_py = project_root / "main.py"
    if not main_py.exists():
        raise RuntimeError(f"Cannot find main.py at {main_py}")
    _server_process = subprocess.Popen([sys.executable, str(main_py)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(project_root))
    return _server_process


def wait_for_api_ready(timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_api_running():
            return True
        time.sleep(0.5)
    return False


def shutdown_server():
    global _server_process
    if _server_process is not None:
        try:
            _server_process.terminate()
            _server_process.wait(timeout=5)
        except:
            try:
                _server_process.kill()
            except:
                pass
        _server_process = None


# ─────────────────────────────────────────────────────────────────────────────
# Splashscreen
# ─────────────────────────────────────────────────────────────────────────────

ASCII_CHARS = ['.', '+', '*', 'o', "'", '-', '~', '|']


def clear_screen_and_scrollback() -> None:
    """Clear visible screen and scrollback buffer."""
    print("\033[2J\033[3J\033[H", end="", flush=True)


def show_splashscreen(start_server: bool = False) -> bool:
    """
    Animated ASCII splashscreen matching web loading animation.

    If start_server=True, starts the API server and animates until ready.
    Returns True if server was started, False otherwise.
    """
    if console.width < 40:
        if start_server and not is_api_running():
            start_api_server()
            return wait_for_api_ready()
        return False

    clear_screen_and_scrollback()

    width = console.width
    frame_delay = 0.05
    min_frames = 40  # Minimum 2 seconds of animation
    max_frames = int(SERVER_STARTUP_TIMEOUT / frame_delay)  # Max based on server timeout

    # Initialize character line (sparse like web version)
    chars = []
    for i in range(width):
        if i == 0 or i == width - 1 or random.random() < 0.2:
            chars.append(random.choice(ASCII_CHARS))
        else:
            chars.append(' ')

    # Center vertically
    vertical_pos = console.height // 2

    # Start server if requested
    server_started = False
    if start_server and not is_api_running():
        start_api_server()
        server_started = True

    frame = 0
    server_ready = not start_server or is_api_running()  # Already ready if not starting

    while frame < max_frames:
        # Check if server is ready (after minimum animation time)
        if server_started and frame >= min_frames:
            if is_api_running():
                server_ready = True
                break

        # Randomly mutate some characters each frame
        for i in range(width):
            if random.random() < 0.15:
                if random.random() < 0.3:
                    chars[i] = random.choice(ASCII_CHARS)
                else:
                    chars[i] = ' '

        # Render frame
        line = ''.join(chars)
        print(f"\033[{vertical_pos};1H", end="")  # Move cursor to center row
        console.print(line, style="bright_green", end="", highlight=False)
        time.sleep(frame_delay)
        frame += 1

        # If not waiting for server, just run minimum frames
        if not server_started and frame >= min_frames:
            break

    clear_screen_and_scrollback()
    return server_started and server_ready


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_delimiter() -> None:
    """Render a delimiter between message exchanges - dark grey ═══ line."""
    console.print("═" * console.width, style="dim")


def render_message(text: str, prefix: str, style: str) -> None:
    """Render a message with colored prefix, left-aligned with proper continuation indent."""
    lines = text.split('\n')
    content = Text()
    content.append(f"{prefix}: ", style=f"{style} bold")
    content.append(lines[0])
    indent = " " * (len(prefix) + 2)  # +2 for ": "
    for line in lines[1:]:
        content.append(f"\n{indent}{line}")
    console.print(content)


def render_user_message(text: str) -> None:
    """Render a user message - magenta YOU: prefix, left-aligned."""
    render_message(text, "YOU", "magenta")


def render_mira_message(text: str, is_error: bool = False) -> None:
    """Render a MIRA message - green MIRA: prefix, left-aligned."""
    render_message(text, "MIRA", "red" if is_error else "bright_green")


def render_tool_message(tool_name: str) -> None:
    """Render a tool call indicator - cyan TOOL: prefix, left-aligned."""
    render_message(f"⚙ {tool_name}", "TOOL", "cyan")


def render_status_bar(tier: str, available_tiers: list[TierInfo], enabled_docs: list[str] = None) -> None:
    """Render status bar with friendly tier name and enabled domaindocs."""
    # Find friendly description for current tier from available_tiers
    tier_display = next((t.description for t in available_tiers if t.name == tier), tier)
    left_parts = [Text(f" {tier_display}", style="magenta")]

    # Add enabled domaindocs pipe-separated in bright yellow
    if enabled_docs:
        for doc in enabled_docs:
            left_parts.append(Text(" | ", style="dim"))
            left_parts.append(Text(doc, style="bright_yellow"))

    left = Text.assemble(*left_parts)
    right = Text("/help • ctrl+c quit", style="dim")

    padding = console.width - len(left.plain) - len(right.plain)
    console.print(Text.assemble(left, " " * max(padding, 1), right))
    console.print("─" * console.width, style="dim")


def render_screen(
    history: list[tuple[str, list[str], str]],
    tier: str,
    available_tiers: list[TierInfo],
    pending_user_msg: str = None,
    show_thinking: bool = False,
    enabled_docs: list[str] = None
) -> None:
    """Clear and render the full screen with status bar always at bottom."""
    clear_screen_and_scrollback()

    # Calculate content height
    content_lines = len(history) * 8 + 2
    if pending_user_msg:
        content_lines += 4
    if show_thinking:
        content_lines += 4
    terminal_height = console.height

    # Push content to bottom if not enough to fill screen
    if content_lines < terminal_height - 2:
        blank_lines = terminal_height - content_lines - 2
        console.print("\n" * blank_lines, end="")

    # Render history
    for i, (user_msg, tools_used, mira_msg) in enumerate(history):
        render_user_message(user_msg)
        for tool in tools_used:
            render_tool_message(tool)
        render_mira_message(mira_msg)
        console.print()  # Blank line after MIRA's response
        if i < len(history) - 1:
            render_delimiter()
            console.print()  # Blank line after delimiter

    # Render pending message (not yet in history)
    if pending_user_msg:
        if history:
            render_delimiter()
            console.print()
        render_user_message(pending_user_msg)
        console.print()

    # Render thinking indicator
    if show_thinking:
        render_thinking()
        console.print()

    # Status bar always last
    render_status_bar(tier, available_tiers, enabled_docs)


class ThinkingAnimation:
    """Animated bouncing face while waiting for response."""

    FACE = "^_^"
    WIDTH = 12  # Inner width for bouncing

    def __init__(self):
        self.running = False
        self.thread = None
        self.position = 0
        self.direction = 1

    def start(self):
        """Start the animation in a background thread."""
        self.running = True
        self.position = 0
        self.direction = 1
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the animation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        # Clear the animation line
        print(f"\033[1A\033[2K", end="", flush=True)

    def _animate(self):
        """Animation loop - bounces face back and forth."""
        max_pos = self.WIDTH - len(self.FACE)
        while self.running:
            # Build the frame
            left_pad = " " * self.position
            right_pad = " " * (max_pos - self.position)
            frame = f"[ {left_pad}{self.FACE}{right_pad} ]"

            # Print in place (move up, clear line, print)
            print(f"\r{frame}", end="", flush=True)

            # Update position
            self.position += self.direction
            if self.position >= max_pos:
                self.direction = -1
            elif self.position <= 0:
                self.direction = 1

            time.sleep(0.1)


_thinking_animation = ThinkingAnimation()


def render_thinking() -> None:
    """Show thinking indicator (static fallback)."""
    print("[ ^_^        ]", flush=True)


def render_update_alert(latest_version: str) -> None:
    """Render an update available alert."""
    current = get_current_version()
    console.print()
    console.print(f"[yellow]Update available:[/yellow] {current} → {latest_version}")
    console.print("[dim]Visit https://miraos.org to update[/dim]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────

_prompt_session: PromptSession | None = None


def get_user_input(prompt_text: str = "\x1b[36m>\x1b[0m ") -> str:
    """Get user input with paste support and backslash line continuation.

    Uses prompt_toolkit for robust terminal handling:
    - Proper bracketed paste support (fixes macOS libedit issues)
    - Backslash continuation: end line with \\ to continue on next line
    - Command history preserved across prompts
    """
    global _prompt_session
    if _prompt_session is None:
        _prompt_session = PromptSession()

    lines = []
    current_prompt = prompt_text
    continuation_prompt = "  "  # Indent for continuation lines

    while True:
        line = _prompt_session.prompt(ANSI(current_prompt))
        if line.rstrip().endswith('\\'):
            # Line continuation - remove trailing backslash, keep collecting
            lines.append(line.rstrip()[:-1])
            current_prompt = continuation_prompt
        else:
            lines.append(line)
            break

    return '\n'.join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Chat Loop
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(token: str) -> None:
    history: list[tuple[str, list[str], str]] = load_recent_pairs(token, pair_count=5)
    current_tier, available_tiers = get_tier_info(token)
    enabled_docs = get_enabled_domaindocs(token)

    # Mutable state for resize handler (closures capture by reference for mutables)
    prefs = {'tier': current_tier, 'tiers': available_tiers, 'docs': enabled_docs}

    def handle_resize(signum, frame):
        render_screen(history, prefs['tier'], prefs['tiers'], enabled_docs=prefs['docs'])

    # SIGWINCH is Unix-only (terminal window resize)
    if hasattr(signal, 'SIGWINCH'):
        signal.signal(signal.SIGWINCH, handle_resize)

    render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)

    while True:
        try:
            user_input = get_user_input()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'bye'):
            console.print("[dim]Goodbye![/dim]")
            break

        # Slash commands
        if user_input.startswith('/'):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower() if parts else ""
            arg = parts[1].lower() if len(parts) > 1 else None

            if cmd == "help":
                console.print()
                render_mira_message("/tier [name] - view or change model tier\n/domaindoc list|create|enable|disable\n/collapse - end current conversation segment\n/status\n/clear\nquit, exit, bye")
                console.print()

            elif cmd == "status":
                # Fetch all status data
                health = fetch_health(token)
                memory_count, has_more = fetch_memory_stats(token)
                user_info = fetch_user_info(token)
                segment_status = fetch_segment_status(token)
                current_tier, available_tiers = get_tier_info(token)
                prefs['tier'] = current_tier
                prefs['tiers'] = available_tiers

                # Find current tier details
                tier_desc = next((t.description for t in available_tiers if t.name == current_tier), current_tier)

                # Build status lines
                lines = []

                # System health
                health_data = health.get("data", {})
                system_status = health_data.get("status", "unknown")
                db_latency = health_data.get("components", {}).get("database", {}).get("latency_ms")
                latency_str = f" ({db_latency}ms)" if db_latency else ""
                lines.append(f"System      {system_status}{latency_str}")

                # Memory count - show "100+" if there are more
                count_str = f"{memory_count}+" if has_more else str(memory_count)
                lines.append(f"Memories    {count_str} stored")

                # Segment timeout info
                if segment_status:
                    if segment_status.get("has_active_segment"):
                        collapse_at = segment_status.get("collapse_at")
                        if collapse_at:
                            time_remaining = format_time_remaining(collapse_at)
                            postponed = " (extended)" if segment_status.get("is_postponed") else ""
                            lines.append(f"Segment     collapses in {time_remaining}{postponed}")
                        else:
                            lines.append("Segment     active")
                    else:
                        lines.append("Segment     collapsed")

                lines.append("")  # Spacer

                # Tier info
                lines.append(f"Tier        {current_tier} ({tier_desc})")

                lines.append("")  # Spacer

                # User info
                if user_info:
                    profile = user_info.get("profile", {})
                    user_prefs = user_info.get("preferences", {})
                    if profile.get("email"):
                        lines.append(f"User        {profile['email']}")
                    if user_prefs.get("timezone"):
                        lines.append(f"Timezone    {user_prefs['timezone']}")

                console.print()
                render_mira_message("\n".join(lines))
                console.print()

            elif cmd == "tier":
                # Refresh tier info from API
                current_tier, available_tiers = get_tier_info(token)
                prefs['tier'] = current_tier
                prefs['tiers'] = available_tiers
                accessible_tiers = [t for t in available_tiers if t.accessible]
                accessible_names = [t.name for t in accessible_tiers]

                # Resolve arg to tier name (accept either tier name or model string)
                resolved_tier = None
                if arg:
                    if arg in accessible_names:
                        resolved_tier = arg
                    else:
                        # Check if arg matches a model string
                        matched = next((t.name for t in accessible_tiers if t.model == arg), None)
                        if matched:
                            resolved_tier = matched

                if resolved_tier:
                    if set_tier(token, resolved_tier):
                        current_tier = prefs['tier'] = resolved_tier
                        render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)
                    else:
                        console.print()
                        render_mira_message("Failed to set tier", is_error=True)
                        console.print()
                elif arg:
                    # Check if tier exists but is locked
                    locked_tier = next((t for t in available_tiers if (t.name == arg or t.model == arg) and not t.accessible), None)
                    if locked_tier and locked_tier.locked_message:
                        console.print()
                        render_mira_message(f"Tier '{arg}' is locked: {locked_tier.locked_message}", is_error=True)
                        console.print()
                    else:
                        options = ", ".join(accessible_names)
                        console.print()
                        render_mira_message(f"Options: {options}", is_error=True)
                        console.print()
                else:
                    # Show current tier and available options with model names
                    current_model = next((t.model for t in available_tiers if t.name == current_tier), current_tier)
                    tier_lines = [f"Current: {current_tier} ({current_model})", ""]
                    for t in available_tiers:
                        if t.accessible:
                            marker = "→" if t.name == current_tier else " "
                            tier_lines.append(f"  {marker} {t.name}: {t.model}")
                    tier_lines.append("")
                    tier_lines.append("Use /tier <name> to switch")
                    console.print()
                    render_mira_message("\n".join(tier_lines))
                    console.print()

            elif cmd == "clear":
                history.clear()
                render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)

            elif cmd == "collapse":
                resp = call_action(token, "continuum", "collapse_segment", {})
                if resp.get("collapsed"):
                    collapse_msg = (
                        "The previous conversation segment has been collapsed. "
                        "Feel free to continue in a new direction or come back later. "
                        "MIRA will be ready when you return."
                    )
                    history.append(("/collapse", [], collapse_msg))
                    render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)
                else:
                    error_msg = resp.get("error", {}).get("message", "No active segment to collapse")
                    console.print()
                    render_mira_message(f"Failed to collapse segment: {error_msg}", is_error=True)
                    console.print()

            elif cmd == "domaindoc":
                # Get original (non-lowercased) arg for create description
                raw_arg = parts[1] if len(parts) > 1 else None

                if not arg:
                    console.print()
                    render_mira_message("/domaindoc list\n/domaindoc create <label> \"<description>\"\n/domaindoc enable <label>\n/domaindoc disable <label>")
                    console.print()

                elif arg == "list":
                    resp = call_action(token, "domain_knowledge", "list", {})
                    if resp.get("success"):
                        data = resp.get("data", {})
                        docs = data.get("domaindocs", [])
                        if docs:
                            lines = []
                            for d in docs:
                                status = "✓" if d.get("enabled") else "○"
                                lines.append(f"{status} {d['label']}: {d.get('description', '')}")
                            console.print()
                            render_mira_message("\n".join(lines))
                            console.print()
                        else:
                            console.print()
                            render_mira_message("No domaindocs found. Create one with /domaindoc create <label> \"<description>\"")
                            console.print()
                    else:
                        console.print()
                        render_mira_message(f"Error: {resp.get('error', {}).get('message', 'Unknown')}", is_error=True)
                        console.print()

                elif arg.startswith("create "):
                    # Parse: create label "description"
                    import shlex
                    try:
                        create_parts = shlex.split(raw_arg[7:])  # Skip "create "
                        if len(create_parts) >= 2:
                            label = create_parts[0]
                            description = create_parts[1]
                            resp = call_action(token, "domain_knowledge", "create", {"label": label, "description": description})
                            if resp.get("success"):
                                console.print()
                                render_mira_message(f"Created domaindoc '{label}'")
                                console.print()
                            else:
                                console.print()
                                render_mira_message(f"Error: {resp.get('error', {}).get('message', 'Unknown')}", is_error=True)
                                console.print()
                        else:
                            console.print()
                            render_mira_message("Usage: /domaindoc create <label> \"<description>\"", is_error=True)
                            console.print()
                    except ValueError as e:
                        console.print()
                        render_mira_message(f"Error parsing command: {e}", is_error=True)
                        console.print()

                elif arg == "enable" or arg.startswith("enable "):
                    label = arg[7:].strip() if arg.startswith("enable ") else ""
                    if not label:
                        console.print()
                        render_mira_message("Usage: /domaindoc enable <label>", is_error=True)
                        console.print()
                    else:
                        resp = call_action(token, "domain_knowledge", "enable", {"label": label})
                        if resp.get("success"):
                            enabled_docs = get_enabled_domaindocs(token)
                            prefs['docs'] = enabled_docs
                            render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)
                        else:
                            console.print()
                            render_mira_message(f"Error: {resp.get('error', {}).get('message', 'Unknown')}", is_error=True)
                            console.print()

                elif arg == "disable" or arg.startswith("disable "):
                    label = arg[8:].strip() if arg.startswith("disable ") else ""
                    if not label:
                        console.print()
                        render_mira_message("Usage: /domaindoc disable <label>", is_error=True)
                        console.print()
                        continue
                    resp = call_action(token, "domain_knowledge", "disable", {"label": label})
                    if resp.get("success"):
                        enabled_docs = get_enabled_domaindocs(token)
                        prefs['docs'] = enabled_docs
                        render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)
                    else:
                        console.print()
                        render_mira_message(f"Error: {resp.get('error', {}).get('message', 'Unknown')}", is_error=True)
                        console.print()

                else:
                    console.print()
                    render_mira_message(f"Unknown domaindoc command: {arg}\nTry: /domaindoc list, create, enable, disable", is_error=True)
                    console.print()

            else:
                console.print()
                render_mira_message(f"Unknown: /{cmd}", is_error=True)
                console.print()

            continue

        # Regular message - show thinking animation
        # Note: Don't use show_thinking=True here - the animation handles its own rendering
        # Using both creates duplicate indicators (one static above status bar, one animated below)
        render_screen(history, current_tier, available_tiers, pending_user_msg=user_input, show_thinking=False, enabled_docs=enabled_docs)
        _thinking_animation.start()

        result = send_message(token, user_input)
        _thinking_animation.stop()

        if result.get("success"):
            data = result.get("data", {})
            response = strip_emotion_tag(data.get("response", ""))
            tools_used = data.get("metadata", {}).get("tools_used", [])
            history.append((user_input, tools_used, response))
        else:
            error = result.get("error", {}).get("message", "Unknown error")
            history.append((user_input, [], f"Error: {error}"))

        render_screen(history, current_tier, available_tiers, enabled_docs=enabled_docs)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def one_shot(token: str, message: str) -> None:
    result = send_message(token, message)
    if result.get("success"):
        print(strip_emotion_tag(result.get("data", {}).get("response", "")))
    else:
        print(f"Error: {result.get('error', {}).get('message', 'Unknown')}", file=sys.stderr)
        sys.exit(1)


def show_api_key() -> None:
    """Display the MIRA API key and exit."""
    try:
        token = get_api_key('mira_api')
        print(f"\nYour MIRA API Key: {token}\n")
        print("Use with curl:")
        print(f'  curl -H "Authorization: Bearer {token}" \\')
        print(f'       -H "Content-Type: application/json" \\')
        print(f'       -d \'{{"message": "Hello!"}}\' \\')
        print(f'       {MIRA_API_URL}/v0/api/chat\n')
    except Exception as e:
        print(f"Error: Could not retrieve API key from Vault: {e}", file=sys.stderr)
        print("\nMake sure:", file=sys.stderr)
        print("  1. Vault is running and unsealed", file=sys.stderr)
        print("  2. MIRA has been started at least once (creates the API key)", file=sys.stderr)
        print("  3. VAULT_ROLE_ID and VAULT_SECRET_ID are set", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MIRA Chat")
    parser.add_argument('--headless', type=str, help="One-shot message")
    parser.add_argument('--show-key', action='store_true', help="Display API key and exit")
    args = parser.parse_args()

    # Handle --show-key flag (exits immediately)
    if args.show_key:
        show_api_key()
        sys.exit(0)

    server_started = False
    if not args.headless:
        # Splashscreen handles server startup during animation
        need_server = not is_api_running()
        server_ready = show_splashscreen(start_server=need_server)

        if need_server:
            if not server_ready:
                console.print("[red]Server failed to start[/red]", style="bold")
                shutdown_server()
                sys.exit(1)
            server_started = True
            atexit.register(shutdown_server)
            signal.signal(signal.SIGINT, lambda s, f: (shutdown_server(), sys.exit(0)))
            signal.signal(signal.SIGTERM, lambda s, f: (shutdown_server(), sys.exit(0)))

    try:
        token = get_api_key('mira_api')
    except Exception as e:
        console.print(f"[red]Failed to get API token: {e}[/red]")
        if server_started:
            shutdown_server()
        sys.exit(1)

    if args.headless:
        one_shot(token, args.headless)
    else:
        # Check for updates (only in interactive mode)
        update_available, latest_version = check_for_update()
        if update_available and latest_version:
            render_update_alert(latest_version)

        chat_loop(token)

    if server_started:
        shutdown_server()


if __name__ == "__main__":
    main()
