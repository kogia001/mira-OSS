"""
MIRA-OSS Codex WSL Tool

Shells out to the Codex CLI inside WSL to send prompts and return replies.
Follows MIRA tool patterns from HOW_TO_BUILD_A_TOOL.md.
"""

import logging
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import config
from tools.repo import Tool
from tools.registry import registry
from utils.user_context import has_user_context

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class CodexWSLConfig(BaseModel):
    """Configuration for the Codex WSL tool."""
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    codex_command: str = Field(default="wsl.exe", description="Executable for Codex CLI")
    codex_command_args: List[str] = Field(
        default_factory=lambda: ["-d", "Ubuntu", "codex"],
        description="Args to prepend after codex_command (e.g., WSL invocation flags)",
    )
    default_args: List[str] = Field(
        default_factory=lambda: ["exec", "-", "--full-auto", "--skip-git-repo-check"],
        description="Default CLI args for Codex",
    )
    default_model: str = Field(
        default="gpt-5.2",
        description="Default model for general tasks",
    )
    coding_model: str = Field(
        default="gpt-5.2-codex",
        description="Model for coding tasks",
    )
    input_mode: str = Field(
        default="stdin",
        description="How to send prompt: 'stdin' or 'arg'",
    )
    timeout_seconds: int = Field(default=300, description="Max seconds to wait for a reply")
    max_output_chars: int = Field(default=8000, description="Max output characters to return")
    sensitive_keywords: List[str] = Field(
        default_factory=lambda: [
            "password",
            "passwd",
            "secret",
            "api key",
            "token",
            "private key",
            "ssh",
            "aws_access_key",
            "gcp",
            "azure",
            "vault",
            "jwt",
            "oauth",
            "cookie",
            "session",
            "credit card",
            "ssn",
            "sudo",
            "root",
            "/etc/",
            "/var/",
            "/usr/",
            "/bin/",
            "/sbin/",
            "/root",
            "rm -rf",
            "mkfs",
            "dd ",
            "chmod 777",
            "chown",
            "systemctl",
            "iptables",
            "firewall",
            "regedit",
            "diskpart",
            "format ",
        ],
        description="Keywords that trigger a sensitive-request block",
    )


registry.register("codex_wsl", CodexWSLConfig)


# =============================================================================
# Tool Implementation
# =============================================================================

class CodexWSLTool(Tool):
    """
    Send queries to the Codex CLI running in WSL and return replies.

    Operations:
    - query: send a prompt and return the CLI output
    - health: verify Codex CLI availability
    """

    name = "codex_wsl"
    simple_description = "Send prompts to Codex CLI in WSL"

    anthropic_schema = {
        "name": "codex_wsl",
        "description": (
            "Send prompts to the Codex CLI in WSL and return the reply. "
            "Configure command and default args via tool config."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["query", "health"],
                    "description": "Operation to perform: 'query' or 'health'.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Prompt to send to Codex (required for query).",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional CLI args to append for this request.",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["general", "coding"],
                    "description": "Choose model by task type (default: general).",
                },
                "model": {
                    "type": "string",
                    "description": "Override model for this request (e.g., 'gpt-5.2-codex').",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional override for command timeout.",
                },
                "max_output_chars": {
                    "type": "integer",
                    "description": "Optional override for response truncation.",
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        # Resolve tool config once per instance (mirrors other tool patterns)
        self.config = config.get_tool_config(self.name)

        # No tables required, but keep the pattern consistent
        if has_user_context():
            pass

    def run(self, **kwargs) -> Dict[str, Any]:
        operation = (kwargs.get("operation") or "").lower()

        if not operation:
            return {
                "success": False,
                "message": "Operation is required. Use: query or health",
            }

        try:
            if operation == "health":
                return self._health()
            if operation == "query":
                return self._query(
                    prompt=kwargs.get("prompt"),
                    args=kwargs.get("args"),
                    task_type=kwargs.get("task_type"),
                    model=kwargs.get("model"),
                    timeout_seconds=kwargs.get("timeout_seconds"),
                    max_output_chars=kwargs.get("max_output_chars"),
                )

            return {
                "success": False,
                "message": f"Unknown operation: '{operation}'. Use: query or health",
            }
        except ValueError as exc:
            self.logger.error(f"Validation error in {operation}: {exc}")
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            self.logger.exception(f"Error in codex_wsl.{operation}: {exc}")
            return {"success": False, "message": f"Error: {str(exc)}"}

    # =========================================================================
    # Operations
    # =========================================================================

    def _health(self) -> Dict[str, Any]:
        command = self._get_command()
        found = shutil.which(command) if command else None

        if not found:
            return {
                "success": False,
                "message": (
                    f"Codex CLI not found on PATH: '{command}'. "
                    "Update codex_command or PATH in tool config."
                ),
            }

        return {
            "success": True,
            "message": f"Codex CLI found: {found}",
            "path": found,
        }

    def _query(
        self,
        prompt: Optional[str],
        args: Optional[List[str]],
        task_type: Optional[str],
        model: Optional[str],
        timeout_seconds: Optional[int],
        max_output_chars: Optional[int],
    ) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("prompt is required for query")

        sensitive_matches = self._find_sensitive_terms(prompt, args)
        if sensitive_matches:
            return {
                "success": False,
                "message": (
                    "Sensitive request detected. Please run Codex directly in WSL "
                    "so you can review permissions and context explicitly."
                ),
                "matches": sensitive_matches[:10],
            }

        command = self._build_command(args, task_type=task_type, model=model)
        input_mode = self._get_input_mode()

        input_data = None
        if input_mode == "stdin":
            input_data = prompt
        elif input_mode == "arg":
            command = command + [prompt]
        else:
            raise ValueError("input_mode must be 'stdin' or 'arg'")

        timeout = timeout_seconds if timeout_seconds is not None else self._get_timeout()
        max_chars = max_output_chars if max_output_chars is not None else self._get_max_output()

        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            stdout = self._coerce_text(getattr(exc, "stdout", "")) or ""
            stderr = self._coerce_text(getattr(exc, "stderr", "")) or ""
            return {
                "success": False,
                "message": f"Request timed out after {timeout} seconds",
                "error_type": "timeout",
                "stdout": self._truncate(stdout, max_chars),
                "stderr": self._truncate(stderr, max_chars),
                "elapsed_ms": elapsed_ms,
                "command": command,
                "hint": "Increase timeout_seconds or ensure Codex CLI is authenticated and responsive.",
            }

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0:
            return {
                "success": False,
                "message": f"Codex CLI exited with code {result.returncode}",
                "stdout": self._truncate(stdout, max_chars),
                "stderr": self._truncate(stderr, max_chars),
                "elapsed_ms": elapsed_ms,
                "command": command,
            }

        return {
            "success": True,
            "reply": self._truncate(stdout, max_chars),
            "stderr": self._truncate(stderr, max_chars),
            "elapsed_ms": elapsed_ms,
            "command": command,
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_command(self) -> str:
        command = self.config.codex_command.strip()
        if not command:
            raise ValueError("codex_command is empty")
        if any(ch.isspace() for ch in command):
            raise ValueError(
                "codex_command must be a single executable, not a shell string. "
                "Use codex_command_args for additional flags."
            )
        return command

    def _build_command(
        self,
        extra_args: Optional[List[str]],
        task_type: Optional[str],
        model: Optional[str],
    ) -> List[str]:
        command = [self._get_command()]
        command.extend(self._get_command_args())
        command.extend(self.config.default_args or [])
        command.extend(extra_args or [])
        model_to_use = self._resolve_model(command, task_type=task_type, model=model)
        if model_to_use:
            command.extend(["--model", model_to_use])
        return command

    def _resolve_model(
        self,
        command: List[str],
        task_type: Optional[str],
        model: Optional[str],
    ) -> Optional[str]:
        if self._has_model_arg(command):
            return None

        if model:
            return model

        normalized = (task_type or "general").strip().lower()
        if normalized == "coding":
            return self.config.coding_model

        return self.config.default_model

    @staticmethod
    def _has_model_arg(command: List[str]) -> bool:
        for arg in command[1:]:
            if arg in ("-m", "--model"):
                return True
            if arg.startswith("--model="):
                return True
        return False

    def _get_input_mode(self) -> str:
        return (self.config.input_mode or "stdin").strip().lower()

    def _get_command_args(self) -> List[str]:
        return [str(arg) for arg in (self.config.codex_command_args or []) if str(arg).strip()]

    def _get_timeout(self) -> int:
        return max(1, int(self.config.timeout_seconds))

    def _get_max_output(self) -> int:
        return max(256, int(self.config.max_output_chars))

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    def _find_sensitive_terms(
        self,
        prompt: str,
        args: Optional[List[str]],
    ) -> List[str]:
        haystack = prompt.lower()
        matches: List[str] = []

        for term in self.config.sensitive_keywords:
            if term in haystack:
                matches.append(term)

        for arg in args or []:
            if arg in ("--dangerously-bypass-approvals-and-sandbox", "--yolo"):
                matches.append(arg)
            if arg.startswith("--sandbox") or arg.startswith("--add-dir"):
                matches.append(arg)

        return list(dict.fromkeys(matches))
