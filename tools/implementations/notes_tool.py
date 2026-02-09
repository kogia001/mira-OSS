"""
Notes tool for local vaults.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from config.config_manager import config
from tools.repo import Tool
from tools.registry import registry


def _split_env_paths(value: str) -> List[str]:
    if not value:
        return []
    return [p for p in value.split(os.pathsep) if p]


# -------------------- CONFIGURATION --------------------

class NotesToolConfig(BaseModel):
    """Configuration for the notes_tool."""
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled by default"
    )
    allowed_roots: List[str] = Field(
        default_factory=lambda: ["/mnt/c/Obsidian Vault"],
        description="Absolute directories this tool can read"
    )
    default_root: Optional[str] = Field(
        default="/mnt/c/Obsidian Vault",
        description="Default root to use when none is provided (must be within allowed_roots)"
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".md"],
        description="File extensions to include (lowercase with dot)"
    )
    max_results: int = Field(
        default=200,
        description="Max results returned for list/search operations"
    )
    max_bytes: int = Field(
        default=200000,
        description="Max bytes to read from a file"
    )
    max_file_size_bytes: int = Field(
        default=2000000,
        description="Max file size allowed for list/search operations"
    )
    follow_symlinks: bool = Field(
        default=False,
        description="Whether to follow symlinked directories when listing/searching"
    )


registry.register("notes_tool", NotesToolConfig)


# -------------------- MAIN TOOL CLASS --------------------

class NotesTool(Tool):
    """Read/search markdown notes and create notes within allowed root directories."""

    name = "notes_tool"
    description = "Read/search markdown notes and create notes within allowed root directories."
    simple_description = (
        "Access local markdown notes in allowed vaults. "
        "List/read/search notes, and create notes within configured roots."
    )
    anthropic_schema = {
        "name": "notes_tool",
        "description": "Read/search markdown notes and create notes inside allowed vault directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list_roots", "list_notes", "read_note", "search_notes", "create_note"],
                    "description": "The operation to perform"
                },
                "root": {
                    "type": "string",
                    "description": "Root vault path to use"
                },
                "path": {
                    "type": "string",
                    "description": "Path to a note (absolute or relative to root)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat query as regex (default false)"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive search (default false)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return"
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Max bytes to read from a file"
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Override allowed extensions"
                },
                "content": {
                    "type": "string",
                    "description": "File content for create_note (default: empty)"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether create_note can overwrite existing file (default false)"
                },
                "mkdir_parents": {
                    "type": "boolean",
                    "description": "Create parent directories if missing (default true)"
                }
            },
            "required": ["operation"],
            "additionalProperties": True
        }
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _get_allowed_roots(self) -> List[Path]:
        tool_config = config.get_tool_config(self.name)
        env_roots = _split_env_paths(os.environ.get("MIRA_NOTES_ALLOWED_ROOTS", ""))

        # Env override should be authoritative so runtime deployment can
        # relocate vault roots without inheriting baked-in defaults.
        if env_roots:
            roots = env_roots
        else:
            roots = list(tool_config.allowed_roots or [])

        normalized: List[Path] = []
        seen = set()
        for root in roots:
            if not root:
                continue
            try:
                resolved = Path(root).expanduser().resolve(strict=False)
            except Exception:
                resolved = Path(root).expanduser()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                normalized.append(resolved)
        return normalized

    def _get_default_root(self) -> Optional[Path]:
        tool_config = config.get_tool_config(self.name)
        env_root = os.environ.get("MIRA_NOTES_DEFAULT_ROOT")
        if env_root:
            root = env_root
        else:
            env_roots = _split_env_paths(os.environ.get("MIRA_NOTES_ALLOWED_ROOTS", ""))
            # If allowed roots are overridden via env and no explicit default is set,
            # default to the first allowed env root for consistent behavior.
            root = env_roots[0] if env_roots else tool_config.default_root
        if not root:
            return None
        try:
            return Path(root).expanduser().resolve(strict=False)
        except Exception:
            return Path(root).expanduser()

    def _ensure_under_allowed(self, path: Path, allowed_roots: List[Path]) -> Path:
        try:
            resolved = path.expanduser().resolve(strict=False)
        except Exception:
            resolved = path.expanduser()
        for root in allowed_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise ValueError(f"Path is outside allowed roots: {resolved}")

    def _resolve_root(self, root: Optional[str]) -> Path:
        allowed_roots = self._get_allowed_roots()
        if not allowed_roots:
            raise ValueError(
                "No allowed roots configured for notes_tool. "
                "Set allowed_roots or MIRA_NOTES_ALLOWED_ROOTS."
            )

        if root:
            candidate = Path(root).expanduser()
        else:
            default_root = self._get_default_root()
            if default_root:
                candidate = default_root
            elif len(allowed_roots) == 1:
                candidate = allowed_roots[0]
            else:
                raise ValueError(
                    "Multiple allowed roots. Provide 'root' explicitly or set MIRA_NOTES_DEFAULT_ROOT."
                )

        resolved = self._ensure_under_allowed(candidate, allowed_roots)
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"Root does not exist or is not a directory: {resolved}")
        return resolved

    def _normalize_extensions(self, extensions: Optional[List[str]]) -> List[str]:
        tool_config = config.get_tool_config(self.name)
        exts = extensions if extensions else tool_config.allowed_extensions
        normalized: List[str] = []
        for ext in exts or []:
            if not ext:
                continue
            ext = ext.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext not in normalized:
                normalized.append(ext)
        return normalized

    def _iter_notes(self, root: Path, extensions: List[str], max_results: int) -> List[Path]:
        tool_config = config.get_tool_config(self.name)
        follow_symlinks = bool(tool_config.follow_symlinks)
        max_file_size = int(tool_config.max_file_size_bytes)

        results: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            if not follow_symlinks:
                dirnames[:] = [d for d in dirnames if not Path(dirpath, d).is_symlink()]

            for name in filenames:
                path = Path(dirpath) / name
                if not path.is_file():
                    continue
                if extensions and path.suffix.lower() not in extensions:
                    continue
                try:
                    if path.stat().st_size > max_file_size:
                        continue
                except OSError:
                    continue
                try:
                    self._ensure_under_allowed(path, [root])
                except ValueError:
                    continue

                results.append(path)
                if len(results) >= max_results:
                    return results
        return results

    def _read_file(self, path: Path, max_bytes: int) -> Dict[str, Any]:
        size = path.stat().st_size
        truncated = size > max_bytes
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
        return {"content": content, "truncated": truncated, "size_bytes": size}

    def _resolve_note_path(self, root: Optional[str], path_value: str) -> Tuple[Path, Path]:
        """Resolve note path to (root_path, absolute_note_path) with allowed-root enforcement."""
        candidate_path = Path(path_value)
        if candidate_path.is_absolute():
            candidate = candidate_path
            root_path = self._resolve_root(root) if root else None
        else:
            root_path = self._resolve_root(root)
            candidate = root_path / path_value

        allowed_roots = self._get_allowed_roots()
        resolved = self._ensure_under_allowed(candidate, allowed_roots)
        if root_path is None:
            # Choose the nearest allowed root for relative path reporting.
            for allowed in allowed_roots:
                try:
                    resolved.relative_to(allowed)
                    root_path = allowed
                    break
                except ValueError:
                    continue
        if root_path is None:
            raise ValueError(f"Unable to determine root for note path: {resolved}")
        return root_path, resolved

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute notes_tool operations.

        Args:
            operation: list_roots | list_notes | read_note | search_notes | create_note
            **kwargs: Operation-specific params (root, path, query, etc.)

        Returns:
            Dict with success, message, and payload.
        """
        try:
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in kwargs for notes_tool: {e}")
                    raise ValueError(f"Invalid JSON in kwargs: {e}")

            if operation == "list_roots":
                allowed_roots = [str(p) for p in self._get_allowed_roots()]
                default_root = self._get_default_root()
                return {
                    "success": True,
                    "message": "Allowed roots retrieved",
                    "allowed_roots": allowed_roots,
                    "default_root": str(default_root) if default_root else None
                }

            if operation == "list_notes":
                root = self._resolve_root(kwargs.get("root"))
                max_results = int(
                    kwargs.get("max_results") or config.get_tool_config(self.name).max_results
                )
                extensions = self._normalize_extensions(kwargs.get("extensions"))
                notes = self._iter_notes(root, extensions, max_results)

                items = []
                for path in notes:
                    try:
                        stat = path.stat()
                        items.append({
                            "path": str(path.relative_to(root)),
                            "size_bytes": stat.st_size,
                            "modified": stat.st_mtime
                        })
                    except OSError:
                        continue

                return {
                    "success": True,
                    "message": f"Found {len(items)} notes",
                    "root": str(root),
                    "count": len(items),
                    "notes": items
                }

            if operation == "read_note":
                path_value = kwargs.get("path")
                if not path_value:
                    raise ValueError("path is required for read_note")

                root_path, resolved = self._resolve_note_path(kwargs.get("root"), path_value)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"Note not found: {resolved}")

                max_bytes = int(
                    kwargs.get("max_bytes") or config.get_tool_config(self.name).max_bytes
                )
                read = self._read_file(resolved, max_bytes)

                rel_path = str(resolved.relative_to(root_path))
                return {
                    "success": True,
                    "message": "Note read",
                    "root": str(root_path),
                    "path": rel_path,
                    "absolute_path": str(resolved),
                    **read
                }

            if operation == "search_notes":
                root = self._resolve_root(kwargs.get("root"))
                query = kwargs.get("query")
                if not query:
                    raise ValueError("query is required for search_notes")

                case_sensitive = bool(kwargs.get("case_sensitive", False))
                use_regex = bool(kwargs.get("regex", False))
                max_results = int(
                    kwargs.get("max_results") or config.get_tool_config(self.name).max_results
                )
                extensions = self._normalize_extensions(kwargs.get("extensions"))

                results = []
                if use_regex:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    pattern = re.compile(query, flags=flags)
                else:
                    pattern = None
                    if not case_sensitive:
                        query = query.lower()

                for path in self._iter_notes(root, extensions, max_results=max_results):
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            for idx, line in enumerate(f, start=1):
                                if pattern:
                                    matched = bool(pattern.search(line))
                                else:
                                    hay = line if case_sensitive else line.lower()
                                    matched = query in hay

                                if matched:
                                    results.append({
                                        "path": str(path.relative_to(root)),
                                        "line": idx,
                                        "snippet": line.rstrip("\n")[:200]
                                    })
                                    if len(results) >= max_results:
                                        break
                        if len(results) >= max_results:
                            break
                    except OSError:
                        continue

                return {
                    "success": True,
                    "message": f"Found {len(results)} matches",
                    "root": str(root),
                    "count": len(results),
                    "matches": results
                }

            if operation == "create_note":
                path_value = kwargs.get("path")
                if not path_value:
                    raise ValueError("path is required for create_note")

                root_path, resolved = self._resolve_note_path(kwargs.get("root"), path_value)
                if resolved.exists() and resolved.is_dir():
                    raise ValueError(f"Cannot create note because target is a directory: {resolved}")

                allowed_exts = self._normalize_extensions(kwargs.get("extensions"))
                if allowed_exts and resolved.suffix.lower() not in allowed_exts:
                    raise ValueError(
                        f"Disallowed extension '{resolved.suffix}'. Allowed: {', '.join(allowed_exts)}"
                    )

                overwrite = bool(kwargs.get("overwrite", False))
                if resolved.exists() and not overwrite:
                    raise ValueError(
                        f"Note already exists: {resolved}. Pass overwrite=true to replace it."
                    )

                mkdir_parents = bool(kwargs.get("mkdir_parents", True))
                parent = resolved.parent
                if not parent.exists():
                    if mkdir_parents:
                        parent.mkdir(parents=True, exist_ok=True)
                    else:
                        raise ValueError(f"Parent directory does not exist: {parent}")

                content = kwargs.get("content")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    raise ValueError("content must be a string")

                with open(resolved, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())

                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"Note write verification failed: {resolved}")

                rel_path = str(resolved.relative_to(root_path))
                return {
                    "success": True,
                    "message": "Note created" if not overwrite else "Note overwritten",
                    "root": str(root_path),
                    "path": rel_path,
                    "absolute_path": str(resolved),
                    "bytes_written": len(content.encode("utf-8")),
                    "created": True,
                    "overwritten": bool(overwrite),
                }

            raise ValueError(f"Unknown operation '{operation}' for notes_tool")

        except Exception as e:
            self.logger.error(f"notes_tool error: {e}")
            raise ValueError(str(e))
