import os
from pathlib import Path

import pytest

from tools.implementations.notes_tool import NotesTool


@pytest.fixture
def notes_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "vault"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MIRA_NOTES_ALLOWED_ROOTS", str(root))
    monkeypatch.setenv("MIRA_NOTES_DEFAULT_ROOT", str(root))
    return root


def test_create_note_writes_file(notes_root: Path):
    tool = NotesTool()

    result = tool.run(
        operation="create_note",
        path="test.md",
        content="# Test\nhello world",
    )

    assert result["success"] is True
    assert result["path"] == "test.md"
    assert result["root"] == str(notes_root.resolve())
    assert result["absolute_path"] == str((notes_root / "test.md").resolve())

    created = notes_root / "test.md"
    assert created.exists()
    assert created.read_text(encoding="utf-8") == "# Test\nhello world"


def test_create_note_rejects_path_outside_allowed_root(notes_root: Path):
    tool = NotesTool()
    outside = notes_root.parent / "outside.md"

    with pytest.raises(ValueError, match="outside allowed roots"):
        tool.run(
            operation="create_note",
            path=str(outside),
            content="nope",
        )


def test_create_note_rejects_disallowed_extension(notes_root: Path):
    tool = NotesTool()

    with pytest.raises(ValueError, match="Disallowed extension"):
        tool.run(
            operation="create_note",
            path="not-allowed.txt",
            content="hello",
        )


def test_create_note_overwrite_requires_flag(notes_root: Path):
    tool = NotesTool()

    first = notes_root / "existing.md"
    first.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        tool.run(
            operation="create_note",
            path="existing.md",
            content="new",
        )

    result = tool.run(
        operation="create_note",
        path="existing.md",
        content="new",
        overwrite=True,
    )
    assert result["overwritten"] is True
    assert first.read_text(encoding="utf-8") == "new"


def test_env_allowed_roots_override_defaults_for_root_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root_a = tmp_path / "vault_a"
    root_b = tmp_path / "vault_b"
    root_a.mkdir(parents=True, exist_ok=True)
    root_b.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv(
        "MIRA_NOTES_ALLOWED_ROOTS",
        os.pathsep.join([str(root_a), str(root_b)]),
    )
    monkeypatch.delenv("MIRA_NOTES_DEFAULT_ROOT", raising=False)

    tool = NotesTool()
    roots = tool.run(operation="list_roots")

    assert roots["allowed_roots"] == [str(root_a.resolve()), str(root_b.resolve())]
    assert roots["default_root"] == str(root_a.resolve())

    result = tool.run(
        operation="create_note",
        path="env-default.md",
        content="env root default",
    )
    assert result["root"] == str(root_a.resolve())
    assert (root_a / "env-default.md").exists()
    assert not (root_b / "env-default.md").exists()


def test_env_default_root_takes_precedence_over_first_allowed_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root_a = tmp_path / "vault_a"
    root_b = tmp_path / "vault_b"
    root_a.mkdir(parents=True, exist_ok=True)
    root_b.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv(
        "MIRA_NOTES_ALLOWED_ROOTS",
        os.pathsep.join([str(root_a), str(root_b)]),
    )
    monkeypatch.setenv("MIRA_NOTES_DEFAULT_ROOT", str(root_b))

    tool = NotesTool()
    roots = tool.run(operation="list_roots")

    assert roots["default_root"] == str(root_b.resolve())

    result = tool.run(
        operation="create_note",
        path="explicit-default.md",
        content="env default root",
    )
    assert result["root"] == str(root_b.resolve())
    assert (root_b / "explicit-default.md").exists()
    assert not (root_a / "explicit-default.md").exists()
