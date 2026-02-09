"""Tests for strict output-claim guard helper methods."""

from cns.services.orchestrator import ContinuumOrchestrator
from cns.core.stream_events import ToolCompletedEvent


class TestOutputClaimGuardHelpers:
    def test_detects_file_creation_claim(self):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        text = "I created /tmp/report.txt and saved it to $OUTPUT_DIR/report.txt."
        assert orchestrator._response_claims_file_creation(text) is True

    def test_ignores_non_file_creation_text(self):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        text = "I analyzed your request and here is the plan."
        assert orchestrator._response_claims_file_creation(text) is False

    def test_ignores_echoed_user_request_text(self):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        text = (
            "YOU: great, can you create test.md?\n"
            "MIRA: I can help with that once you confirm location."
        )
        assert orchestrator._response_claims_file_creation(text) is False

    def test_ignores_uncertainty_disclaimer(self):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        text = (
            "I cannot confirm file creation for this turn because no verifiable output evidence was returned."
        )
        assert orchestrator._response_claims_file_creation(text) is False

    def test_detects_bare_success_claim(self):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        text = "Created test.md and saved it."
        assert orchestrator._response_claims_file_creation(text) is True

    def test_collects_only_existing_local_files(self, tmp_path, monkeypatch):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)

        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "real.txt"
        target.write_text("ok", encoding="utf-8")

        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
        response = (
            "Created files: /tmp/does_not_exist.txt and "
            "$OUTPUT_DIR/real.txt and /tmp/also_missing.md"
        )
        verified = orchestrator._collect_verified_local_file_paths(response)

        assert str(target.resolve()) in verified
        assert all("does_not_exist" not in path for path in verified)

    def test_collects_verified_paths_from_notes_tool_create_results(self, tmp_path):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        created = tmp_path / "created.md"
        created.write_text("ok", encoding="utf-8")

        event = ToolCompletedEvent(
            tool_name="notes_tool",
            tool_id="toolu_1",
            result=(
                "{'success': True, 'created': True, 'root': '" + str(tmp_path) +
                "', 'path': 'created.md', 'absolute_path': '" + str(created) + "'}"
            ),
        )

        verified = orchestrator._collect_verified_local_file_paths_from_tool_events([event])
        assert verified == [str(created.resolve())]

    def test_ignores_non_create_notes_tool_results(self, tmp_path):
        orchestrator = ContinuumOrchestrator.__new__(ContinuumOrchestrator)
        existing = tmp_path / "existing.md"
        existing.write_text("hello", encoding="utf-8")

        event = ToolCompletedEvent(
            tool_name="notes_tool",
            tool_id="toolu_2",
            result=(
                "{'success': True, 'message': 'Note read', 'root': '" + str(tmp_path) +
                "', 'path': 'existing.md', 'absolute_path': '" + str(existing) + "'}"
            ),
        )

        verified = orchestrator._collect_verified_local_file_paths_from_tool_events([event])
        assert verified == []
