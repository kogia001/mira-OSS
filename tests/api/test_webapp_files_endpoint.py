"""
Tests for webapp output file browsing and download endpoints.
"""

from fastapi.testclient import TestClient


class TestWebappOutputFiles:
    def test_webapp_page_available_without_auth(self, test_client: TestClient):
        response = test_client.get("/webapp")
        assert response.status_code == 200
        assert "MIRA Local Webapp" in response.text

    def test_webapp_page_mentions_history_and_actions_endpoints(self, test_client: TestClient):
        response = test_client.get("/webapp")
        assert response.status_code == 200
        assert "GET /v0/api/data?type=history" in response.text
        assert "POST /v0/api/actions" in response.text

    def test_webapp_page_includes_slash_command_hint(self, test_client: TestClient):
        response = test_client.get("/webapp")
        assert response.status_code == 200
        assert "Type your message or /help." in response.text

    def test_webapp_page_includes_load_older_button(self, test_client: TestClient):
        response = test_client.get("/webapp")
        assert response.status_code == 200
        assert "Load Older" in response.text

    def test_webapp_page_sanitizes_think_and_tool_call_tags(self, test_client: TestClient):
        response = test_client.get("/webapp")
        assert response.status_code == 200
        assert "replace(/<think" in response.text
        assert "replace(/<\\\\/?think" in response.text
        assert "replace(/<tool_call" in response.text
        assert "replace(/<\\\\/?tool_call" in response.text

    def test_files_endpoint_requires_authentication(self, test_client: TestClient):
        response = test_client.get("/v0/api/webapp/files")
        assert response.status_code in [401, 403]

    def test_files_endpoint_lists_output_dir(self, authenticated_client: TestClient, monkeypatch, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "artifact.txt").write_text("hello", encoding="utf-8")
        (output_dir / "nested").mkdir()
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

        response = authenticated_client.get("/v0/api/webapp/files")
        assert response.status_code == 200
        data = response.json()

        assert data["output_root"] == str(output_dir.resolve())
        assert isinstance(data["entries"], list)
        names = [entry["name"] for entry in data["entries"]]
        assert "artifact.txt" in names
        assert "nested" in names

    def test_download_endpoint_returns_file(self, authenticated_client: TestClient, monkeypatch, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "report.md"
        target.write_text("payload", encoding="utf-8")
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

        response = authenticated_client.get("/v0/api/webapp/download", params={"path": "report.md"})
        assert response.status_code == 200
        assert response.content == b"payload"

    def test_files_endpoint_blocks_path_traversal(self, authenticated_client: TestClient, monkeypatch, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

        response = authenticated_client.get("/v0/api/webapp/files", params={"path": "../"})
        assert response.status_code == 400

    def test_artifacts_endpoint_requires_authentication(self, test_client: TestClient):
        response = test_client.get("/v0/api/webapp/artifacts")
        assert response.status_code in [401, 403]

    def test_artifacts_endpoint_empty_when_no_cache(self, authenticated_client: TestClient, monkeypatch):
        from api import webapp as webapp_api

        monkeypatch.setattr(webapp_api, "_get_continuum_id_for_current_user", lambda: "continuum_test")
        monkeypatch.setattr(webapp_api, "_load_cached_artifacts", lambda _continuum_id: [])

        response = authenticated_client.get("/v0/api/webapp/artifacts")
        assert response.status_code == 200
        data = response.json()
        assert data["continuum_id"] == "continuum_test"
        assert data["entries"] == []
