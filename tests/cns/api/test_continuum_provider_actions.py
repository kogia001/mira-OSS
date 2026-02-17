"""Unit tests for continuum provider-management actions."""

import pytest

from cns.api.actions import ContinuumDomainHandler
from cns.api.base import ValidationError
from utils.user_context import set_current_user_id
import utils.user_context as user_context


@pytest.fixture
def postgres_state(monkeypatch):
    import clients.postgres_client as postgres_client

    state = {
        "database_name": None,
        "queries": [],
        "updates": [],
        "query_rows": [],
        "update_rowcount": 1,
    }

    class FakePostgresClient:
        def __init__(self, database_name):
            state["database_name"] = database_name

        def execute_query(self, query, params=None):
            state["queries"].append((query, params))
            return state["query_rows"]

        def execute_update(self, query, params=None):
            state["updates"].append((query, params))
            return state["update_rowcount"]

    monkeypatch.setattr(postgres_client, "PostgresClient", FakePostgresClient)
    return state


@pytest.fixture
def vault_state(monkeypatch):
    import clients.vault_client as vault_client

    state = {
        "set_calls": [],
        "has_calls": [],
        "has_results": {},
        "set_error": None,
    }

    def fake_set_api_key(key_name, value):
        if state["set_error"] is not None:
            raise state["set_error"]
        state["set_calls"].append((key_name, value))

    def fake_has_api_key(key_name):
        state["has_calls"].append(key_name)
        return state["has_results"].get(key_name, False)

    monkeypatch.setattr(vault_client, "set_api_key", fake_set_api_key)
    monkeypatch.setattr(vault_client, "has_api_key", fake_has_api_key)
    return state


@pytest.fixture
def handler():
    set_current_user_id("test-user")
    return ContinuumDomainHandler()


def _run_action(handler, action, data):
    validated = handler.validate_action(action, data)
    return handler.execute_action(action, validated)


def test_get_tier_provider_status_returns_expected_shape(handler, postgres_state, vault_state):
    postgres_state["query_rows"] = [
        {
            "name": "fast",
            "provider": "generic",
            "endpoint_url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key_name": "provider_key_1",
            "model": "llama-3.1-8b-instant",
            "thinking_budget": 0,
        },
        {
            "name": "balanced",
            "provider": "generic",
            "endpoint_url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key_name": None,
            "model": "qwen/qwen3-32b",
            "thinking_budget": 0,
        },
        {
            "name": "oss",
            "provider": "generic",
            "endpoint_url": "https://api.groq.com/openai/v1/chat/completions",
            "api_key_name": "provider_key_2",
            "model": "openai/gpt-oss-120b",
            "thinking_budget": 0,
        },
    ]
    vault_state["has_results"]["provider_key_1"] = True
    vault_state["has_results"]["provider_key_2"] = False

    result = _run_action(handler, "get_tier_provider_status", {})

    assert result["success"] is True
    assert [tier["name"] for tier in result["tiers"]] == ["fast", "balanced", "oss"]
    assert result["tiers"][0]["has_api_key"] is True
    assert result["tiers"][1]["has_api_key"] is None
    assert result["tiers"][2]["has_api_key"] is False
    assert vault_state["has_calls"] == ["provider_key_1", "provider_key_2"]
    assert postgres_state["database_name"] == "mira_service"
    assert len(postgres_state["queries"]) == 1


def test_set_provider_key_calls_slot_key_write(handler, vault_state):
    result = _run_action(
        handler,
        "set_provider_key",
        {"selection_id": 1, "api_key": "  secret-value  "},
    )

    assert result == {"success": True, "key_name": "provider_key_1"}
    assert vault_state["set_calls"] == [("provider_key_1", "secret-value")]


def test_set_provider_key_sanitizes_vault_write_failure(handler, vault_state):
    vault_state["set_error"] = PermissionError("forbidden with secret-value")

    with pytest.raises(ValidationError) as exc:
        _run_action(
            handler,
            "set_provider_key",
            {"selection_id": 2, "api_key": "secret-value"},
        )

    assert exc.value.message == "Vault write failed: PermissionError"
    assert "secret-value" not in exc.value.message


def test_set_provider_key_rejects_blank_input(handler):
    with pytest.raises(ValidationError, match="api_key must be non-empty"):
        _run_action(
            handler,
            "set_provider_key",
            {"selection_id": 1, "api_key": "   "},
        )


def test_set_tier_provider_selection_updates_db_and_clears_cache(handler, postgres_state):
    user_context._tiers_cache = {"fast": object()}

    result = _run_action(
        handler,
        "set_tier_provider_selection",
        {"tier": "FAST", "selection_id": 2},
    )

    assert result["success"] is True
    assert result["tier"] == "fast"
    assert result["api_key_name"] == "provider_key_2"
    assert user_context._tiers_cache is None
    assert postgres_state["updates"][0][1] == (
        "generic",
        "https://api.groq.com/openai/v1/chat/completions",
        "provider_key_2",
        "fast",
    )


def test_set_tier_provider_selection_rejects_non_v1_tiers(handler):
    with pytest.raises(ValidationError, match="tier must be one of: fast, balanced, oss"):
        _run_action(
            handler,
            "set_tier_provider_selection",
            {"tier": "nuanced", "selection_id": 1},
        )


def test_set_tier_model_updates_db_and_clears_cache(handler, postgres_state):
    user_context._tiers_cache = {"balanced": object()}

    result = _run_action(
        handler,
        "set_tier_model",
        {"tier": "balanced", "model": "qwen/qwen3-32b"},
    )

    assert result == {"success": True, "tier": "balanced", "model": "qwen/qwen3-32b"}
    assert user_context._tiers_cache is None
    assert postgres_state["updates"][0][1] == ("qwen/qwen3-32b", "balanced")


def test_set_tier_model_rejects_non_v1_tiers(handler):
    with pytest.raises(ValidationError, match="tier must be one of: fast, balanced, oss"):
        _run_action(handler, "set_tier_model", {"tier": "nuanced", "model": "x"})


def test_set_tier_model_rejects_blank_model(handler):
    with pytest.raises(ValidationError, match="model must be non-empty"):
        _run_action(handler, "set_tier_model", {"tier": "fast", "model": "   "})


def test_set_tier_model_normalizes_oss_alias(handler, postgres_state):
    user_context._tiers_cache = {"oss": object()}

    result = _run_action(
        handler,
        "set_tier_model",
        {"tier": "oss", "model": "gpt-openai/gpt-oss-120b"},
    )

    assert result == {"success": True, "tier": "oss", "model": "openai/gpt-oss-120b"}
    assert user_context._tiers_cache is None
    assert postgres_state["updates"][0][1] == ("openai/gpt-oss-120b", "oss")


def test_set_tier_thinking_budget_updates_db_and_clears_cache(handler, postgres_state):
    user_context._tiers_cache = {"oss": object()}

    result = _run_action(
        handler,
        "set_tier_thinking_budget",
        {"tier": "OSS", "thinking_budget": 2048},
    )

    assert result == {"success": True, "tier": "oss", "thinking_budget": 2048}
    assert user_context._tiers_cache is None
    assert postgres_state["updates"][0][1] == (2048, "oss")


def test_set_tier_thinking_budget_rejects_non_v1_tiers(handler):
    with pytest.raises(ValidationError, match="tier must be one of: fast, balanced, oss"):
        _run_action(
            handler,
            "set_tier_thinking_budget",
            {"tier": "nuanced", "thinking_budget": 2048},
        )


def test_set_tier_thinking_budget_rejects_negative_budget(handler):
    with pytest.raises(ValidationError, match="thinking_budget must be an integer >= 0"):
        _run_action(
            handler,
            "set_tier_thinking_budget",
            {"tier": "oss", "thinking_budget": -1},
        )


def test_reload_tier_cache_action(handler):
    user_context._tiers_cache = {"fast": object()}

    result = _run_action(handler, "reload_tier_cache", {})

    assert result == {"success": True}
    assert user_context._tiers_cache is None
