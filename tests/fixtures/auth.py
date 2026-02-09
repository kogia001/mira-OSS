"""
Backward-compatible auth test helpers for OSS mode.

Some legacy tests still import auth symbols from this module.
"""

from __future__ import annotations

import logging
from typing import Optional

from tests.fixtures.core import TEST_USER_EMAIL, ensure_test_user_exists

logger = logging.getLogger(__name__)


def _resolve_test_user_id() -> str:
    """Resolve a stable test user ID without failing module import."""
    try:
        user = ensure_test_user_exists()
        user_id = str(user.get("id", "")).strip()
        if user_id:
            return user_id
    except Exception as exc:
        logger.warning("Could not resolve TEST_USER_ID during import: %s", exc)

    # Fallback for environments where DB bootstrap is unavailable during collection.
    return "00000000-0000-0000-0000-000000000001"


TEST_USER_ID = _resolve_test_user_id()


def get_test_api_key() -> Optional[str]:
    """
    Return the configured bearer token used by OSS auth.

    In OSS mode this is the same API key accepted by auth/api.py.
    """
    try:
        from clients.vault_client import _ensure_vault_client

        vault_client = _ensure_vault_client()
        secret_data = vault_client.client.secrets.kv.v2.read_secret_version(
            path="mira/api_keys"
        )
        return secret_data["data"]["data"].get("mira_api")
    except Exception as exc:
        logger.debug("Could not resolve test API key from Vault: %s", exc)
        return None

