"""
Test isolation fixtures for cleaning up global state between tests.

Ensures tests don't interfere with each other through shared global state.
"""
import pytest
import logging
from clients.valkey_client import get_valkey

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Automatically reset global state before and after each test.

    This fixture runs for every test and ensures:
    - User locks are cleared from Valkey
    - Any other global state is reset
    """
    
    # Clear distributed locks before test when Valkey is available.
    try:
        valkey = get_valkey()
    except Exception as e:
        logger.warning(f"Valkey unavailable during test isolation setup: {e}")
        valkey = None

    try:
        if valkey is not None:
            for key in valkey.scan_iter(match="user_lock:*"):
                valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
    
    yield
    
    # Clear after test
    try:
        if valkey is not None:
            for key in valkey.scan_iter(match="user_lock:*"):
                valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
    


@pytest.fixture
def clean_user_locks():
    """
    Fixture to ensure user locks are clean for specific tests.
    
    Use this when you need explicit control over lock state.
    """
    try:
        valkey = get_valkey()
    except Exception as e:
        logger.warning(f"Valkey unavailable during lock cleanup fixture: {e}")
        valkey = None
    
    # Clear all user locks before test
    try:
        if valkey is not None:
            for key in valkey.scan_iter(match="user_lock:*"):
                valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
    
    yield
    
    # Clear all user locks after test
    try:
        if valkey is not None:
            for key in valkey.scan_iter(match="user_lock:*"):
                valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
