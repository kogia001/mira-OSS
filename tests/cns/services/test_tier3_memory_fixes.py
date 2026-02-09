"""Regression tests for Tier 3 memory fixes."""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from config.config import ProactiveConfig
from lt_memory.db_access import LTMemoryDB
from lt_memory.hybrid_search import HybridSearcher
from lt_memory.proactive import ProactiveService


class _DummyEntity:
    def __init__(self, name: str, entity_type: str):
        self.id = uuid4()
        self.name = name
        self.entity_type = entity_type


class _DummyDBForHybrid:
    def __init__(self):
        self.current_user = "user-a"
        self.calls = []
        self.entities_by_user = {
            "user-a": [_DummyEntity("Alice", "PERSON")],
            "user-b": [_DummyEntity("Bob", "PERSON")],
        }

    def _resolve_user_id(self):
        return self.current_user

    def get_active_entities(self, limit=100, user_id=None):
        self.calls.append((limit, user_id))
        return self.entities_by_user.get(user_id, [])


def test_hybrid_searcher_entity_cache_scoped_by_user():
    """Entity cache should refresh automatically when user context changes."""
    db = _DummyDBForHybrid()
    searcher = HybridSearcher(db_access=db, entity_extractor=None)

    # User A loads/cache matches
    matched_a = searcher._find_fuzzy_entity_matches([("Alice", "PERSON")], {})
    assert len(matched_a) == 1
    assert db.calls == [(100, "user-a")]

    # Switching to user B should invalidate old cache and fetch B's entities
    db.current_user = "user-b"
    matched_b_wrong_name = searcher._find_fuzzy_entity_matches([("Alice", "PERSON")], {})
    assert matched_b_wrong_name == {}
    assert db.calls == [(100, "user-a"), (100, "user-b")]

    # Same user B request should reuse cache (no extra DB fetch)
    matched_b = searcher._find_fuzzy_entity_matches([("Bob", "PERSON")], {})
    assert len(matched_b) == 1
    assert db.calls == [(100, "user-a"), (100, "user-b")]


def test_hybrid_searcher_clear_entity_cache_resets_user_scope():
    """Manual clear should drop both entity cache and user marker."""
    db = _DummyDBForHybrid()
    searcher = HybridSearcher(db_access=db, entity_extractor=None)

    searcher._find_fuzzy_entity_matches([("Alice", "PERSON")], {})
    assert db.calls == [(100, "user-a")]

    searcher.clear_entity_cache()
    searcher._find_fuzzy_entity_matches([("Alice", "PERSON")], {})
    assert db.calls == [(100, "user-a"), (100, "user-a")]


class _FakeSession:
    def __init__(self, first_update_count: int):
        self.first_update_count = first_update_count
        self.execute_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def transaction(self):
        return self

    def execute_update(self, query, params=None):
        self.execute_calls.append((query, params))
        if len(self.execute_calls) == 1:
            return self.first_update_count
        return 1


class _FakeSessionManager:
    def __init__(self, session):
        self._session = session
        self.requested_user_ids = []

    def get_session(self, user_id):
        self.requested_user_ids.append(user_id)
        return self._session


def test_link_memory_to_entity_dedup_skips_entity_counter_on_duplicate():
    """Duplicate memory->entity links should not increment entity link_count."""
    session = _FakeSession(first_update_count=0)
    manager = _FakeSessionManager(session)
    db = LTMemoryDB(manager)

    db.link_memory_to_entity(
        memory_id=uuid4(),
        entity_id=uuid4(),
        entity_name="Alice",
        entity_type="PERSON",
        user_id="user-1",
    )

    assert manager.requested_user_ids == ["user-1"]
    assert len(session.execute_calls) == 1
    first_query, first_params = session.execute_calls[0]
    assert "NOT EXISTS" in first_query
    assert first_params["entity_id"]


def test_link_memory_to_entity_new_link_updates_entity_counter():
    """New memory->entity links should still increment entity link_count."""
    session = _FakeSession(first_update_count=1)
    manager = _FakeSessionManager(session)
    db = LTMemoryDB(manager)

    db.link_memory_to_entity(
        memory_id=uuid4(),
        entity_id=uuid4(),
        entity_name="Alice",
        entity_type="PERSON",
        user_id="user-1",
    )

    assert manager.requested_user_ids == ["user-1"]
    assert len(session.execute_calls) == 2
    first_query, _ = session.execute_calls[0]
    second_query, _ = session.execute_calls[1]
    assert "NOT EXISTS" in first_query
    assert "SET link_count = link_count + 1" in second_query


class _DummyMemory:
    def __init__(self, importance_score: float):
        self.id = uuid4()
        self.text = f"memory-{self.id}"
        self.importance_score = importance_score
        self.similarity_score = 0.8
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = None
        self.access_count = 0
        self.happens_at = None
        self.expires_at = None
        self.inbound_links = []
        self.outbound_links = []
        self.linked_memories = []


class _DummyVectorOps:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class _DummyLinkingService:
    def traverse_related(self, memory_id, depth):
        return []


class _DummyDBForProactive:
    def __init__(self):
        self.access_updates = []

    def update_access_stats(self, memory_id):
        self.access_updates.append(memory_id)


def test_proactive_service_relies_on_hybrid_min_importance_filter():
    """Service should not apply a second in-memory importance filter."""
    low_importance = _DummyMemory(importance_score=0.01)
    high_importance = _DummyMemory(importance_score=0.9)
    vector_ops = _DummyVectorOps([low_importance, high_importance])
    db = _DummyDBForProactive()
    service = ProactiveService(
        config=ProactiveConfig(min_importance_score=0.5, max_memories=2),
        vector_ops=vector_ops,
        linking_service=_DummyLinkingService(),
        db=db,
    )

    result = service.search_with_embedding(
        embedding=np.zeros(768, dtype=np.float32),
        fingerprint="test fingerprint",
        limit=2,
    )

    assert len(result) == 2
    returned_ids = {item["id"] for item in result}
    assert str(low_importance.id) in returned_ids
    assert str(high_importance.id) in returned_ids

    assert len(vector_ops.calls) == 1
    assert vector_ops.calls[0]["min_importance"] == 0.5
    assert vector_ops.calls[0]["limit"] == 4
