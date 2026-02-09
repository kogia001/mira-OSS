"""Unit tests for refinement candidate selection via DB-side filtering."""

from types import SimpleNamespace
from uuid import uuid4

from config.config import RefinementConfig
from lt_memory.refinement import RefinementService


class _FakeDB:
    def __init__(self):
        self.verbose_call = None
        self.hub_call = None

    def get_verbose_refinement_candidates(self, **kwargs):
        self.verbose_call = kwargs
        return [
            SimpleNamespace(
                id=uuid4(),
                text="x" * 120,
            )
        ]

    def get_consolidation_hub_candidates(self, **kwargs):
        self.hub_call = kwargs
        return [
            SimpleNamespace(
                id=uuid4(),
                text="Hub memory",
                importance_score=0.8,
            )
        ]


class _FakeVectorOps:
    def find_similar_to_memory(self, **kwargs):
        return [
            SimpleNamespace(
                id=uuid4(),
                text="Similar memory",
                similarity_score=0.95,
            )
        ]


def _build_service(fake_db, fake_vector_ops):
    # Build without __init__ to avoid prompt-file loading in this fast unit test.
    service = RefinementService.__new__(RefinementService)
    service.config = RefinementConfig(
        verbose_threshold_chars=70,
        refinement_cooldown_days=30,
        min_age_for_refinement_days=7,
        min_access_count_for_refinement=3,
        max_rejection_count=3,
        min_cluster_size=2,
        max_cluster_size=5,
        consolidation_similarity_threshold=0.88,
        consolidation_confidence_threshold=0.8,
    )
    service.db = fake_db
    service.vector_ops = fake_vector_ops
    service.llm_provider = None
    return service


def test_identify_verbose_memories_uses_db_side_filtering():
    fake_db = _FakeDB()
    service = _build_service(fake_db, _FakeVectorOps())

    result = service.identify_verbose_memories(limit=5)

    assert len(result) == 1
    assert fake_db.verbose_call is not None
    assert fake_db.verbose_call["verbose_threshold_chars"] == 70
    assert fake_db.verbose_call["min_age_days"] == 7
    assert fake_db.verbose_call["refinement_cooldown_days"] == 30
    assert fake_db.verbose_call["min_access_count"] == 3
    assert fake_db.verbose_call["max_rejection_count"] == 3
    assert fake_db.verbose_call["limit"] == 5


def test_identify_consolidation_clusters_uses_db_hub_query():
    fake_db = _FakeDB()
    service = _build_service(fake_db, _FakeVectorOps())

    clusters = service.identify_consolidation_clusters()

    assert len(clusters) == 1
    assert fake_db.hub_call is not None
    assert fake_db.hub_call["min_importance"] == 0.3
    assert fake_db.hub_call["min_access_count"] == 5
    assert fake_db.hub_call["min_inbound_links"] == 5
    assert fake_db.hub_call["limit"] == 50

