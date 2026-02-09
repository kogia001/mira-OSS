# Memory Architecture Pickup Notes

Date: 2026-02-09  
Repo of record: `/opt/mira/app` (active runtime), not `mira-OSS`

## Current State

### Completed (already implemented in `/opt/mira/app`)
- Tier 1 + Tier 2 fixes:
  - `lt_memory/refinement.py`: removed invalid undefined `user_id` usage in consolidation similarity call.
  - `lt_memory/db_access.py`: fixed SQL interval parameterization in batch cleanup queries.
  - `lt_memory/models.py` + related callsites: reconciled link taxonomy mismatch.
  - `working_memory/trinkets/proactive_memory_trinket.py`: escaped memory/link text in XML scaffold.
  - `cns/services/orchestrator.py`: Tier-1 overflow now recomposes prompt in-turn.
  - `cns/services/orchestrator.py`: retrieval limit now follows config instead of hardcoded `20`.

- Tier 3 fixes:
  - `lt_memory/hybrid_search.py`: entity cache is now user-scoped with automatic invalidation on user change.
  - `lt_memory/db_access.py`: `link_memory_to_entity()` now deduplicates and only increments `link_count` on new insert.
  - `lt_memory/proactive.py`: removed redundant second importance filter after hybrid search.

- Tier 4 (scalability item #11):
  - Added DB-side filtering methods in `lt_memory/db_access.py`:
    - `get_verbose_refinement_candidates(...)`
    - `get_consolidation_hub_candidates(...)`
  - Updated `lt_memory/refinement.py` to use these methods instead of loading all memories into Python.

## Validation Snapshot
- `py_compile` passed for modified files.
- Tests passed:
  - `tests/cns/services/test_tier3_memory_fixes.py`
  - `tests/cns/services/test_refinement_db_filtering.py`
  - `tests/cns/services/test_memory_retention.py`
  - `tests/working_memory/test_notification_center.py`
  - `tests/working_memory/test_trinket_access.py`

## Open Backlog (next)

1. API cleanup: split polymorphic `get_all_memories()` return behavior (`lt_memory/db_access.py`).
2. Hardening: whitelist allowed fields in `update_memory()` to avoid SQL field injection risk.
3. Typing: replace untyped link dicts with typed link structures (`TypedDict` / model).
4. Design decision: decide whether `_track_memory_access` failures should remain best-effort or fail-fast.
5. Coverage: expand tests for extraction/linking/refinement/batching service flows.

## Quick Resume Commands

```bash
cd /opt/mira/app
venv/bin/pytest -q tests/cns/services/test_tier3_memory_fixes.py tests/cns/services/test_refinement_db_filtering.py
venv/bin/pytest -q tests/cns/services/test_memory_retention.py tests/working_memory/test_notification_center.py tests/working_memory/test_trinket_access.py
```

