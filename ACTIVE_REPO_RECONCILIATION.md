# Active Repo Reconciliation (vs External Review)

Date: 2026-02-07  
Scope: `/opt/mira/app` only (runtime source of truth)

## Context
External review notes were produced against a different repository state.  
This file reconciles findings against the currently active codebase.

## Findings Status

### Confirmed in active repo
1. `refinement.py` consolidation path has invalid `user_id` usage
- File: `lt_memory/refinement.py:230`
- Status: **Present**
- Notes:
  - `user_id` is undefined in method scope.
  - `VectorOps.find_similar_to_memory()` does not accept `user_id` kwarg.

2. SQL interval parameterization is broken in cleanup queries
- File: `lt_memory/db_access.py:1663`
- File: `lt_memory/db_access.py:1862`
- Status: **Present**
- Notes:
  - `INTERVAL '%(retention_hours)s hours'` places a bind placeholder inside a quoted SQL literal.

3. Link taxonomy mismatch across validators / callsites
- File: `lt_memory/models.py:112`
- File: `lt_memory/linking.py:331`
- File: `lt_memory/batch_result_handlers.py:314`
- Status: **Present**
- Notes:
  - `MemoryLink` validator allows one set, callsites/documentation and relationship outputs include others.

4. Proactive memory XML does not escape memory text
- File: `working_memory/trinkets/proactive_memory_trinket.py:93`
- File: `working_memory/trinkets/proactive_memory_trinket.py:166`
- Status: **Present**

5. Tier-1 overflow remediation does not recompose prompt for same-turn retry
- File: `cns/services/orchestrator.py:1396`
- File: `cns/services/orchestrator.py:1422`
- Status: **Present**

6. Chat retrieval limit bypasses proactive config default
- File: `config/config.py:409`
- File: `cns/services/orchestrator.py:203`
- Status: **Present**

7. Entity cache is not user-scoped
- File: `lt_memory/hybrid_search.py:409`
- File: `lt_memory/hybrid_search.py:475`
- Status: **Present**

8. `link_memory_to_entity()` appends without dedup
- File: `lt_memory/db_access.py:1138`
- Status: **Present**

9. `get_all_memories()` polymorphic return type
- File: `lt_memory/db_access.py:388`
- Status: **Present**

10. `refinement.py` uses full-memory loads for candidate discovery
- File: `lt_memory/refinement.py:127`
- File: `lt_memory/refinement.py:199`
- Status: **Present**

11. Dynamic field interpolation in `update_memory()`
- File: `lt_memory/db_access.py:281`
- Status: **Present**

12. Access tracking failures in proactive search are swallowed
- File: `lt_memory/proactive.py:124`
- Status: **Present**

### Corrected interpretation (important)
1. Retention fallback behavior
- File: `cns/services/fingerprint_generator.py:343`
- Status: **Pin-all fallback in this repo**
- Notes:
  - When `<memory_retention>` is missing, code currently keeps all prior surfaced memories.
  - External claim that this branch drops all pins does not match current active code.

### Nuanced / defense-in-depth
1. Scoring formula entity join user-boundary concern
- File: `lt_memory/scoring_formula.sql:113`
- Status: **Potential hardening needed**
- Notes:
  - Current flows usually run user-scoped sessions, but explicit user constraint in subquery would be safer.

## Execution Plan
1. Apply Tier 1 + Tier 2 fixes in active repo:
   - refinement runtime bug
   - SQL interval bugs
   - link taxonomy mismatch
   - XML escaping
   - overflow Tier-1 in-turn recomposition
   - retrieval limit from config
2. Validate by compile + targeted tests.
3. Track Tier 3+ items in follow-up backlog.
