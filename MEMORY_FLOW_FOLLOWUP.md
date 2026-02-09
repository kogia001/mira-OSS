# Memory Layer Follow-up

Date: 2026-02-07

## Why this note exists
Capture current risks in memory/context flow so we can resume improvements quickly.

## Key findings

1. Tier-1 overflow remediation currently does not reduce prompt size in the same turn.
- File: `cns/services/orchestrator.py:1396`
- Details: Tier-1 evacuation updates trinket cache, but the composed system blocks are not rebuilt before retry. The remediation returns unchanged `messages_for_llm`.
- Risk: repeated overflow retries with limited effect until later turn.

2. Memory XML rendering is not escaping memory text.
- File: `working_memory/trinkets/proactive_memory_trinket.py:93`
- File: `working_memory/trinkets/proactive_memory_trinket.py:166`
- Details: raw memory text is interpolated into `<text>...</text>`.
- Risk: malformed XML-like prompt structure and prompt-injection surface from stored memory text.

3. Retention fallback can pin all prior memories when retention block is missing.
- File: `cns/services/fingerprint_generator.py:343`
- Details: when `<memory_retention>` is absent, fallback keeps all previously surfaced memories.
- Risk: context bloat and weaker decay dynamics.

4. Chat retrieval limit exceeds configured proactive max.
- File: `config/config.py:409`
- File: `cns/services/orchestrator.py:203`
- Details: config default `max_memories=10`, but chat path requests `limit=20`.
- Risk: extra context pressure and token cost.

5. Access boosting may over-reinforce surfaced memories.
- File: `lt_memory/proactive.py:108`
- File: `lt_memory/proactive.py:124`
- Details: every surfaced primary memory gets `update_access_stats`, even without explicit user reference.
- Risk: feedback loop that can bias long-term importance toward repeatedly surfaced items.

## Suggested work order

1. Fix Tier-1 overflow remediation to recompose system prompt before retry.
2. Escape/sanitize memory text before XML insertion.
3. Tighten retention fallback (bounded fallback instead of pin-all).
4. Align retrieval limit with config or make it policy-driven.
5. Adjust access boosting to favor explicit reference/mention signals over passive surfacing.

## Validation checklist for the future patch

- Overflow test: Tier-1 retry actually lowers estimated token usage in same turn.
- Injection test: memory text containing `<system>` or `</text>` is safely rendered.
- Retention test: missing retention block does not pin all memories.
- Retrieval test: surfaced count follows configured policy.
- Drift test: long sessions do not trend toward monotonically increasing pinned memory sets.
