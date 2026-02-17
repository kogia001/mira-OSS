# MIRA Change Record

## 2026-02-16

- Updated local runtime tier routing to use Groq-backed V1 model IDs:
  - `fast` -> `meta-llama/llama-3.1-8b-instruct`
  - `balanced` -> `qwen/qwen3-32b`
  - `nuanced` left on Opus (unchanged)
- Resolved startup blockers during local bring-up:
  - Missing `config/system_prompt.txt` path issue when launched from repo root
  - Vault auth failures from missing `VAULT_ROLE_ID` / `VAULT_SECRET_ID`
  - Vault sealed state handling during restarts
- Diagnosed and resolved `permission denied for table account_tiers` during `/provider use ...` actions.
- Wrote progress summary at `/opt/mira/20260216-progress.md`.

## 2026-02-17

- Fixed web output sanitization so internal reasoning/tool envelope text is not rendered to users:
  - Strip `<think>...</think>` blocks
  - Strip `<tool_call>...</tool_call>` blocks
  - Strip stray `<think>`, `</think>`, `<tool_call>`, `</tool_call>` tags
  - File: `api/webapp.py`
- Added regression coverage for sanitizer behavior:
  - File: `tests/api/test_webapp_files_endpoint.py`
  - Test: `test_webapp_page_sanitizes_think_and_tool_call_tags`
- Reduced log noise while preserving important signals:
  - Added `MIRA_DEBUG_LOGS` env toggle for verbose diagnostics
  - APScheduler routine heartbeat logs quiet by default
  - `Received update without task_id` downgraded by default, warning only in debug mode
  - Lattice missing-package warning now explicit/actionable when federation is enabled
  - Files: `main.py`, `working_memory/trinkets/getcontext_trinket.py`
- Silenced hvac deprecation warning by explicitly setting:
  - `raise_on_deleted_version=True` in Vault KV read path (`main.py`)
- Created local restart helper wrapper:
  - `/opt/mira/mira-api.sh` (starts/unseals Vault, exports AppRole env vars, starts API)

## Repo Status

- Committed on branch `main`:
  - Commit: `d671879`
  - Message: `Fix web output sanitization and add debug-log controls`
  - Files in commit:
    - `api/webapp.py`
    - `main.py`
    - `tests/api/test_webapp_files_endpoint.py`
    - `working_memory/trinkets/getcontext_trinket.py`
- Push to GitHub was not completed in this environment due DNS/network resolution failure.
