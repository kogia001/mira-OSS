# Reload Handoff: Pytest Setup

## Where to reload
- Reload environment in: `/opt/mira/app`

## Why
- Config loader expects `config/system_prompt.txt` relative to current working directory.
- Running from `/opt/mira` causes:
  - `System prompt not found: config/system_prompt.txt`

## Commands to run after reload
```bash
cd /opt/mira/app
venv/bin/pytest
```

## Known blocker after CWD fix
- Test bootstrap currently fails on:
  - `ModuleNotFoundError: No module named 'tests.fixtures.auth'`
- Source:
  - `tests/conftest.py` imports `tests.fixtures.auth`, but that file/module is not present in this checkout.

## Resume plan after reload
1. Re-run a targeted test to confirm current failure mode.
2. Fix missing `tests.fixtures.auth` import/module issue.
3. Re-run targeted tests.
4. Run broader pytest subset if stable.
