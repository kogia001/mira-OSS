# MIRA Architecture Overview

This document explains how the OSS MIRA codebase is structured, how requests flow through the system, and where major responsibilities live.

## 1) System Shape

MIRA is a FastAPI application with:
- A single-user auth model in OSS mode
- A CNS orchestration core for chat processing
- Event-driven working memory composition
- A dynamic tool system
- Multi-layer storage (Postgres, Valkey, per-user SQLite)
- Long-term memory extraction and retrieval services

Primary entry point:
- `main.py`

Primary runtime domains:
- `cns/` (orchestration, events, continuum infrastructure)
- `working_memory/` (prompt composition and trinkets)
- `tools/` (tool base classes, repository, implementations)
- `lt_memory/` (long-term memory extraction, linking, vector ops)
- `clients/` (Postgres, Valkey, Vault, embeddings, LLM adapters)
- `api/` and `cns/api/` (HTTP endpoints)
- `auth/` (single-user token auth)
- `utils/` (user context, schedulers, locks, support services)

## 2) Boot and App Wiring

Startup is defined in `main.py`:
- `ensure_single_user()` creates/validates the single OSS user and API key state.
- `lifespan()` initializes core services and singletons.
- `create_app()` registers middleware, exception handlers, and routers.

Routers are mounted under:
- `/v0/api/health`
- `/v0/api/chat`
- `/v0/api/data`
- `/v0/api/actions`
- `/v0/api/tool_config`
- `/v0/api/federation`
- `/webapp` and related file/artifact endpoints

## 3) Auth and Request Context

Auth dependency:
- `auth/api.py:get_current_user`

Behavior:
1. Validates Bearer token against `app.state.api_key`.
2. Sets ambient user context via `utils/user_context.py`.
3. Returns the authenticated user object.

Ambient user context is critical:
- Many DB/tool operations assume `get_current_user_id()` exists.
- Isolation is applied through contextvars + DB-level controls.

## 4) Chat Request Flow

Main endpoint:
- `cns/api/chat.py` (`POST /v0/api/chat`)

Flow:
1. Validate/sanitize text and optional image/document payloads.
2. Acquire per-user distributed request lock.
3. Load current continuum from `ContinuumPool`.
4. Increment segment turn number.
5. Build multimodal inference/storage payloads when needed.
6. Call `ContinuumOrchestrator.process_message(...)`.
7. Commit Unit of Work (batched persistence).
8. Return response text + structured metadata.
9. Release request lock in `finally`.

## 5) Core Orchestrator

Core class:
- `cns/services/orchestrator.py:ContinuumOrchestrator`

Responsibilities:
- Add user/assistant/tool messages to continuum state
- Surface relevant long-term memories
- Compose system prompt sections
- Provide tool schemas to model
- Execute LLM response generation and tool loops
- Parse tags/metadata from model output
- Publish turn events and return metadata to API layer

Global singleton access:
- `initialize_orchestrator(...)`
- `get_orchestrator()`

## 6) Event-Driven Working Memory

Components:
- `cns/integration/event_bus.py` (synchronous in-process event bus)
- `cns/core/events.py` (typed domain events)
- `working_memory/core.py` (trinket coordinator)
- `working_memory/composer.py` (prompt section assembly)

How it works:
1. Orchestrator publishes `ComposeSystemPromptEvent`.
2. `WorkingMemory` requests updates from registered trinkets.
3. Trinkets publish `TrinketContentEvent`.
4. Composer merges sections in configured order.
5. Result is split into:
   - cached system content
   - non-cached system content
   - notification center content
6. `SystemPromptComposedEvent` is published back to orchestrator flow.

## 7) Tool System

Core:
- `tools/repo.py` (`Tool`, `ToolRepository`)

Key behavior:
- Discovers tool classes from `tools/implementations/*`.
- Enables "essential tools" at startup from config.
- Supports gated tools with runtime availability checks.
- Builds model tool definitions dynamically each turn.
- Invokes tools with per-request user context.

Notable pattern:
- `invokeother_tool` + dynamic tool loading allows smaller active toolsets.

## 8) Continuum State, Cache, and Persistence

Continuum aggregate:
- `cns/core/continuum.py`

Persistence:
- `cns/infrastructure/continuum_repository.py`

Session/cache coordinator:
- `cns/infrastructure/continuum_pool.py`

Pattern:
- Continuum messages are managed in-memory during the turn.
- `UnitOfWork` batches writes.
- Valkey-backed cache is used for session continuity.
- Repository methods persist continuums/messages with user isolation.

## 9) Data Backends

### Postgres
- `clients/postgres_client.py`
- Connection pooling and raw SQL.
- Per-connection `app.current_user_id` is set/reset for RLS context isolation.

### Valkey
- `clients/valkey_client.py`
- Unified cache + TTL monitoring + retry semantics.
- Used for hot cache/session state and other coordination patterns.

### Vault
- `clients/vault_client.py`
- Secret retrieval for database URLs, API keys, service config.
- AppRole env vars are required in integration-style environments.

### Per-user SQLite
- `utils/userdata_manager.py`
- Each user gets `data/users/<user_id>/userdata.db`.
- Tool/domain-specific local data with optional field encryption.

Schema bootstrap:
- `tools/schema_distribution.py`

## 10) Segment Lifecycle and Scheduled Work

Scheduled task registration:
- `utils/scheduled_tasks.py`

Segment timeout/collapse path:
1. Timeout job emits `SegmentTimeoutEvent`.
2. `SegmentCollapseHandler` handles event.
3. Active segment is summarized + embedded.
4. Sentinel/boundary metadata is updated.
5. Downstream processing events are triggered.
6. Cache invalidation and related cleanup run.

Main handler:
- `cns/services/segment_collapse_handler.py`

## 11) Long-Term Memory Stack (`lt_memory/`)

Factory:
- `lt_memory/factory.py:LTMemoryFactory`

Core services:
- `lt_memory/db_access.py` (DB gateway)
- `lt_memory/vector_ops.py` (embedding/similarity operations)
- `lt_memory/extraction.py`, `linking.py`, `refinement.py`
- `lt_memory/processing/*` (batch/execution orchestration)

Design intent:
- Explicit dependency wiring in factory
- Separation between real-time chat path and memory processing path
- Batch-capable processing for heavier extraction/linking tasks

## 12) Data and Action APIs

Read endpoint:
- `cns/api/data.py` (`GET /v0/api/data`)
- Type-routed retrieval for history, memories, dashboard, user profile, domaindocs, working memory.

Mutation endpoint:
- `cns/api/actions.py` (`POST /v0/api/actions`)
- Domain-routed command handling (reminder, memory, contacts, etc.) using tool/service calls.

## 13) Local Web UI

UI and file/artifact APIs:
- `api/webapp.py`

Provides:
- Browser chat page (`/webapp`)
- Authenticated file listing/download for output roots
- Anthropic code execution artifact listing/download

## 14) Testing Notes

Pytest configuration:
- `pytest.ini`

Current local baseline in this environment:
- `340 passed`
- `517 skipped`

Most skips are integration-gated tests requiring external services/credentials (Vault, Valkey, or provider keys) not present in this sandbox.

## 15) Suggested Reading Order

If you want to understand the codebase quickly, read in this order:
1. `main.py`
2. `cns/api/chat.py`
3. `cns/services/orchestrator.py`
4. `working_memory/core.py` and `working_memory/composer.py`
5. `tools/repo.py`
6. `cns/infrastructure/continuum_pool.py` and `cns/infrastructure/continuum_repository.py`
7. `clients/postgres_client.py`, `clients/valkey_client.py`, `clients/vault_client.py`
8. `lt_memory/factory.py` and `lt_memory/db_access.py`
