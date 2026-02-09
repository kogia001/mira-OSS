# Plan: Lower-Cost Main LLM Routing + OAuth OpenAI/Codex

## Objective
Reduce inference cost for primary chat and internal tasks by shifting eligible traffic to lower-cost providers (Groq-first), while adding user-scoped OAuth options for OpenAI and Codex in the normal MIRA flow.

## Expected Benefit
- Lower average cost per turn by routing non-specialized turns to Groq/OpenAI-compatible endpoints.
- Keep high-capability paths available (Anthropic) only when required (for example: Files API/code execution artifacts).
- Give users bring-your-own-provider control without exposing raw API keys in plaintext.
- Improve resilience with explicit provider failover paths instead of single-provider dependency.

## Current State (from codebase)
- Generic provider routing already exists through `LLMProvider.generate_response(...)` with `endpoint_url` + `model_override`.
- User-facing model selection is DB-backed via `account_tiers` (`provider`, `endpoint_url`, `api_key_name`, `model`).
- Internal tasks are DB-backed via `internal_llm` (`analysis`, `summary`, etc.).
- Sensitive gap: several features are Anthropic-bound today:
  - document upload/file lifecycle via `FilesManager` (Anthropic Files API)
  - artifact list/download endpoints in `api/webapp.py` use `orchestrator.llm_provider.anthropic_client`
- Auth model is single-user bearer in OSS mode.
- User-scoped encrypted credential storage already exists (`UserCredentialService` on top of encrypted user SQLite).

## Target Architecture

## 1) Capability-Aware Provider Router
Introduce an explicit provider capability matrix and route by capability need, not only by tier.

Example routing policy:
- If turn needs Anthropic-only capability (Files API upload, code_execution artifact lifecycle): force Anthropic.
- Else: prefer low-cost provider (Groq/OpenAI-compatible).
- If selected provider fails (401/403/5xx/rate limit): fail over to configured backup.

Capabilities to model:
- tool_calling
- streaming
- image_input
- file_upload_api
- code_execution_artifacts
- reasoning_blocks (optional display)
- prompt_caching support

## 2) Cost Policy Layer
Add a policy that decides model/provider by:
- user tier
- turn complexity (tool-heavy vs text-only)
- capability requirements
- live budget guardrails (daily/monthly target cost ceiling)

Initial pragmatic policy:
- `fast`: Groq model
- `balanced`: Groq model by default, Anthropic fallback when capability-required
- `nuanced`: Anthropic by default, optional user override to OpenAI/Codex OAuth if connected

## 3) Provider Credentials Model
Keep two credential classes:
- Global infra keys (existing Vault keys): default deployment-level routing.
- User OAuth connections (new): per-user OpenAI/Codex access.

Use existing encrypted credential layer first (no new secret system required):
- `credential_type = "llm_oauth"`
- `service_name in {"openai","codex"}`
- value JSON:
  - access_token
  - refresh_token (if provided)
  - expires_at
  - scope
  - account metadata (email/org/id)

## 4) OAuth Integration Flow
For each provider (OpenAI, Codex if supported):
1. Start endpoint generates PKCE challenge + state.
2. User completes provider consent screen.
3. Callback validates state, exchanges code, stores encrypted tokens.
4. Runtime token accessor handles refresh before expiry.
5. Router can use user token when user selected that provider.

Suggested endpoints:
- `POST /v0/api/providers/{provider}/oauth/start`
- `GET /v0/api/providers/{provider}/oauth/callback`
- `GET /v0/api/providers/connections`
- `DELETE /v0/api/providers/{provider}/disconnect`

## 5) UX/Command Surface
Extend existing tier flow with provider visibility.

Examples:
- `/provider` shows active provider and auth source (vault vs oauth).
- `/provider use groq|anthropic|openai|codex` (validated by tier + available auth).
- `/provider connect openai` and `/provider connect codex` open OAuth flow.

## Delivery Plan (No-Code Plan)

## Phase 0: Baseline + Guardrails
Scope:
- Add/verify observability fields for every LLM call:
  - provider
  - model
  - endpoint
  - token usage
  - latency
  - fallback reason
  - capability flags
- Define target KPIs:
  - cost per 1k user turns
  - p50/p95 latency
  - tool success rate
  - model-error rate

Exit criteria:
- We can compare Anthropic-only vs mixed-provider traffic with hard numbers.

## Phase 1: Low-Risk Cost Shift (DB/Config-First)
Scope:
- Move more internal tasks to Groq/OpenAI-compatible models where quality is acceptable.
- Set `fast` and optionally `balanced` to generic low-cost models via `account_tiers`.
- Keep Anthropic as fallback for capability-required turns.

Exit criteria:
- Cost reduction achieved with no regression in completion rate/tool loop stability.

## Phase 2: OAuth OpenAI + Codex Connections
Scope:
- Add provider connection endpoints and PKCE OAuth state management.
- Store user tokens in encrypted credential storage.
- Add runtime resolver: choose user token when provider is user-selected and connected.

Exit criteria:
- User can connect/disconnect OpenAI (and Codex if OAuth available) and run chat on connected provider.

## Phase 3: Capability Split Hardening
Scope:
- Make Anthropic-only paths explicit and deterministic:
  - document uploads
  - artifact retrieval endpoints
- Add clear user-facing fallback messaging when selected provider cannot satisfy capability.

Exit criteria:
- No silent failures when provider lacks file/code-exec features; routing remains predictable.

## Phase 4: Rollout + Safety Controls
Scope:
- Feature flags:
  - `provider_router_enabled`
  - `oauth_openai_enabled`
  - `oauth_codex_enabled`
  - `balanced_generic_default_enabled`
- Progressive rollout:
  - internal testing
  - selected users
  - full release

Exit criteria:
- Stable metrics across 7+ days and rollback tested.

## Key Design Decisions to Make Early
1. Should `balanced` default to Groq immediately or only after Phase 2 metrics?
2. Should provider choice be tier-bound only, or allow per-conversation override?
3. For Codex: API-provider route, CLI route, or both?
4. Token residence policy: encrypted user SQLite only, or mirrored to Vault for multi-node deployments?

## Risks and Mitigations
- Feature mismatch between providers.
  - Mitigation: capability matrix + forced Anthropic routing for unsupported features.
- OAuth token lifecycle failures.
  - Mitigation: refresh-before-expiry, retry once, clear re-auth prompts.
- Quality drift from cheaper models.
  - Mitigation: keep nuanced tier on Anthropic; add quality gates and fallback.
- Operational complexity from multi-provider routing.
  - Mitigation: strong tracing tags and explicit fallback reason codes.

## Test Strategy
- Unit:
  - routing policy decisions
  - capability gate checks
  - OAuth token storage/refresh logic
- Integration:
  - tier selection -> provider call path
  - fallback behavior on 401/403/429/5xx
  - Anthropic-required flows still work under mixed routing
- Regression:
  - chat endpoint structure
  - tool-call execution loop behavior
  - webapp artifact/file endpoints

## Suggested First Implementation Slice
Smallest high-impact slice:
1. Keep all Anthropic-bound file/artifact flows unchanged.
2. Route all eligible text-only turns in `fast` to Groq.
3. Add provider/latency/cost telemetry fields.
4. Validate savings and error rate for 1 week.
5. Then add OAuth OpenAI, then OAuth Codex.

This sequencing gives immediate cost benefit with minimal blast radius.
