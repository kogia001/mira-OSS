# MIRA-OSS Security Assessment

**Scope**: Single-user deployment with tools enabled
**Date**: 2026-02-21
**Methodology**: Static analysis of application source code across authentication, authorization, database security, tool execution, API surface, credential management, and dependency supply chain.

---

## Executive Summary

MIRA's security architecture is **well-designed for its intended single-user, self-hosted deployment model**. The combination of PostgreSQL Row-Level Security, contextvar-based user isolation, HashiCorp Vault credential management, and parameterized queries creates genuine defense-in-depth. No critical vulnerabilities were found that would allow remote code execution or full system compromise.

However, seven findings require attention - two at **HIGH** severity and five at **MEDIUM**. The most impactful issues are the unauthenticated federation endpoint (which allows arbitrary message injection) and the deterministic encryption key derivation (which renders per-user SQLite encryption ineffective against anyone with the user_id).

### Findings Summary

| ID | Severity | Category | Finding |
|----|----------|----------|---------|
| SEC-01 | HIGH | Authentication | Federation endpoint accepts unauthenticated requests |
| SEC-02 | HIGH | Cryptography | SQLite encryption key derived solely from user_id (no secret material) |
| SEC-03 | MEDIUM | Injection | IMAP search string injection in email_tool.py |
| SEC-04 | MEDIUM | Information Disclosure | Thread dump endpoint exposes internals without authentication |
| SEC-05 | MEDIUM | DoS | Rate limiting configured but not enforced |
| SEC-06 | MEDIUM | HTTP Security | Missing security headers (CSP, HSTS, X-Frame-Options) |
| SEC-07 | MEDIUM | Supply Chain | Dependencies unpinned - vulnerable to supply chain attacks |

---

## Architecture Strengths

Before detailing findings, these architectural decisions deserve recognition - they represent solid security engineering:

**1. PostgreSQL RLS with contextvar propagation** (`clients/postgres_client.py`, `deploy/mira_service_schema.sql`)
Every user-scoped table enforces `user_id = current_setting('app.current_user_id')::uuid`. The context flows from `auth/api.py` through Python contextvars to `SET app.current_user_id = %s` on every pooled connection. Pooled connections explicitly `RESET app.current_user_id` when no user context exists, preventing cross-request data leakage. Separate `mira_admin` role with BYPASSRLS for batch operations is correctly isolated.

**2. Vault-first credential management** (`clients/vault_client.py`)
AppRole authentication, no hardcoded secrets, no environment variable credentials for application secrets. Fail-fast on missing credentials at startup. API key generation uses `secrets.token_urlsafe(32)` (~256 bits entropy).

**3. Parameterized queries everywhere** (`clients/postgres_client.py`, `lt_memory/db_access.py`, `cns/infrastructure/continuum_repository.py`)
No SQL injection vectors found. All user input passes through `%s` or `%(name)s` parameter binding. Even the dynamic WHERE clause construction in `continuum_repository.py:514` parameterizes values correctly.

**4. Tool user isolation** (`tools/repo.py`)
Fresh tool instances per invocation (no cross-user caching). User-scoped file paths via `self.user_data_path`. Database access through `self.db` automatically scoped to current user. Code execution uses Anthropic's sandboxed environment, not local shell.

**5. Multi-layered prompt injection defense** (`utils/prompt_injection_defense.py`)
Pattern-based fast detection, optional LLM-based semantic analysis, and structural XML wrapping with tag escaping. Fails closed when LLM detection is unavailable.

**6. Path traversal protection** (`tools/implementations/notes_tool.py`, `api/webapp.py`)
`Path.resolve()` + `relative_to()` boundary checking. Symlink following disabled by default. Extension whitelisting and file size limits.

**7. SSRF protection** (`tools/implementations/web_tool.py`)
Private network pattern matching (localhost, 10.x, 172.16-31.x, 192.168.x, link-local), scheme validation (http/https only), domain allowlist/blocklist support.

---

## Detailed Findings

### SEC-01: Unauthenticated Federation Endpoint [HIGH]

**Location**: `api/federation.py:38-97`

**Description**: The `POST /v0/api/federation/deliver` endpoint accepts messages without any authentication. It takes a `to_user_id` in the request body, instantiates a `PagerTool` for that user, and delivers arbitrary content to their pager.

```python
@router.post("/federation/deliver", response_model=FederationDeliveryResponse)
def receive_federation_delivery(payload: FederationDeliveryPayload):
    # No authentication - no Depends(get_current_user)
    pager = PagerTool(user_id=payload.to_user_id)
    result = pager.deliver_federated_message(
        from_address=payload.from_address,  # Attacker-controlled
        content=payload.content,            # Attacker-controlled
        priority=payload.priority,
        metadata=payload.metadata
    )
```

**Impact**: Any network-reachable attacker can inject messages into a user's pager by knowing (or guessing) their user_id. In single-user mode the user_id is a static UUID created at first startup. The injected message content is attacker-controlled with no sanitization, creating a prompt injection vector if pager messages feed into LLM context.

**Recommendation**: Add bearer token authentication (`Depends(get_current_user)`) or a shared secret between Lattice and MIRA. Alternatively, if federation is not used in OSS single-user mode, remove this endpoint entirely.

---

### SEC-02: Deterministic Encryption Key Derivation [HIGH]

**Location**: `utils/userdata_manager.py:445-451`

**Description**: The encryption key for per-user SQLite databases is derived deterministically from the user_id alone, with no secret material:

```python
def derive_session_key(user_id: str) -> bytes:
    key_material = f"userdata_encryption_{user_id}".encode()
    return hashlib.sha256(key_material).digest()
```

**Impact**: Anyone with access to the filesystem who knows the user_id (a UUID stored in PostgreSQL, printed in logs, present in file paths like `data/users/{user_id}/`) can derive the encryption key and decrypt all "encrypted" fields in `userdata.db`. This includes credentials stored via `UserCredentialService` (email passwords, tool API keys). The Fernet encryption provides no actual confidentiality.

**Recommendation**: Derive the key from a secret stored in Vault combined with the user_id:
```python
def derive_session_key(user_id: str) -> bytes:
    vault_secret = get_auth_secret("userdata_encryption_key")
    key_material = f"{vault_secret}:{user_id}".encode()
    return hashlib.sha256(key_material).digest()
```

---

### SEC-03: IMAP Search String Injection [MEDIUM]

**Location**: `tools/implementations/email_tool.py:1068-1077` (and `1574-1583`)

**Description**: User-supplied search parameters are interpolated directly into IMAP search strings without escaping:

```python
search_parts.append(f'FROM "{sender}"')
search_parts.append(f'SUBJECT "{subject}"')
search_parts.append(f'SINCE "{start_date}"')
search_parts.append(f'BEFORE "{end_date}"')
```

**Impact**: An LLM-driven tool call with crafted parameters (e.g., `sender = 'x" UNSEEN ALL'`) could manipulate the IMAP search criteria to return unintended emails. Impact is limited to information disclosure within the user's own mailbox (reading emails they didn't intend to search for), since the IMAP connection is authenticated as that user. In single-user mode this is lower impact, but the principle of correct input handling still applies.

**Recommendation**: Escape double quotes in search parameters before interpolation:
```python
safe_sender = sender.replace('"', '\\"')
search_parts.append(f'FROM "{safe_sender}"')
```

---

### SEC-04: Unauthenticated Thread Dump Endpoint [MEDIUM]

**Location**: `cns/api/health.py:182-215`

**Description**: `GET /v0/api/health/thread-dump` returns complete thread state information (code locations, stack traces, operation names) without requiring authentication. It also writes the dump to a predictable path in `/tmp/`.

```python
@router.get("/health/thread-dump")
def thread_dump_endpoint():
    dump = ThreadMonitor.dump_thread_states()
    dump_file = f"/tmp/thread_dump_api_{int(time.time())}.txt"
    with open(dump_file, 'w') as f:
        f.write(dump)
```

**Impact**: Exposes internal code paths, function names, active operations, and system architecture to unauthenticated requests. This aids reconnaissance. The `/tmp/` file writes also create a local file accumulation issue.

**Recommendation**: Add `Depends(get_current_user)` to both `/health/threads` and `/health/thread-dump` endpoints. Remove the `/tmp/` file writing or make it configurable.

---

### SEC-05: Rate Limiting Configured But Not Enforced [MEDIUM]

**Location**: `config/config.py:65-66` (config), entire API surface (missing middleware)

**Description**: Configuration defines `rate_limit_rpm=10` and `burst_limit=5`, and Valkey has `atomic_increment()` infrastructure ready for rate limiting, but no middleware actually enforces these limits on any endpoint.

**Impact**: Unauthenticated endpoints (health, federation) are fully exposed to DoS. Authenticated endpoints have no per-user rate limiting beyond the single-request distributed lock on `/chat`. An attacker (or runaway client) could exhaust server resources, database connections, or LLM API quotas.

**Recommendation**: Implement rate limiting middleware using the existing Valkey infrastructure and config values. At minimum, rate-limit the `/chat` endpoint (LLM API cost) and unauthenticated endpoints (DoS).

---

### SEC-06: Missing HTTP Security Headers [MEDIUM]

**Location**: `main.py:500-507` (middleware configuration)

**Description**: The application configures CORS but adds no other security headers. Missing:
- `X-Frame-Options: DENY` (clickjacking protection)
- `X-Content-Type-Options: nosniff` (MIME sniffing prevention)
- `Strict-Transport-Security` (HTTPS enforcement)
- `Content-Security-Policy` (XSS mitigation for `/webapp`)
- `X-XSS-Protection: 0` (disable legacy XSS filter)

**Impact**: The `/webapp` endpoint serves an inline HTML page. Without CSP, any XSS in the rendered content could execute arbitrary JavaScript. Without X-Frame-Options, the webapp could be embedded in a malicious iframe for clickjacking.

**Recommendation**: Add a security headers middleware:
```python
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"
    return response
```

---

### SEC-07: Unpinned Dependencies [MEDIUM]

**Location**: `requirements.txt`

**Description**: Of 38+ dependencies, only 2 specify any version constraint (`spacy>=3.7.0`, `Pillow>=10.0.0`). The rest install whatever version is latest at `pip install` time. There is no `requirements.lock`, `pip-audit` configuration, or Dependabot setup. No Dockerfile exists to pin base images.

The dependency surface includes security-sensitive packages:
- `cryptography` (encryption primitives)
- `PyJWT` (token handling)
- `anthropic` / `openai` (API clients with credential handling)
- `psycopg2-binary` (database driver)
- `hvac` (Vault client)

Additionally, the `--extra-index-url https://download.pytorch.org/whl/cpu` adds a secondary package index, which creates a dependency confusion risk if any package on PyPI has the same name as a PyTorch-index-only package.

**Impact**: A compromised or typosquatted package version could execute arbitrary code during installation. Without pinning, builds are not reproducible and a previously-safe deployment could become compromised on re-deploy.

**Recommendation**:
1. Pin all dependencies to exact versions (`package==x.y.z`)
2. Generate a lock file (`pip freeze > requirements.lock`)
3. Run `pip-audit` in CI to catch known CVEs
4. Consider using `--require-hashes` for integrity verification

---

## Additional Observations (Low Severity / Informational)

### Content-Disposition Header Injection
**Location**: `api/webapp.py:1638`
Filename from Anthropic API metadata is interpolated into `Content-Disposition` header without sanitizing quotes or newlines. Low impact since the metadata comes from Anthropic's API (not user input), but worth fixing for defense-in-depth.

### API Key Printed to Console
**Location**: `main.py:146`
The generated API key is printed to stdout on startup. Expected for local OSS usage, but could leak into container logs or process monitoring in more complex deployments.

### Bearer Token Has No TTL
**Location**: `auth/api.py`
The API key never expires and has no rotation mechanism. Acceptable for single-user local deployments but worth noting for anyone extending to multi-user.

### Codex Tool Sensitive Keyword Blocklist
**Location**: `tools/implementations/codex_tool_mira.py:54-93`
Uses a blacklist of ~80 keywords to filter dangerous commands. Blacklists are inherently bypassable (encoding, variable expansion, aliasing). The subprocess list-based command execution prevents shell injection regardless, so this is a secondary defense layer. A whitelist approach would be stronger.

### Vault TLS Disabled Locally
**Location**: `config/vault.hcl:5`
`tls_disable = 1` for the local Vault listener. Expected for localhost-only development, but the Vault docs should note this must change for any non-localhost deployment.

---

## Scope Limitations

This assessment covers static code analysis only. The following were **not** evaluated:
- Runtime behavior under load or attack
- Network configuration and firewall rules
- Operating system and kernel hardening
- Vault policy granularity and audit logging
- PostgreSQL configuration hardening (pg_hba.conf, SSL)
- Actual versions of installed dependencies (only constraints in requirements.txt)
- Third-party service security (Anthropic API, OpenAI API, Google Maps API)
