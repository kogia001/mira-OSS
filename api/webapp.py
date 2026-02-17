"""
Local web UI for chatting with MIRA and browsing/downloading sandbox output files.

UI endpoint:
- GET /webapp

Authenticated API endpoints:
- GET /v0/api/webapp/files
- GET /v0/api/webapp/download
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response

from auth.api import get_current_user

router = APIRouter()


def _get_output_roots() -> Dict[str, Path]:
    """Resolve candidate output roots and expose stable root keys."""
    roots: Dict[str, Path] = {}

    configured = os.getenv("OUTPUT_DIR")
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if configured_path.exists():
            roots["configured"] = configured_path

    # Common tool output locations in local/containerized runs
    for key, raw in (
        ("files_output", "/files/output"),
        ("files_root", "/files"),
        ("tmp", "/tmp"),
    ):
        path = Path(raw).resolve()
        if path.exists():
            roots[key] = path

    if not roots:
        fallback = Path("/tmp").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        roots["tmp"] = fallback

    return roots


def _select_output_root(root_key: Optional[str]) -> tuple[str, Path]:
    """Pick the active root by key (or default)."""
    roots = _get_output_roots()

    if root_key:
        normalized = root_key.strip()
        if normalized not in roots:
            raise HTTPException(status_code=400, detail="Invalid root")
        return normalized, roots[normalized]

    # Prefer configured output dir when available.
    if "configured" in roots:
        return "configured", roots["configured"]

    first_key = next(iter(roots))
    return first_key, roots[first_key]


def _ensure_within_root(path: Path, root: Path) -> None:
    """Reject path traversal outside configured output root."""
    if path != root and root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid path")


def _resolve_relative_path(relative_path: str, root: Path) -> Path:
    """Resolve a user-supplied relative path under output root."""
    cleaned = (relative_path or "").strip().lstrip("/")
    resolved = (root / cleaned).resolve()
    _ensure_within_root(resolved, root)
    return resolved


def _get_continuum_id_for_current_user() -> str:
    """Get current user's active continuum ID."""
    from cns.infrastructure.continuum_pool import get_continuum_pool

    continuum = get_continuum_pool().get_or_create()
    return str(continuum.id)


def _load_cached_artifacts(continuum_id: str) -> List[Dict[str, str]]:
    """Load cached code_execution artifact IDs from Valkey."""
    from clients.valkey_client import get_valkey

    raw = get_valkey().get(f"codeexec_artifacts:{continuum_id}")
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    artifacts: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        file_id = item.get("file_id")
        if isinstance(file_id, str) and file_id:
            artifacts.append(
                {
                    "file_id": file_id,
                    "captured_at": str(item.get("captured_at") or ""),
                }
            )
    return artifacts


@router.get("/webapp", response_class=HTMLResponse)
def webapp_index() -> str:
    """Serve browser UI for chat and output file access."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MIRA Local Webapp</title>
  <style>
    :root {
      --bg: #edf2f7;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #d9e2ec;
      --accent: #0f766e;
      --accent-2: #134e4a;
      --danger: #7f1d1d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 0%, #d9f2ea 0%, #edf2f7 45%, #eef2ff 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1040px;
      margin: 26px auto;
      padding: 0 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
      padding: 16px;
    }
    h1 {
      margin: 0 0 10px 0;
      font-size: 1.35rem;
      letter-spacing: 0.01em;
    }
    .muted { color: var(--muted); font-size: 0.92rem; }
    .mono { font-family: ui-monospace, Menlo, Consolas, monospace; }
    .tabs {
      display: flex;
      gap: 8px;
      margin: 14px 0;
    }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--accent-2);
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font-size: 0.92rem;
    }
    .tab.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .panel { display: none; }
    .panel.active { display: block; }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      margin: 10px 0;
    }
    .row-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 10px 0;
    }
    input, button, textarea {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 0.95rem;
      font-family: inherit;
    }
    input, textarea { width: 100%; }
    button {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: var(--accent-2);
      border-color: var(--line);
    }
    button:disabled {
      opacity: 0.65;
      cursor: not-allowed;
    }
    .status {
      min-height: 22px;
      margin-top: 6px;
      color: var(--danger);
      font-size: 0.9rem;
    }
    .chat-log {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f8fafc;
      height: min(56vh, 560px);
      overflow: auto;
      padding: 10px;
    }
    .bubble {
      margin: 10px 0;
      padding: 10px 12px;
      border-radius: 10px;
      white-space: pre-wrap;
      line-height: 1.35;
      font-size: 0.95rem;
      max-width: 92%;
    }
    .bubble.user {
      background: #def7ec;
      border: 1px solid #b7e4d3;
      margin-left: auto;
    }
    .bubble.assistant {
      background: #fff;
      border: 1px solid #d7dee7;
      margin-right: auto;
    }
    .bubble .who {
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      margin-bottom: 4px;
    }
    .chat-input-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      margin-top: 10px;
      align-items: end;
    }
    .chat-history-row {
      display: flex;
      justify-content: flex-end;
      margin-top: 10px;
    }
    textarea { min-height: 74px; resize: vertical; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      vertical-align: middle;
      font-size: 0.92rem;
    }
    th { color: var(--muted); font-weight: 600; }
    .name-btn {
      border: 0;
      background: transparent;
      color: var(--accent-2);
      padding: 0;
      cursor: pointer;
      font: inherit;
      text-decoration: underline;
    }
    @media (max-width: 760px) {
      .chat-input-row { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .row-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>MIRA Local Webapp</h1>
      <div class="muted">Connect with your MIRA API key, chat on localhost, and browse files from OUTPUT_DIR.</div>

      <div class="row">
        <input id="apiKey" placeholder="Paste MIRA API key (mira_...)" />
        <button id="saveKeyBtn">Save Key</button>
      </div>

      <div class="tabs">
        <button class="tab active" id="chatTab" data-tab="chat">Chat</button>
        <button class="tab" id="filesTab" data-tab="files">Files</button>
        <button class="tab" id="domaindocsTab" data-tab="domaindocs">DomainDocs</button>
      </div>

      <div class="panel active" id="chatPanel">
        <div class="muted">Endpoints: <span class="mono">POST /v0/api/chat</span>, <span class="mono">GET /v0/api/data?type=history</span>, <span class="mono">POST /v0/api/actions</span></div>
        <div class="status" id="chatStatus"></div>
        <div class="chat-history-row">
          <button class="secondary" id="loadOlderBtn">Load Older</button>
        </div>
        <div class="chat-log" id="chatLog"></div>
        <div class="chat-input-row">
          <textarea id="chatInput" placeholder="Type your message or /help. Shift+Enter for a new line."></textarea>
          <button id="sendBtn">Send</button>
          <button class="secondary" id="clearBtn">Clear</button>
        </div>
      </div>

      <div class="panel" id="filesPanel">
        <div class="muted">Endpoint: <span class="mono">GET /v0/api/webapp/files</span></div>
        <div class="row">
          <select id="rootSelect"></select>
          <button class="secondary" id="reloadBtn">Reload</button>
        </div>
        <div class="row">
          <input id="pathInput" placeholder="Path relative to OUTPUT_DIR (blank for root)" />
          <button class="secondary" id="goBtn">Go</button>
        </div>
        <div class="muted">Output root: <span id="rootPath" class="mono">-</span></div>
        <div class="muted">Current path: <span id="currentPath" class="mono">/</span></div>
        <div class="status" id="filesStatus"></div>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Size</th>
              <th>Modified</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>

        <div style="height: 12px"></div>
        <div class="muted">Anthropic sandbox artifacts (code_execution): <span class="mono">GET /v0/api/webapp/artifacts</span></div>
        <div class="row">
          <button class="secondary" id="reloadArtifactsBtn">Reload Artifacts</button>
        </div>
        <div class="status" id="artifactStatus"></div>
        <table>
          <thead>
            <tr>
              <th>File</th>
              <th>Size</th>
              <th>Created</th>
              <th>Captured</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="artifactRows"></tbody>
        </table>
      </div>

      <div class="panel" id="domaindocsPanel">
        <div class="muted">DomainDoc editor via <span class="mono">POST /v0/api/actions</span> (<span class="mono">domain_knowledge</span>)</div>
        <div class="status" id="domaindocsStatus"></div>

        <div class="row">
          <select id="docSelect"></select>
          <button class="secondary" id="reloadDocsBtn">Reload Docs</button>
        </div>

        <div class="row-2">
          <input id="newDocLabelInput" placeholder="new_doc_label" />
          <input id="newDocDescInput" placeholder="Description for the new DomainDoc" />
        </div>
        <div class="row">
          <button id="createDocBtn">Create Doc</button>
          <button class="secondary" id="enableDocBtn">Enable</button>
        </div>
        <div class="row">
          <div class="muted" id="docMeta">Select a DomainDoc to edit sections.</div>
          <button class="secondary" id="disableDocBtn">Disable</button>
        </div>

        <hr style="border:0;border-top:1px solid var(--line);margin:10px 0;" />

        <div class="row">
          <select id="sectionSelect"></select>
          <button class="secondary" id="reloadSectionsBtn">Reload Sections</button>
        </div>
        <div class="row">
          <input id="newSectionInput" placeholder="New section header (e.g. API_NOTES)" />
          <button class="secondary" id="createSectionBtn">Create Section</button>
        </div>
        <div class="row">
          <button class="secondary" id="expandSectionBtn">Expand</button>
          <button class="secondary" id="collapseSectionBtn">Collapse</button>
        </div>

        <div class="row">
          <textarea id="sectionContentInput" placeholder="Section content..." style="min-height: 260px;"></textarea>
          <button id="saveSectionBtn">Save Section</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const apiKeyInput = document.getElementById('apiKey');
    const saveKeyBtn = document.getElementById('saveKeyBtn');

    const chatTab = document.getElementById('chatTab');
    const filesTab = document.getElementById('filesTab');
    const domaindocsTab = document.getElementById('domaindocsTab');
    const chatPanel = document.getElementById('chatPanel');
    const filesPanel = document.getElementById('filesPanel');
    const domaindocsPanel = document.getElementById('domaindocsPanel');

    const chatStatusEl = document.getElementById('chatStatus');
    const loadOlderBtn = document.getElementById('loadOlderBtn');
    const chatLogEl = document.getElementById('chatLog');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');

    const pathInput = document.getElementById('pathInput');
    const rootSelect = document.getElementById('rootSelect');
    const filesStatusEl = document.getElementById('filesStatus');
    const rowsEl = document.getElementById('rows');
    const currentPathEl = document.getElementById('currentPath');
    const rootPathEl = document.getElementById('rootPath');
    const goBtn = document.getElementById('goBtn');
    const reloadBtn = document.getElementById('reloadBtn');
    const artifactStatusEl = document.getElementById('artifactStatus');
    const artifactRowsEl = document.getElementById('artifactRows');
    const reloadArtifactsBtn = document.getElementById('reloadArtifactsBtn');

    const domaindocsStatusEl = document.getElementById('domaindocsStatus');
    const docSelect = document.getElementById('docSelect');
    const reloadDocsBtn = document.getElementById('reloadDocsBtn');
    const newDocLabelInput = document.getElementById('newDocLabelInput');
    const newDocDescInput = document.getElementById('newDocDescInput');
    const createDocBtn = document.getElementById('createDocBtn');
    const enableDocBtn = document.getElementById('enableDocBtn');
    const disableDocBtn = document.getElementById('disableDocBtn');
    const docMetaEl = document.getElementById('docMeta');
    const sectionSelect = document.getElementById('sectionSelect');
    const reloadSectionsBtn = document.getElementById('reloadSectionsBtn');
    const newSectionInput = document.getElementById('newSectionInput');
    const createSectionBtn = document.getElementById('createSectionBtn');
    const expandSectionBtn = document.getElementById('expandSectionBtn');
    const collapseSectionBtn = document.getElementById('collapseSectionBtn');
    const sectionContentInput = document.getElementById('sectionContentInput');
    const saveSectionBtn = document.getElementById('saveSectionBtn');

    const COMMAND_HELP_TEXT = '/tier [name] - view or change model tier\\n/domaindoc list|create|enable|disable\\n/collapse - end current conversation segment\\n/status\\n/clear\\nquit, exit, bye';

    let currentPath = '';
    let currentRoot = '';
    let currentDocLabel = '';
    let currentSectionHeader = '';
    let docsCache = [];
    let historyBootstrapped = false;
    let historyOffset = 0;
    let historyHasMore = false;
    const historyPageSize = 100;

    const savedKey = localStorage.getItem('mira_api_key');
    if (savedKey) apiKeyInput.value = savedKey;

    function authHeaders(withJson = false) {
      const key = apiKeyInput.value.trim();
      if (!key) throw new Error('API key is required');
      const headers = { Authorization: 'Bearer ' + key };
      if (withJson) headers['Content-Type'] = 'application/json';
      return headers;
    }

    function setTab(tabName) {
      const chatActive = tabName === 'chat';
      const filesActive = tabName === 'files';
      const domaindocsActive = tabName === 'domaindocs';
      chatTab.classList.toggle('active', chatActive);
      filesTab.classList.toggle('active', filesActive);
      domaindocsTab.classList.toggle('active', domaindocsActive);
      chatPanel.classList.toggle('active', chatActive);
      filesPanel.classList.toggle('active', filesActive);
      domaindocsPanel.classList.toggle('active', domaindocsActive);
    }

    function createBubble(who, text) {
      const div = document.createElement('div');
      div.className = 'bubble ' + who;
      const whoLabel = who === 'user' ? 'You' : 'MIRA';
      div.innerHTML = '<span class="who">' + whoLabel + '</span>' + escapeHtml(text || '');
      return div;
    }

    function appendBubble(who, text) {
      const div = createBubble(who, text);
      chatLogEl.appendChild(div);
      chatLogEl.scrollTop = chatLogEl.scrollHeight;
    }

    function prependBubble(who, text) {
      const div = createBubble(who, text);
      chatLogEl.insertBefore(div, chatLogEl.firstChild);
    }

    function updateLoadOlderButton() {
      loadOlderBtn.disabled = !historyHasMore || !apiKeyInput.value.trim();
      loadOlderBtn.textContent = historyHasMore ? 'Load Older' : 'No Older Messages';
    }

    function escapeHtml(str) {
      return String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function stripEmotionTag(text) {
      return String(text || '')
        .replace(/<think(?:\\s[^>]*)?>[\\s\\S]*?<\\/think>/gi, '')
        .replace(/<\\/?think(?:\\s[^>]*)?>/gi, '')
        .replace(/<tool_call(?:\\s[^>]*)?>[\\s\\S]*?<\\/tool_call>/gi, '')
        .replace(/<\\/?tool_call(?:\\s[^>]*)?>/gi, '')
        .replace(/\\n?<mira:my_emotion>[\\s\\S]*?<\\/mira:my_emotion>/g, '')
        .trim();
    }

    function contentToText(content) {
      if (typeof content === 'string') {
        return content;
      }

      if (!Array.isArray(content)) {
        return '';
      }

      const parts = [];
      for (const block of content) {
        if (!block || typeof block !== 'object') {
          continue;
        }
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push(block.text);
          continue;
        }
        if (block.type === 'image') {
          parts.push('[image]');
          continue;
        }
        if (block.type === 'document') {
          parts.push('[document]');
          continue;
        }
        if (block.type === 'container_upload') {
          parts.push('[file]');
          continue;
        }
        parts.push('[media]');
      }

      return parts.join('\\n').trim();
    }

    function parseErrorDetail(payload, status) {
      return payload?.error?.message || payload?.detail || ('Request failed (' + status + ')');
    }

    async function fetchJson(url, options = {}) {
      const res = await fetch(url, options);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(parseErrorDetail(payload, res.status));
      }
      return payload;
    }

    async function callAction(domain, action, data = {}) {
      const payload = await fetchJson('/v0/api/actions', {
        method: 'POST',
        headers: authHeaders(true),
        body: JSON.stringify({ domain, action, data }),
      });
      if (payload.success === false) {
        throw new Error(payload?.error?.message || 'Action failed');
      }
      return payload.data || {};
    }

    async function callData(query) {
      const params = new URLSearchParams(query);
      const payload = await fetchJson('/v0/api/data?' + params.toString(), {
        headers: authHeaders(),
      });
      if (payload.success === false) {
        throw new Error(payload?.error?.message || 'Data request failed');
      }
      return payload.data || {};
    }

    async function callHealth() {
      try {
        const payload = await fetchJson('/v0/api/health');
        return payload.data || {};
      } catch (_err) {
        return { status: 'unknown', components: {} };
      }
    }

    async function safeCall(fn, fallback) {
      try {
        return await fn();
      } catch (_err) {
        return fallback;
      }
    }

    async function loadChatHistory(options = {}) {
      const reset = options.reset !== false;
      const limit = options.limit || historyPageSize;
      chatStatusEl.textContent = '';
      try {
        localStorage.setItem('mira_api_key', apiKeyInput.value.trim());
        if (reset) {
          historyOffset = 0;
          historyHasMore = false;
          chatLogEl.innerHTML = '';
        }

        const historyData = await callData({
          type: 'history',
          offset: String(historyOffset),
          limit: String(limit),
          message_type: 'regular',
        });

        const rawMessages = Array.isArray(historyData.messages) ? historyData.messages : [];
        const normalized = rawMessages
          .filter((msg) => msg && (msg.role === 'user' || msg.role === 'assistant'))
          .map((msg) => {
            const rawText = contentToText(msg.content);
            const cleanedText = msg.role === 'assistant' ? stripEmotionTag(rawText) : rawText;
            return { role: msg.role, text: cleanedText };
          })
          .filter((msg) => msg.text);

        historyOffset += rawMessages.length;
        historyHasMore = Boolean(historyData?.meta?.has_more);
        updateLoadOlderButton();

        if (normalized.length === 0) {
          if (reset) {
            appendBubble('assistant', 'No prior messages found. Send a message or run /help.');
          } else {
            chatStatusEl.textContent = 'No older messages found.';
          }
          return;
        }

        const chunkChronological = normalized.reverse();
        if (reset) {
          for (const msg of chunkChronological) {
            appendBubble(msg.role === 'user' ? 'user' : 'assistant', msg.text);
          }
          chatStatusEl.textContent = 'Loaded ' + normalized.length + ' message(s) from history.';
          return;
        }

        const beforeHeight = chatLogEl.scrollHeight;
        const beforeTop = chatLogEl.scrollTop;
        for (let idx = chunkChronological.length - 1; idx >= 0; idx -= 1) {
          const msg = chunkChronological[idx];
          prependBubble(msg.role === 'user' ? 'user' : 'assistant', msg.text);
        }
        const addedHeight = chatLogEl.scrollHeight - beforeHeight;
        chatLogEl.scrollTop = beforeTop + addedHeight;
        chatStatusEl.textContent = 'Loaded ' + normalized.length + ' older message(s).';
      } catch (err) {
        chatStatusEl.textContent = err.message || String(err);
        updateLoadOlderButton();
      }
    }

    async function bootstrapChatWithCurrentKey(force = false) {
      if (!apiKeyInput.value.trim()) {
        return false;
      }

      if (historyBootstrapped && !force) {
        return true;
      }

      await loadChatHistory({ reset: true });
      historyBootstrapped = true;
      return true;
    }

    async function handleStatusCommand() {
      const [health, memoryData, userData, segmentData, tierData] = await Promise.all([
        callHealth(),
        safeCall(() => callData({ type: 'memories', limit: '100' }), { meta: { total_returned: 0, has_more: false } }),
        safeCall(() => callData({ type: 'user' }), {}),
        safeCall(() => callAction('continuum', 'get_segment_status', {}), {}),
        safeCall(() => callAction('continuum', 'get_llm_tier', {}), { tier: 'balanced', available_tiers: [] }),
      ]);

      const healthStatus = health.status || 'unknown';
      const dbLatency = health?.components?.database?.latency_ms;
      const latencyLabel = dbLatency ? ' (' + dbLatency + 'ms)' : '';

      const memoryMeta = memoryData.meta || {};
      const memoryCount = Number(memoryMeta.total_returned || 0);
      const hasMoreMemories = memoryMeta.has_more === true;
      const memoryLabel = hasMoreMemories ? (memoryCount + '+') : String(memoryCount);

      const tier = tierData.tier || 'balanced';
      const availableTiers = Array.isArray(tierData.available_tiers) ? tierData.available_tiers : [];
      const tierDescription = (availableTiers.find((t) => t.name === tier) || {}).description || tier;

      const lines = [];
      lines.push('System      ' + healthStatus + latencyLabel);
      lines.push('Memories    ' + memoryLabel + ' stored');

      if (segmentData && segmentData.has_active_segment === true) {
        if (segmentData.collapse_at) {
          lines.push('Segment     collapses at ' + segmentData.collapse_at);
        } else {
          lines.push('Segment     active');
        }
      } else if (segmentData && segmentData.has_active_segment === false) {
        lines.push('Segment     collapsed');
      }

      lines.push('');
      lines.push('Tier        ' + tier + ' (' + tierDescription + ')');
      lines.push('');

      if (userData && userData.profile && userData.profile.email) {
        lines.push('User        ' + userData.profile.email);
      }
      if (userData && userData.preferences && userData.preferences.timezone) {
        lines.push('Timezone    ' + userData.preferences.timezone);
      }

      appendBubble('assistant', lines.join('\\n'));
    }

    async function handleTierCommand(argRaw) {
      const tierData = await callAction('continuum', 'get_llm_tier', {});
      const currentTier = tierData.tier || 'balanced';
      const availableTiers = Array.isArray(tierData.available_tiers) ? tierData.available_tiers : [];
      const accessibleTiers = availableTiers.filter((t) => t.accessible);
      const accessibleNames = accessibleTiers.map((t) => t.name);

      const arg = (argRaw || '').trim();
      if (!arg) {
        const currentModel = (availableTiers.find((t) => t.name === currentTier) || {}).model || currentTier;
        const lines = ['Current: ' + currentTier + ' (' + currentModel + ')', ''];
        for (const tier of accessibleTiers) {
          const marker = tier.name === currentTier ? '->' : '  ';
          lines.push(marker + ' ' + tier.name + ': ' + (tier.model || tier.description || tier.name));
        }
        lines.push('');
        lines.push('Use /tier <name> to switch');
        appendBubble('assistant', lines.join('\\n'));
        return;
      }

      const loweredArg = arg.toLowerCase();
      const targetTier = accessibleTiers.find((tier) => {
        const tierName = String(tier.name || '').toLowerCase();
        const tierModel = String(tier.model || '').toLowerCase();
        return tierName === loweredArg || tierModel === loweredArg;
      });

      if (targetTier) {
        await callAction('continuum', 'set_llm_tier', { tier: targetTier.name });
        appendBubble('assistant', "Tier set to '" + targetTier.name + "'.");
        return;
      }

      const lockedTier = availableTiers.find((tier) => {
        const tierName = String(tier.name || '').toLowerCase();
        const tierModel = String(tier.model || '').toLowerCase();
        return (tierName === loweredArg || tierModel === loweredArg) && !tier.accessible;
      });

      if (lockedTier && lockedTier.locked_message) {
        appendBubble('assistant', "Tier '" + arg + "' is locked: " + lockedTier.locked_message);
        return;
      }

      appendBubble('assistant', 'Options: ' + accessibleNames.join(', '));
    }

    function parseDomaindocCreateArg(rawArg) {
      const createExpr = rawArg.trim().slice('create '.length).trim();
      const match = createExpr.match(/^([A-Za-z0-9_]+)\\s+\"([\\s\\S]+)\"$/);
      if (!match) {
        return null;
      }
      return { label: match[1], description: match[2] };
    }

    async function handleDomaindocCommand(argRaw) {
      const rawArg = (argRaw || '').trim();
      if (!rawArg) {
        appendBubble('assistant', '/domaindoc list\\n/domaindoc create <label> "<description>"\\n/domaindoc enable <label>\\n/domaindoc disable <label>');
        return;
      }

      const lowered = rawArg.toLowerCase();
      if (lowered === 'list') {
        const result = await callAction('domain_knowledge', 'list', {});
        const docs = Array.isArray(result.domaindocs) ? result.domaindocs : [];
        if (docs.length === 0) {
          appendBubble('assistant', 'No domaindocs found. Create one with /domaindoc create <label> "<description>"');
          return;
        }

        const lines = docs.map((doc) => {
          const status = doc.enabled ? '[enabled]' : '[disabled]';
          return status + ' ' + doc.label + ': ' + (doc.description || '');
        });
        appendBubble('assistant', lines.join('\\n'));
        return;
      }

      if (lowered.startsWith('create ')) {
        const parsed = parseDomaindocCreateArg(rawArg);
        if (!parsed) {
          appendBubble('assistant', 'Usage: /domaindoc create <label> "<description>"');
          return;
        }
        await callAction('domain_knowledge', 'create', parsed);
        appendBubble('assistant', "Created domaindoc '" + parsed.label + "'.");
        return;
      }

      if (lowered === 'enable' || lowered.startsWith('enable ')) {
        const label = rawArg.slice('enable'.length).trim();
        if (!label) {
          appendBubble('assistant', 'Usage: /domaindoc enable <label>');
          return;
        }
        await callAction('domain_knowledge', 'enable', { label });
        appendBubble('assistant', "Enabled domaindoc '" + label + "'.");
        return;
      }

      if (lowered === 'disable' || lowered.startsWith('disable ')) {
        const label = rawArg.slice('disable'.length).trim();
        if (!label) {
          appendBubble('assistant', 'Usage: /domaindoc disable <label>');
          return;
        }
        await callAction('domain_knowledge', 'disable', { label });
        appendBubble('assistant', "Disabled domaindoc '" + label + "'.");
        return;
      }

      appendBubble('assistant', 'Unknown domaindoc command: ' + rawArg + '\\nTry: /domaindoc list, create, enable, disable');
    }

    async function executeSlashCommand(rawMessage) {
      const stripped = rawMessage.slice(1).trim();
      if (!stripped) {
        appendBubble('assistant', 'Empty command. Try /help.');
        return;
      }

      const splitAt = stripped.indexOf(' ');
      const cmd = (splitAt >= 0 ? stripped.slice(0, splitAt) : stripped).toLowerCase();
      const argRaw = splitAt >= 0 ? stripped.slice(splitAt + 1) : '';

      if (cmd === 'help') {
        appendBubble('assistant', COMMAND_HELP_TEXT);
        return;
      }
      if (cmd === 'status') {
        await handleStatusCommand();
        return;
      }
      if (cmd === 'tier') {
        await handleTierCommand(argRaw);
        return;
      }
      if (cmd === 'clear') {
        chatLogEl.innerHTML = '';
        historyOffset = 0;
        historyHasMore = false;
        historyBootstrapped = false;
        updateLoadOlderButton();
        chatStatusEl.textContent = 'Chat log cleared.';
        return;
      }
      if (cmd === 'collapse') {
        const result = await callAction('continuum', 'collapse_segment', {});
        if (result.collapsed) {
          appendBubble('assistant', 'The previous conversation segment has been collapsed. Feel free to continue.');
          return;
        }
        appendBubble('assistant', 'Failed to collapse segment.');
        return;
      }
      if (cmd === 'domaindoc') {
        await handleDomaindocCommand(argRaw);
        return;
      }
      appendBubble('assistant', 'Unknown: /' + cmd);
    }

    async function sendChatMessage() {
      chatStatusEl.textContent = '';
      const message = chatInput.value.trim();
      if (!message) {
        chatStatusEl.textContent = 'Message cannot be empty.';
        return;
      }

      try {
        const headers = authHeaders(true);
        localStorage.setItem('mira_api_key', apiKeyInput.value.trim());

        appendBubble('user', message);
        chatInput.value = '';
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';

        if (message.startsWith('/')) {
          await executeSlashCommand(message);
          return;
        }

        const payload = await fetchJson('/v0/api/chat', {
          method: 'POST',
          headers,
          body: JSON.stringify({ message }),
        });

        if (payload.success === false) {
          throw new Error(payload?.error?.message || 'Chat request failed');
        }

        const responseText = stripEmotionTag(payload?.data?.response || payload?.response || '(No response text)');
        const responseMeta = payload?.data?.metadata || {};
        appendBubble('assistant', responseText);

        const toolsUsed = Array.isArray(responseMeta.tools_used) ? responseMeta.tools_used : [];
        const artifactIds = Array.isArray(responseMeta.code_execution_file_ids) ? responseMeta.code_execution_file_ids : [];
        if (responseMeta.output_claim_blocked === true) {
          chatStatusEl.textContent = 'Strict guard blocked an unverified file-creation claim.';
        } else if (toolsUsed.includes('code_execution')) {
          if (artifactIds.length > 0) {
            chatStatusEl.textContent = 'code_execution produced ' + artifactIds.length + ' downloadable artifact(s).';
          } else {
            chatStatusEl.textContent = 'code_execution ran, but no artifact file_id was returned.';
          }
        } else if (artifactIds.length > 0) {
          chatStatusEl.textContent = 'Received ' + artifactIds.length + ' artifact id(s).';
        } else {
          chatStatusEl.textContent = 'No code_execution artifacts were captured for this turn.';
        }
        await loadArtifacts();
      } catch (err) {
        chatStatusEl.textContent = err.message || String(err);
      } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
      }
    }

    function fmtBytes(size) {
      if (size === null || size === undefined) return '-';
      if (size < 1024) return size + ' B';
      if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KB';
      return (size / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function renderRootOptions(options, selected) {
      rootSelect.innerHTML = '';
      for (const item of options || []) {
        const opt = document.createElement('option');
        opt.value = item.key;
        opt.textContent = item.label;
        if (item.key === selected) {
          opt.selected = true;
        }
        rootSelect.appendChild(opt);
      }
    }

    async function loadFiles(path = '') {
      filesStatusEl.textContent = '';
      rowsEl.innerHTML = '';
      try {
        const headers = authHeaders();
        localStorage.setItem('mira_api_key', apiKeyInput.value.trim());
        const rootParam = rootSelect.value || currentRoot;
        let url = '/v0/api/webapp/files?path=' + encodeURIComponent(path);
        if (rootParam) {
          url += '&root=' + encodeURIComponent(rootParam);
        }
        const data = await fetchJson(url, { headers });
        currentPath = data.current_path || '';
        currentRoot = data.root_key || '';
        pathInput.value = currentPath;
        currentPathEl.textContent = '/' + currentPath;
        rootPathEl.textContent = data.output_root;
        renderRootOptions(data.available_roots, currentRoot);
        renderRows(data);
      } catch (err) {
        filesStatusEl.textContent = err.message || String(err);
      }
    }

    function renderRows(payload) {
      if (payload.parent_path !== null) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td><button class="name-btn" data-open="' + payload.parent_path + '">..</button></td><td>dir</td><td>-</td><td>-</td><td>-</td>';
        rowsEl.appendChild(tr);
      }

      for (const entry of payload.entries) {
        const tr = document.createElement('tr');
        const openBtn = entry.type === 'dir'
          ? '<button class="name-btn" data-open="' + entry.path + '">' + escapeHtml(entry.name) + '/</button>'
          : '<span class="mono">' + escapeHtml(entry.name) + '</span>';
        const actionBtn = entry.type === 'file'
          ? '<button data-download="' + entry.path + '">Download</button>'
          : '-';
        tr.innerHTML =
          '<td>' + openBtn + '</td>' +
          '<td>' + entry.type + '</td>' +
          '<td>' + fmtBytes(entry.size_bytes) + '</td>' +
          '<td>' + (entry.modified_at || '-') + '</td>' +
          '<td>' + actionBtn + '</td>';
        rowsEl.appendChild(tr);
      }
    }

    async function downloadFile(path) {
      filesStatusEl.textContent = '';
      try {
        const headers = authHeaders();
        const rootParam = rootSelect.value || currentRoot;
        let requestUrl = '/v0/api/webapp/download?path=' + encodeURIComponent(path);
        if (rootParam) {
          requestUrl += '&root=' + encodeURIComponent(rootParam);
        }
        const res = await fetch(requestUrl, { headers });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || ('Download failed (' + res.status + ')'));
        }

        const blob = await res.blob();
        const blobUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const filename = path.split('/').pop() || 'download';
        link.href = blobUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(blobUrl);
      } catch (err) {
        filesStatusEl.textContent = err.message || String(err);
      }
    }

    function renderArtifactRows(entries) {
      artifactRowsEl.innerHTML = '';
      if (!entries || entries.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="5" class="muted">No captured sandbox artifacts yet.</td>';
        artifactRowsEl.appendChild(tr);
        return;
      }

      for (const entry of entries) {
        const tr = document.createElement('tr');
        const fileLabel = escapeHtml(entry.filename || entry.file_id || 'artifact');
        const actionBtn = entry.downloadable === false
          ? '-'
          : '<button data-artifact-download="' + escapeHtml(entry.file_id || '') + '" data-artifact-name="' + fileLabel + '">Download</button>';
        tr.innerHTML =
          '<td class="mono">' + fileLabel + '</td>' +
          '<td>' + fmtBytes(entry.size_bytes) + '</td>' +
          '<td>' + escapeHtml(entry.created_at || '-') + '</td>' +
          '<td>' + escapeHtml(entry.captured_at || '-') + '</td>' +
          '<td>' + actionBtn + '</td>';
        artifactRowsEl.appendChild(tr);
      }
    }

    async function loadArtifacts() {
      artifactStatusEl.textContent = '';
      artifactRowsEl.innerHTML = '';
      try {
        const data = await fetchJson('/v0/api/webapp/artifacts', { headers: authHeaders() });
        renderArtifactRows(data.entries || []);
      } catch (err) {
        artifactStatusEl.textContent = err.message || String(err);
      }
    }

    async function downloadArtifact(fileId, fallbackName) {
      artifactStatusEl.textContent = '';
      try {
        const headers = authHeaders();
        const res = await fetch('/v0/api/webapp/artifacts/download?file_id=' + encodeURIComponent(fileId), { headers });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || ('Artifact download failed (' + res.status + ')'));
        }

        const blob = await res.blob();
        const blobUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const filename = fallbackName || (fileId + '.bin');
        link.href = blobUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(blobUrl);
      } catch (err) {
        artifactStatusEl.textContent = err.message || String(err);
      }
    }

    function setDomaindocsStatus(message, isError = false) {
      domaindocsStatusEl.textContent = message || '';
      domaindocsStatusEl.style.color = isError ? 'var(--danger)' : 'var(--muted)';
    }

    function renderDocOptions(docs, selectedLabel) {
      docSelect.innerHTML = '';
      if (!docs || docs.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No DomainDocs found';
        docSelect.appendChild(opt);
        return;
      }
      for (const doc of docs) {
        const opt = document.createElement('option');
        opt.value = doc.label;
        opt.textContent = (doc.enabled ? '[on] ' : '[off] ') + doc.label;
        if (doc.label === selectedLabel) {
          opt.selected = true;
        }
        docSelect.appendChild(opt);
      }
    }

    function renderSectionOptions(sections, selectedHeader) {
      sectionSelect.innerHTML = '';
      if (!sections || sections.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No sections';
        sectionSelect.appendChild(opt);
        return;
      }
      for (const section of sections) {
        const opt = document.createElement('option');
        opt.value = section.header;
        const collapsedTag = section.collapsed ? ' [collapsed]' : '';
        opt.textContent = section.header + collapsedTag;
        if (section.header === selectedHeader) {
          opt.selected = true;
        }
        sectionSelect.appendChild(opt);
      }
    }

    async function loadDomaindocs(options = {}) {
      setDomaindocsStatus('Loading DomainDocs...');
      const result = await callAction('domain_knowledge', 'list', {});
      const docs = Array.isArray(result.domaindocs) ? result.domaindocs : [];
      docsCache = docs;

      if (docs.length === 0) {
        currentDocLabel = '';
        currentSectionHeader = '';
        renderDocOptions([], '');
        renderSectionOptions([], '');
        sectionContentInput.value = '';
        docMetaEl.textContent = 'No DomainDocs yet. Create one above.';
        setDomaindocsStatus('No DomainDocs found.');
        return;
      }

      const requestedLabel = options.selectedLabel || currentDocLabel;
      const selected = docs.find((d) => d.label === requestedLabel) || docs[0];
      currentDocLabel = selected.label;
      renderDocOptions(docs, currentDocLabel);
      docMetaEl.textContent = (selected.enabled ? 'Enabled' : 'Disabled') + ' | ' + (selected.description || '');
      await loadSections(currentDocLabel, options.selectedSection);
      setDomaindocsStatus('Loaded ' + docs.length + ' DomainDoc(s).');
    }

    async function loadSections(label, selectedSection) {
      if (!label) {
        currentSectionHeader = '';
        renderSectionOptions([], '');
        sectionContentInput.value = '';
        return;
      }
      setDomaindocsStatus('Loading sections for ' + label + '...');
      const result = await callAction('domain_knowledge', 'list_sections', { label });
      const sections = Array.isArray(result.sections) ? result.sections : [];
      if (sections.length === 0) {
        currentSectionHeader = '';
        renderSectionOptions([], '');
        sectionContentInput.value = '';
        setDomaindocsStatus('No sections found. Create one.');
        return;
      }
      const picked = sections.find((s) => s.header === selectedSection) || sections[0];
      currentSectionHeader = picked.header;
      renderSectionOptions(sections, currentSectionHeader);
      await loadSectionContent();
    }

    async function loadSectionContent() {
      if (!currentDocLabel || !currentSectionHeader) {
        sectionContentInput.value = '';
        return;
      }
      const result = await callAction('domain_knowledge', 'get_section', {
        label: currentDocLabel,
        section: currentSectionHeader,
      });
      sectionContentInput.value = result.content || '';
      const collapsedState = result.collapsed ? 'collapsed' : 'expanded';
      setDomaindocsStatus('Editing section "' + currentSectionHeader + '" (' + collapsedState + ').');
    }

    async function createDomaindoc() {
      const label = newDocLabelInput.value.trim();
      const description = newDocDescInput.value.trim();
      if (!label || !description) {
        throw new Error('Both label and description are required.');
      }
      await callAction('domain_knowledge', 'create', { label, description });
      newDocLabelInput.value = '';
      newDocDescInput.value = '';
      await loadDomaindocs({ selectedLabel: label });
      setDomaindocsStatus('Created DomainDoc "' + label + '".');
    }

    async function setDomaindocEnabled(enabled) {
      if (!currentDocLabel) {
        throw new Error('Select a DomainDoc first.');
      }
      const action = enabled ? 'enable' : 'disable';
      await callAction('domain_knowledge', action, { label: currentDocLabel });
      await loadDomaindocs({ selectedLabel: currentDocLabel, selectedSection: currentSectionHeader });
      setDomaindocsStatus((enabled ? 'Enabled ' : 'Disabled ') + '"' + currentDocLabel + '".');
    }

    async function createSection() {
      if (!currentDocLabel) {
        throw new Error('Select a DomainDoc first.');
      }
      const section = newSectionInput.value.trim();
      if (!section) {
        throw new Error('Section header is required.');
      }
      await callAction('domain_knowledge', 'create_section', {
        label: currentDocLabel,
        section,
        content: '',
      });
      newSectionInput.value = '';
      await loadSections(currentDocLabel, section);
      setDomaindocsStatus('Created section "' + section + '".');
    }

    async function setSectionCollapsed(collapsed) {
      if (!currentDocLabel || !currentSectionHeader) {
        throw new Error('Select a section first.');
      }
      const action = collapsed ? 'collapse_section' : 'expand_section';
      await callAction('domain_knowledge', action, {
        label: currentDocLabel,
        section: currentSectionHeader,
      });
      await loadSectionContent();
      setDomaindocsStatus((collapsed ? 'Collapsed ' : 'Expanded ') + '"' + currentSectionHeader + '".');
    }

    async function saveSection() {
      if (!currentDocLabel || !currentSectionHeader) {
        throw new Error('Select a section first.');
      }
      await callAction('domain_knowledge', 'update_section', {
        label: currentDocLabel,
        section: currentSectionHeader,
        content: sectionContentInput.value,
      });
      setDomaindocsStatus('Saved "' + currentSectionHeader + '".');
    }

    saveKeyBtn.addEventListener('click', async () => {
      const key = apiKeyInput.value.trim();
      if (!key) {
        chatStatusEl.textContent = 'API key is required.';
        return;
      }
      localStorage.setItem('mira_api_key', key);
      chatStatusEl.textContent = 'API key saved in browser local storage.';
      await bootstrapChatWithCurrentKey(true);
      await loadFiles(pathInput.value.trim());
      await loadArtifacts();
      await loadDomaindocs();
    });

    chatTab.addEventListener('click', () => setTab('chat'));
    filesTab.addEventListener('click', () => {
      setTab('files');
      loadFiles(pathInput.value.trim());
      loadArtifacts();
    });
    domaindocsTab.addEventListener('click', () => {
      setTab('domaindocs');
      loadDomaindocs().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });

    sendBtn.addEventListener('click', sendChatMessage);
    loadOlderBtn.addEventListener('click', async () => {
      if (!apiKeyInput.value.trim()) {
        chatStatusEl.textContent = 'API key is required.';
        return;
      }
      if (!historyHasMore) {
        chatStatusEl.textContent = 'No older messages available.';
        updateLoadOlderButton();
        return;
      }
      loadOlderBtn.disabled = true;
      loadOlderBtn.textContent = 'Loading...';
      await loadChatHistory({ reset: false });
      updateLoadOlderButton();
    });
    clearBtn.addEventListener('click', () => {
      chatStatusEl.textContent = '';
      chatLogEl.innerHTML = '';
      historyOffset = 0;
      historyHasMore = false;
      historyBootstrapped = false;
      updateLoadOlderButton();
      chatInput.value = '';
      chatInput.focus();
    });

    chatInput.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey) {
        ev.preventDefault();
        sendChatMessage();
      }
    });

    goBtn.addEventListener('click', () => loadFiles(pathInput.value.trim()));
    reloadBtn.addEventListener('click', () => loadFiles(pathInput.value.trim()));
    reloadArtifactsBtn.addEventListener('click', loadArtifacts);
    rootSelect.addEventListener('change', () => loadFiles(''));
    rowsEl.addEventListener('click', (ev) => {
      const target = ev.target;
      if (target.dataset && target.dataset.open !== undefined) {
        loadFiles(target.dataset.open);
      } else if (target.dataset && target.dataset.download !== undefined) {
        downloadFile(target.dataset.download);
      }
    });
    artifactRowsEl.addEventListener('click', (ev) => {
      const target = ev.target;
      if (target.dataset && target.dataset.artifactDownload !== undefined) {
        downloadArtifact(target.dataset.artifactDownload, target.dataset.artifactName || '');
      }
    });

    reloadDocsBtn.addEventListener('click', () => {
      loadDomaindocs().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    docSelect.addEventListener('change', () => {
      currentDocLabel = docSelect.value || '';
      if (!currentDocLabel) {
        renderSectionOptions([], '');
        sectionContentInput.value = '';
        return;
      }
      loadDomaindocs({ selectedLabel: currentDocLabel }).catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    sectionSelect.addEventListener('change', () => {
      currentSectionHeader = sectionSelect.value || '';
      loadSectionContent().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    reloadSectionsBtn.addEventListener('click', () => {
      loadSections(currentDocLabel, currentSectionHeader).catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    createDocBtn.addEventListener('click', () => {
      createDomaindoc().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    enableDocBtn.addEventListener('click', () => {
      setDomaindocEnabled(true).catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    disableDocBtn.addEventListener('click', () => {
      setDomaindocEnabled(false).catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    createSectionBtn.addEventListener('click', () => {
      createSection().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    expandSectionBtn.addEventListener('click', () => {
      setSectionCollapsed(false).catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    collapseSectionBtn.addEventListener('click', () => {
      setSectionCollapsed(true).catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });
    saveSectionBtn.addEventListener('click', () => {
      saveSection().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    });

    if (savedKey) {
      bootstrapChatWithCurrentKey();
      loadFiles('');
      loadArtifacts();
      loadDomaindocs().catch((err) => setDomaindocsStatus(err.message || String(err), true));
    } else {
      appendBubble('assistant', 'Connected to local webapp. Paste API key, save it, then send a message or run /help.');
    }
    updateLoadOlderButton();

    // Handle browser/password-manager autofill that can happen after initial script run.
    setTimeout(() => {
      if (!historyBootstrapped && apiKeyInput.value.trim()) {
        bootstrapChatWithCurrentKey();
        loadFiles('');
        loadArtifacts();
        loadDomaindocs().catch((err) => setDomaindocsStatus(err.message || String(err), true));
      }
    }, 350);
  </script>
</body>
</html>"""


@router.get("/v0/api/webapp/files")
def list_output_files(
    path: str = Query(default="", description="Path relative to OUTPUT_DIR"),
    root: Optional[str] = Query(default=None, description="Output root selector key"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """List directories and files under OUTPUT_DIR."""
    del current_user
    root_key, root_path = _select_output_root(root)
    target = _resolve_relative_path(path, root_path)

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path must be a directory")

    entries: List[Dict[str, Any]] = []
    try:
        iter_entries = list(target.iterdir())
    except OSError as exc:
        raise HTTPException(status_code=403, detail=f"Cannot read directory: {exc}") from exc

    for entry in sorted(iter_entries, key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            stat = entry.stat()
            rel = entry.relative_to(root_path).as_posix()
            is_dir = entry.is_dir()
        except OSError:
            # Skip unreadable entries so one bad node doesn't break the whole listing.
            continue

        entries.append(
            {
                "name": entry.name,
                "path": rel,
                "type": "dir" if is_dir else "file",
                "size_bytes": None if is_dir else stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )

    rel_current = target.relative_to(root_path).as_posix()
    if rel_current == ".":
        rel_current = ""

    parent_path: Optional[str] = None
    if target != root_path:
        parent_rel = target.parent.relative_to(root_path).as_posix()
        parent_path = "" if parent_rel == "." else parent_rel

    roots = _get_output_roots()
    available_roots = [
        {"key": key, "label": f"{key}: {value}"}
        for key, value in roots.items()
    ]

    return {
        "output_root": str(root_path),
        "root_key": root_key,
        "available_roots": available_roots,
        "current_path": rel_current,
        "parent_path": parent_path,
        "entries": entries,
    }


@router.get("/v0/api/webapp/download")
def download_output_file(
    path: str = Query(..., description="File path relative to OUTPUT_DIR"),
    root: Optional[str] = Query(default=None, description="Output root selector key"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    """Download a file from OUTPUT_DIR."""
    del current_user
    _root_key, root_path = _select_output_root(root)
    target = _resolve_relative_path(path, root_path)

    if not target.exists():
        raise HTTPException(status_code=404, detail="File does not exist")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path must be a file")

    return FileResponse(path=str(target), filename=target.name)


@router.get("/v0/api/webapp/artifacts")
def list_code_execution_artifacts(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """List Anthropic code_execution artifacts captured for this continuum."""
    del current_user
    continuum_id = _get_continuum_id_for_current_user()
    cached = _load_cached_artifacts(continuum_id)

    # Show newest first in UI.
    cached = list(reversed(cached))[:100]

    if not cached:
        return {"continuum_id": continuum_id, "entries": []}

    from cns.services.orchestrator import get_orchestrator

    client = get_orchestrator().llm_provider.anthropic_client
    entries: List[Dict[str, Any]] = []

    for item in cached:
        file_id = item["file_id"]
        entry: Dict[str, Any] = {
            "file_id": file_id,
            "captured_at": item.get("captured_at", ""),
            "filename": file_id,
            "mime_type": "application/octet-stream",
            "size_bytes": None,
            "created_at": None,
            "downloadable": True,
        }

        try:
            metadata = client.beta.files.retrieve_metadata(file_id=file_id)
            entry["filename"] = getattr(metadata, "filename", file_id)
            entry["mime_type"] = getattr(metadata, "mime_type", "application/octet-stream")
            entry["size_bytes"] = getattr(metadata, "size_bytes", None)
            created_at = getattr(metadata, "created_at", None)
            entry["created_at"] = created_at.isoformat() if created_at else None
            entry["downloadable"] = getattr(metadata, "downloadable", True)
        except Exception as exc:
            entry["downloadable"] = False
            entry["error"] = str(exc)

        entries.append(entry)

    return {"continuum_id": continuum_id, "entries": entries}


@router.get("/v0/api/webapp/artifacts/download")
def download_code_execution_artifact(
    file_id: str = Query(..., description="Anthropic file_id from code_execution output"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Download an Anthropic code_execution artifact by file ID."""
    del current_user
    from cns.services.orchestrator import get_orchestrator

    client = get_orchestrator().llm_provider.anthropic_client

    try:
        metadata = client.beta.files.retrieve_metadata(file_id=file_id)
        binary = client.beta.files.download(file_id=file_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Artifact download failed: {exc}") from exc

    filename = Path(getattr(metadata, "filename", file_id) or file_id).name
    media_type = getattr(metadata, "mime_type", "application/octet-stream")

    try:
        payload = binary.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed reading artifact payload: {exc}") from exc

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type=media_type, headers=headers)
