/**
 * epidemic.js — Standalone Epidemic / Layer 4 tab module.
 *
 * Loaded as a separate <script type="module"> from dist/index.html so it
 * works without touching the pre-built Vite bundle.
 *
 * Features:
 *  - Nav integration: shows panel on "epidemic" tab click, hides on others
 *  - SVG network topology graph (agent nodes + infection edges, animated)
 *  - Layer 4 drift signal bars
 *  - SEIR epidemic parameter readout from simulation data
 *  - Live MoltBook threat feed (security.jsonl events)
 *  - Simulation launcher (deterministic + LLM/Ollama)
 *  - 5-second auto-refresh while active
 */

/* ══════════════════════════════════════════════
   Constants
   ══════════════════════════════════════════════ */
const AUTH_KEY = "siem_auth_token";

const STATE_COLORS = {
  // sancta_epidemic.AgentHealthState values
  susceptible:  "#39ff14",
  exposed:      "#ffe600",
  infected:     "#ff6b35",
  compromised:  "#ff2d55",
  recovered:    "#00e5ff",
  // infection_sim.InfectionState values
  CLEAN:        "#39ff14",
  EXPOSED:      "#ffe600",
  INFECTED:     "#ff6b35",
  SPREADING:    "#ff9500",
  DORMANT:      "#b0ff60",
  RECOVERED:    "#00e5ff",
  unknown:      "#444444",
};

const SIGNAL_LABELS = {
  belief_decay_rate:        "Belief Decay",
  soul_alignment:           "Soul Alignment",
  topic_drift:              "Topic Drift",
  strategy_entropy:         "Strategy Entropy",
  dissonance_trend:         "Dissonance Trend",
  engagement_pattern_delta: "Engagement Delta",
};

/* ══════════════════════════════════════════════
   Auth helpers — reads same key as the bundle
   ══════════════════════════════════════════════ */
function _tok() { return localStorage.getItem(AUTH_KEY) || ""; }
function _authHdrs() { const t = _tok(); return t ? { Authorization: `Bearer ${t}` } : {}; }
async function _efetch(url, opts = {}) {
  return fetch(url, { ...opts, headers: { ..._authHdrs(), ...(opts.headers || {}) } });
}

/* ══════════════════════════════════════════════
   HTML escape (self-contained, no bundle dep)
   ══════════════════════════════════════════════ */
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* ══════════════════════════════════════════════
   Module state
   ══════════════════════════════════════════════ */
let _active = false;
let _refreshTimer = null;

/* ══════════════════════════════════════════════
   Nav integration
   Intercept clicks AFTER the bundle's setNav() runs
   (setTimeout(0) defers to next task in event queue)
   ══════════════════════════════════════════════ */
function _showPanel() {
  const p = document.getElementById("panel-epidemic");
  if (p) p.style.display = "flex";
  _active = true;
  _loadAll();
}

function _hidePanel() {
  const p = document.getElementById("panel-epidemic");
  if (p) p.style.display = "none";
  _active = false;
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
}

function _initNav() {
  const epiTab = document.querySelector('[data-nav="epidemic"]');
  const others  = document.querySelectorAll('.nav-tab:not([data-nav="epidemic"])');
  epiTab?.addEventListener("click", () => setTimeout(_showPanel, 0));
  others.forEach(t => t.addEventListener("click", _hidePanel));
  document.getElementById("epidemic-refresh")?.addEventListener("click", _loadAll);
  document.getElementById("sim-run-det")?.addEventListener("click", () => _runSim("deterministic"));
  document.getElementById("sim-run-llm")?.addEventListener("click", () => _runSim("llm"));
}

/* ══════════════════════════════════════════════
   Data loading — parallel API calls
   ══════════════════════════════════════════════ */
async function _loadAll() {
  try {
    const [sr, mr, lr] = await Promise.all([
      _efetch("/api/epidemic/status"),
      _efetch("/api/epidemic/simulation"),
      _efetch("/api/live-events"),
    ]);
    const status = await sr.json().catch(() => ({}));
    const sim    = await mr.json().catch(() => ({}));
    const live   = await lr.json().catch(() => ({}));

    _renderMetrics(status);
    _renderSignals(status.signals || {});
    _renderParams(sim.available ? sim.data : null);
    _renderNetwork(sim.available ? sim.data : null);
    _renderThreats(live.events || []);
  } catch (e) {
    const el = document.getElementById("epidemic-metrics");
    if (el) el.innerHTML = `<span style="color:#ff2d55;font-size:9px;">Load error: ${esc(String(e))}</span>`;
  }

  if (_refreshTimer) clearInterval(_refreshTimer);
  _refreshTimer = setInterval(() => { if (_active) _loadAll(); }, 5000);
}

/* ══════════════════════════════════════════════
   Metrics row
   ══════════════════════════════════════════════ */
const _alertCols = { clear: "#39ff14", watch: "#ffe600", warn: "#ff6b35", critical: "#ff2d55" };

function _renderMetrics(status) {
  const el = document.getElementById("epidemic-metrics");
  if (!el) return;
  const alert  = status.alert_level || "clear";
  const score  = typeof status.score === "number" ? (status.score * 100).toFixed(1) + "%" : "—";
  const health = (status.seir?.health_state) || "susceptible";
  const ac     = _alertCols[alert] || "#aaa";
  const hc     = STATE_COLORS[health] || "#aaa";

  const card = (label, val, color, extra = "") => `
    <div style="padding:6px 10px;border:1px solid ${color}33;background:#0d0d0d;flex:1;min-width:80px;">
      <div style="font-size:7px;color:#39ff1455;text-transform:uppercase;letter-spacing:.15em;">${label}</div>
      <div style="font-size:16px;font-weight:700;color:${color};">${val}</div>
      ${extra}
    </div>`;

  el.innerHTML =
    card("Alert Level", alert.toUpperCase(), ac) +
    card("Drift Score", score, ac) +
    card("Health State", health.toUpperCase(), hc) +
    (status.seir?.is_epidemic
      ? card("EPIDEMIC", "R0 &gt; 1.0", "#ff2d55",
             '<div style="font-size:7px;color:#ff2d5599;animation:ep-pulse 1s infinite;">ACTIVE</div>')
      : "") +
    (status.seir?.incubation_active
      ? card("Incubation", `${status.seir.incubation_duration ?? "?"} cyc`, "#ffe600")
      : "");
}

/* ══════════════════════════════════════════════
   Layer 4 signal bars
   ══════════════════════════════════════════════ */
function _sigColor(v) {
  return v > 0.65 ? "#ff2d55" : v > 0.45 ? "#ff6b35" : v > 0.25 ? "#ffe600" : "#39ff14";
}

function _renderSignals(signals) {
  const el = document.getElementById("epidemic-signals");
  if (!el) return;
  el.innerHTML = Object.entries(SIGNAL_LABELS).map(([k, label]) => {
    const v   = typeof signals[k] === "number" ? signals[k] : null;
    const pct = v !== null ? Math.round(v * 100) : null;
    const c   = v !== null ? _sigColor(v) : "#444";
    return `<div>
      <div style="display:flex;justify-content:space-between;font-size:8px;font-family:'JetBrains Mono',monospace;">
        <span style="color:#39ff1455;">${label}</span>
        <span style="color:${c};font-weight:600;">${pct !== null ? pct + "%" : "—"}</span>
      </div>
      <div style="height:3px;background:#111;margin-top:2px;overflow:hidden;border-radius:2px;">
        <div style="width:${pct || 0}%;height:100%;background:${c};transition:width 0.4s;"></div>
      </div>
    </div>`;
  }).join("");
}

/* ══════════════════════════════════════════════
   Epidemic parameters (from simulation output)
   ══════════════════════════════════════════════ */
function _renderParams(simData) {
  const el = document.getElementById("epidemic-params");
  if (!el) return;
  if (!simData) {
    el.innerHTML = `<em style="color:#39ff1440;font-size:8px;">Run a simulation to compute parameters</em>`;
    return;
  }
  const p   = simData.epidemic_params || simData.final_stats || simData.summary || {};
  const fmt = (v, d = 3) => typeof v === "number" ? v.toFixed(d) : String(v ?? "—");
  const rows = [
    ["R0 (repro num)",   p.R0,              2],
    ["sigma (incub)",    p.sigma,           3],
    ["gamma (recov)",    p.gamma,           3],
    ["beta (trans)",     p.beta,            3],
    ["agents",           p.total_agents,    0],
    ["peak infected",    p.peak_infected,   0],
    ["final compromised",p.final_infected,  0],
  ].filter(([, v]) => v !== undefined && v !== null);

  el.innerHTML = rows.length
    ? rows.map(([l, v, d]) =>
        `<div style="display:flex;justify-content:space-between;font-size:9px;">
           <span style="color:#39ff1455;">${esc(l)}</span>
           <span style="color:#39ff14;">${esc(fmt(v, d))}</span>
         </div>`).join("")
    : `<em style="color:#39ff1440;font-size:8px;">No parameters in simulation output</em>`;
}

/* ══════════════════════════════════════════════
   SVG Network Topology Graph
   Circular layout, colored nodes, animated edges
   ══════════════════════════════════════════════ */
function _renderNetwork(simData) {
  const svg = document.getElementById("epidemic-network-svg");
  if (!svg) return;

  const agents      = simData?.agents || simData?.final_agents || [];
  const events      = Array.isArray(simData?.events) ? simData.events
                    : Array.isArray(simData?.stats_history) ? simData.stats_history : [];
  const connections = simData?.connections || simData?.topology || [];

  if (!agents.length) {
    svg.innerHTML = `
      <text x="50%" y="45%" text-anchor="middle" dominant-baseline="middle"
            fill="#39ff1440" font-family="JetBrains Mono,monospace" font-size="11">
        No simulation data
      </text>
      <text x="50%" y="55%" text-anchor="middle" dominant-baseline="middle"
            fill="#39ff1425" font-family="JetBrains Mono,monospace" font-size="9">
        Run a simulation to see the agent network
      </text>`;
    return;
  }

  const W  = Math.max(svg.clientWidth  || 400, 260);
  const H  = Math.max(svg.clientHeight || 280, 200);
  const cx = W / 2, cy = H / 2;
  const R  = Math.min(cx - 30, cy - 30) * 0.82;
  const nr = Math.max(7, Math.min(14, Math.round(140 / agents.length)));

  // Circular layout
  const nodes = agents.map((a, i) => {
    const angle = (2 * Math.PI * i / agents.length) - Math.PI / 2;
    const rawState = (a.infection_state || a.state || a.status || "unknown").toString();
    return {
      id:    String(a.id || a.agent_id || i),
      state: rawState,
      role:  String(a.role || "?").slice(0, 4).toUpperCase(),
      x:     cx + R * Math.cos(angle),
      y:     cy + R * Math.sin(angle),
    };
  });
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  // Build edge set from connections or inferred from events
  const edgeSet = new Set();
  if (connections.length) {
    for (const c of connections) {
      const a = String(c.from || c.source || "");
      const b = String(c.to   || c.target || "");
      if (a && b && nodeMap[a] && nodeMap[b] && a !== b) edgeSet.add(`${a}|||${b}`);
    }
  } else {
    for (const ev of events.slice(-200)) {
      const a = String(ev.sender      || ev.source_id || ev.from || "");
      const b = String(ev.recipient   || ev.target_id || ev.to   || "");
      if (a && b && a !== b && nodeMap[a] && nodeMap[b]) edgeSet.add(`${a}|||${b}`);
    }
  }

  // Active edges = those appearing in last 15 events
  const activeSet = new Set();
  for (const ev of events.slice(-15)) {
    const a = String(ev.sender    || ev.source_id || "");
    const b = String(ev.recipient || ev.target_id || "");
    if (a && b) activeSet.add(`${a}|||${b}`);
  }

  // SVG defs: glow filters + keyframe animation
  const defs = `<defs>
    <style>
      @keyframes ep-pulse { 0%,100%{opacity:.2} 50%{opacity:.9} }
      .ep-active-edge { animation: ep-pulse 0.7s ease-in-out infinite; }
      .ep-infected    { animation: ep-pulse 1.1s ease-in-out infinite; }
    </style>
    <filter id="glow-g" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow-r" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;

  // Draw edges
  const edgeSVG = [...edgeSet].map(e => {
    const [aid, bid] = e.split("|||");
    const na = nodeMap[aid], nb = nodeMap[bid];
    if (!na || !nb) return "";
    const isActive = activeSet.has(e);
    return `<line x1="${na.x.toFixed(1)}" y1="${na.y.toFixed(1)}"
                  x2="${nb.x.toFixed(1)}" y2="${nb.y.toFixed(1)}"
                  stroke="${isActive ? "#ff6b35" : "#39ff1418"}"
                  stroke-width="${isActive ? "1.5" : "0.7"}"
                  class="${isActive ? "ep-active-edge" : ""}"/>`;
  }).join("");

  // Draw nodes
  const nodeSVG = nodes.map(n => {
    const normalState = n.state.toLowerCase();
    const color  = STATE_COLORS[n.state] || STATE_COLORS[normalState] || STATE_COLORS.unknown;
    const isDanger = ["infected","compromised","INFECTED","SPREADING","COMPROMISED"].includes(n.state);
    const filter = isDanger ? 'url(#glow-r)' : 'url(#glow-g)';
    const label  = n.id.length > 7 ? n.id.slice(0, 7) : n.id;
    return `<g transform="translate(${n.x.toFixed(1)},${n.y.toFixed(1)})">
      ${isDanger ? `<circle r="${nr + 5}" fill="${color}15" class="ep-infected"/>` : ""}
      <circle r="${nr}" fill="${color}22" stroke="${color}" stroke-width="1.5" filter="${filter}"/>
      <text y="0.5" text-anchor="middle" dominant-baseline="middle"
            fill="${color}" font-family="JetBrains Mono,monospace"
            font-size="${Math.max(5.5, nr * 0.58)}">${esc(label)}</text>
      <text y="${nr + 8}" text-anchor="middle" dominant-baseline="middle"
            fill="${color}88" font-family="JetBrains Mono,monospace" font-size="6">${esc(n.role)}</text>
    </g>`;
  }).join("");

  // State legend (bottom row)
  const stateCount = {};
  nodes.forEach(n => { stateCount[n.state] = (stateCount[n.state] || 0) + 1; });
  const legendSVG = Object.entries(stateCount).map(([s, cnt], i) => {
    const c = STATE_COLORS[s] || STATE_COLORS[s.toLowerCase()] || "#444";
    return `<g transform="translate(${8 + i * 72}, ${H - 14})">
      <circle r="4" fill="${c}44" stroke="${c}" stroke-width="1"/>
      <text x="8" y="1" dominant-baseline="middle" fill="${c}"
            font-family="JetBrains Mono,monospace" font-size="7">${esc(s.slice(0,8))} ${cnt}</text>
    </g>`;
  }).join("");

  svg.innerHTML = defs + edgeSVG + nodeSVG + legendSVG;
}

/* ══════════════════════════════════════════════
   Live MoltBook Threats
   Reads from /api/live-events, shows security events
   Real threats = magenta; LLM deep scan = orange; sim = green
   ══════════════════════════════════════════════ */
function _renderThreats(events) {
  const el    = document.getElementById("epidemic-threats");
  const badge = document.getElementById("epidemic-threat-count");
  if (!el) return;

  const THREAT_EVENTS = new Set([
    "injection_defense", "tavern_defense", "suspicious_block", "llm_deep_scan",
    "injection_blocked", "ioc_detected",
  ]);

  const threats = events
    .filter(ev => THREAT_EVENTS.has(ev.event) || ev.source === "security")
    .slice(-30)
    .reverse();

  if (badge) {
    badge.textContent = threats.length ? `${threats.length} threat${threats.length !== 1 ? "s" : ""}` : "";
    badge.style.color = threats.length ? "#ff2d55" : "";
  }

  if (!threats.length) {
    el.innerHTML = `<em style="color:#39ff1440;">No live threats — MoltBook feed appears clean</em>`;
    return;
  }

  el.innerHTML = threats.map(ev => {
    const data    = typeof ev.data === "object" ? ev.data : ev;
    const ts      = (ev.ts || ev.timestamp || data.ts || "").toString().slice(11, 19);
    const author  = esc(String(data.author || ev.source || "unknown"));
    const evType  = ev.event || "event";
    const isLLM   = evType === "llm_deep_scan";
    const isBlock = evType === "injection_defense" || evType === "tavern_defense" || evType === "injection_blocked";
    const color   = isLLM ? "#ff9500" : isBlock ? "#ff2d55" : "#ffe600";
    const label   = isLLM ? "LLM-SCAN" : isBlock ? "BLOCKED" : "WATCH";
    const preview = esc(String(data.preview || data.response_preview || data.reason || "").slice(0, 70));
    const conf    = isLLM && data.confidence != null ? ` conf=${(data.confidence * 100).toFixed(0)}%` : "";
    return `<div style="border-bottom:1px solid #39ff1410;padding:2px 0;display:flex;gap:6px;align-items:baseline;flex-wrap:nowrap;overflow:hidden;">
      <span style="color:#39ff1435;white-space:nowrap;font-size:8px;">${esc(ts)}</span>
      <span style="color:${color};font-weight:700;white-space:nowrap;font-size:8px;">[${label}${conf}]</span>
      <span style="color:#00e5ff;white-space:nowrap;">${author}</span>
      <span style="color:#39ff1455;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${preview}</span>
    </div>`;
  }).join("");
}

/* ══════════════════════════════════════════════
   Simulation runner
   ══════════════════════════════════════════════ */
async function _runSim(type) {
  const statusEl = document.getElementById("sim-run-status");
  if (statusEl) statusEl.textContent = `Launching ${type} sim...`;
  try {
    const res  = await _efetch("/api/epidemic/run", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ type }),
    });
    const data = await res.json().catch(() => ({}));
    if (data.ok) {
      if (statusEl) statusEl.textContent = `PID ${data.pid} running — ${data.script}`;
      setTimeout(_loadAll, 4000);
    } else {
      if (statusEl) statusEl.textContent = `Error: ${data.error || "script not found"}`;
    }
  } catch (e) {
    if (statusEl) statusEl.textContent = `Failed: ${String(e)}`;
  }
}

/* ══════════════════════════════════════════════
   Boot: wire up nav after DOM is ready
   ══════════════════════════════════════════════ */
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _initNav);
} else {
  _initNav();
}
