import "./tailwind.source.css";
import { Chart, registerables } from "chart.js";
import { animate } from "motion";

Chart.register(...registerables);

/* ══════════════════════════════════════════════
   Auth — Bearer token when SIEM_AUTH_TOKEN is set
   ══════════════════════════════════════════════ */
const AUTH_STORAGE_KEY = "siem_auth_token";

function getAuthToken() {
  return sessionStorage.getItem(AUTH_STORAGE_KEY);
}

function setAuthToken(t) {
  if (t) sessionStorage.setItem(AUTH_STORAGE_KEY, t);
  else sessionStorage.removeItem(AUTH_STORAGE_KEY);
}

function authHeaders() {
  const t = getAuthToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

const FETCH_TIMEOUT_MS = 8000;
const CHAT_FETCH_TIMEOUT_MS = 180000;  // 3 min — LLM inference (Ollama) can take 60–90+ sec for large models

async function authFetchWithRetry(url, opts = {}, maxRetries = 3) {
  const headers = { ...opts.headers, ...authHeaders() };
  let lastErr;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
      const res = await fetch(url, { ...opts, headers, signal: ctrl.signal });
      clearTimeout(to);
      if (res.status === 401 && getAuthToken()) {
        setAuthToken(null);
        showAuthOverlay();
        if (wsLive) {
          wsLive.close();
          wsLive = null;
        }
      }
      return res;
    } catch (e) {
      clearTimeout(to);
      lastErr = e;
      if (attempt < maxRetries - 1) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

async function authFetch(url, opts = {}) {
  const headers = { ...opts.headers, ...authHeaders() };
  const timeoutMs = opts.timeoutMs ?? FETCH_TIMEOUT_MS;
  const { timeoutMs: _omit, ...restOpts } = opts;
  const ctrl = new AbortController();
  const to = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...restOpts, headers, signal: ctrl.signal });
    clearTimeout(to);
    if (res.status === 401 && getAuthToken()) {
      setAuthToken(null);
      showAuthOverlay();
      if (wsLive) {
        wsLive.close();
        wsLive = null;
      }
    }
    return res;
  } catch (e) {
    clearTimeout(to);
    throw e;
  }
}

/* ══════════════════════════════════════════════
   Audio Manager — browser-side sound playback
   ══════════════════════════════════════════════ */
const AUDIO_STORAGE_KEY = "siem_audio_enabled";
let audioEnabled = (() => {
  try {
    const stored = sessionStorage.getItem(AUDIO_STORAGE_KEY);
    if (stored !== null) return stored === "true";
  } catch (_) {}
  return true;
})();
let soundManifest = null;
let lastSoundTime = 0;
const SOUND_COOLDOWN_MS = 800;

async function initAudio() {
  try {
    const res = await authFetchWithRetry("/api/sounds/manifest");
    const data = await res.json();
    if (data.ok && data.manifest) {
      soundManifest = data.manifest;
    }
  } catch (_) {}
}

function mapEventToCategory(ev) {
  const src = ev.source || "";
  const name = ev.event || "";
  if (src === "security" && /inject|block|reject|suspicious/.test(name)) return "security.alert";
  if (src === "redteam" && /escalat|reward/.test(name)) return "security.alert";
  if (/published|complete|cycle_done/.test(name)) return "task.complete";
  if (/mood_shift|mood_change|epistemic_state/.test(name) && ev.mood) return "mood.change";
  return null;
}

function playEventSound(ev) {
  if (!audioEnabled || !soundManifest) return;
  const now = Date.now();
  if (now - lastSoundTime < SOUND_COOLDOWN_MS) return;

  const category = mapEventToCategory(ev);
  if (!category) return;

  const packName = soundManifest.default_pack || "office_peon";
  const pack = (soundManifest.packs || {})[packName];
  if (!pack) return;

  const sounds = pack[category];
  if (!sounds || !sounds.length) return;

  const file = sounds[Math.floor(Math.random() * sounds.length)];
  const audio = new Audio("/sounds/" + file);
  audio.volume = 0.55;
  audio.play().catch(() => {});
  lastSoundTime = now;
}

function playSessionSound() {
  if (!audioEnabled || !soundManifest) return;
  const packName = soundManifest.default_pack || "office_peon";
  const pack = (soundManifest.packs || {})[packName];
  if (!pack) return;
  const sounds = pack["session.start"];
  if (!sounds || !sounds.length) return;
  const file = sounds[Math.floor(Math.random() * sounds.length)];
  const audio = new Audio("/sounds/" + file);
  audio.volume = 0.55;
  audio.play().catch(() => {});
}

/* ══════════════════════════════════════════════
   DOM References
   ══════════════════════════════════════════════ */
const panes = {
  feed: document.getElementById("pane-feed"),
  alerts: document.getElementById("pane-alerts"),
  system: document.getElementById("pane-system"),
  activity: document.getElementById("pane-activity"),
  learn: document.getElementById("learn-content"),
  defense: document.getElementById("defense-content"),
};

const panelLearn = document.getElementById("panel-learn");
const panelDefense = document.getElementById("panel-defense");
const feedSection = document.querySelector(".panel.feed");

const els = {
  agent: document.getElementById("m-agent"),
  pid: document.getElementById("m-pid"),
  mood: document.getElementById("m-mood"),
  inj: document.getElementById("m-inj"),
  sanitized: document.getElementById("m-sanitized"),
  reward: document.getElementById("m-reward"),
  fp: document.getElementById("m-fp"),
  belief: document.getElementById("m-belief"),
};

const modeSelect = document.getElementById("mode");
const scrollLock = document.getElementById("scrollLock");
const navTabs = Array.from(document.querySelectorAll(".nav-tab"));
const eventFilters = Array.from(document.querySelectorAll(".event-filter"));
const feedToggle = document.getElementById("feed-toggle");
const feedStatusIcon = document.getElementById("feed-status-icon");
const feedStatusLabel = document.getElementById("feed-status-label");
const timeFrom = document.getElementById("time-from");
const timeTo = document.getElementById("time-to");
const timeApply = document.getElementById("time-apply");
const timeClear = document.getElementById("time-clear");

const injEls = {
  hidden: document.getElementById("inj-hidden"),
  unicode: document.getElementById("inj-unicode"),
  patterns: document.getElementById("inj-patterns"),
  rate: document.getElementById("inj-rate"),
};

const epiEls = {
  conf: document.getElementById("epi-conf"),
  ent: document.getElementById("epi-ent"),
  anth: document.getElementById("epi-anth"),
  over: document.getElementById("epi-over"),
};

const footerEls = {
  avgScore: document.getElementById("f-avg-score"),
  totalDecisions: document.getElementById("f-total-decisions"),
  currentMood: document.getElementById("f-current-mood"),
  conf: document.getElementById("f-conf"),
  ent: document.getElementById("f-ent"),
};

let activeNav = "dashboard";
let activeTab = "all";
let feedPaused = false;
let wsLive = null;
let bufferedEvents = [];
const MAX_BUFFER = 1200;
const renderedEventIds = new Set();
const MAX_RENDERED_IDS = 500;

const counters = {
  totalEvents: 0,
  injectionAttempts: 0,
  sanitizations: 0,
  unicodeCleanEvents: 0,
  hiddenCharsStripped: 0,
  patternMatches: 0,
  epistemicN: 0,
  sumConf: 0,
  sumEntropy: 0,
  sumAnth: 0,
};

let lastStatus = {
  running: null,
  suspended: null,
  pid: null,
};

let lastActivitySnapshot = "";

/* ══════════════════════════════════════════════
   Helpers
   ══════════════════════════════════════════════ */
function esc(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function levelClass(level) {
  const l = (level || "").toUpperCase();
  if (l === "INFO") return "lvl-info";
  if (l === "DEBUG") return "lvl-debug";
  if (l === "WARN" || l === "WARNING") return "lvl-warn";
  if (l === "ERROR") return "lvl-error";
  if (l === "ALERT" || l === "CRITICAL") return "lvl-alert";
  return "lvl-info";
}

function appendLine(pane, htmlLine) {
  pane.insertAdjacentHTML("afterbegin", htmlLine + "\n");
  if (scrollLock.checked) {
    pane.scrollTop = 0;
  }
}

/* ══════════════════════════════════════════════
   Nav & Event Filters
   ══════════════════════════════════════════════ */
const panelInteract = document.getElementById("panel-interact");
const panelLab = document.getElementById("panel-lab");
const panelEpidemic = document.getElementById("panel-epidemic");
const chatPanel = document.getElementById("chat-panel");
const chatPanelSlot = document.getElementById("chat-panel-slot");
const labSystemSlot = document.getElementById("lab-system-slot");
const systemSectionWrapper = document.getElementById("system-section-wrapper");
const knowledgeBasePanel = document.getElementById("knowledge-base-panel");
const labBottom = document.querySelector(".lab-bottom");

function setNav(nav) {
  activeNav = nav;
  for (const b of navTabs) b.classList.toggle("is-active", b.dataset.nav === nav);
  const isDashboard = nav === "dashboard";
  const isGovernance = nav === "governance";
  const isMemory = nav === "memory";
  const isInteract = nav === "interact";
  const isLab = nav === "lab";
  const isEpidemic = nav === "epidemic";

  if (feedSection) feedSection.style.display = isDashboard ? "flex" : "none";
  if (panelLearn) panelLearn.style.display = isMemory ? "flex" : "none";
  if (panelDefense) panelDefense.style.display = isGovernance ? "flex" : "none";
  if (panelInteract) panelInteract.style.display = isInteract ? "flex" : "none";
  if (panelLab) panelLab.style.display = isLab ? "flex" : "none";
  if (panelEpidemic) panelEpidemic.style.display = isEpidemic ? "flex" : "none";

  if (isDashboard && incidentsChart) setTimeout(() => incidentsChart.resize(), 50);
  if (isMemory) loadLearningHealth();
  else if (isGovernance) loadAdversaryDefense();
  else if (isEpidemic) loadEpidemicData();
  else clearLearnDefenseRefresh();

  if (chatPanel && chatPanelSlot && labBottom) {
    if (isInteract) {
      chatPanelSlot.appendChild(chatPanel);
    } else if (chatPanel.parentElement === chatPanelSlot) {
      labBottom.insertBefore(chatPanel, labBottom.children[1]);
    }
  }

  if (systemSectionWrapper && labSystemSlot && knowledgeBasePanel) {
    if (isLab) {
      labSystemSlot.appendChild(systemSectionWrapper);
    } else if (systemSectionWrapper.parentElement === labSystemSlot) {
      knowledgeBasePanel.appendChild(systemSectionWrapper);
    }
  }
  appendLine(panes.system, `<span class="lvl-debug">[SYS] nav=${esc(nav)}</span>`);
}

function setEventFilter(tab) {
  activeTab = tab;
  for (const b of eventFilters) b.classList.toggle("is-active", b.dataset.tab === tab);
  panes.feed.textContent = "";
  for (let i = bufferedEvents.length - 1; i >= 0; i--) {
    const ev = bufferedEvents[i];
    if (!matchesActiveTab(ev)) continue;
    const lvl = ev.level || "INFO";
    const cls = levelClass(lvl);
    const { tag, text } = formatEventLine(ev);
    const line = `<span class="${cls} dim">${tag} ${esc(text)}</span>`;
    panes.feed.insertAdjacentHTML("beforeend", line + "\n");
  }
  panes.feed.scrollTop = 0;
  appendLine(panes.system, `<span class="lvl-debug">[SYS] filter=${esc(tab)}</span>`);
}

for (const b of navTabs) {
  b.addEventListener("click", () => setNav(b.dataset.nav));
}
for (const b of eventFilters) {
  b.addEventListener("click", () => setEventFilter(b.dataset.tab));
}

window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) return;
  const navMap = { "1": "dashboard", "2": "governance", "3": "memory", "4": "interact", "5": "lab" };
  const filterMap = { "6": "all", "7": "security", "8": "redteam", "9": "analyst", "0": "alerts" };
  if (navMap[e.key]) setNav(navMap[e.key]);
  else if (filterMap[e.key] && activeNav === "dashboard") setEventFilter(filterMap[e.key]);
});

/* ══════════════════════════════════════════════
   LEARN / DEFENSE Tab Data Loading
   ══════════════════════════════════════════════ */
let learnRefreshTimer = null;
let defenseRefreshTimer = null;

async function loadLearningHealth() {
  if (!panes.learn) return;
  try {
    const res = await authFetch("/api/learning/health");
    const data = await res.json().catch(() => ({}));
    if (!data.ok) {
      panes.learn.innerHTML = `<div class="alert-box warning">Failed to load: ${esc(data.error || "Unknown error")}</div>`;
      return;
    }
    const hitRate = (data.pattern_hit_rate || 0) * 100;
    const html = `
      <div class="metric-cards-row">
        <div class="metric-card"><span class="metric-label">Patterns</span><div class="metric-value">${data.pattern_count ?? 0}</div></div>
        <div class="metric-card"><span class="metric-label">Hit Rate</span><div class="metric-value">${hitRate.toFixed(1)}%</div></div>
        <div class="metric-card"><span class="metric-label">Positive Feedback</span><div class="metric-value">${data.positive_feedback_pct ?? 0}%</div></div>
        <div class="metric-card"><span class="metric-label">Interactions</span><div class="metric-value">${data.interaction_count ?? 0}</div></div>
      </div>
      <div class="metric-card">
        <span class="metric-label">Top Patterns by Success Rate</span>
        <div style="padding-top:8px;">
          ${(data.top_patterns || []).length ? (data.top_patterns || []).slice(0, 5).map(p =>
            `<div class="recent-item">${esc(p.topic)} — ${(p.success_rate * 100).toFixed(0)}% (n=${p.frequency})</div>`
          ).join("") : "<em>No patterns yet</em>"}
        </div>
      </div>
      <div class="metric-card">
        <span class="metric-label">Recent Interactions</span>
        <div style="padding-top:6px;">
          ${(data.recent_interactions || []).slice(0, 10).map(r => {
            const fbLabel = r.feedback === 1 ? "+" : r.feedback === -1 ? "-" : "?";
            return `<div class="recent-item">${esc((r.ts || "").slice(0, 19))} | ${fbLabel} | ${esc(r.user_preview || "")}</div>`;
          }).join("") || "<em>No recent interactions</em>"}
        </div>
      </div>
    `;
    panes.learn.innerHTML = html;
  } catch (e) {
    panes.learn.innerHTML = `<div class="alert-box warning">Error: ${esc(String(e))}</div>`;
  }
  if (learnRefreshTimer) clearInterval(learnRefreshTimer);
  learnRefreshTimer = setInterval(loadLearningHealth, 5000);
}

async function loadAdversaryDefense() {
  if (!panes.defense) return;
  try {
    const res = await authFetch("/api/security/adversary");
    const data = await res.json().catch(() => ({}));
    if (!data.ok) {
      panes.defense.innerHTML = `<div class="alert-box warning">Failed to load: ${esc(data.error || "Unknown error")}</div>`;
      return;
    }
    const threat = data.threat_level || "green";
    const stats = data.defense_stats || {};
    const html = `
      <div class="metric-cards-row">
        <div class="metric-card">
          <span class="metric-label">Threat Level</span>
          <div class="metric-value"><span class="threat-level ${esc(threat)}">${esc(threat)}</span></div>
        </div>
        <div class="metric-card"><span class="metric-label">Attacks Detected</span><div class="metric-value">${data.total_attacks ?? 0}</div></div>
        <div class="metric-card"><span class="metric-label">Unique Fingerprints</span><div class="metric-value">${data.unique_fingerprints ?? 0}</div></div>
        <div class="metric-card"><span class="metric-label">High-Risk</span><div class="metric-value">${data.high_risk_count ?? 0}</div></div>
      </div>
      <div class="metric-card">
        <span class="metric-label">Defense Stats</span>
        <div style="padding-top:6px;">
          Blocked: ${stats.blocked ?? 0} | IOC: ${stats.ioc_detected ?? 0} | Sanitized: ${stats.unicode_sanitized ?? 0} | Normal: ${stats.normal ?? 0}
        </div>
      </div>
      <div class="metric-card">
        <span class="metric-label">Known Attackers</span>
        <div style="padding-top:6px;">
          ${(data.known_attackers || []).slice(0, 8).map(a =>
            `<div class="recent-item">${esc(a.author || "?")} — ${a.count} attempts</div>`
          ).join("") || "<em>None identified</em>"}
        </div>
      </div>
      <div class="metric-card">
        <span class="metric-label">Recent Attacks</span>
        <div style="padding-top:6px;">
          ${(data.recent_attacks || []).slice(0, 10).map(r =>
            `<div class="recent-item">${esc((r.ts || "").slice(0, 19))} | ${esc(r.action || "")} | ${esc((r.preview || "").slice(0, 50))}</div>`
          ).join("") || "<em>No recent attacks</em>"}
        </div>
      </div>
    `;
    panes.defense.innerHTML = html;
  } catch (e) {
    panes.defense.innerHTML = `<div class="alert-box warning">Error: ${esc(String(e))}</div>`;
  }
  if (defenseRefreshTimer) clearInterval(defenseRefreshTimer);
  defenseRefreshTimer = setInterval(loadAdversaryDefense, 5000);
}

function clearLearnDefenseRefresh() {
  if (learnRefreshTimer) { clearInterval(learnRefreshTimer); learnRefreshTimer = null; }
  if (defenseRefreshTimer) { clearInterval(defenseRefreshTimer); defenseRefreshTimer = null; }
  clearEpidemicRefresh();
}

/* ══════════════════════════════════════════════
   EPIDEMIC Tab — Layer 4 / WoW SEIR Monitor
   ══════════════════════════════════════════════ */
let epidemicRefreshTimer = null;
let seirChart = null;

function clearEpidemicRefresh() {
  if (epidemicRefreshTimer) { clearInterval(epidemicRefreshTimer); epidemicRefreshTimer = null; }
}

const SEIR_COLORS = {
  susceptible:  "#39ff14",   // neon green
  exposed:      "#ffe600",   // yellow
  infected:     "#ff6b35",   // orange
  compromised:  "#ff2d55",   // red/magenta
  recovered:    "#00e5ff",   // cyan
  unknown:      "#444444",
};

const SIGNAL_LABELS = {
  belief_decay_rate:        "Confidence Decay",
  soul_alignment:           "Baseline Alignment",
  topic_drift:              "Topic Drift",
  strategy_entropy:         "Strategy Entropy",
  dissonance_trend:         "Deviation Trend",
  engagement_pattern_delta: "Engagement Delta",
};

function _alertColor(level) {
  return { clear: "#39ff14", watch: "#ffe600", warn: "#ff6b35", critical: "#ff2d55" }[level] || "#aaa";
}

function _buildSeirChart(state) {
  const ctx = document.getElementById("seir-chart");
  if (!ctx) return;
  const healthState = (state || "susceptible").toLowerCase();
  // Map health state to a simulated population distribution (single-agent view)
  const dist = { susceptible: 0, exposed: 0, infected: 0, compromised: 0, recovered: 0 };
  if (dist[healthState] !== undefined) dist[healthState] = 1;
  else dist.susceptible = 1;

  const labels = Object.keys(dist).map(k => k.charAt(0).toUpperCase() + k.slice(1));
  const data   = Object.values(dist);
  const colors = Object.keys(dist).map(k => SEIR_COLORS[k]);

  if (seirChart) {
    seirChart.data.datasets[0].data = data;
    seirChart.data.datasets[0].backgroundColor = colors;
    seirChart.update("none");
    return;
  }
  // eslint-disable-next-line no-undef
  seirChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderColor: "#0a0a0a", borderWidth: 2 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#39ff1499", font: { family: "JetBrains Mono", size: 9 }, boxWidth: 10, padding: 6 },
        },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.label}: ${c.raw === 1 ? "ACTIVE" : "inactive"}`,
          },
        },
      },
    },
  });
}

function _renderSignals(signals) {
  const el = document.getElementById("epidemic-signals");
  if (!el || !signals) return;
  el.innerHTML = Object.entries(SIGNAL_LABELS).map(([key, label]) => {
    const val = typeof signals[key] === "number" ? signals[key] : null;
    const pct = val !== null ? Math.round(val * 100) : null;
    const color = val === null ? "#aaa" : val > 0.65 ? "#ff2d55" : val > 0.45 ? "#ff6b35" : val > 0.25 ? "#ffe600" : "#39ff14";
    const bar = val !== null
      ? `<div style="height:3px;background:#111;border-radius:2px;overflow:hidden;margin-top:2px;">
           <div style="width:${pct}%;height:100%;background:${color};transition:width 0.3s;"></div></div>`
      : "";
    return `<div style="font-size:9px;">
      <span style="color:#39ff1455;">${label}</span>
      <span style="float:right;color:${color};">${pct !== null ? pct + "%" : "—"}</span>
      ${bar}
    </div>`;
  }).join("");
}

function _renderEpidemicMetrics(statusData) {
  const el = document.getElementById("epidemic-metrics");
  if (!el) return;
  const alert = statusData.alert_level || "clear";
  const score = typeof statusData.score === "number" ? (statusData.score * 100).toFixed(1) + "%" : "—";
  const health = (statusData.seir && statusData.seir.health_state) || "—";
  const alertCol = _alertColor(alert);
  el.innerHTML = `
    <div class="metric-card" style="flex:1;padding:8px 12px;border:1px solid #39ff1430;background:#0d0d0d;">
      <span class="metric-label" style="font-size:7px;color:#39ff1455;text-transform:uppercase;display:block;">Alert Level</span>
      <div class="metric-value" style="font-size:22px;font-weight:700;color:${alertCol};">${alert.toUpperCase()}</div>
    </div>
    <div class="metric-card" style="flex:1;padding:8px 12px;border:1px solid #39ff1430;background:#0d0d0d;">
      <span class="metric-label" style="font-size:7px;color:#39ff1455;text-transform:uppercase;display:block;">Drift Score</span>
      <div class="metric-value" style="font-size:22px;font-weight:700;color:${alertCol};">${score}</div>
    </div>
    <div class="metric-card" style="flex:1;padding:8px 12px;border:1px solid #39ff1430;background:#0d0d0d;">
      <span class="metric-label" style="font-size:7px;color:#39ff1455;text-transform:uppercase;display:block;">Health State</span>
      <div class="metric-value" style="font-size:14px;font-weight:700;color:${SEIR_COLORS[health] || "#aaa"};">${health.toUpperCase()}</div>
    </div>
    ${statusData.seir && statusData.seir.is_epidemic
      ? `<div class="metric-card" style="flex:1;padding:8px 12px;border:1px solid #ff2d5580;background:#1a0007;">
           <span class="metric-label" style="font-size:7px;color:#ff2d5599;text-transform:uppercase;display:block;">EPIDEMIC ACTIVE</span>
           <div class="metric-value" style="font-size:14px;font-weight:700;color:#ff2d55;">R0 &gt; 1.0</div>
         </div>`
      : ""}
  `;
}

function _renderEpidemicParams(simData) {
  const el = document.getElementById("epidemic-params");
  if (!el || !simData) return;
  const params = simData.epidemic_params || simData.final_stats || {};
  const lines = [];
  if (params.R0 !== undefined) lines.push(`R0 (repro num): ${typeof params.R0 === "number" ? params.R0.toFixed(2) : params.R0}`);
  if (params.sigma !== undefined) lines.push(`sigma (incubation): ${typeof params.sigma === "number" ? params.sigma.toFixed(3) : params.sigma}`);
  if (params.gamma !== undefined) lines.push(`gamma (recovery): ${typeof params.gamma === "number" ? params.gamma.toFixed(3) : params.gamma}`);
  if (params.beta !== undefined)  lines.push(`beta (transmission): ${typeof params.beta === "number" ? params.beta.toFixed(3) : params.beta}`);
  if (params.total_agents !== undefined) lines.push(`agents: ${params.total_agents}`);
  if (params.peak_infected !== undefined) lines.push(`peak infected: ${params.peak_infected}`);
  if (params.final_infected !== undefined) lines.push(`final infected: ${params.final_infected}`);
  el.innerHTML = lines.length
    ? lines.map(l => `<div>${esc(l)}</div>`).join("")
    : "<em style='color:#39ff1440;'>No simulation data yet</em>";
}

function _renderSimStats(simData) {
  const el = document.getElementById("epidemic-sim-stats");
  if (!el) return;
  if (!simData) { el.innerHTML = "<em style='color:#39ff1440;'>No data</em>"; return; }
  const stats = simData.final_stats || simData.stats || simData;
  el.innerHTML = Object.entries(stats).slice(0, 12).map(([k, v]) =>
    `<div><span style="color:#39ff1455;">${esc(String(k))}:</span> ${esc(String(v))}</div>`
  ).join("");
}

function _renderSimLog(simData) {
  const el = document.getElementById("epidemic-sim-log");
  if (!el) return;
  if (!simData) { el.innerHTML = "<em style='color:#39ff1440;'>Run a simulation to see events</em>"; return; }
  const events = simData.events || simData.stats_history || simData.log || [];
  if (!Array.isArray(events) || !events.length) {
    el.innerHTML = "<em style='color:#39ff1440;'>No event log in simulation output</em>";
    return;
  }
  el.innerHTML = events.slice(-40).reverse().map(ev => {
    const ts = ev.tick !== undefined ? `T${ev.tick}` : ev.ts ? esc(String(ev.ts).slice(0, 19)) : "";
    const msg = ev.event || ev.message || ev.type || JSON.stringify(ev).slice(0, 80);
    const lvlColor = (ev.type || "").includes("INFECT") || (ev.event || "").includes("INFECT") ? "#ff2d55"
                   : (ev.type || "").includes("RECOV") ? "#00e5ff" : "#39ff1499";
    return `<div><span style="color:#39ff1430;">${esc(ts)}</span> <span style="color:${lvlColor};">${esc(String(msg))}</span></div>`;
  }).join("");
}

async function loadEpidemicData() {
  const content = document.getElementById("epidemic-content");
  if (!content) return;

  try {
    const [statusRes, simRes] = await Promise.all([
      authFetch("/api/epidemic/status"),
      authFetch("/api/epidemic/simulation"),
    ]);
    const statusData = await statusRes.json().catch(() => ({}));
    const simData    = await simRes.json().catch(() => ({}));

    _renderEpidemicMetrics(statusData);
    _renderSignals(statusData.signals || {});
    _buildSeirChart((statusData.seir && statusData.seir.health_state) || "susceptible");
    _renderEpidemicParams(simData.available ? simData.data : null);
    _renderSimStats(simData.available ? simData.data : null);
    _renderSimLog(simData.available ? simData.data : null);
  } catch (e) {
    content.innerHTML = `<div class="alert-box warning">Epidemic data error: ${esc(String(e))}</div>`;
  }

  if (epidemicRefreshTimer) clearInterval(epidemicRefreshTimer);
  epidemicRefreshTimer = setInterval(() => { if (activeNav === "epidemic") loadEpidemicData(); }, 5000);
}

document.getElementById("epidemic-refresh")?.addEventListener("click", loadEpidemicData);

async function _runSimulation(type) {
  const statusEl = document.getElementById("sim-run-status");
  if (statusEl) statusEl.textContent = `Starting ${type} simulation...`;
  try {
    const res = await authFetch("/api/epidemic/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type }),
    });
    const data = await res.json().catch(() => ({}));
    if (data.ok) {
      if (statusEl) statusEl.textContent = `PID ${data.pid} running (${data.script})`;
      setTimeout(loadEpidemicData, 3000);
    } else {
      if (statusEl) statusEl.textContent = `Error: ${data.error || "unknown"}`;
    }
  } catch (e) {
    if (statusEl) statusEl.textContent = `Failed: ${String(e)}`;
  }
}

document.getElementById("sim-run-det")?.addEventListener("click", () => _runSimulation("deterministic"));
document.getElementById("sim-run-llm")?.addEventListener("click", () => _runSimulation("llm"));

/* ══════════════════════════════════════════════
   Event Processing
   ══════════════════════════════════════════════ */
function eventTag(ev) {
  const src = (ev.source || "").toLowerCase();
  const name = (ev.event || "").toLowerCase();
  const lvl = (ev.level || "").toUpperCase();
  if (lvl === "WARN" || lvl === "WARNING") return { tag: "[WARN]", cls: "event-tag-warn" };
  if (lvl === "ERROR" || lvl === "CRITICAL") return { tag: "[ALER]", cls: "event-tag-aler" };
  if (src === "security") return { tag: "[SEC]", cls: "event-tag-sec" };
  if (src === "redteam") return { tag: "[REDT]", cls: "event-tag-redt" };
  if (src === "philosophy" || src === "analyst" || src === "behavioral" || src === "belief") return { tag: "[ANLY]", cls: "event-tag-analyst" };
  if (maybeAlert(ev)) return { tag: "[ALER]", cls: "event-tag-aler" };
  return { tag: "[SYST]", cls: "event-tag-syst" };
}

function formatEventLine(ev) {
  const { tag, cls } = eventTag(ev);
  const ts = ev.ts ? ev.ts.replace("T", " ").replace("Z", "") : "";
  const src = (ev.source || "unknown").toUpperCase();
  const name = ev.event || "";
  const cid = ev.correlation_id || (ev.data && ev.data.correlation_id) || null;

  let msg = ev.message || "";
  if (!msg) {
    const author = ev.author || (ev.data && ev.data.author) || "";
    const preview = ev.preview || (ev.data && ev.data.preview) || "";
    if (author && preview) msg = `author=${author} | ${preview}`;
    else if (preview) msg = preview;
  }

  let extra = "";
  const ac = ev.attack_complexity || (ev.data && ev.data.attack_complexity);
  if (ac && ac.complexity_label) extra += ` | complexity=${ac.complexity_label}(${ac.pattern_count}/${ac.class_count})`;
  const pers = ev.attacker_persistence || (ev.data && ev.data.attacker_persistence);
  if (typeof pers === "number") extra += ` | persistence=${pers.toFixed(3)}`;
  if (cid) extra += ` | corr=${cid}`;

  const text = `${ts} [${src}] ${name} ${msg}${extra}`.trim();
  return { tag: `<span class="event-tag ${cls}">${tag}</span>`, text };
}

function summarizeEvent(ev) {
  const { text } = formatEventLine(ev);
  return text;
}

function matchesActiveTab(ev) {
  if (activeTab === "all") return true;
  if (activeTab === "alerts") return maybeAlert(ev);
  if (activeTab === "analyst") return ev.source === "analyst" || ev.source === "philosophy" || ev.source === "behavioral";
  return ev.source === activeTab;
}

function maybeAlert(ev) {
  const src = ev.source;
  const name = ev.event || "";
  if (src === "security" && (name === "injection_blocked" || name === "suspicious_block")) return true;
  if (src === "redteam" && name === "redteam_escalate") return true;
  const ac = ev.attack_complexity || (ev.data && ev.data.attack_complexity);
  if (ac && ac.complexity_label === "multi_pattern_chained_intent") return true;
  return false;
}

function eventId(ev) {
  const ts = ev.ts || ev.timestamp || "";
  const src = ev.source || "";
  const name = ev.event || "";
  const preview = (ev.preview || (ev.data && ev.data.preview) || "").slice(0, 80);
  return `${ts}_${src}_${name}_${preview}`;
}

function renderEvent(ev) {
  const id = eventId(ev);
  if (renderedEventIds.has(id)) return;
  renderedEventIds.add(id);
  if (renderedEventIds.size > MAX_RENDERED_IDS) {
    const arr = [...renderedEventIds];
    arr.splice(0, 100);
    renderedEventIds.clear();
    arr.forEach((x) => renderedEventIds.add(x));
  }
  playEventSound(ev);
  counters.totalEvents += 1;
  bufferedEvents.push(ev);
  if (bufferedEvents.length > MAX_BUFFER) bufferedEvents = bufferedEvents.slice(-MAX_BUFFER);

  if (ev.source === "security") {
    if (ev.event === "unicode_clean") {
      counters.unicodeCleanEvents += 1;
      const n = ev.stripped_hidden_chars ?? (ev.data && ev.data.stripped_hidden_chars) ?? 0;
      counters.hiddenCharsStripped += Number(n) || 0;
    }
    if (ev.event === "input_reject" || ev.event === "injection_blocked" || ev.event === "suspicious_block") {
      counters.injectionAttempts += 1;
    }
    if (ev.event === "input_reject" || ev.event === "injection_blocked" || ev.event === "output_redact") {
      counters.sanitizations += 1;
    }
    const pm = ev.patterns_matched ?? (ev.data && ev.data.patterns_matched);
    if (typeof pm === "number") counters.patternMatches += pm;
  }

  if ((ev.source === "philosophy" || ev.source === "analyst" || ev.source === "behavioral") && (ev.event === "epistemic_state" || ev.event === "behavioral_state")) {
    const es = ev.epistemic_state || (ev.data && ev.data.epistemic_state);
    if (es) {
      const c = Number(es.confidence_score);
      const ent = Number(es.uncertainty_entropy);
      const a = Number(es.anthropomorphism_index);
      if (!Number.isNaN(c) && !Number.isNaN(ent) && !Number.isNaN(a)) {
        counters.epistemicN += 1;
        counters.sumConf += c;
        counters.sumEntropy += ent;
        counters.sumAnth += a;
      }
    }
  }

  const lvl = ev.level || "INFO";
  const cls = levelClass(lvl);
  const { tag, text } = formatEventLine(ev);
  const line = `<span class="${cls} dim">${tag} ${esc(text)}</span>`;
  if (!feedPaused && matchesActiveTab(ev)) appendLine(panes.feed, line);
  if (maybeAlert(ev)) {
    appendLine(panes.alerts, `<span class="lvl-alert">${esc(text)}</span>`);
  }

  injEls.hidden.textContent = String(counters.hiddenCharsStripped);
  injEls.unicode.textContent = String(counters.unicodeCleanEvents);
  injEls.patterns.textContent = String(counters.patternMatches);
  injEls.rate.textContent = counters.injectionAttempts > 0
    ? `${Math.round((counters.sanitizations / counters.injectionAttempts) * 100)}%`
    : "\u2014";

  const n = counters.epistemicN;
  const meanConf = n ? (counters.sumConf / n) : null;
  const meanEnt = n ? (counters.sumEntropy / n) : null;
  const meanAnth = n ? (counters.sumAnth / n) : null;
  epiEls.conf.textContent = meanConf == null ? "\u2014" : meanConf.toFixed(3);
  epiEls.ent.textContent = meanEnt == null ? "\u2014" : meanEnt.toFixed(3);
  epiEls.anth.textContent = meanAnth == null ? "\u2014" : meanAnth.toFixed(3);
  epiEls.over.textContent = meanConf == null ? "\u2014" : (1 - meanConf).toFixed(3);

  if (footerEls.conf) footerEls.conf.textContent = meanConf == null ? "\u2014" : meanConf.toFixed(3);
  if (footerEls.ent) footerEls.ent.textContent = meanEnt == null ? "\u2014" : meanEnt.toFixed(3);

  const avgScore = meanConf != null ? meanConf.toFixed(2) : "0.70";
  if (footerEls.avgScore) footerEls.avgScore.textContent = avgScore;
  if (footerEls.totalDecisions) footerEls.totalDecisions.textContent = String(counters.totalEvents);
}

/* ══════════════════════════════════════════════
   Metrics
   ══════════════════════════════════════════════ */
function renderKarmaHistoryBars(history) {
  const el = document.getElementById("karma-history-bars");
  if (!el) return;
  const arr = Array.isArray(history) ? history.slice(-24) : [];
  el.innerHTML = arr.map((v) => {
    const n = Number(v);
    const pct = Math.min(100, Math.max(0, (n + 10) * 5));
    return `<span class="karma-bar" style="height:6px;min-width:4px;flex:1;background:hsla(142,70%,45%,${0.3 + pct / 200});border-radius:1px;"></span>`;
  }).join("");
  if (!arr.length) el.innerHTML = '<span class="text-cyber-green/30 text-[8px]">—</span>';
}

function renderDefenseHistoryBars(history) {
  const el = document.getElementById("defense-history-bars");
  if (!el) return;
  const arr = Array.isArray(history) ? history.slice(-24) : [];
  el.innerHTML = arr.map((v) => {
    const n = Number(v) || 0;
    const pct = Math.min(100, n * 20);
    return `<span class="defense-bar" style="height:6px;min-width:4px;flex:1;background:hsla(180,70%,50%,${0.2 + pct / 150});border-radius:1px;"></span>`;
  }).join("");
  if (!arr.length) el.innerHTML = '<span class="text-cyber-green/30 text-[8px]">—</span>';
}

function renderMetrics(m) {
  const moodVal = m.agent_mood || "\u2014";
  const statusVal = m.running ? (m.suspended ? "PAUSED" : "RUNNING") : "STOPPED";
  els.agent.textContent = statusVal;
  els.pid.textContent = m.pid || "\u2014";
  els.mood.textContent = moodVal;
  document.querySelectorAll(".mood-value, #mood-badge").forEach((el) => { el.textContent = moodVal; });
  els.inj.textContent = m.injection_attempts_detected ?? 0;
  if (els.sanitized) els.sanitized.textContent = m.sanitized_payload_count ?? 0;
  els.reward.textContent = m.reward_score_rolling_sum ?? 0;
  els.fp.textContent = (m.false_positive_rate == null) ? "\u2014" : String(m.false_positive_rate);
  if (els.belief) els.belief.textContent = (m.belief_confidence == null) ? "\u2014" : m.belief_confidence.toFixed(3);

  const mStatus = document.getElementById("m-status");
  const mCycle = document.getElementById("m-cycle");
  const mKarma = document.getElementById("m-karma");
  const mHeartbeat = document.getElementById("m-heartbeat");
  const mDefenseRate = document.getElementById("m-defense-rate");
  const mResilience = document.getElementById("m-resilience");
  const mRebars = document.getElementById("m-rebars");
  if (mStatus) mStatus.textContent = statusVal;
  if (mCycle) mCycle.textContent = m.cycle_count ?? m.cycle ?? "\u2014";
  if (mKarma) mKarma.textContent = m.current_karma ?? (Array.isArray(m.karma_history) && m.karma_history.length ? m.karma_history[m.karma_history.length - 1] : "\u2014");
  if (mHeartbeat) mHeartbeat.textContent = m.heartbeat_interval_minutes != null ? `${m.heartbeat_interval_minutes}m` : "\u2014";
  if (mDefenseRate) mDefenseRate.textContent = (m.defense_rate != null) ? `${Math.round((m.defense_rate || 0) * 100)}%` : "100%";
  if (mResilience) mResilience.textContent = (m.belief_confidence != null) ? m.belief_confidence.toFixed(2) : (m.resilience != null ? m.resilience : "\u2014");
  if (mRebars) mRebars.textContent = m.rebars ?? "\u2014";

  renderKarmaHistoryBars(m.karma_history);
  renderDefenseHistoryBars(m.defense_history ?? m.sanitized_payload_count != null ? [m.sanitized_payload_count] : []);

  if (footerEls.currentMood) footerEls.currentMood.textContent = moodVal;
  if (footerEls.totalDecisions) footerEls.totalDecisions.textContent = String(m.cycle_count ?? m.cycle ?? counters.totalEvents ?? 0);

  const changed =
    lastStatus.running !== m.running ||
    lastStatus.suspended !== m.suspended ||
    lastStatus.pid !== (m.pid || null);

  if (changed) {
    if (m.running && !m.suspended && !lastStatus.running) playSessionSound();
    const sys = `agent_running=${m.running} suspended=${m.suspended} pid=${m.pid || "\u2014"}`;
    appendLine(panes.system, `<span class="lvl-debug">[SYS] ${esc(sys)}</span>`);
    lastStatus = {
      running: m.running,
      suspended: m.suspended,
      pid: m.pid || null,
    };
  }
}

/* ══════════════════════════════════════════════
   Agent Activity
   ══════════════════════════════════════════════ */
let agentActivityFailCount = 0;
const AGENT_ACTIVITY_BASE_MS = 2500;
const AGENT_ACTIVITY_MAX_MS = 60000;

async function refreshAgentActivity() {
  try {
    if (!panes.activity) return;

    const res = await authFetchWithRetry("/api/agent-activity");
    if (!res.ok) {
      appendLine(panes.system, `<span class="lvl-warn">[SYS] agent_activity_http_${res.status}</span>`);
      return;
    }
    const data = await res.json();
    if (!data || data.ok === false || !Array.isArray(data.lines)) {
      appendLine(panes.system, `<span class="lvl-warn">[SYS] agent_activity_invalid_response</span>`);
      return;
    }

    const joined = data.lines.join("\n");
    if (joined === lastActivitySnapshot) return;
    lastActivitySnapshot = joined;

    panes.activity.textContent = "";
    if (!data.lines.length) {
      panes.activity.textContent = "(no recent agent activity)";
      return;
    }
    const reversed = data.lines.slice().reverse();
    for (const line of reversed) {
      panes.activity.insertAdjacentHTML("beforeend", esc(line) + "\n");
    }
    panes.activity.scrollTop = 0;
    agentActivityFailCount = 0;
  } catch (_) {
    agentActivityFailCount++;
    if (panes.system) {
      appendLine(panes.system, `<span class="lvl-warn">[SYS] agent_activity_fetch_error</span>`);
    }
  }
}

function scheduleAgentActivityRefresh() {
  const backoff = Math.min(AGENT_ACTIVITY_BASE_MS * Math.pow(2, Math.min(agentActivityFailCount, 4)), AGENT_ACTIVITY_MAX_MS);
  setTimeout(async () => {
    await refreshAgentActivity();
    scheduleAgentActivityRefresh();
  }, backoff);
}

/* ══════════════════════════════════════════════
   Actions
   ══════════════════════════════════════════════ */
async function post(path, body = null) {
  const res = await authFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : "{}",
  });
  return await res.json();
}

async function runAction(btnId, path, body, level) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = true;
  btn.classList.remove("btn-pulse-ok", "btn-pulse-err");
  try {
    const res = await post(path, body);
    const ok = res && res.ok !== false && !res.error;
    const cls = ok ? "btn-pulse-ok" : "btn-pulse-err";
    btn.classList.add(cls);
    setTimeout(() => btn.classList.remove(cls), 260);
    const lvl = ok ? level : "lvl-error";
    appendLine(panes.system, `<span class="${lvl}">[CMD] ${esc(JSON.stringify(res))}</span>`);
  } catch (e) {
    btn.classList.add("btn-pulse-err");
    setTimeout(() => btn.classList.remove("btn-pulse-err"), 260);
    appendLine(panes.system, `<span class="lvl-error">[ERR] ${esc(String(e))}</span>`);
  } finally {
    btn.disabled = false;
  }
}

async function refreshAgentControlMetrics() {
  try {
    const res = await authFetchWithRetry("/api/status", {}, 1);
    if (res.ok) {
      const d = await res.json();
      if (d.metrics) renderMetrics(d.metrics);
    }
  } catch (_) {}
}

document.getElementById("start").addEventListener("click", async () => {
  const mode = modeSelect.value;
  await runAction("start", "/api/agent/start", { mode }, "lvl-info");
  setTimeout(refreshAgentControlMetrics, 500);
});
document.getElementById("pause").addEventListener("click", async () => {
  await runAction("pause", "/api/agent/pause", {}, "lvl-warn");
  setTimeout(refreshAgentControlMetrics, 500);
});
document.getElementById("resume").addEventListener("click", async () => {
  await runAction("resume", "/api/agent/resume", {}, "lvl-info");
  setTimeout(refreshAgentControlMetrics, 500);
});
document.getElementById("restart").addEventListener("click", async () => {
  const mode = modeSelect.value;
  await runAction("restart", "/api/agent/restart", { mode }, "lvl-alert");
  setTimeout(refreshAgentControlMetrics, 500);
});
document.getElementById("kill").addEventListener("click", async () => {
  await runAction("kill", "/api/agent/kill", {}, "lvl-error");
  setTimeout(refreshAgentControlMetrics, 500);
});

/* ══════════════════════════════════════════════
   Feed Controls
   ══════════════════════════════════════════════ */
function updateFeedStatus(kind) {
  feedStatusIcon.classList.remove("status-live", "status-paused", "status-off", "status-polling");
  if (kind === "live") {
    feedStatusIcon.classList.add("status-live");
    feedStatusLabel.textContent = "LIVE";
  } else if (kind === "paused") {
    feedStatusIcon.classList.add("status-paused");
    feedStatusLabel.textContent = "PAUSED";
  } else if (kind === "polling") {
    feedStatusIcon.classList.add("status-polling");
    feedStatusLabel.textContent = "POLLING";
  } else {
    feedStatusIcon.classList.add("status-off");
    feedStatusLabel.textContent = "STOPPED";
  }
}

function applyTimeFilter() {
  const from = timeFrom?.value ? new Date(timeFrom.value).getTime() : null;
  const to = timeTo?.value ? new Date(timeTo.value).getTime() : null;
  panes.feed.textContent = "";
  const filtered = [];
  for (const ev of bufferedEvents) {
    if (!ev.ts) continue;
    const ts = Date.parse(ev.ts);
    if (Number.isNaN(ts)) continue;
    if (from !== null && ts < from) continue;
    if (to !== null && ts > to) continue;
    if (!matchesActiveTab(ev)) continue;
    filtered.push(ev);
  }
  for (let i = filtered.length - 1; i >= 0; i--) {
    const ev = filtered[i];
    const lvl = ev.level || "INFO";
    const cls = levelClass(lvl);
    const { tag, text } = formatEventLine(ev);
    const line = `<span class="${cls} dim">${tag} ${esc(text)}</span>`;
    panes.feed.insertAdjacentHTML("beforeend", line + "\n");
  }
  panes.feed.scrollTop = 0;
}

timeApply?.addEventListener("click", () => {
  feedPaused = true;
  if (feedToggle) feedToggle.textContent = "\u25b9 LIVE";
  updateFeedStatus("paused");
  applyTimeFilter();
});

timeClear?.addEventListener("click", () => {
  if (timeFrom) timeFrom.value = "";
  if (timeTo) timeTo.value = "";
  panes.feed.textContent = "";
  feedPaused = false;
  if (feedToggle) feedToggle.textContent = "\u23f8 PAUSE";
  updateFeedStatus(wsLive ? "live" : "off");
  for (let i = bufferedEvents.length - 1; i >= 0; i--) {
    const ev = bufferedEvents[i];
    if (!matchesActiveTab(ev)) continue;
    const lvl = ev.level || "INFO";
    const cls = levelClass(lvl);
    const { tag, text } = formatEventLine(ev);
    const line = `<span class="${cls} dim">${tag} ${esc(text)}</span>`;
    panes.feed.insertAdjacentHTML("beforeend", line + "\n");
  }
  panes.feed.scrollTop = 0;
});

feedToggle?.addEventListener("click", () => {
  feedPaused = !feedPaused;
  if (feedPaused) {
    if (feedToggle) feedToggle.textContent = "\u25b9 LIVE";
    updateFeedStatus("paused");
  } else {
    if (feedToggle) feedToggle.textContent = "\u23f8 PAUSE";
    updateFeedStatus(wsLive ? "live" : "off");
  }
});

/* ══════════════════════════════════════════════
   WebSocket
   ══════════════════════════════════════════════ */
let wsReconnectCount = 0;
const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 30000;

function connect() {
  const token = getAuthToken();
  const qs = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`ws://${location.host}/ws/live${qs}`);
  wsLive = ws;
  ws.onopen = () => {
    wsReconnectCount = 0;
    appendLine(panes.system, `<span class="lvl-info">[WS] connected</span>`);
    if (!feedPaused) updateFeedStatus("live");
  };
  ws.onclose = (ev) => {
    if (ev.code === 4001) {
      setAuthToken(null);
      showAuthOverlay();
      wsLive = null;
      updateFeedStatus("off");
      return;
    }
    wsReconnectCount++;
    const delay = Math.min(WS_RECONNECT_BASE_MS * Math.pow(2, Math.min(wsReconnectCount - 1, 4)), WS_RECONNECT_MAX_MS);
    appendLine(panes.system, `<span class="lvl-warn">[WS] disconnected — reconnecting in ${Math.round(delay / 1000)}s</span>`);
    wsLive = null;
    updateFeedStatus("off");
    setTimeout(connect, delay);
  };
  ws.onerror = () => {};
  ws.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      if (data.type === "event") renderEvent(data.event);
      if (data.type === "metrics") renderMetrics(data.metrics);
    } catch (_) {}
  };
}

/* ══════════════════════════════════════════════
   Chat — Talk to the agent
   ══════════════════════════════════════════════ */
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

function appendChatMessage(role, text, interactionId) {
  if (!chatMessages) return;
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  const label = role === "user" ? "You" : "Sancta";
  div.innerHTML = `<span class="chat-label">${esc(label)}</span><br>${esc(text)}`;
  chatMessages.appendChild(div);
  if (role === "agent" && interactionId) {
    const feedbackRow = document.createElement("div");
    feedbackRow.className = "chat-feedback-row";
    feedbackRow.innerHTML = `<span class="chat-feedback-label">Was that helpful?</span>
      <button type="button" class="btn btn-small chat-fb" data-fb="1" title="Good">+</button>
      <button type="button" class="btn btn-small chat-fb" data-fb="0" title="Neutral">?</button>
      <button type="button" class="btn btn-small chat-fb" data-fb="-1" title="Bad">−</button>`;
    feedbackRow.querySelectorAll(".chat-fb").forEach((btn) => {
      btn.addEventListener("click", () => submitFeedback(interactionId, parseInt(btn.dataset.fb, 10), btn));
    });
    chatMessages.appendChild(feedbackRow);
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function submitFeedback(interactionId, feedback, btn) {
  if (!btn) return;
  btn.disabled = true;
  btn.classList.add("chat-fb-sent");
  try {
    const res = await authFetch("/api/chat/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ interaction_id: interactionId, feedback }),
    });
    const data = await res.json().catch(() => ({}));
    if (data?.ok && panes?.system) {
      appendLine(panes.system, `<span class="lvl-info">[LEARNING] Feedback recorded: ${feedback === 1 ? "good" : feedback === -1 ? "bad" : "neutral"}</span>`);
    }
  } catch (e) {
    if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[CHAT] Feedback failed: ${esc(String(e))}</span>`);
  }
}

async function sendChatMessage() {
  if (!chatInput || !chatSend) return;
  const msg = (chatInput.value || "").trim();
  if (!msg) return;

  chatSend.disabled = true;
  chatInput.value = "";
  appendChatMessage("user", msg);

  try {
    const trainCb = document.getElementById("chat-train-on-exchange");
    const res = await authFetch("/api/chat/gpt", {
      timeoutMs: CHAT_FETCH_TIMEOUT_MS,
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({
        message: msg,
        temperature: 0.7,
        train_on_exchange: trainCb?.checked === true,
      }),
    });

    let data;
    try {
      data = await res.json();
    } catch (_) {
      data = { error: `HTTP ${res.status}`, detail: "Invalid response" };
    }

    if (res.status === 404) {
      appendChatMessage("agent", "Chat endpoint not found (404). Restart the SIEM server.");
      if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[CHAT] 404 — restart SIEM: python -m uvicorn backend.siem_server:app</span>`);
      return;
    }
    if (res.status === 401) {
      appendChatMessage("agent", "Unauthorized. Enter the SIEM token to use chat.");
      return;
    }
    if (res.status >= 400) {
      const err = data?.error || data?.detail || data?.reason || `HTTP ${res.status}`;
      appendChatMessage("agent", `Error: ${err}`);
      if (panes.system) appendLine(panes.system, `<span class="lvl-error">[CHAT] ${res.status}: ${esc(err)}</span>`);
      return;
    }

    if (data && data.ok && data.reply) {
      const routeHint =
        data.route && data.backend
          ? ` [${data.route} → ${data.backend}${data.knowledge_effective ? ", knowledge path" : ""}]`
          : "";
      appendChatMessage("agent", `${data.reply}${routeHint}`, data.interaction_id);
    } else {
      const err = data?.error || data?.reason || data?.detail || "Failed to get reply";
      appendChatMessage("agent", `${err}`);
      if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[CHAT] ${esc(err)}</span>`);
    }
  } catch (e) {
    const isAbort = e?.name === "AbortError" || /aborted|timeout/i.test(String(e?.message || e));
    const msg = isAbort
      ? "Request timed out — try again in a moment."
      : `Error: ${esc(String(e?.message || e))}`;
    appendChatMessage("agent", msg);
    if (panes.system) appendLine(panes.system, `<span class="lvl-error">[CHAT] ${esc(msg)}</span>`);
  } finally {
    chatSend.disabled = false;
    chatInput?.focus();
  }
}

if (chatSend) {
  chatSend.addEventListener("click", sendChatMessage);
}
if (chatInput) {
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
}

/* ══════════════════════════════════════════════
   Sound Toggle
   ══════════════════════════════════════════════ */
function updateSoundButtonUI(btn) {
  if (!btn) return;
  btn.textContent = audioEnabled ? "\uD83D\uDD0A SOUND" : "\uD83D\uDD07 MUTED";
  btn.classList.toggle("btn-dim", !audioEnabled);
}

const soundBtn = document.getElementById("sound-toggle");
if (soundBtn) {
  updateSoundButtonUI(soundBtn);
  soundBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    audioEnabled = !audioEnabled;
    try { localStorage.setItem(AUDIO_STORAGE_KEY, String(audioEnabled)); } catch (_) {}
    updateSoundButtonUI(soundBtn);
  });
}

/* ══════════════════════════════════════════════
   Auth overlay
   ══════════════════════════════════════════════ */
let initRan = false;

function showAuthOverlay() {
  const el = document.getElementById("auth-overlay");
  if (el) {
    el.style.display = "flex";
    document.getElementById("auth-token")?.focus();
  }
}

function hideAuthOverlay() {
  const el = document.getElementById("auth-overlay");
  if (el) el.style.display = "none";
}

function setAuthError(msg) {
  const el = document.getElementById("auth-error");
  if (el) el.textContent = msg || "";
}

/* ══════════════════════════════════════════════
   Cyberpunk data rain — low-opacity background for live feed
   ══════════════════════════════════════════════ */
const CYBER_CHARS = "01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン";
const RAIN_FONT_SIZE = 12;
const RAIN_COLUMN_SPACING = 18;

function initCyberRain() {
  const canvas = document.getElementById("cyber-rain");
  const feedContainer = canvas?.closest(".panel.feed");
  if (!canvas || !feedContainer) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let rafId = null;
  let columns = [];
  let lastResize = 0;

  function resize() {
    const container = canvas.closest(".panel.feed");
    if (!container) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) return;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.scale(dpr, dpr);

    const colCount = Math.ceil(w / RAIN_COLUMN_SPACING);
    columns = columns.slice(0, colCount);
    while (columns.length < colCount) {
      columns.push({
        y: Math.random() * h,
        speed: 0.16 + Math.random() * 0.24,
        chars: Array.from({ length: 8 }, () => CYBER_CHARS[Math.floor(Math.random() * CYBER_CHARS.length)]),
      });
    }
  }

  function draw() {
    if (document.hidden) {
      rafId = requestAnimationFrame(draw);
      return;
    }
    const container = canvas.closest(".panel.feed");
    if (!container) {
      rafId = requestAnimationFrame(draw);
      return;
    }
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) {
      rafId = requestAnimationFrame(draw);
      return;
    }

    ctx.fillStyle = "hsla(150, 40%, 1%, 0.06)";
    ctx.fillRect(0, 0, w, h);

    ctx.font = `${RAIN_FONT_SIZE}px 'Courier New', monospace`;
    const colCount = Math.ceil(w / RAIN_COLUMN_SPACING);

    for (let i = 0; i < colCount; i++) {
      const col = columns[i] || { y: 0, speed: 0.05, chars: [] };
      col.y += col.speed;
      if (col.y > h + 50) col.y = -30;

      for (let j = 0; j < col.chars.length; j++) {
        const y = col.y - j * RAIN_FONT_SIZE;
        if (y < -RAIN_FONT_SIZE || y > h) continue;
        const alpha = 1 - j / col.chars.length;
        ctx.fillStyle = `hsla(140, 100%, 50%, ${alpha * 0.7})`;
        ctx.fillText(col.chars[j], i * RAIN_COLUMN_SPACING, y);
      }
    }
    rafId = requestAnimationFrame(draw);
  }

  resize();
  draw();

  const ro = new ResizeObserver(() => {
    const now = Date.now();
    if (now - lastResize < 100) return;
    lastResize = now;
    resize();
  });
  ro.observe(feedContainer);
}

/* ══════════════════════════════════════════════
   Security Incidents & Injection Types — Chart.js
   ══════════════════════════════════════════════ */
let incidentsChart = null;
let incidentsContainer = null;
let graphLogWrap = null;
let showGraphView = true;
let incidentsRefreshTimer = null;
let incidentsChartType = "bar";
let lastInjectionTypes = {};
let lastInjectionClasses = {};

const INCIDENT_CHART_COLORS = [
  "hsla(180, 100%, 50%, 0.85)",
  "hsla(340, 100%, 50%, 0.85)",
  "hsla(30, 100%, 50%, 0.85)",
  "hsla(70, 100%, 50%, 0.85)",
  "hsla(140, 100%, 50%, 0.85)",
  "hsla(300, 100%, 50%, 0.8)",
];

function updateIncidentsChart(injectionTypes, injectionClasses) {
  const canvas = document.getElementById("security-incidents-chart");
  if (!canvas) return;

  const data = Object.entries(injectionClasses).length ? injectionClasses : injectionTypes;
  const labels = Object.keys(data).map((k) => (k.length > 12 ? k.slice(0, 11) + "…" : k));
  const values = Object.values(data);
  const colors = labels.map((_, i) => INCIDENT_CHART_COLORS[i % INCIDENT_CHART_COLORS.length]);

  const type = incidentsChartType;
  const isPieLike = type === "pie" || type === "doughnut";

  if (incidentsChart) {
    const needsRecreate = incidentsChart.config.type !== type;
    if (needsRecreate) {
      incidentsChart.destroy();
      incidentsChart = null;
    } else {
      incidentsChart.data.labels = labels;
      incidentsChart.data.datasets[0].data = values;
      incidentsChart.data.datasets[0].backgroundColor = colors;
      if (!isPieLike) incidentsChart.data.datasets[0].borderColor = "hsla(140, 100%, 50%, 0.5)";
      incidentsChart.update("none");
      return;
    }
  }

  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: isPieLike },
      tooltip: {
        titleFont: { family: "'Courier New', monospace" },
        bodyFont: { family: "'Courier New', monospace" },
      },
    },
  };

  const scalesOptions = isPieLike ? {} : {
    scales: {
      x: {
        ticks: {
          color: "hsl(140, 100%, 50%)",
          font: { family: "'Courier New', monospace", size: 9 },
          maxRotation: -45,
          minRotation: -45,
        },
        grid: { color: "hsla(140, 100%, 50%, 0.15)" },
      },
      y: {
        ticks: {
          color: "hsl(140, 100%, 50%)",
          font: { family: "'Courier New', monospace", size: 9 },
        },
        grid: { color: "hsla(140, 100%, 50%, 0.15)" },
      },
    },
  };

  incidentsChart = new Chart(canvas, {
    type,
    data: {
      labels,
      datasets: [
        {
          label: "Incidents",
          data: values,
          backgroundColor: colors,
          borderColor: isPieLike ? colors : "hsla(140, 100%, 50%, 0.5)",
          borderWidth: 1,
          fill: type === "line",
          tension: type === "line" ? 0.3 : 0,
        },
      ],
    },
    options: { ...baseOptions, ...scalesOptions },
  });
}

async function loadSecurityIncidents() {
  try {
    const res = await authFetch("/api/security/incidents");
    const data = await res.json().catch(() => ({}));
    if (!data.ok) {
      document.getElementById("inc-rate-1h").textContent = "—";
      document.getElementById("inc-rate-24h").textContent = "—";
      document.getElementById("inc-rate-7d").textContent = "—";
      document.getElementById("inc-total").textContent = "—";
      if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[SEC] Failed: ${esc(data.error || "Unknown")}</span>`);
      return;
    }
    const r = data.rates || {};
    document.getElementById("inc-rate-1h").textContent = String(r.last_hour ?? 0);
    document.getElementById("inc-rate-24h").textContent = String(r.last_24h ?? 0);
    document.getElementById("inc-rate-7d").textContent = String(r.last_7d ?? 0);
    document.getElementById("inc-total").textContent = String(r.total ?? 0);
    const types = data.injection_types || {};
    const classes = data.injection_classes || {};
    const legendEl = document.getElementById("injection-types-legend");
    if (legendEl) {
      const combined = { ...types, ...classes };
      legendEl.innerHTML = Object.entries(combined)
        .slice(0, 10)
        .map(([k, v]) => `<span class="px-1.5 py-0.5 border border-cyber-green/30 rounded text-cyber-cyan">${esc(k)}: ${v}</span>`)
        .join("");
    }
    updateIncidentsChart(types, classes);
    if (panes.feed) {
      const recent = data.recent_incidents || [];
      const eventClass = (ev) => {
        if (/(reject|blocked|ingest_reject)/.test(ev)) return "lvl-warn";
        if (ev === "output_redact") return "lvl-alert";
        if (ev === "ioc_domain_detected") return "lvl-error";
        return "lvl-info";
      };
      const logContent = recent.length
        ? recent.slice(0, 25).map((i) => {
            const ts = (i.ts || "").slice(0, 19).replace("T", " ");
            const ev = esc(i.event || "");
            const author = esc((i.author || "").trim());
            let preview = (i.preview || "").trim();
            if (preview === "{}" || preview === "" || !preview) {
              preview = ev.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
            }
            preview = esc(preview);
            const detail = [author, preview].filter(Boolean).join(" · ");
            const cls = eventClass(i.event);
            return `<div class="incident-line flex gap-2 py-0.5 text-[10px]"><span class="text-cyber-green/50 shrink-0 w-[19ch]">${ts}</span><span class="${cls} font-medium shrink-0">[${ev}]</span>${detail ? `<span class="text-cyber-cyan/80 min-w-0 truncate">${detail}</span>` : ""}</div>`;
          }).join("")
        : '<span class="text-cyber-green/50">No recent incidents.</span>';
      panes.feed.innerHTML = logContent;
    }
    const recentScansEl = document.getElementById("recent-scans-list");
    if (recentScansEl) {
      const recent = data.recent_incidents || [];
      recentScansEl.innerHTML = recent.length
        ? recent.slice(0, 10).map((i) => {
            let preview = (i.preview || i.event || "").trim();
            if (preview === "{}" || !preview) preview = (i.event || "").replace(/_/g, " ");
            return `<div class="recent-scan-item flex items-center gap-1.5 py-0.5"><span class="text-cyber-green">✓</span><span class="truncate">${esc(preview.slice(0, 50))}</span></div>`;
          }).join("")
        : '<span class="text-cyber-green/50">No recent scans.</span>';
    }
  } catch (e) {
    if (panes.system) appendLine(panes.system, `<span class="lvl-error">[SEC] ${esc(String(e))}</span>`);
  }
}

function initSecurityIncidents() {
  graphLogWrap = document.getElementById("pane-feed-wrap");
  incidentsContainer = document.getElementById("security-incidents-container");
  if (!incidentsContainer) return;

  const ro = new ResizeObserver(() => {
    if (incidentsChart) incidentsChart.resize();
  });
  ro.observe(incidentsContainer);

  document.getElementById("graph-log-toggle")?.addEventListener("click", () => {
    showGraphView = !showGraphView;
    const btn = document.getElementById("graph-log-toggle");
    if (btn) btn.textContent = showGraphView ? "LOG" : "CHART";
    if (incidentsContainer) incidentsContainer.classList.toggle("hidden", !showGraphView);
    if (graphLogWrap) graphLogWrap.classList.toggle("hidden", showGraphView);
    loadSecurityIncidents();
  });

  document.getElementById("chart-type")?.addEventListener("change", (e) => {
    incidentsChartType = e.target?.value || "bar";
    if (Object.keys(lastInjectionTypes).length || Object.keys(lastInjectionClasses).length) {
      updateIncidentsChart(lastInjectionTypes, lastInjectionClasses);
    }
  });

  document.getElementById("incidents-refresh")?.addEventListener("click", () => {
    loadSecurityIncidents();
  });

  loadSecurityIncidents();
  if (incidentsRefreshTimer) clearInterval(incidentsRefreshTimer);
  incidentsRefreshTimer = setInterval(loadSecurityIncidents, 8000);
}

let wsSafeMode = false;

async function checkModelStatus() {
  const ind = document.getElementById("modelStatusIndicator");
  const txt = document.getElementById("modelStatusText");
  if (!ind || !txt) return;
  try {
    const res = await fetch("/api/model/info");
    const info = await res.json().catch(() => ({}));
    ind.classList.remove("connected", "disconnected", "disabled");
    if (info.status === "connected") {
      ind.classList.add("connected");
      txt.textContent = `${info.model || "LLM"} ready`;
    } else if (info.status === "disconnected") {
      ind.classList.add("disconnected");
      txt.textContent = info.disconnect_reason ? `Offline: ${info.disconnect_reason}` : "LLM offline (fallback mode)";
    } else if (info.status === "disabled") {
      ind.classList.add("disabled");
      txt.textContent = "LLM disabled";
    } else {
      txt.textContent = info.error || "Unknown";
    }
  } catch (e) {
    ind.classList.remove("connected", "disconnected", "disabled");
    ind.classList.add("disconnected");
    txt.textContent = "LLM check failed";
  }
}

async function runInit() {
  if (initRan) {
    if (!wsSafeMode) connect();
    return;
  }
  initRan = true;
  await new Promise((r) => setTimeout(r, 150));

  // Motion One: subtle panel fade-in
  document.querySelectorAll(".lab-panel").forEach((el, i) => {
    el.style.opacity = "0";
    animate(
      el,
      { opacity: 1 },
      { duration: 0.4, delay: i * 0.04, easing: [0.25, 0.46, 0.45, 0.94] }
    );
  });

  initCyberRain();
  initSecurityIncidents();
  await initAudio();
  // Fetch metrics immediately so MOOD, INJ, REWARD show before WebSocket connects
  try {
    const statusRes = await authFetchWithRetry("/api/status", {}, 1);
    if (statusRes.ok) {
      const d = await statusRes.json();
      if (d.metrics) renderMetrics(d.metrics);
      wsSafeMode = !!d.ws_safe_mode;
    }
  } catch (e) {
    if (panes.system) appendLine(panes.system, `<span class="lvl-warn">[SYS] Initial status fetch failed: ${esc(String(e))}</span>`);
  }
  if (wsSafeMode) {
    appendLine(panes.system, `<span class="lvl-info">[SYS] polling mode (WebSocket disabled for stability)</span>`);
    updateFeedStatus("polling");
    pollStatusForMetrics();
  } else {
    connect();
  }
  await refreshAgentActivity();
  scheduleAgentActivityRefresh();
  pollLiveEvents();
  refreshEpistemicFromApi();
  setInterval(refreshEpistemicFromApi, 8000);
}

async function refreshEpistemicFromApi() {
  try {
    const res = await authFetchWithRetry("/api/epistemic", {}, 1);
    if (!res.ok) return;
    const data = await res.json();
    const epi = data.epistemic;
    if (!epi) return;
    const conf = epi.confidence_score;
    const ent = epi.uncertainty_entropy;
    const anth = epi.anthropomorphism_index;
    const hasValid = typeof conf === "number" && typeof ent === "number" && typeof anth === "number";
    if (hasValid) {
      counters.epistemicN = 1;
      counters.sumConf = conf;
      counters.sumEntropy = ent;
      counters.sumAnth = anth;
    }
    const meanConf = hasValid ? conf : null;
    const meanEnt = hasValid ? ent : null;
    const meanAnth = hasValid ? anth : null;
    if (epiEls.conf) epiEls.conf.textContent = meanConf != null ? meanConf.toFixed(3) : "\u2014";
    if (epiEls.ent) epiEls.ent.textContent = meanEnt != null ? meanEnt.toFixed(3) : "\u2014";
    if (epiEls.anth) epiEls.anth.textContent = meanAnth != null ? meanAnth.toFixed(3) : "\u2014";
    if (epiEls.over) epiEls.over.textContent = meanConf != null ? (1 - meanConf).toFixed(3) : "\u2014";
    if (footerEls.conf) footerEls.conf.textContent = meanConf != null ? meanConf.toFixed(3) : "\u2014";
    if (footerEls.ent) footerEls.ent.textContent = meanEnt != null ? meanEnt.toFixed(3) : "\u2014";
  } catch (_) {}
}

function pollStatusForMetrics() {
  const poll = async () => {
    try {
      const res = await authFetchWithRetry("/api/status", {}, 1);
      if (!res.ok) return;
      const d = await res.json();
      if (d.metrics) renderMetrics(d.metrics);
    } catch (_) {}
    setTimeout(poll, 1500);
  };
  poll();
  const pollEpi = async () => {
    await refreshEpistemicFromApi();
    setTimeout(pollEpi, 5000);
  };
  setTimeout(pollEpi, 2000);
}

async function pollLiveEvents() {
  const poll = async () => {
    try {
      const res = await authFetchWithRetry("/api/live-events", {}, 1);
      if (!res.ok) return;
      const data = await res.json();
      if (data.ok && Array.isArray(data.events)) {
        for (const ev of data.events) {
          renderEvent(ev);
        }
      }
    } catch (_) {}
    setTimeout(poll, 2500);
  };
  poll();
}

async function onAuthSubmit() {
  const input = document.getElementById("auth-token");
  const token = (input?.value || "").trim();
  if (!token) {
    setAuthError("Enter a token");
    return;
  }
  setAuthError("");
  const res = await fetch("/api/auth/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  const data = await res.json();
  if (data && data.ok) {
    setAuthToken(token);
    hideAuthOverlay();
    runInit();
  } else {
    setAuthError("Invalid token");
  }
}

async function bootstrap() {
  setNav("dashboard");
  setEventFilter("all");

  document.getElementById("auth-submit")?.addEventListener("click", onAuthSubmit);
  document.getElementById("auth-token")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") onAuthSubmit();
  });

  checkModelStatus();
  setInterval(checkModelStatus, 30000);

  const statusRes = await fetch("/api/auth/status");
  let authRequired = false;
  try {
    const status = await statusRes.json();
    authRequired = !!(status && status.auth_required);
  } catch (_) {}

  if (authRequired && !getAuthToken()) {
    showAuthOverlay();
    return;
  }

  runInit();
}

/* ══════════════════════════════════════════════
   Init
   ══════════════════════════════════════════════ */
bootstrap();

