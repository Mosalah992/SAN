/* ─── Control Tab ─────────────────────────────────────────── */
import { S } from '../state.js';
import * as api from '../api.js';
import * as notifications from '../components/notifications.js';

/* ── State ───────────────────────────────────────────────── */
let _allLogs      = [];       // merged log objects {ts, source, level, text}
let _services     = {};       // last fetched services map
let _activeSource = 'all';   // 'all' | 'security' | 'redteam' | 'analyst' | 'activity'
let _activeLevel  = 'all';   // 'all' | 'info' | 'warn' | 'error'
let _searchText   = '';
let _pollTimer    = null;
const MAX_LOGS    = 300;

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ── Log normalisation ───────────────────────────────────── */
function _levelOf(ev) {
  const src = (ev.source || ev.type || '').toLowerCase();
  const cat = (ev.category || '').toLowerCase();
  if (cat === 'redteam' || src.includes('block') || src.includes('threat')) return 'error';
  if (cat === 'security' || src.includes('warn') || src.includes('suspicious')) return 'warn';
  return 'info';
}

function _sourceOf(ev) {
  const cat = (ev.category || ev.type || '').toLowerCase();
  if (cat.includes('redteam') || cat.includes('red_team')) return 'redteam';
  if (cat.includes('security') || cat.includes('block') || cat.includes('threat')) return 'security';
  if (cat.includes('analyst') || cat.includes('behavioral') || cat.includes('philosophy') || cat.includes('soul') || cat.includes('belief')) return 'analyst';
  return 'activity';
}

function _wrapActivityLine(line, i) {
  const lvl = /error|fail|block/i.test(line) ? 'error'
             : /warn|suspicious/i.test(line) ? 'warn' : 'info';
  return { ts: Date.now() - i * 1000, source: 'activity', level: lvl, text: String(line) };
}

function _mergeLogs(events, actLines) {
  const evObjs = (events || []).map(ev => ({
    ts:     ev.ts || ev.timestamp || Date.now(),
    source: _sourceOf(ev),
    level:  _levelOf(ev),
    text:   ev.message || ev.msg || ev.text || JSON.stringify(ev),
  }));
  const actObjs = (actLines || []).map((l, i) => _wrapActivityLine(l, i));
  const merged  = [...evObjs, ...actObjs];
  merged.sort((a, b) => b.ts - a.ts);
  _allLogs = merged.slice(0, MAX_LOGS);
}

/* ── Rendering ───────────────────────────────────────────── */
const LEVEL_CLASS  = { error: 'ev-block', warn: 'ev-warn', info: 'ev-info' };
const SOURCE_TAG   = { security: 'SEC', redteam: 'RT', analyst: 'ANALYST', activity: 'ACT' };
const SOURCE_COLOR = {
  security: 'var(--red)', redteam: 'var(--magenta)',
  analyst: 'var(--purple)', activity: 'var(--cyan)',
};

function _renderLogs() {
  const feed    = document.getElementById('ctrl-activity-feed');
  const countEl = document.getElementById('ctrl-log-count');
  if (!feed) return;

  let logs = _allLogs;
  if (_activeSource !== 'all') logs = logs.filter(l => l.source === _activeSource);
  if (_activeLevel  !== 'all') logs = logs.filter(l => l.level  === _activeLevel);
  if (_searchText)             logs = logs.filter(l => l.text.toLowerCase().includes(_searchText));

  if (countEl) countEl.textContent = String(logs.length);

  if (!logs.length) {
    feed.innerHTML = '<div class="term-event ev-info"><span class="term-msg">No entries match filter</span></div>';
    return;
  }

  feed.innerHTML = logs.slice(0, 80).map(l => {
    const tag   = SOURCE_TAG[l.source]   || 'LOG';
    const color = SOURCE_COLOR[l.source] || 'var(--text-muted)';
    const time  = l.ts ? new Date(l.ts).toLocaleTimeString([], { hour12: false }) : '';
    return `<div class="term-event ${LEVEL_CLASS[l.level] || 'ev-info'}">` +
      `<span class="ctrl-log-ts">${esc(time)}</span>` +
      `<span class="ctrl-log-src" style="color:${color}">[${tag}]</span>` +
      `<span class="term-msg">${esc(l.text)}</span>` +
      `</div>`;
  }).join('');
}

function _svcStateClass(svc) {
  if (!svc)                    return 'ctrl-svc-offline';
  if (svc.status === 'paused') return 'ctrl-svc-paused';
  if (svc.running)             return 'ctrl-svc-online';
  return 'ctrl-svc-offline';
}

function _ledClass(svc) {
  if (!svc)                    return 'ctrl-led ctrl-led-off';
  if (svc.status === 'paused') return 'ctrl-led ctrl-led-pause';
  if (svc.running)             return 'ctrl-led ctrl-led-on';
  return 'ctrl-led ctrl-led-off';
}

function _renderProcessMatrix() {
  const table   = document.getElementById('ctrl-proc-table');
  const countEl = document.getElementById('ctrl-proc-online-count');
  if (!table) return;

  const SVC_DEFS = [
    { key: 'sancta',  label: 'Sancta Agent',      canStop: true, canPause: true,  canResume: true  },
    { key: 'curiosity', label: 'Curiosity Pipeline', canStop: true, canPause: false, canResume: false },
    { key: 'phenomenology', label: 'Phenomenology', canStop: true, canPause: false, canResume: false },
    { key: 'ollama',  label: 'Ollama LLM',         canStop: true, canPause: false, canResume: false },
    { key: 'siem',    label: 'SIEM Server',        canStop: false, canPause: false, canResume: false },
  ];

  let onlineCount = 0;
  const rows = SVC_DEFS.map(def => {
    const svc = _services[def.key];
    if (svc?.running) onlineCount++;
    const stateLabel = !svc ? 'OFFLINE'
      : svc.status === 'paused' ? 'PAUSED'
      : svc.running ? 'ONLINE' : 'OFFLINE';
    const pid = svc?.pid != null ? `PID ${svc.pid}` : '—';

    /* API stoppable overrides static defs (keeps SIEM matrix aligned with sancta_launcher) */
    const allowPause = def.canPause && (svc?.pauseable !== false);
    const allowResume = def.canResume && (svc?.pauseable !== false);
    const allowStop = def.canStop && (svc?.stoppable !== false);
    const showStop = allowStop || svc?.stoppable === true;

    const btns = [];
    if (allowPause)  btns.push(`<button class="ctrl-svc-btn btn-xs" data-svc="${def.key}" data-action="pause">Pause</button>`);
    if (allowResume) btns.push(`<button class="ctrl-svc-btn btn-xs" data-svc="${def.key}" data-action="resume">Resume</button>`);
    if (showStop)    btns.push(`<button class="ctrl-svc-btn btn-xs btn-xs-danger" data-svc="${def.key}" data-action="stop">Stop</button>`);

    return `<div class="ctrl-proc-row">` +
      `<span class="${_ledClass(svc)}"></span>` +
      `<span class="ctrl-svc-name">${esc(def.label)}</span>` +
      `<span class="ctrl-svc-pid">${pid}</span>` +
      `<span class="ctrl-svc-state ${_svcStateClass(svc)}">${stateLabel}</span>` +
      `<span class="ctrl-svc-btns">${btns.join('')}</span>` +
      `</div>`;
  });

  table.innerHTML = rows.join('');
  if (countEl) countEl.textContent = String(onlineCount);
}

/* ── Data Fetching ───────────────────────────────────────── */
async function _fetchLogs() {
  try {
    const [evRes, actRes] = await Promise.allSettled([
      api.fetchLiveEvents(),
      api.fetchActivity(),
    ]);
    const events   = evRes.status   === 'fulfilled' ? (evRes.value?.events  || evRes.value?.data  || []) : [];
    const actLines = actRes.status  === 'fulfilled' ? (actRes.value?.lines  || actRes.value?.data || S.activityLines || []) : (S.activityLines || []);
    _mergeLogs(events, actLines);
    _renderLogs();
  } catch (_) { /* silent — use cached */ }
}

async function _fetchServices() {
  try {
    const data = await api.fetchServicesStatus();
    if (data?.services) {
      _services = { ...data.services };
    }
  } catch (_) { /* fallback to S state */ }

  /* Merge UI agent state with API (launcher + SIEM both write .agent.pid) */
  _services.sancta = _services.sancta || {};
  const apiRun = !!_services.sancta.running;
  const uiRun = !!S.agentRunning;
  _services.sancta.running = apiRun || uiRun;
  _services.sancta.status = S.agentSuspended ? 'paused' : (_services.sancta.running ? 'running' : 'stopped');
  _services.sancta.pid = _services.sancta.pid ?? S.agentPid;

  _renderProcessMatrix();
}

async function _handleSvcAction(svcKey, action) {
  try {
    let result;
    if (svcKey === 'sancta') {
      if (action === 'pause')  result = await api.pauseAgent();
      if (action === 'resume') result = await api.resumeAgent();
      if (action === 'stop')   result = await api.killAgent();
    } else if (action === 'stop') {
      if (svcKey === 'ollama' && !confirm('Stop Ollama? This kills the process listening on :11434 (same as launcher STOP).')) return;
      result = await api.stopService(svcKey);
    }
    if (result?.ok === false && typeof window.showToast === 'function') {
      window.showToast(result.error || 'Action failed', 'error');
    }
    await _fetchServices();
  } catch (e) {
    if (typeof window.showToast === 'function') window.showToast(e.message || String(e), 'error');
  }
}

function _poll() {
  _fetchServices();
  _fetchLogs();
}

/* ── Exports ─────────────────────────────────────────────── */
export function init() {
  /* Lifecycle buttons */
  const ctrlModeSelect = document.getElementById('ctrl-mode-select');
  const getMode = () => (ctrlModeSelect?.value || 'passive').trim() || 'passive';

  const runAction = async (fn) => {
    try {
      const data = await fn();
      if (data?.ok === false) {
        const msg = data?.error ?? data?.detail ?? 'Action failed';
        if (typeof window.showToast === 'function') window.showToast(msg, 'error');
      }
      refresh();
    } catch (e) {
      if (typeof window.showToast === 'function') window.showToast(e?.message || String(e), 'error');
    }
  };

  document.getElementById('ctrl-start')?.addEventListener('click',   () => runAction(() => api.startAgent(getMode())));
  document.getElementById('ctrl-pause')?.addEventListener('click',   () => runAction(api.pauseAgent));
  document.getElementById('ctrl-resume')?.addEventListener('click',  () => runAction(api.resumeAgent));
  document.getElementById('ctrl-stop')?.addEventListener('click',    () => {
    if (S.agentRunning && !confirm('Stop the Sancta agent? This will terminate the process.')) return;
    runAction(api.killAgent);
  });
  document.getElementById('ctrl-restart')?.addEventListener('click', () => {
    if (S.agentRunning && !confirm('Restart the agent? This will stop and start it again.')) return;
    runAction(() => api.restartAgent(getMode()));
  });

  /* Log source tabs */
  document.getElementById('ctrl-log-tabs')?.addEventListener('click', e => {
    const btn = e.target.closest('.ctrl-log-tab');
    if (!btn) return;
    _activeSource = btn.dataset.src || 'all';
    document.querySelectorAll('.ctrl-log-tab').forEach(b => b.classList.toggle('active', b === btn));
    _renderLogs();
  });

  /* Level filter */
  document.getElementById('ctrl-log-level')?.addEventListener('change', e => {
    _activeLevel = e.target.value;
    _renderLogs();
  });

  /* Text search with 200ms debounce */
  let searchDebounce;
  document.getElementById('ctrl-log-search')?.addEventListener('input', e => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => {
      _searchText = e.target.value.trim().toLowerCase();
      _renderLogs();
    }, 200);
  });

  /* Process matrix — event delegation for per-service buttons */
  document.getElementById('ctrl-proc-table')?.addEventListener('click', e => {
    const btn = e.target.closest('.ctrl-svc-btn');
    if (!btn) return;
    const { svc, action } = btn.dataset;
    if (svc && action) _handleSvcAction(svc, action);
  });

  /* Notification toggles */
  const notifToggle = document.getElementById('ctrl-notif-toggle');
  const soundToggle = document.getElementById('ctrl-sound-toggle');

  if (notifToggle) {
    notifToggle.checked = notifications.isEnabled();
    notifToggle.addEventListener('change', async () => {
      if (notifToggle.checked) {
        const perm = await notifications.requestPermission();
        if (perm !== 'granted') {
          notifToggle.checked = false;
          notifications.setEnabled(false);
          if (typeof window.showToast === 'function') window.showToast('Notification permission denied', 'error');
          return;
        }
      }
      notifications.setEnabled(notifToggle.checked);
    });
  }

  if (soundToggle) {
    soundToggle.checked = notifications.isSoundsEnabled();
    soundToggle.addEventListener('change', () => {
      notifications.setSoundsEnabled(soundToggle.checked);
    });
  }

  /* Initial load */
  _fetchServices();
  _fetchLogs();

  /* Poll every 5s */
  _pollTimer = setInterval(_poll, 5000);
}

export function refresh() {
  const running   = S.agentRunning && !S.agentSuspended;
  const suspended = S.agentRunning && S.agentSuspended;

  /* Lifecycle button states */
  const setDisabled = (id, val) => { const el = document.getElementById(id); if (el) el.disabled = val; };
  setDisabled('ctrl-start',   !!running);
  setDisabled('ctrl-pause',   !running || !!suspended);
  setDisabled('ctrl-resume',  !suspended);
  setDisabled('ctrl-stop',    !running);
  setDisabled('ctrl-restart', !running);

  /* Process info panel */
  const piStatus = document.getElementById('pi-status');
  if (piStatus) {
    piStatus.textContent = S.agentRunning ? (S.agentSuspended ? 'PAUSED' : 'ONLINE') : 'OFFLINE';
    piStatus.className = 'proc-val ' + (
      S.agentRunning && !S.agentSuspended ? 'pv-online' :
      S.agentSuspended ? 'pv-paused' : 'pv-offline'
    );
  }
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val ?? '—'; };
  set('pi-pid',   S.agentPid);
  set('pi-cycle', S.cycle);
  set('pi-karma', S.karma);
  set('pi-mood',  S.mood);
  set('pi-model', S.modelInfo);

  /* Sync sancta row with live UI state + last API snapshot */
  _services.sancta = _services.sancta || {};
  _services.sancta.running = !!(_services.sancta.running || S.agentRunning);
  _services.sancta.status  = S.agentSuspended ? 'paused' : (_services.sancta.running ? 'running' : 'stopped');
  _services.sancta.pid     = _services.sancta.pid ?? S.agentPid;
  _renderProcessMatrix();

  /* Community stats */
  const ctrlCommunity = document.getElementById('ctrl-community');
  if (ctrlCommunity) {
    ctrlCommunity.innerHTML =
      `<div class="ks-item"><span class="ks-key">inner circle</span><span class="ks-val">${S.innerCircle ?? '—'}</span></div>` +
      `<div class="ks-item"><span class="ks-key">recruited</span><span class="ks-val">${S.recruited ?? '—'}</span></div>`;
  }

  const ctrlKnowledge = document.getElementById('ctrl-knowledge');
  if (ctrlKnowledge) {
    ctrlKnowledge.innerHTML =
      `<div class="ks-item"><span class="ks-key">entries</span><span class="ks-val">—</span></div>` +
      `<div class="ks-item"><span class="ks-key">patterns</span><span class="ks-val">—</span></div>`;
  }
}

export function destroy() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}
