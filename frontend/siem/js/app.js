/* ─── SIEM App Entry ─────────────────────────────────────── */
import { S, pushEvent, evSeverity, EVENTS_BUFFER_MAX } from './state.js';
import * as api from './api.js';
import { runBoot } from './boot.js';
import { connectWS, disconnectWS } from './websocket.js';
import * as dashboard from './tabs/dashboard.js';
import * as security from './tabs/security.js';
import * as analyst from './tabs/analyst.js';
import * as chat from './tabs/chat.js';
import * as lab from './tabs/lab.js';
import * as epidemic from './tabs/epidemic.js';
import * as profiles from './tabs/profiles.js';
import * as knowledge from './tabs/knowledge.js';
import * as control from './tabs/control.js';
import * as notifications from './components/notifications.js';

const AUTH_STORAGE_KEY = 'siem_auth_token';
const TAB_MODULES = { dashboard, security, analyst, chat, lab, epidemic, profiles, knowledge, control };

function showToast(msg, type = 'info') {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg || '';
  el.className = 'toast toast-' + (type === 'error' ? 'error' : type === 'success' ? 'success' : '');
  el.classList.remove('hidden');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => { el.classList.add('hidden'); }, 4000);
}
if (typeof window !== 'undefined') window.showToast = showToast;

function getToken() {
  return sessionStorage.getItem(AUTH_STORAGE_KEY) || '';
}

function setToken(t) {
  if (t) sessionStorage.setItem(AUTH_STORAGE_KEY, t);
  else sessionStorage.removeItem(AUTH_STORAGE_KEY);
}

function showAuthModal() {
  document.getElementById('auth-modal')?.classList.remove('hidden');
  document.getElementById('auth-input')?.focus();
}

function hideAuthModal() {
  document.getElementById('auth-modal')?.classList.add('hidden');
}

function setAuthError(msg) {
  const el = document.getElementById('auth-error');
  if (!el) return;
  el.textContent = msg || '';
  el.classList.toggle('hidden', !msg);
}

async function applyStatus(data) {
  if (!data) return;
  const m = data.metrics || data;
  // agent running/pid/suspended lives in data.agent (not data.metrics)
  const ag = (data.agent && typeof data.agent === 'object') ? data.agent : {};
  S.cycle = m.cycle_count ?? m.cycle ?? S.cycle;
  S.karma = m.current_karma ?? (Array.isArray(m.karma_history) && m.karma_history.length ? m.karma_history[m.karma_history.length - 1] : S.karma);
  S.mood = m.agent_mood ?? m.mood ?? S.mood;
  S.innerCircle = m.inner_circle ?? m.inner_circle_count ?? S.innerCircle;
  S.recruited = m.recruited ?? m.recruited_count ?? S.recruited;
  // Prefer data.agent.running over metrics.running (metrics doesn't have it)
  S.agentRunning = ('running' in ag) ? !!ag.running : (!!m.running || S.agentRunning);
  S.agentPid = ag.pid ?? m.pid ?? S.agentPid;
  S.agentSuspended = ('suspended' in ag) ? !!ag.suspended : !!m.suspended;
  S.modelInfo = m.model_info ?? data.model ?? S.modelInfo;
  S.injections = m.injection_attempts_detected ?? m.injections ?? S.injections;
  S.defenseRate = m.defense_rate ?? S.defenseRate;
  S.fpRate = m.false_positive_rate ?? S.fpRate;
  if (data.seir) S.seir = data.seir.health_state || data.seir;
  if (typeof data.drift_score === 'number') S.driftScore = data.drift_score;
  if (data.alert_level) S.alertLevel = data.alert_level;
  if (data.epistemic && typeof data.epistemic === 'object') Object.assign(S.epistemic, data.epistemic);
  if (data.beliefs) S.beliefs = data.beliefs;
  if (data.journal_entries) S.journalEntries = data.journal_entries;
  refreshAll();
}

let _refreshScheduled = false;
let _lastRefreshAt = 0;
let _refreshDebounce = null;
const REFRESH_MIN_MS = 800;

function refreshAll(opts = {}) {
  const { syncFeeds = false, adversary } = (typeof opts === 'object' && opts) ? opts : {};
  const now = Date.now();
  if (!syncFeeds && now - _lastRefreshAt < REFRESH_MIN_MS) {
    if (!_refreshDebounce) {
      _refreshDebounce = setTimeout(() => {
        _refreshDebounce = null;
        refreshAll();
      }, REFRESH_MIN_MS - (now - _lastRefreshAt));
    }
    return;
  }
  if (_refreshDebounce) {
    clearTimeout(_refreshDebounce);
    _refreshDebounce = null;
  }
  if (_refreshScheduled && !syncFeeds) return;
  _refreshScheduled = true;
  _lastRefreshAt = now;
  requestAnimationFrame(() => {
    _refreshScheduled = false;
    const tab = S.activeTab || 'dashboard';
    // Pass context data to active tab
    if (tab === 'security') {
      security.refresh(adversary || S._adversaryData);
    } else if (tab === 'epidemic') {
      epidemic.refresh();
    } else {
      const mod = TAB_MODULES[tab];
      if (mod?.refresh) mod.refresh();
    }
    // Always refresh dashboard stats + control process info + bottom bar
    dashboard.refresh({ syncFeeds });
    control.refresh();
    updateBottombar();
    updateBadges();
  });
}

function updateBottombar() {
  const bbStatus = document.getElementById('bb-status');
  const bbCycle = document.getElementById('bb-cycle');
  const bbKarma = document.getElementById('bb-karma');
  const bbMood = document.getElementById('bb-mood');
  const bbThreats = document.getElementById('bb-threats');
  const bbSeir = document.getElementById('bb-seir');

  if (bbStatus) {
    const online = S.agentRunning && !S.agentSuspended;
    bbStatus.textContent = online ? '⬤ ONLINE' : '⬤ OFFLINE';
    bbStatus.className = 'bb-item ' + (online ? 'online' : 'offline');
  }
  if (bbCycle) bbCycle.textContent = `cycle ${S.cycle}`;
  if (bbKarma) bbKarma.textContent = `karma ${S.karma}`;
  if (bbMood) bbMood.textContent = S.mood ?? '—';
  const threatCount = (S.events || []).filter(e => evSeverity(e) === 'block' || evSeverity(e) === 'warn').length;
  if (bbThreats) {
    bbThreats.textContent = `threats ${threatCount}`;
    bbThreats.classList.toggle('has-threats', threatCount > 0);
  }
  if (bbSeir) {
    bbSeir.textContent = `SEIR ${(S.seir || 'SUSCEPTIBLE').toUpperCase()}`;
    const seirL = (S.seir || '').toLowerCase();
    bbSeir.className = 'bb-item ' + (seirL === 'compromised' ? 'compromised' : (seirL === 'infected' || seirL === 'exposed') ? 'infected' : '');
  }
}

function updateBadges() {
  const badgeModel = document.getElementById('badge-model');
  const badgeMood = document.getElementById('badge-mood');
  const badgeAgent = document.getElementById('badge-agent');

  if (badgeModel) badgeModel.textContent = S.modelInfo || '—';
  if (badgeMood) badgeMood.textContent = S.mood || '—';
  if (badgeAgent) {
    badgeAgent.textContent = S.agentRunning && !S.agentSuspended ? 'ONLINE' : 'OFFLINE';
    badgeAgent.className = 'badge ' + (S.agentRunning && !S.agentSuspended ? 'badge-online' : 'badge-offline');
  }
}

function setTab(tab) {
  S.activeTab = tab;
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.dataset.tab === tab);
  });
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  const mod = TAB_MODULES[tab];
  if (mod?.refresh) mod.refresh();
}

function onEvent(ev) {
  pushEvent(ev);
  dashboard.appendEvent(ev);
  updateBottombar();
  if (S.activeTab === 'epidemic' && epidemic.onLiveEvent) {
    epidemic.onLiveEvent(ev);
  }
  // Critical event notifications
  const sev = evSeverity(ev);
  const inner = (ev && typeof ev.event === 'object') ? ev.event : ev;
  const msg = inner?.message || inner?.preview || inner?.event || '';
  if (sev === 'block') {
    notifications.notify('Threat Blocked', msg.slice(0, 120), 'critical');
    notifications.toastAlert('BLOCKED: ' + msg.slice(0, 80), 'critical');
  } else if (sev === 'warn' && /quarantine|infected|compromised/i.test(msg)) {
    notifications.notify('Security Warning', msg.slice(0, 120), 'warning');
    notifications.toastAlert(msg.slice(0, 80), 'warning');
  }
}

function onMetrics(msg) {
  if (msg.agent && typeof msg.agent === 'object') {
    if ('running' in msg.agent) S.agentRunning = !!msg.agent.running;
    if ('pid' in msg.agent) S.agentPid = msg.agent.pid;
    if ('suspended' in msg.agent) S.agentSuspended = !!msg.agent.suspended;
  }
  if (msg.metrics) applyStatus({ metrics: msg.metrics });
  else applyStatus(msg);
}

async function bootstrap() {
  S.token = getToken();

  const authRequired = await api.checkAuthRequired().then(r => r.auth_required).catch(() => false);
  S.authRequired = !!authRequired;

  if (authRequired && !S.token) {
    showAuthModal();
    return;
  }

  if (S.token) {
    try {
      const v = await api.verifyToken(S.token);
      if (!v?.ok) { setToken(''); showAuthModal(); setAuthError('Invalid token'); return; }
    } catch (_) {
      setToken(''); showAuthModal(); setAuthError('Verification failed');
      return;
    }
  }

  document.getElementById('auth-submit')?.addEventListener('click', async () => {
    const input = document.getElementById('auth-input');
    const token = (input?.value || '').trim();
    if (!token) { setAuthError('Enter a token'); return; }
    setAuthError('');
    try {
      const v = await api.verifyToken(token);
      if (v?.ok) {
        setToken(token);
        S.token = token;
        hideAuthModal();
        bootstrap();
      } else {
        setAuthError('Invalid token');
      }
    } catch (e) {
      setAuthError(String(e?.message || 'Verification failed'));
    }
  });

  document.getElementById('auth-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') document.getElementById('auth-submit')?.click();
  });

  const bootDynamic = {};
  try {
    const status = await api.fetchStatus();
    applyStatus(status);
    bootDynamic.agent = !!(status?.agent?.running);
    bootDynamic.ws = !status?.ws_safe_mode;
  } catch (_) {
    bootDynamic.ws = true;
  }
  if (bootDynamic.ws === undefined) bootDynamic.ws = true;

  await runBoot(bootDynamic);

  document.getElementById('boot-screen')?.classList.add('fading');
  document.getElementById('app')?.classList.remove('hidden');

  TAB_MODULES.dashboard.init();
  TAB_MODULES.security.init();
  TAB_MODULES.analyst.init();
  chat.init();
  lab.init();
  epidemic.init();
  profiles.init();
  knowledge.init();
  control.init();
  notifications.init();

  const statusData = await api.fetchStatus().catch(() => null);
  if (statusData?.ws_safe_mode) {
    S.wsConnected = false;
    // In WS_SAFE_MODE, live events come via HTTP polling only.
    // We do NOT replace the entire buffer every 4s — instead, diff
    // and prepend only genuinely new events to avoid DOM thrash.
    let _lastSeenTs = '';
    const pollEvents = () => {
      api.fetchLiveEvents().then(d => {
        if (!d?.events?.length) return;
        const incoming = d.events.slice(0, EVENTS_BUFFER_MAX);
        if (!S.events.length) {
          // First load: replace buffer, full render
          S.events = incoming;
          dashboard.refresh({ syncFeeds: true });
        } else {
          // Subsequent polls: find new events by timestamp and prepend
          const newEvents = [];
          for (const ev of incoming) {
            const ts = ev.ts || ev.timestamp || '';
            if (ts && ts > _lastSeenTs) {
              newEvents.push(ev);
            }
          }
          if (newEvents.length > 0 && newEvents.length < incoming.length) {
            // Prepend only new events (live append, no DOM rebuild)
            for (let i = newEvents.length - 1; i >= 0; i--) {
              S.events.unshift(newEvents[i]);
              if (S.activeTab === 'dashboard') dashboard.appendEvent(newEvents[i]);
            }
            // Trim buffer
            if (S.events.length > EVENTS_BUFFER_MAX) S.events.length = EVENTS_BUFFER_MAX;
          } else if (newEvents.length === incoming.length) {
            // All events are new (rare: buffer was stale) — full replace
            S.events = incoming;
            dashboard.refresh({ syncFeeds: true });
          }
          // else: no new events, do nothing (no flicker)
        }
        if (incoming[0]) _lastSeenTs = incoming[0].ts || incoming[0].timestamp || _lastSeenTs;
        updateBottombar();
        updateBadges();
      }).catch(() => {});
    };
    pollEvents();
    setInterval(pollEvents, 4000);
  } else {
    connectWS(onEvent, onMetrics);
  }

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });

  window.addEventListener('keydown', e => {
    if (e.target?.closest('input, textarea, select')) return;
    if (e.key === 'Tab') { e.preventDefault(); cycleTab(); }
    if (e.key === 'r' || e.key === 'R') { e.preventDefault(); fetchAll(); }
    if (e.key === '/') { e.preventDefault(); setTab('chat'); document.getElementById('chat-input')?.focus(); }
  });

  const tabs = ['dashboard', 'security', 'analyst', 'chat', 'lab', 'epidemic', 'profiles', 'knowledge', 'control'];
  function cycleTab() {
    const i = tabs.indexOf(S.activeTab);
    setTab(tabs[(i + 1) % tabs.length]);
  }

  async function fetchAll() {
    try {
      const results = await Promise.allSettled([
        api.fetchStatus(),
        api.fetchLiveEvents(),
        api.fetchActivity(),
        api.fetchEpistemic(),
        api.fetchEpidemicStatus().catch(() => null),
        api.fetchEpidemicSim().catch(() => null),
        api.fetchSecAdversary().catch(() => null),
        api.fetchModelInfo().catch(() => null),
      ]);
      const [status, events, activity, epistemic, epidStatus, epidSim, adversary, modelInfo] = results.map(r => r.status === 'fulfilled' ? r.value : null);
      // Update model info
      if (modelInfo?.model) S.modelInfo = modelInfo.model + (modelInfo.status === 'connected' ? '' : ' (offline)');
      applyStatus(status);
      // Merge events from API — only replace on first load; subsequent polls handled by pollEvents diff
      if (events?.events) {
        if (!S.events.length) {
          S.events = events.events.slice(0, EVENTS_BUFFER_MAX);
        }
        // Don't replace on subsequent fetchAll — pollEvents handles live append
      }
      if (activity?.lines) S.activityLines = activity.lines;
      if (epistemic?.epistemic) {
        const e = epistemic.epistemic;
        S.epistemic.confidence = e.confidence_score ?? S.epistemic.confidence;
        S.epistemic.dissonance = (e.uncertainty_entropy != null) ? Math.min(1, e.uncertainty_entropy) : S.epistemic.dissonance;
        S.epistemic.curiosity  = e.anthropomorphism_index ?? S.epistemic.curiosity;
        S.epistemic.coherence  = (e.confidence_score != null) ? e.confidence_score : S.epistemic.coherence;
      }
      if (epidStatus?.ok) {
        S.seir        = epidStatus.seir?.health_state ?? epidStatus.health_state ?? S.seir;
        S.driftScore  = epidStatus.score ?? epidStatus.drift_score ?? S.driftScore;
        S.alertLevel  = epidStatus.alert_level ?? S.alertLevel;
        S.epidSignals = epidStatus.signals ?? S.epidSignals;
        S.epidParams  = epidStatus.params ?? epidStatus.epidemic_params ?? S.epidParams;
        if (epidStatus.epidemic_report) S.epidReport = epidStatus.epidemic_report;
      }
      if (epidSim?.available && epidSim?.data) S.epidSimData = epidSim.data;
      // Cache adversary data for security tab
      if (adversary?.ok) S._adversaryData = adversary;
      refreshAll({ syncFeeds: true, adversary });
    } catch (_) {}
  }

  await fetchAll();
  setInterval(fetchAll, 6000);

  async function fetchAgentActivity() {
    try {
      const activity = await api.fetchActivity();
      if (activity?.lines) {
        S.activityLines = activity.lines;
        refreshAll();
      }
    } catch (_) {}
    setTimeout(fetchAgentActivity, 4000);
  }
  setTimeout(fetchAgentActivity, 1500);
}

bootstrap();
