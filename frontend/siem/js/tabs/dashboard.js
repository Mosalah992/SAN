/* ─── Dashboard Tab ───────────────────────────────────────── */
import { S, evSeverity, EVENTS_BUFFER_MAX } from '../state.js';

const FEED_LIMIT = EVENTS_BUFFER_MAX;

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function tagClass(sev) {
  if (sev === 'block' || sev === 'warn') return 'tag-block';
  if (sev === 'ok') return 'tag-ok';
  if (sev === 'analyst') return 'tag-analyst';
  return 'tag-info';
}

function evClass(sev) {
  if (sev === 'block') return 'ev-block';
  if (sev === 'warn') return 'ev-warn';
  if (sev === 'ok') return 'ev-ok';
  if (sev === 'analyst') return 'ev-analyst';
  return 'ev-info';
}

function formatTs(ts) {
  if (!ts) return '';
  const s = String(ts).trim();
  try {
    const d = new Date(s);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleString(undefined, {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
    }
  } catch (_) { /* fall through */ }
  return s.replace('T', ' ').slice(0, 19);
}

function _safeStr(v) {
  if (v == null) return '';
  return typeof v === 'string' ? v : (typeof v === 'object' ? '' : String(v));
}

/** Unwrap WS event wrapper { type, event } to inner payload */
function _inner(ev) {
  return (ev && typeof ev.event === 'object') ? ev.event : ev;
}

function renderEvent(ev) {
  if (!ev || typeof ev !== 'object') return '';
  const inner = _inner(ev);
  const sev = evSeverity(ev);
  const ts = formatTs(ev.ts ?? inner?.ts ?? ev.timestamp ?? inner?.timestamp);
  const tag = _safeStr(inner?.source ?? ev.source ?? inner?.event ?? ev.event ?? 'event').toUpperCase().slice(0, 8);
  const msg = _safeStr(inner?.message ?? ev.message ?? inner?.preview ?? ev.preview ?? inner?.event ?? ev.event);
  return `<div class="term-event ${evClass(sev)}">
    <span class="term-ts">${esc(ts)}</span>
    <span class="term-tag ${tagClass(sev)}">${esc(tag)}</span>
    <span class="term-msg">${esc(msg)}</span>
  </div>`;
}

export function init() {
  // Handled by app.js
}

/** Append a single event to feeds (live stream). Newest at top. */
export function appendEvent(ev) {
  const html = renderEvent(ev);
  if (!html) return;
  const feedActivity = document.getElementById('feed-activity');
  const feedThreats = document.getElementById('feed-threats');
  const activityCount = document.getElementById('activity-count');
  const threatCount = document.getElementById('threat-count');

  if (feedActivity) {
    const empty = feedActivity.querySelector('.term-event.ev-info .term-msg');
    if (empty && empty.textContent === 'No activity yet') feedActivity.innerHTML = '';
    const div = document.createElement('div');
    div.innerHTML = html;
    const node = div.firstElementChild;
    if (node) feedActivity.insertBefore(node, feedActivity.firstChild);
    _trimFeed(feedActivity, FEED_LIMIT);
  }
  if (evSeverity(ev) === 'block' || evSeverity(ev) === 'warn') {
    if (feedThreats) {
      const empty = feedThreats.querySelector('.term-event.ev-info .term-msg');
      if (empty && empty.textContent === 'No threats') feedThreats.innerHTML = '';
      const div = document.createElement('div');
      div.innerHTML = html;
      const node = div.firstElementChild;
      if (node) feedThreats.insertBefore(node, feedThreats.firstChild);
      _trimFeed(feedThreats, FEED_LIMIT);
    }
  }
  _updateFeedCounts(activityCount, threatCount);
}

function _trimFeed(el, max) {
  if (!el) return;
  while (el.children.length > max) el.removeChild(el.lastChild);
}

function _updateFeedCounts(activityCount, threatCount) {
  const all = S.events || [];
  const threats = all.filter(e => evSeverity(e) === 'block' || evSeverity(e) === 'warn');
  if (activityCount) activityCount.textContent = String(all.length);
  if (threatCount) threatCount.textContent = String(threats.length);
}

/** Full sync of feeds (on bulk load). Stats + behavioral metrics always updated. */
export function refresh(opts = {}) {
  const { syncFeeds = false } = typeof opts === 'object' ? opts : {};
  const feedActivity = document.getElementById('feed-activity');
  const feedThreats = document.getElementById('feed-threats');
  const activityCount = document.getElementById('activity-count');
  const threatCount = document.getElementById('threat-count');

  const scAgentVal = document.getElementById('sc-agent-val');
  const scAgentSub = document.getElementById('sc-agent-sub');
  const scSecVal = document.getElementById('sc-sec-val');
  const scKarmaVal = document.getElementById('sc-karma-val');
  const scKarmaSub = document.getElementById('sc-karma-sub');
  const scSeirVal = document.getElementById('sc-seir-val');
  const scSeirSub = document.getElementById('sc-seir-sub');

  const esCoherence = document.getElementById('es-coherence');
  const esDissonance = document.getElementById('es-dissonance');
  const esCuriosity = document.getElementById('es-curiosity');
  const esConfidence = document.getElementById('es-confidence');
  const esMoodText = document.getElementById('es-mood-text');

  // Stat cards
  if (scAgentVal) {
    if (S.agentRunning) {
      scAgentVal.textContent = S.agentSuspended ? 'PAUSED' : 'RUNNING';
      scAgentVal.className = 'sc-num num-online';
    } else {
      scAgentVal.textContent = 'OFFLINE';
      scAgentVal.className = 'sc-num num-offline';
    }
  }
  if (scAgentSub) scAgentSub.textContent = `cycle ${S.cycle} · karma ${S.karma}`;
  if (scSecVal) scSecVal.textContent = String(S.injections ?? 0);
  if (scKarmaVal) scKarmaVal.textContent = S.karma ?? '—';
  if (scKarmaSub) scKarmaSub.textContent = `inner circle ${S.innerCircle ?? '—'}`;
  if (scSeirVal) {
    const seirLower = (S.seir || '').toLowerCase();
    scSeirVal.textContent = (S.seir || 'SUSCEPTIBLE').toUpperCase();
    if (seirLower === 'compromised') scSeirVal.className = 'sc-num seir-c';
    else if (seirLower === 'infected' || seirLower === 'exposed') scSeirVal.className = 'sc-num seir-i';
    else if (seirLower === 'recovered') scSeirVal.className = 'sc-num seir-r';
    else scSeirVal.className = 'sc-num seir-s';
  }
  if (scSeirSub) scSeirSub.textContent = `drift ${(S.driftScore ?? 0).toFixed(3)} · ${S.alertLevel ?? 'clear'}`;

  // Behavioral strip
  const ep = S.epistemic || {};
  if (esCoherence) esCoherence.style.width = `${Math.round((ep.coherence ?? 0.7) * 100)}%`;
  if (esDissonance) esDissonance.style.width = `${Math.round((ep.dissonance ?? 0.1) * 100)}%`;
  if (esCuriosity) esCuriosity.style.width = `${Math.round((ep.curiosity ?? 0.6) * 100)}%`;
  if (esConfidence) esConfidence.style.width = `${Math.round((ep.confidence ?? 0.75) * 100)}%`;
  if (esMoodText) esMoodText.textContent = S.mood ?? '—';

  // Feeds — only full replace when syncFeeds (bulk load). Live stream uses appendEvent().
  if (syncFeeds) {
    const allEvents = S.events || [];
    const threats = allEvents.filter(e => evSeverity(e) === 'block' || evSeverity(e) === 'warn');
    const activity = allEvents.slice(0, FEED_LIMIT);
    const threatsDisplay = threats.slice(0, FEED_LIMIT);

    if (feedActivity) feedActivity.innerHTML = activity.map(renderEvent).join('') || '<div class="term-event ev-info"><span class="term-msg">No activity yet</span></div>';
    if (feedThreats) feedThreats.innerHTML = threatsDisplay.map(renderEvent).join('') || '<div class="term-event ev-info"><span class="term-msg">No threats</span></div>';
    if (activityCount) activityCount.textContent = String(activity.length);
    if (threatCount) threatCount.textContent = String(threatsDisplay.length);
  } else {
    _updateFeedCounts(activityCount, threatCount);
  }
}
