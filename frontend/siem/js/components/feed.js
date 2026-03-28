/* ─── Shared Terminal Feed Renderer ──────────────────────── */
import { evSeverity, EVENTS_BUFFER_MAX } from '../state.js';

const MAX_ROWS = EVENTS_BUFFER_MAX;

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmtTs(raw) {
  if (!raw) return '--:--:--';
  const s = String(raw).trim();
  if (/^\d{2}:\d{2}:\d{2}/.test(s)) return s.slice(0, 8);
  try {
    const d = new Date(s);
    if (!Number.isNaN(d.getTime())) {
      return d.toLocaleString(undefined, {
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
      });
    }
  } catch (_) { /* ignore */ }
  return s.slice(-14);
}

function getEventTag(ev) {
  const e = String(ev.event || ev.type || '').toLowerCase();
  if (/suspicious_block|injection/.test(e))  return 'BLOCK';
  if (/llm_deep_scan/.test(e))               return 'LLM';
  if (/injection_defense|tavern_defense|defend/.test(e)) return 'DEF';
  if (/red_team|redteam/.test(e))             return 'RT';
  if (/mood|analyst|belief/.test(e))           return 'ANALYST';
  if (/epidemic|seir|infect/.test(e))         return 'EPID';
  if (/learn|epistemic|behavioral/.test(e))    return 'LEARN';
  if (/activity|cycle|task/.test(e))          return 'ACT';
  if (/analyst|behavior/.test(e))             return 'ANALYST';
  if (/warn|anomal|drift/.test(e))            return 'WARN';
  if (/ok|pass|clean/.test(e))                return 'OK';
  const src = String(ev.source || '').toUpperCase();
  return src || 'INFO';
}

function tagClass(tag) {
  const t = tag.toLowerCase();
  if (/block|rt/.test(t))      return 'tag-block';
  if (/warn|anomal/.test(t))   return 'tag-warn';
  if (/def|ok/.test(t))        return 'tag-ok';
  if (/analyst/.test(t))       return 'tag-analyst';
  if (/llm/.test(t))           return 'tag-llm';
  if (/epid/.test(t))          return 'tag-info';
  if (/learn/.test(t))         return 'tag-info';
  return 'tag-default';
}

function getMsg(ev) {
  // Prefer human-readable message fields
  const d = ev.data || ev.details || {};
  return (
    ev.message || ev.msg || ev.content ||
    d.message || d.reason || d.verdict ||
    ev.event || ev.type || '—'
  );
}

/** Build a single .term-event row HTML string */
export function makeEventRow(ev) {
  const ts  = fmtTs(ev.ts || ev.timestamp || '');
  const tag = getEventTag(ev);
  const msg = getMsg(ev);
  const sev = evSeverity(ev);

  return `<div class="term-event ev-${sev}">` +
    `<span class="term-ts">${escHtml(ts)}</span>` +
    `<span class="term-tag ${tagClass(tag)}">${escHtml(tag)}</span>` +
    `<span class="term-msg">${escHtml(msg)}</span>` +
    `</div>`;
}

/** Prepend a single event row to a feed element (live update) */
export function prependToFeed(feedEl, ev) {
  if (!feedEl) return;
  const row = document.createElement('div');
  row.className = 'term-event ev-' + evSeverity(ev);
  const ts  = fmtTs(ev.ts || ev.timestamp || '');
  const tag = getEventTag(ev);
  const msg = getMsg(ev);
  row.innerHTML =
    `<span class="term-ts">${escHtml(ts)}</span>` +
    `<span class="term-tag ${tagClass(tag)}">${escHtml(tag)}</span>` +
    `<span class="term-msg">${escHtml(msg)}</span>`;
  feedEl.prepend(row);
  // Trim to MAX_ROWS
  while (feedEl.children.length > MAX_ROWS) {
    feedEl.removeChild(feedEl.lastChild);
  }
}

/** Bulk-render an array of events into a feed element */
export function renderFeed(feedEl, events) {
  if (!feedEl) return;
  feedEl.innerHTML = events.slice(0, MAX_ROWS).map(makeEventRow).join('');
}

/** Filter events by source/type predicates */
export function filterEvents(events, predicate) {
  return events.filter(predicate);
}
