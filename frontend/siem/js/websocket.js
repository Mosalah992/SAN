/* ─── WebSocket + Polling Fallback ───────────────────────── */
import { S, pushEvent } from './state.js';

let _ws = null;
let _pollTimer = null;
let _onEvent = null;
let _onMetrics = null;
let _reconnectDelay = 2000;
const MAX_RECONNECT = 30000;

export function connectWS(onEvent, onMetrics) {
  _onEvent   = onEvent;
  _onMetrics = onMetrics;
  _tryConnect();
}

function _tryConnect() {
  if (_ws) { try { _ws.close(); } catch (_) {} }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url   = `${proto}://${location.host}/ws/live${S.token ? '?token=' + encodeURIComponent(S.token) : ''}`;

  try {
    _ws = new WebSocket(url);
  } catch (_) {
    _fallbackPoll();
    return;
  }

  _ws.onopen = () => {
    S.wsConnected = true;
    _reconnectDelay = 2000;
    _stopPoll();
  };

  _ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      _handleMessage(msg);
    } catch (_) {}
  };

  _ws.onerror = () => {
    S.wsConnected = false;
  };

  _ws.onclose = () => {
    S.wsConnected = false;
    _fallbackPoll();
    // Reconnect with backoff
    setTimeout(() => {
      _reconnectDelay = Math.min(_reconnectDelay * 1.5, MAX_RECONNECT);
      _tryConnect();
    }, _reconnectDelay);
  };
}

function _handleMessage(msg) {
  if (!msg || typeof msg !== 'object') return;
  // WS sends individual events or bulk metric snapshots (validate structure)
  if (msg.type === 'metrics' || (msg.metrics !== undefined && msg.metrics !== null)) {
    if (_onMetrics) _onMetrics(msg);
    return;
  }
  if (msg.agent !== undefined && typeof msg.agent === 'object' && (_onMetrics)) {
    _onMetrics(msg);
    return;
  }
  // Individual event — unwrap { type: "event", event: {...} } from backend
  const ev = msg.data ?? (msg.type === 'event' && msg.event ? msg.event : msg);
  pushEvent(ev);
  if (_onEvent) _onEvent(ev);
}

/* ── Polling fallback (when WS unavailable) ──────────────── */
let _lastPollEventTs = '';

function _fallbackPoll() {
  if (_pollTimer) return; // already polling
  _doPoll();
  _pollTimer = setInterval(_doPoll, 6000);
}

function _stopPoll() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

async function _doPoll() {
  try {
    const hdrs = S.token ? { Authorization: `Bearer ${S.token}` } : {};
    const r = await fetch('/api/live-events', { headers: hdrs });
    if (!r.ok) return;
    const { events } = await r.json();
    if (!events) return;

    let newCount = 0;
    for (const ev of events) {
      const ts = ev.ts || ev.timestamp || '';
      if (ts && ts <= _lastPollEventTs) continue;
      pushEvent(ev);
      if (_onEvent) _onEvent(ev);
      newCount++;
    }
    if (events.length > 0) {
      const latest = events[0];
      _lastPollEventTs = latest.ts || latest.timestamp || '';
    }
  } catch (_) {}
}

export function disconnectWS() {
  _stopPoll();
  if (_ws) { try { _ws.close(); } catch (_) {} _ws = null; }
}
