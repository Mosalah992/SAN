/* ─── Desktop & In-App Notification System ─────────────────── */

const LS_ENABLED = 'sancta_notifications_enabled';
const LS_SOUNDS  = 'sancta_sounds_enabled';
const MAX_TOASTS = 5;
const TOAST_TTL  = 5000;
const NOTIF_TTL  = 8000;

let _audioCtx = null;

/* ── Init ────────────────────────────────────────────────── */
export function init() {
  // Just check availability — don't request permission yet
  if (typeof Notification === 'undefined') {
    console.warn('[notifications] Notification API not available');
  }
}

/* ── Permission ─────────────────────────────────────────── */
export async function requestPermission() {
  if (typeof Notification === 'undefined') return 'denied';
  const result = await Notification.requestPermission();
  return result;
}

/* ── Desktop notification + sound ───────────────────────── */
export function notify(title, body, severity) {
  if (!isEnabled()) return;
  if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return;

  const n = new Notification(title, {
    body: body || '',
    icon: '/favicon.ico',
    tag: 'sancta-' + Date.now(),
  });
  setTimeout(() => n.close(), NOTIF_TTL);

  _playSound(severity);
}

/* ── Sound via Web Audio API ────────────────────────────── */
function _playSound(severity) {
  if (localStorage.getItem(LS_SOUNDS) === 'false') return;

  try {
    if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    const freq     = severity === 'critical' ? 880 : severity === 'warning' ? 660 : 440;
    const duration = severity === 'critical' ? 0.2 : severity === 'warning' ? 0.15 : 0.1;

    const osc  = _audioCtx.createOscillator();
    const gain = _audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.value = 0.15;

    osc.connect(gain);
    gain.connect(_audioCtx.destination);

    osc.start();
    osc.stop(_audioCtx.currentTime + duration);
  } catch (_) {
    // Audio may be blocked by browser policy — ignore
  }
}

/* ── In-App Toast ───────────────────────────────────────── */
export function toastAlert(msg, severity) {
  let container = document.getElementById('notif-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'notif-container';
    document.body.appendChild(container);
  }

  // Evict oldest if at capacity
  while (container.children.length >= MAX_TOASTS) {
    container.removeChild(container.firstChild);
  }

  const toast = document.createElement('div');
  toast.className = 'notif-toast notif-' + (severity || 'info');
  toast.textContent = msg || '';
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('notif-fadeout');
    setTimeout(() => { toast.remove(); }, 320);
  }, TOAST_TTL);
}

/* ── State helpers ──────────────────────────────────────── */
export function isEnabled() {
  return localStorage.getItem(LS_ENABLED) === 'true';
}

export function setEnabled(bool) {
  localStorage.setItem(LS_ENABLED, bool ? 'true' : 'false');
}

export function setSoundsEnabled(bool) {
  localStorage.setItem(LS_SOUNDS, bool ? 'true' : 'false');
}

export function isSoundsEnabled() {
  return localStorage.getItem(LS_SOUNDS) !== 'false';
}
