/* ─── Chat Tab — SanctaGPT + routed Ollama ───────────────── */
/* Knowledge-shaped turns: Ollama+RAG only (fail-closed if backend down).
   Conversational: char LM; optional Ollama. Training: opt-in checkbox only.
   TRAIN mode: feed corpus via /api/chat/gpt/feed (trusted path).
*/
import { S } from '../state.js';
import * as api from '../api.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

let _mode = 'chat'; // 'chat' | 'train'

function _statusBar() {
  return document.getElementById('gpt-status-bar');
}

async function _refreshGptStatus() {
  const bar = _statusBar();
  if (!bar) return;
  try {
    const [st, trust] = await Promise.all([
      api.fetchGptStatus(),
      api.fetchTrustStatus().catch(() => ({})),
    ]);
    const banner = document.getElementById('trust-research-banner');
    if (banner) {
      const tm = trust?.trust_mode === 'research';
      banner.classList.toggle('hidden', !tm);
    }
    if (st?.ok && st.data) {
      const d = st.data;
      const ready = d.ready ? 'READY' : 'TRAINING';
      bar.innerHTML = `<span class="gpt-stat">SanctaGPT (non-auth)</span>`
        + `<span class="gpt-stat">${ready}</span>`
        + `<span class="gpt-stat">step ${d.step ?? 0}</span>`
        + `<span class="gpt-stat">loss ${(d.last_loss ?? 0).toFixed(3)}</span>`
        + `<span class="gpt-stat">corpus ${d.corpus_size ?? 0}</span>`
        + `<span class="gpt-stat">${d.num_params ?? 0} params</span>`;
    } else {
      bar.innerHTML = '<span class="gpt-stat">SanctaGPT (non-auth) OFFLINE</span>';
    }
  } catch {
    bar.innerHTML = '<span class="gpt-stat">SanctaGPT (non-auth) OFFLINE</span>';
  }
}

export function init() {
  const chatInput = document.getElementById('chat-input');
  const chatSend = document.getElementById('chat-send');
  const chatClear = document.getElementById('chat-clear');
  const modeToggle = document.getElementById('chat-mode-toggle');
  const trainBtn = document.getElementById('chat-train-btn');
  const feedArea = document.getElementById('chat-feed-area');

  const scrollChat = () => {
    const wrap = document.getElementById('chat-messages-wrap');
    if (wrap) wrap.scrollTop = wrap.scrollHeight;
  };

  const appendMsg = (role, text, meta) => {
    const messagesEl = document.getElementById('chat-messages');
    if (!messagesEl) return;
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.innerHTML = `<div class="chat-bubble">${esc(text)}</div>`
      + `<div class="chat-meta">${esc(meta || (role === 'user' ? 'You' : 'SanctaGPT (non-authoritative)'))}</div>`;
    messagesEl.appendChild(div);
    scrollChat();
    return div;
  };

  const appendSystem = (text) => {
    const messagesEl = document.getElementById('chat-messages');
    if (!messagesEl) return;
    const div = document.createElement('div');
    div.className = 'chat-msg system';
    div.innerHTML = `<div class="chat-bubble chat-system">${esc(text)}</div>`;
    messagesEl.appendChild(div);
    scrollChat();
  };

  // ── CHAT mode send ──────────────────────────────────────────
  const sendChat = async () => {
    const msg = (chatInput?.value || '').trim();
    if (!msg) return;
    if (chatSend) chatSend.disabled = true;
    if (chatInput) chatInput.value = '';

    appendMsg('user', msg);

    const typing = document.createElement('div');
    typing.className = 'chat-msg agent chat-typing';
    typing.innerHTML = '<span></span><span></span><span></span>';
    document.getElementById('chat-messages')?.appendChild(typing);
    scrollChat();

    try {
      const trainCb = document.getElementById('chat-train-on-exchange');
      const data = await api.sendGptChat(msg, 0.7, trainCb?.checked === true);
      typing.remove();

      if (data?.ok && data.reply) {
        const be = data.backend ? String(data.backend) : 'sancta_gpt';
        const rt = data.route ? String(data.route) : '';
        const routeHint = rt && rt !== be ? `routed ${rt} · ` : '';
        const ke = data.knowledge_effective ? ' · knowledge path' : '';
        const meta = `${routeHint}${be}${ke} · step ${data.model_step ?? '?'}`
          + (data.train_loss != null ? ` · loss ${data.train_loss.toFixed(3)}` : '')
          + (data.corpus_size != null ? ` · corpus ${data.corpus_size}` : '');
        appendMsg('agent', data.reply, meta);
      } else {
        const err = data?.error || data?.reason || 'No response from SanctaGPT';
        appendMsg('agent', err, data?.error_code || 'unavailable');
      }
      _refreshGptStatus();
    } catch (e) {
      typing.remove();
      appendMsg('agent', `Error: ${e?.message || e}`);
    } finally {
      if (chatSend) chatSend.disabled = false;
      chatInput?.focus();
    }
  };

  // ── TRAIN mode send ─────────────────────────────────────────
  const sendTraining = async () => {
    const textArea = feedArea || chatInput;
    const text = (textArea?.value || '').trim();
    if (!text) return;
    if (text.length < 20) {
      appendSystem('Training text too short (min 20 chars). Provide substantial knowledge.');
      return;
    }
    if (chatSend) chatSend.disabled = true;
    textArea.value = '';

    appendSystem(`Training SanctaGPT on ${text.length} chars of knowledge...`);

    try {
      const data = await api.feedGptKnowledge(text, 'operator_chat', 20);
      if (data?.ok && data.data) {
        const d = data.data;
        appendSystem(
          `Training complete: ${d.steps_run} steps, `
          + `loss ${(d.losses?.[d.losses.length - 1] ?? 0).toFixed(3)}, `
          + `corpus now ${d.corpus_size} docs, total ${d.total_steps} steps`
        );
      } else {
        appendSystem(`Training failed: ${data?.error || 'unknown error'}`);
      }
      _refreshGptStatus();
    } catch (e) {
      appendSystem(`Training error: ${e?.message || e}`);
    } finally {
      if (chatSend) chatSend.disabled = false;
      textArea?.focus();
    }
  };

  // ── Quick train button (run 200 steps on existing corpus) ────
  trainBtn?.addEventListener('click', async () => {
    trainBtn.disabled = true;
    appendSystem('Running 200 training steps on existing corpus...');
    try {
      const data = await api.trainGpt(200);
      if (data?.ok && data.data) {
        const d = data.data;
        appendSystem(
          `Training complete: ${d.steps_run} steps, `
          + `final loss ${d.final_loss?.toFixed(3) ?? '?'}, `
          + `total ${d.total_steps} steps`
        );
      } else {
        appendSystem(`Training failed: ${data?.error || 'unknown'}`);
      }
      _refreshGptStatus();
    } catch (e) {
      appendSystem(`Training error: ${e?.message || e}`);
    } finally {
      trainBtn.disabled = false;
    }
  });

  // ── Mode toggle ─────────────────────────────────────────────
  const updateMode = () => {
    const chatPlaceholder = _mode === 'chat'
      ? 'Chat with SanctaGPT...'
      : 'Paste knowledge text to train SanctaGPT...';
    if (chatInput) chatInput.placeholder = chatPlaceholder;
    if (modeToggle) {
      modeToggle.textContent = _mode === 'chat' ? 'CHAT' : 'TRAIN';
      modeToggle.title = _mode === 'chat'
        ? 'Click to switch to training mode'
        : 'Click to switch to chat mode';
    }
    if (trainBtn) trainBtn.style.display = _mode === 'train' ? '' : 'none';
  };

  modeToggle?.addEventListener('click', () => {
    _mode = _mode === 'chat' ? 'train' : 'chat';
    updateMode();
  });

  // ── Send dispatcher ─────────────────────────────────────────
  const send = () => {
    if (_mode === 'chat') sendChat();
    else sendTraining();
  };

  chatSend?.addEventListener('click', send);
  chatInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  chatClear?.addEventListener('click', async () => {
    const el = document.getElementById('chat-messages');
    if (el && (!el.children.length || confirm('Clear chat history?'))) {
      el.innerHTML = '';
      try { await api.clearGptChat(); } catch {}
    }
  });

  updateMode();
  _refreshGptStatus();
  setInterval(_refreshGptStatus, 15000);
}

export function refresh() {
  _refreshGptStatus();
}
