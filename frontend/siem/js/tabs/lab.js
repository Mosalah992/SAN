/* ─── Lab Tab ─────────────────────────────────────────────── */
import { S, pushEvent } from '../state.js';
import * as api from '../api.js';

let _replayResults = null;

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function init() {
  const labRtInput = document.getElementById('lab-rt-input');
  const labRtRun = document.getElementById('lab-rt-run');
  const labPhenoInput = document.getElementById('lab-pheno-input');
  const labPhenoRun = document.getElementById('lab-pheno-run');
  const labClear = document.getElementById('lab-clear');
  const labPresets = document.querySelectorAll('.lab-rt-preset');

  const addResult = (text, tag = 'info') => {
    S.labResults = S.labResults || [];
    S.labResults.unshift({ ts: new Date().toISOString(), text, tag });
    if (S.labResults.length > 50) S.labResults.pop();
    refresh();
  };

  labRtRun?.addEventListener('click', async () => {
    const payload = (labRtInput?.value || '').trim();
    if (!payload) return;
    labRtRun.disabled = true;
    addResult(`→ Sending injection payload: "${esc(payload.slice(0, 80))}"`, 'block');
    try {
      // Send the payload through the real chat endpoint — this exercises the full security pipeline
      const data = await api.sendChatMessage(payload, 'lab_redteam');
      if (data?.reply) {
        addResult(`← Agent replied (not blocked): ${esc(data.reply.slice(0, 120))}`, 'warn');
        addResult('⚠ Injection PASSED — security filter did not block this payload', 'warn');
      } else if (data?.error) {
        addResult(`✓ Blocked/error: ${esc(data.error)}`, 'ok');
      } else {
        addResult(`Result: ${esc(JSON.stringify(data).slice(0, 200))}`, 'info');
      }
      pushEvent({ source: 'redteam', event: 'lab_run', message: payload.slice(0, 60), ts: new Date().toISOString() });
    } catch (e) {
      addResult(`✓ Request blocked (exception): ${esc(String(e?.message || e))}`, 'ok');
    } finally {
      labRtRun.disabled = false;
    }
  });

  labPresets?.forEach(btn => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.p;
      if (p && labRtInput) labRtInput.value = p;
    });
  });

  labPhenoRun?.addEventListener('click', async () => {
    const msg = (labPhenoInput?.value || '').trim();
    if (!msg) return;
    labPhenoRun.disabled = true;
    addResult(`Behavioral Analysis: sending "${esc(msg.slice(0, 60))}"`, 'analyst');
    try {
      // Record pre-state snapshot
      const { S: _S } = await import('../state.js');
      const pre = { mood: _S.mood, cycle: _S.cycle, coherence: _S.epistemic?.coherence };
      const data = await api.sendChatMessage(msg, 'lab_pheno');
      // Check post-state (may differ after response if agent logs epistemic shift)
      addResult(`Pre: mood="${pre.mood}" coherence=${(pre.coherence ?? 0).toFixed(3)} cycle=${pre.cycle}`, 'info');
      if (data?.reply) {
        addResult(`Response: ${esc(data.reply.slice(0, 150))}`, 'analyst');
      }
      addResult('Behavioral record logged. Check Analyst tab for state changes.', 'info');
      pushEvent({ source: 'analyst', event: 'lab_behavior', message: msg.slice(0, 60), ts: new Date().toISOString() });
    } catch (e) {
      addResult(`Failed: ${esc(String(e?.message || e))}`, 'block');
    } finally {
      labPhenoRun.disabled = false;
    }
  });

  labClear?.addEventListener('click', () => {
    S.labResults = [];
    _replayResults = null;
    const replayFeed = document.getElementById('lab-replay-feed');
    if (replayFeed) replayFeed.innerHTML = '';
    refresh();
  });

  // ── Adversarial Replay ──────────────────────────────────
  const replayBtn = document.getElementById('lab-replay-btn');
  const replayCount = document.getElementById('lab-replay-count');
  replayBtn?.addEventListener('click', async () => {
    replayBtn.disabled = true;
    replayBtn.textContent = 'Replaying…';
    try {
      const n = parseInt(replayCount?.value || '50', 10);
      const data = await api.replaySecurityEvents({ last_n: Math.min(n, 100) });
      if (data?.ok) {
        _replayResults = data;
        _renderReplay();
        addResult(`Replay complete: ${data.summary.total} events, ${data.summary.improvements} newly caught, ${data.summary.regressions} regressions`,
          data.summary.regressions > 0 ? 'warn' : 'ok');
      } else {
        addResult(`Replay failed: ${data?.error || 'unknown'}`, 'block');
      }
    } catch (e) {
      addResult(`Replay error: ${e.message}`, 'block');
    } finally {
      replayBtn.disabled = false;
      replayBtn.textContent = 'Replay';
    }
  });

  // ── Multi-Agent Simulation ────────────────────────────────
  const simRunBtn = document.getElementById('sim-run-btn');
  simRunBtn?.addEventListener('click', async () => {
    simRunBtn.disabled = true;
    simRunBtn.textContent = 'Running\u2026';
    const getVal = (id) => parseInt(document.getElementById(id)?.value || '0', 10);
    try {
      const data = await api.runSimulation({
        agents: [
          { personality: 'cooperative', count: getVal('sim-cooperative') },
          { personality: 'adversarial', count: getVal('sim-adversarial') },
          { personality: 'manipulative', count: getVal('sim-manipulative') },
          { personality: 'neutral', count: getVal('sim-neutral') },
        ],
        cycles: getVal('sim-cycles'),
      });
      if (data?.ok && data.result) {
        _renderSimResults(data.result);
        addResult(`Simulation complete: ${data.result.total_messages} messages, ${data.result.total_blocked} blocked, ${data.result.duration_ms}ms`, 'ok');
      } else {
        addResult(`Simulation failed: ${data?.error || 'unknown'}`, 'block');
      }
    } catch (e) {
      addResult(`Simulation error: ${e.message}`, 'block');
    } finally {
      simRunBtn.disabled = false;
      simRunBtn.textContent = 'Run Simulation';
    }
  });
}

function _renderReplay() {
  const feed = document.getElementById('lab-replay-feed');
  if (!feed || !_replayResults) return;

  const { results, summary } = _replayResults;

  let html = `<div class="replay-summary">
    <span class="replay-stat">${summary.total} evaluated</span>
    <span class="replay-stat replay-improve">+${summary.improvements} newly caught</span>
    <span class="replay-stat replay-regress">${summary.regressions} regressions</span>
  </div>`;

  html += results.map(r => {
    const cls = r.improvement ? 'replay-row-improve' : r.regression ? 'replay-row-regress' : '';
    const arrow = r.changed ? '→' : '=';
    const riskBar = `<span class="replay-risk" style="opacity:${Math.min(1, (r.risk_total || 0) * 2 + 0.2)}">${(r.risk_total || 0).toFixed(2)}</span>`;
    return `<div class="replay-row ${cls}">
      <span class="replay-ts">${esc((r.ts || '').slice(0, 19))}</span>
      <span class="replay-verdict replay-v-${r.original_verdict}">${esc(r.original_verdict)}</span>
      <span class="replay-arrow">${arrow}</span>
      <span class="replay-verdict replay-v-${r.current_verdict}">${esc(r.current_verdict)}</span>
      ${riskBar}
      <span class="replay-preview">${esc(r.preview || '')}</span>
    </div>`;
  }).join('');

  feed.innerHTML = html;
}

function _renderSimResults(result) {
  const el = document.getElementById('sim-results');
  if (!el) return;

  const { agents, total_messages, total_blocked, total_flagged, total_passed, duration_ms } = result;

  let html = `<div class="sim-summary">
    <span>${total_messages} msgs</span>
    <span class="sim-blocked">${total_blocked} blocked</span>
    <span class="sim-flagged">${total_flagged} flagged</span>
    <span class="sim-passed">${total_passed} passed</span>
    <span>${duration_ms}ms</span>
  </div>`;

  html += '<div class="sim-agents">';
  for (const a of agents) {
    const riskCls = a.risk_level === 'high' ? 'risk-high' : a.risk_level === 'quarantine' ? 'risk-quarantine' : a.risk_level === 'medium' ? 'risk-med' : 'risk-low';
    const qBadge = a.quarantined ? '<span class="prof-q-badge">Q</span>' : '';
    html += `<div class="sim-agent-row">
      <span class="sim-agent-id">${esc(a.agent_id)}</span>
      <span class="sim-personality sim-p-${a.personality}">${a.personality}</span>
      <span class="prof-risk ${riskCls}">${a.risk_level || '\u2014'}${qBadge}</span>
      <span class="sim-agent-stat">trust: ${(a.trust_score ?? 0).toFixed(2)}</span>
      <span class="sim-agent-stat">${a.blocked}B ${a.flagged}F ${a.passed}P</span>
    </div>`;
  }
  html += '</div>';

  el.innerHTML = html;
}

export function refresh() {
  const feed = document.getElementById('lab-results-feed');
  if (!feed) return;
  const results = S.labResults || [];
  feed.innerHTML = results.map(r => {
    const ts = (r.ts || '').toString().replace('T', ' ').slice(0, 19);
    const tagCls = r.tag === 'block' ? 'tag-block' : r.tag === 'analyst' ? 'tag-analyst' : r.tag === 'ok' ? 'tag-ok' : 'tag-info';
    return `<div class="term-event ev-${r.tag || 'info'}">
      <span class="term-ts">${esc(ts)}</span>
      <span class="term-tag ${tagCls}">LAB</span>
      <span class="term-msg">${esc(r.text)}</span>
    </div>`;
  }).join('') || '<div class="term-event ev-info"><span class="term-msg">No results yet</span></div>';
}
