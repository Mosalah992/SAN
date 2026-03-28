/* ─── Analyst Tab ───────────────────────────────────────────── */
import { S, evSeverity } from '../state.js';
import * as api from '../api.js';

let _driftEvents = [];

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function init() {}

export async function refresh() {
  const analystMoodVal = document.getElementById('analyst-mood-val');
  const analystMoodMeta = document.getElementById('analyst-mood-meta');
  const analystEnergyMeta = document.getElementById('analyst-energy-meta');
  const analystEpiBody = document.getElementById('analyst-epi-body');
  const analystBeliefsBody = document.getElementById('analyst-beliefs-body');
  const analystBeliefCount = document.getElementById('analyst-belief-count');
  const analystJournalFeed = document.getElementById('analyst-journal-feed');

  if (analystMoodVal) analystMoodVal.textContent = S.mood || '—';
  if (analystMoodMeta) analystMoodMeta.textContent = `cycle ${S.cycle} · deviation ${((S.epistemic?.dissonance ?? 0.1) * 100).toFixed(0)}%`;
  if (analystEnergyMeta) analystEnergyMeta.textContent = `energy — · patience —`;

  const ep = S.epistemic || {};
  if (analystEpiBody) {
    analystEpiBody.innerHTML = `
      <div class="prog-row">
        <span class="prog-label">coherence</span>
        <div class="prog-track"><div class="prog-fill pf-cyan" style="width:${(ep.coherence ?? 0.7) * 100}%"></div></div>
        <span class="prog-val">${((ep.coherence ?? 0.7) * 100).toFixed(0)}%</span>
      </div>
      <div class="prog-row">
        <span class="prog-label">deviation</span>
        <div class="prog-track"><div class="prog-fill pf-amber" style="width:${(ep.dissonance ?? 0.1) * 100}%"></div></div>
        <span class="prog-val">${((ep.dissonance ?? 0.1) * 100).toFixed(0)}%</span>
      </div>
      <div class="prog-row">
        <span class="prog-label">query drive</span>
        <div class="prog-track"><div class="prog-fill pf-purple" style="width:${(ep.curiosity ?? 0.6) * 100}%"></div></div>
        <span class="prog-val">${((ep.curiosity ?? 0.6) * 100).toFixed(0)}%</span>
      </div>
      <div class="prog-row">
        <span class="prog-label">confidence</span>
        <div class="prog-track"><div class="prog-fill pf-green" style="width:${(ep.confidence ?? 0.75) * 100}%"></div></div>
        <span class="prog-val">${((ep.confidence ?? 0.75) * 100).toFixed(0)}%</span>
      </div>
    `;
  }

  const beliefs = S.beliefs || {};
  const beliefKeys = Object.keys(beliefs);
  if (analystBeliefsBody) {
    if (!beliefKeys.length) {
      analystBeliefsBody.innerHTML = '<div style="padding:12px;color:var(--text-muted);font-family:var(--font-terminal);font-size:11px">No analytical positions loaded yet.</div>';
    } else {
      analystBeliefsBody.innerHTML = '<div class="belief-grid">' +
        beliefKeys.slice(0, 12).map(topic => {
          const raw = beliefs[topic];
          const conf = typeof raw === 'number' ? raw : (raw?.confidence ?? raw?.strength ?? 0.5);
          const pct = Math.round(Math.min(1, Math.max(0, +conf)) * 100);
          return `<div class="belief-card">
            <div class="belief-topic">${esc(topic)}</div>
            <div class="belief-conf-bar"><div class="belief-conf-fill" style="width:${pct}%"></div></div>
            <div class="belief-conf-val">${(+conf).toFixed(3)}</div>
          </div>`;
        }).join('') + '</div>';
    }
  }
  if (analystBeliefCount) analystBeliefCount.textContent = String(beliefKeys.length);

  let journal = S.journalEntries || [];
  // Fallback: pull analyst/behavioral events from the event buffer
  if (!journal.length) {
    journal = (S.events || [])
      .filter(e => {
        const src = String(e?.source ?? '').toLowerCase();
        const ev = String(e?.event ?? '').toLowerCase();
        return src === 'analyst' || src === 'behavioral' || ev.includes('mood') || ev.includes('belief');
      })
      .slice(0, 20)
      .map(e => ({ ts: e.ts || e.timestamp || '', message: e.message || e.preview || e.event || '' }));
  }
  if (analystJournalFeed) {
    analystJournalFeed.innerHTML = journal.slice(0, 30).map(entry => {
      const ts = (entry.ts || entry.timestamp || '').toString().slice(0, 19);
      const text = entry.message || entry.text || JSON.stringify(entry);
      return `<div class="analyst-journal-entry"><div class="sje-ts">${esc(ts)}</div>${esc(text)}</div>`;
    }).join('') || '<div class="analyst-journal-entry"><div class="sje-ts">—</div>No journal entries yet</div>';
  }

  // Drift forensics timeline
  try {
    const drift = await api.fetchDriftTimeline();
    if (drift?.ok && drift.events) {
      _driftEvents = drift.events;
      const countEl = document.getElementById('analyst-drift-count');
      if (countEl) countEl.textContent = String(drift.count || _driftEvents.length);
      _renderDriftTimeline();
    }
  } catch (_) {}
}

function _renderDriftTimeline() {
  const feed = document.getElementById('analyst-drift-feed');
  if (!feed) return;

  if (!_driftEvents.length) {
    feed.innerHTML = '<div class="prof-no-data">No drift events recorded yet.</div>';
    return;
  }

  // Show most recent first
  const recent = _driftEvents.slice(-50).reverse();
  feed.innerHTML = recent.map(ev => {
    const ts = (ev.ts || '').replace('T', ' ').slice(0, 19);
    const delta = ev.delta || 0;
    const deltaSign = delta > 0 ? '+' : '';
    const deltaCls = delta > 0 ? 'drift-positive' : delta < 0 ? 'drift-negative' : '';
    const riskCls = ev.agent_risk_level === 'high' ? 'risk-high' :
                    ev.agent_risk_level === 'quarantine' ? 'risk-quarantine' :
                    ev.agent_risk_level === 'medium' ? 'risk-med' : '';
    const agent = ev.source_agent || '\u2014';
    const topic = ev.topic || '\u2014';
    const bar = Math.min(100, Math.abs(delta) * 500); // scale for visibility

    return `<div class="drift-row">
      <span class="drift-ts">${esc(ts)}</span>
      <span class="drift-topic">${esc(topic)}</span>
      <span class="drift-agent ${riskCls}">${esc(agent)}</span>
      <div class="drift-bar-wrap">
        <div class="drift-bar ${deltaCls}" style="width:${bar}%"></div>
      </div>
      <span class="drift-delta ${deltaCls}">${deltaSign}${delta.toFixed(4)}</span>
      <span class="drift-conf">${(ev.new_confidence || 0).toFixed(2)}</span>
    </div>`;
  }).join('');
}
