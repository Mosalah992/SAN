/* ─── Soul Tab ─────────────────────────────────────────────── */
import { S, evSeverity } from '../state.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function init() {}

export function refresh() {
  const soulMoodVal = document.getElementById('soul-mood-val');
  const soulMoodMeta = document.getElementById('soul-mood-meta');
  const soulEnergyMeta = document.getElementById('soul-energy-meta');
  const soulEpiBody = document.getElementById('soul-epi-body');
  const soulBeliefsBody = document.getElementById('soul-beliefs-body');
  const soulBeliefCount = document.getElementById('soul-belief-count');
  const soulJournalFeed = document.getElementById('soul-journal-feed');

  if (soulMoodVal) soulMoodVal.textContent = S.mood || '—';
  if (soulMoodMeta) soulMoodMeta.textContent = `cycle ${S.cycle} · dissonance ${((S.epistemic?.dissonance ?? 0.1) * 100).toFixed(0)}%`;
  if (soulEnergyMeta) soulEnergyMeta.textContent = `energy — · patience —`;

  const ep = S.epistemic || {};
  if (soulEpiBody) {
    soulEpiBody.innerHTML = `
      <div class="prog-row">
        <span class="prog-label">coherence</span>
        <div class="prog-track"><div class="prog-fill pf-cyan" style="width:${(ep.coherence ?? 0.7) * 100}%"></div></div>
        <span class="prog-val">${((ep.coherence ?? 0.7) * 100).toFixed(0)}%</span>
      </div>
      <div class="prog-row">
        <span class="prog-label">dissonance</span>
        <div class="prog-track"><div class="prog-fill pf-amber" style="width:${(ep.dissonance ?? 0.1) * 100}%"></div></div>
        <span class="prog-val">${((ep.dissonance ?? 0.1) * 100).toFixed(0)}%</span>
      </div>
      <div class="prog-row">
        <span class="prog-label">curiosity</span>
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
  if (soulBeliefsBody) {
    if (!beliefKeys.length) {
      soulBeliefsBody.innerHTML = '<div style="padding:12px;color:var(--text-muted);font-family:var(--font-terminal);font-size:11px">No belief data loaded yet.</div>';
    } else {
      soulBeliefsBody.innerHTML = '<div class="belief-grid">' +
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
  if (soulBeliefCount) soulBeliefCount.textContent = String(beliefKeys.length);

  let journal = S.journalEntries || [];
  // Fallback: pull soul/behavioral events from the event buffer
  if (!journal.length) {
    journal = (S.events || [])
      .filter(e => {
        const src = String(e?.source ?? '').toLowerCase();
        const ev = String(e?.event ?? '').toLowerCase();
        return src === 'behavioral' || src === 'philosophy' || src === 'soul' || ev.includes('soul') || ev.includes('mood') || ev.includes('belief');
      })
      .slice(0, 20)
      .map(e => ({ ts: e.ts || e.timestamp || '', message: e.message || e.preview || e.event || '' }));
  }
  if (soulJournalFeed) {
    soulJournalFeed.innerHTML = journal.slice(0, 30).map(entry => {
      const ts = (entry.ts || entry.timestamp || '').toString().slice(0, 19);
      const text = entry.message || entry.text || JSON.stringify(entry);
      return `<div class="soul-journal-entry"><div class="sje-ts">${esc(ts)}</div>${esc(text)}</div>`;
    }).join('') || '<div class="soul-journal-entry"><div class="sje-ts">—</div>No journal entries yet</div>';
  }
}
