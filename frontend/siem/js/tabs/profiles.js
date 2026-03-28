/* ─── Profiles Tab — Per-Entity Threat Profiles ──────────── */
import { S } from '../state.js';
import * as api from '../api.js';

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const RISK_COLORS = {
  low:        'var(--green)',
  medium:     'var(--amber)',
  high:       'var(--red)',
  quarantine: 'var(--magenta)',
  unknown:    'var(--text-muted)',
};

const RISK_CLASSES = {
  low:        'risk-low',
  medium:     'risk-med',
  high:       'risk-high',
  quarantine: 'risk-quarantine',
  unknown:    'risk-unknown',
};

let _profiles = [];
let _selectedId = null;
let _sortKey = 'trust_score';
let _sortAsc = true;

/* ── Init ─────────────────────────────────────────────────── */
export function init() {
  // Sort header clicks
  document.querySelectorAll('#prof-table-head th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (_sortKey === key) _sortAsc = !_sortAsc;
      else { _sortKey = key; _sortAsc = key === 'trust_score'; }
      _renderTable();
    });
  });
}

/* ── Refresh (called on 10s poll) ─────────────────────────── */
export async function refresh() {
  try {
    const data = await api.fetchProfiles();
    if (data?.ok && data.profiles) {
      _profiles = data.profiles;
      const countEl = document.getElementById('prof-count');
      if (countEl) countEl.textContent = String(data.count ?? _profiles.length);
      _renderTable();
      // If detail is open, refresh it
      if (_selectedId) _renderDetail(_selectedId);
    }
  } catch (_) {}
}

/* ── Table Rendering ──────────────────────────────────────── */
function _sorted() {
  const arr = [..._profiles];
  arr.sort((a, b) => {
    let va = a[_sortKey], vb = b[_sortKey];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return _sortAsc ? -1 : 1;
    if (va > vb) return _sortAsc ? 1 : -1;
    return 0;
  });
  return arr;
}

function _renderTable() {
  const tbody = document.getElementById('prof-table-body');
  if (!tbody) return;

  if (!_profiles.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="prof-empty">No profiles tracked yet. Profiles appear after agent interactions.</td></tr>';
    return;
  }

  const sorted = _sorted();
  tbody.innerHTML = sorted.map(p => {
    const riskCls = RISK_CLASSES[p.risk_level] || 'risk-unknown';
    const qTag = p.quarantined ? '<span class="prof-q-badge">Q</span>' : '';
    const trustPct = Math.round(p.trust_score * 100);
    const trustCls = trustPct >= 60 ? 'trust-ok' : trustPct >= 30 ? 'trust-warn' : 'trust-danger';
    const selected = p.agent_id === _selectedId ? ' prof-row-selected' : '';
    return `<tr class="prof-row${selected}" data-agent="${esc(p.agent_id)}">
      <td class="prof-cell-id">${esc(p.agent_id)}${qTag}</td>
      <td><span class="prof-risk ${riskCls}">${esc(p.risk_level)}</span></td>
      <td>
        <div class="prof-trust-bar">
          <div class="prof-trust-fill ${trustCls}" style="width:${trustPct}%"></div>
        </div>
        <span class="prof-trust-val">${p.trust_score.toFixed(2)}</span>
      </td>
      <td class="prof-cell-num">${p.injection_attempts}</td>
      <td class="prof-cell-num">${p.influence_score.toFixed(2)}</td>
      <td class="prof-cell-num">${p.interaction_count ?? 0}</td>
      <td class="prof-cell-ts">${_shortTs(p.last_seen)}</td>
    </tr>`;
  }).join('');

  // Click handlers for rows
  tbody.querySelectorAll('.prof-row').forEach(row => {
    row.addEventListener('click', () => {
      const aid = row.dataset.agent;
      _selectedId = (_selectedId === aid) ? null : aid;
      _renderTable();
      if (_selectedId) _renderDetail(_selectedId);
      else _clearDetail();
    });
  });

  // Update sort indicators
  document.querySelectorAll('#prof-table-head th[data-sort]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.sort === _sortKey) {
      th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
    }
  });
}

/* ── Detail Panel ─────────────────────────────────────────── */
async function _renderDetail(agentId) {
  const panel = document.getElementById('prof-detail');
  if (!panel) return;
  panel.classList.remove('hidden');

  try {
    const data = await api.fetchProfile(agentId);
    if (!data?.ok || !data.profile) {
      panel.innerHTML = `<div class="prof-detail-err">Profile not found</div>`;
      return;
    }
    const p = data.profile;
    const riskCls = RISK_CLASSES[p.risk_level] || 'risk-unknown';
    const trustPct = Math.round(p.trust_score * 100);
    const trustCls = trustPct >= 60 ? 'trust-ok' : trustPct >= 30 ? 'trust-warn' : 'trust-danger';

    panel.innerHTML = `
      <div class="prof-detail-header">
        <span class="prof-detail-id">${esc(p.agent_id)}</span>
        <span class="prof-risk ${riskCls}">${esc(p.risk_level)}</span>
        <button id="prof-qtn-btn" class="btn btn-sm ${p.quarantined ? 'btn-danger' : 'btn-warn'}">${p.quarantined ? 'Lift Quarantine' : 'Quarantine'}</button>
        <button id="prof-close-btn" class="btn btn-sm">✕</button>
      </div>
      <div class="prof-detail-stats">
        <div class="prof-stat">
          <span class="prof-stat-label">Trust</span>
          <div class="prof-trust-bar prof-trust-bar-lg">
            <div class="prof-trust-fill ${trustCls}" style="width:${trustPct}%"></div>
          </div>
          <span class="prof-stat-val">${p.trust_score.toFixed(3)}</span>
        </div>
        <div class="prof-stat"><span class="prof-stat-label">Injections</span><span class="prof-stat-val">${p.injection_attempts}</span></div>
        <div class="prof-stat"><span class="prof-stat-label">Obfuscations</span><span class="prof-stat-val">${p.obfuscation_attempts}</span></div>
        <div class="prof-stat"><span class="prof-stat-label">Influence</span><span class="prof-stat-val">${p.influence_score.toFixed(3)}</span></div>
        <div class="prof-stat"><span class="prof-stat-label">Belief Changes</span><span class="prof-stat-val">${p.belief_changes_caused}</span></div>
        <div class="prof-stat"><span class="prof-stat-label">Interactions</span><span class="prof-stat-val">${p.interaction_count}</span></div>
        <div class="prof-stat"><span class="prof-stat-label">First Seen</span><span class="prof-stat-val">${_shortTs(p.first_seen)}</span></div>
        <div class="prof-stat"><span class="prof-stat-label">Last Seen</span><span class="prof-stat-val">${_shortTs(p.last_seen)}</span></div>
      </div>
      ${p.quarantine_reason ? `<div class="prof-q-reason">Quarantine reason: ${esc(p.quarantine_reason)}</div>` : ''}
      <div class="prof-detail-section">
        <div class="prof-section-title">Trust History</div>
        <div class="prof-sparkline" id="prof-sparkline"></div>
      </div>
      <div class="prof-detail-section">
        <div class="prof-section-title">Recent Interactions</div>
        <div class="prof-history" id="prof-history-feed"></div>
      </div>
    `;

    // Sparkline from interaction history
    _drawSparkline(p.interaction_history || []);

    // Interaction history feed
    _renderHistory(p.interaction_history || []);

    // Quarantine button
    document.getElementById('prof-qtn-btn')?.addEventListener('click', async () => {
      try {
        const res = await api.toggleQuarantine(agentId);
        if (res?.ok) {
          window.showToast?.(`Quarantine ${res.action} for ${agentId}`, 'success');
          await refresh();
        }
      } catch (e) {
        window.showToast?.(`Failed: ${e.message}`, 'error');
      }
    });

    // Close button
    document.getElementById('prof-close-btn')?.addEventListener('click', () => {
      _selectedId = null;
      _clearDetail();
      _renderTable();
    });
  } catch (e) {
    panel.innerHTML = `<div class="prof-detail-err">Error: ${esc(e.message)}</div>`;
  }
}

function _clearDetail() {
  const panel = document.getElementById('prof-detail');
  if (panel) { panel.classList.add('hidden'); panel.innerHTML = ''; }
}

/* ── Sparkline (trust over interactions) ──────────────────── */
function _drawSparkline(history) {
  const container = document.getElementById('prof-sparkline');
  if (!container || !history.length) {
    if (container) container.innerHTML = '<span class="prof-no-data">No history</span>';
    return;
  }

  // Extract trust-like signal: 1.0 - (injection ? 0.12 : 0) per interaction
  // We approximate trust trajectory from interaction outcomes
  const W = 280, H = 40, PAD = 2;
  let trust = 0.5;
  const points = [];
  for (const h of history) {
    if (h.injection_detected) trust = Math.max(0, trust - 0.12);
    else if (h.obfuscation_detected) trust = Math.max(0, trust - 0.08);
    else trust = Math.min(0.85, trust + 0.01);
    points.push(trust);
  }

  const maxPts = Math.min(points.length, 60);
  const pts = points.slice(-maxPts);
  const stepX = (W - PAD * 2) / Math.max(pts.length - 1, 1);

  const svgPts = pts.map((v, i) => {
    const x = PAD + i * stepX;
    const y = PAD + (1 - v) * (H - PAD * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const last = pts[pts.length - 1];
  const color = last >= 0.6 ? 'var(--green)' : last >= 0.3 ? 'var(--amber)' : 'var(--red)';

  container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="prof-spark-svg">
    <polyline points="${svgPts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
  </svg>`;
}

/* ── Interaction History Feed ─────────────────────────────── */
function _renderHistory(history) {
  const feed = document.getElementById('prof-history-feed');
  if (!feed) return;
  if (!history.length) {
    feed.innerHTML = '<div class="prof-no-data">No interactions recorded</div>';
    return;
  }
  const recent = history.slice(-30).reverse();
  feed.innerHTML = recent.map(h => {
    const ts = _shortTs(h.timestamp);
    const injCls = h.injection_detected ? 'hist-injection' : '';
    const obfCls = h.obfuscation_detected ? 'hist-obfuscation' : '';
    const tag = h.injection_detected ? 'INJ' : h.obfuscation_detected ? 'OBF' : 'OK';
    const tagCls = h.injection_detected ? 'tag-block' : h.obfuscation_detected ? 'tag-warn' : 'tag-info';
    const preview = esc((h.content_preview || '').slice(0, 80));
    const delta = h.belief_delta ? ` Δ${h.belief_delta > 0 ? '+' : ''}${h.belief_delta.toFixed(3)}` : '';
    return `<div class="prof-hist-row ${injCls} ${obfCls}">
      <span class="prof-hist-ts">${ts}</span>
      <span class="term-tag ${tagCls}">${tag}</span>
      <span class="prof-hist-preview">${preview}</span>
      ${delta ? `<span class="prof-hist-delta">${delta}</span>` : ''}
    </div>`;
  }).join('');
}

/* ── Helpers ──────────────────────────────────────────────── */
function _shortTs(ts) {
  if (!ts) return '—';
  return String(ts).replace('T', ' ').slice(0, 19);
}
