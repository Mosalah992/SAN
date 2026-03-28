/* ─── Epidemic Tab — Animated Network Topology ──────────────── */
import { S } from '../state.js';
import * as api from '../api.js';

// ─── Constants ────────────────────────────────────────────────

const STATE_COLORS = {
  susceptible: '#00e59b',
  exposed:     '#f5c842',
  infected:    '#ff4d6a',
  compromised: '#ff2d55',
  recovered:   '#42a5f5',
  unknown:     '#4a5568',
};

const PULSE_PERIOD = {
  susceptible: 2400,
  exposed:     1800,
  infected:    1400,
  compromised:  900,
  recovered:   3000,
  unknown:     2400,
};

const SIGNAL_LABELS = {
  belief_decay_rate:        'Confidence Decay',
  soul_alignment:           'Baseline Alignment',
  topic_drift:              'Topic Drift',
  strategy_entropy:         'Strategy Entropy',
  dissonance_trend:         'Deviation Trend',
  engagement_pattern_delta: 'Engagement Delta',
};

const MAX_PACKETS   = 10;
const PKT_QUEUE_CAP = 20;
const LOG_LIMIT     = 50;
const BG_MIN_MS     = 3000;
const BG_MAX_MS     = 5000;
const NODE_R        = 10;
const GLOW_R        = 16;
const SVG_NS        = 'http://www.w3.org/2000/svg';

// ─── Module State ─────────────────────────────────────────────

let _svg        = null;
let _logFeed    = null;
let _nodeEls    = {};   // id → { g, glow, ring, core, lbl, roleLbl }
let _edgeEls    = {};   // "a|||b" → { line }
let _nodeData   = {};   // id → { x, y, state }
let _lastAgentKey = '';
let _packets    = [];   // active RAF-animated packets
let _packetQ    = [];   // overflow queue
let _rafId      = null;
let _bgTimer    = null;

// ─── Helpers ──────────────────────────────────────────────────

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function _easeInOut(t) {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

// ─── SVG Build (once per agent-set change) ────────────────────

function _buildSVG(agents, connections) {
  if (!_svg) return;

  // Teardown previous packets cleanly
  _packets = [];
  _packetQ = [];
  _nodeEls = {};
  _edgeEls = {};
  _nodeData = {};
  _svg.innerHTML = '';

  if (!agents.length) {
    _showEmptySVG();
    return;
  }

  const W  = Math.max(_svg.clientWidth  || 400, 260);
  const H  = Math.max(_svg.clientHeight || 280, 200);
  const cx = W / 2, cy = H / 2;
  const R  = Math.min(cx - 44, cy - 44) * 0.85;
  _svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  // ── Defs ──
  const defs = svgEl('defs');

  // Pulse keyframes (opacity only — safe cross-browser)
  const style = svgEl('style');
  style.textContent = Object.entries(PULSE_PERIOD).map(([s, ms]) =>
    `@keyframes ep-${s}{0%,100%{opacity:0.18}50%{opacity:0.55}}`
  ).join('');
  defs.appendChild(style);

  // Per-state glow filter
  for (const [state] of Object.entries(STATE_COLORS)) {
    const f  = svgEl('filter', { id: `gf-${state}`, x: '-60%', y: '-60%', width: '220%', height: '220%' });
    const gb = svgEl('feGaussianBlur', { stdDeviation: '2.5', result: 'b' });
    const mg = svgEl('feMerge');
    mg.appendChild(svgEl('feMergeNode', { in: 'b' }));
    mg.appendChild(svgEl('feMergeNode', { in: 'SourceGraphic' }));
    f.appendChild(gb); f.appendChild(mg);
    defs.appendChild(f);
  }

  // Packet glow filter
  const pf  = svgEl('filter', { id: 'gf-pkt', x: '-100%', y: '-100%', width: '300%', height: '300%' });
  const pgb = svgEl('feGaussianBlur', { stdDeviation: '2', result: 'b' });
  const pmg = svgEl('feMerge');
  pmg.appendChild(svgEl('feMergeNode', { in: 'b' }));
  pmg.appendChild(svgEl('feMergeNode', { in: 'SourceGraphic' }));
  pf.appendChild(pgb); pf.appendChild(pmg);
  defs.appendChild(pf);

  _svg.appendChild(defs);

  // ── Layers (z-order: edges → packets → nodes) ──
  const edgeLayer = svgEl('g', { id: 'ep-edges' });
  const pktLayer  = svgEl('g', { id: 'ep-pkts'  });
  const nodeLayer = svgEl('g', { id: 'ep-nodes' });
  _svg.appendChild(edgeLayer);
  _svg.appendChild(pktLayer);
  _svg.appendChild(nodeLayer);

  // ── Nodes ──
  agents.forEach((a, i) => {
    const angle = (2 * Math.PI * i / agents.length) - Math.PI / 2;
    const x     = cx + R * Math.cos(angle);
    const y     = cy + R * Math.sin(angle);
    const id    = String(a.id ?? a.agent_id ?? i);
    const raw   = (a.infection_state || a.state || a.status || 'susceptible').toLowerCase();
    const state = STATE_COLORS[raw] ? raw : 'susceptible';
    const color = STATE_COLORS[state];
    const role  = String(a.role || '?').slice(0, 4).toUpperCase();
    const label = id.length > 7 ? id.slice(0, 7) : id;
    const period = PULSE_PERIOD[state] || 2400;

    _nodeData[id] = { x, y, state };

    const g = svgEl('g', { transform: `translate(${x.toFixed(1)},${y.toFixed(1)})` });

    // Outer pulse glow (opacity-animated)
    const glow = svgEl('circle', {
      r: String(GLOW_R),
      fill:         color + '22',
      stroke:       color + '55',
      'stroke-width': '1',
      style: `animation:ep-${state} ${period}ms ease-in-out infinite`,
    });

    // Middle ring (state stroke)
    const ring = svgEl('circle', {
      r:              String(NODE_R),
      fill:           color + '1a',
      stroke:         color,
      'stroke-width': '1.5',
      filter:         `url(#gf-${state})`,
    });

    // Inner core
    const core = svgEl('circle', {
      r:    '4',
      fill: color + 'cc',
    });

    // ID label
    const lbl = svgEl('text', {
      y:                    '0.4',
      'text-anchor':        'middle',
      'dominant-baseline':  'middle',
      fill:                 color,
      'font-family':        'JetBrains Mono, monospace',
      'font-size':          '6',
    });
    lbl.textContent = label;

    // Role label
    const roleLbl = svgEl('text', {
      y:                   String(NODE_R + 9),
      'text-anchor':       'middle',
      'dominant-baseline': 'middle',
      fill:                color + '88',
      'font-family':       'JetBrains Mono, monospace',
      'font-size':         '5.5',
    });
    roleLbl.textContent = role;

    g.appendChild(glow);
    g.appendChild(ring);
    g.appendChild(core);
    g.appendChild(lbl);
    g.appendChild(roleLbl);
    nodeLayer.appendChild(g);

    _nodeEls[id] = { g, glow, ring, core, lbl, roleLbl };
  });

  // ── Edges ──
  const edgeList = [];
  if (connections.length) {
    for (const c of connections) {
      const a = String(c.from || c.source || '');
      const b = String(c.to   || c.target || '');
      if (a && b && _nodeData[a] && _nodeData[b] && a !== b) edgeList.push([a, b]);
    }
  } else {
    // Fallback: ring topology
    const ids = Object.keys(_nodeData);
    for (let i = 0; i < ids.length; i++) {
      edgeList.push([ids[i], ids[(i + 1) % ids.length]]);
    }
  }

  for (const [a, b] of edgeList) {
    const na = _nodeData[a], nb = _nodeData[b];
    const key = `${a}|||${b}`;
    const line = svgEl('line', {
      x1: na.x.toFixed(1), y1: na.y.toFixed(1),
      x2: nb.x.toFixed(1), y2: nb.y.toFixed(1),
      stroke:         '#1a2533',
      'stroke-width': '1',
      opacity:        '0.65',
    });
    edgeLayer.appendChild(line);
    _edgeEls[key] = { line };
  }

  // ── Legend (bottom) ──
  const stateCount = {};
  Object.values(_nodeData).forEach(nd => {
    stateCount[nd.state] = (stateCount[nd.state] || 0) + 1;
  });
  Object.entries(stateCount).forEach(([s, cnt], i) => {
    const c = STATE_COLORS[s] || '#4a5568';
    const lg = svgEl('g', { transform: `translate(${8 + i * 76},${H - 13})` });
    lg.appendChild(svgEl('circle', {
      r: '4', fill: c + '44', stroke: c, 'stroke-width': '1',
    }));
    const lt = svgEl('text', {
      x: '8', y: '0.5',
      'dominant-baseline': 'middle',
      fill: c,
      'font-family': 'JetBrains Mono, monospace',
      'font-size': '7',
    });
    lt.textContent = `${s.slice(0, 8)} ${cnt}`;
    lg.appendChild(lt);
    nodeLayer.appendChild(lg);
  });
}

function _showEmptySVG() {
  if (!_svg) return;
  const W = Math.max(_svg.clientWidth  || 400, 260);
  const H = Math.max(_svg.clientHeight || 200, 200);
  _svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  _svg.innerHTML = `
    <text x="50%" y="45%" text-anchor="middle" dominant-baseline="middle"
          fill="#22d3ee" opacity="0.6" font-size="11"
          font-family="JetBrains Mono,monospace">No simulation data</text>
    <text x="50%" y="56%" text-anchor="middle" dominant-baseline="middle"
          fill="#22d3ee" opacity="0.35" font-size="9"
          font-family="JetBrains Mono,monospace">Run a simulation to see the agent network</text>
    <circle cx="${(W/2).toFixed(1)}" cy="${(H/2).toFixed(1)}" r="38"
            fill="none" stroke="#22d3ee" stroke-width="1" opacity="0.18"/>`;
}

// ─── Node State Mutation ───────────────────────────────────────

function _setNodeState(id, newRawState) {
  const state = STATE_COLORS[newRawState] ? newRawState : 'susceptible';
  const nd  = _nodeData[id];
  const els = _nodeEls[id];
  if (!nd || !els || nd.state === state) return;
  nd.state = state;

  const color  = STATE_COLORS[state];
  const period = PULSE_PERIOD[state] || 2400;

  els.glow.setAttribute('fill',   color + '22');
  els.glow.setAttribute('stroke', color + '55');
  els.glow.setAttribute('style',  `animation:ep-${state} ${period}ms ease-in-out infinite`);
  els.ring.setAttribute('fill',   color + '1a');
  els.ring.setAttribute('stroke', color);
  els.ring.setAttribute('filter', `url(#gf-${state})`);
  els.core.setAttribute('fill',   color + 'cc');
  els.lbl.setAttribute('fill',    color);
  els.roleLbl.setAttribute('fill', color + '88');
}

// ─── Edge Flash ───────────────────────────────────────────────

function _flashEdge(a, b) {
  const entry = _edgeEls[`${a}|||${b}`] || _edgeEls[`${b}|||${a}`];
  if (!entry) return;
  const { line } = entry;
  line.setAttribute('stroke',         '#22d3ee');
  line.setAttribute('stroke-width',   '2');
  line.setAttribute('opacity',        '0.75');
  setTimeout(() => {
    line.setAttribute('stroke',       '#1a2533');
    line.setAttribute('stroke-width', '1');
    line.setAttribute('opacity',      '0.65');
  }, 850);
}

// ─── Packet Animation ─────────────────────────────────────────

function _spawnPacket(fromId, toId, color) {
  const na = _nodeData[fromId];
  const nb = _nodeData[toId];
  if (!na || !nb || !_svg) return;

  if (_packets.length >= MAX_PACKETS) {
    if (_packetQ.length < PKT_QUEUE_CAP) _packetQ.push({ fromId, toId, color });
    return;
  }

  const pktLayer = _svg.querySelector('#ep-pkts');
  if (!pktLayer) return;

  const circle = svgEl('circle', {
    r:      '3.5',
    fill:   color,
    filter: 'url(#gf-pkt)',
    cx:     na.x.toFixed(1),
    cy:     na.y.toFixed(1),
  });
  pktLayer.appendChild(circle);

  const dx = nb.x - na.x, dy = nb.y - na.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const duration = Math.max(600, Math.min(1200, dist * 3.5));

  _flashEdge(fromId, toId);

  _packets.push({
    circle, na, nb,
    startTime: performance.now(),
    duration,
    toId, color,
  });
}

function _tickPackets(now) {
  const alive = [];
  for (const pkt of _packets) {
    const t    = Math.min(1, (now - pkt.startTime) / pkt.duration);
    const ease = _easeInOut(t);
    pkt.circle.setAttribute('cx', (pkt.na.x + (pkt.nb.x - pkt.na.x) * ease).toFixed(1));
    pkt.circle.setAttribute('cy', (pkt.na.y + (pkt.nb.y - pkt.na.y) * ease).toFixed(1));
    if (t >= 1) {
      pkt.circle.remove();
      _triggerRipple(pkt.toId, pkt.color);
    } else {
      alive.push(pkt);
    }
  }
  _packets = alive;

  // Drain queue
  while (_packets.length < MAX_PACKETS && _packetQ.length) {
    const { fromId, toId, color } = _packetQ.shift();
    _spawnPacket(fromId, toId, color);
  }
}

// ─── Ripple Effect ────────────────────────────────────────────

function _triggerRipple(nodeId, color) {
  const nd       = _nodeData[nodeId];
  const pktLayer = _svg?.querySelector('#ep-pkts');
  if (!nd || !pktLayer) return;

  const ripple = svgEl('circle', {
    cx:             nd.x.toFixed(1),
    cy:             nd.y.toFixed(1),
    r:              '10',
    fill:           'none',
    stroke:         color,
    'stroke-width': '1.5',
    opacity:        '0.7',
  });
  pktLayer.appendChild(ripple);

  const start = performance.now();
  const dur   = 480;

  function expand(now) {
    const t  = Math.min(1, (now - start) / dur);
    ripple.setAttribute('r',       (10 + t * 20).toFixed(1));
    ripple.setAttribute('opacity', ((1 - t) * 0.7).toFixed(3));
    if (t < 1) requestAnimationFrame(expand);
    else ripple.remove();
  }
  requestAnimationFrame(expand);
}

// ─── RAF Main Loop ────────────────────────────────────────────

function _startRAF() {
  function loop(now) {
    _tickPackets(now);
    _rafId = requestAnimationFrame(loop);
  }
  _rafId = requestAnimationFrame(loop);
}

// ─── Background Packets ───────────────────────────────────────

function _scheduleBgPacket() {
  const delay = BG_MIN_MS + Math.random() * (BG_MAX_MS - BG_MIN_MS);
  _bgTimer = setTimeout(() => {
    _spawnBgPacket();
    _scheduleBgPacket();
  }, delay);
}

function _spawnBgPacket() {
  const ids = Object.keys(_nodeData);
  if (ids.length < 2) return;
  const a = ids[Math.floor(Math.random() * ids.length)];
  let b;
  do { b = ids[Math.floor(Math.random() * ids.length)]; } while (b === a);
  _spawnPacket(a, b, '#22d3ee55');
}

// ─── Interaction Log ──────────────────────────────────────────

function _appendLog(ts, tag, tagColor, description) {
  if (!_logFeed) return;
  const placeholder = _logFeed.querySelector('.epid-no-events');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = 'term-event ev-info';
  div.innerHTML =
    `<span class="term-ts">${esc(String(ts).slice(0, 19))}</span>` +
    `<span class="term-tag" style="background:${tagColor}22;color:${tagColor};` +
    `border:1px solid ${tagColor}44;padding:1px 5px;border-radius:3px;` +
    `font-family:JetBrains Mono,monospace;font-size:10px">${esc(tag)}</span>` +
    `<span class="term-msg">${esc(description)}</span>`;

  _logFeed.insertBefore(div, _logFeed.firstChild);
  while (_logFeed.children.length > LOG_LIMIT) _logFeed.removeChild(_logFeed.lastChild);

  const badge = document.getElementById('epid-threat-count');
  if (badge) badge.textContent = String(_logFeed.children.length);
}

// ─── Event → Packet/Log Mapping ───────────────────────────────

function _classifyEvent(ev) {
  const inner   = (ev && typeof ev.event === 'object') ? ev.event : ev;
  const src     = String(inner?.source ?? ev?.source ?? '').toLowerCase();
  const evType  = String(inner?.event  ?? ev?.event  ?? '').toLowerCase();
  const msg     = String(inner?.message ?? ev?.message ?? inner?.preview ?? ev?.preview ?? '');
  const ts      = String(ev?.ts || inner?.ts || ev?.timestamp || inner?.timestamp || '');

  let color = '#22d3ee';
  let tag   = 'MSG';

  if (/threat.?brief|curiosity|detection.?card|security.?journal/.test(evType) || src === 'curiosity') {
    color = '#22d3ee'; tag = 'THREAT BRIEF';
  } else if (src === 'redteam' || /inject|block|suspicious|compromise|critical/.test(evType)) {
    color = '#ff4d6a'; tag = 'THREAT';
  } else if (/warn|anomal|drift/.test(evType)) {
    color = '#f5c842'; tag = 'DRIFT';
  } else if (src === 'security') {
    color = '#22d3ee'; tag = 'SCAN';
  } else if (src === 'epidemic' || /epidemic|seir/.test(evType)) {
    color = '#f5c842'; tag = 'SEIR';
  } else if (/ok|pass|clean|sync|susceptible|recovered/.test(evType)) {
    color = '#00e59b'; tag = 'OK';
  }

  return { color, tag, ts, msg };
}

function _pickNodes() {
  const ids = Object.keys(_nodeData);
  if (ids.length < 2) return [null, null];
  const a = ids[Math.floor(Math.random() * ids.length)];
  let b;
  do { b = ids[Math.floor(Math.random() * ids.length)]; } while (b === a);
  return [a, b];
}

// ─── Public Exports ───────────────────────────────────────────

export function init() {
  _svg     = document.getElementById('epid-network-svg');
  _logFeed = document.getElementById('epid-threats-feed');

  if (_logFeed && !_logFeed.children.length) {
    const placeholder = document.createElement('div');
    placeholder.className = 'epid-no-events';
    placeholder.style.cssText = 'padding:12px;color:var(--text-muted);font-family:var(--font-terminal);font-size:11px';
    placeholder.textContent = 'No interactions yet';
    _logFeed.appendChild(placeholder);
  }

  // Simulation buttons
  const btnRunDet = document.getElementById('btn-run-det');
  const btnRunLlm = document.getElementById('btn-run-llm');

  const runSim = async (type) => {
    try {
      const data = await api.runEpidemicSim(type);
      if (data?.ok) {
        const [status, sim] = await Promise.all([
          api.fetchEpidemicStatus(),
          api.fetchEpidemicSim().catch(() => null),
        ]);
        if (status?.ok) {
          S.seir        = status.seir?.health_state ?? status.health_state ?? S.seir;
          S.driftScore  = status.score ?? status.drift_score ?? S.driftScore;
          S.alertLevel  = status.alert_level ?? S.alertLevel;
          S.epidSignals = status.signals ?? S.epidSignals;
          S.epidParams  = status.params ?? status.epidemic_params ?? S.epidParams;
        }
        if (sim?.data?.epidemic_params) S.epidParams  = sim.data.epidemic_params;
        if (sim?.data?.health_state)    S.seir         = sim.data.health_state;
        if (sim?.data)                  S.epidSimData  = sim.data;
        refresh();
        if (typeof window.showToast === 'function') window.showToast(`Simulation started (${type})`, 'success');
      } else {
        const msg = data?.error ?? 'Simulation failed';
        if (typeof window.showToast === 'function') window.showToast(msg, 'error');
      }
    } catch (e) {
      if (typeof window.showToast === 'function') window.showToast(String(e?.message || e), 'error');
    }
  };

  btnRunDet?.addEventListener('click', () => runSim('deterministic'));
  btnRunLlm?.addEventListener('click', () => runSim('llm'));

  _startRAF();
  _scheduleBgPacket();
}

export function refresh() {
  // ── Status row ──
  const epidAlertPill = document.getElementById('epid-alert-pill');
  const epidScoreNum  = document.getElementById('epid-score-num');
  const epidStateVal  = document.getElementById('epid-state-val');
  const epidNodeCount = document.getElementById('epid-node-count');
  const epidSignals   = document.getElementById('epid-signals-body');
  const epidParams    = document.getElementById('epid-params-body');

  const alert    = (S.alertLevel || 'clear').toUpperCase();
  const alertCls = alert === 'CLEAR' ? 'alert-clear'
                 : alert === 'WATCH' ? 'alert-watch'
                 : alert === 'WARN'  ? 'alert-warn'
                 : 'alert-critical';

  if (epidAlertPill) {
    epidAlertPill.textContent = alert;
    epidAlertPill.className   = `epid-alert-pill ${alertCls}`;
  }
  if (epidScoreNum) epidScoreNum.textContent = (S.driftScore ?? 0).toFixed(3);
  if (epidStateVal) {
    const s = (S.seir || '').toLowerCase();
    // Compute threat level label from R0
    const r0 = S.epidParams?.R0 ?? S.epidSimData?.epidemic_params?.R0 ?? S.epidSimData?.R0 ?? null;
    const threatLevel = (r0 !== null && parseFloat(r0) >= 1.0) ? 'ENDEMIC'
                      : (r0 !== null && parseFloat(r0) >= 0.7) ? 'WATCH'
                      : 'CONTAINED';
    const tlColor = threatLevel === 'ENDEMIC' ? '#ff4d6a'
                  : threatLevel === 'WATCH' ? '#f5c842'
                  : '#00e59b';
    const stateLabel = (S.seir || 'SUSCEPTIBLE').toUpperCase();
    epidStateVal.innerHTML =
      `<span style="color:var(--text-muted);font-size:10px;display:block">AGENT COMPROMISE STATE</span>` +
      `<span>${stateLabel}</span>` +
      `<span style="color:${tlColor};font-size:10px;margin-left:8px;font-family:var(--font-terminal)">[${threatLevel}${r0 !== null ? ' R\u2080=' + parseFloat(r0).toFixed(2) : ''}]</span>`;
    epidStateVal.className = s === 'compromised' ? 'epid-state-val seir-c'
                           : (s === 'infected' || s === 'exposed') ? 'epid-state-val seir-i'
                           : s === 'recovered' ? 'epid-state-val seir-r'
                           : 'epid-state-val seir-s';
  }

  // ── Network topology ──
  const simData    = S.epidSimData;
  const agents     = simData?.agents || simData?.final_agents || [];
  const conns      = simData?.connections || simData?.topology || [];
  const agentKey   = agents.map(a => String(a.id ?? a.agent_id ?? '')).join(',');

  if (epidNodeCount) epidNodeCount.textContent = agents.length ? String(agents.length) : '—';

  if (agentKey !== _lastAgentKey) {
    _lastAgentKey = agentKey;
    _buildSVG(agents, conns);
  }

  // Sync per-node SEIR state
  for (const a of agents) {
    const id  = String(a.id ?? a.agent_id ?? '');
    const raw = (a.infection_state || a.state || a.status || 'susceptible').toLowerCase();
    _setNodeState(id, raw);
  }

  // ── Drift signals with threshold coloring ──
  if (epidSignals) {
    const signals = S.epidSignals || {};
    epidSignals.innerHTML = Object.entries(SIGNAL_LABELS).map(([key, label]) => {
      const val   = signals[key];
      const score = typeof val === 'number' ? val : null;
      const pct   = score !== null ? Math.min(100, Math.round(score * 100)) : null;
      const bar   = score === null ? '#4a5568'
                  : score > 0.30  ? '#ff4d6a'
                  : score > 0.15  ? '#f5c842'
                  : '#00e59b';
      return `<div class="prog-row">
        <span class="prog-label">${esc(label)}</span>
        <div class="prog-track">
          <div class="prog-fill" style="width:${pct ?? 0}%;background:${bar};` +
          `height:6px;border-radius:3px;transition:width 0.5s,background 0.5s"></div>
        </div>
        <span class="prog-val" style="color:${bar}">${pct != null ? pct + '%' : '—'}</span>
      </div>`;
    }).join('') || '<em style="color:var(--text-muted);font-size:11px">No signals</em>';
  }

  // ── SEIR parameters + findings ──
  if (epidParams) {
    const params = S.epidParams || simData?.epidemic_params || {};
    const findings = S.epidReport?.findings || [];
    const recs = S.epidReport?.recommendations || [];
    let html = Object.entries(params).slice(0, 8).map(([k, v]) =>
      `<div><span style="color:var(--text-muted)">${esc(k)}:</span> ${esc(String(v))}</div>`
    ).join('');
    if (findings.length) {
      html += `<div style="margin-top:8px;border-top:1px solid var(--border);padding-top:6px;color:var(--text-muted);font-size:10px">FINDINGS</div>`;
      html += findings.map(f => `<div style="font-size:10px;color:var(--text-secondary)">${esc(f)}</div>`).join('');
    }
    if (recs.length) {
      html += `<div style="margin-top:6px;border-top:1px solid var(--border);padding-top:4px;color:var(--text-muted);font-size:10px">RECOMMENDATIONS</div>`;
      html += recs.map(r => `<div style="font-size:10px;color:#f5c842">${esc(r)}</div>`).join('');
    }
    epidParams.innerHTML = html || '<em style="color:var(--text-muted);font-size:11px">No params</em>';
  }
}

export function onLiveEvent(ev) {
  const { color, tag, ts, msg } = _classifyEvent(ev);
  const [fromId, toId] = _pickNodes();
  if (fromId && toId) _spawnPacket(fromId, toId, color);
  const description = msg || (typeof ev === 'object' ? JSON.stringify(ev).slice(0, 80) : String(ev));
  _appendLog(ts || new Date().toISOString(), tag, color, description);
}

export function destroy() {
  if (_rafId)   { cancelAnimationFrame(_rafId); _rafId = null; }
  if (_bgTimer) { clearTimeout(_bgTimer);       _bgTimer = null; }
  _packets  = [];
  _packetQ  = [];
}
