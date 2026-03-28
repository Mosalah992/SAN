/* ─── Knowledge Tab — Force-Directed Knowledge Graph ─────── */
import * as api from '../api.js';

function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

let _canvas = null;
let _ctx = null;
let _nodes = [];
let _edges = [];
let _rafId = null;
let _selectedNode = null;
let _dragNode = null;
let _panX = 0, _panY = 0;
let _zoom = 1;
let _mouseX = 0, _mouseY = 0;
let _isDragging = false;
let _settled = false;
let _frameCount = 0;

const NODE_R = 8;
const SPRING_K = 0.008;
const REPULSION = 800;
const DAMPING = 0.88;
const SETTLE_FRAMES = 300;

export function init() {
  _canvas = document.getElementById('kg-canvas');
  if (!_canvas) return;
  _ctx = _canvas.getContext('2d');

  _canvas.addEventListener('mousemove', _onMouseMove);
  _canvas.addEventListener('mousedown', _onMouseDown);
  _canvas.addEventListener('mouseup', _onMouseUp);
  _canvas.addEventListener('wheel', _onWheel, { passive: false });
  _canvas.addEventListener('click', _onClick);

  _startLoop();
}

export async function refresh() {
  // Let the tab layout (display:grid / flex) commit so canvas parent has non-zero size.
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  try {
    const data = await api.fetchKnowledgeGraph();
    const countEl = document.getElementById('kg-count');
    if (!data?.ok) {
      if (countEl) countEl.textContent = '—';
      console.warn('[knowledge graph]', data?.error || 'request failed');
      return;
    }
    const nodes = Array.isArray(data.nodes) ? data.nodes : [];
    const edges = Array.isArray(data.edges) ? data.edges : [];
    if (countEl) countEl.textContent = String(data.count ?? nodes.length);
    _initGraph(nodes, edges);
  } catch (e) {
    console.warn('[knowledge graph]', e?.message || e);
    const countEl = document.getElementById('kg-count');
    if (countEl) countEl.textContent = '—';
  }
}

export function destroy() {
  if (_rafId) cancelAnimationFrame(_rafId);
  _rafId = null;
}

function _initGraph(nodes, edges) {
  const W = _canvas?.width || 600;
  const H = _canvas?.height || 400;

  // Only reinit positions if graph changed
  const nodeKey = nodes.map(n => n.id).join(',');
  if (_nodes.length && _nodes.map(n => n.id).join(',') === nodeKey) {
    // Just update metadata
    for (const n of nodes) {
      const existing = _nodes.find(e => e.id === n.id);
      if (existing) Object.assign(existing, { frequency: n.frequency, last_seen: n.last_seen, sources: n.sources });
    }
    _edges = edges;
    return;
  }

  const maxFreq = Math.max(1, ...nodes.map(n => n.frequency));

  _nodes = nodes.map((n) => ({
    ...n,
    x: W / 2 + (Math.random() - 0.5) * W * 0.6,
    y: H / 2 + (Math.random() - 0.5) * H * 0.6,
    vx: 0, vy: 0,
    r: NODE_R + (n.frequency / maxFreq) * 12,
    color: _ageColor(n.last_seen),
  }));
  _edges = edges;
  _settled = false;
  _frameCount = 0;
  _panX = 0;
  _panY = 0;
  _zoom = 1;
}

function _ageColor(lastSeen) {
  if (!lastSeen) return '#444460';
  try {
    const age = (Date.now() - new Date(lastSeen).getTime()) / (1000 * 3600);
    if (age < 24) return '#00fff2';      // hot — last 24h
    if (age < 168) return '#39ff14';     // warm — last week
    if (age < 720) return '#ffb800';     // cooling — last month
    return '#ff2d7b';                     // stale — older
  } catch {
    return '#444460';
  }
}

function _startLoop() {
  function tick() {
    _rafId = requestAnimationFrame(tick);
    if (!_ctx || !_canvas) return;

    // Resize canvas — avoid 0×0 (invisible graph) when flex/grid hasn’t resolved yet
    const rect = _canvas.parentElement?.getBoundingClientRect();
    if (rect) {
      const w = Math.max(280, Math.floor(rect.width));
      const h = Math.max(220, Math.floor(rect.height));
      if (_canvas.width !== w || _canvas.height !== h) {
        _canvas.width = w;
        _canvas.height = h;
        _settled = false;
        _frameCount = 0;
      }
    }

    if (!_settled) {
      _physics();
      _frameCount++;
      if (_frameCount > SETTLE_FRAMES) _settled = true;
    }

    _render();
  }
  tick();
}

function _physics() {
  const nodes = _nodes;
  // Repulsion
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      let dx = nodes[j].x - nodes[i].x;
      let dy = nodes[j].y - nodes[i].y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      let force = REPULSION / (dist * dist);
      let fx = (dx / dist) * force;
      let fy = (dy / dist) * force;
      nodes[i].vx -= fx; nodes[i].vy -= fy;
      nodes[j].vx += fx; nodes[j].vy += fy;
    }
  }

  // Springs (edges)
  const nodeMap = {};
  for (const n of nodes) nodeMap[n.id] = n;

  for (const e of _edges) {
    const a = nodeMap[e.source], b = nodeMap[e.target];
    if (!a || !b) continue;
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dist = Math.sqrt(dx * dx + dy * dy) || 1;
    let targetDist = 100 + 50 / (e.weight || 1);
    let force = SPRING_K * (dist - targetDist);
    let fx = (dx / dist) * force;
    let fy = (dy / dist) * force;
    a.vx += fx; a.vy += fy;
    b.vx -= fx; b.vy -= fy;
  }

  // Center gravity
  const W = _canvas.width, H = _canvas.height;
  for (const n of nodes) {
    n.vx += (W / 2 - n.x) * 0.0003;
    n.vy += (H / 2 - n.y) * 0.0003;
    n.vx *= DAMPING;
    n.vy *= DAMPING;
    if (n !== _dragNode) {
      n.x += n.vx;
      n.y += n.vy;
    }
  }
}

function _render() {
  const ctx = _ctx, W = _canvas.width, H = _canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.translate(_panX, _panY);
  ctx.scale(_zoom, _zoom);

  // Edges
  const nodeMap = {};
  for (const n of _nodes) nodeMap[n.id] = n;

  for (const e of _edges) {
    const a = nodeMap[e.source], b = nodeMap[e.target];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = `rgba(255,255,255,${Math.min(0.15, 0.03 * (e.weight || 1))})`;
    ctx.lineWidth = Math.min(2, 0.5 + (e.weight || 1) * 0.3);
    ctx.stroke();
  }

  // Nodes
  for (const n of _nodes) {
    const isHover = _isHover(n);
    const isSelected = _selectedNode === n;

    // Glow
    if (isHover || isSelected) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 6, 0, Math.PI * 2);
      ctx.fillStyle = n.color + '22';
      ctx.fill();
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = isHover || isSelected ? n.color : n.color + '88';
    ctx.fill();
    ctx.strokeStyle = n.color;
    ctx.lineWidth = isSelected ? 2 : 0.5;
    ctx.stroke();

    // Label
    if (n.r > 10 || isHover || isSelected) {
      ctx.font = `${isHover ? 11 : 9}px "JetBrains Mono", monospace`;
      ctx.fillStyle = isHover || isSelected ? '#e0e0f0' : '#8888aa';
      ctx.textAlign = 'center';
      ctx.fillText(n.label, n.x, n.y + n.r + 12);
    }
  }

  ctx.restore();

  // Detail panel for selected node
  if (_selectedNode) {
    _renderDetailOverlay(_selectedNode);
  }
}

function _renderDetailOverlay(node) {
  const el = document.getElementById('kg-detail');
  if (!el) return;
  el.classList.remove('hidden');
  el.innerHTML = `
    <div class="kg-detail-title">${esc(node.label)}</div>
    <div class="kg-detail-stat">Frequency: <span>${node.frequency}</span></div>
    <div class="kg-detail-stat">Last seen: <span>${esc(node.last_seen ? node.last_seen.replace('T', ' ').slice(0, 19) : '\u2014')}</span></div>
    <div class="kg-detail-stat">Sources: <span>${esc((node.sources || []).join(', ') || '\u2014')}</span></div>
    <div class="kg-detail-stat">Connections: <span>${_edges.filter(e => e.source === node.id || e.target === node.id).length}</span></div>
  `;
}

function _canvasToWorld(cx, cy) {
  return { x: (cx - _panX) / _zoom, y: (cy - _panY) / _zoom };
}

function _isHover(node) {
  const wp = _canvasToWorld(_mouseX, _mouseY);
  const dx = wp.x - node.x, dy = wp.y - node.y;
  return dx * dx + dy * dy < (node.r + 4) * (node.r + 4);
}

function _findNodeAt(cx, cy) {
  const wp = _canvasToWorld(cx, cy);
  for (const n of _nodes) {
    const dx = wp.x - n.x, dy = wp.y - n.y;
    if (dx * dx + dy * dy < (n.r + 4) * (n.r + 4)) return n;
  }
  return null;
}

function _onMouseMove(e) {
  const rect = _canvas.getBoundingClientRect();
  _mouseX = e.clientX - rect.left;
  _mouseY = e.clientY - rect.top;

  if (_dragNode) {
    const wp = _canvasToWorld(_mouseX, _mouseY);
    _dragNode.x = wp.x;
    _dragNode.y = wp.y;
    _dragNode.vx = 0;
    _dragNode.vy = 0;
    _settled = false;
    _frameCount = SETTLE_FRAMES - 30;
  } else if (_isDragging) {
    _panX += e.movementX;
    _panY += e.movementY;
  }

  _canvas.style.cursor = _findNodeAt(_mouseX, _mouseY) ? 'pointer' : 'grab';
}

function _onMouseDown(e) {
  const node = _findNodeAt(e.offsetX, e.offsetY);
  if (node) {
    _dragNode = node;
  } else {
    _isDragging = true;
  }
}

function _onMouseUp() {
  _dragNode = null;
  _isDragging = false;
}

function _onClick(e) {
  const node = _findNodeAt(e.offsetX, e.offsetY);
  if (node) {
    _selectedNode = (_selectedNode === node) ? null : node;
    if (!_selectedNode) {
      const el = document.getElementById('kg-detail');
      if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
    }
  } else {
    _selectedNode = null;
    const el = document.getElementById('kg-detail');
    if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
  }
}

function _onWheel(e) {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  _zoom = Math.max(0.3, Math.min(3, _zoom * delta));
}
