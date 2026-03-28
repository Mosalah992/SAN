/* ─── Risk Heatmap — Canvas-based 5D risk visualization ──── */

const DIMENSIONS = [
  { key: 'injection',              label: 'Injection',  color: '#ff2d7b' },
  { key: 'authority_manipulation', label: 'Authority',  color: '#ff8800' },
  { key: 'emotional_coercion',     label: 'Emotional',  color: '#ffb800' },
  { key: 'obfuscation',           label: 'Obfuscation', color: '#a855f7' },
  { key: 'long_term_influence',    label: 'Influence',  color: '#3b82f6' },
];

const ROW_H = 22;
const LABEL_W = 88;
const PAD = 4;

/** Column count scales with data so cells stay visible (was fixed 100 → tiny slivers). */
function heatmapCols(vectorCount) {
  if (!vectorCount) return 24;
  return Math.min(96, Math.max(20, vectorCount + 4));
}

export function render(canvasEl, vectors) {
  if (!canvasEl) return;
  const ctx = canvasEl.getContext('2d');
  if (!ctx) return;

  const raw = Array.isArray(vectors) ? vectors : [];
  const rows = DIMENSIONS.length;
  const COLS = heatmapCols(raw.length);
  const totalH = rows * ROW_H + PAD * 2;
  const totalW = canvasEl.parentElement?.clientWidth || 520;
  canvasEl.width = Math.floor(totalW);
  canvasEl.height = totalH;

  const cellW = (totalW - LABEL_W - PAD * 2) / COLS;

  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, totalW, totalH);

  if (!raw.length) {
    ctx.fillStyle = '#5a5a78';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textBaseline = 'middle';
    ctx.fillText('No risk timeline yet — security events will appear here', LABEL_W + 8, totalH / 2);
    for (let r = 0; r < rows; r++) {
      const y = PAD + r * ROW_H;
      ctx.fillStyle = '#444460';
      ctx.fillText(DIMENSIONS[r].label, 4, y + ROW_H / 2);
    }
    return;
  }

  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'middle';

  for (let r = 0; r < rows; r++) {
    const y = PAD + r * ROW_H;
    ctx.fillStyle = '#8b8ba8';
    ctx.fillText(DIMENSIONS[r].label, 4, y + ROW_H / 2);
  }

  const data = raw.slice(-COLS);
  const padCount = COLS - data.length;

  for (let r = 0; r < rows; r++) {
    const dim = DIMENSIONS[r];
    const y = PAD + r * ROW_H;

    for (let c = 0; c < COLS; c++) {
      const x = LABEL_W + PAD + c * cellW;
      const idx = c - padCount;

      if (idx < 0) {
        ctx.fillStyle = '#12141c';
        ctx.fillRect(x, y + 1, Math.max(1, cellW - 1), ROW_H - 2);
        continue;
      }

      const row = data[idx] || {};
      const val = Math.min(1, Math.max(0, Number(row[dim.key]) || 0));

      if (val < 0.02) {
        ctx.fillStyle = '#161a24';
      } else {
        const r_c = parseInt(dim.color.slice(1, 3), 16);
        const g_c = parseInt(dim.color.slice(3, 5), 16);
        const b_c = parseInt(dim.color.slice(5, 7), 16);
        const alpha = Math.min(0.92, 0.18 + val * 0.82);
        ctx.fillStyle = `rgba(${r_c},${g_c},${b_c},${alpha})`;
      }

      ctx.fillRect(x, y + 1, Math.max(1, cellW - 1), ROW_H - 2);
    }
  }
}

export function renderTooltip(canvasEl, vectors, mouseX, mouseY) {
  if (!canvasEl || !vectors || !vectors.length) return null;

  const raw = vectors;
  const COLS = heatmapCols(raw.length);
  const rect = canvasEl.getBoundingClientRect();
  const x = mouseX - rect.left;
  const y = mouseY - rect.top;

  const cellW = (canvasEl.width - LABEL_W - PAD * 2) / COLS;
  const col = Math.floor((x - LABEL_W - PAD) / cellW);
  const row = Math.floor((y - PAD) / ROW_H);

  if (col < 0 || col >= COLS || row < 0 || row >= DIMENSIONS.length) return null;

  const data = raw.slice(-COLS);
  const padCount = COLS - data.length;
  const idx = col - padCount;
  if (idx < 0) return null;

  const v = data[idx] || {};
  const dim = DIMENSIONS[row];
  return {
    dimension: dim.label,
    value: (Number(v[dim.key]) || 0).toFixed(3),
    total: (Number(v.total) || 0).toFixed(3),
    ts: v.ts || '—',
  };
}
