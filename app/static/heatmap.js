// Virtualised canvas heatmap: odors (columns) x units (rows), with a column header
// that stays at the top and row labels that stay at the left while scrolling.
// Only the visible cells are drawn, so 10k+ rows are fine.
//
// model = { rows: [{label, group}], cols: [{label, group, sub}], values: Float32Array (row-major),
//           rowDendro, colDendro (optional, from cluster.js layoutDendrogram) }
// options = { cellW, cellH, lut, norm, nanColor, onHover(cellInfo|null) }
import { fmtNum } from './colormaps.js';

const FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

function cssVar(el, name, fallback) {
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}

export class Heatmap {
  constructor(container, options = {}) {
    this.container = container;
    this.options = Object.assign({ cellW: 16, cellH: 16, nanColor: '#e4e4e4', onHover: null }, options);
    this.model = null;
    this._raf = 0;
    this._hover = null;
    this._build();
  }

  _build() {
    const c = this.container;
    c.classList.add('hm');
    c.innerHTML = `
      <div class="hm-scroll" tabindex="0" aria-label="heatmap">
        <div class="hm-content">
          <div class="hm-head"><canvas></canvas></div>
          <div class="hm-view"><canvas class="hm-cells"></canvas><canvas class="hm-overlay"></canvas></div>
        </div>
      </div>
      <div class="hm-tooltip" hidden></div>`;
    this.scroll = c.querySelector('.hm-scroll');
    this.content = c.querySelector('.hm-content');
    this.head = c.querySelector('.hm-head');
    this.headCanvas = this.head.querySelector('canvas');
    this.view = c.querySelector('.hm-view');
    this.cellsCanvas = c.querySelector('.hm-cells');
    this.overlay = c.querySelector('.hm-overlay');
    this.tooltip = c.querySelector('.hm-tooltip');

    this._onScroll = () => { if (this._hover || !this.tooltip.hidden) { this._hover = null; this.tooltip.hidden = true; } this._schedule(); };
    this.scroll.addEventListener('scroll', this._onScroll, { passive: true });
    this._ro = new ResizeObserver(() => this._layout());
    this._ro.observe(this.scroll);
    this._onMove = (e) => this._mouse(e);
    this._onLeave = () => this._setHover(null);
    this.view.addEventListener('mousemove', this._onMove);
    this.view.addEventListener('mouseleave', this._onLeave);
    this.head.addEventListener('mousemove', (e) => this._mouseHead(e));
    this.head.addEventListener('mouseleave', this._onLeave);
  }

  destroy() {
    this._ro.disconnect();
    this.scroll.removeEventListener('scroll', this._onScroll);
    this.container.innerHTML = '';
  }

  setModel(model, options) {
    this.model = model;
    if (options) Object.assign(this.options, options);
    this._measure();
    this._layout();
  }

  setOptions(options) {
    const sizeChanged = (options.cellW && options.cellW !== this.options.cellW) || (options.cellH && options.cellH !== this.options.cellH);
    Object.assign(this.options, options);
    if (sizeChanged) this._measure();
    this._layout();
  }

  // --------------------------------------------------------------- layout
  _measure() {
    const m = this.model;
    if (!m) return;
    const ctx = this.headCanvas.getContext('2d');
    const { cellW, cellH } = this.options;
    this.rowLabelFont = `${Math.min(12, Math.max(8, cellH - 4))}px ${FONT_FAMILY}`;
    this.groupLabelFont = `11px ${FONT_FAMILY}`;
    this.colLabelFont = `${Math.min(12, Math.max(8, cellW - 4))}px ${FONT_FAMILY}`;
    this.showRowLabels = cellH >= 9;
    this.showColLabels = cellW >= 8;
    this.hasRowGroups = m.rows.some((r) => r.group != null);
    this.hasColGroups = m.cols.some((c) => c.group != null);
    this.hasColSubs = m.cols.some((c) => c.sub != null);

    ctx.font = this.rowLabelFont;
    let labelW = 0, groupW = 0;
    if (this.showRowLabels) for (const r of m.rows) labelW = Math.max(labelW, ctx.measureText(r.label).width);
    ctx.font = this.groupLabelFont;
    if (this.hasRowGroups) for (const r of m.rows) groupW = Math.max(groupW, ctx.measureText(r.group).width);
    this.rowGroupColW = this.hasRowGroups ? Math.min(150, Math.ceil(groupW) + 12) : 0;
    this.rowLabelColW = this.showRowLabels ? Math.min(160, Math.ceil(labelW) + 12) : 0;
    this.rowDendroW = m.rowDendro ? 56 : 0;
    this.labelW = this.rowDendroW + Math.max(this.rowGroupColW + this.rowLabelColW, this.hasRowGroups || this.showRowLabels ? 40 : 8);

    ctx.font = this.colLabelFont;
    let colW = 0;
    if (this.showColLabels) for (const c of m.cols) colW = Math.max(colW, ctx.measureText(c.label).width);
    this.colLabelBand = this.showColLabels ? Math.min(200, Math.ceil(colW) + 10) : 6;
    this.colGroupBand = this.hasColGroups ? 18 : 0;
    this.colSubBand = this.hasColSubs && cellW >= 10 ? 13 : 0;
    this.colDendroH = m.colDendro ? 44 : 0;
    this.headerH = this.colDendroH + this.colGroupBand + this.colLabelBand + this.colSubBand;

    // group runs (row and column) for separators and labels
    this.rowRuns = runs(m.rows.map((r) => r.group));
    this.colRuns = runs(m.cols.map((c) => c.group));
    this.colLabelRuns = runs(m.cols.map((c) => `${c.group}\u0000${c.label}`));
  }

  _layout() {
    const m = this.model;
    if (!m) return;
    const { cellW, cellH } = this.options;
    const totalW = this.labelW + m.cols.length * cellW;
    const rowsH = m.rows.length * cellH;
    this.content.style.width = `${totalW}px`;
    this.content.style.height = `${this.headerH + rowsH}px`;
    const clientW = this.scroll.clientWidth;
    const clientH = this.scroll.clientHeight;
    const viewW = Math.min(clientW, totalW);
    const viewH = Math.max(0, Math.min(clientH - this.headerH, rowsH));
    this.viewW = viewW; this.viewH = viewH;
    this.head.style.width = `${viewW}px`;
    this.head.style.height = `${this.headerH}px`;
    this.view.style.top = `${this.headerH}px`;
    this.view.style.width = `${viewW}px`;
    this.view.style.height = `${viewH}px`;
    const dpr = window.devicePixelRatio || 1;
    for (const [cv, w, h] of [[this.headCanvas, viewW, this.headerH], [this.cellsCanvas, viewW, viewH], [this.overlay, viewW, viewH]]) {
      cv.style.width = `${w}px`; cv.style.height = `${h}px`;
      cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.max(1, Math.round(h * dpr));
    }
    this._schedule();
  }

  _schedule() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }

  // ---------------------------------------------------------------- draw
  _draw() {
    const m = this.model;
    if (!m) return;
    const o = this.options;
    const { cellW, cellH } = o;
    const dpr = window.devicePixelRatio || 1;
    const sl = this.scroll.scrollLeft, st = this.scroll.scrollTop;
    const nRows = m.rows.length, nCols = m.cols.length;
    const surface = cssVar(this.container, '--hm-surface', '#ffffff');
    const ink = cssVar(this.container, '--hm-ink', '#222');
    const inkMuted = cssVar(this.container, '--hm-ink-muted', '#666');
    const rule = cssVar(this.container, '--hm-rule', '#d0d0d0');

    const firstCol = Math.max(0, Math.floor(sl / cellW));
    const lastCol = Math.min(nCols - 1, Math.ceil((sl + this.viewW - this.labelW) / cellW));
    const firstRow = Math.max(0, Math.floor(st / cellH));
    const lastRow = Math.min(nRows - 1, Math.ceil((st + this.viewH) / cellH));
    const nvc = Math.max(0, lastCol - firstCol + 1), nvr = Math.max(0, lastRow - firstRow + 1);
    this._visible = { firstCol, lastCol, firstRow, lastRow };

    // ---- cells
    const ctx = this.cellsCanvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, this.viewW, this.viewH);
    if (nvc > 0 && nvr > 0) {
      const img = ctx.createImageData(nvc, nvr);
      const px = img.data;
      const lut = o.lut, norm = o.norm;
      const nan = parseColor(o.nanColor);
      let k = 0;
      for (let r = 0; r < nvr; r++) {
        const base = (firstRow + r) * nCols + firstCol;
        for (let c = 0; c < nvc; c++) {
          const v = m.values[base + c];
          const t = norm.toUnit(v);
          if (t !== t) { px[k++] = nan[0]; px[k++] = nan[1]; px[k++] = nan[2]; px[k++] = 255; continue; }
          const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
          px[k++] = lut[i]; px[k++] = lut[i + 1]; px[k++] = lut[i + 2]; px[k++] = 255;
        }
      }
      const off = this._offscreen || (this._offscreen = document.createElement('canvas'));
      off.width = nvc; off.height = nvr;
      off.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      const x0 = this.labelW + firstCol * cellW - sl, y0 = firstRow * cellH - st;
      ctx.drawImage(off, 0, 0, nvc, nvr, x0, y0, nvc * cellW, nvr * cellH);

      // group separators (gaps in the surface colour)
      ctx.fillStyle = surface;
      if (this.hasColGroups) for (const run of this.colRuns) {
        if (run.start === 0) continue;
        const x = this.labelW + run.start * cellW - sl;
        if (x >= this.labelW - 2 && x <= this.viewW) ctx.fillRect(x - 1, 0, 2, this.viewH);
      }
      if (this.hasRowGroups) for (const run of this.rowRuns) {
        if (run.start === 0) continue;
        const y = run.start * cellH - st;
        if (y >= -2 && y <= this.viewH) ctx.fillRect(this.labelW, y - 1, this.viewW, 2);
      }
      if (this.hasColSubs && cellW >= 6) for (const run of this.colLabelRuns) {
        if (run.start === 0) continue;
        const x = this.labelW + run.start * cellW - sl;
        if (x >= this.labelW - 1 && x <= this.viewW) ctx.fillRect(x, 0, 1, this.viewH);
      }
    }

    // ---- row labels (opaque column at the left)
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, this.labelW, this.viewH);
    ctx.textBaseline = 'middle';
    if (this.showRowLabels) {
      ctx.font = this.rowLabelFont;
      ctx.fillStyle = ink;
      ctx.textAlign = 'right';
      for (let r = firstRow; r <= lastRow; r++) {
        const y = r * cellH - st + cellH / 2;
        ctx.fillText(ellipsize(ctx, m.rows[r].label, this.rowLabelColW - 8), this.labelW - 6, y);
      }
    }
    if (this.hasRowGroups) {
      ctx.font = this.groupLabelFont;
      ctx.textAlign = 'left';
      const x0 = this.rowDendroW;
      for (const run of this.rowRuns) {
        const top = run.start * cellH - st, bottom = run.end * cellH - st;
        if (bottom < 0 || top > this.viewH) continue;
        const visTop = Math.max(top, 0), visBottom = Math.min(bottom, this.viewH);
        if (run.start > 0) { ctx.fillStyle = rule; ctx.fillRect(x0, top, this.labelW - x0, 1); }
        const h = visBottom - visTop;
        if (h >= 11) {
          ctx.fillStyle = inkMuted;
          const label = ellipsize(ctx, String(run.key), this.rowGroupColW - 8);
          const y = h >= 24 ? visTop + 10 : visTop + h / 2;
          ctx.fillText(label, x0 + 6, y);
        }
      }
    }
    if (m.rowDendro) this._drawDendro(ctx, m.rowDendro, true, st, inkMuted);
    ctx.fillStyle = rule;
    ctx.fillRect(this.labelW - 1, 0, 1, this.viewH);

    // ---- header
    const hc = this.headCanvas.getContext('2d');
    hc.setTransform(dpr, 0, 0, dpr, 0, 0);
    hc.fillStyle = surface;
    hc.fillRect(0, 0, this.viewW, this.headerH);
    hc.save();
    hc.beginPath();
    hc.rect(this.labelW, 0, this.viewW - this.labelW, this.headerH);
    hc.clip();
    if (m.colDendro) this._drawDendro(hc, m.colDendro, false, sl, inkMuted);
    // panel / group band
    const bandY = this.colDendroH;
    if (this.hasColGroups) {
      hc.font = `600 11px ${FONT_FAMILY}`;
      hc.textBaseline = 'middle';
      hc.textAlign = 'center';
      for (const run of this.colRuns) {
        const x0 = this.labelW + run.start * cellW - sl, x1 = this.labelW + run.end * cellW - sl;
        if (x1 < this.labelW || x0 > this.viewW) continue;
        if (run.start > 0) { hc.fillStyle = rule; hc.fillRect(x0 - 1, bandY, 1, this.headerH - bandY); }
        const vis0 = Math.max(x0, this.labelW), vis1 = Math.min(x1, this.viewW);
        hc.fillStyle = inkMuted;
        const label = ellipsize(hc, String(run.key ?? ''), vis1 - vis0 - 6);
        if (label) hc.fillText(label, (vis0 + vis1) / 2, bandY + this.colGroupBand / 2 + 1);
      }
      hc.fillStyle = rule;
      hc.fillRect(this.labelW, bandY + this.colGroupBand - 1, this.viewW, 1);
    }
    // rotated odor labels (one per label run; a run spans the repeats of one odor)
    if (this.showColLabels) {
      hc.font = this.colLabelFont;
      hc.fillStyle = ink;
      hc.textAlign = 'left';
      hc.textBaseline = 'middle';
      const baseY = bandY + this.colGroupBand + this.colLabelBand - 5;
      for (const run of this.colLabelRuns) {
        if (run.end <= firstCol || run.start > lastCol + 1) continue;
        const xc = this.labelW + ((run.start + run.end) / 2) * cellW - sl;
        const label = ellipsize(hc, m.cols[run.start].label, this.colLabelBand - 10);
        hc.save();
        hc.translate(xc, baseY);
        hc.rotate(-Math.PI / 2);
        hc.fillText(label, 0, 0);
        hc.restore();
      }
    }
    if (this.colSubBand) {
      hc.font = `9px ${FONT_FAMILY}`;
      hc.fillStyle = inkMuted;
      hc.textAlign = 'center';
      hc.textBaseline = 'middle';
      const y = this.headerH - this.colSubBand / 2;
      for (let c = firstCol; c <= lastCol; c++) {
        const x = this.labelW + (c + 0.5) * cellW - sl;
        hc.fillText(String(m.cols[c].sub ?? ''), x, y);
      }
    }
    hc.restore();
    // corner
    hc.fillStyle = surface;
    hc.fillRect(0, 0, this.labelW, this.headerH);
    hc.fillStyle = rule;
    hc.fillRect(0, this.headerH - 1, this.viewW, 1);
    hc.fillRect(this.labelW - 1, 0, 1, this.headerH);
    if (this.hasRowGroups || this.showRowLabels) {
      hc.font = `11px ${FONT_FAMILY}`;
      hc.fillStyle = inkMuted;
      hc.textAlign = 'left';
      hc.textBaseline = 'bottom';
      const t = [m.rowGroupTitle, m.rowTitle].filter(Boolean).join(' / ');
      hc.fillText(ellipsize(hc, t, this.labelW - 10), 6, this.headerH - 6);
    }
    this._drawOverlay();
  }

  // Dendrogram in the band left of the row labels (rows) or above the column header,
  // leaves next to the labels; only segments in view are drawn.
  _drawDendro(ctx, d, rows, scroll, color) {
    const { cellW, cellH } = this.options;
    const cell = rows ? cellH : cellW;
    const origin = (rows ? 0 : this.labelW) - scroll;
    const lo = rows ? 0 : this.labelW, hi = rows ? this.viewH : this.viewW;
    const band = rows ? this.rowDendroW : this.colDendroH;
    const base = band - 3, scale = (band - 7) / d.maxH;
    const s = d.segs;
    ctx.save();
    ctx.beginPath();
    if (rows) ctx.rect(0, 0, band, this.viewH); else ctx.rect(this.labelW, 0, this.viewW - this.labelW, band);
    ctx.clip();
    ctx.beginPath();
    for (let i = 0; i < s.length; i += 4) {
      const p1 = origin + s[i] * cell, p2 = origin + s[i + 2] * cell;
      if ((p1 < lo && p2 < lo) || (p1 > hi && p2 > hi)) continue;
      const h1 = base - s[i + 1] * scale, h2 = base - s[i + 3] * scale;
      if (rows) { ctx.moveTo(h1, p1); ctx.lineTo(h2, p2); } else { ctx.moveTo(p1, h1); ctx.lineTo(p2, h2); }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  _drawOverlay() {
    const ctx = this.overlay.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.viewW, this.viewH);
    const h = this._hover;
    if (!h) return;
    const { cellW, cellH } = this.options;
    const x = this.labelW + h.col * cellW - this.scroll.scrollLeft;
    const y = h.row * cellH - this.scroll.scrollTop;
    if (x < this.labelW - cellW) return;
    ctx.strokeStyle = cssVar(this.container, '--hm-highlight', '#111');
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
    // faint crosshair along the row and column
    ctx.fillStyle = 'rgba(0,0,0,0.06)';
    ctx.fillRect(this.labelW, y, this.viewW - this.labelW, cellH);
    ctx.fillRect(x, 0, cellW, this.viewH);
  }

  // ---------------------------------------------------------------- hover
  _mouse(e) {
    const m = this.model;
    if (!m) return;
    const rect = this.view.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const { cellW, cellH } = this.options;
    const col = Math.floor((x - this.labelW + this.scroll.scrollLeft) / cellW);
    const row = Math.floor((y + this.scroll.scrollTop) / cellH);
    if (x < this.labelW) {
      // over the row labels: still show the row's identity
      if (row >= 0 && row < m.rows.length) this._setHover({ row, col: -1 }, e);
      else this._setHover(null);
      return;
    }
    if (col < 0 || col >= m.cols.length || row < 0 || row >= m.rows.length) { this._setHover(null); return; }
    this._setHover({ row, col }, e);
  }

  _mouseHead(e) {
    const m = this.model;
    if (!m) return;
    const rect = this.head.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const col = Math.floor((x - this.labelW + this.scroll.scrollLeft) / this.options.cellW);
    if (x < this.labelW || col < 0 || col >= m.cols.length) { this._setHover(null); return; }
    this._setHover({ row: -1, col }, e);
  }

  _setHover(h, e) {
    const prev = this._hover;
    this._hover = h && h.row >= 0 && h.col >= 0 ? h : null;
    if ((prev && prev.row) !== (this._hover && this._hover.row) || (prev && prev.col) !== (this._hover && this._hover.col)) this._drawOverlay();
    if (!h) { this.tooltip.hidden = true; if (this.options.onHover) this.options.onHover(null); return; }
    const m = this.model;
    const row = h.row >= 0 ? m.rows[h.row] : null, col = h.col >= 0 ? m.cols[h.col] : null;
    const v = h.row >= 0 && h.col >= 0 ? m.values[h.row * m.cols.length + h.col] : NaN;
    const info = { row, col, value: v, rowIndex: h.row, colIndex: h.col };
    const parts = [];
    if (col) parts.push(`<b>${esc(col.label)}</b>${col.sub != null ? ` <span class="muted">repeat ${esc(col.sub)}</span>` : ''}${col.group != null ? `<span class="muted"> in ${esc(col.group)}</span>` : ''}`);
    if (row) parts.push(`${row.group != null ? esc(row.group) + ' ' : ''}<b>${esc(row.label)}</b>`);
    if (row && col) parts.push(`<span class="val">${Number.isFinite(v) ? fmtNum(v) : 'missing'}</span>`);
    this.tooltip.innerHTML = parts.join('<br>');
    this.tooltip.hidden = false;
    const cw = this.container.getBoundingClientRect();
    let tx = e.clientX - cw.left + 14, ty = e.clientY - cw.top + 14;
    const tw = this.tooltip.offsetWidth, th = this.tooltip.offsetHeight;
    if (tx + tw > cw.width - 4) tx = e.clientX - cw.left - tw - 10;
    if (ty + th > cw.height - 4) ty = e.clientY - cw.top - th - 10;
    this.tooltip.style.left = `${tx}px`;
    this.tooltip.style.top = `${ty}px`;
    if (this.options.onHover) this.options.onHover(info);
  }
}

// ---------------------------------------------------------------- helpers
function runs(keys) {
  const out = [];
  let start = 0;
  for (let i = 1; i <= keys.length; i++) {
    if (i === keys.length || keys[i] !== keys[start]) {
      out.push({ key: keys[start], start, end: i });
      start = i;
    }
  }
  return out;
}

function ellipsize(ctx, text, maxW) {
  if (maxW <= 0) return '';
  if (ctx.measureText(text).width <= maxW) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(text.slice(0, mid) + '…').width <= maxW) lo = mid; else hi = mid - 1;
  }
  return lo > 0 ? text.slice(0, lo) + '…' : '';
}

export function parseColor(c) {
  if (c.startsWith('#')) {
    const n = parseInt(c.slice(1), 16);
    return c.length === 7 ? [(n >> 16) & 255, (n >> 8) & 255, n & 255] : [((n >> 8) & 15) * 17, ((n >> 4) & 15) * 17, (n & 15) * 17];
  }
  const m = c.match(/(\d+)[,\s]+(\d+)[,\s]+(\d+)/);
  return m ? [+m[1], +m[2], +m[3]] : [228, 228, 228];
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
}
