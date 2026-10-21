// Virtualised canvas heatmap: odors (columns) x units (rows), or the other way around,
// with a column header that stays at the top and row labels that stay at the left while
// scrolling. Only the visible cells are drawn, so 10k+ rows are fine.
//
// model = { rows: [{label, group, sub, key, detail (for the tooltip)}], cols: (the same), values: Float32Array (row-major),
//           rowDendro, colDendro (optional, from cluster.js layoutDendrogram),
//           rowTitle, rowGroupTitle, transposed (odors are the rows) }
// Neighbouring items with the same key (the trials of one odor) share one label; `sub`
// (a repeat number) is shown next to each.
// options = { cellW, cellH, lut, norm, nanColor, onHover(cellInfo|null),
//             reorder: {rows, cols} (labels of these may be dragged), onReorder(axis, keys),
//             onLabelMenu({axis, level, items, keys, x, y}) (right-click on a label),
//             onLabelRemove({axis, level, items, keys}) (ctrl-click on a label),
//             onCellClick({row, col, rowIndex, colIndex}) (a click on a cell),
//             minHeaderH (px: the header at least this high, to line up with another's) }
// A label dragged moves its items within their group, and a group label its group;
// onReorder then gets the axis's keys in their new order.
import { fmtNum } from './colormaps.js';

const FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const SUB_FONT = `9px ${FONT_FAMILY}`;
const DRAG_SLOP = 4; // px the pointer moves before a press on a label becomes a drag
const DRAG_EDGE = 28; // px from the edge of the view where a drag scrolls it

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
    this.mark = null; // {rows, cols}: indices of a row and a column shaded (see setMark)
    this._build();
  }

  _build() {
    const c = this.container;
    c.classList.add('hm');
    c.innerHTML = `
      <div class="hm-scroll" tabindex="0" aria-label="heatmap">
        <div class="hm-content">
          <div class="hm-head"><canvas class="hm-head-cells"></canvas><canvas class="hm-overlay hm-head-overlay"></canvas></div>
          <div class="hm-view"><canvas class="hm-cells"></canvas><canvas class="hm-overlay"></canvas></div>
        </div>
      </div>
      <div class="hm-tooltip" hidden></div>`;
    this.scroll = c.querySelector('.hm-scroll');
    this.content = c.querySelector('.hm-content');
    this.head = c.querySelector('.hm-head');
    this.headCanvas = c.querySelector('.hm-head-cells');
    this.headOverlay = c.querySelector('.hm-head-overlay');
    this.view = c.querySelector('.hm-view');
    this.cellsCanvas = c.querySelector('.hm-cells');
    this.overlay = c.querySelector('.hm-view .hm-overlay');
    this.tooltip = c.querySelector('.hm-tooltip');

    this._onScroll = () => {
      if (this._drag?.active) this._dragUpdate();
      else if (this._hover || !this.tooltip.hidden) { this._hover = null; this.tooltip.hidden = true; }
      this._schedule();
    };
    this.scroll.addEventListener('scroll', this._onScroll, { passive: true });
    this._ro = new ResizeObserver(() => this._layout());
    this._ro.observe(this.scroll);
    this._onMove = (e) => this._mouse(e);
    this._onLeave = () => { if (!this._drag?.active) this._setHover(null); };
    this.view.addEventListener('mousemove', this._onMove);
    this.view.addEventListener('mouseleave', this._onLeave);
    this.view.addEventListener('click', (e) => this._click(e));
    this.head.addEventListener('mousemove', (e) => this._mouseHead(e));
    this.head.addEventListener('mouseleave', this._onLeave);
    for (const el of [this.view, this.head]) {
      el.addEventListener('pointerdown', (e) => this._pointerDown(e, el));
      el.addEventListener('pointermove', (e) => this._dragMove(e));
      el.addEventListener('pointerup', (e) => this._dragEnd(e, true));
      el.addEventListener('pointercancel', (e) => this._dragEnd(e, false));
      el.addEventListener('lostpointercapture', (e) => this._dragEnd(e, false));
      el.addEventListener('contextmenu', (e) => this._labelMenu(e, el));
    }
    this._onKey = (e) => { if (e.key === 'Escape' && this._drag) this._dragEnd(e, false); };
    window.addEventListener('keydown', this._onKey);
  }

  destroy() {
    this._ro.disconnect();
    this.scroll.removeEventListener('scroll', this._onScroll);
    window.removeEventListener('keydown', this._onKey);
    this.container.innerHTML = '';
  }

  setModel(model, options) {
    if (this._drag) this._dragEnd({}, false); // its items are not these
    this.model = model;
    if (options) Object.assign(this.options, options);
    this._measure();
    this._layout();
  }

  setOptions(options) {
    const sizeChanged = (options.cellW && options.cellW !== this.options.cellW) || (options.cellH && options.cellH !== this.options.cellH);
    Object.assign(this.options, options);
    if (sizeChanged || 'minHeaderH' in options) this._measure();
    this._layout();
  }

  // a row and a column (indices, either null) shaded, as those hovered in another heatmap
  // of the same items
  setMark(mark) {
    this.mark = mark;
    if (this.model) this._drawOverlay();
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
    this.hasRowSubs = m.rows.some((r) => r.sub != null);

    // group runs (row and column) for separators and labels, and runs of items sharing a label
    this.rowRuns = runs(m.rows.map((r) => r.group));
    this.colRuns = runs(m.cols.map((c) => c.group));
    this.rowLabelRuns = runs(m.rows.map((r) => r.key));
    this.colLabelRuns = runs(m.cols.map((c) => c.key));
    this.rowRunOf = runIndex(this.rowRuns, m.rows.length);
    this.colRunOf = runIndex(this.colRuns, m.cols.length);
    this.rowLabelRunOf = runIndex(this.rowLabelRuns, m.rows.length);
    this.colLabelRunOf = runIndex(this.colLabelRuns, m.cols.length);

    ctx.font = this.rowLabelFont;
    let labelW = 0, groupW = 0, subW = 0;
    if (this.showRowLabels) for (const run of this.rowLabelRuns) labelW = Math.max(labelW, ctx.measureText(m.rows[run.start].label).width);
    ctx.font = this.groupLabelFont;
    if (this.hasRowGroups) for (const run of this.rowRuns) groupW = Math.max(groupW, ctx.measureText(String(run.key ?? '')).width);
    ctx.font = SUB_FONT;
    if (this.hasRowSubs) for (const r of m.rows) subW = Math.max(subW, ctx.measureText(String(r.sub ?? '')).width);
    this.rowGroupColW = this.hasRowGroups ? Math.min(150, Math.ceil(groupW) + 12) : 0;
    this.rowLabelColW = this.showRowLabels ? Math.min(160, Math.ceil(labelW) + 12) : 0;
    this.rowSubColW = this.hasRowSubs && cellH >= 10 ? Math.ceil(subW) + 8 : 0;
    this.rowDendroW = m.rowDendro ? 56 : 0;
    this.labelW = this.rowDendroW + Math.max(this.rowGroupColW + this.rowLabelColW + this.rowSubColW, this.hasRowGroups || this.showRowLabels ? 40 : 8);

    ctx.font = this.colLabelFont;
    let colW = 0;
    if (this.showColLabels) for (const run of this.colLabelRuns) colW = Math.max(colW, ctx.measureText(m.cols[run.start].label).width);
    this.colLabelBand = this.showColLabels ? Math.min(200, Math.ceil(colW) + 10) : 6;
    this.colGroupBand = this.hasColGroups ? 18 : 0;
    this.colSubBand = this.hasColSubs && cellW >= 10 ? 13 : 0;
    this.colDendroH = m.colDendro ? 44 : 0;
    this.naturalHeaderH = this.colDendroH + this.colGroupBand + this.colLabelBand + this.colSubBand;
    // any height more than its own above the column labels, which stay next to the cells
    this.colLabelBand += Math.max(0, (this.options.minHeaderH || 0) - this.naturalHeaderH);
    this.headerH = this.colDendroH + this.colGroupBand + this.colLabelBand + this.colSubBand;
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
    for (const [cv, w, h] of [[this.headCanvas, viewW, this.headerH], [this.headOverlay, viewW, this.headerH], [this.cellsCanvas, viewW, viewH], [this.overlay, viewW, viewH]]) {
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
      if (this.hasRowSubs && cellH >= 6) for (let k = this.rowLabelRunOf[firstRow] + 1; k <= this.rowLabelRunOf[lastRow]; k++) {
        ctx.fillRect(this.labelW, this.rowLabelRuns[k].start * cellH - st, this.viewW, 1);
      }

      // text in the cells (model.cellText(row, col): lines, the first larger), when they fit
      if (m.cellText && cellW >= CELL_TEXT_MIN.w && cellH >= CELL_TEXT_MIN.h) {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (let r = firstRow; r <= lastRow; r++) {
          for (let c = firstCol; c <= lastCol; c++) {
            const lines = cellLines(m.cellText(r, c), cellH);
            if (!lines.length) continue;
            const t = norm.toUnit(m.values[r * nCols + c]), i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
            ctx.fillStyle = textOn(t === t ? [lut[i], lut[i + 1], lut[i + 2]] : nan);
            const x = this.labelW + (c + 0.5) * cellW - sl, y = (r + 0.5) * cellH - st - ((lines.length - 1) * CELL_LINE) / 2;
            lines.forEach((line, k) => {
              ctx.font = `${k ? 9.5 : 11.5}px ${FONT_FAMILY}`;
              ctx.fillText(ellipsize(ctx, line, cellW - 4), x, y + k * CELL_LINE);
            });
          }
        }
        ctx.textAlign = 'start';
      }
    }

    // ---- row labels (opaque column at the left)
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, this.labelW, this.viewH);
    ctx.textBaseline = 'middle';
    if (this.showRowLabels && nvr > 0) {
      ctx.font = this.rowLabelFont;
      ctx.fillStyle = ink;
      ctx.textAlign = 'right';
      for (let k = this.rowLabelRunOf[firstRow]; k <= this.rowLabelRunOf[lastRow]; k++) {
        const run = this.rowLabelRuns[k];
        const y = ((run.start + run.end) / 2) * cellH - st;
        ctx.fillText(ellipsize(ctx, m.rows[run.start].label, this.rowLabelColW - 8), this.labelW - this.rowSubColW - 6, y);
      }
    }
    if (this.rowSubColW && nvr > 0) {
      ctx.font = SUB_FONT;
      ctx.fillStyle = inkMuted;
      ctx.textAlign = 'center';
      const x = this.labelW - this.rowSubColW / 2 - 1;
      for (let r = firstRow; r <= lastRow; r++) ctx.fillText(String(m.rows[r].sub ?? ''), x, r * cellH - st + cellH / 2);
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
      hc.font = SUB_FONT;
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
    const dpr = window.devicePixelRatio || 1;
    const hc = this.headOverlay.getContext('2d');
    hc.setTransform(dpr, 0, 0, dpr, 0, 0);
    hc.clearRect(0, 0, this.viewW, this.headerH);
    const ctx = this.overlay.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.viewW, this.viewH);
    if (this._drag?.active) { this._drawDrag(ctx, hc); return; }
    const { cellW, cellH } = this.options;
    const mk = this.mark;
    if (mk) {
      ctx.fillStyle = hc.fillStyle = 'rgba(0,0,0,0.1)';
      if (mk.rows != null) ctx.fillRect(0, mk.rows * cellH - this.scroll.scrollTop, this.viewW, cellH);
      const x = this.labelW + mk.cols * cellW - this.scroll.scrollLeft;
      if (mk.cols != null && x > this.labelW - cellW) {
        ctx.fillRect(x, 0, cellW, this.viewH);
        hc.fillRect(x, 0, cellW, this.headerH);
      }
    }
    const h = this._hover;
    if (!h) return;
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

  // the items being dragged shaded, and a line where they would go
  _drawDrag(ctx, hc) {
    const d = this._drag;
    const { cellW, cellH } = this.options;
    const color = cssVar(this.container, '--hm-drag', '#0b6e7a');
    const { start, end } = d.run;
    ctx.fillStyle = hc.fillStyle = color;
    if (d.axis === 'rows') {
      const st = this.scroll.scrollTop;
      const x0 = this.rowDendroW + (d.level === 'group' ? 0 : this.rowGroupColW);
      ctx.globalAlpha = 0.16;
      ctx.fillRect(x0, start * cellH - st, this.viewW - x0, (end - start) * cellH);
      ctx.globalAlpha = 1;
      if (d.to != null) ctx.fillRect(x0, d.to * cellH - st - 1, this.viewW - x0, 2);
      return;
    }
    const sl = this.scroll.scrollLeft;
    const y0 = this.colDendroH + (d.level === 'group' ? 0 : this.colGroupBand);
    const x = this.labelW + start * cellW - sl, w = (end - start) * cellW;
    for (const [c, top, h] of [[hc, y0, this.headerH - y0], [ctx, 0, this.viewH]]) {
      c.save();
      c.beginPath();
      c.rect(this.labelW, 0, this.viewW - this.labelW, top + h);
      c.clip();
      c.globalAlpha = 0.16;
      c.fillRect(x, top, w, h);
      c.globalAlpha = 1;
      if (d.to != null) c.fillRect(this.labelW + d.to * cellW - sl - 1, top, 2, h);
      c.restore();
    }
  }

  // ---------------------------------------------------------------- hover
  _mouse(e) {
    const m = this.model;
    if (!m || this._drag?.active) return;
    const rect = this.view.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const { cellW, cellH } = this.options;
    const col = Math.floor((x - this.labelW + this.scroll.scrollLeft) / cellW);
    const row = Math.floor((y + this.scroll.scrollTop) / cellH);
    this._grabCursor(this.view, x, y);
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
    if (!m || this._drag?.active) return;
    const rect = this.head.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const col = Math.floor((x - this.labelW + this.scroll.scrollLeft) / this.options.cellW);
    this._grabCursor(this.head, x, e.clientY - rect.top);
    if (x < this.labelW || col < 0 || col >= m.cols.length) { this._setHover(null); return; }
    this._setHover({ row: -1, col }, e);
  }

  _grabCursor(el, x, y) {
    const hit = this._labelAt(el, x, y);
    el.style.cursor = hit && this.options.reorder?.[hit.axis] ? 'grab' : this.options.onCellClick && this._cellAt(el, x, y) ? 'pointer' : '';
  }

  // the cell at (x, y) in the view: {rowIndex, colIndex}, or null (labels, outside the matrix)
  _cellAt(el, x, y) {
    const m = this.model;
    if (el !== this.view || !m || x < this.labelW) return null;
    const colIndex = Math.floor((x - this.labelW + this.scroll.scrollLeft) / this.options.cellW);
    const rowIndex = Math.floor((y + this.scroll.scrollTop) / this.options.cellH);
    return colIndex >= 0 && colIndex < m.cols.length && rowIndex >= 0 && rowIndex < m.rows.length ? { rowIndex, colIndex } : null;
  }

  // a click on a cell, but for the one ending a drag (or removing a label, with ctrl)
  _click(e) {
    if (!this.options.onCellClick || e.ctrlKey || performance.now() - (this._dragEndedAt || 0) < 300) return;
    const rect = this.view.getBoundingClientRect();
    const at = this._cellAt(this.view, e.clientX - rect.left, e.clientY - rect.top);
    if (at) this.options.onCellClick({ ...at, row: this.model.rows[at.rowIndex], col: this.model.cols[at.colIndex] });
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
    // the odor first, then the unit, whichever way round they are shown
    const [odor, unit] = m.transposed ? [row, col] : [col, row];
    const parts = [];
    const detail = (it) => (it.detail ? ` <span class="muted">${esc(it.detail)}</span>` : '');
    if (odor) parts.push(`<b>${esc(odor.label)}</b>${odor.sub != null ? ` <span class="muted">repeat ${esc(odor.sub)}</span>` : ''}${odor.group != null ? `<span class="muted"> in ${esc(odor.group)}</span>` : ''}${detail(odor)}`);
    if (unit) parts.push(`${unit.group != null ? esc(unit.group) + ' ' : ''}<b>${esc(unit.label)}</b>${detail(unit)}`);
    if (row && col) parts.push(`<span class="val">${Number.isFinite(v) ? fmtNum(v) : 'missing'}</span>`);
    this._tooltipAt(parts.join('<br>'), e);
    if (this.options.onHover) this.options.onHover(info);
  }

  _tooltipAt(html, e) {
    this.tooltip.innerHTML = html;
    this.tooltip.hidden = false;
    const cw = this.container.getBoundingClientRect();
    let tx = e.clientX - cw.left + 14, ty = e.clientY - cw.top + 14;
    const tw = this.tooltip.offsetWidth, th = this.tooltip.offsetHeight;
    if (tx + tw > cw.width - 4) tx = e.clientX - cw.left - tw - 10;
    if (ty + th > cw.height - 4) ty = e.clientY - cw.top - th - 10;
    this.tooltip.style.left = `${tx}px`;
    this.tooltip.style.top = `${ty}px`;
  }

  // ------------------------------------------------------------ labels
  // The label at (x, y) in the row labels (el: the view) or the column header:
  // {axis, level: 'item' | 'group', run: the items it labels} or null (repeat numbers,
  // dendrograms and hidden labels are not labels here).
  _labelAt(el, x, y) {
    const m = this.model;
    if (!m) return null;
    const { cellW, cellH } = this.options;
    if (el === this.view) {
      const i = Math.floor((y + this.scroll.scrollTop) / cellH);
      if (x >= this.labelW || i < 0 || i >= m.rows.length) return null;
      const labels = this.rowDendroW + this.rowGroupColW;
      if (this.hasRowGroups && x >= this.rowDendroW && x < labels) return { axis: 'rows', level: 'group', run: this.rowRuns[this.rowRunOf[i]] };
      if (this.showRowLabels && x >= labels && x < this.labelW - this.rowSubColW) return { axis: 'rows', level: 'item', run: this.rowLabelRuns[this.rowLabelRunOf[i]] };
      return null;
    }
    const j = Math.floor((x - this.labelW + this.scroll.scrollLeft) / cellW);
    if (x < this.labelW || j < 0 || j >= m.cols.length) return null;
    const labels = this.colDendroH + this.colGroupBand;
    if (this.hasColGroups && y >= this.colDendroH && y < labels) return { axis: 'cols', level: 'group', run: this.colRuns[this.colRunOf[j]] };
    if (this.showColLabels && y >= labels && y < labels + this.colLabelBand) return { axis: 'cols', level: 'item', run: this.colLabelRuns[this.colLabelRunOf[j]] };
    return null;
  }

  // the label under a pointer event, with the items it labels: {axis, level, items, keys}
  _labelHit(e, el) {
    const rect = el.getBoundingClientRect();
    const hit = this._labelAt(el, e.clientX - rect.left, e.clientY - rect.top);
    if (!hit) return null;
    const items = this.model[hit.axis].slice(hit.run.start, hit.run.end);
    return { axis: hit.axis, level: hit.level, items, keys: uniqueKeys(items) };
  }

  _labelMenu(e, el) {
    // macOS makes a right click of a ctrl-click, whose label was removed already
    if (this._removedByCtrl) { this._removedByCtrl = false; e.preventDefault(); return; }
    if (!this.options.onLabelMenu || this._drag?.active) return;
    const hit = this._labelHit(e, el);
    if (!hit) return;
    e.preventDefault();
    this._setHover(null);
    this.options.onLabelMenu({ ...hit, x: e.clientX, y: e.clientY });
  }

  // ctrl-click (with either button) on a label removes what it labels; a press on one
  // that may be dragged starts a drag
  _pointerDown(e, el) {
    this._removedByCtrl = false;
    if (!e.ctrlKey) { this._dragStart(e, el); return; }
    if (e.button !== 0 && e.button !== 2) return;
    const hit = this.options.onLabelRemove && !this._drag && this._labelHit(e, el);
    if (!hit) return;
    e.preventDefault();
    this._removedByCtrl = true;
    this._setHover(null);
    this.options.onLabelRemove(hit);
  }

  // ------------------------------------------------------------- drag
  _dragStart(e, el) {
    if (e.button !== 0 || e.metaKey || this._drag) return;
    const rect = el.getBoundingClientRect();
    const hit = this._labelAt(el, e.clientX - rect.left, e.clientY - rect.top);
    if (!hit || !this.options.reorder?.[hit.axis]) return;
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    // where it may go: between the labels of its group, or between groups
    const n = this.model[hit.axis].length;
    const groups = hit.axis === 'rows' ? this.rowRuns : this.colRuns;
    const groupOf = hit.axis === 'rows' ? this.rowRunOf : this.colRunOf;
    const within = hit.level === 'group' ? { start: 0, end: n } : groups[groupOf[hit.run.start]];
    const slots = hit.level === 'group' ? groups : hit.axis === 'rows' ? this.rowLabelRuns : this.colLabelRuns;
    const slotOf = hit.level === 'group' ? groupOf : hit.axis === 'rows' ? this.rowLabelRunOf : this.colLabelRunOf;
    this._drag = { ...hit, el, pointerId: e.pointerId, x0: e.clientX, y0: e.clientY, cx: e.clientX, cy: e.clientY, within, slots, slotOf, active: false, to: null, raf: 0 };
  }

  _dragMove(e) {
    const d = this._drag;
    if (!d || e.pointerId !== d.pointerId) return;
    d.cx = e.clientX; d.cy = e.clientY;
    if (!d.active) {
      if (Math.hypot(d.cx - d.x0, d.cy - d.y0) < DRAG_SLOP) return;
      d.active = true;
      d.el.style.cursor = 'grabbing';
      this._hover = null;
      d.raf = requestAnimationFrame(() => this._dragScroll());
    }
    this._dragUpdate();
  }

  // the boundary between labels nearest the pointer, and the label next to the pointer
  _dragUpdate() {
    const d = this._drag;
    const rect = this.view.getBoundingClientRect();
    const p = d.axis === 'rows'
      ? (d.cy - rect.top + this.scroll.scrollTop) / this.options.cellH
      : (d.cx - rect.left - this.labelW + this.scroll.scrollLeft) / this.options.cellW;
    const { start, end } = d.within;
    if (p <= start) d.to = start;
    else if (p >= end) d.to = end;
    else { const s = d.slots[d.slotOf[Math.floor(p)]]; d.to = p - s.start < s.end - p ? s.start : s.end; }
    this._drawOverlay();
    const it = this.model[d.axis][d.run.start];
    this._tooltipAt(`<b>${esc(d.level === 'group' ? it.group : it.label)}</b>`, { clientX: d.cx, clientY: d.cy });
  }

  // scrolls while the pointer is near the edge of the view, in the direction of the drag
  _dragScroll() {
    const d = this._drag;
    if (!d?.active) return;
    const r = this.view.getBoundingClientRect();
    const speed = (inside) => Math.min(40, Math.max(2, (DRAG_EDGE - inside) / 2));
    if (d.axis === 'rows') {
      if (d.cy < r.top + DRAG_EDGE) this.scroll.scrollTop -= speed(d.cy - r.top);
      else if (d.cy > r.top + this.viewH - DRAG_EDGE) this.scroll.scrollTop += speed(r.top + this.viewH - d.cy);
    } else {
      if (d.cx < r.left + this.labelW + DRAG_EDGE) this.scroll.scrollLeft -= speed(d.cx - r.left - this.labelW);
      else if (d.cx > r.left + this.viewW - DRAG_EDGE) this.scroll.scrollLeft += speed(r.left + this.viewW - d.cx);
    }
    d.raf = requestAnimationFrame(() => this._dragScroll());
  }

  _dragEnd(e, drop) {
    const d = this._drag;
    if (!d || (e.pointerId != null && e.pointerId !== d.pointerId)) return;
    this._drag = null;
    cancelAnimationFrame(d.raf);
    if (d.el.hasPointerCapture(d.pointerId)) d.el.releasePointerCapture(d.pointerId);
    d.el.style.cursor = '';
    if (!d.active) return;
    this._dragEndedAt = performance.now();
    this.tooltip.hidden = true;
    this._drawOverlay();
    const { start, end } = d.run;
    if (drop && d.to != null && d.to !== start && d.to !== end && this.options.onReorder) {
      const items = this.model[d.axis];
      const moved = items.slice(start, end), rest = items.slice(0, start).concat(items.slice(end));
      rest.splice(d.to < start ? d.to : d.to - (end - start), 0, ...moved);
      this.options.onReorder(d.axis, uniqueKeys(rest));
    }
  }
}

// ---------------------------------------------------------------- helpers
// text in cells: shown when they are at least this large; its lines' spacing
export const CELL_TEXT_MIN = { w: 26, h: 14 };
const CELL_LINE = 12;

// the lines of a cell's text that fit in its height
export function cellLines(text, cellH) {
  if (text == null || text === '') return [];
  return String(text).split('\n').slice(0, Math.max(1, Math.floor((cellH - 2) / CELL_LINE)));
}

// a colour to write on a cell of colour rgb ([r, g, b], 0-255): dark on light, white on dark
export function textOn(rgb) {
  const lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
  return lum > 0.55 ? '#1d1f22' : '#ffffff';
}

function uniqueKeys(items) {
  const seen = new Set();
  for (const it of items) seen.add(it.key);
  return [...seen];
}

// for each item, the index of the run it is in
function runIndex(rs, n) {
  const of = new Int32Array(n);
  rs.forEach((r, k) => of.fill(k, r.start, r.end));
  return of;
}

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
