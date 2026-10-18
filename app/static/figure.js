// Figure export of a heatmap model as SVG, PDF or PNG: the whole matrix at the viewer's
// cell size, with the labels, group separators and dendrograms shown on screen, plus a
// colorbar. Cells are embedded as an image drawn without smoothing, so they stay crisp
// at any zoom; text and lines are vector in SVG and PDF. The PDF is written here (one
// page, the standard Helvetica font) to keep the frontend free of dependencies.
//
// model: as for Heatmap (rows, cols, values, rowTitle, rowGroupTitle, rowDendro, colDendro)
// (either way round)
// opts: { cellW, cellH, lut, norm, nanColor, valueLabel, title }
import { fmtNum } from './colormaps.js';
import { parseColor } from './heatmap.js';

const FONT = 'Helvetica, Arial, sans-serif';
const INK = '#1d1f22', INK_MUTED = '#5c6066', RULE = '#cfd1cf', SURFACE = '#ffffff', LINE = '#444444';
const GROUP_FONT = 11, SUB_FONT = 9, TICK_FONT = 10;
const MID = 0.36; // baseline offset (in em) that centres Helvetica text on a line
const MARGIN = 10;
const PNG_SCALE = 3, MAX_IMAGE_SIDE = 16384, MAX_PNG_PIXELS = 120e6, MAX_SVG_IMAGE_PIXELS = 40e6;

export async function exportFigure(format, model, opts) {
  const L = layoutFigure(model, opts);
  let fig, note = '';
  if (format === 'svg') fig = new SvgFigure(L.width, L.height, opts.title);
  else if (format === 'pdf') fig = new PdfFigure(L.width, L.height, opts.title);
  else {
    const scale = Math.min(PNG_SCALE, MAX_IMAGE_SIDE / L.width, MAX_IMAGE_SIDE / L.height, Math.sqrt(MAX_PNG_PIXELS / (L.width * L.height)));
    if (scale < 0.05) throw new Error('the matrix is too large for a PNG at this cell size; export SVG or PDF, or use smaller cells');
    if (scale < 1) note = `PNG scaled to ${Math.round(scale * 100)}% to fit the browser's image size limit`;
    fig = new CanvasFigure(L.width, L.height, scale);
  }
  drawFigure(fig, L, model, opts);
  return { blob: await fig.finish(), note };
}

// ------------------------------------------------------------------ text metrics
// Helvetica advance widths (1/1000 em) of ASCII 32..126, from the standard AFM
const HELVETICA = [
  278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
  556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
  1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
  667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
  333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
  556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
];
// non-ASCII characters of the PDF's WinAnsi encoding: [code, width]
const WINANSI = {
  '…': [0x85, 1000], '–': [0x96, 556], '—': [0x97, 1000], '‘': [0x91, 222], '’': [0x92, 222],
  '“': [0x93, 333], '”': [0x94, 333], '•': [0x95, 350], '×': [0xd7, 584], '±': [0xb1, 584],
  '°': [0xb0, 400], 'µ': [0xb5, 556], '·': [0xb7, 278],
};

function charInfo(ch) {
  const c = ch.codePointAt(0);
  if (c >= 32 && c <= 126) return [c, HELVETICA[c - 32]];
  if (WINANSI[ch]) return WINANSI[ch];
  if (c >= 0xa0 && c <= 0xff) return [c, 556];
  return [63, 556]; // '?'
}

function textWidth(s, size) {
  let w = 0;
  for (const ch of s) w += charInfo(ch)[1];
  return (w * size) / 1000;
}

function ellipsize(s, maxW, size) {
  if (maxW <= 0) return '';
  if (textWidth(s, size) <= maxW) return s;
  const chars = Array.from(s);
  let lo = 0, hi = chars.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (textWidth(chars.slice(0, mid).join('') + '…', size) <= maxW) lo = mid; else hi = mid - 1;
  }
  return lo > 0 ? chars.slice(0, lo).join('') + '…' : '';
}

// --------------------------------------------------------------------- layout
// Same bands and sizes as the on-screen heatmap, but labels are allowed to be longer.
function layoutFigure(model, o) {
  const { cellW, cellH } = o;
  const L = { cellW, cellH, nR: model.rows.length, nC: model.cols.length };
  const widest = (strs, size) => { let w = 0; for (const s of strs) w = Math.max(w, textWidth(String(s ?? ''), size)); return Math.ceil(w); };
  L.rowFont = Math.min(12, Math.max(8, cellH - 4));
  L.colFont = Math.min(12, Math.max(8, cellW - 4));
  L.showRowLabels = cellH >= 9;
  L.showColLabels = cellW >= 8;
  L.hasRowGroups = model.rows.some((r) => r.group != null);
  L.hasColGroups = model.cols.some((c) => c.group != null);
  L.anySub = model.cols.some((c) => c.sub != null);
  L.anyRowSub = model.rows.some((r) => r.sub != null);
  L.rowRuns = runs(model.rows.map((r) => r.group));
  L.colRuns = runs(model.cols.map((c) => c.group));
  L.rowLabelRuns = runs(model.rows.map((r) => r.key));
  L.labelRuns = runs(model.cols.map((c) => c.key));

  L.dendroW = model.rowDendro ? 56 : 0;
  L.dendroH = model.colDendro ? 44 : 0;
  L.groupColW = L.hasRowGroups ? Math.min(220, widest(L.rowRuns.map((r) => r.key), GROUP_FONT) + 12) : 0;
  L.labelColW = L.showRowLabels ? Math.min(300, widest(L.rowLabelRuns.map((r) => model.rows[r.start].label), L.rowFont) + 12) : 0;
  L.rowSubW = L.anyRowSub && cellH >= 10 ? widest(model.rows.map((r) => r.sub), SUB_FONT) + 8 : 0;
  const labelW = L.dendroW + Math.max(L.groupColW + L.labelColW + L.rowSubW, L.hasRowGroups || L.showRowLabels ? 40 : 4);
  L.colGroupBand = L.hasColGroups ? 18 : 0;
  L.colLabelBand = L.showColLabels ? Math.min(300, widest(L.labelRuns.map((r) => model.cols[r.start].label), L.colFont) + 10) : 4;
  L.colSubBand = L.anySub && cellW >= 10 ? 13 : 0;
  L.x0 = MARGIN;
  L.y0 = MARGIN;
  L.gx = MARGIN + labelW;
  L.gy = MARGIN + L.dendroH + L.colGroupBand + L.colLabelBand + L.colSubBand;
  L.gw = L.nC * cellW;
  L.gh = L.nR * cellH;

  L.hasNaN = model.values.some((v) => v !== v);
  L.cbX = L.gx + L.gw + 18;
  L.cbY = L.gy;
  L.cbW = 12;
  L.cbH = Math.max(60, Math.min(220, L.gh));
  // one label per distinct value, preferring the middle tick (white = 0 when there are
  // no negative values gives two ticks at 0)
  L.ticks = o.norm.ticks
    .filter((t, i, all) => !all.some((u, j) => j !== i && u.value === t.value && Math.abs(u.t - 0.5) < Math.abs(t.t - 0.5)))
    .map((t) => ({ t: t.t, label: fmtNum(t.value) }));
  let right = L.cbX + L.cbW + 6 + widest(L.ticks.map((t) => t.label), TICK_FONT);
  let bottom = L.cbY + L.cbH + 4;
  if (o.valueLabel) { right = Math.max(right, L.cbX + textWidth(o.valueLabel, TICK_FONT)); bottom += 16; }
  if (L.hasNaN) { right = Math.max(right, L.cbX + 16 + textWidth('missing', TICK_FONT)); bottom += 16; }
  L.width = Math.ceil(right + MARGIN);
  L.height = Math.ceil(Math.max(L.gy + L.gh, bottom) + MARGIN);
  return L;
}

// ----------------------------------------------------------------------- draw
function drawFigure(f, L, model, o) {
  const { cellW, cellH, gx, gy, gw, gh } = L;

  f.image(cellPixels(model, o), L.nC, L.nR, gx, gy, gw, gh, { crisp: true });
  // gaps between groups, in the background colour (as on screen)
  if (L.hasColGroups) for (const r of L.colRuns.slice(1)) f.rect(gx + r.start * cellW - 1, gy, 2, gh, SURFACE);
  if (L.hasRowGroups) for (const r of L.rowRuns.slice(1)) f.rect(gx, gy + r.start * cellH - 1, gw, 2, SURFACE);
  if (L.anySub && cellW >= 6) for (const r of L.labelRuns.slice(1)) f.rect(gx + r.start * cellW, gy, 1, gh, SURFACE);
  if (L.anyRowSub && cellH >= 6) for (const r of L.rowLabelRuns.slice(1)) f.rect(gx, gy + r.start * cellH, gw, 1, SURFACE);

  // rows: labels (and repeat numbers right of them), then groups (with a rule between
  // them) left of the labels
  if (L.showRowLabels) {
    for (const r of L.rowLabelRuns) {
      f.text(ellipsize(String(model.rows[r.start].label), L.labelColW - 8, L.rowFont), gx - L.rowSubW - 6,
        gy + ((r.start + r.end) / 2) * cellH + MID * L.rowFont, { size: L.rowFont, color: INK, anchor: 'end' });
    }
  }
  if (L.rowSubW) {
    model.rows.forEach((r, i) => f.text(String(r.sub ?? ''), gx - L.rowSubW / 2 - 1, gy + (i + 0.5) * cellH + MID * SUB_FONT,
      { size: SUB_FONT, color: INK_MUTED, anchor: 'middle' }));
  }
  if (L.hasRowGroups) {
    const x = L.x0 + L.dendroW;
    for (const r of L.rowRuns) {
      const top = gy + r.start * cellH, h = (r.end - r.start) * cellH;
      if (r.start > 0) f.rect(x, top - 0.5, gx - x, 1, RULE);
      if (h >= 11) f.text(ellipsize(String(r.key), L.groupColW - 8, GROUP_FONT), x + 6, top + h / 2 + MID * GROUP_FONT, { size: GROUP_FONT, color: INK_MUTED });
    }
  }

  // header: group (panel) band, rotated odor labels, repeat numbers
  const bandY = L.y0 + L.dendroH;
  if (L.hasColGroups) {
    for (const r of L.colRuns) {
      const x0 = gx + r.start * cellW, x1 = gx + r.end * cellW;
      if (r.start > 0) f.rect(x0 - 0.5, bandY, 1, gy - bandY, RULE);
      const label = ellipsize(String(r.key ?? ''), x1 - x0 - 6, GROUP_FONT);
      if (label) f.text(label, (x0 + x1) / 2, bandY + L.colGroupBand / 2 + MID * GROUP_FONT, { size: GROUP_FONT, color: INK_MUTED, anchor: 'middle' });
    }
    f.rect(gx, bandY + L.colGroupBand - 1, gw, 1, RULE);
  }
  if (L.showColLabels) {
    const baseY = bandY + L.colGroupBand + L.colLabelBand - 5;
    for (const r of L.labelRuns) {
      const xc = gx + ((r.start + r.end) / 2) * cellW;
      f.text(ellipsize(String(model.cols[r.start].label), L.colLabelBand - 10, L.colFont), xc + MID * L.colFont, baseY,
        { size: L.colFont, color: INK, rotate: true });
    }
  }
  if (L.colSubBand) {
    const y = gy - L.colSubBand / 2 + MID * SUB_FONT;
    model.cols.forEach((c, i) => f.text(String(c.sub ?? ''), gx + (i + 0.5) * cellW, y, { size: SUB_FONT, color: INK_MUTED, anchor: 'middle' }));
  }
  if (L.hasRowGroups || L.showRowLabels) {
    const title = [model.rowGroupTitle, model.rowTitle].filter(Boolean).join(' / ');
    f.text(ellipsize(title, gx - L.x0 - 6, GROUP_FONT), L.x0, gy - 6, { size: GROUP_FONT, color: INK_MUTED });
  }

  if (model.rowDendro) drawDendrogram(f, model.rowDendro, true, L);
  if (model.colDendro) drawDendrogram(f, model.colDendro, false, L);

  // colorbar (high values at the top), ticks, value label, missing-value swatch
  const { cbX, cbY, cbW, cbH } = L, { lut } = o;
  const bar = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const k = (255 - i) * 3;
    bar.set([lut[k], lut[k + 1], lut[k + 2], 255], i * 4);
  }
  f.image(bar, 1, 256, cbX, cbY, cbW, cbH, { crisp: false });
  f.lines([[cbX, cbY, cbX + cbW, cbY], [cbX + cbW, cbY, cbX + cbW, cbY + cbH], [cbX + cbW, cbY + cbH, cbX, cbY + cbH], [cbX, cbY + cbH, cbX, cbY]], '#999999', 0.5);
  for (const t of L.ticks) {
    const y = cbY + (1 - t.t) * cbH;
    f.lines([[cbX + cbW, y, cbX + cbW + 3, y]], INK, 0.75);
    f.text(t.label, cbX + cbW + 6, y + MID * TICK_FONT, { size: TICK_FONT, color: INK });
  }
  let y = cbY + cbH + 16;
  if (o.valueLabel) { f.text(o.valueLabel, cbX, y, { size: TICK_FONT, color: INK }); y += 16; }
  if (L.hasNaN) {
    f.rect(cbX, y - 9, 10, 10, o.nanColor);
    f.text('missing', cbX + 16, y, { size: TICK_FONT, color: INK });
  }
}

// leaves next to the labels, root towards the edge of the figure (as on screen)
function drawDendrogram(f, d, rows, L) {
  const band = rows ? L.dendroW : L.dendroH;
  const base = (rows ? L.x0 : L.y0) + band - 3, scale = (band - 7) / d.maxH;
  const cell = rows ? L.cellH : L.cellW, origin = rows ? L.gy : L.gx;
  const segs = [];
  for (let i = 0; i < d.segs.length; i += 4) {
    const p1 = origin + d.segs[i] * cell, h1 = base - d.segs[i + 1] * scale;
    const p2 = origin + d.segs[i + 2] * cell, h2 = base - d.segs[i + 3] * scale;
    segs.push(rows ? [h1, p1, h2, p2] : [p1, h1, p2, h2]);
  }
  f.lines(segs, LINE, 0.75);
}

function cellPixels(model, { lut, norm, nanColor }) {
  const { values } = model;
  const px = new Uint8ClampedArray(values.length * 4);
  const nan = parseColor(nanColor);
  for (let i = 0, k = 0; i < values.length; i++) {
    const t = norm.toUnit(values[i]);
    if (t !== t) { px[k++] = nan[0]; px[k++] = nan[1]; px[k++] = nan[2]; } else {
      const j = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
      px[k++] = lut[j]; px[k++] = lut[j + 1]; px[k++] = lut[j + 2];
    }
    px[k++] = 255;
  }
  return px;
}

function runs(keys) {
  const out = [];
  let start = 0;
  for (let i = 1; i <= keys.length; i++) {
    if (i === keys.length || keys[i] !== keys[start]) { out.push({ key: keys[start], start, end: i }); start = i; }
  }
  return out;
}

// ------------------------------------------------------------------- backends
// Each draws in figure units (CSS px = PDF pt), y down, text positioned by baseline:
// rect(x, y, w, h, fill), lines([[x1, y1, x2, y2], ...], color, width),
// text(s, x, y, {size, color, anchor: start|middle|end, rotate: reads upwards}),
// image(rgba, iw, ih, x, y, w, h, {crisp}); finish() -> Promise<Blob>
const num = (v) => String(Math.round(v * 100) / 100);
const xmlEsc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

function rgbaCanvas(px, iw, ih) {
  const c = document.createElement('canvas');
  c.width = iw; c.height = ih;
  c.getContext('2d').putImageData(new ImageData(px, iw, ih), 0, 0);
  return c;
}

function canvasBlob(c, type = 'image/png') {
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('the browser could not encode the image (too large?)'))), type));
}

function blobDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

class SvgFigure {
  constructor(w, h, title) { this.w = w; this.h = h; this.title = title || 'heatmap'; this.parts = []; this.pending = []; }
  rect(x, y, w, h, fill) { this.parts.push(`<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" fill="${fill}"/>`); }
  lines(segs, color, width) {
    if (!segs.length) return;
    const d = segs.map(([x1, y1, x2, y2]) => `M${num(x1)} ${num(y1)}L${num(x2)} ${num(y2)}`).join('');
    this.parts.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}"/>`);
  }
  text(s, x, y, { size, color, anchor = 'start', rotate = false }) {
    const pos = rotate ? `transform="translate(${num(x)} ${num(y)}) rotate(-90)"` : `x="${num(x)}" y="${num(y)}"`;
    this.parts.push(`<text ${pos} font-size="${size}" fill="${color}"${anchor === 'start' ? '' : ` text-anchor="${anchor}"`}>${xmlEsc(s)}</text>`);
  }
  image(px, iw, ih, x, y, w, h, { crisp }) {
    // upscale so that each cell is whole pixels: crisp even where image-rendering is ignored
    let kx = crisp ? Math.max(1, Math.round(w / iw)) : 1, ky = crisp ? Math.max(1, Math.round(h / ih)) : 1;
    if (iw * kx > MAX_IMAGE_SIDE || ih * ky > MAX_IMAGE_SIDE || iw * kx * ih * ky > MAX_SVG_IMAGE_PIXELS) { kx = 1; ky = 1; }
    const i = this.parts.push('') - 1;
    this.pending.push((async () => {
      let c = rgbaCanvas(px, iw, ih);
      if (kx > 1 || ky > 1) {
        const big = document.createElement('canvas');
        big.width = iw * kx; big.height = ih * ky;
        const ctx = big.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(c, 0, 0, big.width, big.height);
        c = big;
      }
      const url = await blobDataUrl(await canvasBlob(c));
      const style = crisp ? ' style="image-rendering:pixelated" image-rendering="optimizeSpeed"' : '';
      this.parts[i] = `<image x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" preserveAspectRatio="none"${style} xlink:href="${url}"/>`;
    })());
  }
  async finish() {
    await Promise.all(this.pending);
    const svg = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${this.w}" height="${this.h}" viewBox="0 0 ${this.w} ${this.h}" font-family="${FONT}">`,
      `<title>${xmlEsc(this.title)}</title>`,
      ...this.parts,
      '</svg>',
      '',
    ].join('\n');
    return new Blob([svg], { type: 'image/svg+xml' });
  }
}

const latin1 = (s) => Uint8Array.from(s, (ch) => ch.charCodeAt(0) & 0xff);

function concatBytes(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function deflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function pdfText(s) {
  let out = '';
  for (const ch of s) {
    const c = charInfo(ch)[0];
    out += (c === 0x28 || c === 0x29 || c === 0x5c ? '\\' : '') + String.fromCharCode(c);
  }
  return out;
}

function pdfColor(hex) { return parseColor(hex).map((v) => (v / 255).toFixed(3)).join(' '); }

class PdfFigure {
  constructor(w, h, title) { this.w = w; this.h = h; this.title = title || 'heatmap'; this.ops = []; this.images = []; }
  rect(x, y, w, h, fill) { this.ops.push(`${pdfColor(fill)} rg ${num(x)} ${num(this.h - y - h)} ${num(w)} ${num(h)} re f`); }
  lines(segs, color, width) {
    if (!segs.length) return;
    const ops = [`${pdfColor(color)} RG ${num(width)} w`];
    for (const [x1, y1, x2, y2] of segs) ops.push(`${num(x1)} ${num(this.h - y1)} m ${num(x2)} ${num(this.h - y2)} l`);
    this.ops.push(ops.join('\n'), 'S');
  }
  text(s, x, y, { size, color, anchor = 'start', rotate = false }) {
    const w = textWidth(s, size);
    const shift = anchor === 'middle' ? -w / 2 : anchor === 'end' ? -w : 0;
    const m = rotate ? `0 1 -1 0 ${num(x)} ${num(this.h - y + shift)}` : `1 0 0 1 ${num(x + shift)} ${num(this.h - y)}`;
    this.ops.push(`BT /F1 ${num(size)} Tf ${pdfColor(color)} rg ${m} Tm (${pdfText(s)}) Tj ET`);
  }
  image(px, iw, ih, x, y, w, h, { crisp }) {
    const rgb = new Uint8Array(iw * ih * 3);
    for (let i = 0, j = 0; i < px.length; i += 4) { rgb[j++] = px[i]; rgb[j++] = px[i + 1]; rgb[j++] = px[i + 2]; }
    this.images.push({ rgb, iw, ih, crisp });
    this.ops.push(`q ${num(w)} 0 0 ${num(h)} ${num(x)} ${num(this.h - y - h)} cm /Im${this.images.length} Do Q`);
  }
  async finish() {
    const stream = (dict, data) => concatBytes([latin1(`<< ${dict} /Length ${data.length} /Filter /FlateDecode >>\nstream\n`), data, latin1('\nendstream')]);
    const content = await deflate(latin1(this.ops.join('\n')));
    const images = await Promise.all(this.images.map(async (im) => ({ ...im, data: await deflate(im.rgb) })));
    const now = new Date(), p2 = (v) => String(v).padStart(2, '0');
    const date = `${now.getFullYear()}${p2(now.getMonth() + 1)}${p2(now.getDate())}${p2(now.getHours())}${p2(now.getMinutes())}${p2(now.getSeconds())}`;
    // 1 catalog, 2 pages, 3 page, 4 font, 5 info, 6 content, 7... images
    const objects = [
      latin1('<< /Type /Catalog /Pages 2 0 R >>'),
      latin1('<< /Type /Pages /Kids [3 0 R] /Count 1 >>'),
      latin1(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(this.w)} ${num(this.h)}] /Resources << /Font << /F1 4 0 R >> ` +
        `/XObject << ${images.map((_, i) => `/Im${i + 1} ${7 + i} 0 R`).join(' ')} >> >> /Contents 6 0 R >>`),
      latin1('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>'),
      latin1(`<< /Title (${pdfText(this.title)}) /Producer (Odor response database) /CreationDate (D:${date}) >>`),
      stream('', content),
      ...images.map((im) => stream(`/Type /XObject /Subtype /Image /Width ${im.iw} /Height ${im.ih} /ColorSpace /DeviceRGB ` +
        `/BitsPerComponent 8 /Interpolate ${im.crisp ? 'false' : 'true'}`, im.data)),
    ];
    const chunks = [latin1('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')];
    let offset = chunks[0].length;
    const offsets = objects.map((body, i) => {
      const at = offset;
      chunks.push(latin1(`${i + 1} 0 obj\n`), body, latin1('\nendobj\n'));
      offset += `${i + 1} 0 obj\n`.length + body.length + '\nendobj\n'.length;
      return at;
    });
    const xref = ['xref', `0 ${objects.length + 1}`, '0000000000 65535 f ', ...offsets.map((o) => `${String(o).padStart(10, '0')} 00000 n `)];
    chunks.push(latin1(`${xref.join('\n')}\ntrailer\n<< /Size ${objects.length + 1} /Root 1 0 R /Info 5 0 R >>\nstartxref\n${offset}\n%%EOF\n`));
    return new Blob(chunks, { type: 'application/pdf' });
  }
}

class CanvasFigure {
  constructor(w, h, scale) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = Math.round(w * scale);
    this.canvas.height = Math.round(h * scale);
    this.ctx = this.canvas.getContext('2d');
    if (!this.ctx) throw new Error('the browser could not make a canvas this large');
    this.ctx.scale(scale, scale);
    this.ctx.fillStyle = SURFACE;
    this.ctx.fillRect(0, 0, w, h);
  }
  rect(x, y, w, h, fill) { this.ctx.fillStyle = fill; this.ctx.fillRect(x, y, w, h); }
  lines(segs, color, width) {
    const c = this.ctx;
    c.beginPath();
    for (const [x1, y1, x2, y2] of segs) { c.moveTo(x1, y1); c.lineTo(x2, y2); }
    c.strokeStyle = color; c.lineWidth = width;
    c.stroke();
  }
  text(s, x, y, { size, color, anchor = 'start', rotate = false }) {
    const c = this.ctx;
    c.save();
    c.font = `${size}px ${FONT}`;
    c.fillStyle = color;
    c.textAlign = anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left';
    c.textBaseline = 'alphabetic';
    c.translate(x, y);
    if (rotate) c.rotate(-Math.PI / 2);
    c.fillText(s, 0, 0);
    c.restore();
  }
  image(px, iw, ih, x, y, w, h, { crisp }) {
    this.ctx.imageSmoothingEnabled = !crisp;
    this.ctx.drawImage(rgbaCanvas(px, iw, ih), x, y, w, h);
    this.ctx.imageSmoothingEnabled = true;
  }
  finish() { return canvasBlob(this.canvas); }
}
