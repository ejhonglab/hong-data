// Colormaps + normalisation for the heatmap viewer.
//
// The default diverging map is built in Oklab so that lightness is a linear function
// of |value|: white at 0, darkening at the same rate towards a blue (negative) and a
// red (positive) pole of equal lightness. That makes "twice as dark" mean "twice as
// large" on either side, which the ColorBrewer / Moreland maps only approximate.
import { MPL_LUTS } from './mpl_luts.js';

// ---- Oklab helpers -------------------------------------------------------------
function oklabToLinearSrgb(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}
function linearToSrgb8(c) {
  c = Math.min(1, Math.max(0, c));
  const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.round(v * 255);
}
function oklchToRgb8(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  // shrink chroma until inside the sRGB gamut
  for (let k = 0; k < 24; k++) {
    const [r, g, b] = oklabToLinearSrgb(L, C * Math.cos(h), C * Math.sin(h));
    if (r >= -1e-4 && r <= 1.0001 && g >= -1e-4 && g <= 1.0001 && b >= -1e-4 && b <= 1.0001) {
      return [linearToSrgb8(r), linearToSrgb8(g), linearToSrgb8(b)];
    }
    C *= 0.92;
  }
  const [r, g, b] = oklabToLinearSrgb(L, 0, 0);
  return [linearToSrgb8(r), linearToSrgb8(g), linearToSrgb8(b)];
}

// Diverging: index 0 = negative pole, 127/128 = white, 255 = positive pole.
function oklabDiverging({ lEnd = 0.36, cEnd = 0.15, hueNeg = 262, huePos = 27 } = {}) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i - 127.5) / 127.5; // -1 .. 1
    const t = Math.abs(x);
    const L = 1 - t * (1 - lEnd);
    const C = cEnd * t;
    const [r, g, b] = oklchToRgb8(L, C, x < 0 ? hueNeg : huePos);
    lut[i * 3] = r; lut[i * 3 + 1] = g; lut[i * 3 + 2] = b;
  }
  return lut;
}

// Sequential single-hue, white -> dark, lightness linear in value.
function oklabSequential({ hue = 250, lStart = 0.985, lEnd = 0.28, cMax = 0.14 } = {}) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const L = lStart - t * (lStart - lEnd);
    const C = cMax * Math.sin(Math.PI * Math.min(1, t * 1.15)) * 0.9 + cMax * 0.1 * t;
    const [r, g, b] = oklchToRgb8(L, C, hue);
    lut[i * 3] = r; lut[i * 3 + 1] = g; lut[i * 3 + 2] = b;
  }
  return lut;
}

function fromMpl(name) {
  return Uint8ClampedArray.from(MPL_LUTS[name]);
}

function reversedLut(lut) {
  const out = new Uint8ClampedArray(lut.length);
  for (let i = 0; i < 256; i++) out.set(lut.subarray((255 - i) * 3, (256 - i) * 3), i * 3);
  return out;
}

function binary(absent = [241, 241, 241], present = [31, 95, 139]) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = i < 128 ? absent : present;
    lut[i * 3] = c[0]; lut[i * 3 + 1] = c[1]; lut[i * 3 + 2] = c[2];
  }
  return lut;
}

export const COLORMAPS = {
  'oklab-div': { label: 'Blue–white–red (Oklab, lightness-linear)', kind: 'diverging', lut: oklabDiverging() },
  'rdbu': { label: 'RdBu (ColorBrewer)', kind: 'diverging', lut: fromMpl('RdBu_r') },
  'coolwarm': { label: 'coolwarm (Moreland)', kind: 'diverging', lut: fromMpl('coolwarm') },
  'viridis': { label: 'viridis', kind: 'sequential', lut: fromMpl('viridis') },
  'plasma-r': { label: 'plasma, reversed (lowest brightest)', kind: 'sequential', lut: reversedLut(fromMpl('plasma')) },
  'oklab-seq': { label: 'White–blue (Oklab)', kind: 'sequential', lut: oklabSequential() },
  'binary': { label: 'Absent / present', kind: 'binary', lut: binary() },
};

export function colormapsOfKind(kind) {
  return Object.entries(COLORMAPS).filter(([, v]) => v.kind === kind).map(([k, v]) => ({ id: k, label: v.label }));
}

// ---- normalisation -------------------------------------------------------------
// kind: 'twoslope' | 'symmetric' | 'log' | 'linear'.
// Returns {toUnit(v) -> 0..1 | NaN, vmin, vmax, ticks:[{t, value}]}
export function makeNorm(kind, vmin, vmax) {
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) { vmin = 0; vmax = 1; }
  if (kind === 'twoslope') {
    // white at 0; negative side scaled by vmin (if any negatives), positive by vmax
    const neg = Math.min(vmin, 0), pos = Math.max(vmax, 0);
    const lo = neg < 0 ? neg : 0, hi = pos > 0 ? pos : 0;
    return {
      kind, vmin: lo, vmax: hi,
      toUnit(v) {
        if (!Number.isFinite(v)) return NaN;
        if (v < 0) return lo < 0 ? 0.5 * (1 - Math.min(1, v / lo)) : 0.5;
        return hi > 0 ? 0.5 + 0.5 * Math.min(1, v / hi) : 0.5;
      },
      ticks: [{ t: 0, value: lo }, { t: 0.5, value: 0 }, { t: 1, value: hi }],
    };
  }
  if (kind === 'symmetric') {
    const m = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1;
    return {
      kind, vmin: -m, vmax: m,
      toUnit(v) { return Number.isFinite(v) ? 0.5 + 0.5 * Math.max(-1, Math.min(1, v / m)) : NaN; },
      ticks: [{ t: 0, value: -m }, { t: 0.5, value: 0 }, { t: 1, value: m }],
    };
  }
  if (kind === 'log') {
    // of positive values: 0 and below the lowest color
    if (!(vmax > 0)) vmax = 1;
    if (!(vmin > 0) || vmin >= vmax) vmin = vmax / 1e3;
    const lo = Math.log10(vmin), hi = Math.log10(vmax), span = hi - lo || 1;
    return {
      kind, vmin, vmax,
      toUnit(v) {
        if (!Number.isFinite(v)) return NaN;
        return v > 0 ? Math.max(0, Math.min(1, (Math.log10(v) - lo) / span)) : 0;
      },
      ticks: logTicks(lo, hi),
    };
  }
  if (kind === 'log1p') {
    // of counts: log of 1 + each, so 0 is the lowest color and 1 is apart from it
    if (!(vmax > 0)) vmax = 1;
    const hi = Math.log10(1 + vmax);
    const unit = (v) => Math.log10(1 + v) / hi;
    const decades = [];
    for (let v = 1; v < vmax; v *= 10) decades.push({ t: unit(v), value: v });
    return {
      kind, vmin: 0, vmax,
      toUnit(v) { return Number.isFinite(v) ? Math.max(0, Math.min(1, unit(Math.max(0, v)))) : NaN; },
      ticks: [{ t: 0, value: 0 }, ...decades, { t: 1, value: vmax }],
    };
  }
  // linear
  const span = vmax - vmin || 1;
  return {
    kind: 'linear', vmin, vmax,
    toUnit(v) { return Number.isFinite(v) ? Math.max(0, Math.min(1, (v - vmin) / span)) : NaN; },
    ticks: [{ t: 0, value: vmin }, { t: 0.5, value: vmin + span / 2 }, { t: 1, value: vmax }],
  };
}

// ticks of a log scale from 10^lo to 10^hi: its ends, and every power of ten between them
// (those whose labels would overlap left out where they are drawn: `spacedTicks`)
function logTicks(lo, hi) {
  const span = hi - lo || 1, decades = [];
  for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) if (k > lo && k < hi) decades.push({ t: (k - lo) / span, value: 10 ** k });
  return [{ t: 0, value: 10 ** lo }, ...decades, { t: 1, value: 10 ** hi }];
}

// Of a colorbar's ticks (`norm.ticks`), those whose labels do not overlap as drawn: its two
// ends, and of the others, every one, else every 2nd, 3rd... (evenly, as powers of ten
// are best left out), the first way none overlap by less than `gap`. extent(tick, index,
// n): where its label is along the bar, [from, to].
export function spacedTicks(ticks, extent, gap = 4) {
  const n = ticks.length;
  if (n <= 2) return ticks;
  const spans = ticks.map((tk, i) => extent(tk, i, n));
  const clear = (a, b) => (a === 0 && b === n - 1) || spans[a][1] + gap <= spans[b][0] || spans[a][0] >= spans[b][1] + gap;
  const inner = ticks.slice(1, -1).map((_, k) => k + 1);
  for (let step = 1; step <= inner.length; step++) {
    for (let off = 0; off < step; off++) {
      const kept = [0, ...inner.filter((_, k) => k % step === off), n - 1];
      if (kept.every((a, j) => kept.slice(j + 1).every((b) => clear(a, b)))) return kept.map((i) => ticks[i]);
    }
  }
  return [ticks[0], ticks[n - 1]];
}

export function drawColorbar(canvas, lut, norm, { nanColor = '#e6e6e6' } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 220, h = canvas.clientHeight || 34;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  const barH = 10, pad = 2, barW = w - 2 * pad;
  // the bar in whole pixels of the screen: drawn as columns a CSS pixel wide, on a display
  // scaled by a fraction (125%, 150%) they would leave seams that the page shows through,
  // darkening (or lightening) every color
  const x0 = Math.round(pad * dpr), y0 = Math.round(2 * dpr);
  const bw = Math.round((pad + barW) * dpr) - x0, bh = Math.round((2 + barH) * dpr) - y0;
  const img = ctx.createImageData(bw, bh);
  for (let x = 0; x < bw; x++) {
    const i = Math.max(0, Math.min(255, Math.round((x / Math.max(1, bw - 1)) * 255))) * 3;
    for (let y = 0; y < bh; y++) img.data.set([lut[i], lut[i + 1], lut[i + 2], 255], (y * bw + x) * 4);
  }
  ctx.putImageData(img, x0, y0);
  ctx.scale(dpr, dpr);
  ctx.strokeStyle = 'rgba(0,0,0,0.25)';
  ctx.lineWidth = 1;
  ctx.strokeRect(pad + 0.5, 2.5, barW - 1, barH - 1);
  const ink = getComputedStyle(canvas).color || '#333';
  ctx.fillStyle = ctx.strokeStyle = ink;
  ctx.font = '10px ' + (getComputedStyle(canvas).fontFamily || 'sans-serif');
  ctx.textBaseline = 'top';
  // each tick a mark under the bar and its value, left- and right-aligned at its ends
  const at = (tk) => pad + tk.t * (barW - 1), n = norm.ticks.length;
  const align = (i) => (i === 0 ? 'left' : i === n - 1 ? 'right' : 'center');
  const extent = (tk, i) => {
    const x = at(tk), w = ctx.measureText(tickLabel(tk.value)).width;
    return align(i) === 'left' ? [x, x + w] : align(i) === 'right' ? [x - w, x] : [x - w / 2, x + w / 2];
  };
  const shown = new Set(spacedTicks(norm.ticks, extent));
  norm.ticks.forEach((tk, i) => {
    if (!shown.has(tk)) return;
    const x = at(tk);
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, 2 + barH);
    ctx.lineTo(Math.round(x) + 0.5, 2 + barH + 3);
    ctx.stroke();
    ctx.textAlign = align(i);
    ctx.fillText(tickLabel(tk.value), x, barH + 6);
  });
  void nanColor;
}

// a colorbar's tick's value as labelled: as fmtNum, but without trailing zeros (0.01, not 0.010)
export function tickLabel(v) {
  const s = fmtNum(v);
  return s.includes('.') && !s.includes('e') ? s.replace(/\.?0+$/, '') : s;
}

export function fmtNum(v) {
  if (!Number.isFinite(v)) return 'NaN';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5) return v.toExponential(1).replace('e+', 'e').replace('.0e', 'e');
  if (a >= 100 || Number.isInteger(v)) return v.toFixed(0); // (counts, and powers of ten of a log scale, as such)
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toExponential(1).replace('.0e', 'e');
}
