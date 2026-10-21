// Bootstrap p-values of correlations between similarities (Compare tab), off the page's
// thread. Each test: a correlation (Pearson or Spearman) against 0, by resampling with
// replacement either its pairs of odors ('pairs': {xs, ys}) or its odors ('odors': {n,
// subA, subB}: then the pairs of the odors drawn, but for an odor drawn twice with itself).
// Two-sided: twice the smaller share of resampled correlations on either side of 0 (each
// count plus one, over the resamplings plus one), at most 1. Each test's random numbers
// are the same each time (from its seed), so its p-value is too.
//
// message in: {tests: [{seed, and xs, ys or n, subA, subB}], method, resamplings, unit};
// out: {progress} (0-1) as it goes, then {ps, valid} (the resamplings with a correlation)

import { pearson } from './similarity.js';

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ranks (from 1, ties the mean of theirs) of the first n of src into out, by `sorted` (scratch)
function rank(src, n, out, sorted) {
  const s = sorted.subarray(0, n);
  s.set(src.subarray(0, n));
  s.sort();
  for (let i = 0; i < n; i++) {
    const v = src[i];
    let lo = 0, hi = n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] < v) lo = m + 1; else hi = m; }
    let up = lo, top = n;
    while (up < top) { const m = (up + top) >> 1; if (s[m] <= v) up = m + 1; else top = m; }
    out[i] = (lo + up + 1) / 2;
  }
}

self.onmessage = ({ data: { tests, method, resamplings: B, unit } }) => {
  const ps = [], valids = [];
  let done = 0;
  const total = tests.length * B;
  tests.forEach((t) => {
    const cap = unit === 'odors' ? (t.n * (t.n - 1)) / 2 : t.xs.length;
    if (!(cap >= 3)) { ps.push(NaN); valids.push(0); done += B; return; }
    const random = mulberry32(t.seed);
    const bx = new Float64Array(cap), by = new Float64Array(cap), rx = new Float64Array(cap), ry = new Float64Array(cap), sorted = new Float64Array(cap);
    const draw = new Int32Array(unit === 'odors' ? t.n : 0);
    let below = 0, above = 0, valid = 0;
    for (let b = 0; b < B; b++) {
      let m = 0;
      if (unit === 'odors') {
        for (let k = 0; k < t.n; k++) draw[k] = Math.floor(random() * t.n);
        for (let p = 0; p < t.n; p++) {
          for (let q = p + 1; q < t.n; q++) {
            if (draw[p] === draw[q]) continue;
            const x = t.subA[draw[p] * t.n + draw[q]], y = t.subB[draw[p] * t.n + draw[q]];
            if (x === x && y === y) { bx[m] = x; by[m] = y; m++; }
          }
        }
      } else {
        for (; m < cap; m++) { const k = Math.floor(random() * cap); bx[m] = t.xs[k]; by[m] = t.ys[k]; }
      }
      let r = NaN;
      if (m >= 3) {
        if (method === 'spearman') { rank(bx, m, rx, sorted); rank(by, m, ry, sorted); r = pearson(rx.subarray(0, m), ry.subarray(0, m)); } else r = pearson(bx.subarray(0, m), by.subarray(0, m));
      }
      if (r === r) { valid++; if (r <= 0) below++; if (r >= 0) above++; }
      if (++done % 500 === 0) self.postMessage({ progress: done / total });
    }
    ps.push(valid ? Math.min(1, (2 * (Math.min(below, above) + 1)) / (valid + 1)) : NaN);
    valids.push(valid);
  });
  self.postMessage({ ps, valid: valids });
};
