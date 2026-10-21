// p-values of correlations between similarities (Compare tab), off the page's thread. Each
// test: a correlation (Pearson or Spearman) of two spaces' similarities against 0,
// two-sided, by `unit`:
// - 'pairs': a bootstrap of its pairs of odors ({xs, ys}), with replacement;
// - 'odors': a bootstrap of its odors ({n, subA, subB}), with replacement: then the pairs of
//   the odors drawn, but for an odor drawn twice with itself;
// - 'permute': a permutation test (Mantel's) of the odors of the second space (n, subA,
//   subB), shuffled against the first's.
// By bootstrap, p is twice the smaller share of resampled correlations on either side of 0
// (each count plus one, over the resamplings plus one), at most 1; by permutation, the
// share as far from 0 as the correlation, or further (plus one, over the permutations plus
// one). Each test's random numbers are the same each time (from its seed), so its p-value
// is too.
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
    const n = t.n, cap = unit === 'pairs' ? t.xs.length : (n * (n - 1)) / 2;
    const bx = new Float64Array(cap), by = new Float64Array(cap), rx = new Float64Array(cap), ry = new Float64Array(cap), sorted = new Float64Array(cap);
    const corr = (m) => {
      if (m < 3) return NaN;
      if (method !== 'spearman') return pearson(bx.subarray(0, m), by.subarray(0, m));
      rank(bx, m, rx, sorted);
      rank(by, m, ry, sorted);
      return pearson(rx.subarray(0, m), ry.subarray(0, m));
    };
    // the pairs of odors p < q, as odors ia[p], ia[q] of the first space and ib[p], ib[q] of
    // the second (an odor drawn twice not with itself), neither value missing: how many
    const collect = (ia, ib) => {
      let m = 0;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (ia[p] === ia[q]) continue;
          const x = t.subA[ia[p] * n + ia[q]], y = t.subB[ib[p] * n + ib[q]];
          if (x === x && y === y) { bx[m] = x; by[m] = y; m++; }
        }
      }
      return m;
    };
    const ident = Int32Array.from({ length: unit === 'pairs' ? 0 : n }, (_, k) => k), draw = ident.slice();
    const r0 = unit === 'permute' ? corr(collect(ident, ident)) : NaN;
    if (!(cap >= 3) || (unit === 'permute' && r0 !== r0)) { ps.push(NaN); valids.push(0); done += B; return; }
    const random = mulberry32(t.seed);
    let below = 0, above = 0, extreme = 0, valid = 0;
    for (let b = 0; b < B; b++) {
      let m = 0;
      if (unit === 'odors') {
        for (let k = 0; k < n; k++) draw[k] = Math.floor(random() * n);
        m = collect(draw, draw);
      } else if (unit === 'permute') {
        for (let k = n - 1; k > 0; k--) { const j = Math.floor(random() * (k + 1)); [draw[k], draw[j]] = [draw[j], draw[k]]; }
        m = collect(ident, draw);
      } else {
        for (; m < cap; m++) { const k = Math.floor(random() * cap); bx[m] = t.xs[k]; by[m] = t.ys[k]; }
      }
      const r = corr(m);
      if (r === r) {
        valid++;
        if (r <= 0) below++;
        if (r >= 0) above++;
        if (Math.abs(r) >= Math.abs(r0) - 1e-12) extreme++; // as far from 0, but for rounding
      }
      if (++done % 500 === 0) self.postMessage({ progress: done / total });
    }
    if (!valid) ps.push(NaN);
    else if (unit === 'permute') ps.push((extreme + 1) / (valid + 1));
    else ps.push(Math.min(1, (2 * (Math.min(below, above) + 1)) / (valid + 1)));
    valids.push(valid);
  });
  self.postMessage({ ps, valid: valids });
};
