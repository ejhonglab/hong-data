// How alike odors are (Compare tab), within a dataset or another space of chemicals, and
// how alike those likenesses are between two spaces. NaN is a value missing.

// ranks of xs, from 1, ties given the mean of theirs
// (the values sorted by a typed array's own sort, each found in them by bisection: millions of pairs of KCs in a second)
export function ranks(xs) {
  const sorted = Float64Array.from(xs).sort(), n = sorted.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    let lo = 0, hi = n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] < x) lo = m + 1; else hi = m; }
    const first = lo;
    hi = n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] <= x) lo = m + 1; else hi = m; }
    out[i] = (first + lo - 1) / 2 + 1;
  }
  return out;
}

// Pearson correlation of xs and ys (as long as each other, none missing); NaN if either is constant
export function pearson(xs, ys) {
  const n = xs.length;
  let mx = 0, my = 0;
  for (let k = 0; k < n; k++) { mx += xs[k]; my += ys[k]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let k = 0; k < n; k++) {
    const dx = xs[k] - mx, dy = ys[k] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : NaN;
}

export const spearman = (xs, ys) => pearson(ranks(xs), ranks(ys));

function cosine(xs, ys) {
  let sxy = 0, sxx = 0, syy = 0;
  for (let k = 0; k < xs.length; k++) { sxy += xs[k] * ys[k]; sxx += xs[k] * xs[k]; syy += ys[k] * ys[k]; }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : NaN;
}

// how alike two odors are, by their responses (or values) in a space
export const SIMILARITIES = {
  pearson: { label: 'Pearson correlation', of: pearson },
  spearman: { label: 'Spearman correlation', of: spearman },
  cosine: { label: 'cosine similarity', of: cosine },
};

// the similarity (by `metric`, a key of SIMILARITIES) of vectors a and b over the entries
// both have, if they have at least 3
export function similarity(metric, a, b) {
  const xs = [], ys = [];
  for (let k = 0; k < a.length; k++) if (a[k] === a[k] && b[k] === b[k]) { xs.push(a[k]); ys.push(b[k]); }
  return xs.length >= 3 ? SIMILARITIES[metric].of(xs, ys) : NaN;
}

// a vector with nothing missing (of 3 or more entries) made so that the similarity (`metric`'s) of two such is their dot
// product: ranked (Spearman's), centered (Pearson's and Spearman's), and of length 1; null if it has something missing (or
// too few), false if it is constant (none then: NaN, as `similarity` gives)
function unitVector(v, metric) {
  if (v.length < 3) return null;
  for (let k = 0; k < v.length; k++) if (v[k] !== v[k]) return null;
  const x = Float64Array.from(metric === 'spearman' ? ranks(Array.from(v)) : v);
  if (metric !== 'cosine') {
    let m = 0;
    for (let k = 0; k < x.length; k++) m += x[k];
    m /= x.length;
    for (let k = 0; k < x.length; k++) x[k] -= m;
  }
  let ss = 0;
  for (let k = 0; k < x.length; k++) ss += x[k] * x[k];
  if (!(ss > 0)) return false;
  const norm = Math.sqrt(ss);
  for (let k = 0; k < x.length; k++) x[k] /= norm;
  return x;
}

// n × n similarities (row-major) of n vectors, by a function of two of them (by default `metric`'s; then of two
// vectors with nothing missing, as their unit vectors' dot product, `unitVector`: thousands of KCs at once)
export function similarityMatrix(vectors, metric, of = null) {
  const n = vectors.length, out = new Float32Array(n * n);
  const units = !of && SIMILARITIES[metric] ? vectors.map((v) => unitVector(v, metric)) : null;
  of = of || ((a, b) => similarity(metric, a, b));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const a = units?.[i], b = units?.[j];
      let v;
      if (a != null && b != null && a.length === b.length) {
        if (a === false || b === false) v = NaN;
        else {
          let dot = 0;
          for (let k = 0; k < a.length; k++) dot += a[k] * b[k];
          v = Math.max(-1, Math.min(1, dot));
        }
      } else v = of(vectors[i], vectors[j]);
      out[i * n + j] = v;
      out[j * n + i] = v;
    }
  }
  return out;
}

// Jaccard similarity (1 - the Jaccard distance; Tanimoto's for fingerprints) of two sets of
// bits, as arrays sorted ascending
export function tanimoto(a, b) {
  let i = 0, j = 0, both = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { both++; i++; j++; } else if (a[i] < b[j]) i++; else j++;
  }
  const either = a.length + b.length - both;
  return either ? both / either : NaN;
}

// The values of each pair of odors two spaces both have (one side of their matrices,
// without the diagonal), over the pairs neither lacks: {xs, ys}; and the spaces' whole
// similarity matrices over those n odors (n × n, row-major), for resampling odors.
// spaces: {ids: [odor id], sim: n × n}
export function sharedSimilarities(a, b) {
  const at = new Map(b.ids.map((id, i) => [id, i]));
  const shared = a.ids.map((id, i) => [i, at.get(id)]).filter(([, j]) => j !== undefined);
  const n = shared.length, nA = a.ids.length, nB = b.ids.length;
  const subA = new Float32Array(n * n), subB = new Float32Array(n * n), xs = [], ys = [];
  for (let p = 0; p < n; p++) {
    for (let q = 0; q < n; q++) {
      const x = a.sim[shared[p][0] * nA + shared[q][0]], y = b.sim[shared[p][1] * nB + shared[q][1]];
      subA[p * n + q] = x;
      subB[p * n + q] = y;
      if (q > p && x === x && y === y) { xs.push(x); ys.push(y); }
    }
  }
  return { xs: Float64Array.from(xs), ys: Float64Array.from(ys), n, subA, subB };
}

// How alike the similarities of two spaces are: the correlation (`method`: 'pearson' or
// 'spearman') of their values for each pair of the odors both have (sharedSimilarities').
// -> {r, nOdors, nPairs}
export function correlateSimilarities(method, a, b) {
  const { xs, ys, n } = sharedSimilarities(a, b);
  return { r: xs.length >= 3 ? (method === 'spearman' ? spearman : pearson)(xs, ys) : NaN, nOdors: n, nPairs: xs.length };
}

// Holm–Bonferroni adjustment of p-values (NaN: no test, left out of the family): the
// smallest times the number of tests, the next times one fewer, ..., each at least the one
// before it, at most 1 (Holm 1979, https://www.jstor.org/stable/4615733)
export function holm(ps) {
  const order = ps.map((_, i) => i).filter((i) => ps[i] === ps[i]).sort((a, b) => ps[a] - ps[b]);
  const out = ps.map(() => NaN);
  let prev = 0;
  order.forEach((i, k) => { prev = Math.min(1, Math.max(prev, (order.length - k) * ps[i])); out[i] = prev; });
  return out;
}
