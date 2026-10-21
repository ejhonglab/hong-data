// How alike odors are (Compare tab), within a dataset or another space of chemicals, and
// how alike those likenesses are between two spaces. NaN is a value missing.

// ranks of xs, from 1, ties given the mean of theirs
export function ranks(xs) {
  const idx = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
  const out = new Array(xs.length);
  for (let s = 0; s < idx.length;) {
    let e = s;
    while (e + 1 < idx.length && xs[idx[e + 1]] === xs[idx[s]]) e++;
    for (let k = s; k <= e; k++) out[idx[k]] = (s + e) / 2 + 1;
    s = e + 1;
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

// n × n similarities (row-major) of n vectors, by a function of two of them (by default `metric`'s)
export function similarityMatrix(vectors, metric, of = (a, b) => similarity(metric, a, b)) {
  const n = vectors.length, out = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = of(vectors[i], vectors[j]);
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

// How alike the similarities of two spaces are: the correlation (`method`: 'pearson' or
// 'spearman') of their values for each pair of the odors both have (one side of the
// matrices, without the diagonal), over the pairs neither lacks.
// spaces: {ids: [odor id], sim: n × n} -> {r, nOdors, nPairs}
export function correlateSimilarities(method, a, b) {
  const at = new Map(b.ids.map((id, i) => [id, i]));
  const shared = a.ids.map((id, i) => [i, at.get(id)]).filter(([, j]) => j !== undefined);
  const xs = [], ys = [], nA = a.ids.length, nB = b.ids.length;
  for (let p = 0; p < shared.length; p++) {
    for (let q = p + 1; q < shared.length; q++) {
      const x = a.sim[shared[p][0] * nA + shared[q][0]], y = b.sim[shared[p][1] * nB + shared[q][1]];
      if (x === x && y === y) { xs.push(x); ys.push(y); }
    }
  }
  return { r: xs.length >= 3 ? (method === 'spearman' ? spearman : pearson)(xs, ys) : NaN, nOdors: shared.length, nPairs: xs.length };
}
