// Hierarchical clustering for ordering heatmap rows and columns (run in a worker, see
// cluster-worker.js). Average linkage (UPGMA) on correlation (1 - Pearson r) or
// Euclidean distance. Distances use only the entries both vectors have, like pandas'
// DataFrame.corr, so missing values (NaN) are fine; pairs with too little in common
// get the largest distance found.
//
// merges use scipy's linkage convention: merge k joins clusters a and b (ids < n are
// the input vectors, n + j is the cluster made by merge j) at height h, sorted by h.

export const MAX_CLUSTER_ITEMS = 5000;

// data: Float32Array of n vectors of length d (row-major)
// -> { order: Int32Array (leaf order), merges: Float64Array [a0, b0, h0, a1, ...] }
export function cluster(data, n, d, metric = 'correlation') {
  if (n > MAX_CLUSTER_ITEMS) throw new Error(`too many items to cluster (${n} > ${MAX_CLUSTER_ITEMS})`);
  if (n < 2) return { order: Int32Array.from({ length: n }, (_, i) => i), merges: new Float64Array(0) };
  const merges = averageLinkage(pairwiseDistances(data, n, d, metric), n);
  return { order: leafOrder(merges, n), merges };
}

// condensed distance matrix (pair i < j at n*i - i*(i+1)/2 + j - i - 1, as scipy's pdist)
export function pairwiseDistances(data, n, d, metric) {
  const corr = metric === 'correlation';
  const out = new Float32Array((n * (n - 1)) / 2);
  // vectors without missing values are prepared once (centred and scaled to unit
  // length for correlation), so that their pairs need only one pass
  const complete = new Uint8Array(n);
  const z = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let s = 0, ok = true;
    for (let k = 0; k < d; k++) { const v = data[o + k]; if (v !== v) { ok = false; break; } s += v; }
    if (!ok) continue;
    complete[i] = 1;
    if (!corr) { for (let k = 0; k < d; k++) z[o + k] = data[o + k]; continue; }
    const mean = s / d;
    let ss = 0;
    for (let k = 0; k < d; k++) { const c = data[o + k] - mean; z[o + k] = c; ss += c * c; }
    const norm = Math.sqrt(ss);
    for (let k = 0; k < d; k++) z[o + k] = norm > 0 ? z[o + k] / norm : NaN;
  }
  let p = 0, maxD = 0;
  for (let i = 0; i < n - 1; i++) {
    const oi = i * d;
    for (let j = i + 1; j < n; j++) {
      const oj = j * d;
      let dist;
      if (complete[i] && complete[j]) {
        let acc = 0;
        if (corr) { for (let k = 0; k < d; k++) acc += z[oi + k] * z[oj + k]; dist = 1 - acc; }
        else { for (let k = 0; k < d; k++) { const t = z[oi + k] - z[oj + k]; acc += t * t; } dist = Math.sqrt(acc); }
      } else {
        // pairwise-complete
        let m = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        for (let k = 0; k < d; k++) {
          const x = data[oi + k], y = data[oj + k];
          if (x !== x || y !== y) continue;
          m++;
          if (corr) { sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; } else { const t = x - y; sxx += t * t; }
        }
        if (corr) {
          const den = Math.sqrt((m * sxx - sx * sx) * (m * syy - sy * sy));
          dist = m >= 2 && den > 0 ? 1 - (m * sxy - sx * sy) / den : NaN;
        } else {
          dist = m ? Math.sqrt((sxx * d) / m) : NaN;
        }
      }
      if (corr && dist === dist) dist = Math.min(2, Math.max(0, dist));
      if (dist > maxD) maxD = dist;
      out[p++] = dist;
    }
  }
  for (let k = 0; k < out.length; k++) if (out[k] !== out[k]) out[k] = maxD;
  return out;
}

// Average linkage with the nearest-neighbour chain algorithm (Müllner 2011, as scipy):
// O(n^2) time, and it updates the condensed matrix D in place.
export function averageLinkage(D, n) {
  const at = (i, j) => (i < j ? n * i - ((i * (i + 1)) >> 1) + j - i - 1 : n * j - ((j * (j + 1)) >> 1) + i - j - 1);
  const size = new Float64Array(n).fill(1);
  const height = new Float64Array(n);
  const active = new Uint8Array(n).fill(1);
  const chain = new Int32Array(n);
  const raw = new Float64Array(3 * (n - 1)); // merges in the order found, as slot indices
  let top = 0, first = 0;
  for (let step = 0; step < n - 1; step++) {
    if (top === 0) { while (!active[first]) first++; chain[top++] = first; }
    let a, b, dab;
    for (;;) {
      a = chain[top - 1];
      const prev = top >= 2 ? chain[top - 2] : -1;
      // nearest active neighbour of a; ties go to the previous chain element, which
      // guarantees the chain ends in a pair of reciprocal nearest neighbours
      b = prev;
      dab = prev >= 0 ? D[at(a, prev)] : Infinity;
      for (let x = 0; x < n; x++) {
        if (!active[x] || x === a) continue;
        const v = D[at(a, x)];
        if (v < dab) { dab = v; b = x; }
      }
      if (b === prev) { top -= 2; break; }
      chain[top++] = b;
    }
    // the merged cluster takes slot b
    const sa = size[a], sb = size[b];
    for (let x = 0; x < n; x++) {
      if (!active[x] || x === a || x === b) continue;
      const ib = at(b, x);
      D[ib] = (sa * D[at(a, x)] + sb * D[ib]) / (sa + sb);
    }
    active[a] = 0;
    size[b] = sa + sb;
    const h = Math.max(dab, height[a], height[b]);
    height[b] = h;
    raw[3 * step] = a; raw[3 * step + 1] = b; raw[3 * step + 2] = h;
  }
  // sort by height and relabel slots as cluster ids (scipy's `label`)
  const m = n - 1;
  const byHeight = Array.from({ length: m }, (_, k) => k).sort((p, q) => raw[3 * p + 2] - raw[3 * q + 2] || p - q);
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const id = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const merges = new Float64Array(3 * m);
  byHeight.forEach((k, t) => {
    const ra = find(raw[3 * k]), rb = find(raw[3 * k + 1]);
    merges[3 * t] = Math.min(id[ra], id[rb]);
    merges[3 * t + 1] = Math.max(id[ra], id[rb]);
    merges[3 * t + 2] = raw[3 * k + 2];
    parent[ra] = rb;
    id[rb] = n + t;
  });
  return merges;
}

// Leaves left to right; of two children, the one holding the earlier input comes first,
// so ties keep the input order.
export function leafOrder(merges, n) {
  const m = merges.length / 3;
  const first = new Int32Array(n + m);
  for (let i = 0; i < n; i++) first[i] = i;
  for (let k = 0; k < m; k++) first[n + k] = Math.min(first[merges[3 * k]], first[merges[3 * k + 1]]);
  const out = new Int32Array(n);
  const stack = [n + m - 1];
  let p = 0;
  while (stack.length) {
    const v = stack.pop();
    if (v < n) { out[p++] = v; continue; }
    const a = merges[3 * (v - n)], b = merges[3 * (v - n) + 1];
    if (first[a] <= first[b]) stack.push(b, a); else stack.push(a, b);
  }
  return out;
}

// Line segments of one or more dendrograms, for drawing.
// trees: [{ merges, leafPos }], leafPos[i] = position of leaf i along the axis (in cells)
// -> { segs: Float64Array [pos1, h1, pos2, h2, ...], maxH } or null if nothing to draw
export function layoutDendrogram(trees) {
  let nSeg = 0, maxH = 0;
  for (const t of trees) {
    const m = t.merges.length / 3;
    nSeg += 3 * m;
    if (m) maxH = Math.max(maxH, t.merges[3 * m - 1]);
  }
  if (!nSeg) return null;
  const segs = new Float64Array(4 * nSeg);
  let s = 0;
  for (const t of trees) {
    const n = t.leafPos.length, m = t.merges.length / 3;
    const pos = new Float64Array(n + m), h = new Float64Array(n + m);
    pos.set(t.leafPos);
    for (let k = 0; k < m; k++) {
      const a = t.merges[3 * k], b = t.merges[3 * k + 1], hk = t.merges[3 * k + 2];
      pos[n + k] = (pos[a] + pos[b]) / 2;
      h[n + k] = hk;
      segs.set([pos[a], h[a], pos[a], hk, pos[b], h[b], pos[b], hk, pos[a], hk, pos[b], hk], s);
      s += 12;
    }
  }
  return { segs, maxH: maxH || 1 };
}
