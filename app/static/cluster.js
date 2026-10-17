// Hierarchical clustering for ordering heatmap rows and columns. Average linkage
// (UPGMA) on correlation (1 - Pearson r) or Euclidean distance. Distances use only the
// entries both vectors have, like pandas' DataFrame.corr, so missing values (NaN) are
// fine; pairs with too little in common get the largest distance found.
//
// ClusterPool runs jobs in web workers (cluster-worker.js): each job's distance matrix
// is computed in blocks of rows spread over the workers, and its linkage then runs in
// one worker directly on those blocks, never copied into one buffer. Workers cannot
// share memory here (that needs HTTPS and cross-origin isolation), so blocks are moved
// (transferred) between them instead.
//
// merges use scipy's linkage convention: merge k joins clusters a and b (ids < n are
// the input vectors, n + j is the cluster made by merge j) at height h, sorted by h.

export const MAX_CLUSTER_ITEMS = 15000;

// Everything in the calling thread (small inputs, tests).
// data: Float32Array of n vectors of length d (row-major)
// -> { order: Int32Array (leaf order), merges: Float64Array [a0, b0, h0, a1, ...] }
export function cluster(data, n, d, metric = 'correlation') {
  if (n > MAX_CLUSTER_ITEMS) throw new Error(`too many items to cluster (${n} > ${MAX_CLUSTER_ITEMS})`);
  if (n < 2) return trivial(n);
  const { block, maxD } = rowDistances(prepare(data, n, d, metric), 0, n - 1);
  const merges = averageLinkage([{ r0: 0, r1: n - 1, data: block }], n, maxD);
  return { order: leafOrder(merges, n), merges };
}

const trivial = (n) => ({ order: Int32Array.from({ length: n }, (_, i) => i), merges: new Float64Array(0) });

// ------------------------------------------------------------------- distances
// The vectors, prepared for rowDistances. For correlation, those without missing values
// are centred and scaled to unit length once, so that their pairs need one dot product.
// For pairs with missing values: each vector with NaN as 0 (x0), its sum and sum of
// squares, and where its NaNs are (row i's at nanIdx[nanStart[i] .. nanStart[i + 1]]).
export function prepare(data, n, d, metric) {
  const corr = metric === 'correlation';
  const complete = new Uint8Array(n);
  const z = corr ? new Float32Array(n * d) : data;
  const x0 = new Float32Array(n * d), sum = new Float64Array(n), sumSq = new Float64Array(n);
  const nanStart = new Int32Array(n + 1), nans = [];
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let s = 0, ss = 0;
    for (let k = 0; k < d; k++) {
      const v = data[o + k];
      if (v !== v) { nans.push(k); continue; }
      x0[o + k] = v; s += v; ss += v * v;
    }
    sum[i] = s; sumSq[i] = ss;
    nanStart[i + 1] = nans.length;
    if (nanStart[i + 1] > nanStart[i]) continue;
    complete[i] = 1;
    if (!corr) continue;
    const mean = s / d;
    let cs = 0;
    for (let k = 0; k < d; k++) { const c = data[o + k] - mean; cs += c * c; }
    const scale = cs > 0 ? 1 / Math.sqrt(cs) : NaN; // constant vector: correlation undefined
    for (let k = 0; k < d; k++) z[o + k] = (data[o + k] - mean) * scale;
  }
  return { data, n, d, corr, complete, z, x0, sum, sumSq, nanStart, nanIdx: Int32Array.from(nans) };
}

// number of condensed-matrix entries in rows r0 <= i < r1 (row i holds the n - 1 - i pairs (i, j > i))
export const pairsInRows = (n, r0, r1) => ((r1 - r0) * (2 * n - r0 - r1 - 1)) / 2;

// Condensed distance-matrix entries of rows r0 <= i < r1, in scipy's pdist order
// ((i, j) for every j > i). Undefined distances are NaN.
// -> { block: Float32Array, maxD: largest finite distance }
//
// Pairs are visited in tiles of rows (TILE_I rows x enough later rows to fill ~32 kB),
// so that the vectors being compared stay in the CPU cache; going row by row instead
// streams every later vector through memory once per row, which several workers at
// once make memory-bound.
const TILE_I = 32;
export function rowDistances({ data, n, d, corr, complete, z, x0, sum, sumSq, nanStart, nanIdx }, r0, r1) {
  const block = new Float32Array(pairsInRows(n, r0, r1));
  const base = new Float64Array(r1 - r0); // entry (i, j) is block[base[i - r0] + j]
  for (let i = r0, s = 0; i < r1; i++) { base[i - r0] = s - i - 1; s += n - 1 - i; }
  const tileJ = Math.max(32, Math.floor(8192 / d));
  let maxD = 0;
  for (let i0 = r0; i0 < r1; i0 += TILE_I) {
    const i1 = Math.min(r1, i0 + TILE_I);
    for (let j0 = i0 + 1; j0 < n; j0 += tileJ) {
      const j1 = Math.min(n, j0 + tileJ);
      for (let i = i0; i < i1; i++) {
        const oi = i * d, bi = base[i - r0];
        for (let j = Math.max(j0, i + 1); j < j1; j++) {
          const oj = j * d;
          let dist;
          if (complete[i] && complete[j]) {
            dist = corr ? 1 - dot(z, oi, z, oj, d) : Math.sqrt(sqDiff(z, oi, z, oj, d));
          } else {
            // pairwise-complete: the dot product with NaN as 0 only has shared terms;
            // sums over the shared entries are whole-row sums less the other row's NaNs
            const sxy = dot(x0, oi, x0, oj, d);
            let m = d - (nanStart[i + 1] - nanStart[i]), sx = sum[i], sxx = sumSq[i], sy = sum[j], syy = sumSq[j];
            for (let q = nanStart[j]; q < nanStart[j + 1]; q++) {
              const k = nanIdx[q], v = x0[oi + k];
              if (data[oi + k] === data[oi + k]) { m--; sx -= v; sxx -= v * v; }
            }
            for (let q = nanStart[i]; q < nanStart[i + 1]; q++) { const v = x0[oj + nanIdx[q]]; sy -= v; syy -= v * v; }
            if (corr) {
              const den = Math.sqrt((m * sxx - sx * sx) * (m * syy - sy * sy));
              dist = m >= 2 && den > 0 ? 1 - (m * sxy - sx * sy) / den : NaN;
            } else {
              dist = m ? Math.sqrt((Math.max(0, sxx + syy - 2 * sxy) * d) / m) : NaN;
            }
          }
          if (corr && dist === dist) dist = Math.min(2, Math.max(0, dist));
          if (dist > maxD) maxD = dist;
          block[bi + j] = dist;
        }
      }
    }
  }
  return { block, maxD };
}

// Sums over k < d with four partial sums, so that the additions can overlap.
function dot(a, oa, b, ob, d) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
  for (; k + 3 < d; k += 4) {
    s0 += a[oa + k] * b[ob + k]; s1 += a[oa + k + 1] * b[ob + k + 1];
    s2 += a[oa + k + 2] * b[ob + k + 2]; s3 += a[oa + k + 3] * b[ob + k + 3];
  }
  for (; k < d; k++) s0 += a[oa + k] * b[ob + k];
  return s0 + s1 + (s2 + s3);
}

function sqDiff(a, oa, b, ob, d) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
  for (; k + 3 < d; k += 4) {
    const t0 = a[oa + k] - b[ob + k], t1 = a[oa + k + 1] - b[ob + k + 1];
    const t2 = a[oa + k + 2] - b[ob + k + 2], t3 = a[oa + k + 3] - b[ob + k + 3];
    s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3;
  }
  for (; k < d; k++) { const t = a[oa + k] - b[ob + k]; s0 += t * t; }
  return s0 + s1 + (s2 + s3);
}

// --------------------------------------------------------------------- linkage
// Average linkage with Müllner's (2011) "generic" algorithm. The condensed matrix is
// given as row blocks [{r0, r1, data}] covering rows 0..n-2 in any order, and is
// updated in place; NaN entries count as `fill`. onProgress(fraction of merges done)
// is called now and then.
//
// Each row i keeps its nearest neighbour among the rows after it (a lower bound, made
// exact when needed), and a heap gives the closest pair overall. Unlike the nearest-
// neighbour chain (scipy's choice), searches then only read rows, which are contiguous;
// reading a column means a cache miss per entry, and made the chain ~1.5x slower here.
// Same tree as the chain (up to ties), as average linkage is reducible.
export function averageLinkage(blocks, n, fill, onProgress) {
  // entry (i, j), i < j, is rows[i][off[i] + j]
  const rows = new Array(n).fill(blocks[0].data);
  const off = new Int32Array(n);
  for (const b of blocks) {
    const D = b.data;
    for (let k = 0; k < D.length; k++) if (D[k] !== D[k]) D[k] = fill;
    let start = 0;
    for (let i = b.r0; i < b.r1; i++) { rows[i] = D; off[i] = start - i - 1; start += n - 1 - i; }
  }
  const size = new Float64Array(n).fill(1);
  const height = new Float64Array(n);
  const active = new Uint8Array(n).fill(1);
  const nghbr = new Int32Array(n).fill(-1); // nearest active j > i
  const mindist = new Float64Array(n).fill(Infinity); // <= distance to it; equal when up to date
  const rowMin = (i) => {
    let best = Infinity, arg = -1;
    const r = rows[i], o = off[i];
    for (let j = i + 1; j < n; j++) if (active[j]) { const v = r[o + j]; if (v < best) { best = v; arg = j; } }
    nghbr[i] = arg;
    mindist[i] = best;
  };
  // binary min-heap of rows 0..n-2 by (mindist, index)
  const heap = new Int32Array(n), pos = new Int32Array(n).fill(-1);
  let hn = 0;
  const less = (i, j) => mindist[i] < mindist[j] || (mindist[i] === mindist[j] && i < j);
  const place = (k, i) => { heap[k] = i; pos[i] = k; };
  const up = (k) => {
    const i = heap[k];
    while (k > 0) { const p = (k - 1) >> 1; if (!less(i, heap[p])) break; place(k, heap[p]); k = p; }
    place(k, i);
  };
  const down = (k) => {
    const i = heap[k];
    for (;;) {
      let c = 2 * k + 1;
      if (c >= hn) break;
      if (c + 1 < hn && less(heap[c + 1], heap[c])) c++;
      if (!less(heap[c], i)) break;
      place(k, heap[c]);
      k = c;
    }
    place(k, i);
  };
  const update = (i) => { if (pos[i] >= 0) { up(pos[i]); down(pos[i]); } };
  const remove = (i) => {
    const k = pos[i];
    pos[i] = -1;
    if (k === --hn) return;
    place(k, heap[hn]);
    update(heap[k]);
  };
  for (let i = 0; i < n - 1; i++) { rowMin(i); place(hn++, i); }
  for (let k = (hn >> 1) - 1; k >= 0; k--) down(k);

  const raw = new Float64Array(3 * (n - 1)); // merges in the order found, as slot indices
  const every = Math.max(1, Math.floor(n / 64));
  for (let step = 0; step < n - 1; step++) {
    // closest pair: the top row's bound is the smallest, so once exact it is the minimum
    let a, b;
    for (;;) {
      a = heap[0];
      b = nghbr[a];
      if (b >= 0 && rows[a][off[a] + b] === mindist[a]) break;
      rowMin(a);
      update(a);
    }
    const dab = mindist[a];
    // the merged cluster takes slot b (> a); slot a goes away
    const sa = size[a], sb = size[b], inv = 1 / (sa + sb);
    const rowA = rows[a], offA = off[a], rowB = rows[b], offB = off[b];
    for (let x = 0; x < a; x++) { // (x, a) and (x, b), in row x
      if (!active[x]) continue;
      const r = rows[x], o = off[x];
      r[o + b] = (sa * r[o + a] + sb * r[o + b]) * inv;
      if (nghbr[x] === a) nghbr[x] = b; // the new distance is >= mindist[x]: still a bound
    }
    for (let x = a + 1; x < b; x++) { // (a, x) in row a, (x, b) in row x
      if (!active[x]) continue;
      const r = rows[x], o = off[x];
      const v = Math.fround((sa * rowA[offA + x] + sb * r[o + b]) * inv);
      r[o + b] = v;
      if (v < mindist[x]) { mindist[x] = v; nghbr[x] = b; update(x); }
    }
    for (let x = b + 1; x < n; x++) { // rows a and b
      if (active[x]) rowB[offB + x] = (sa * rowA[offA + x] + sb * rowB[offB + x]) * inv;
    }
    active[a] = 0;
    remove(a);
    size[b] = sa + sb;
    rowMin(b);
    update(b);
    const h = Math.max(dab, height[a], height[b]);
    height[b] = h;
    raw[3 * step] = a; raw[3 * step + 1] = b; raw[3 * step + 2] = h;
    if (onProgress && step % every === 0) onProgress(step / (n - 1));
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

// ------------------------------------------------------------------ worker pool
const MIN_BLOCK_PAIRS = 1 << 20; // smaller jobs are one block (one worker)
const BLOCKS_PER_WORKER = 4; // for load balancing
// distance matrices of jobs in progress at once stay under this (a larger job runs alone)
const MEMORY_BUDGET = 600e6;
const IDLE_MS = 60_000; // workers are stopped after this long unused

// Rows split into blocks of roughly equal numbers of pairs.
function planBlocks(n, workers) {
  const target = Math.max(MIN_BLOCK_PAIRS, Math.ceil((n * (n - 1)) / 2 / (BLOCKS_PER_WORKER * workers)));
  const out = [];
  let r0 = 0, acc = 0;
  for (let i = 0; i < n - 1; i++) {
    acc += n - 1 - i;
    if (acc >= target || i === n - 2) { out.push({ r0, r1: i + 1 }); r0 = i + 1; acc = 0; }
  }
  return out;
}

export class ClusterPool {
  constructor(size = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8))) {
    this.size = size;
    this.workers = null;
    this.current = null; // {reject} of the run in progress
    this.nextId = 1;
    this.idleTimer = 0;
  }

  // Clusters jobs [{data, n, d, metric}] (data is copied, not consumed). Resolves to
  // [{order, merges}] in job order; onProgress({distances, linkage}) gets the fraction of
  // each phase done. A new run cancels the one in progress (its promise rejects with
  // error.cancelled set).
  run(jobs, onProgress = () => {}) {
    if (this.current) this.stop(Object.assign(new Error('cancelled'), { cancelled: true }));
    clearTimeout(this.idleTimer);
    const big = jobs.find((j) => j.n > MAX_CLUSTER_ITEMS);
    if (big) return Promise.reject(new Error(`too many items to cluster (${big.n} > ${MAX_CLUSTER_ITEMS})`));
    const results = jobs.map((j) => (j.n < 2 ? trivial(j.n) : null));
    const js = jobs.map((j, k) => ({ ...j, k, id: this.nextId++, bytes: 2 * j.n * (j.n - 1) })).filter((j) => !results[j.k]);
    if (!js.length) return Promise.resolve(results);
    if (!this.workers) this.workers = Array.from({ length: this.size }, () => new Worker(new URL('./cluster-worker.js', import.meta.url), { type: 'module' }));
    const W = this.workers;
    return new Promise((resolve, reject) => {
      this.current = { reject };
      const busy = new Int32Array(W.length); // tasks sent to each worker and not yet answered
      const byId = new Map();
      for (const j of js) {
        Object.assign(j, { blocks: planBlocks(j.n, W.length), got: [], maxD: 0, holders: new Set(), linked: 0 });
        j.left = j.blocks.length;
        byId.set(j.id, j);
      }
      const waiting = js.slice().sort((a, b) => b.n - a.n);
      const tasks = [];
      let inFlight = 0, done = 0, pairsDone = 0;
      const pairsTotal = js.reduce((s, j) => s + (j.n * (j.n - 1)) / 2, 0);
      const n2Total = js.reduce((s, j) => s + j.n * j.n, 0);

      const start = () => { // jobs whose distance matrices fit in the memory budget
        while (waiting.length && (!inFlight || inFlight + waiting[0].bytes <= MEMORY_BUDGET)) {
          const j = waiting.shift();
          inFlight += j.bytes;
          for (const b of j.blocks) tasks.push({ j, b });
        }
      };
      const pump = () => { // distance blocks to idle workers
        for (let w = 0; w < W.length && tasks.length; w++) {
          if (busy[w]) continue;
          const { j, b } = tasks.shift();
          busy[w]++;
          const msg = { type: 'dist', job: j.id, r0: b.r0, r1: b.r1 };
          if (!j.holders.has(w)) { j.holders.add(w); Object.assign(msg, { vectors: j.data, n: j.n, d: j.d, metric: j.metric }); }
          W[w].postMessage(msg);
        }
      };
      const report = () => onProgress({
        distances: pairsDone / pairsTotal,
        linkage: js.reduce((s, j) => s + j.linked * j.n * j.n, 0) / n2Total,
      });

      W.forEach((worker, w) => {
        worker.onerror = (e) => { e.preventDefault(); this.stop(new Error(e.message || 'a clustering worker failed')); };
        worker.onmessage = ({ data: m }) => {
          const j = byId.get(m.job);
          if (m.type === 'error') {
            const oom = j && (m.outOfMemory || /memory|allocat/i.test(m.message));
            this.stop(new Error(oom
              ? `not enough memory to cluster ${j.n.toLocaleString()} items (their distances take ${Math.round(j.bytes / 1e6)} MB)`
              : m.message));
            return;
          }
          if (m.type === 'block') {
            busy[w]--;
            pairsDone += m.block.length;
            j.got.push({ r0: m.r0, r1: m.r1, data: m.block });
            j.maxD = Math.max(j.maxD, m.maxD);
            if (--j.left === 0) {
              // all blocks in: the worker that sent the last one (now idle) does the linkage
              for (const v of j.holders) W[v].postMessage({ type: 'forget', job: j.id });
              busy[w]++;
              W[w].postMessage({ type: 'link', job: j.id, n: j.n, fill: j.maxD, blocks: j.got }, j.got.map((x) => x.data.buffer));
              j.got = null;
            }
          } else if (m.type === 'progress') {
            j.linked = m.fraction;
          } else if (m.type === 'result') {
            busy[w]--;
            results[j.k] = { order: m.order, merges: m.merges };
            j.linked = 1;
            inFlight -= j.bytes;
            if (++done === js.length) {
              this.current = null;
              this.idleTimer = setTimeout(() => this.stop(), IDLE_MS);
              report();
              resolve(results);
              return;
            }
            start();
          }
          pump();
          report();
        };
      });
      start();
      pump();
      report();
    });
  }

  // Terminates the workers (they are started again when needed); rejects the run in
  // progress, if any, with `err`.
  stop(err = Object.assign(new Error('cancelled'), { cancelled: true })) {
    clearTimeout(this.idleTimer);
    this.workers?.forEach((w) => w.terminate());
    this.workers = null;
    const c = this.current;
    this.current = null;
    c?.reject(err);
  }
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
