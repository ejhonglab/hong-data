// Naming a fly's ROIs again as al-diag-seg does (its identity.assign_identities), from what the fit worker kept of every
// ROI and glomerulus (naming.py's make_costs: each term of the cost of each pair, and what the layout term is made of),
// under any weights of the terms and any most a name may cost: for the naming explorer (naming.js), which shows what
// changes as those do. No page needed: tests/naming_solve.test.mjs runs it against al-diag-seg's own names.

export const TERMS = ['tuning', 'position', 'shape', 'atlas', 'layout'];
const CAP = 4; // the SDs al-diag-seg's position, shape and layout terms are capped at (and divided by)

// The costs as kept (`doc`, naming.py's) in typed arrays: `nL` ROIs (rows) × `nN` glomeruli (columns); a term with no
// value for a pair is NaN there (the atlas term of a glomerulus the atlas has no map of: nothing is added to its cost).
export function prepare(doc) {
  const nL = doc.labels.length, nN = doc.names.length, terms = {};
  for (const k of TERMS) {
    const a = new Float64Array(nL * nN);
    for (let i = 0; i < nL; i++) for (let j = 0; j < nN; j++) a[i * nN + j] = doc.terms[k][i][j] ?? NaN;
    terms[k] = a;
  }
  const col = new Map(doc.names.map((n, j) => [n, j]));
  return { nL, nN, terms, labels: doc.labels, names: doc.names, reference: doc.reference, hasAtlas: doc.weights.atlas != null,
    rounds: doc.rounds, roiPos: doc.roi_pos, expPos: doc.exp_pos, posSd: doc.pos_sd, neighbors: doc.neighbors,
    assigned: Int32Array.from(doc.assigned, (n) => (n == null ? -1 : col.get(n))), weights: { ...doc.weights }, maxCost: doc.max_cost };
}

// The cheapest way to give each of `nL` rows a column of `cost` (nL × nN, row by row) or none, none costing `maxCost`:
// a row's column, or -1. As al-diag-seg's _solve: a linear sum assignment over the costs and one more column per row
// that costs `maxCost`. Where several ways cost the same (ROIs far from every glomerulus of the reference atlas alone
// cost the same for each of them), which one comes out is the algorithm's: so this is SciPy's own
// (linear_sum_assignment: Crouse's shortest augmenting paths, as its rectangular_lsap.cpp has them), step for step.
export function assign(cost, nL, nN, maxCost) {
  const nr = nL, nc = nN + nL;
  const u = new Float64Array(nr), v = new Float64Array(nc), shortest = new Float64Array(nc);
  const path = new Int32Array(nc).fill(-1), col4row = new Int32Array(nr).fill(-1), row4col = new Int32Array(nc).fill(-1);
  const inRows = new Uint8Array(nr), inCols = new Uint8Array(nc), remaining = new Int32Array(nc);
  for (let cur = 0; cur < nr; cur++) {
    // the shortest augmenting path from row `cur` to a column with no row
    let minVal = 0, i = cur, left = nc, sink = -1;
    for (let it = 0; it < nc; it++) remaining[it] = nc - it - 1;
    inRows.fill(0);
    inCols.fill(0);
    shortest.fill(Infinity);
    while (sink === -1) {
      let index = -1, lowest = Infinity;
      inRows[i] = 1;
      for (let it = 0; it < left; it++) {
        const j = remaining[it];
        const r = minVal + (j < nN ? cost[i * nN + j] : maxCost) - u[i] - v[j];
        if (r < shortest[j]) { path[j] = i; shortest[j] = r; }
        // (of several columns as near, one with no row: a new sink)
        if (shortest[j] < lowest || (shortest[j] === lowest && row4col[j] === -1)) { lowest = shortest[j]; index = it; }
      }
      minVal = lowest;
      const j = remaining[index];
      if (row4col[j] === -1) sink = j; else i = row4col[j];
      inCols[j] = 1;
      remaining[index] = remaining[--left];
    }
    // the dual variables, then the rows and columns along the path matched anew
    u[cur] += minVal;
    for (let k = 0; k < nr; k++) if (inRows[k] && k !== cur) u[k] += minVal - shortest[col4row[k]];
    for (let j = 0; j < nc; j++) if (inCols[j]) v[j] -= minVal - shortest[j];
    for (let j = sink; ;) {
      const k = path[j], was = col4row[k];
      row4col[j] = k;
      col4row[k] = j;
      j = was;
      if (k === cur) break;
    }
  }
  return Int32Array.from(col4row, (j) => (j < nN ? j : -1));
}

// The layout term of every pair, given the names the ROIs have (`names`: each row's column, or -1): for ROI i as
// glomerulus j, how far i's offsets to its named neighbors are from the offsets between j and their glomeruli where
// they are expected, in SDs (capped at 4, divided by 4), averaged over those neighbors; 0 for an ROI with none named.
export function layout(c, names) {
  const { nL, nN, roiPos, expPos, posSd } = c, out = new Float64Array(nL * nN), acc = new Float64Array(nN);
  for (let i = 0; i < nL; i++) {
    let n = 0;
    acc.fill(0);
    for (const k of c.neighbors[i]) {
      const a = names[k];
      if (a < 0) continue;
      const o0 = roiPos[i][0] - roiPos[k][0], o1 = roiPos[i][1] - roiPos[k][1], o2 = roiPos[i][2] - roiPos[k][2], e = expPos[a];
      for (let j = 0; j < nN; j++) {
        const d0 = expPos[j][0] - e[0] - o0, d1 = expPos[j][1] - e[1] - o1, d2 = expPos[j][2] - e[2] - o2;
        const scale = Math.sqrt(posSd[j] * posSd[j] + posSd[a] * posSd[a]) + 1e-8;
        acc[j] += Math.min(Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2) / scale, CAP) / CAP;
      }
      n += 1;
    }
    if (n) for (let j = 0; j < nN; j++) out[i * nN + j] = acc[j] / n;
  }
  return out;
}

// The sum of the weights the cost is divided by: of every term that is on (the atlas' only with an atlas).
export const weightSum = (c, w) => w.tuning + w.position + w.shape + w.layout + (c.hasAtlas ? w.atlas : 0);

// Names the ROIs under `weights` ({term: weight}) and `maxCost`, as al-diag-seg would: matched by the terms that do
// not depend on the other names, then up to `c.rounds` more times with the layout term of the names so far, until
// they no longer change. Returns `names` (each row's column, or -1), `cost` (every pair's, with the layout term of
// those names: what al-diag-seg reports), `layout` (that term), `sum` (of the weights) and `settled` (whether the
// names stopped changing: if not, the costs are of the last names, which were matched by the layout of those before).
export function solve(c, weights, maxCost) {
  const { nL, nN, terms } = c, sum = weightSum(c, weights), n = nL * nN;
  const base = new Float64Array(n), atlas = c.hasAtlas && weights.atlas ? weights.atlas : 0;
  for (let k = 0; k < n; k++) {
    const a = terms.atlas[k];
    base[k] = (weights.tuning * terms.tuning[k] + weights.position * terms.position[k] + weights.shape * terms.shape[k]
      + (atlas && !Number.isNaN(a) ? atlas * a : 0)) / sum;
  }
  const wl = weights.layout / sum, cost = new Float64Array(n);
  let names = assign(base, nL, nN, maxCost), lay = new Float64Array(n), settled = !wl;
  for (let r = 0; r < (wl ? c.rounds : 0); r++) {
    lay = layout(c, names);
    for (let k = 0; k < n; k++) cost[k] = base[k] + wl * lay[k];
    const next = assign(cost, nL, nN, maxCost);
    if (next.every((v, i) => v === names[i])) { settled = true; break; }
    names = next;
  }
  if (wl && !settled) lay = layout(c, names);
  for (let k = 0; k < n; k++) cost[k] = base[k] + wl * lay[k];
  return { names, cost, layout: lay, sum, settled };
}

// Why ROI `i` has no name under a solution (`s`: solve's) at `maxCost`: 'costly' (every glomerulus costs more than a
// name may), 'taken' (those that cost no more all name other ROIs) or 'unsettled' (one that costs no more is free: the
// costs are of the last names, which were matched by the layout term of those before); null for one named.
export function whyUnnamed(c, s, i, maxCost) {
  if (s.names[i] >= 0) return null;
  const taken = new Set(s.names);
  let under = false;
  for (let j = 0; j < c.nN; j++) {
    if (s.cost[i * c.nN + j] <= maxCost) {
      if (!taken.has(j)) return 'unsettled';
      under = true;
    }
  }
  return under ? 'taken' : 'costly';
}

// The columns of row `i` by cost, least first (`k`: how many; all by default).
export function cheapest(c, cost, i, k = c.nN) {
  const row = cost.subarray(i * c.nN, (i + 1) * c.nN);
  return Array.from({ length: c.nN }, (_, j) => j).sort((a, b) => row[a] - row[b]).slice(0, k);
}
