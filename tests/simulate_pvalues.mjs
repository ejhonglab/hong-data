// How often each test of the Compare tab's correlations between similarities (permutation,
// bootstraps of odor pairs and of odors: app/static/pvalue-worker.js) gives p < 0.05 and
// p < 0.2 for unrelated spaces, which a valid test does for 5% and 20% of them. The
// tables of design.md ("Statistical testing") are its output.
//
// Each space has n odors (15 or 30), each 30 values (as across 30 glomeruli); its
// similarities are their Pearson correlations, and two spaces' are compared by Spearman
// correlation. Odors of a space are made:
// - alike: every value independent, standard normal;
// - saturated: g·u + e, u the space's pattern, g each odor's gain (2 × standard normal),
//   e noise;
// - broad: as saturated, with g = exp(0.7 × standard normal), so some odors respond much
//   more strongly than others.
// The numbers are seeded (each row by its odors and n), so a run gives the same tables.
//
// Run from the repository (a few minutes; --quick for a smaller run, to check it works):
//   gjs -m tests/simulate_pvalues.mjs [--quick]

import { similarityMatrix, sharedSimilarities } from '../app/static/similarity.js';

const out = globalThis.print ?? console.log;
const progress = globalThis.printerr ?? console.error;
const args = globalThis.ARGV ?? globalThis.process?.argv.slice(2) ?? [];
const QUICK = args.includes('--quick');
const SPACES = QUICK ? 100 : 500; // pairs of unrelated spaces per row
const RESAMPLINGS = QUICK ? 200 : 1000;
const DIMS = 30;
const ODORS = [15, 30];
const KINDS = ['alike', 'saturated', 'broad'];
const TESTS = [['pairs', 'bootstrap of odor pairs'], ['odors', 'bootstrap of odors'], ['permute', 'permutation (Mantel)']];

// the worker, run here: its `self` this script's, set before it is loaded
let answer = null;
globalThis.self = { postMessage: (m) => { if (m.ps) answer = m; } };
await import('../app/static/pvalue-worker.js');

// the worker's answer to one message: {ps, valid}
function pValues(tests, unit) {
  answer = null;
  self.onmessage({ data: { tests, method: 'spearman', resamplings: RESAMPLINGS, unit } });
  return answer;
}

// sfc32 (Chris Doty-Humphrey's small fast counting generator), its state from one seed by
// splitmix32: uniform numbers in [0, 1)
function generator(seed) {
  const split = () => {
    seed = (seed + 0x9e3779b9) | 0;
    let z = seed;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
  let a = split(), b = split(), c = split(), d = split();
  const next = () => {
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (b + (b << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let k = 0; k < 15; k++) next();
  return next;
}

function hash(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

// a space of n odors of a kind: their similarities, keyed by odor 0..n-1
function space(kind, n, random) {
  const normal = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random()); // Box–Muller
  const u = Array.from({ length: DIMS }, normal);
  const gain = { alike: () => 0, saturated: () => 2 * normal(), broad: () => Math.exp(0.7 * normal()) }[kind];
  const vectors = Array.from({ length: n }, () => { const g = gain(); return u.map((x) => g * x + normal()); });
  return { ids: vectors.map((_, i) => i), sim: similarityMatrix(vectors, 'pearson') };
}

const pct = (x) => `${(100 * x).toFixed(1)}%`;
const rows = [];
const started = Date.now();
for (const kind of KINDS) {
  const row = { kind, at: new Map() }; // test -> [[p < 0.05, p < 0.2] for each n]
  for (const n of ODORS) {
    const random = generator(hash(`${kind}:${n}`));
    const pairs = Array.from({ length: SPACES }, () => sharedSimilarities(space(kind, n, random), space(kind, n, random)));
    for (const [unit] of TESTS) {
      const tests = pairs.map((t, k) => (unit === 'pairs' ? { seed: k + 1, xs: t.xs, ys: t.ys } : { seed: k + 1, n: t.n, subA: t.subA, subB: t.subB }));
      const ps = pValues(tests, unit).ps.filter((p) => p === p);
      const share = (alpha) => ps.filter((p) => p < alpha).length / ps.length;
      row.at.set(unit, [...(row.at.get(unit) || []), [share(0.05), share(0.2)]]);
      progress(`${kind}, ${n} odors, ${unit}: p < 0.05 ${pct(share(0.05))}, p < 0.2 ${pct(share(0.2))} (${Math.round((Date.now() - started) / 1000)} s)`);
    }
  }
  rows.push(row);
}

const NAMES = { alike: 'alike', saturated: 'saturated', broad: 'broad responders' };
for (const [k, alpha] of [[0, '0.05'], [1, '0.2']]) {
  out(`\nUnrelated spaces with p < ${alpha}, for ${ODORS.join(' / ')} odors (${SPACES} pairs of spaces, ${RESAMPLINGS} resamplings or permutations each):\n`);
  out(`| Odors | ${TESTS.map(([, name]) => name).join(' | ')} |`);
  out(`|---|${TESTS.map(() => '---').join('|')}|`);
  for (const r of rows) out(`| ${NAMES[r.kind]} | ${TESTS.map(([unit]) => r.at.get(unit).map((s) => pct(s[k])).join(' / ')).join(' | ')} |`);
}
