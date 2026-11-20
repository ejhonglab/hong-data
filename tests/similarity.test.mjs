// Tests of how alike vectors are (app/static/similarity.js): `similarityMatrix`'s quick path (vectors with nothing
// missing, as unit vectors' dot products) gives what `similarity` gives pair by pair.
// Run with: node tests/similarity.test.mjs (tests/test_compare.py does, where there is a node).
import assert from 'node:assert/strict';
import { similarity, similarityMatrix } from '../app/static/similarity.js';

// a seeded random number generator (mulberry32)
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(1);
// vectors of spike counts (ties, zeros), some with values missing, one constant, one all 0, one too short
const vectors = Array.from({ length: 40 }, (_, i) => Float32Array.from({ length: 12 }, () => (i % 7 === 3 && rand() < 0.3 ? NaN : Math.floor(rand() * 4))));
vectors.push(new Float32Array(12).fill(2), new Float32Array(12), Float32Array.from([1, 2, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]));
for (const metric of ['pearson', 'spearman', 'cosine']) {
  const n = vectors.length, m = similarityMatrix(vectors, metric);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const want = Math.fround(similarity(metric, vectors[i], vectors[j])), got = m[i * n + j];
      if (Number.isNaN(want)) assert.ok(Number.isNaN(got), `${metric} ${i} ${j}: ${got}, not NaN`);
      else assert.ok(Math.abs(got - want) < 1e-6, `${metric} ${i} ${j}: ${got}, not ${want}`);
    }
  }
}
// a function of two given: used for every pair
assert.deepEqual(Array.from(similarityMatrix([[1], [2]], null, (a, b) => a[0] + b[0])), [2, 3, 3, 4]);
console.log('ok');
