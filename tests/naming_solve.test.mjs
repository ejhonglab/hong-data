// Tests of naming ROIs again in the page (app/static/naming_solve.js), against what SciPy's linear_sum_assignment gives as
// al-diag-seg uses it (tests/naming_solve_fixture.json, made and checked by tests/test_naming.py): the matching alone,
// ties and all, and the ROIs named under several weights of the terms and most-a-name-may-cost.
// Run with: node tests/naming_solve.test.mjs (tests/test_naming.py does, where there is a node).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { assign, cheapest, layout, prepare, solve, weightSum, whyUnnamed } from '../app/static/naming_solve.js';

const fixture = JSON.parse(fs.readFileSync(new URL('./naming_solve_fixture.json', import.meta.url)));

// the matching: each row's column or none, as SciPy's (which of several as cheap comes out is the algorithm's)
for (const [n, m] of fixture.matchings.entries()) {
  const got = assign(Float64Array.from(m.cost.flat()), m.cost.length, m.cost[0].length, m.max_cost);
  assert.deepEqual([...got], m.expected, `matching ${n}`);
}

// the ROIs named again: the names, and what each name costs
let settings = 0;
for (const [n, d] of fixture.docs.entries()) {
  const c = prepare(d.doc);
  assert.equal(c.nL, d.doc.labels.length);
  assert.equal(c.hasAtlas, d.doc.weights.atlas != null);
  for (const [k, e] of d.expected.entries()) {
    const s = solve(c, e.weights, e.max_cost), names = [...s.names].map((j) => (j < 0 ? null : c.names[j]));
    assert.deepEqual(names, e.names, `doc ${n}, setting ${k}`);
    assert.equal(s.settled, e.settled, `doc ${n}, setting ${k}: settled`);
    assert.ok(Math.abs(s.sum - weightSum(c, e.weights)) < 1e-12);
    s.names.forEach((j, i) => { if (j >= 0) assert.ok(Math.abs(s.cost[i * c.nN + j] - e.costs[i]) < 1e-9, `doc ${n}, setting ${k}, ROI ${i}: cost`); });
    // an ROI left unnamed says why; one whose every glomerulus costs more than a name may is 'costly'
    for (let i = 0; i < c.nL; i++) {
      const why = whyUnnamed(c, s, i, e.max_cost), least = s.cost[i * c.nN + cheapest(c, s.cost, i, 1)[0]];
      assert.equal(why == null, s.names[i] >= 0);
      if (why) assert.equal(why === 'costly', least > e.max_cost);
      if (why === 'unsettled') assert.equal(s.settled, false);
    }
    settings += 1;
  }
  // the model's own weights and threshold: the names al-diag-seg gave (the doc's)
  const own = solve(c, c.weights, c.maxCost);
  assert.deepEqual([...own.names], [...c.assigned], `doc ${n}: the model's names`);
}

// the layout term: 0 for an ROI with no named neighbor; for one with one, the distance of its offset from the expected one, in SDs
{
  const c = { nL: 2, nN: 2, roiPos: [[0, 0, 0], [0, 3, 4]], expPos: [[0, 0, 0], [0, 6, 8]], posSd: [3, 4], neighbors: [[1], [0]] };
  assert.deepEqual([...layout(c, [-1, -1])], [0, 0, 0, 0]);
  const l = layout(c, [0, -1]); // ROI 0 is glomerulus 0: ROI 1, 5 away, as glomerulus 1 (expected 10 away) is 5 off, over sqrt(9 + 16) = 5
  assert.deepEqual([...l.slice(0, 2)], [0, 0]);
  assert.ok(Math.abs(l[3] - 1 / 4) < 1e-7 && Math.abs(l[2] - Math.min(5 / Math.sqrt(18), 4) / 4) < 1e-7);
}
console.log(`ok: ${fixture.matchings.length} matchings, ${settings} namings of ${fixture.docs.length} flies`);
