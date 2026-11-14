// Tests of what the movie player edits a prediction's ROIs with (app/static/roi_edit.js).
// Run with: node tests/roi_edit.test.mjs (tests/test_roi_edit.py does, where there is a node).
import assert from 'node:assert/strict';
import { History, area, deletePoint, inside, nearestPoint, simplify, splitPoint } from '../app/static/roi_edit.js';

const close = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
// the distance from p to the outline
const toOutline = (p, pts) => Math.min(...pts.map((a, i) => {
  const b = pts[(i + 1) % pts.length], dx = b[0] - a[0], dy = b[1] - a[1], len = dx * dx + dy * dy;
  const t = len ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len)) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}));

// a square with a point every half pixel along its sides: its four corners are what is left
const square = [];
for (let x = 0; x < 10; x += 0.5) square.push([x, 0]);
for (let y = 0; y < 10; y += 0.5) square.push([10, y]);
for (let x = 10; x > 0; x -= 0.5) square.push([x, 10]);
for (let y = 10; y > 0; y -= 0.5) square.push([0, y]);
assert.deepEqual(simplify(square, 0.25).sort(), [[0, 0], [0, 10], [10, 0], [10, 10]].sort());
assert.equal(simplify(square, 0).length, square.length); // no tolerance: as it is
assert.deepEqual(simplify([...square, square[0]], 0), square); // (a point repeated: once)
assert.equal(simplify(square, 100).length, 3); // never fewer than three points
assert.deepEqual(simplify([[0, 0], [1, 0], [0, 1]], 5), [[0, 0], [1, 0], [0, 1]]);

// the outline of a blob as marching squares gives it (a point per pixel edge crossed): far fewer
// points, each one of its own, none of those left out further than the tolerance from it, the
// same wherever the outline starts
const blob = [];
for (let i = 0; i < 400; i++) {
  const a = (2 * Math.PI * i) / 400, r = 20 + 3 * Math.sin(3 * a) + 2 * Math.cos(5 * a);
  blob.push([Math.round((50 + r * Math.cos(a)) * 2) / 2, Math.round((50 + r * Math.sin(a)) * 2) / 2]);
}
for (const tol of [0.5, 1, 2]) {
  const s = simplify(blob, tol);
  assert.ok(s.length >= 3 && s.length < blob.length / 4, `${s.length} points at ${tol}`);
  assert.ok(s.every((p) => blob.some((q) => q[0] === p[0] && q[1] === p[1])));
  assert.ok(blob.every((p) => toOutline(p, s) <= tol + 1e-9));
  assert.ok(Math.abs(area(s) - area(blob)) / area(blob) < 0.1 * tol);
  const turned = [...blob.slice(137), ...blob.slice(0, 137)];
  assert.deepEqual(new Set(simplify(turned, tol).map(String)), new Set(s.map(String)));
}
assert.ok(simplify(blob, 2).length < simplify(blob, 1).length && simplify(blob, 1).length < simplify(blob, 0.5).length);
const before = JSON.stringify(blob);
simplify(blob, 1)[0][0] = -1; // (a copy: the outline given is not changed)
assert.equal(JSON.stringify(blob), before);

// as ImageJ: a point Shift-clicked becomes two, a third of the way to each of its neighbours; one Alt-clicked goes
const tri = [[0, 0], [9, 0], [0, 9]];
assert.deepEqual(splitPoint(tri, 1), [[0, 0], [6, 0], [6, 3], [0, 9]]);
const first = splitPoint(tri, 0); // (the point before the first is the last)
assert.ok(close(first[0][0], 0) && close(first[0][1], 3) && close(first[1][0], 3) && close(first[1][1], 0) && first.length === 4);
assert.equal(deletePoint(tri, 0), null); // (three are the fewest)
assert.deepEqual(deletePoint([...tri, [5, 5]], 3), tri);
assert.deepEqual(tri, [[0, 0], [9, 0], [0, 9]]);

assert.equal(nearestPoint(tri, 8.5, 0.5, 1), 1);
assert.equal(nearestPoint(tri, 5, 5, 1), -1);
assert.equal(nearestPoint([[0, 0], [1, 0], [0.4, 0]], 0.45, 0, 1), 2); // the nearest of those in reach
assert.ok(inside(tri, 2, 2) && !inside(tri, 7, 7) && !inside(tri, -1, 2));
assert.ok(close(area(tri), 40.5) && close(area([...tri].reverse()), 40.5));

const h = new History('a', 3);
assert.ok(!h.canUndo && !h.canRedo && h.undo() === undefined && h.current === 'a');
h.push('b'); h.push('c');
assert.equal(h.undo(), 'b');
assert.equal(h.undo(), 'a');
assert.equal(h.redo(), 'b');
h.push(null); // (a state may be none; those undone are gone)
assert.ok(h.current === null && !h.canRedo && h.undo() === 'b');
h.redo(); h.push('d'); // (the oldest go, past its limit)
assert.deepEqual(h.states, ['b', null, 'd']);

console.log('roi_edit.js: ok');
