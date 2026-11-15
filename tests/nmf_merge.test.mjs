// Tests of what the movie player merges ROIs found by NMF into a prediction's edited ROIs with
// (app/static/nmf_merge.js). Run with: node tests/nmf_merge.test.mjs (tests/test_roi_edit.py does).
import assert from 'node:assert/strict';
import { changes, copyPredicted, kindOf, merge, mergedIds, mergedOf, provisional, shownName, what } from '../app/static/nmf_merge.js';

// a square with a point at the middle of each side too (which simplifying leaves out)
const sq = (x, y, s = 4) => [[x, y], [x + s / 2, y], [x + s, y], [x + s, y + s / 2], [x + s, y + s], [x + s / 2, y + s], [x, y + s], [x, y + s / 2]];
const names = (rois) => rois.map((r) => `${r.name}@${r.z}${r.nmf == null ? '' : `#${r.nmf}`}`).sort();

// a prediction's ROIs: DM1 (label 1) on planes 0 and 1, DM2 (2) on plane 0, and one left unnamed (3) on plane 1
const predicted = [{ name: 'DM1', z: 0, kind: 'glomerulus', paths: [sq(0, 0)] }, { name: 'DM1', z: 1, kind: 'glomerulus', paths: [sq(0, 0)] },
  { name: 'DM2', z: 0, kind: 'glomerulus', paths: [sq(10, 0, 8)] }, { name: '3', z: 1, kind: 'other', paths: [sq(20, 0)] }];
// what a look found: a new ROI (label 4) on both planes, more of DM2 on plane 1, and a part cut off DM2 on plane 0 (label 5)
const found = { run: 'r1', predicted: { 1: 'DM1', 2: 'DM2', 3: null },
  candidates: [
    { id: 1, kind: 'added', label: 4, of: null, planes: [0, 1], name: 'DM3', outlines: [{ z: 0, points: sq(30, 0) }, { z: 1, points: sq(30, 0) }] },
    { id: 2, kind: 'extended', label: 2, of: 2, of_name: 'DM2', planes: [1], name: 'DM2', outlines: [{ z: 1, points: sq(10, 0) }] },
    { id: 3, kind: 'split', label: 5, of: 2, of_name: 'DM2', planes: [0], name: null, outlines: [{ z: 0, points: sq(14, 0) }], rest: [{ z: 0, points: sq(10, 0) }] }],
  names: { kept: { 1: 'DM1', 2: 'DM2', 3: null, 4: 'DM3', 5: null }, joint: { 1: 'DM1', 2: 'DM5', 3: 'DM4', 4: 'DM3', 5: 'DM2' } },
  kinds: { DM1: 'glomerulus', DM2: 'glomerulus', DM3: 'glomerulus', DM4: 'glomerulus', DM5: 'glomerulus', odd: 'other' } };

assert.equal(shownName({ 1: 'DM1', 3: null }, 1), 'DM1');
assert.equal(shownName({ 1: 'DM1', 3: null }, 3), '3');
assert.deepEqual([kindOf(found.kinds, 'DM1'), kindOf(found.kinds, '12'), kindOf(found.kinds, 'odd'), kindOf(found.kinds, 'VA9'), kindOf(null, '7')],
  ['glomerulus', 'other', 'other', 'glomerulus', 'other']);
assert.deepEqual(found.candidates.map(what), ['new', 'more of DM2', 'part of DM2']);

// those not merged, to show: every outline of each, named as the look named them (numbered where it gave none)
let shown = provisional(found, null);
assert.deepEqual(shown.map((r) => [r.name, r.z, r.cand, r.first, r.kind]), [['DM3', 0, 1, true, 'glomerulus'], ['DM3', 1, 1, false, 'glomerulus'],
  ['DM2', 1, 2, true, 'glomerulus'], ['5', 0, 3, true, 'other']]);
assert.equal(shown[0].points.length, 8);
assert.deepEqual(provisional(null, null), []);

// a copy of the predicted ROIs with nothing merged: as before there were looks
let state = { rois: copyPredicted(predicted, found, null, 1), simplified: 1, pointsEdited: false, nmf: null };
assert.deepEqual(names(state.rois), ['3@1', 'DM1@0', 'DM1@1', 'DM2@0']);
assert.ok(state.rois.every((r) => r.points.length === 4 && r.nmf === undefined));
// (those to show are simplified as the edited ROIs are)
assert.ok(provisional(found, state).every((r) => r.points.length === 4));
assert.equal(mergedIds(found, state).size, 0);

// two merged with the predicted ROIs' names kept: added, the ROIs as they were (an outline edited by hand too)
state.rois[0] = { ...state.rois[0], points: [[0, 0], [9, 0], [9, 9]] };
state.pointsEdited = true;
const kept = { names: { 1: 'DM1', 2: 'DM2', 3: null, 4: 'DM3' }, kinds: {} };
let c = changes(found, state, kept, [1, 2]);
assert.deepEqual(c.added.map((r) => [r.id, r.kind, r.name, r.named, r.planes, r.of]), [[1, 'added', 'DM3', true, [0, 1], null], [2, 'extended', 'DM2', true, [1], 'DM2']]);
assert.deepEqual([c.renamed, c.kept], [[], 3]);
const first = merge(state, found, kept, [1, 2], false);
assert.deepEqual(names(first.rois), ['3@1', 'DM1@0', 'DM1@1', 'DM2@0', 'DM2@1#2', 'DM3@0#1', 'DM3@1#1']);
assert.deepEqual(first.nmf, { run: 'r1', merged: [1, 2], joint: false, names: kept.names });
assert.deepEqual(first.rois[0].points, [[0, 0], [9, 0], [9, 9]]);
assert.ok(first.rois.filter((r) => r.nmf != null).every((r) => r.points.length === 4) && first.pointsEdited && first.simplified === 1);
assert.deepEqual([...mergedIds(found, first)], [1, 2]);
assert.deepEqual(provisional(found, first).map((r) => r.cand), [3]);
assert.equal(state.rois.length, 4); // (the state before is as it was: to undo to)

// then the part cut off, all named again together: names change by label, DM2's outline on that plane is what is left of it
const joint = { names: found.names.joint, kinds: {} };
c = changes(found, first, joint, [1, 2, 3]);
assert.deepEqual(c.added.map((r) => [r.id, r.kind, r.name, r.of]), [[3, 'split', 'DM2', 'DM2']]);
assert.deepEqual(c.renamed, [{ label: 2, from: 'DM2', to: 'DM5', nmf: false }, { label: 3, from: '3', to: 'DM4', nmf: false }]);
assert.equal(c.kept, 2);
const second = merge(first, found, joint, [1, 2, 3], true);
assert.deepEqual(names(second.rois), ['DM1@0', 'DM1@1', 'DM2@0#3', 'DM3@0#1', 'DM3@1#1', 'DM4@1', 'DM5@0#3', 'DM5@1#2']);
assert.equal(second.rois.find((r) => r.name === 'DM4').kind, 'glomerulus');
assert.deepEqual(second.nmf.merged, [1, 2, 3]);
assert.ok(second.nmf.joint);
assert.deepEqual(provisional(found, second), []);
// (one merged before whose name changes is said to be from NMF)
const again = changes(found, second, { names: { ...found.names.joint, 4: null }, kinds: {} }, [1, 2, 3]);
assert.deepEqual(again.renamed, [{ label: 4, from: 'DM3', to: '4', nmf: true }]);
assert.deepEqual(again.added, []);

// the predicted ROIs copied again (simplified anew) have the same merged in, named the same
assert.deepEqual(names(copyPredicted(predicted, found, second.nmf, 0.5)), names(second.rois));
assert.deepEqual(names(copyPredicted(predicted, found, first.nmf, null)), names(first.rois));
assert.equal(copyPredicted(predicted, found, first.nmf, null).find((r) => r.nmf === 1).points.length, 8);

// what a state merged of another look says nothing of this one's candidates
const other = { ...second, nmf: { ...second.nmf, run: 'r0' } };
assert.equal(mergedOf(found, other), null);
assert.equal(provisional(found, other).length, 4);
assert.deepEqual(names(copyPredicted(predicted, found, other.nmf, null)), ['3@1', 'DM1@0', 'DM1@1', 'DM2@0']);
// (and its ROIs, whose names are not this look's, are left as they are when more is merged)
const mixed = merge(other, found, kept, [1], false);
assert.equal(mixed.rois.filter((r) => r.name === 'DM5').length, 2);
assert.deepEqual(mixed.nmf.merged, [1]);

console.log('nmf_merge.js: ok');
