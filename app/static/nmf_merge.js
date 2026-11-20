// What the movie player does with the candidates of a look for glomeruli a prediction's ROIs miss
// (nmf_check.py; `found`: as /api/predictions/<id>/nmf gives it): shows those not merged beside the
// ROIs, and merges those chosen into the edited ROIs (player.js: a state of its history), named as
// a plan says (/api/predictions/<id>/nmf/plan: { names: {label: name or null}, kinds }).
// An edited ROI is known by its name, which says its label (in the prediction's label volumes, or the
// look's): a glomerulus' name, or the label's number for an ROI left unnamed, as in the RoiSets. What a
// state has merged is its `nmf`: { run (the look, by when it finished), merged (candidates' ids),
// joint (whether all ROIs were named again together), names (of every ROI then, by label) }.

import { simplify } from './roi_edit.js';

// an ROI's name as RoiSets have it, by a naming ({label: name or null})
export const shownName = (names, label) => names[label] ?? String(label);

// whether a name is a glomerulus' to the database ('glomerulus', 'tentative' or 'other'), by `kinds`
// ({name: kind}, as the server says); numbers are of ROIs left unnamed
export const kindOf = (kinds, name) => kinds?.[name] ?? (/^\d+$/.test(name) ? 'other' : 'glomerulus');

// what a candidate is, in a few words
export const what = (c) => (c.kind === 'added' ? 'new' : `${c.kind === 'split' ? 'part' : 'more'} of ${c.of_name ?? c.of}`);

// a state's `nmf`, if it is of this look (else null: nothing of it is merged, as far as can be told)
export const mergedOf = (found, state) => (found && state?.nmf && state.nmf.run === found.run ? state.nmf : null);

// the ids of the candidates a state has merged
export const mergedIds = (found, state) => new Set(mergedOf(found, state)?.merged || []);

const outline = (points, tolerance) => (tolerance == null ? points.map((p) => [p[0], p[1]]) : simplify(points, tolerance));

// the labels of a naming's ROIs, by the names shown
function labelsByName(names) {
  const out = new Map();
  for (const label of Object.keys(names)) out.set(shownName(names, label), Number(label));
  return out;
}

// a candidate's outlines as edited ROIs, named by `names`: its own and, of a part cut off an ROI,
// what is left of that ROI on its plane (in place of the ROI's outline there)
function pushCandidate(rois, c, names, kinds, tolerance) {
  const add = (name, outlines) => {
    for (const o of outlines) {
      const points = outline(o.points, tolerance);
      if (points.length >= 3) rois.push({ name, z: o.z, kind: kindOf(kinds, name), points, nmf: c.id });
    }
  };
  add(shownName(names, c.label), c.outlines);
  if (c.kind === 'split') add(shownName(names, c.of), c.rest || []);
}

// The candidates a state has not merged, as ROIs to show beside the others: [{ name, z, kind, points,
// cand (its id), what, first (its first outline: where its tick box goes) }], named as the look named
// them (the predicted ROIs' names kept), their outlines simplified as the state's are.
export function provisional(found, state) {
  if (!found) return [];
  const merged = mergedIds(found, state), tolerance = state?.simplified ?? null;
  return found.candidates.filter((c) => !merged.has(c.id)).flatMap((c) => {
    const name = c.name ?? String(c.label);
    return c.outlines.map((o) => ({ name, z: o.z, kind: kindOf(found.kinds, name), points: outline(o.points, tolerance), cand: c.id, what: what(c) }))
      .filter((r) => r.points.length >= 3).map((r, k) => ({ ...r, first: k === 0 }));
  });
}

// A copy of the predicted ROIs (`predicted`: [{ name, z, kind, paths }]) to edit, each outline
// simplified to within `tolerance` pixels (null: as it is), with what `nmf` (a state's, of this look;
// null: nothing) has merged in again: the predicted ROIs named as it names them, an ROI a part was cut
// off replaced on that plane by what is left of it, and the candidates' own outlines.
export function copyPredicted(predicted, found, nmf, tolerance) {
  const on = nmf && found && nmf.run === found.run ? nmf : null;
  const byName = on ? labelsByName(found.predicted) : null;
  const merged = on ? found.candidates.filter((c) => on.merged.includes(c.id)) : [];
  const cut = new Set(merged.filter((c) => c.kind === 'split').flatMap((c) => c.planes.map((z) => `${c.of}|${z}`)));
  const rois = [];
  for (const r of predicted) {
    const label = byName?.get(r.name);
    if (label != null && cut.has(`${label}|${r.z}`)) continue;
    const name = label != null ? shownName(on.names, label) : r.name;
    for (const path of r.paths) {
      const points = outline(path, tolerance);
      if (points.length >= 3) rois.push({ name, z: r.z, kind: label != null ? kindOf(found.kinds, name) : r.kind, points });
    }
  }
  for (const c of merged) pushCandidate(rois, c, on.names, found.kinds, tolerance);
  return rois;
}

// A state with the candidates `chosen` (ids: those it has merged already, and more) merged in, named
// by `plan`: its ROIs renamed (each by its label, which its name before says), the outline of an ROI a
// part is cut off replaced by what is left of it, and the new candidates' outlines added, simplified
// as the state's are. Outlines edited by hand stay as they are, but for an ROI's a part is cut off; ROIs
// marked `unlinked` (drawn, renamed or numbered again: their names no longer say their labels) as they are.
export function merge(state, found, plan, chosen, joint) {
  const before = mergedOf(found, state), had = new Set(before?.merged || []);
  const byName = labelsByName(before ? before.names : found.predicted);
  const more = found.candidates.filter((c) => chosen.includes(c.id) && !had.has(c.id));
  const cut = new Set(more.filter((c) => c.kind === 'split').flatMap((c) => c.planes.map((z) => `${c.of}|${z}`)));
  const kinds = { ...found.kinds, ...plan.kinds };
  const rois = [];
  for (const r of state.rois) {
    const label = r.unlinked ? null : byName.get(r.name); // (drawn, renamed or numbered again in the viewer: its name kept)
    if (label == null) { rois.push(r); continue; }
    if (r.nmf == null && cut.has(`${label}|${r.z}`)) continue;
    const name = shownName(plan.names, label);
    rois.push(name === r.name ? r : { ...r, name, kind: kindOf(kinds, name) });
  }
  for (const c of more) pushCandidate(rois, c, plan.names, kinds, state.simplified ?? null);
  return { ...state, rois, nmf: { run: found.run, merged: [...had, ...more.map((c) => c.id)].sort((a, b) => a - b), joint, names: plan.names } };
}

// What merging the candidates `chosen` into a state changes, to show before it is done: `added` (the
// candidates not merged yet: { id, kind, name (the one it gets), named (whether it gets one), planes,
// of (the ROI it is more of, or cut from, as named before) }), `renamed` (ROIs whose names change:
// { label, from, to, nmf (whether it is one merged from the look before, not a predicted one) }) and
// `kept` (how many ROIs keep their names).
export function changes(found, state, plan, chosen) {
  const before = mergedOf(found, state), had = new Set(before?.merged || []);
  const prev = before ? before.names : found.predicted;
  const own = new Set(found.candidates.filter((c) => c.kind !== 'extended').map((c) => c.label));
  const added = found.candidates.filter((c) => chosen.includes(c.id) && !had.has(c.id)).map((c) => ({
    id: c.id, kind: c.kind, name: shownName(plan.names, c.label), named: plan.names[c.label] != null, planes: c.planes,
    of: c.of == null ? null : shownName(prev, c.of) }));
  const both = Object.keys(prev).filter((l) => l in plan.names);
  const renamed = both.filter((l) => (prev[l] ?? null) !== (plan.names[l] ?? null))
    .map((l) => ({ label: Number(l), from: shownName(prev, l), to: shownName(plan.names, l), nmf: own.has(Number(l)) }));
  return { added, renamed, kept: both.length - renamed.length };
}
