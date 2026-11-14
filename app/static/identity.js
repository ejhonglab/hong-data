// What naming a predicted ROI as a glomerulus cost (al-diag-seg's identity.assign_identities, as
// /api/predictions/<id>/identity gives it), in a box opened by a double click on the ROI's name
// drawn on the movie player's image, and kept open until closed (its ×, Escape, or a click outside it): the terms of
// the cost, their weights, and each term's value for the glomeruli that cost least.

// the terms, in the order they are added; `value`: what each one's value is (0: a perfect fit)
export const TERMS = [
  { key: 'tuning', label: 'Odor tuning', value: "1 − the correlation of the ROI's odor tuning with the glomerulus's mean tuning in the training flies (can reach 2)" },
  { key: 'position', label: 'Position', value: "The distance from the ROI's centroid to where the glomerulus is expected (in the atlas registered to the fly, if the model has one), in SDs of the glomerulus's position, capped at 4, divided by 4" },
  { key: 'shape', label: 'Shape', value: "The distance of the ROI's log volume and log axis lengths from the glomerulus's, in SDs, capped at 4, divided by 4" },
  { key: 'atlas', label: 'Atlas overlap', value: "1 − the mean probability of the glomerulus, in the atlas registered to the fly, over the ROI's voxels" },
  { key: 'layout', label: 'Layout', value: "The mismatch between the ROI's offsets to its neighbours and the offsets between their glomeruli in the template, in SDs, capped at 4, divided by 4 (the mean over its named neighbours)" },
];

const num = (v) => (v == null ? '—' : (+v.toFixed(3)).toString());

function wire(dialog) {
  if (dialog.dataset.wired) return;
  dialog.dataset.wired = '1';
  dialog.querySelector('.roi-identity-close').addEventListener('click', () => dialog.close());
  // (a click outside the box is on the dialog itself: its backdrop)
  dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
}

// Shows `html` in the box, under `title` (plain text).
function show(dialog, title, html) {
  wire(dialog);
  dialog.querySelector('.roi-identity-title').textContent = title;
  dialog.querySelector('.roi-identity-body').innerHTML = html;
  if (!dialog.open) dialog.showModal();
}

// Why nothing can be shown of an ROI's name (`why`, HTML).
export function showNoIdentity(dialog, title, why) {
  show(dialog, title, `<p>${why}</p>`);
}

// The table of ROI `label`'s name: `terms` as the server gives them (with `rois`), `esc` the app's.
export function showIdentity(dialog, title, terms, label, esc) {
  const roi = terms.rois[label], w = terms.weights, max = terms.max_cost;
  const on = (t) => w[t.key] != null && w[t.key] !== 0;
  const best = roi.candidates[0], got = roi.candidates.find((c) => c.name === roi.assigned);
  let summary;
  if (got) {
    summary = `Named <b>${esc(got.name)}</b>, at a cost of ${num(got.cost)}: no more than the most a name may cost
      (${num(max)}, set when the model was fit).`;
  } else if (!best) summary = 'Not named: the model has no glomerulus to name it.';
  else if (best.cost > max) {
    summary = `Not named (numbered ${esc(label)}): its best match, <b>${esc(best.name)}</b>, costs ${num(best.cost)}, more than the
      most a name may cost (${num(max)}, set when the model was fit).`;
  } else {
    summary = `Not named (numbered ${esc(label)}), though ${roi.candidates.filter((c) => c.cost <= max).map((c) => `<b>${esc(c.name)}</b>`).join(', ')}
      cost${roi.candidates.filter((c) => c.cost <= max).length === 1 ? 's' : ''} no more than a name may (${num(max)}): each glomerulus
      names one ROI, and the names were given to the ROIs that, all together, cost least.`;
  }
  const head = TERMS.map((t) => `<th title="${esc(t.value)}">${esc(t.label)}</th>`).join('');
  const weights = TERMS.map((t) => (w[t.key] == null
    ? '<td class="muted" title="Off: the model has no atlas">—</td>'
    : `<td${on(t) ? '' : ' class="muted" title="Off (weight 0): not part of the cost"'}>${num(w[t.key])}</td>`)).join('');
  const cell = (c, t) => {
    const fixed = c.reference && (t.key === 'tuning' || t.key === 'shape');
    const tip = c[t.key] == null ? (w[t.key] ? 'The atlas has no map of this glomerulus: nothing is added to its cost' : 'Not computed: this term is off')
      : fixed ? `Fixed at ${num(terms.unknown_cost)}: this glomerulus is only in the reference atlas, so its tuning and shape are not known` : '';
    return `<td${on(t) ? '' : ' class="muted"'}${tip ? ` title="${esc(tip)}"` : ''}>${num(c[t.key])}${fixed ? '*' : ''}</td>`;
  };
  const rows = roi.candidates.map((c) => {
    const mine = c.name === roi.assigned;
    const note = mine ? ' <span class="roi-identity-mark">named this</span>'
      : c.taken_by != null ? ' <span class="muted" title="Each glomerulus names one ROI: this one names another of the fly\'s ROIs">(another ROI\'s)</span>' : '';
    return `<tr${mine ? ' class="roi-identity-got"' : ''}><th scope="row">${esc(c.name)}${note}</th>${TERMS.map((t) => cell(c, t)).join('')}
      <td${c.cost > max ? ` class="muted" title="More than the most a name may cost (${num(max)})"` : ''}>${num(c.cost)}</td></tr>`;
  }).join('');
  const sum = TERMS.filter(on).map((t) => `${num(w[t.key])} × ${t.label.toLowerCase()}`).join(' + ');
  show(dialog, title, `<p>${summary}</p>
    <table class="odor-table roi-identity-table"><thead>
      <tr><th scope="row">Term</th>${head}<th title="The weighted mean of the terms: (${esc(sum)}) / ${num(terms.weight_sum)}. A name is given only at ${num(max)} or less">Cost</th></tr>
      <tr class="roi-identity-weights"><th scope="row">Weight</th>${weights}<td class="muted" title="The sum of the weights, which the cost is divided by">/ ${num(terms.weight_sum)}</td></tr>
    </thead><tbody>${rows}</tbody></table>
    <p class="muted small">The ${Math.min(5, roi.candidates.length)} glomeruli that cost least for this ROI${roi.candidates.length > 5 ? ', and the one it was named' : ''},
      least first; each value is 0 for a perfect fit (hover over a term for what it is). The values are of the ROI as a 3D object: its
      outlines on all its planes.${roi.candidates.some((c) => c.reference) ? ' * Only in the reference atlas.' : ''}</p>`);
}
