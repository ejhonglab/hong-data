// The AL segmentation tab's box of how a fit's model did on each glomerulus (by_glomerulus.py): over the flies of one
// experimenter that the model predicted and that have ROIs of their own, per name the experimenter gives glomeruli, in
// how many flies an ROI was found on it and what that ROI was named, and where else the model gave the name. The
// predictions' ROIs are matched to the flies' own by the fit worker, the first time they are summed (`_load`).

const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' }).compare;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const pct = (n, of) => (of ? `${Math.round((100 * n) / of)}%` : '—');
const count = (n) => n.toLocaleString('en-US');
const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);
// 'DC1 3, VA2 1' of { name: flies }, most first (the first `most` of them, then how many more names)
const tally = (o, esc, most = Infinity) => {
  const all = Object.entries(o).sort(([a, n], [b, m]) => m - n || natural(a, b));
  return all.slice(0, most).map(([g, n]) => `${esc(g)} ${n}`).join(', ') + (all.length > most ? `, +${all.length - most}` : '');
};
// a fly's name without whose it is: '2024-05-24/1'
const shortFly = (fly) => fly.replace(/^.*_(\d{4}-\d{2}-\d{2})_(\d+)$/, '$1/$2');

// The parts of the two bars, in the order drawn: [class, key of a glomerulus' count (or how to get it), what it is]
const DRAWN = [
  ['right', (r) => r.named, 'named right'],
  ['unnamed', (r) => r.unnamed, 'an ROI on it, not named'],
  ['none', (r) => r.none, 'no ROI on it'],
  ['wrong', (r) => sum(r.other), 'named as another'],
];
const GIVEN = [
  ['right', (r) => r.given_right, "on the fly's own of that name"],
  ['right weak', (r) => r.given_weak + r.given_stacked, 'barely on it, or over or under it'],
  ['unnamed', (r) => r.given_undrawn, 'in a fly where it is not drawn'],
  ['wrong weak', (r) => r.given_elsewhere, 'elsewhere, though the fly has it drawn'],
  ['wrong', (r) => sum(r.given_wrong), 'on another glomerulus'],
];
// what a fly is listed under, when a glomerulus' flies are shown: [state, heading, whether its ROI's name is said]
const STATES = [
  ['other', 'Named as another', true], ['unnamed', 'An ROI on it, not named', false], ['none', 'No ROI on it', true], ['named', 'Named right', false],
  ['wrong', 'The model gave its name to an ROI on another glomerulus', true], ['elsewhere', 'The model gave its name elsewhere, though the fly has it drawn', true],
  ['weak', 'The model gave its name to an ROI barely on it', false],
  ['stacked', 'The model gave its name to an ROI over or under it, on planes where it is not drawn', true],
  ['undrawn', 'The model gave its name in a fly where it is not drawn', true],
];

export class ByGlomerulusPanel {
  // helpers: { $, esc, api, msgBox, naming (the naming explorer: naming.js) }
  constructor(h) {
    this.h = h;
    this.data = null;
    this.seq = 0;
    this.asked = new Set(); // predictions the fit worker was asked to match, since the box was last loaded
    this.note = null; // what went wrong matching them
    this.open = new Set(); // the glomeruli whose flies are shown
    this.sort = { key: 'name', dir: 1 };
    this.signature = null;
    const { $ } = h;
    for (const id of ['#byglom-fit', '#byglom-who', '#byglom-flies', '#byglom-rois']) {
      $(id).addEventListener('change', () => { if (id === '#byglom-fit') $('#byglom-who').value = ''; this.open.clear(); this.load(); });
    }
    $('#byglom-model-only').addEventListener('change', () => this._renderTable());
    const box = $('#byglom-table');
    box.addEventListener('click', (e) => {
      const th = e.target.closest('th[data-sort]'), fly = e.target.closest('[data-p]'), row = e.target.closest('tr[data-glom]');
      if (th) {
        const key = th.dataset.sort;
        this.sort = { key, dir: this.sort.key === key ? -this.sort.dir : key === 'name' ? 1 : -1 };
        this._renderTable();
      } else if (fly) {
        const p = this.data.predictions[fly.dataset.p];
        h.naming().open({ id: +fly.dataset.p, fit_id: this.data.fit, fly: p.fly }, p.explorer && fly.dataset.roi ? +fly.dataset.roi : null);
      } else if (row) {
        const g = row.dataset.glom;
        if (this.open.has(g)) this.open.delete(g); else this.open.add(g);
        this._renderTable();
      }
    });
    // what a part of a bar is, by the pointer
    this.tip = document.createElement('div');
    this.tip.className = 'naming-tip';
    this.tip.hidden = true;
    document.body.append(this.tip);
    box.addEventListener('mousemove', (e) => {
      const t = e.target.closest?.('[data-tip]');
      this.tip.hidden = !t;
      if (!t) return;
      this.tip.textContent = t.dataset.tip;
      const w = this.tip.offsetWidth, ht = this.tip.offsetHeight;
      this.tip.style.left = `${clamp(e.clientX + 14, 4, window.innerWidth - w - 4)}px`;
      this.tip.style.top = `${e.clientY + 18 + ht > window.innerHeight ? e.clientY - ht - 10 : e.clientY + 18}px`;
    });
    box.addEventListener('mouseleave', () => { this.tip.hidden = true; });
  }

  // when the tab is shown, or something is chosen
  load() {
    this.asked = new Set();
    this.note = null;
    return this._load();
  }

  // The tab's predictions changed (`preds`: as its list has them): summed again if those done of flies with ROIs of their own
  // are others than they were, or refined otherwise.
  predictionsChanged(preds) {
    const sig = (preds || []).filter((p) => p.status === 'done' && p.result?.scores).map((p) => `${p.id}:${p.result.final}:${p.finished_at}`).join(' ');
    if (this.signature != null && sig !== this.signature && !this.h.$('#tab-al-segmentation').hidden) this.load();
    this.signature = sig;
  }

  // The summary as chosen; then, while it has predictions whose ROIs are not matched to their flies' own, the fit worker matches
  // those (as many at a time as it gets to in a while) and it is read again.
  async _load() {
    const { $, api, esc } = this.h, seq = ++this.seq;
    const q = new URLSearchParams({ flies: $('#byglom-flies').value || 'new', rois: $('#byglom-rois').value || 'final' });
    if ($('#byglom-fit').value) q.set('fit', $('#byglom-fit').value);
    if ($('#byglom-who').value) q.set('who', $('#byglom-who').value);
    let d;
    try { d = await api(`/api/fits/by_glomerulus?${q}`); } catch (e) {
      if (seq === this.seq) $('#byglom-status').replaceChildren(this.h.msgBox('error', `Could not sum the predictions by glomerulus: ${esc(e.message)}`));
      return;
    }
    if (seq !== this.seq) return;
    if (d.fit == null && d.choices.length && q.has('fit')) { // (the fit chosen has no such predictions any more: the newest that has)
      $('#byglom-fit').replaceChildren();
      $('#byglom-who').replaceChildren();
      return this._load();
    }
    this.data = d;
    this.matching = false;
    const todo = (d.counts?.missing || []).filter((i) => !this.asked.has(i));
    if (todo.length && d.worker && !this.note) this.matching = true;
    this.render();
    if (!this.matching) return;
    let r;
    try { r = await api('/api/fits/by_glomerulus', { method: 'POST', json: { predictions: todo } }); } catch (e) {
      if (seq !== this.seq) return;
      this.note = e.message;
      this.matching = false;
      this.render();
      return;
    }
    if (seq !== this.seq) return;
    for (const i of r.made) this.asked.add(i);
    if (!r.made.length) for (const i of todo) this.asked.add(i); // (none of them can be: not asked again)
    await this._load();
  }

  render() {
    const { $, esc } = this.h, d = this.data;
    $('#byglom-step').hidden = !d?.choices.length;
    if (!d?.choices.length) return;
    const fit = d.choices.find((c) => c.id === d.fit);
    const opt = (v, text, sel) => `<option value="${esc(String(v))}"${sel ? ' selected' : ''}>${text}</option>`;
    $('#byglom-fit').innerHTML = d.choices.map((c) => opt(c.id, `Fit #${c.id}${c.by ? `, ${esc(c.by)}'s` : ' (deleted)'}`, c.id === d.fit)).join('');
    $('#byglom-who').innerHTML = fit.experimenters.map((e) => opt(e.name, `${esc(e.name)} (${e.scored} ${e.scored === 1 ? 'fly' : 'flies'} with ROIs)`, e.name === d.experimenter)).join('');
    const c = d.counts;
    $('#byglom-flies').innerHTML = opt('new', `Those it was not trained on (${c.scored - c.trained})`, d.flies === 'new')
      + opt('all', `All it predicted (${c.scored})`, d.flies === 'all');
    $('#byglom-rois').value = d.rois;
    this._renderStatus();
    this._renderSummary();
    this._renderTable();
  }

  _renderStatus() {
    const { $, esc, msgBox } = this.h, d = this.data, c = d.counts, out = [];
    const left = c.missing.length;
    if (this.matching) {
      out.push(msgBox('info', `Matching each prediction's ROIs to its fly's own: ${c.ready} of ${c.chosen} flies done… <span class="muted">(once: they are kept with the predictions)</span>`));
    } else if (left) {
      out.push(msgBox('warn', `${left} of the ${c.chosen} flies are not in the sums: their predictions' ROIs are not matched to their own yet, which the fit worker does${
        this.note ? ` (${esc(this.note)})` : d.worker ? ', and it could not' : ', and it is not running'}.`));
    }
    if (d.failed.length) {
      const one = d.failed.length === 1, why = (f) => `${esc(shortFly(f.fly))} <span class="muted">(${esc(f.why)})</span>`, rest = d.failed.slice(2);
      out.push(msgBox('warn', `${d.failed.length} ${one ? 'fly is' : 'flies are'} left out: ${one ? 'its' : 'their'} ROIs cannot be matched to ${one ? 'its' : 'their'} own.
        ${d.failed.slice(0, 2).map(why).join('; ')}${rest.length ? `<details class="msg-fold"><summary>The other ${rest.length}</summary><ul>${
        rest.map((f) => `<li>${why(f)}</li>`).join('')}</ul></details>` : ''}`));
    }
    $('#byglom-status').replaceChildren(...out);
  }

  _renderSummary() {
    const { $, esc } = this.h, d = this.data, c = d.counts, t = d.total, who = esc(d.experimenter);
    const kinds = { pred: "the model's own", split: 'those with outlines over two glomeruli split', clean: 'those cleaned of their neighbors' };
    const of = Object.entries(c.kinds).map(([k, n]) => `${kinds[k] || esc(k)}${Object.keys(c.kinds).length > 1 ? ` (${n})` : ''}`).join(', ');
    const fit = d.choices.find((x) => x.id === d.fit), e = fit.experimenters.find((x) => x.name === d.experimenter);
    const parts = [`Fit #${d.fit}'s model predicted ${e.predicted} of ${who}'s flies, <b>${c.scored}</b> of them with ROIs of their own${
      c.held_out ? ` (${c.held_out} held out of the fit)` : ''}${c.trained && d.flies === 'new' ? `; the ${c.trained} it was trained on are left out` : ''}.${
      c.ready < c.scored ? ` Summed here: <b>${c.ready}</b> of them.` : ''} Their ROIs: ${of || '—'}.${
      c.changed ? ` ${c.changed} had other ROIs of their own when they were predicted: all are matched to those they have now.` : ''}`];
    const n = (v, total, label) => `${label} <b>${count(v)}</b> <span class="muted">${pct(v, total)}</span>`;
    if (t.drawn) {
      parts.push(`${who} drew <b>${count(t.drawn)}</b> glomeruli in them: ${[n(t.found, t.drawn, 'found'), n(t.named, t.drawn, 'named right'),
        n(t.other, t.drawn, 'named as another'), n(t.unnamed, t.drawn, 'an ROI on it, not named'), n(t.none, t.drawn, 'no ROI on it')].join(' · ')}.${
        t.moved ? ` For ${count(t.moved)} of those not named right, the model gave the name to another ROI of the fly.` : ''}`);
    }
    if (t.given) {
      parts.push(`The model gave these names to <b>${count(t.given)}</b> ROIs: ${[n(t.given_right, t.given, "on the fly's own of that name"), n(t.given_weak, t.given, 'barely on it'),
        n(t.given_stacked, t.given, 'over or under it'), n(t.given_undrawn, t.given, 'in a fly where it is not drawn'),
        n(t.given_elsewhere, t.given, 'elsewhere, though the fly has it drawn'), n(t.given_wrong, t.given, 'on another glomerulus')].join(' · ')}.`);
    }
    $('#byglom-summary').innerHTML = c.ready ? `<ul class="small byglom-sums">${parts.map((p) => `<li>${p}</li>`).join('')}</ul>` : '';
    const key = (parts_, what) => `<span class="byglom-key-of">${what}:</span> ${parts_.map(([cls, , label]) => `<span><i class="byglom-sw ${cls}"></i>${label}</span>`).join('')}`;
    $('#byglom-key').innerHTML = c.ready ? `<div>${key(DRAWN, 'Of the flies it is drawn in')}</div><div>${key(GIVEN, 'Of the ROIs the model named it')}</div>` : '';
    $('#byglom-only').hidden = !d.only_model.length;
    $('#byglom-only-n').textContent = d.only_model.length ? `${d.only_model.length} ${d.only_model.length === 1 ? 'name' : 'names'} the model gave that ${d.experimenter} never did` : '';
  }

  // a bar of the parts of `parts` with something in them, each as wide as its share of `of`
  _bar(r, parts, of, what) {
    if (!of) return '';
    const segs = parts.map(([cls, get, label]) => [cls, get(r), label]).filter(([, n]) => n);
    return `<span class="byglom-bar" role="img" aria-label="${this.h.esc(segs.map(([, n, label]) => `${label}: ${n}`).join('; '))}">${segs.map(([cls, n, label]) => `<i class="${cls}" style="flex:${n}"
      data-tip="${this.h.esc(`${r.name}, ${what}: ${label} in ${n} of ${of} (${pct(n, of)})`)}"></i>`).join('')}</span>`;
  }

  _renderTable() {
    const { $, esc } = this.h, d = this.data, box = $('#byglom-table');
    if (!d?.counts?.ready) { box.replaceChildren(); return; }
    const who = esc(d.experimenter), n = d.counts.ready, { key, dir } = this.sort;
    const val = {
      name: null, drawn: (r) => r.drawn, found: (r) => (r.drawn ? r.found / r.drawn : null), dice: (r) => r.dice,
      named: (r) => (r.drawn ? r.named / r.drawn : null), other: (r) => (r.drawn ? sum(r.other) / r.drawn : null), given: (r) => r.given,
      right: (r) => (r.given ? r.given_right / r.given : null), model: (r) => (r.model ? (r.model.trained ?? (r.model.atlas ? 0 : -1)) : null),
    }[key];
    const sorted = (rows) => [...rows].sort((a, b) => (val ? (val(a) == null) - (val(b) == null) || dir * (val(a) - val(b)) || natural(a.name, b.name) : dir * natural(a.name, b.name)));
    const th = (k, label, tip, cls = '') => `<th data-sort="${k}" class="${cls}" title="${esc(tip)}"${key === k ? ` aria-sort="${dir > 0 ? 'ascending' : 'descending'}"` : ''}>${label}${
      key === k ? (dir > 0 ? ' ▴' : ' ▾') : ''}</th>`;
    const model = (r) => (!r.model ? '—' : r.model.trained != null ? `<span title="Training flies with it: the model knows its odor tuning, shape and place from them">${r.model.trained}</span>`
      : r.model.atlas ? '<span class="muted" title="Only in the reference atlas (Grabe et al. 2016): no training fly has it, so the model knows its place but not its tuning or shape">ref</span>'
        : '<span class="muted" title="The model does not have it: it can never give this name">not in it</span>');
    const row = (r, cls = '') => {
      const open = this.open.has(r.name), other = sum(r.other), off = r.given - r.given_right;
      return `<tr data-glom="${esc(r.name)}" class="${cls}${open ? ' open' : ''}" aria-expanded="${open}"><th scope="row">${esc(r.name)}</th>
        <td>${r.drawn || '<span class="muted">0</span>'}</td>
        <td>${r.drawn ? `${r.found} <span class="muted">${pct(r.found, r.drawn)}</span>` : '—'}</td><td>${r.dice == null ? '—' : r.dice.toFixed(2)}</td>
        <td class="byglom-barcell">${this._bar(r, DRAWN, r.drawn, `of the ${r.drawn} flies it is drawn in`) || `<span class="muted">${r.site ? 'not drawn in these flies' : `never drawn by ${who}`}</span>`}</td>
        <td>${r.drawn ? `${r.named} <span class="muted">${pct(r.named, r.drawn)}</span>` : '—'}</td>
        <td class="byglom-names">${other ? `${other} <span class="muted">${tally(r.other, esc, 2)}</span>` : r.drawn ? '<span class="muted">0</span>' : '—'}</td>
        <td class="byglom-barcell">${this._bar(r, GIVEN, r.given, `of the ${r.given} ROIs the model named it`) || '<span class="muted">never given</span>'}</td>
        <td>${r.given ? `${r.given} <span class="muted">${off ? `${off} off` : 'all on it'}</span>` : '<span class="muted">0</span>'}</td>
        <td>${model(r)}</td></tr>${open ? `<tr class="byglom-detail"><td colspan="10">${this._detail(r)}</td></tr>` : ''}`;
    };
    const t = d.total, all = { name: 'All', ...t, other: { '': t.other }, given_wrong: { '': t.given_wrong } };
    const extra = $('#byglom-model-only').checked ? d.only_model : [];
    box.innerHTML = `<table class="odor-table byglom-table"><thead><tr>
      ${th('name', 'Glomerulus', `Each name ${d.experimenter} gave an ROI in these flies. Click one for its flies.`)}
      ${th('drawn', 'Drawn', `The flies ${d.experimenter} drew it in, of the ${n}`)}
      ${th('found', 'Found', `Of those, the flies with a predicted ROI on it at an IoU of ${d.detect_iou} or over (as 3D objects; where it is not drawn on a plane, an ROI's part there counts against it only inside another drawn glomerulus)`)}
      ${th('dice', 'Dice', 'The mean Dice of those found (2 IoU / (1 + IoU))')}
      <th class="byglom-barcell" title="Each fly it is drawn in, by the predicted ROI overlapping it most">Of the flies it is drawn in</th>
      ${th('named', 'Named right', `The flies where the predicted ROI overlapping it most (IoU over ${d.name_iou}) has its name`)}
      ${th('other', 'As another', 'The flies where that ROI has another name (the commonest)', 'byglom-names')}
      <th class="byglom-barcell" title="Each ROI the model gave this name, by the fly's own ROI it overlaps most">Of the ROIs the model named it</th>
      ${th('given', 'Named it', `The ROIs the model gave this name (at most one a fly), and how many of them are not on the fly's own of that name (IoU over ${d.name_iou})`)}
      ${th('model', 'In model', "How many of the model's training flies have it")}</tr></thead>
      <tbody>${sorted(d.glomeruli).map((r) => row(r)).join('')}${extra.length ? `<tr class="group"><td colspan="10">Names the model gave that ${who} never did</td></tr>
        ${sorted(extra).map((r) => row(r)).join('')}` : ''}</tbody>
      <tfoot><tr><th scope="row">All ${d.glomeruli.length}</th><td>${count(t.drawn)}</td><td>${count(t.found)} <span class="muted">${pct(t.found, t.drawn)}</span></td><td></td>
        <td class="byglom-barcell">${this._bar(all, DRAWN, t.drawn, `of the ${count(t.drawn)} glomeruli drawn`)}</td><td>${count(t.named)} <span class="muted">${pct(t.named, t.drawn)}</span></td>
        <td class="byglom-names">${count(t.other)}</td><td class="byglom-barcell">${this._bar(all, GIVEN, t.given, `of the ${count(t.given)} ROIs given these names`)}</td>
        <td>${count(t.given)} <span class="muted">${count(t.given - t.given_right)} off</span></td><td></td></tr></tfoot></table>`;
  }

  // a glomerulus' flies, by what became of it in each: each a button that opens the fly's naming at the ROI
  _detail(r) {
    const { esc } = this.h, d = this.data, iou = (v) => (v == null ? '' : `IoU ${v.toFixed(2)}`);
    const fly = (f, named) => {
      const p = d.predictions[f.p], on = f.on !== undefined;
      // of an ROI the model named it: the fly's own ROI it is on; of one drawn: the name of the ROI on it
      const what = on ? (f.state === 'wrong' ? `on ${esc(f.on)}` : f.state === 'weak' ? '' : f.on ? `barely on ${esc(f.on)}` : 'on none drawn')
        : named ? (f.roi == null ? 'none touches it' : f.as ? esc(f.as) : 'unnamed') : '';
      // (of one drawn and not named right: the ROI the model gave its name instead)
      const moved = !on && f.has != null && f.has !== f.roi ? `its name is on ROI ${f.has}` : '';
      const about = [what, f.roi == null || (on && !f.on) ? '' : iou(f.iou), moved].filter(Boolean).join(', ');
      return `<button type="button" class="byglom-fly" data-p="${f.p}"${f.roi == null ? '' : ` data-roi="${f.roi}"`} title="${esc(p.fly)}${f.roi == null ? '' : `, ROI ${f.roi}`}: how its ROIs were named${
        p.explorer || f.roi == null ? '' : ' (its naming shows other ROIs than these: the ROI is not marked)'}">${esc(shortFly(p.fly))}${about ? ` <span class="muted">${about}</span>` : ''}</button>`;
    };
    const lists = STATES.map(([state, head, named]) => {
      const flies = r.flies.filter((f) => f.state === state);
      return flies.length ? `<div class="byglom-flies"><span class="byglom-flies-head">${head} <span class="muted">${flies.length}</span></span>${flies.map((f) => fly(f, named)).join('')}</div>` : '';
    }).join('');
    const m = r.model, knows = !m ? '' : m.trained != null ? `${m.trained} of the model's training flies have it.`
      : m.atlas ? 'No training fly has it: the model has it from the reference atlas (Grabe et al. 2016) alone.' : 'The model does not have this name, so it can never give it.';
    const drawn = r.drawn ? `Drawn in ${r.drawn} of the ${d.counts.ready} flies: found (IoU ≥ ${d.detect_iou}) in ${r.found}, named right in <b>${r.named}</b>.${
      r.none ? ` Where no ROI is on it (none overlaps it by an IoU over ${d.name_iou}), ${r.untouched ? `none touches it at all in ${r.untouched}` : 'one still touches it'}.` : ''}${
      r.moved ? ` In ${r.moved} of the ${r.drawn - r.named} not named right, the model gave its name to another ROI.` : ''}`
      : r.site ? `Not drawn in these flies: ${esc(d.experimenter)} drew it in ${r.site} of the flies of the site's datasets.` : `${esc(d.experimenter)} never drew it.`;
    const given = r.given ? `The model named an ROI ${esc(r.name)} in ${r.given} flies, in <b>${r.given_right}</b> on the fly's own ${esc(r.name)}.` : 'The model never gave this name in these flies.';
    return `<p class="small">${drawn} ${given} ${knows}</p>${lists}`;
  }
}
