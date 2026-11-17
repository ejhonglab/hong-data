// Predictions of a fly's glomerulus ROIs by a fit's model, in the AL segmentation tab
// (predictions.py, fit_worker.run_prediction): asking for one (a model, and a fly of the site's
// datasets or any on tensor), the one of the model and fly chosen, every one made, and viewing
// one in the movie player (player.js: the ROIs predicted and the fly's own, of the plane shown), and
// refining one's ROIs (refinement.py: split, trimmed, kept apart), with their scores before and after.
// Also fills in the example, under Fit a model, of running a downloaded model yourself.

import { both, BRACKETS_NOTE, fmtDuration, LEAK_TIP, manualValue, PURITY_COLUMNS, scoringNote } from './fits.js';
import { listLink, tip } from './tips.js';

const STATUS_NAMES = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', interrupted: 'Interrupted' };
const STAGE_NAMES = { load: 'loading the fly', predict: 'predicting its ROIs' };
const OTHER = 'other'; // the fly menu's entry for a fly's folders entered by hand
const ACTIVE = new Set(['queued', 'running']);
const num2 = (v) => (v == null ? '—' : v.toFixed(2));
const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);

// what the scores of predicted ROIs against the fly's own mean (al-diag-seg's Model.evaluate)
const SCORE_TIPS = {
  found: `Of the fly's own ROIs, the share the predicted ones match. Each of the fly's ROIs is paired with at most one predicted ROI (the pairing that overlaps most overall), and a pair matches if they overlap by an IoU of at least 0.5: the area they share over the area of the two together.

Per plane, each glomerulus's outline on each plane is an ROI, as ROIs are drawn; as 3D objects, its outlines on all its planes are one.`,
  right: `Of the predicted ROIs, the share that match one of the fly's own: the same one-to-one pairing as found (an IoU of at least 0.5), but out of the predicted ROIs instead of the fly's. An ROI is right by its outline alone, whatever it is named.

So a low found means glomeruli missed; a low right, ROIs where the fly has none, or outlines too far off. (Unless scored strictly, predicted ROIs lying mostly outside all of the fly's, where a glomerulus may be but is not drawn, are left out.)`,
  dice: `How closely the outlines of the matched (found) pairs agree, on average: twice the area they share over the sum of their two areas, 1 if they are the same and 0 if they do not touch.`,
  named: `Of the predicted ROIs that overlap one of the fly's own enough, those given its name. Each predicted ROI is compared with the one of the fly's ROIs it overlaps most (by IoU, as 3D objects), and counts if that IoU is over 0.3. E.g. 12/20: 20 predicted ROIs overlap one of the fly's by that much, and 12 of them are named as it is (one left unnamed counts as wrong).

Unlike right, this is only about names: the outline need not match as well (IoU over 0.3, not 0.5), and the pairing is not one to one.`,
};
const scoreTip = (key, label) => tip(label, SCORE_TIPS[key]);
// a refinement running: what it is doing (refinement.STAGES)
const REFINE_STAGES = { load: 'loading the fly', register: 'registering the atlas to it', refine: 'refining', score: 'scoring' };
// the look for glomeruli a prediction's ROIs miss (nmf_check.py), if one was asked for: how it is, in a sentence
function nmfNote(p, esc, fmtWhen) {
  const n = p.nmf;
  if (!n) return '';
  const by = n.name ? `, asked for by ${esc(n.name)}` : '';
  if (n.status === 'queued') return ` Glomeruli its ROIs miss are to be looked for in its movies (waiting for the fit worker${by}).`;
  if (n.status === 'running') return ` Glomeruli its ROIs miss are being looked for in its movies (${Math.round((n.fraction || 0) * 100)}%${by}).`;
  if (n.status === 'done') {
    const k = n.result?.candidates ?? 0, merged = p.edited?.nmf?.merged.length || 0;
    return ` Glomeruli its ROIs miss were looked for in its movies ${esc(fmtWhen(n.finished_at))}${by}: ${k ? `${k} found${merged ? `, ${merged} of them merged into its edited ROIs`
      : ''} (view the ROIs to see them, and merge them or not)` : 'none found'}.`;
  }
  return ` Looking for glomeruli its ROIs miss ${n.status === 'failed' ? 'failed' : 'was interrupted'}${n.error ? ` (${esc(n.error)})` : ''}: view the ROIs to ask again.`;
}
// the odors of its model that a prediction's fly lacks, if any, and what that means for its ROIs
function missingNote(r, esc) {
  const m = r.missing_odors || [];
  if (!m.length) return '';
  const img = (r.missing_image_odors || []).length;
  return ` Its fly lacks ${m.length} of the model's ${r.model_odors} odors (${m.map(esc).join(', ')}): its ROIs are joined into 3D and named by those it has${
    img ? `, and found from images of ${img} fewer odor${img === 1 ? '' : 's'} than Cellpose was trained on, so fewer may be found than in a fly with all of them`
      : ', exactly as by a model fit to the same flies without them (Cellpose\'s images are of none of them)'}.`;
}
// the steps a refinement was asked for (predictions.options), in the order they are done, with their settings
function stepsText(o) {
  return [...(o?.split ? ['outlines over two glomeruli split'] : []), ...(o?.clean || []).map((st) => {
    const ps = Object.entries(o.clean_params?.[st] || {}).map(([k, v]) => `${k} ${v}`).join(', ');
    return ps ? `${st} (${ps})` : st;
  })].join(', then ');
}
const plural = (n, what) => `${n} ${what}${n === 1 ? '' : 's'}`;
// each kind of a refined prediction's ROIs (the model's own, split, refined) by the scores the fit's held-out flies are shown
// with: how they match the fly's own ROIs, where it has some, and how well they keep from their neighbors
function refineTable(r, esc) {
  const kinds = r?.result?.kinds;
  if (!kinds) return '<p class="muted small">Its scores were not kept (refined before they were).</p>';
  const scored = kinds.pred?.recall != null, final = r.result.final;
  const label = (k) => (k === 'pred' ? "The model's own" : k === final ? 'Refined' : 'Split only');
  const hint = { pred: 'As the model predicted them', split: 'Outlines over two glomeruli cut in two, joined into 3D and named again',
    clean: 'After every step asked for' };
  // (of a fly without ROIs of its own, no manual values to put in brackets)
  const cell = (c, x) => (scored ? c[2](x) : c[2](x).replace(/ <span class="muted"[^>]*>\(—\)<\/span>/, ''));
  const head = `<th>ROIs</th>${scored ? ['Found', 'Right', 'Dice'].map((h) => `<th>${h}${kinds.pred.recall_2d == null ? ''
    : ' <span class="muted">(per plane · 3D)</span>'}</th>`).join('') : ''}<th title="${esc(LEAK_TIP)}">Leak</th>${PURITY_COLUMNS.map((c) => `<th
    title="${esc(c[1])}">${c[0]}</th>`).join('')}${scored ? '<th>Named</th>' : ''}`;
  const rows = ['pred', 'split', 'clean'].filter((k) => kinds[k]).map((k) => {
    const x = kinds[k];
    return `<tr${k === final ? ' class="refine-final"' : ''}><td title="${hint[k]}">${label(k)} <span class="muted" title="${x.n} ROIs (as 3D objects)">(${
      x.n})</span></td>${scored ? `<td>${both(x, 'recall')}</td><td>${both(x, 'precision')}</td><td>${both(x, 'dice', num2)}</td>` : ''}<td>${num2(x.leak)}${
      scored ? ` ${manualValue(x.manual_leak)}` : ''}</td>${PURITY_COLUMNS.map((c) => `<td>${cell(c, x)}</td>`).join('')}${scored
      ? `<td>${x.named ?? '—'}/${x.named_of ?? '—'}</td>` : ''}</tr>`;
  }).join('');
  return `${scored ? `<p class="muted small">${BRACKETS_NOTE}</p>` : ''}<table class="odor-table refine-table"><thead><tr>${head}</tr></thead><tbody>${
    rows}</tbody></table>${scored ? ''
    : '<p class="muted small">The fly has no ROIs of its own: only how the ROIs keep from their neighbors is measured.</p>'}`;
}
const trim = (s) => (s || '').trim().replace(/\/+$/, '');

// The rows of a list (`items`, newest first: `row(item)` each one's) grouped by the batch each is of (`batchOf(item)`: { id,
// label }, or null): a batch's rows after a row of its own, where its newest would be, which opens or folds them (`open`: the
// batches open; folded by default), saying how many there are of each status (`statusOf`); the others as they are
export function batchGroups(items, batchOf, row, open, cols, statusOf) {
  const seen = new Set(), out = [];
  for (const it of items) {
    const b = batchOf(it);
    if (!b) { out.push(row(it)); continue; }
    if (seen.has(b.id)) continue;
    seen.add(b.id);
    const members = items.filter((x) => batchOf(x)?.id === b.id), on = open.has(b.id), counts = {};
    for (const m of members) counts[statusOf(m)] = (counts[statusOf(m)] || 0) + 1;
    out.push(`<tr class="batch-head"><td colspan="${cols}"><button type="button" class="link" data-act="batch-toggle" data-batch-id="${b.id}"
      aria-expanded="${on}">${on ? '▾' : '▸'} ${b.label}</button> <span class="muted">(${members.length}: ${Object.entries(counts)
      .map(([k, n]) => `${n} ${k}`).join(', ')})</span></td></tr>`);
    out.push(...members.map((m) => row(m).replace(/<tr(?=[\s>])/g, `<tr data-in-batch="${b.id}"${on ? '' : ' hidden'}`)));
  }
  return out.join('');
}

export class PredictPanel {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, getName (the name in Settings, '' if none),
  // player (the movie player), flies (the tab's flies, as from /api/roi_training; null until found) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/predictions
    this.mine = { name: '', flies: [] }; // the flies in the folders of whoever is named in Settings (/api/user/flies)
    this.listed = []; // the flies of the menu
    this.timer = null;
    const { $ } = h;
    $('#predict-model').addEventListener('change', () => this.renderFlies());
    $('#predict-fly').addEventListener('change', () => { this.flyChosen = $('#predict-fly').value; this._renderPair(); });
    for (const id of ['#predict-dir', '#predict-raw']) $(id).addEventListener('input', () => this._renderPair());
    $('#predict-clean').addEventListener('change', () => {
      const o = this._cleanOptions(), on = [...(o.split ? ['split'] : []), ...o.clean];
      $('#predict-clean-summary').textContent = on.length ? `(${on.join(', ')})` : '';
    });
    this.shown = new Set(); // predictions whose refinement's scores are shown in the list
    // (the refine dialog's steps: the form's, as they are in it)
    const steps = $('#refine-steps');
    for (const n of $('#predict-clean').children) if (!['SUMMARY', 'P'].includes(n.tagName)) steps.append(n.cloneNode(true));
    steps.querySelectorAll('[id]').forEach((n) => n.removeAttribute('id'));
    steps.addEventListener('change', () => this._refineChecked());
    $('#predict-start').addEventListener('click', () => this.start());
    $('#predict-all').addEventListener('click', () => this.startAll());
    $('#predict-all-mine').addEventListener('change', () => this._renderBatch());
    $('#predict-batch').addEventListener('click', (e) => { if (e.target.closest('[data-act=forget-batch]')) this._forgetBatch(+e.target.closest('[data-batch]').dataset.batch); });
    this.batchSeen = new Map(); // batch id -> { over, open }: whether its flies' details are shown (`_renderBatch`)
    this.openBatches = new Set(); // batches whose predictions are shown in the list (`batchGroups`)
    $('#predict-batch').addEventListener('toggle', (e) => {
      const d = e.target.closest?.('.predict-batch-more'), id = +e.target.closest?.('[data-batch]')?.dataset.batch;
      if (d && this.batchSeen.has(id)) this.batchSeen.get(id).open = d.open;
    }, true);
    this.missing = new Map(); // `${fit}|${dir}` -> what the model's odors the fly lacks are (/api/predictions/missing), or a promise of it
    $('#predict-view').addEventListener('click', () => { const p = this._pair(); if (p) this.view(p); });
    $('#predict-list').addEventListener('click', (e) => this._action(e));
  }

  // while the tab is shown: every 3 s while one is queued or running, else every 30 s
  async load() {
    this._loadMine();
    await this.refresh();
    if (this.timer) return;
    let n = 0;
    this.timer = setInterval(() => {
      if (this.h.$('#tab-al-segmentation').hidden) return;
      n += 1;
      const busy = this.data?.predictions.some((p) => ACTIVE.has(p.status) || ACTIVE.has(p.nmf?.status) || ACTIVE.has(p.refine?.status))
        || (this.data?.batches || []).some((b) => b.extract?.status === 'running');
      if (busy || n % 10 === 0) this.refresh();
    }, 3000);
  }

  async refresh() {
    try { this.data = await this.h.api('/api/predictions'); } catch (e) {
      this.h.$('#predict-status').replaceChildren(this.h.msgBox('error', `Could not ask the server about predictions: ${this.h.esc(e.message)}`));
      return;
    }
    this.render();
  }

  render() {
    const { $, esc } = this.h;
    if (!this.data) return;
    const name = this.h.getName();
    $('#predict-who').innerHTML = name
      ? `Predictions say who asked for them: this one will be <b>${esc(name)}</b>'s (as set in <a href="#settings/name">Settings</a>).`
      : 'Enter your first name in <a href="#settings/name">Settings</a> first: predictions say who asked for them.';
    const sel = $('#predict-model'), had = sel.value, models = this.data.models;
    sel.innerHTML = models.length ? models.map((m) => `<option value="${m.id}">Fit #${m.id}: ${m.train.length + m.held_out.length} flies
      (${m.train.length} trained on, ${m.held_out.length} held out), ${esc(m.name)}'s, ${esc(this.h.fmtWhen(m.finished_at))}</option>`).join('')
      : '<option value="">No fit is done yet: fit a model below first</option>';
    sel.disabled = !models.length;
    // (the last fit done, unless another was chosen)
    if (models.some((m) => String(m.id) === had)) sel.value = had;
    this.renderFlies();
    this._renderList();
    this._renderHowto();
    this.h.onData?.(); // (what else lists the predictions: Extract signals' ROIs)
  }

  // the flies on tensor of whoever is named in Settings, in the folders saved there: asked for when the tab is shown
  async _loadMine() {
    const name = this.h.getName();
    let flies = [];
    try { if (name) flies = (await this.h.api(`/api/user/flies?name=${encodeURIComponent(name)}`)).flies; } catch { return; }
    if (name !== this.h.getName()) return;
    const changed = JSON.stringify(flies) !== JSON.stringify(this.mine.flies) || name !== this.mine.name;
    this.mine = { name, flies };
    if (changed && this.data) this.renderFlies();
    if (changed) this.h.onMine?.();
  }

  _model() { return this.data?.models.find((m) => String(m.id) === this.h.$('#predict-model').value) || null; }

  // a fly's name, as a fit's (fits.fly_names): its top folder on tensor, its date and number
  _name(f) { return `${f.dir.split('/')[0]}_${f.date}_${f.fly_num}`; }

  // the fly menu: the flies of the site's datasets (by uploader; with what the model chosen did
  // with each, and whether it has predicted it) and, with them, the others in the folders of whoever
  // is named in Settings (with whether each has ROIs), then any other on tensor. That person's
  // flies first; of each person's, those without ROIs, then those with ROIs the model was not fit
  // with, those held out of it, and those it was trained on.
  renderFlies() {
    const { $, esc } = this.h;
    const sel = $('#predict-fly'), flies = this.h.flies(), m = this._model();
    const same = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase();
    const preds = new Map((this.data?.predictions || []).filter((p) => m && p.fit_id === m.id).map((p) => [trim(p.tensor_dir), p.status]));
    const made = (f) => ({ done: ' · predicted', queued: ' · queued', running: ' · being predicted' }[preds.get(trim(f.dir))] || '');
    const groups = new Map();
    for (const f of flies || []) {
      const who = [...new Set(f.datasets.map((d) => d.uploader || '?'))].join(' + ');
      if (!groups.has(who)) groups.set(who, []);
      groups.get(who).push(f);
    }
    const known = new Set((flies || []).map((f) => trim(f.dir)));
    const more = this.mine.flies.filter((f) => !known.has(trim(f.dir)));
    if (more.length) {
      const who = [...groups.keys()].find((w) => same(w, this.mine.name)) || this.mine.name;
      groups.set(who, [...(groups.get(who) || []), ...more]);
    }
    this.listed = [...groups.values()].flat();
    const isMine = (who) => who.split(' + ').some((w) => same(w, this.mine.name || this.h.getName()));
    const rank = (f) => (m?.train.includes(this._name(f)) ? 3 : m?.held_out.includes(this._name(f)) ? 2 : f.datasets || f.rois ? 1 : 0);
    const role = (f) => [' · no ROIs', f.datasets ? '' : ' · has ROIs', ' · held out of it', ' · trained on'][rank(f)];
    const opts = [...groups].sort(([a], [b]) => isMine(b) - isMine(a) || a.localeCompare(b)).map(([who, fs]) => `<optgroup label="${esc(who)}'s flies">${fs
      .sort((a, b) => rank(a) - rank(b) || a.date.localeCompare(b.date) || a.fly_num - b.fly_num)
      .map((f) => `<option value="${esc(f.dir)}">${esc(f.date)} fly ${f.fly_num}${role(f)}${made(f)}</option>`).join('')}</optgroup>`);
    sel.innerHTML = `${opts.join('')}${flies ? '' : '<option value="" disabled>(the site\'s flies: looking on tensor…)</option>'}
      <option value="${OTHER}">Another fly on tensor: its folders…</option>`;
    const values = [...sel.options].filter((o) => !o.disabled).map((o) => o.value);
    // (the one chosen, else the first held out of the model, else the first)
    const first = (flies || []).find((f) => m?.held_out.includes(this._name(f)));
    sel.value = values.includes(this.flyChosen) ? this.flyChosen : first ? first.dir : values[0];
    this._renderPair();
  }

  // the fly chosen: {dir, raw_dir} (from the top of tensor, or as entered, if not on tensor), or null
  _fly() {
    const { $ } = this.h, v = $('#predict-fly').value;
    if (v === OTHER) {
      const top = trim(this.data?.tensor_root), rel = (s) => (top && s.startsWith(`${top}/`) ? s.slice(top.length + 1) : s);
      const dir = rel(trim($('#predict-dir').value));
      return dir ? { dir, raw_dir: rel(trim($('#predict-raw').value)) } : null;
    }
    const f = this.listed.find((x) => x.dir === v);
    return f ? { dir: f.dir, raw_dir: f.raw_dir } : null;
  }

  // the prediction of the model and fly chosen, if there is one (as far as can be told here: the
  // server knows a fly by its folder, however it was entered)
  _pair() {
    const m = this._model(), fly = this._fly();
    if (!m || !fly || !this.data) return null;
    return this.data.predictions.find((p) => p.fit_id === m.id && trim(p.tensor_dir) === fly.dir) || null;
  }

  // what an estimate of a prediction's time is made from (predictions.estimate)
  _from(e) {
    return e.from ? `estimated from how long the last ${e.from === 1 ? 'one' : e.from} like it took on the ${e.device}`
      : `a rough guess: none has finished on the ${e.device} yet to estimate from`;
  }

  // when a prediction queued or running should end (its `eta`)
  _eta(p) {
    const { esc, fmtWhen } = this.h, e = p.eta;
    if (!e) return '';
    if (p.status === 'running' && !e.seconds) return ` It should end any moment: it has taken longer than expected (${this._from(e)}).`;
    return ` It should end about ${esc(fmtWhen(e.ends_at))} (in ${fmtDuration(e.until)}${p.status === 'queued' && e.until - e.seconds > 1
      ? `, after those before it: it takes about ${fmtDuration(e.seconds)}` : ''}; ${this._from(e)}).`;
  }

  // a prediction's ROIs as edited in the viewer: when they were last saved, and by whom
  _edited(p) {
    const e = p.edited;
    return `last ${this.h.fmtWhen(e.saved_at)}${e.by ? ` by ${e.by}` : ''}`;
  }

  // what is known of a prediction: when, by whom, on what, its ROIs, and their scores against the fly's own
  _about(p) {
    const { esc, fmtWhen } = this.h, r = p.result;
    if (p.status === 'queued') {
      return `Queued, asked for by <b>${esc(p.name)}</b> ${esc(fmtWhen(p.created_at))}: waiting for the fit worker.${
        this.data.worker?.alive ? this._eta(p) : ''}`;
    }
    if (p.status === 'running') {
      const pr = p.progress || {};
      const t0 = Date.parse(p.started_at), t1 = p.eta?.ends_at ? Date.parse(p.eta.ends_at) : NaN;
      const frac = Number.isFinite(t1) && t1 > t0 ? Math.min(0.99, (Date.now() - t0) / (t1 - t0)) : null;
      return `Running: ${esc(STAGE_NAMES[pr.stage] || 'starting')}${pr.device ? `, on the ${esc(pr.device)}` : ''}, since ${esc(fmtWhen(p.started_at))}
        (asked for by <b>${esc(p.name)}</b>).${this._eta(p)}<progress class="predict-one-bar" max="1"${frac == null ? '' : ` value="${frac}"`}></progress>`;
    }
    if (p.status !== 'done') {
      return `${STATUS_NAMES[p.status] || esc(p.status)} ${esc(fmtWhen(p.finished_at))} (asked for by <b>${esc(p.name)}</b>)${p.error
        ? `: ${esc(p.error)}` : ''}. <a href="/api/predictions/${p.id}/log" target="_blank" rel="noopener">Its log</a>.`;
    }
    const s = r.scores;
    return `Predicted ${esc(fmtWhen(p.finished_at))}, asked for by <b>${esc(p.name)}</b>, on the ${esc(r.device)} in ${fmtDuration(r.seconds)}:
      ${r.n} ROIs, ${r.named.length ? listLink(`${r.named.length} of them named`, `Named (${r.named.length})`, r.named) : 'none of them named'}.${s
      ? `<br>Against the fly's own ${s.manual} ROIs${s.recall_2d == null ? '' : ` (${s.manual_2d} outlines on their planes)`}:<br>${scoreTip('found', 'found')} ${both(s, 'recall')},
        ${scoreTip('right', 'right')} ${both(s, 'precision')}, ${scoreTip('dice', 'Dice')} ${both(s, 'dice', num2)} <span class="muted">(${s.recall_2d == null ? '' : 'per plane · as 3D objects; '}${scoringNote(s)})</span>;
        ${scoreTip('named', 'name correct')} ${s.named}/${s.named_of}.${s.edge_share == null && s.gap_um == null ? '' : `<br>Kept from neighbors
        <span class="muted">(the fly's own ROIs' in brackets)</span>: ${PURITY_COLUMNS.map((c) => `${tip(c[0].toLowerCase(), c[1])} ${c[2](s)}`).join(', ')}.`}` : ''}${missingNote(r, esc)}${this._refineNote(p)}${p.edited ? ` Edited in the viewer (${esc(this._edited(p))}:
        ${p.edited.rois} ROIs${p.edited.simplified == null ? '' : `, simplified within ${p.edited.simplified} px`}): its download has those
        too, as <code>${esc(p.fly)}_pred-edited_RoiSet.zip</code>.` : ''}${nmfNote(p, esc, fmtWhen)}`;
  }

  // how long a prediction asked for now should take (the server's `expected`), while the worker is there
  _expected() {
    const e = this.data?.expected;
    if (!e || !this.data.worker?.alive) return '';
    const wait = e.until - e.seconds;
    return ` Predicting them should take about ${fmtDuration(e.seconds)} on the ${e.device}${wait > 1
      ? `, after the ${fmtDuration(wait)} those asked for before have left` : ''} (${this._from(e)}).`;
  }

  // the flies of the menu the model chosen was not trained on (its held-out ones, those with ROIs it was not fit to, those without):
  // what Predict all asks for
  // (with "only my flies" ticked, the default: only those of whoever is named in Settings, `_isMine`)
  _untrained() {
    const m = this._model(), mine = this.h.$('#predict-all-mine').checked;
    return m ? this.listed.filter((f) => f.raw_dir && !m.train.includes(this._name(f)) && (!mine || this._isMine(f))) : [];
  }

  // whether a fly of the menu is of whoever is named in Settings: of a dataset they uploaded, or in their folders on tensor
  _isMine(f) {
    const name = this.h.getName().trim().toLowerCase();
    if (!name) return false;
    return (f.datasets || []).some((d) => (d.uploader || '').trim().toLowerCase() === name)
      || (this.mine.name.trim().toLowerCase() === name && this.mine.flies.some((x) => trim(x.dir) === trim(f.dir)));
  }

  // A yellow warning if the fly chosen lacks odors the model was fit with: how many, and which (asked of the server once a fly and model)
  async _renderMissing() {
    const { $, esc, msgBox, api } = this.h, m = this._model(), fly = this._fly(), box = $('#predict-missing');
    if (!m || !fly) { box.replaceChildren(); return; }
    const key = `${m.id}|${fly.dir}`;
    if (!this.missing.has(key)) this.missing.set(key, api(`/api/predictions/missing?fit=${m.id}&dir=${encodeURIComponent(fly.dir)}`).catch(() => null));
    const r = await this.missing.get(key);
    if (key !== `${this._model()?.id}|${this._fly()?.dir}`) return;
    if (!r || !r.missing.length) { box.replaceChildren(); return; }
    const img = r.missing_image_odors.length;
    box.replaceChildren(msgBox('warn', `This fly lacks <b>${r.missing.length}</b> of the ${r.odors} diagnostic odors fit #${m.id}'s model was trained with:
      ${r.missing.map(esc).join(', ')}. Its ROIs would be joined and named by those it has${img ? `, and found from images of ${img} fewer odor${img === 1 ? '' : 's'}
      than Cellpose was trained on, so fewer may be found` : ''}${r.odors - r.missing.length < 3 ? '; it has under 3 of them, too few to predict' : ''}.`));
  }

  // The last batch of the model chosen (Predict all): a bar of its flies done, and of each fly done the model's odors it lacked, and those that failed
  _renderBatch() {
    const { $, esc, msgBox } = this.h, m = this._model(), box = $('#predict-batch');
    const b = m && (this.data?.batches || []).find((x) => x.fit_id === m.id);
    const untrained = this._untrained().length;
    $('#predict-all').disabled = !(m && untrained && this.h.getName());
    $('#predict-all').textContent = `Predict all ${untrained || ''} ${$('#predict-all-mine').checked ? 'of my flies' : 'flies'} it was not trained on`.replace('  ', ' ');
    if (!b) { box.replaceChildren(); return; }
    const ps = b.ids.map((id) => this.data.predictions.find((p) => p.id === id)).filter(Boolean);
    const over = ps.filter((p) => !ACTIVE.has(p.status)).length, failed = ps.filter((p) => p.status === 'failed' || p.status === 'interrupted');
    const lacking = ps.filter((p) => p.status === 'done' && p.result?.missing_odors?.length);
    const head = `Batch of ${ps.length} flies asked for by <b>${esc(b.name)}</b> ${esc(this.h.fmtWhen(b.created_at))}: <b>${over}</b> of ${ps.length} over${
      failed.length ? `, ${failed.length} failed` : ''}. <button type="button" class="link" data-act="forget-batch" title="Forget this batch (its predictions stay)">forget it</button>`;
    const bar = `<progress class="predict-batch-bar" max="${ps.length}" value="${over}"></progress>`;
    const lack = lacking.length ? `<p class="small">Of the flies done, ${lacking.length} lacked diagnostic odors the model was trained with (each prediction's
      <code>missing_odors.csv</code>, in its download, lists them):</p><ul class="predict-batch-flies">${lacking.map((p) => `<li><b>${esc(p.fly)}</b>: ${
      p.result.missing_odors.length} of ${p.result.model_odors} (${p.result.missing_odors.map(esc).join(', ')})</li>`).join('')}</ul>` : '';
    const fail = failed.length ? `<p class="small">Failed:</p><ul class="predict-batch-flies">${failed.map((p) => `<li><b>${esc(p.fly)}</b>: ${esc(p.error || p.status)}</li>`).join('')}</ul>` : '';
    const kind = lacking.length || failed.length ? 'warn' : over === ps.length ? 'ok' : 'info';
    // (which flies lacked odors and which failed: folded away, unless it was opened, or the batch ended while the page was open)
    const all = over === ps.length, seen = this.batchSeen.get(b.id);
    if (!seen) this.batchSeen.set(b.id, { over: all, open: false });
    else if (!seen.over && all) Object.assign(seen, { over: true, open: true });
    else seen.over = all;
    const more = lack || fail ? `<details class="predict-batch-more"${this.batchSeen.get(b.id).open ? ' open' : ''}><summary class="small">${[
      lacking.length ? `${lacking.length} lacked diagnostic odors` : '', failed.length ? `${failed.length} failed` : ''].filter(Boolean).join(', ')}: which</summary>${lack}${fail}</details>` : '';
    const el = msgBox(kind, head + bar + more);
    el.dataset.batch = b.id;
    box.replaceChildren(el);
  }

  async startAll() {
    const { $, api, msgBox, esc } = this.h, m = this._model(), name = this.h.getName(), flies = this._untrained();
    if (!m || !name || !flies.length) return;
    $('#predict-all').disabled = true;
    try {
      const b = await api('/api/predictions/batch', { method: 'POST', json: { name, fit: m.id, flies: flies.map((f) => ({ dir: f.dir, raw_dir: f.raw_dir })),
        ...this._cleanOptions() } });
      if (b.left_out.length) $('#predict-status').replaceChildren(msgBox('warn', `Left out, not found: ${b.left_out.map((x) => `${esc(x.dir)} (${esc(x.why)})`).join('; ')}`));
    } catch (e) {
      $('#predict-batch').replaceChildren(msgBox('error', `Could not ask for them: ${esc(e.message)}`));
    }
    await this.refresh();
  }

  async _forgetBatch(id) {
    try { await this.h.api(`/api/predictions/batches/${id}`, { method: 'DELETE' }); } catch { /* (listed again below) */ }
    await this.refresh();
  }

  _renderPair() {
    const { $, msgBox } = this.h;
    this._renderMissing();
    this._renderBatch();
    $('#predict-paths').hidden = $('#predict-fly').value !== OTHER;
    const p = this._pair(), m = this._model(), fly = this._fly();
    $('#predict-view').disabled = !(p && p.status === 'done');
    $('#predict-start').disabled = !(m && fly && this.h.getName()) || (p && ACTIVE.has(p.status));
    $('#predict-status').replaceChildren(...(!m || !fly ? [] : p
      ? [msgBox(p.status === 'done' ? 'ok' : ACTIVE.has(p.status) ? 'info' : 'error', this._about(p))]
      : [msgBox('info', `This model has not predicted this fly's ROIs yet.${this._expected()}`)]));
  }

  // every prediction, newest first
  _renderList() {
    const { $, esc, fmtWhen } = this.h, ps = this.data.predictions;
    // (the batch each is of: the newest it is in)
    const batchOf = new Map();
    for (const b of [...(this.data.batches || [])].reverse()) {
      for (const id of b.ids) batchOf.set(id, { id: b.id, label: `Batch #${b.id}: fit #${b.fit_id}'s predictions, asked for by <b>${esc(b.name)}</b> ${esc(fmtWhen(b.created_at))}` });
    }
    $('#predict-list').innerHTML = !ps.length ? '' : `<h3>Predictions</h3><table class="odor-table predictions-table"><thead><tr>
      <th>Fly</th><th>Model</th><th>Asked for by</th><th>When</th><th>Status</th><th>ROIs <span class="muted">(named; ${scoreTip('found', 'found')}, ${scoreTip('right', 'right')}, ${scoreTip('dice', 'Dice')}, ${scoreTip('named', 'name correct')})</span></th>
      <th title="Its ROIs kept from their neighbors: split where an outline holds two glomeruli, trimmed, kept apart. Show: their scores before and after">Refinement</th>
      <th></th></tr></thead><tbody>${batchGroups(ps, (p) => batchOf.get(p.id) || null, (p) => {
      const r = p.result, s = r?.scores;
      const rois = r ? `${r.n} <span class="muted">(${r.named.length} named)</span>${s ? ` <span class="muted">· ${both(s, 'recall')}, ${both(s, 'precision')},
        ${both(s, 'dice', num2)}, ${s.named_of ? pct(s.named / s.named_of) : '—'}</span>` : ''}${
        p.edited ? ` <span class="muted" title="Edited in the viewer, ${esc(this._edited(p))}">· edited</span>` : ''}${
        p.nmf ? ` <span class="muted" title="${esc(nmfNote(p, (x) => x, fmtWhen).trim())}">· NMF${p.nmf.status === 'done' ? `: ${p.nmf.result?.candidates ?? 0} found`
          : p.nmf.status === 'running' ? ` ${Math.round((p.nmf.fraction || 0) * 100)}%` : p.nmf.status === 'queued' ? ' queued' : ' failed'}</span>` : ''}` : '—';
      const left = p.eta && this.data.worker?.alive ? ` <span class="muted" title="${esc(this._from(p.eta))}">· ${p.eta.until
        ? `about ${fmtDuration(p.eta.until)} left` : 'ending any moment'}</span>` : '';
      const status = `${STATUS_NAMES[p.status] || esc(p.status)}${p.status === 'running' && p.progress?.device ? ` <span class="muted">(${esc(p.progress.device)})</span>` : ''}${left}${
        p.error ? ` <span class="muted" title="${esc(p.error)}">(${esc(p.error.slice(0, 60))}${p.error.length > 60 ? '…' : ''})</span>` : ''}`;
      return `<tr data-id="${p.id}"><td title="${esc(p.tensor_dir || p.fly_dir)}">${esc(p.fly)}</td>
        <td>#${p.fit_id}${p.fit_deleted ? ' <span class="muted">(deleted)</span>' : ''}</td><td>${esc(p.name)}</td>
        <td>${esc(fmtWhen(p.finished_at || p.created_at))}</td><td>${status}</td><td>${rois}</td><td class="acts">${this._refineCell(p)}</td>
        <td class="acts"><div class="acts-grid">${p.status === 'done' ? `<button type="button" class="btn small" data-act="view">View</button>
          <a class="btn small" href="/api/predictions/${p.id}/download" download>Download</a>` : ''}
          ${p.status !== 'queued' ? `<a class="btn small" href="/api/predictions/${p.id}/log" target="_blank" rel="noopener">Log</a>` : ''}
          ${p.status !== 'running' ? '<button type="button" class="btn small danger" data-act="delete">Delete</button>' : ''}</div></td></tr>${
        this.shown.has(p.id) && p.refine?.status === 'done' ? `<tr class="refine-details" data-id="${p.id}"><td colspan="8">${this._refineHtml(p, true)}</td></tr>` : ''}`;
    }, this.openBatches, 8, (p) => (STATUS_NAMES[p.status] || p.status).toLowerCase())}</tbody></table>`;
  }

  // a prediction's Refinement cell in the list: Show (its scores), once refined; how it is, while it is being; else Refine
  _refineCell(p) {
    const { esc } = this.h, r = p.refine;
    if (p.status !== 'done') return '—';
    if (r?.status === 'done') {
      const open = this.shown.has(p.id);
      return `<button type="button" class="btn small" data-act="refine-show" aria-expanded="${open}" title="${esc(`Refined: ${stepsText(r.options)}`)}">${
        open ? 'Hide' : 'Show'}</button>`;
    }
    if (r && ACTIVE.has(r.status)) {
      return `<span class="muted">${r.status === 'queued' ? 'Queued' : `Refining${r.progress?.stage ? `: ${REFINE_STAGES[r.progress.stage] || esc(r.progress.stage)}` : '…'}`}</span>`;
    }
    const failed = r ? `<span class="muted" title="${esc(r.error || '')}">${r.status === 'failed' ? 'Failed' : 'Interrupted'}</span> ` : '';
    return p.refinable?.any ? `${failed}<button type="button" class="btn small" data-act="refine">Refine</button>` : `${failed}—`;
  }

  // what a prediction's refinement done did, and its ROIs' scores before and after (`buttons`: with Refine again)
  _refineHtml(p, buttons) {
    const { esc, fmtWhen } = this.h, r = p.refine, res = r.result || {};
    const steps = res.steps || [], did = [...(steps.includes('split') && res.split != null ? [`${plural(res.split, 'ROI')} ${res.split === 1 ? 'is a part'
      : 'are parts'} of an outline cut in two`] : []), ...(steps.includes('trim') && res.trimmed != null ? [`${plural(res.trimmed, 'ROI')} trimmed`] : [])];
    const said = did.length ? ` ${did.join(', and ')}, on some plane (in the viewer, hover over one in the list for what was done to it there).` : '';
    return `<p class="small">Refined ${esc(fmtWhen(r.finished_at))}${r.with_prediction ? ', as it was predicted' : ''}, asked for by <b>${esc(r.name || '?')}</b>:
      ${esc(stepsText(r.options))}${(res.steps || []).length < [...(r.options?.split ? ['split'] : []), ...(r.options?.clean || [])].length
      ? ' <span class="muted">(core left out: Cellpose\'s probabilities were not kept)</span>' : ''}.${said}${r.legacy
      ? ' <span class="muted">Refined before the model\'s own ROIs were kept apart: those here are already split, if split was asked for.</span>' : ''}</p>
      ${refineTable(r, esc)}${buttons ? '<div class="actions"><button type="button" class="btn small" data-act="refine">Refine again…</button></div>' : ''}`;
  }

  // a prediction's refinement, in what is said of it above the list
  _refineNote(p) {
    const { esc } = this.h, r = p.refine;
    if (!r) return '';
    if (r.status === 'done') return `<br>${this._refineHtml(p, false)}`;
    if (ACTIVE.has(r.status)) return ` Its ROIs are ${r.status === 'queued' ? 'to be refined (waiting for the fit worker)' : 'being refined'}: ${esc(stepsText(r.options))}.`;
    return ` Refining its ROIs ${r.status === 'failed' ? 'failed' : 'was interrupted'}${r.error ? ` (${esc(r.error)})` : ''}: they are the model's own.`;
  }

  // the refine dialog: which steps can be ticked (core, only of a prediction that kept what it goes by)
  _refineChecked() {
    const box = this.h.$('#refine-steps'), core = box.querySelector('[data-clean="core"]');
    core.disabled = !this.refining?.refinable?.core;
    if (core.disabled) core.checked = false;
    core.closest('label').title = core.disabled ? "This prediction did not keep Cellpose's cell probabilities, which core goes by (only those made since 2026-11-16 do): predict it again to refine by it" : '';
    const o = this._cleanOptions(box);
    box.closest('form').querySelector('[value="refine"]').disabled = !(o.split || o.clean.length);
  }

  // asks how to refine a prediction's ROIs (as it was refined last, if it was), and asks for it
  async refine(p) {
    const { $, api, msgBox, esc, askDialog, fmtWhen } = this.h;
    const name = this.h.getName();
    if (!name) {
      $('#predict-status').replaceChildren(msgBox('error', 'Enter your first name in <a href="#settings/name">Settings</a> first: refinements say who asked for them.'));
      return;
    }
    const box = $('#refine-steps'), o = p.refine?.options || { clean: [], clean_params: {}, split: false };
    box.querySelector('[data-split]').checked = Boolean(o.split);
    for (const c of box.querySelectorAll('[data-clean]')) c.checked = (o.clean || []).includes(c.dataset.clean);
    for (const i of box.querySelectorAll('[data-clean-param]')) {
      const [step, key] = i.dataset.cleanParam.split('.');
      i.value = o.clean_params?.[step]?.[key] ?? i.defaultValue;
    }
    this.refining = p;
    this._refineChecked();
    $('#refine-dialog-q').textContent = `Refine fit #${p.fit_id}'s ROIs of ${p.fly}`;
    $('#refine-dialog-detail').innerHTML = [
      'Applied in this order: split first, then the steps that only remove pixels, on each plane, from the model\'s own ROIs. Those are kept, to compare with',
      '(the list\'s Show, and the viewer); the refined ones are then the prediction\'s, to view, download and extract signals from.',
      ...(p.refine?.status === 'done' ? [`This replaces its refinement of ${esc(fmtWhen(p.refine.finished_at))} (made again from the model's own).`] : []),
      ...(p.edited ? ['Its ROIs as edited in the viewer stay as they are: a copy of those it has now.'] : []),
      ...(p.nmf?.status === 'done' ? ['The look for glomeruli its ROIs miss, made of those it has now, is deleted: ask for it again in the viewer.'] : []),
    ].join(' ');
    if (await askDialog($('#refine-dialog')) !== 'refine') return;
    try {
      await api(`/api/predictions/${p.id}/refine`, { method: 'POST', json: { name, ...this._cleanOptions(box) } });
    } catch (err) {
      $('#predict-status').replaceChildren(msgBox('error', `Could not ask for it: ${esc(err.message)}`));
    }
    await this.refresh();
  }

  // the example under Fit a model: the last fit done's download, and the al-diag-seg (and hong2p) the worker has
  _renderHowto() {
    const { $ } = this.h, m = this.data.models[0], w = this.data.worker || {};
    const fit = m ? `olfdb_fit_${m.id}` : 'olfdb_fit_<fit>';
    const at = (c) => (c ? `@${c}` : '');
    $('#fit-howto').textContent = `unzip ${fit}.zip              # the download of ${m ? `fit #${m.id}` : 'a fit'}
python3 -m venv venv && . venv/bin/activate
# (the versions the server's fit worker has now: its README.txt says those its model was made with)
pip install "hong2p @ git+https://github.com/ejhonglab/hong2p${at(w.hong2p)}" \\
            "al-diag-seg @ git+https://github.com/tom-f-oconnell/al-diag-seg${at(w.alseg?.commit)}"
alseg-predict ${fit}/model --fly /local/tensor/Ann/analysis/2026-03-31/1 \\
    --raw /local/tensor/Ann/raw/2026-03-31/1 --out predictions/
# predictions/<fly>_RoiSet.zip: the fly's ROIs, named by glomerulus (alseg-predict -h says more)`;
  }

  async view(p) {
    const { api, msgBox, esc, $ } = this.h;
    let v;
    try { v = await api(`/api/predictions/${p.id}/view`); } catch (e) {
      $('#predict-status').replaceChildren(msgBox('error', `Could not open it: ${esc(e.message)}`));
      return;
    }
    // (its ROIs can be edited there: saved as whoever is named in Settings, and said here once they are)
    this.h.player().openPrediction(v, `Glomerulus ROIs of ${p.fly}, predicted by fit #${p.fit_id}'s model`,
      { name: () => this.h.getName(), onSaved: () => this.refresh() });
  }

  // asks whether to view a prediction there is already of the model and fly, or make it again: 'view', 'rerun' or ''
  _askAgain(p) {
    const { $, fmtWhen, askDialog } = this.h;
    $('#predict-again-detail').textContent = `Fit #${p.fit_id}'s model predicted ${p.fly}'s ROIs ${fmtWhen(p.finished_at)}, as ${p.name} asked.
      Predicting them again replaces those${p.edited ? `, and deletes them as edited in the viewer (${this._edited(p)})` : ''}.`;
    return askDialog($('#predict-again'));
  }

  // how the prediction asked for refines its ROIs, from the form (or the refine dialog's `box`): the server's predictions.options
  _cleanOptions(box = this.h.$('#predict-clean')) {
    const clean = [...box.querySelectorAll('[data-clean]')].filter((c) => c.checked && !c.disabled).map((c) => c.dataset.clean);
    const clean_params = {};
    for (const i of box.querySelectorAll('[data-clean-param]')) {
      const [step, key] = i.dataset.cleanParam.split('.');
      if (clean.includes(step) && i.value !== '' && Number.isFinite(Number(i.value))) (clean_params[step] ||= {})[key] = Number(i.value);
    }
    return { clean, clean_params, split: box.querySelector('[data-split]').checked };
  }

  async start() {
    const { $ } = this.h;
    const name = this.h.getName(), m = this._model(), fly = this._fly();
    if (!name || !m || !fly) { this._renderPair(); return; }
    const known = this._pair();
    if (known && ACTIVE.has(known.status)) return;
    let overwrite = Boolean(known); // (one that failed: again, without asking)
    if (known?.status === 'done') {
      const a = await this._askAgain(known);
      if (a === 'view') this.view(known);
      if (a !== 'rerun') return;
    }
    $('#predict-start').disabled = true;
    await this._post({ name, fit: m.id, ...fly, overwrite, ...this._cleanOptions() });
  }

  // asks for a prediction; if the server has one of that model and fly already (the same fly,
  // entered otherwise), asks again, as `start` does
  async _post(body) {
    const { $, api, msgBox, esc } = this.h;
    try { await api('/api/predictions', { method: 'POST', json: body }); } catch (err) {
      const p = err.status === 409 ? err.body?.prediction : null;
      if (p && !body.overwrite) {
        const a = p.status === 'done' ? await this._askAgain(p) : 'rerun';
        if (a === 'view') this.view(p);
        if (a === 'rerun') return this._post({ ...body, overwrite: true });
      } else {
        $('#predict-status').replaceChildren(msgBox('error', `Could not ask for it: ${esc(err.message)}`));
        $('#predict-start').disabled = false;
        return undefined;
      }
    }
    return this.refresh();
  }

  async _action(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    if (b.dataset.act === 'batch-toggle') {
      const k = +b.dataset.batchId;
      if (this.openBatches.has(k)) this.openBatches.delete(k); else this.openBatches.add(k);
      this._renderList();
      return;
    }
    const id = Number(b.closest('[data-id]').dataset.id), p = this.data.predictions.find((x) => x.id === id);
    const { $, api, msgBox, esc, askDialog } = this.h;
    if (b.dataset.act === 'view') { this.view(p); return; }
    if (b.dataset.act === 'refine') { this.refine(p); return; }
    if (b.dataset.act === 'refine-show') {
      if (this.shown.has(id)) this.shown.delete(id); else this.shown.add(id);
      this._renderList();
      return;
    }
    if (b.dataset.act === 'delete') {
      $('#delete-confirm-q').textContent = `Delete fit #${p.fit_id}'s prediction of ${p.fly}'s ROIs?`;
      $('#delete-confirm-detail').textContent = 'Its ROIs, and what they were made from, are deleted from the server, for everyone.';
      $('#delete-confirm-warn').replaceChildren();
      if (await askDialog($('#delete-confirm')) !== 'delete') return;
      try { await api(`/api/predictions/${id}`, { method: 'DELETE' }); } catch (err) {
        $('#predict-status').replaceChildren(msgBox('error', esc(err.message)));
      }
      await this.refresh();
    }
  }
}
