// Fits of al-diag-seg models in the AL segmentation tab (fits.py, fit_worker.py): asking for one
// (as the name set in Settings), those queued and running (who asked, from where, when, how far,
// and when it should end), and past ones above them (their scores on the held-out
// flies, and their model to download, or to delete).

const STAGE_NAMES = { load: 'Finding and loading the flies', train: 'Training Cellpose', atlas: 'Building the atlas',
  evaluate: 'Scoring the held-out flies' };
const STATUS_NAMES = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', cancelled: 'Stopped',
  interrupted: 'Interrupted' };
const REF_NAMES = { grabe2016: 'Grabe et al. 2016' }; // fits.REFERENCES
// the odors asked of a fit's flies unless unticked (of those some of the tab's flies have and some lack): all the flies recorded
// since 2024-07 have them, so a model of those flies can know them
const REQUIRE_CHECKED = ['acetophenone @ -4', 'valeric acid @ -3'];

// a fit's reference atlases, by name
const refNames = (o) => (o.refs || []).map((r) => REF_NAMES[r] || r);
// how a fit's ROIs are made (fits.ROI_DEFAULTS), if it was asked for since they were options
const minus = (v) => String(v).replace('-', '−');
const roiSettings = (o) => (o.cellprob_threshold == null ? '' : ` Its ROIs: cell probability threshold ${minus(o.cellprob_threshold)},
  flow threshold ${o.flow_threshold}, grown by ${o.dilate} px; planes joined at overlap ${o.join_overlap} and tuning correlation
  ${minus(o.join_corr)}${o.max_cost == null ? '' : '; named where a name costs ' + o.max_cost + ' or less'}.`);

// a duration in seconds as '1 h 5 min', '12 min', '40 s'
export function fmtDuration(s) {
  if (s == null || !Number.isFinite(s)) return '—';
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ''}`;
}

const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
const num2 = (v) => (v == null ? '—' : v.toFixed(2));
// the final ROIs, by their kind (al-diag-seg's)
const FINAL_NAMES = { pred: "Cellpose's own, named", refined_prior: 'refined with the atlas', refined_tuning: 'refined by tuning' };

// a score plane by plane and as 3D objects ('46% · 30%'), or as 3D objects alone (fits before
// al-diag-seg scored them plane by plane)
export function both(x, key, fmt = pct) {
  return x[`${key}_2d`] == null ? fmt(x[key]) : `${fmt(x[`${key}_2d`])} <span class="muted">· ${fmt(x[key])}</span>`;
}

// how a fit's or a prediction's ROIs were scored (al-diag-seg's): as glomeruli drawn on some of their
// planes only (often their clearest), so not counted wrong where none is drawn; or strictly (before)
// a score of the predicted ROIs with the fly's own ROIs' in brackets ('0.30 (0.15)'); '' for fits scored before it was
const withManual = (x, key, fmt = num2, manual = `manual_${key}`) => (x[key] == null ? '—'
  : `${fmt(x[key])} <span class="muted">(${fmt(x[manual])})</span>`);
const um = (v) => (v == null ? '—' : `${v.toFixed(1)} µm`);
// how far ROIs keep from each other and how much of a neighbour they carry (al-diag-seg's purity scores): the
// table's columns after Leak, each [heading, what it is, a fly's cell]
export const PURITY_COLUMNS = [
  ['Gap', "Distance from an ROI to the nearest other ROI on its plane: the median over ROIs' outlines on their planes", (x) => withManual(x, 'gap_um', um)],
  ['Touching', 'Outlines on a plane that touch another ROI there', (x) => withManual(x, 'touching', pct)],
  ['Neighbor at edge', `Of the signal of an ROI's edge facing its nearest neighbor (the outer 1 µm, within 3 µm of it), the share that is the neighbor's: the edge's odor tuning as a sum of its own core's and the neighbor's core's (cores: 1.5 µm from the edge). Median over outlines with a core and such a neighbor. 0: all its own`, (x) => withManual(x, 'edge_share')],
  ['Neighbor in ROI', "The same of the whole ROI's mean signal: what reaches its responses", (x) => withManual(x, 'whole_share')],
  ['Responding', "Of ROIs matching one of the fly's own on a plane (IoU at least 0.5) and the odors, the pairs with a mean response over 0.3 dF/F; in brackets the same of the fly's own ROIs they match. Higher than them: tuned more broadly", (x) => withManual(x, 'breadth', pct, 'breadth_manual')],
];

export const partlyDrawn = (x) => x.scoring === 'partly drawn';
export const scoringNote = (x) => (partlyDrawn(x) ? 'not counted wrong where no glomerulus is drawn'
  : 'scored strictly: also counted wrong where no glomerulus is drawn');

export class FitsPanel {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, fmtSize, getName (the name in Settings, '' if none) }
  constructor(h) {
    this.h = h;
    this.data = null;
    this.timer = null;
    this.open = new Set(); // past fits whose details are shown
    this.require = new Set(REQUIRE_CHECKED); // the odors asked of its flies (those ticked in the form)
    const { $ } = h;
    $('#fit-odors').addEventListener('change', (e) => {
      const o = e.target.dataset.odor;
      if (o) { if (e.target.checked) this.require.add(o); else this.require.delete(o); }
      this._renderOdorsTotal();
      this._renderExpected();
    });
    $('#fit-start').addEventListener('click', () => this.start());
    const atlas = $('#fit-atlas'), grabe = $('#fit-grabe');
    const sync = () => { grabe.disabled = !atlas.checked; grabe.closest('label').classList.toggle('disabled', !atlas.checked); };
    atlas.addEventListener('change', sync);
    sync();
    for (const id of ['#fit-held-out', '#fit-epochs', '#fit-atlas', '#fit-grabe']) $(id).addEventListener('input', () => this._renderExpected());
    $('#fits-list').addEventListener('click', (e) => this._action(e));
    $('#fits-past').addEventListener('click', (e) => this._action(e));
  }

  // while the tab is shown: every 5 s if a fit is queued or running, else every 30 s
  async load() {
    await this.refresh();
    if (this.timer) return;
    let n = 0;
    this.timer = setInterval(() => {
      if (this.h.$('#tab-al-segmentation').hidden) return;
      n += 1;
      const busy = this.data?.fits.some((f) => f.status === 'queued' || f.status === 'running');
      if (busy || n % 6 === 0) this.refresh();
    }, 5000);
  }

  async refresh() {
    try { this.data = await this.h.api('/api/fits'); } catch (e) {
      this.h.$('#fit-worker').replaceChildren(this.h.msgBox('error', `Could not ask the server about fits: ${this.h.esc(e.message)}`));
      return;
    }
    this.render();
  }

  render() {
    const { $, esc, msgBox, fmtWhen } = this.h;
    const name = this.h.getName();
    $('#fit-who').innerHTML = name
      ? `Fits say who asked for them: this one will be <b>${esc(name)}</b>'s (as set in <a href="#settings/name">Settings</a>).`
      : 'Enter your first name in <a href="#settings/name">Settings</a> first: fits say who asked for them.';
    $('#fit-start').disabled = !name;
    this._renderExpected();
    const w = this.data?.worker;
    $('#fit-worker').replaceChildren(!w ? '' : w.alive
      ? msgBox('ok', `The fit worker is running on ${esc(w.host || '?')}: ${w.gpu ? `GPU ${esc(w.gpu)}` : 'no GPU (training would be very slow)'}${
        w.alseg?.commit ? `, al-diag-seg <code>${esc(w.alseg.commit.slice(0, 7))}</code>` : ''}.`)
      : msgBox('warn', `No fit worker is running${w.seen ? ` (last heard from ${esc(fmtWhen(w.seen))})` : ''}: fits asked for wait
        until one starts. It runs in a container of its own, with the GPU (DEPLOY.md).`));
    const fits = this.data?.fits || [];
    const active = fits.filter((f) => f.status === 'queued' || f.status === 'running').reverse(); // oldest first
    $('#fits-list').innerHTML = active.length ? active.map((f) => this._activeHtml(f)).join('') : '';
    const past = fits.filter((f) => f.status !== 'queued' && f.status !== 'running');
    $('#fits-past-step').hidden = !past.length;
    $('#fits-past').innerHTML = past.length ? `<table class="odor-table fits-table"><thead><tr><th>Fit</th><th>Asked for by</th>
      <th>From</th><th>Started</th><th>Took</th><th>Status</th><th title="Of the held-out flies, the final ROIs: found (IoU ≥ 0.5) of their own ROIs, and of the ROIs predicted; mean Dice of those found (each plane by plane: a glomerulus's outline on each plane, as ROIs are drawn; then as 3D objects: its outlines on all its planes as one); and ROIs named as the ROI of theirs they overlap">Held-out flies:
      found, right, Dice <span class="muted">(per plane · 3D)</span>, named</th><th></th></tr></thead><tbody>${past.map((f) => this._pastHtml(f)).join('')}</tbody></table>` : '';
  }

  // The tab's flies with the odors of their diagnostic recordings (null until they are found, or as found before those were
  // kept with them), and whose each is: [{ who, date, odors: Set }]
  _odorFlies() {
    const flies = this.h.flies?.();
    if (!flies?.length || !flies.every((f) => f.odors)) return null;
    return flies.map((f) => ({ who: [...new Set(f.datasets.map((d) => d.uploader || '?'))].sort().join(' + '), date: f.date, odors: new Set(f.odors) }));
  }

  // flies as 'Sam 29 (2024-07-16 to 2026-03-31), Tom 9 (…)': how many of each person's, recorded from when to when
  _span(flies) {
    const by = new Map();
    for (const f of flies) by.set(f.who, [...(by.get(f.who) || []), f.date]);
    return [...by].sort(([a], [b]) => a.localeCompare(b)).map(([who, d]) => { d.sort(); return `${this.h.esc(who)} ${d.length} (${d[0] === d.at(-1) ? d[0] : `${d[0]} to ${d.at(-1)}`})`; }).join(', ');
  }

  // the odors a fit can ask of its flies (the odors ticked in the form, of those the tab's flies have): those of flies found
  _required(flies) { return [...this.require].filter((o) => flies.some((f) => f.odors.has(o))); }

  // The odors that some of the tab's flies have and some lack, each with a box to ask it of the fit's flies: who has it and
  // who does not, recorded when; and what the fit is of with the boxes as they are (`_renderOdorsTotal`).
  renderOdors() {
    const { $, esc } = this.h, box = $('#fit-odors'), flies = this._odorFlies();
    box.hidden = !flies;
    if (!flies) return;
    const count = new Map();
    for (const f of flies) for (const o of f.odors) count.set(o, (count.get(o) || 0) + 1);
    const some = [...count].filter(([, n]) => n < flies.length).sort(([a, n], [b, m]) => m - n || a.localeCompare(b));
    if (!some.length) { box.hidden = true; return; }
    box.innerHTML = `<p><b>Odors asked of its flies.</b> <span class="muted">The model knows the odors all the flies it is fit to share: the
      ${[...count].filter(([, n]) => n === flies.length).length} that all ${flies.length} flies below have, and each odor ticked
      here: the fit is then only to the flies that have it. A fly without an odor of the model can still be predicted, by those it has.</span></p>
      ${some.map(([o, n]) => `<label class="check"><input type="checkbox" data-odor="${esc(o)}"${this.require.has(o) ? ' checked' : ''}>
        <span><b>${esc(o)}</b>: ${n} of the ${flies.length} flies have it: ${this._span(flies.filter((f) => f.odors.has(o)))}.
        <span class="muted">Without it: ${this._span(flies.filter((f) => !f.odors.has(o)))}.</span></span></label>`).join('')}
      <p class="fit-odors-total" id="fit-odors-total"></p>
      <label class="check" title="Cellpose is given three images of a fly: its average image, and two made of its responses to odors. Ticked, those two are made of the odors every fly below has, whichever flies the model is fit to: the odors ticked above are then only in what its ROIs are joined into 3D and named by, so a fly without them gets exactly the ROIs and names that a model fit to the same flies without those odors would give it. Unticked, the images are of all the model's odors, and a fly without one of them is found from images Cellpose was not trained on: fewer of its ROIs are found."><input type="checkbox" id="fit-image-shared" checked>
        <span>Cellpose's images of responses from the odors every fly has only <span class="muted">(so a fly without the odors ticked is predicted exactly as by a model fit without them)</span></span></label>`;
    this._renderOdorsTotal();
    this._renderExpected();
  }

  // what a fit asked for now is of: the flies with every odor ticked, and the odors they share
  _renderOdorsTotal() {
    const flies = this._odorFlies(), box = this.h.$('#fit-odors-total');
    if (!flies || !box) return;
    const req = this._required(flies), chosen = flies.filter((f) => req.every((o) => f.odors.has(o)));
    const shared = chosen.length ? [...chosen[0].odors].filter((o) => chosen.every((f) => f.odors.has(o))).length : 0;
    box.innerHTML = chosen.length >= 2 ? `As ticked, a fit is to <b>${chosen.length}</b> of the ${flies.length} flies (${this._span(chosen)}), and its model has
      the <b>${shared}</b> odors they share.` : `<span class="player-error">As ticked, ${chosen.length} flies have every odor: a fit needs 2 at least.</span>`;
  }

  // how many flies a fit asked for now would be of (those with every odor ticked), if the tab's flies are found
  _chosenFlies() {
    const flies = this._odorFlies();
    if (!flies) return null;
    const req = this._required(flies);
    return flies.filter((f) => req.every((o) => f.odors.has(o))).length;
  }

  // how long a fit asked for now, with the form's options, should take: each stage's size (as fits.sizes: of as many
  // flies as the last fit that listed its own had) times what a unit of it took in the last fits done (the server's
  // `rates`); and when it would end, after those queued and running
  _renderExpected() {
    const { $ } = this.h, r = this.data?.rates, box = $('#fit-expected');
    if (!r) { box.textContent = ''; return; }
    const last = this.data.fits.find((f) => f.flies);
    const n = this._chosenFlies() ?? (last ? last.flies.train.length + last.flies.held_out.length : 80); // (fits.DEFAULT_FLIES)
    const held = Math.max(0, Math.min(+$('#fit-held-out').value || 0, n - 1)), train = n - held, epochs = +$('#fit-epochs').value || 0;
    const atlas = $('#fit-atlas').checked, refs = atlas && $('#fit-grabe').checked ? 1 : 0, s = r.seconds;
    const secs = s.load * n + s.train * epochs * train + (atlas ? s.atlas * (train + (last?.flies.stacks || 0) + refs) : 0) + s.evaluate * held;
    const wait = Math.max(0, ...this.data.fits.filter((f) => f.eta).map((f) => f.eta.until));
    box.textContent = `A fit asked for now should take about ${fmtDuration(secs)}${wait ? `, after the ${fmtDuration(wait)} those before it have left` : ''}
      (${r.from ? `estimated from how long the last ${r.from === 1 ? 'fit' : `${r.from} fits`} took, for the ${n} flies it would be of`
    : 'a rough guess: no fit has finished yet to estimate from'}).`;
  }

  _who(f) {
    const { esc, fmtWhen } = this.h;
    return `<b>${esc(f.name)}</b> <span class="muted">from ${esc(f.ip || '?')}</span>, asked ${esc(fmtWhen(f.created_at))}`;
  }

  _activeHtml(f) {
    const { esc, fmtWhen } = this.h;
    const p = f.progress || {};
    const stage = f.status === 'queued' ? 'Waiting for the worker' : (STAGE_NAMES[p.stage] || 'Starting');
    const frac = p.total ? Math.min(1, p.done / p.total) : 0;
    const eta = f.eta ? `ends about ${esc(fmtWhen(f.eta.ends_at))} (in ${fmtDuration(f.eta.until)}${f.status === 'queued'
      ? `, after those before it: it takes about ${fmtDuration(f.eta.seconds)}` : ''}; ${f.eta.from
      ? `estimated from how long the last ${f.eta.from === 1 ? 'fit' : `${f.eta.from} fits`} took`
      : 'a rough guess: no fit has finished yet to estimate from'})` : '';
    const o = f.options || {};
    return `<div class="fit-active" data-id="${f.id}">
      <div class="fit-head"><span><b>Fit #${f.id}</b> · ${this._who(f)}${f.started_at ? `, started ${esc(fmtWhen(f.started_at))}` : ''}</span>
        <button type="button" class="btn small" data-act="cancel" ${f.cancel ? 'disabled' : ''}>${f.cancel ? 'Stopping…' : f.status === 'queued' ? 'Cancel' : 'Stop'}</button></div>
      <div class="fit-stage">${esc(stage)}${p.total ? ` <span class="muted">(${p.done} of ${p.total})</span>` : ''}</div>
      <progress max="1" value="${frac}"></progress>
      <div class="muted small">${eta}${eta ? ' · ' : ''}${o.held_out} held out, ${o.epochs} epochs${o.atlas
        ? `, atlas${refNames(o).map((r) => ` + ${esc(r)}`).join('')}` : ', no atlas'}${o.tentative ? '' : ', no tentative ROIs'}
        ${f.status === 'running' ? ` · <a href="/api/fits/${f.id}/log" target="_blank" rel="noopener">log</a>` : ''}</div></div>`;
  }

  _pastHtml(f) {
    const { esc, fmtWhen, fmtSize } = this.h;
    const took = f.started_at && f.finished_at ? (new Date(f.finished_at) - new Date(f.started_at)) / 1000 : null;
    const r = f.result;
    const scores = r ? `${both(r.mean, 'recall')}, ${both(r.mean, 'precision')}, ${both(r.mean, 'dice', num2)}, ${r.named}/${r.named_of}${partlyDrawn(r)
      ? '' : ` <span class="muted" title="${scoringNote(r)}">(strict)</span>`}` : '—';
    const status = `${STATUS_NAMES[f.status] || esc(f.status)}${f.error ? ` <span class="muted" title="${esc(f.error)}">(${esc(f.error.slice(0, 80))}${f.error.length > 80 ? '…' : ''})</span>` : ''}`;
    const open = this.open.has(f.id);
    return `<tr data-id="${f.id}"><td>#${f.id}</td><td>${esc(f.name)}</td><td>${esc(f.ip || '?')}</td>
      <td>${f.started_at ? esc(fmtWhen(f.started_at)) : '—'}</td><td>${fmtDuration(took)}</td><td>${status}</td>
      <td>${r ? `<button type="button" class="link" data-act="details">${scores}</button>` : scores}</td>
      <td class="fit-acts">${f.status === 'done' ? `<a class="btn small" href="/api/fits/${f.id}/download" download>Download${f.model_bytes ? ` (${fmtSize(f.model_bytes)})` : ''}</a>` : ''}
        <a class="btn small" href="/api/fits/${f.id}/log" target="_blank" rel="noopener">Log</a>
        <button type="button" class="btn small danger" data-act="delete">Delete</button></td></tr>
      ${r && open ? `<tr class="fit-details"><td colspan="8">${this._detailsHtml(f)}</td></tr>` : ''}`;
  }

  _detailsHtml(f) {
    const { esc } = this.h;
    const r = f.result, o = f.options || {};
    const flies = r.flies.map((x) => `<tr><td>${esc(x.fly)}</td><td>${x.n}</td><td>${both(x, 'recall')}</td><td>${both(x, 'precision')}</td>
      <td>${both(x, 'dice', num2)}</td><td>${num2(x.leak)} <span class="muted">(${num2(x.manual_leak)})</span></td>${PURITY_COLUMNS.map((c) => `<td>${c[2](x)}</td>`).join('')}<td>${x.named}/${x.named_of}</td></tr>`).join('');
    const missed = r.glomeruli.filter((g) => g.detected < g.flies);
    const refs = refNames(o).map(esc);
    const fl = f.flies, of = fl ? fl.train.length + fl.held_out.length : 0;
    // (the odors asked of its flies, and those its model has: of fits made since they could be asked)
    const odors = fl?.odors ? ` Its model has the ${fl.odors.length} odors those flies share${fl.listed > of ? `: only the ${of} of ${fl.listed} flies with ${
      (o.require_odors || []).map(esc).join(' and ')} were fit to` : ''}${fl.image_odors && fl.image_odors.length < fl.odors.length
      ? `; Cellpose's images are of the ${fl.image_odors.length} that all ${fl.listed} had (a fly without the others is predicted as by a model fit without them)` : ''}.` : '';
    return `<p class="small">${f.flies ? `Fit to ${f.flies.train.length} flies, ${o.held_out} held out, ${o.epochs} epochs${o.atlas
      ? `, with the atlas${refs.length ? ` (and ${refs.join(' and ')}'s in it)` : ''}` : ''}.${odors}${roiSettings(o)}` : ''}
      Scored are the final ROIs (${esc(FINAL_NAMES[r.final] || r.final)}), against each held-out fly's own${r.mean.recall_2d == null ? ', as 3D objects'
      : ": plane by plane (each glomerulus's outline on a plane, as ROIs are drawn), then as 3D objects (its outlines on all its planes as one)"}.
      ${partlyDrawn(r) ? `As glomeruli are often drawn on their clearest planes only, nothing counts against an ROI where nothing is drawn:
        on planes its glomerulus has no outline on, only its part in another glomerulus's outline does, and ROIs lying mostly
        outside every drawn glomerulus are left out of Right.` : `Scored strictly, as before that changed: an ROI's parts on planes its
        glomerulus has no outline on count against it, as does every ROI where no glomerulus is drawn, so these are lower than a newer
        fit's would be.`}</p>
      <table class="odor-table"><thead><tr><th>Held-out fly</th><th>ROIs</th>${['Found', 'Right', 'Dice'].map((h) => `<th>${h}${r.mean.recall_2d == null
        ? '' : ' <span class="muted">(per plane · 3D)</span>'}</th>`).join('')}
      <th title="Signal from neighbors in an ROI's border (its manual ROIs' in brackets)">Leak <span class="muted">(manual)</span></th>${PURITY_COLUMNS.map((c) => `<th title="${esc(c[1])}">${c[0]} <span class="muted">(manual)</span></th>`).join('')}<th>Named</th></tr></thead><tbody>${flies}</tbody></table>
      ${missed.length ? `<p class="small">Glomeruli not found (as 3D objects) in every held-out fly with one: ${missed.map((g) => `${esc(g.glomerulus)}
        <span class="muted">${g.detected}/${g.flies}</span>`).join(', ')}.</p>` : '<p class="small">Every glomerulus of the held-out flies was found.</p>'}`;
  }

  async _action(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const id = Number(b.closest('[data-id]').dataset.id);
    const { $, api, msgBox, esc, askDialog } = this.h;
    if (b.dataset.act === 'details') {
      if (this.open.has(id)) this.open.delete(id); else this.open.add(id);
      this.render();
      return;
    }
    try {
      if (b.dataset.act === 'cancel') {
        await api(`/api/fits/${id}/cancel`, { method: 'POST' });
      } else if (b.dataset.act === 'delete') {
        const f = this.data.fits.find((x) => x.id === id);
        $('#delete-confirm-q').textContent = `Delete fit #${id}, ${f.name}'s?`;
        $('#delete-confirm-detail').textContent = 'Its model, its results and its log are deleted from the server, for everyone.';
        $('#delete-confirm-warn').replaceChildren();
        if (await askDialog($('#delete-confirm')) !== 'delete') return;
        await api(`/api/fits/${id}`, { method: 'DELETE' });
        this.open.delete(id);
      }
    } catch (err) {
      $('#fit-status').replaceChildren(msgBox('error', esc(err.message)));
    }
    await this.refresh();
  }

  async start() {
    const { $, api, msgBox, esc } = this.h;
    const name = this.h.getName();
    if (!name) { this.render(); return; }
    const atlas = $('#fit-atlas').checked;
    const options = { held_out: $('#fit-held-out').value, epochs: $('#fit-epochs').value, atlas,
      tentative: $('#fit-tentative').checked, refs: atlas && $('#fit-grabe').checked ? ['grabe2016'] : [],
      cellprob_threshold: $('#fit-cellprob').value, flow_threshold: $('#fit-flow').value, dilate: $('#fit-dilate').value,
      join_overlap: $('#fit-join-overlap').value, join_corr: $('#fit-join-corr').value, max_cost: $('#fit-max-cost').value,
      // (the odors ticked, of those the flies found have: all of them, if the flies are not found yet, for the server to say)
      require_odors: this._odorFlies() ? this._required(this._odorFlies()) : [...this.require],
      image_odors: $('#fit-image-shared')?.checked === false ? 'all' : 'shared' };
    $('#fit-start').disabled = true;
    try {
      const f = await api('/api/fits', { method: 'POST', json: { name, options } });
      $('#fit-status').replaceChildren(msgBox('ok', `Fit #${f.id} asked for: it is below while it waits and runs.`));
    } catch (err) {
      $('#fit-status').replaceChildren(msgBox('error', `Could not ask for a fit: ${esc(err.message)}`));
    }
    await this.refresh();
  }
}
