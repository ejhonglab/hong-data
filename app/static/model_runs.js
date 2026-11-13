// Baseline models run on datasets of glomeruli, in the Models tab (model_runs.py, run by
// model_worker.py one at a time): asking for one (a baseline of the Admin tab's, datasets and
// their panels, tuned or not), how the datasets would be scaled into spike rates (orn_input.py),
// the runs queued and running (with how far along), and those done: their spike counts opened
// in the tab's viewer (as a model's), and saved.

import { fmtDuration } from './fits.js';

const STATUS_NAMES = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', cancelled: 'Stopped', interrupted: 'Interrupted' };
const STAGE_NAMES = { inputs: 'scaling its responses into spike rates', model: 'running the model (fit_and_plot_mb_model)', outputs: 'reading what it wrote' };
const TUNING_NAMES = { fixed: 'as the baseline was tuned', tuned: 'tuned again on these odors' };
const ACTIVE = new Set(['queued', 'running']);
const KEY = 'olfdb.modelRuns'; // {baseline, datasets, tuning}: as last chosen, in this browser

export class ModelRuns {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, fmtSize, LS, getName (the name in Settings, '' if none),
  // setJobs(view) (the done runs' models, for the tab's viewer to open), onChange() (a run was done or deleted) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/models/runs
    this.preview = null; // as from /api/models/runs/preview, of the datasets (and panels) chosen
    this.panels = null; // the panels chosen (null: the datasets' default ones)
    this.timer = null;
    this._seq = 0;
    this.o = { baseline: null, datasets: [], tuning: null, ...h.LS.get(KEY, {}) };
    const { $ } = h;
    $('#mruns-baseline').addEventListener('change', (e) => { this.o.baseline = +e.target.value; this._save(); this.render(); });
    $('#mruns-datasets').addEventListener('change', () => {
      this.o.datasets = [...$('#mruns-datasets').querySelectorAll('input:checked')].map((i) => +i.value);
      this.panels = null;
      this._save();
      this._preview();
    });
    $('#mruns-panels').addEventListener('change', () => {
      this.panels = [...$('#mruns-panels').querySelectorAll('input:checked')].map((i) => i.value);
      this._preview();
    });
    $('#mruns-tuning').addEventListener('change', (e) => { this.o.tuning = e.target.value; this._save(); this._renderStart(); });
    $('#mruns-start').addEventListener('click', () => this.start());
    for (const id of ['#mruns-active', '#mruns-past']) $(id).addEventListener('click', (e) => this._action(e));
  }

  _save() { this.h.LS.set(KEY, this.o); }

  // while the tab is shown: every 3 s while one is queued or running, else every 30 s
  async load() {
    await this.refresh();
    if (this.timer) return;
    let n = 0;
    this.timer = setInterval(() => {
      if (this.h.$('#tab-models').hidden) return;
      n += 1;
      if (this.data?.runs.some((r) => ACTIVE.has(r.status)) || n % 10 === 0) this.refresh();
    }, 3000);
  }

  async refresh() {
    const { $, api, msgBox, esc } = this.h;
    let data;
    try { data = await api('/api/models/runs'); } catch (e) {
      $('#mruns-worker').replaceChildren(msgBox('error', `Could not ask the server about model runs: ${esc(e.message)}`));
      return;
    }
    const done = (d) => (d?.runs || []).filter((r) => r.status === 'done').map((r) => r.id).join(',');
    const changed = done(data) !== done(this.data);
    this.data = data;
    this.h.setJobs(data.view);
    if (changed) this.h.onChange();
    this.render();
    if (!this.preview) this._preview();
  }

  _baseline() { return this.data?.baselines.find((b) => b.id === this.o.baseline) || null; }

  render() {
    const { $, esc, msgBox } = this.h, d = this.data;
    if (!d) return;
    $('#model-runs-step').hidden = !d.baselines.length && !d.runs.length;
    const w = d.worker;
    $('#mruns-worker').replaceChildren(...(!w ? [msgBox('warn', `No model worker has ever started on this server, so nothing can be run yet
        (docker-compose.yml's <code>model</code> service: <code>restart-olfdb</code> builds and starts it).`)]
      : !w.alive ? [msgBox('warn', 'The model worker is not running: runs asked for wait until it is started again.')]
        : w.olfsysm?.problems?.length ? [msgBox('warn', `The model worker's olfsysm is not a clean build of its checkout: ${esc(w.olfsysm.problems.join('; '))}.`)] : []));
    // the baselines: those that cannot be run listed too, said why below
    const sel = $('#mruns-baseline');
    sel.innerHTML = d.baselines.map((b) => `<option value="${b.id}"${b.ok && b.plan.runnable === false ? ' class="muted"' : ''}>${esc(b.name ?? `row ${b.row + 1}`)}
      (${esc(b.experiment.split('/').pop())})${!b.ok || b.plan.runnable === false ? ' — cannot be run' : ''}</option>`).join('');
    if (!d.baselines.some((b) => b.id === this.o.baseline)) this.o.baseline = (d.baselines.find((b) => b.ok && b.plan.runnable) || d.baselines[0])?.id ?? null;
    sel.value = this.o.baseline ?? '';
    const b = this._baseline(), p = b?.ok ? b.plan : null;
    $('#mruns-why').replaceChildren(...(!b ? [] : !b.ok ? [msgBox('warn', `It cannot be read: ${esc(b.error)}.`)]
      : p.runnable === false ? [msgBox('warn', `<b>${esc(b.name)} cannot be run here</b>: ${p.why.map(esc).join('; ')}. Only baselines that
          pyolfsysm's <code>fit_and_plot_mb_model</code> can make from its own settings are run.`)] : []));
    // how it is tuned: what its params.csv row allows
    const tunings = p?.tunings?.length ? p.tunings : d.tunings;
    if (!tunings.includes(this.o.tuning)) this.o.tuning = tunings[0];
    $('#mruns-tuning').innerHTML = d.tunings.map((t) => `<label class="check"><input type="radio" name="mruns-tuning" value="${t}"${t === this.o.tuning ? ' checked' : ''}${tunings.includes(t)
      ? '' : ' disabled'}> ${t === 'fixed' ? `As the baseline was tuned <span class="muted">(its own threshold${p?.fixed ? ` ${(+p.fixed.fixed_thr.toPrecision(4))}` : ''} and APL weights: the same model, on this input)</span>`
      : `Tuned again on these odors <span class="muted">(to ${p?.target != null ? `${+(p.target * 100).toPrecision(3)}% of KC–odor pairs responding, as it was` : 'its response rate'})</span>`}</label>`).join('');
    // the datasets of glomeruli
    const have = new Set(d.datasets.map((x) => x.id));
    this.o.datasets = this.o.datasets.filter((i) => have.has(i));
    $('#mruns-datasets').innerHTML = d.datasets.length ? d.datasets.map((x) => `<label class="check"><input type="checkbox" value="${x.id}"${this.o.datasets.includes(x.id) ? ' checked' : ''}>
      #${x.id} ${esc(x.title)} <span class="muted">(${esc(x.uploader || '?')}, ${x.n_flies} flies)</span></label>`).join('')
      : '<span class="muted">No dataset of glomeruli is uploaded yet.</span>';
    this._renderPreview();
    this._renderLists();
  }

  // how the datasets chosen would be scaled, and their panels (asked again for each choice)
  async _preview() {
    const seq = ++this._seq, ids = this.o.datasets;
    if (!ids.length) { this.preview = null; this._renderPreview(); return; }
    let p;
    try { p = await this.h.api('/api/models/runs/preview', { method: 'POST', json: { datasets: ids, panels: this.panels } }); } catch (e) { p = { panels: [], chosen: [], scaling: null, error: e.message }; }
    if (seq !== this._seq) return;
    this.preview = p;
    this._renderPreview();
  }

  _renderPreview() {
    const { $, esc, msgBox } = this.h, p = this.o.datasets.length ? this.preview : null;
    $('#mruns-panels').innerHTML = !p ? '<span class="muted">Choose a dataset first.</span>' : p.panels.map((x) => `<label class="check"><input type="checkbox" value="${esc(x.name)}"${p.chosen.includes(x.name) ? ' checked' : ''}>
      ${esc(x.name || '(no panel)')} <span class="muted">(${x.n_odors} odors)</span></label>`).join('');
    const s = p?.scaling, box = $('#mruns-scaling');
    if (!p) box.replaceChildren();
    else if (p.error) box.replaceChildren(msgBox('warn', `<b>These cannot be run:</b> ${esc(p.error)}.`));
    else {
      const notes = [`The model's input: ${s.glomeruli.length} glomeruli × ${s.odors.length} odors (each glomerulus's mean response to each odor of the panels chosen, over the flies).`,
        esc(s.said), ...(s.dropped.length ? [`Left out, as not measured for every odor: ${esc(s.dropped.join(', '))}.`] : []),
        ...(s.mixtures.length ? [`Mixtures are left out (${s.mixtures.length}: ${esc(s.mixtures.slice(0, 4).join('; '))}${s.mixtures.length > 4 ? '; …' : ''}).`] : [])];
      box.replaceChildren(msgBox(s.method === 'hallem' ? 'info' : 'warn', notes.join(' ')));
    }
    this._renderStart();
  }

  _renderStart() {
    const { $, esc } = this.h, d = this.data, b = this._baseline(), name = this.h.getName();
    $('#mruns-who').innerHTML = name ? `Runs say who asked for them: this one will be <b>${esc(name)}</b>'s (as set in <a href="#settings/name">Settings</a>).`
      : 'Enter your first name in <a href="#settings/name">Settings</a> first: runs say who asked for them.';
    const ok = !!name && !!b?.ok && b.plan.runnable === true && !!this.o.datasets.length && !!this.preview?.scaling && !this.busy;
    $('#mruns-start').disabled = !ok;
    const waiting = d ? d.runs.filter((r) => ACTIVE.has(r.status)).length : 0;
    $('#mruns-queue').textContent = waiting ? `One model is run at a time: ${waiting} ${waiting === 1 ? 'is' : 'are'} queued or running, so this one would wait its turn.` : 'One model is run at a time.';
  }

  async start() {
    const { $, api, msgBox, esc } = this.h, out = $('#mruns-status');
    this.busy = true;
    this._renderStart();
    try {
      const r = await api('/api/models/runs', { method: 'POST', json: { name: this.h.getName(), baseline: this.o.baseline, datasets: this.o.datasets,
        panels: this.preview?.chosen ?? null, tuning: this.o.tuning } });
      out.replaceChildren(msgBox('ok', `Run #${r.id} is queued: it is listed below, with how far along it is.`));
    } catch (e) {
      out.replaceChildren(msgBox('error', `Not run: ${esc(e.message)}`));
    }
    this.busy = false;
    await this.refresh();
  }

  _who(r) { return `<b>${this.h.esc(r.name)}</b>, asked ${this.h.esc(this.h.fmtWhen(r.created_at))}`; }

  _what(r) {
    const { esc } = this.h, o = r.options, titles = new Map((this.data?.datasets || []).map((d) => [d.id, d.title]));
    return `<b>${esc(o.baseline.name)}</b> on ${o.datasets.map((i) => `<span title="${esc(titles.get(i) || 'deleted since')}">#${i}</span>`).join(', ')}
      <span class="muted">(${esc(o.panels.join(', '))}; ${esc(TUNING_NAMES[o.tuning] || o.tuning)})</span>`;
  }

  _renderLists() {
    const { $, esc, fmtWhen, fmtSize } = this.h, runs = this.data.runs;
    const active = runs.filter((r) => ACTIVE.has(r.status)).sort((a, b) => (a.status === 'running' ? -1 : b.status === 'running' ? 1 : a.position - b.position));
    $('#mruns-active').innerHTML = active.map((r) => {
      const p = r.progress;
      const stage = r.status === 'queued' ? `Waiting its turn${r.position ? ` (${r.position === 1 ? 'next' : `${r.position} in the queue`})` : ''}` : (STAGE_NAMES[p?.stage] || 'Starting');
      const eta = p?.ends_at ? `should end about ${esc(fmtWhen(p.ends_at))}` : p ? 'taking longer than runs like it have' : '';
      return `<div class="fit-active" data-id="${r.id}">
        <div class="fit-head"><span><b>Run #${r.id}</b> · ${this._what(r)} · ${this._who(r)}${r.started_at ? `, started ${esc(fmtWhen(r.started_at))}` : ''}</span>
          <button type="button" class="btn small" data-act="cancel" ${r.cancel ? 'disabled' : ''}>${r.cancel ? 'Stopping…' : r.status === 'queued' ? 'Cancel' : 'Stop'}</button></div>
        <div class="fit-stage">${esc(stage)}${p?.detail ? ` <span class="muted small mruns-detail">${esc(p.detail)}</span>` : ''}</div>
        <progress max="1" value="${p ? p.fraction : 0}" aria-label="Run #${r.id}: how far along"></progress>
        <div class="muted small">${eta}${eta ? ' <span title="From how long the last runs tuned alike took">(estimated)</span>' : ''}</div></div>`;
    }).join('');
    const past = runs.filter((r) => !ACTIVE.has(r.status));
    $('#mruns-past').innerHTML = past.length ? `<table class="odor-table fits-table"><thead><tr><th>Run</th><th>Model, on</th><th>Asked for by</th><th>Started</th><th>Took</th>
      <th>Result</th><th></th></tr></thead><tbody>${past.map((r) => {
    const took = r.started_at && r.finished_at ? (new Date(r.finished_at) - new Date(r.started_at)) / 1000 : null, x = r.result;
    const result = r.status === 'done' ? `${x.n_kcs.toLocaleString()} KCs × ${x.n_odors} odors, ${+(x.rate * 100).toPrecision(3)}% responding;
        input ${x.n_glomeruli} glomeruli, × ${+x.scaling.factor.toPrecision(4)} <span class="muted" title="${esc(x.scaling.said)}">(${x.scaling.method === 'hallem' ? 'by Hallem' : 'by all datasets, then Hallem'})</span>`
      : `${STATUS_NAMES[r.status] || esc(r.status)}${r.error ? `: <span class="mruns-error" title="${esc(r.error)}">${esc(r.error.split('\n')[0].slice(0, 160))}${r.error.length > 160 ? '…' : ''}</span>` : ''}`;
    return `<tr data-id="${r.id}"><td>#${r.id}</td><td>${this._what(r)}</td><td>${esc(r.name)}</td><td>${r.started_at ? esc(fmtWhen(r.started_at)) : '—'}</td>
        <td>${took == null ? '—' : fmtDuration(took)}</td><td>${result}</td>
        <td class="fit-acts">${r.status === 'done' ? `${r.run ? `<a class="btn small" href="#models/${encodeURIComponent(r.run)}" title="Its KCs' spike counts to each odor, below">Show</a>` : ''}
          <a class="btn small" href="/api/models/runs/${r.id}/download?what=csv" download title="Each KC's spike count to each odor, KCs as rows, as the model wrote it">Spike counts (CSV)</a>
          <a class="btn small" href="/api/models/runs/${r.id}/download" download title="Everything fit_and_plot_mb_model wrote (${x.plots} plots), its inputs and how they were scaled, the script that ran it, and its provenance">Everything (zip, ${fmtSize(x.bytes)})</a>` : ''}
          <button type="button" class="btn small danger" data-act="delete">Delete</button></td></tr>`;
  }).join('')}</tbody></table>` : '';
  }

  async _action(e) {
    const { api, askDialog, msgBox, esc, $ } = this.h;
    const b = e.target.closest('[data-act]'), id = +e.target.closest('[data-id]')?.dataset.id;
    if (!b || !id) return;
    const r = this.data.runs.find((x) => x.id === id);
    if (b.dataset.act === 'cancel') {
      try { await api(`/api/models/runs/${id}/cancel`, { method: 'POST' }); } catch (err) { $('#mruns-status').replaceChildren(msgBox('error', `Not stopped: ${esc(err.message)}`)); }
    } else if (b.dataset.act === 'delete') {
      $('#delete-confirm-q').textContent = `Delete model run #${id}?`;
      $('#delete-confirm-detail').innerHTML = r?.status === 'done' ? 'Its spike counts, plots and the rest of its files are deleted: save them first if they are needed (<i>Everything</i>).'
        : 'It and what it left are deleted.';
      $('#delete-confirm-warn').replaceChildren();
      if (await askDialog($('#delete-confirm')) !== 'delete') return;
      try { await api(`/api/models/runs/${id}`, { method: 'DELETE' }); } catch (err) { $('#mruns-status').replaceChildren(msgBox('error', `Not deleted: ${esc(err.message)}`)); }
    }
    this.refresh();
  }
}
