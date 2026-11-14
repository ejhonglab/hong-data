// Signals extracted from a fly's movies for its ROIs, in the AL segmentation tab (extract.py,
// fit_worker.run_extraction): asking for them (a fly, which ROIs: those a model predicted, as
// they are or as edited in the viewer, or its own; and how one signal is made of an ROI drawn on
// several planes), and every extraction made, to download.

import { fmtDuration } from './fits.js';

const STATUS_NAMES = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', interrupted: 'Interrupted' };
const OTHER = 'other'; // the fly menu's entry for a fly's folder entered by hand
const ACTIVE = new Set(['queued', 'running']);
const trim = (s) => (s || '').trim().replace(/\/+$/, '');
// what each way of making one signal of an ROI's planes does (extract.MODES, in more words)
const MODE_NOTES = {
  best_plane: `As al_analysis: in each recording, the plane on which the ROI's largest response over that recording's trials is
    greatest. A glomerulus may so be read on different planes in different recordings of a fly, and the plane is chosen by the
    responses it is then used for, which favours a plane's noise.`,
  best_plane_fly: `The plane on which the ROI's largest response over the trials of all the fly's recordings is greatest, used in
    every one of them: the same pixels throughout the fly. Still chosen by the responses themselves.`,
  all_planes: `The mean over every voxel of the ROI, on all the planes it is drawn on: no plane is chosen, and more pixels mean
    less noise, but planes where it is dim, or overlapped by a neighbour, count as much as its best.`,
  weighted: `Each plane's trace divided by its mean over the recording (so that a brighter plane does not count more), then
    averaged with weights 1 / the variance of that plane's dF/F before the odors in that recording: the average of least noise,
    if the planes hold the same signal. Its traces are then of relative F, not F; dF/F and responses are as for the others.`,
};
const ROI_NAMES = { own: "the fly's own", predicted: 'as predicted', edited: 'as edited' };

export class ExtractPanel {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, getName (the name in Settings, '' if none), flies (the tab's
  // flies, as from /api/roi_training; null until found), predictions (those listed, as from /api/predictions) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/extractions
    this.timer = null;
    const { $ } = h;
    $('#extract-fly').addEventListener('change', () => { this.flyChosen = $('#extract-fly').value; this.roisChosen = null; this.renderRois(); });
    $('#extract-rois').addEventListener('change', () => { this.roisChosen = $('#extract-rois').value; this._renderStart(); });
    $('#extract-dir').addEventListener('input', () => this._renderStart());
    $('#extract-mode').addEventListener('change', () => this._renderMode());
    $('#extract-start').addEventListener('click', () => this.start());
    $('#extract-list').addEventListener('click', (e) => this._action(e));
  }

  // while the tab is shown: every 3 s while one is queued or running, else every 30 s
  async load() {
    await this.refresh();
    if (this.timer) return;
    let n = 0;
    this.timer = setInterval(() => {
      if (this.h.$('#tab-al-segmentation').hidden) return;
      n += 1;
      const busy = this.data?.extractions.some((x) => ACTIVE.has(x.status));
      if (busy || n % 10 === 0) this.refresh();
    }, 3000);
  }

  async refresh() {
    try { this.data = await this.h.api('/api/extractions'); } catch (e) {
      this.h.$('#extract-status').replaceChildren(this.h.msgBox('error', `Could not ask the server about extractions: ${this.h.esc(e.message)}`));
      return;
    }
    this.render();
  }

  render() {
    const { $, esc } = this.h;
    if (!this.data) return;
    const name = this.h.getName();
    $('#extract-who').innerHTML = name
      ? `Extractions say who asked for them: this one will be <b>${esc(name)}</b>'s (as set in <a href="#settings/name">Settings</a>).`
      : 'Enter your first name in <a href="#settings/name">Settings</a> first: extractions say who asked for them.';
    if (!this.menusMade) { // (once: the menus are not made again under whoever is choosing in them, at each refresh)
      this.menusMade = true;
      $('#extract-mode').innerHTML = Object.entries(this.data.modes).map(([k, label]) => `<option value="${esc(k)}">${esc(label)}</option>`).join('');
      this._renderMode();
      this.renderFlies();
    }
    this._renderStart();
    this._renderList();
  }

  _renderMode() {
    this.h.$('#extract-mode-note').textContent = (MODE_NOTES[this.h.$('#extract-mode').value] || '').replace(/\s+/g, ' ');
  }

  // the predictions done, newest first
  _predictions() { return (this.h.predictions() || []).filter((p) => p.status === 'done'); }

  // the predictions were asked of the server again: the menus made again, if what they say of them changed
  predictionsChanged() {
    const sig = JSON.stringify(this._predictions().map((p) => [p.id, p.finished_at, p.edited?.revision]));
    if (sig === this.predSig) return;
    this.predSig = sig;
    if (this.data) this.renderFlies();
  }

  // the fly menu: the flies of the site's datasets (by uploader), those others whose ROIs a model has predicted, then
  // any other on tensor; called again when the tab's flies, or the predictions, are found or change
  renderFlies() {
    const { $, esc } = this.h;
    const sel = $('#extract-fly'), flies = this.h.flies(), preds = this._predictions();
    const predicted = new Set(preds.map((p) => trim(p.tensor_dir)));
    const groups = new Map();
    for (const f of flies || []) {
      const who = [...new Set(f.datasets.map((d) => d.uploader || '?'))].join(' + ');
      if (!groups.has(who)) groups.set(who, []);
      groups.get(who).push(f);
    }
    const known = new Set((flies || []).map((f) => trim(f.dir)));
    const others = [...new Map(preds.filter((p) => p.tensor_dir && !known.has(trim(p.tensor_dir))).map((p) => [trim(p.tensor_dir), p])).values()];
    const opts = [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([who, fs]) => `<optgroup label="${esc(who)}'s flies">${fs
      .sort((a, b) => a.date.localeCompare(b.date) || a.fly_num - b.fly_num)
      .map((f) => `<option value="${esc(trim(f.dir))}">${esc(f.date)} fly ${f.fly_num}${predicted.has(trim(f.dir)) ? ' · ROIs predicted' : ''}</option>`).join('')}</optgroup>`);
    if (others.length) {
      opts.push(`<optgroup label="Other flies with predicted ROIs">${others.map((p) => `<option value="${esc(trim(p.tensor_dir))}">${
        esc(p.fly)}</option>`).join('')}</optgroup>`);
    }
    sel.innerHTML = `${opts.join('')}${flies ? '' : '<option value="" disabled>(the site\'s flies: looking on tensor…)</option>'}
      <option value="${OTHER}">Another fly on tensor: its folder…</option>`;
    const values = [...sel.options].filter((o) => !o.disabled).map((o) => o.value);
    // (the one chosen; else the fly whose ROIs were last predicted, or edited; else the first)
    const last = preds.find((p) => values.includes(trim(p.tensor_dir)));
    sel.value = values.includes(this.flyChosen) ? this.flyChosen : last ? trim(last.tensor_dir) : values[0];
    this.renderRois();
  }

  // the ROIs menu, for the fly chosen: those each model predicted for it (as edited in the viewer, if they were; and
  // as predicted), then its own, if it is a fly of the site's datasets (which have ROIs) or one entered by hand
  renderRois() {
    const { $, esc, fmtWhen } = this.h;
    const fly = $('#extract-fly').value, sel = $('#extract-rois');
    $('#extract-paths').hidden = fly !== OTHER;
    const opts = [];
    for (const p of fly === OTHER ? [] : this._predictions().filter((q) => trim(q.tensor_dir) === fly)) {
      const by = `fit #${p.fit_id}'s model`;
      if (p.edited) {
        opts.push([`edited:${p.id}`, `Predicted by ${by}, as edited in the viewer (last ${fmtWhen(p.edited.saved_at)}${
          p.edited.by ? ` by ${p.edited.by}` : ''})`]);
      }
      opts.push([`predicted:${p.id}`, `Predicted by ${by}, as predicted`]);
    }
    const site = (this.h.flies() || []).some((f) => trim(f.dir) === fly);
    if (site || fly === OTHER) opts.push(['own', "The fly's own: the RoiSet.zip of its recordings on tensor"]);
    sel.innerHTML = opts.length ? opts.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('')
      : '<option value="">(none: choose a fly)</option>';
    if (opts.some(([v]) => v === this.roisChosen)) sel.value = this.roisChosen;
    this._renderStart();
  }

  // what to ask for: { rois, prediction } or { rois: 'own', dir }, or null if nothing can be yet
  _asked() {
    const { $ } = this.h, fly = $('#extract-fly').value, [kind, id] = ($('#extract-rois').value || '').split(':');
    if (!kind) return null;
    if (kind !== 'own') return { rois: kind, prediction: Number(id) };
    if (fly !== OTHER) return { rois: 'own', dir: fly };
    const top = trim(this.data?.tensor_root), dir = trim($('#extract-dir').value);
    return dir ? { rois: 'own', dir: top && dir.startsWith(`${top}/`) ? dir.slice(top.length + 1) : dir } : null;
  }

  _renderStart() {
    this.h.$('#extract-start').disabled = !(this._asked() && this.h.getName() && this.data);
  }

  // when an extraction queued or running should end (its `eta`), and what that is estimated from
  _from(e) {
    return e.from ? `estimated from how long the last ${e.from === 1 ? 'one' : e.from} took for the size of their movies`
      : 'a rough guess: none has finished yet to estimate from';
  }

  // which ROIs an extraction is of
  _rois(x) {
    const r = x.rois;
    return r.kind === 'own' ? ROI_NAMES.own : `fit #${r.fit}'s, ${ROI_NAMES[r.kind]}`;
  }

  // every extraction, newest first
  _renderList() {
    const { $, esc, fmtWhen } = this.h, xs = this.data.extractions, alive = this.data.worker?.alive;
    const modes = this.data.modes;
    $('#extract-list').innerHTML = !xs.length ? '' : `<h3>Signals extracted</h3><table class="odor-table predictions-table"><thead><tr>
      <th>Fly</th><th>ROIs</th><th>Of several planes</th><th>Asked for by</th><th>When</th><th>Status</th><th>Signals</th><th></th></tr></thead><tbody>${xs.map((x) => {
      const r = x.result, pr = x.progress || {};
      const left = x.eta && alive ? ` <span class="muted" title="${esc(this._from(x.eta))}">· ${x.eta.until >= 1
        ? `about ${fmtDuration(x.eta.until)} left` : 'ending any moment'}</span>` : '';
      const at = x.status === 'running' && pr.recording ? ` <span class="muted">(${esc(pr.recording)}${pr.total
        ? `, ${Math.round(100 * pr.done / pr.total)}%` : ''})</span>` : '';
      const status = `${STATUS_NAMES[x.status] || esc(x.status)}${at}${left}${x.status === 'queued' && !alive
        ? ' <span class="muted">(no fit worker is running)</span>' : ''}${
        x.error ? ` <span class="muted" title="${esc(x.error)}">(${esc(x.error.slice(0, 80))}${x.error.length > 80 ? '…' : ''})</span>` : ''}`;
      const got = r ? `${r.rois} ROIs <span class="muted">(${r.glomeruli} named as glomeruli; ${r.roi_planes} outlines)</span>,
        ${r.recordings.length} recording${r.recordings.length === 1 ? '' : 's'}
        <span class="muted" title="${esc(r.recordings.map((c) => `${c.name}: ${c.panel}, ${c.trials} trials`).join('\n'))}">(${
        esc([...new Set(r.recordings.map((c) => c.panel))].join(', '))})</span>, in ${fmtDuration(r.seconds)}` : '—';
      return `<tr data-id="${x.id}"><td title="${esc(x.tensor_dir || x.fly_dir)}">${esc(x.fly)}</td><td>${esc(this._rois(x))}</td>
        <td title="${esc(modes[x.options.mode] || '')}">${esc({ best_plane: 'best plane, per recording', best_plane_fly: 'best plane, per fly',
        all_planes: 'all voxels', weighted: 'weighted by noise' }[x.options.mode] || x.options.mode)}</td>
        <td>${esc(x.name)}</td><td>${esc(fmtWhen(x.finished_at || x.created_at))}</td><td>${status}</td><td>${got}</td>
        <td class="acts">${x.status === 'done' ? `<a class="btn small" href="/api/extractions/${x.id}/download" download>Download</a>` : ''}
          ${x.status !== 'queued' ? `<a class="btn small" href="/api/extractions/${x.id}/log" target="_blank" rel="noopener">Log</a>` : ''}
          ${x.status !== 'running' ? '<button type="button" class="btn small danger" data-act="delete">Delete</button>' : ''}</td></tr>`;
    }).join('')}</tbody></table>`;
  }

  async start() {
    const { $, api, msgBox, esc } = this.h;
    const name = this.h.getName(), asked = this._asked();
    if (!name || !asked) { this._renderStart(); return; }
    $('#extract-start').disabled = true;
    try {
      const x = await api('/api/extractions', { method: 'POST', json: { name, mode: $('#extract-mode').value, ...asked } });
      const rate = this.data?.rate;
      const about = rate && x.bytes ? ` It should take about ${fmtDuration(rate.seconds_per_gb * x.bytes / 2 ** 30)} for its ${(x.bytes / 2 ** 30).toFixed(1)} GB
        of movies (${this._from(rate)}).` : '';
      $('#extract-status').replaceChildren(msgBox('ok', `Asked for: it is below while it waits and runs, then to download.${esc(about)}`));
    } catch (err) {
      $('#extract-status').replaceChildren(msgBox('error', `Could not ask for them: ${esc(err.message)}`));
    }
    await this.refresh();
  }

  async _action(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const id = Number(b.closest('[data-id]').dataset.id), x = this.data.extractions.find((y) => y.id === id);
    const { $, api, msgBox, esc, askDialog } = this.h;
    if (b.dataset.act === 'delete') {
      $('#delete-confirm-q').textContent = `Delete the signals extracted for ${x.fly} (ROIs: ${this._rois(x)})?`;
      $('#delete-confirm-detail').textContent = 'Their tables are deleted from the server, for everyone. They can be extracted again.';
      $('#delete-confirm-warn').replaceChildren();
      if (await askDialog($('#delete-confirm')) !== 'delete') return;
      try { await api(`/api/extractions/${id}`, { method: 'DELETE' }); } catch (err) {
        $('#extract-status').replaceChildren(msgBox('error', esc(err.message)));
      }
      await this.refresh();
    }
  }
}
