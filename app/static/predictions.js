// Predictions of a fly's glomerulus ROIs by a fit's model, in the AL segmentation tab
// (predictions.py, fit_worker.run_prediction): asking for one (a model, and a fly of the site's
// datasets or any on tensor), the one of the model and fly chosen, every one made, and viewing
// one in the movie player (player.js: the ROIs predicted and the fly's own, of the plane shown).
// Also fills in the example, under Fit a model, of running a downloaded model yourself.

import { both, fmtDuration } from './fits.js';

const STATUS_NAMES = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', interrupted: 'Interrupted' };
const STAGE_NAMES = { load: 'loading the fly', predict: 'predicting its ROIs' };
const OTHER = 'other'; // the fly menu's entry for a fly's folders entered by hand
const ACTIVE = new Set(['queued', 'running']);
const num2 = (v) => (v == null ? '—' : v.toFixed(2));
const trim = (s) => (s || '').trim().replace(/\/+$/, '');

export class PredictPanel {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, getName (the name in Settings, '' if none),
  // player (the movie player), flies (the tab's flies, as from /api/roi_training; null until found) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/predictions
    this.timer = null;
    const { $ } = h;
    $('#predict-model').addEventListener('change', () => this.renderFlies());
    $('#predict-fly').addEventListener('change', () => { this.flyChosen = $('#predict-fly').value; this._renderPair(); });
    for (const id of ['#predict-dir', '#predict-raw']) $(id).addEventListener('input', () => this._renderPair());
    $('#predict-start').addEventListener('click', () => this.start());
    $('#predict-view').addEventListener('click', () => { const p = this._pair(); if (p) this.view(p); });
    $('#predict-list').addEventListener('click', (e) => this._action(e));
  }

  // while the tab is shown: every 3 s while one is queued or running, else every 30 s
  async load() {
    await this.refresh();
    if (this.timer) return;
    let n = 0;
    this.timer = setInterval(() => {
      if (this.h.$('#tab-al-segmentation').hidden) return;
      n += 1;
      const busy = this.data?.predictions.some((p) => ACTIVE.has(p.status));
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
  }

  _model() { return this.data?.models.find((m) => String(m.id) === this.h.$('#predict-model').value) || null; }

  // a fly's name, as a fit's (fits.fly_names): its top folder on tensor, its date and number
  _name(f) { return `${f.dir.split('/')[0]}_${f.date}_${f.fly_num}`; }

  // the fly menu: the flies of the site's datasets (by uploader; with what the model chosen did
  // with each, and whether it has predicted it), then any other on tensor
  renderFlies() {
    const { $, esc } = this.h;
    const sel = $('#predict-fly'), flies = this.h.flies(), m = this._model();
    const preds = new Map((this.data?.predictions || []).filter((p) => m && p.fit_id === m.id).map((p) => [trim(p.tensor_dir), p.status]));
    const made = (f) => ({ done: ' · predicted', queued: ' · queued', running: ' · being predicted' }[preds.get(trim(f.dir))] || '');
    const groups = new Map();
    for (const f of flies || []) {
      const who = [...new Set(f.datasets.map((d) => d.uploader || '?'))].join(' + ');
      if (!groups.has(who)) groups.set(who, []);
      groups.get(who).push(f);
    }
    const role = (f) => (!m ? '' : m.held_out.includes(this._name(f)) ? ' · held out of it' : m.train.includes(this._name(f)) ? ' · trained on' : '');
    const opts = [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([who, fs]) => `<optgroup label="${esc(who)}'s flies">${fs
      .sort((a, b) => a.date.localeCompare(b.date) || a.fly_num - b.fly_num)
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
    const f = (this.h.flies() || []).find((x) => x.dir === v);
    return f ? { dir: f.dir, raw_dir: f.raw_dir } : null;
  }

  // the prediction of the model and fly chosen, if there is one (as far as can be told here: the
  // server knows a fly by its folder, however it was entered)
  _pair() {
    const m = this._model(), fly = this._fly();
    if (!m || !fly || !this.data) return null;
    return this.data.predictions.find((p) => p.fit_id === m.id && trim(p.tensor_dir) === fly.dir) || null;
  }

  // what is known of a prediction: when, by whom, on what, its ROIs, and their scores against the fly's own
  _about(p) {
    const { esc, fmtWhen } = this.h, r = p.result;
    if (p.status === 'queued') return `Queued, asked for by <b>${esc(p.name)}</b> ${esc(fmtWhen(p.created_at))}: waiting for the fit worker.`;
    if (p.status === 'running') {
      const pr = p.progress || {};
      return `Running: ${esc(STAGE_NAMES[pr.stage] || 'starting')}${pr.device ? `, on the ${esc(pr.device)}` : ''}, since ${esc(fmtWhen(p.started_at))}
        (asked for by <b>${esc(p.name)}</b>).`;
    }
    if (p.status !== 'done') {
      return `${STATUS_NAMES[p.status] || esc(p.status)} ${esc(fmtWhen(p.finished_at))} (asked for by <b>${esc(p.name)}</b>)${p.error
        ? `: ${esc(p.error)}` : ''}. <a href="/api/predictions/${p.id}/log" target="_blank" rel="noopener">Its log</a>.`;
    }
    const s = r.scores;
    return `Predicted ${esc(fmtWhen(p.finished_at))}, asked for by <b>${esc(p.name)}</b>, on the ${esc(r.device)} in ${fmtDuration(r.seconds)}:
      ${r.n} ROIs, ${r.named.length} of them named${r.named.length ? ` <span class="muted">(${esc(r.named.join(', '))})</span>` : ''}.${s
      ? ` Against the fly's own ${s.manual} ROIs${s.recall_2d == null ? '' : ` (${s.manual_2d} outlines on their planes)`}: found ${both(s, 'recall')},
        right ${both(s, 'precision')}, Dice ${both(s, 'dice', num2)}${s.recall_2d == null ? '' : ' <span class="muted">(per plane · as 3D objects)</span>'};
        named ${s.named}/${s.named_of} as the one they overlap.` : ''}`;
  }

  _renderPair() {
    const { $, msgBox } = this.h;
    $('#predict-paths').hidden = $('#predict-fly').value !== OTHER;
    const p = this._pair(), m = this._model(), fly = this._fly();
    $('#predict-view').disabled = !(p && p.status === 'done');
    $('#predict-start').disabled = !(m && fly && this.h.getName()) || (p && ACTIVE.has(p.status));
    $('#predict-status').replaceChildren(...(!m || !fly ? [] : p
      ? [msgBox(p.status === 'done' ? 'ok' : ACTIVE.has(p.status) ? 'info' : 'error', this._about(p))]
      : [msgBox('info', 'This model has not predicted this fly\'s ROIs yet.')]));
  }

  // every prediction, newest first
  _renderList() {
    const { $, esc, fmtWhen } = this.h, ps = this.data.predictions;
    $('#predict-list').innerHTML = !ps.length ? '' : `<h3>Predictions</h3><table class="odor-table predictions-table"><thead><tr>
      <th>Fly</th><th>Model</th><th>Asked for by</th><th>When</th><th>Status</th><th title="The ROIs predicted (named, of them); and against the fly's own, if it has them: found, and right (per plane · as 3D objects)">ROIs</th>
      <th></th></tr></thead><tbody>${ps.map((p) => {
      const r = p.result, s = r?.scores;
      const rois = r ? `${r.n} <span class="muted">(${r.named.length} named)</span>${s ? ` <span class="muted">· ${both(s, 'recall')}, ${both(s, 'precision')}</span>` : ''}` : '—';
      const status = `${STATUS_NAMES[p.status] || esc(p.status)}${p.status === 'running' && p.progress?.device ? ` <span class="muted">(${esc(p.progress.device)})</span>` : ''}${
        p.error ? ` <span class="muted" title="${esc(p.error)}">(${esc(p.error.slice(0, 60))}${p.error.length > 60 ? '…' : ''})</span>` : ''}`;
      return `<tr data-id="${p.id}"><td title="${esc(p.tensor_dir || p.fly_dir)}">${esc(p.fly)}</td>
        <td>#${p.fit_id}${p.fit_deleted ? ' <span class="muted">(deleted)</span>' : ''}</td><td>${esc(p.name)}</td>
        <td>${esc(fmtWhen(p.finished_at || p.created_at))}</td><td>${status}</td><td>${rois}</td>
        <td class="acts">${p.status === 'done' ? `<button type="button" class="btn small" data-act="view">View</button>
          <a class="btn small" href="/api/predictions/${p.id}/download" download>Download</a>` : ''}
          ${p.status !== 'queued' ? `<a class="btn small" href="/api/predictions/${p.id}/log" target="_blank" rel="noopener">Log</a>` : ''}
          ${p.status !== 'running' ? '<button type="button" class="btn small danger" data-act="delete">Delete</button>' : ''}</td></tr>`;
    }).join('')}</tbody></table>`;
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
    this.h.player().openPrediction(v, `Glomerulus ROIs of ${p.fly}, predicted by fit #${p.fit_id}'s model`);
  }

  // asks whether to view a prediction there is already of the model and fly, or make it again: 'view', 'rerun' or ''
  _askAgain(p) {
    const { $, fmtWhen, askDialog } = this.h;
    $('#predict-again-detail').textContent = `Fit #${p.fit_id}'s model predicted ${p.fly}'s ROIs ${fmtWhen(p.finished_at)}, as ${p.name} asked.
      Predicting them again replaces those.`;
    return askDialog($('#predict-again'));
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
    await this._post({ name, fit: m.id, ...fly, overwrite });
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
    const id = Number(b.closest('[data-id]').dataset.id), p = this.data.predictions.find((x) => x.id === id);
    const { $, api, msgBox, esc, askDialog } = this.h;
    if (b.dataset.act === 'view') { this.view(p); return; }
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
