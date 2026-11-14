// Signals extracted from a fly's movies for its ROIs, in the AL segmentation tab (extract.py,
// fit_worker.run_extraction): asking for them (a fly, which ROIs: those a model predicted, as
// they are or as edited in the viewer, or its own; and how one signal is made of an ROI drawn on
// several planes; and the response window: how many volumes from the odor's onset a response is taken from, by default
// as many as the odor pulse can reach, which the server works out for each recording: extract.window), and every
// extraction made, to download.

import { fmtDuration } from './fits.js';
import { FlyMenu } from './flymenu.js';
import { tip } from './tips.js';

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
const vols = (n) => `${n} volume${n === 1 ? '' : 's'}`;
// recordings' response windows (each as extract.window gives it, with its `recording`) by how they were worked out:
// those of the same number of volumes, from a pulse and volumes of the same lengths, together
const windowGroups = (ws) => {
  const groups = new Map();
  for (const w of ws) {
    const key = JSON.stringify([w.volumes, w.reach, w.pulse_s, w.volume_s, Boolean(w.missing)]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(w);
  }
  return [...groups.values()];
};
// what recordings' response windows were worked out from, in words (for a tip): each working out once, as the server
// says it of the first recording it is of, under the recordings it is of
const windowHow = (ws) => windowGroups(ws).map((g) => `${g.map((w) => w.recording).join(', ')}: ${g[0].how}${
  g.length > 1 ? ` (This is ${g[0].recording}'s; the other${g.length > 2 ? 's are' : ' is'} alike, by ${g.length > 2 ? 'their' : 'its'} own files.)` : ''}`)
  .join('\n\n');
// an extraction's response window, for the list: of one done, what each recording's was (on hover: how it was worked
// out); of one waiting, what was asked for; of one asked for before there was a choice, al_analysis' 2
const windowCell = (x, esc) => {
  const ws = (x.result?.recordings || []).filter((c) => c.window).map((c) => ({ recording: c.name, ...c.window }));
  if (ws.length) {
    const ns = [...new Set(ws.map((w) => w.volumes))].sort((a, b) => a - b);
    return `${tip(esc(ns.length === 1 ? vols(ns[0]) : `${ns[0]} to ${ns[ns.length - 1]} volumes`), windowHow(ws))}${
      ws[0].set ? ' <span class="muted">(set)</span>' : ''}`;
  }
  const w = x.options.window;
  if (w === undefined) return vols(2);
  return w === 'pulse' ? '<span class="muted">those the odor pulse can reach</span>' : `${vols(w)} <span class="muted">(set)</span>`;
};

export class ExtractPanel {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, getName (the name in Settings, '' if none), flies (the tab's
  // flies, as from /api/roi_training; null until found), mine (the flies on tensor of whoever is named in Settings:
  // { name, flies }, as the Predict box has them), predictions (those listed, as from /api/predictions) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/extractions
    this.timer = null;
    const { $ } = h;
    this.listed = new Map(); // the menu's flies, by folder (`_flies`)
    this.menu = new FlyMenu($('#extract-fly'), { esc: h.esc, onPick: (v) => { this.flyChosen = v; this.roisChosen = null; this.renderRois(); } });
    $('#extract-last').addEventListener('click', () => {
      const p = this._lastPrediction();
      if (p && this.listed.has(trim(p.tensor_dir))) { this._chooseLast(p); this.renderRois(); }
    });
    $('#extract-rois').addEventListener('change', () => { this.roisChosen = $('#extract-rois').value; this._renderStart(); });
    for (const id of ['#extract-dir', '#extract-raw-dir', '#extract-window']) $(id).addEventListener('input', () => this._renderStart());
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

  // The flies to choose from, each { dir, raw_dir, date, fly_num, who (whose it is: null for another's
  // fly with predicted ROIs), manual (whether it has ROIs of its own), preds (the predictions done of
  // it, newest first), name }: those of the site's datasets (which have ROIs of their own), and those
  // in the folders of whoever is named in Settings that have ROIs of their own or predicted ones;
  // then any others with predicted ROIs.
  _flies() {
    const flies = this.h.flies() || [], mine = this.h.mine?.() || { name: '', flies: [] }, preds = this._predictions();
    const same = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase();
    const of = (dir) => preds.filter((p) => trim(p.tensor_dir) === dir);
    const out = new Map();
    const whos = new Set(flies.map((f) => [...new Set(f.datasets.map((d) => d.uploader || '?'))].join(' + ')));
    for (const f of flies) {
      out.set(trim(f.dir), { dir: trim(f.dir), raw_dir: f.raw_dir, date: f.date, fly_num: f.fly_num, manual: true, preds: of(trim(f.dir)),
        who: [...new Set(f.datasets.map((d) => d.uploader || '?'))].join(' + ') });
    }
    const mineAs = [...whos].find((w) => same(w, mine.name)) || mine.name;
    for (const f of mine.flies) {
      const dir = trim(f.dir), ps = of(dir);
      if (!out.has(dir) && (f.rois || ps.length)) out.set(dir, { dir, raw_dir: f.raw_dir, date: f.date, fly_num: f.fly_num, manual: f.rois, preds: ps, who: mineAs });
    }
    for (const p of preds) {
      const dir = trim(p.tensor_dir);
      if (p.tensor_dir && !out.has(dir)) out.set(dir, { dir, raw_dir: p.tensor_raw_dir, name: p.fly, manual: false, preds: of(dir), who: null });
    }
    return [...out.values()];
  }

  // the last prediction made: of those whoever is named in Settings asked for, or edited, if there are any (else of
  // all), the one predicted, or edited, last
  _lastPrediction() {
    const me = (this.h.getName() || '').trim().toLowerCase(), when = (p) => (p.edited?.saved_at > p.finished_at ? p.edited.saved_at : p.finished_at) || '';
    const preds = this._predictions().filter((p) => p.tensor_dir);
    const mine = preds.filter((p) => me && [p.name, p.edited?.by].some((n) => (n || '').trim().toLowerCase() === me));
    return (mine.length ? mine : preds).reduce((a, p) => (!a || when(p) > when(a) ? p : a), null);
  }

  // the fly menu: each fly with what ROIs it has ("manual": its own; "predictions, edited", or "predictions,
  // unchecked": none of its predictions was edited in the viewer), by whose it is (those of whoever is named in
  // Settings first; of each person's, those with edited predictions, then unchecked ones, then only their own
  // ROIs), then any other on tensor; called again when the tab's flies, that person's, or the predictions change
  renderFlies() {
    const { $ } = this.h, known = this.h.flies(), list = this._flies(), me = (this.h.getName() || '').trim().toLowerCase();
    this.listed = new Map(list.map((f) => [f.dir, f]));
    const edited = (f) => f.preds.some((p) => p.edited);
    const tags = (f) => [...(f.manual ? [{ text: 'manual', cls: 'tag-ok' }] : []), ...(!f.preds.length ? []
      : [edited(f) ? { text: 'predictions, edited', cls: 'tag-ok' } : { text: 'predictions, unchecked', cls: 'tag-warn' }])];
    const rank = (f) => (edited(f) ? 0 : f.preds.length ? 1 : 2);
    const parts = (f) => f.dir.split('/').slice(-2).join('/'); // (a fly's folder: <date>/<n>, as searched for)
    const item = (f) => ({ value: f.dir, label: f.date ? `${f.date} fly ${f.fly_num}` : f.name, tags: tags(f),
      search: `${f.date ? `${f.date}/${f.fly_num}` : ''} ${parts(f)}` });
    const by = new Map();
    for (const f of list) if (f.who != null) by.set(f.who, [...(by.get(f.who) || []), f]);
    const isMine = (who) => Boolean(me) && who.split(' + ').some((w) => w.trim().toLowerCase() === me);
    const groups = [...by].sort(([a], [b]) => isMine(b) - isMine(a) || a.localeCompare(b)).map(([who, fs]) => ({ label: `${who}'s flies`,
      items: fs.sort((a, b) => rank(a) - rank(b) || a.date.localeCompare(b.date) || a.fly_num - b.fly_num).map(item) }));
    const others = list.filter((f) => f.who == null);
    if (others.length) groups.push({ label: 'Other flies with predicted ROIs', items: others.map(item) });
    this.menu.set(groups, [{ value: OTHER, label: 'Another fly on tensor: its folder…' }], known ? '' : "The site's flies: looking on tensor…");
    // (the one chosen; else the last prediction's fly, with those ROIs; else the first)
    const values = this.menu.values(), last = this._lastPrediction();
    const chosen = values.includes(this.flyChosen), byDefault = !chosen && last && values.includes(trim(last.tensor_dir));
    this.menu.choose(chosen ? this.flyChosen : byDefault ? trim(last.tensor_dir) : values[0]);
    this.roisDefault = byDefault ? this._roisOf(last) : null; // (until a fly is chosen by hand, or by the button)
    const b = $('#extract-last'), f = last && this.listed.get(trim(last.tensor_dir));
    b.disabled = !f;
    b.title = f ? `Chooses ${last.fly}, with the ROIs fit #${last.fit_id}'s model predicted for it${last.edited ? ', as edited in the viewer' : ''}` : 'No prediction is done yet';
    $('#extract-last-fly').textContent = f ? (f.date ? `${f.date}/${f.fly_num}` : last.fly) : 'none yet';
    this.renderRois();
  }

  // the fly of a prediction chosen, with its ROIs: as edited, if they were, else as predicted
  _chooseLast(p) {
    this.flyChosen = trim(p.tensor_dir);
    this.roisChosen = this._roisOf(p);
    this.menu.choose(this.flyChosen);
  }

  // a prediction's entry in the ROIs menu: its ROIs as edited, if they were, else as predicted
  _roisOf(p) { return `${p.edited ? 'edited' : 'predicted'}:${p.id}`; }

  // the ROIs menu, for the fly chosen: those each model predicted for it (as edited in the viewer, if they were; and
  // as predicted), then its own, if it is a fly of the site's datasets (which have ROIs) or one entered by hand
  renderRois() {
    const { $, esc, fmtWhen } = this.h;
    const fly = this.menu.value, sel = $('#extract-rois');
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
    if (this.listed.get(fly)?.manual || fly === OTHER) opts.push(['own', "The fly's own: the RoiSet.zip of its recordings on tensor"]);
    sel.innerHTML = opts.length ? opts.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('')
      : '<option value="">(none: choose a fly)</option>';
    const want = this.roisChosen ?? this.roisDefault;
    if (opts.some(([v]) => v === want)) sel.value = want;
    this._renderStart();
  }

  // the response window entered: undefined for none (as many volumes as the odor pulse can reach), its number of
  // volumes, or null for what is not one
  _window() {
    const v = this.h.$('#extract-window').value.trim(), max = this.data?.window?.max || 100;
    if (!v) return this.h.$('#extract-window').validity.badInput ? null : undefined;
    return /^\d+$/.test(v) && Number(v) >= 1 && Number(v) <= max ? Number(v) : null;
  }

  // what to ask for: { rois, prediction } or { rois: 'own', dir, raw_dir (if known, or entered) }, with the `window`
  // entered (if one is), or null if nothing can be yet
  _asked() {
    const { $ } = this.h, fly = this.menu.value, [kind, id] = ($('#extract-rois').value || '').split(':');
    const win = this._window();
    if (!kind || win === null) return null;
    const opts = win === undefined ? {} : { window: win };
    if (kind !== 'own') return { rois: kind, prediction: Number(id), ...opts };
    if (fly !== OTHER) {
      const raw = this.listed.get(fly)?.raw_dir;
      return { rois: 'own', dir: fly, ...(raw ? { raw_dir: raw } : {}), ...opts };
    }
    const top = trim(this.data?.tensor_root);
    const rel = (p) => (top && p.startsWith(`${top}/`) ? p.slice(top.length + 1) : p);
    const dir = trim($('#extract-dir').value), raw = trim($('#extract-raw-dir').value);
    return dir ? { rois: 'own', dir: rel(dir), ...(raw ? { raw_dir: rel(raw) } : {}), ...opts } : null;
  }

  _renderStart() {
    this.h.$('#extract-start').disabled = !(this._asked() && this.h.getName() && this.data);
    this._previewWindow();
  }

  // under the response window's box: how many volumes that is in each of the fly's recordings, for what is chosen
  // (asked of the server a moment after the last change), with how each was worked out, on hover
  _previewWindow() {
    const { $, api, esc } = this.h, note = $('#extract-window-note');
    const asked = this._asked(), sig = JSON.stringify(asked) + (this._window() === null);
    if (sig === this.winSig || !this.data) return;
    this.winSig = sig;
    clearTimeout(this.winTimer);
    if (!asked) {
      note.innerHTML = this._window() === null
        ? `<span class="muted">A whole number of volumes from 1 to ${this.data.window?.max || 100}, or empty.</span>` : '';
      return;
    }
    this.winTimer = setTimeout(async () => {
      let html;
      try { html = this._windowNote(await api('/api/extractions/window', { method: 'POST', json: asked })); } catch (err) {
        html = `<span class="muted">How many volumes that is, is not known yet: ${esc(err.message)}</span>`;
      }
      if (sig === this.winSig) note.innerHTML = html;
    }, 250);
  }

  // the response windows an extraction would have (as from /api/extractions/window), in words
  _windowNote(res) {
    const { esc } = this.h, ws = res.windows, al = res.al_analysis;
    const recs = ws.length === 1 ? 'its recording' : `each of its ${ws.length} recordings`;
    const named = (g) => esc(g.map((w) => w.recording).join(', '));
    const how = (g, text) => tip(`<b>${text}</b>`, windowHow(g)); // (what to hover over for how it was worked out)
    const first = 'Responses will be taken from the first';
    if (res.window !== 'pulse') {
      const more = ws.filter((w) => w.reach > w.volumes);
      return `${first} ${how(ws, vols(ws[0].volumes))} from the odor's onset in ${recs}, as set (${ws[0].volumes === al
        ? 'as many as al_analysis takes' : `al_analysis takes ${al}`}).${more.length
        ? ` The odor pulse can reach more: ${esc(more.map((w) => `${w.reach} in ${w.recording}`).join(', '))}.` : ''}`;
    }
    const unknown = ws.filter((w) => w.missing);
    if (unknown.length === ws.length) {
      return `${first} ${how(ws, vols(al))} from the odor's onset in ${recs}, as al_analysis: how many the odor pulse can reach
        could not be worked out${res.raw_dir ? '' : " (the fly's raw data folder is not known)"}.`;
    }
    const versus = ws.some((w) => w.volumes !== al) ? `al_analysis takes ${al}` : 'as many as al_analysis takes';
    if (!unknown.length && new Set(ws.map((w) => w.volumes)).size === 1) {
      return `${first} ${how(ws, vols(ws[0].volumes))} from the odor's onset in ${recs}: as many as the odor pulse can reach (${versus}).`;
    }
    // (not the same in all: those of as many volumes together, those not worked out apart)
    const by = new Map();
    for (const w of ws) by.set(`${w.volumes} ${Boolean(w.missing)}`, [...(by.get(`${w.volumes} ${Boolean(w.missing)}`) || []), w]);
    return `${first} volumes from the odor's onset, as many as the odor pulse can reach: ${[...by.values()]
      .map((g) => `${how(g, g[0].volumes)} in ${named(g)}${g[0].missing ? ' (not worked out: as al_analysis)' : ''}`).join('; ')} (${versus}).`;
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
      <th>Fly</th><th>ROIs</th><th>Of several planes</th><th>Response window</th><th>Asked for by</th><th>When</th><th>Status</th><th>Signals</th><th></th></tr></thead><tbody>${xs.map((x) => {
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
        <td>${windowCell(x, esc)}</td><td>${esc(x.name)}</td><td>${esc(fmtWhen(x.finished_at || x.created_at))}</td><td>${status}</td><td>${got}</td>
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
