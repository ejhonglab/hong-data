// Movie player for a dataset's raw and motion-corrected movies on tensor, with the ImageJ
// ROIs drawn on them; or for a prediction of a fly's glomerulus ROIs (the AL segmentation tab),
// its images with the ROIs predicted and the fly's own. Planes are read on the server a few at a
// time (app/movies.py), and kept here while there is room, so only what is shown is ever read.
// A prediction's ROIs can be edited there (roi_edit.js): a copy of them, their outlines simplified
// and their points moved, added and deleted as ImageJ's are, saved at each change
// (predictions.save_edited); those predicted, and the fly's own, are only ever shown.

import { History, area, deletePoint, inside, movePoints, nearestPoint, simplify, splitPoint } from './roi_edit.js';
import { showIdentity, showNoIdentity } from './identity.js';

const TYPED = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array };
const CACHE_BYTES = 256 * 1024 * 1024;
const CHUNK_BYTES = 4 * 1024 * 1024; // planes asked for at once
const AHEAD = 2; // chunks fetched ahead while playing
const ROI_COLOR = '#ffd400';
const SELECTED_COLOR = '#00e5ff';
// ROIs of the planes next to the one shown: the one above (its number less one) and below
const ABOVE_COLOR = '#ff5050';
const BELOW_COLOR = '#4aa3ff';
// a prediction's ROIs, by where they are from (`openPrediction`)
const SOURCES = { predicted: { label: 'predicted', color: '#ff4fd8' }, manual: { label: "the fly's own (drawn by hand)", color: ROI_COLOR },
  edited: { label: 'predicted, as edited here', color: '#a394ff' } };
const HANDLE = 6; // the side of an edited ROI's points' handles, and how near one the pointer takes hold of it (CSS pixels)
const CLICK = 3; // how far the pointer may move between press and release for it to be a click, not a drag
const WHEEL_STEP = 40; // pixels of wheel scrolling (a trackpad's, added up) for one step of a slider
// the movies of recordings with none in the other folder, listed apart in each
const ONLY = { raw: 'Not in the motion-corrected data', mocorr: 'Not in the raw data' };
// brightness and contrast, as ImageJ's B&C window: steps of its sliders, bins of its
// histogram, and the fraction of pixels (1 / this) a histogram bin must have for Auto to
// keep it, halved at each press (more contrast) until the range found is empty
const SLIDER = 1000;
const HIST_BINS = 256;
const AUTO_THRESHOLD = 5000;

const sleep = (ms) => new Promise((r) => { setTimeout(r, ms); });

// a '#rrggbb' colour moved toward the grey of its luminance (by k, from 0 to 1)
const desaturate = (hex, k = 0.6) => {
  const rgb = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)), l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  return `rgb(${rgb.map((c) => Math.round(c + (l - c) * k)).join(',')})`;
};

export class MoviePlayer {
  // el: the player's <dialog>; api, esc, fmtBytes: the app's helpers
  constructor(el, { api, esc, fmtBytes }) {
    Object.assign(this, { el, api, esc, fmtBytes });
    this.$ = (sel) => el.querySelector(sel);
    this.movie = null; // as listed by /api/datasets/<id>/movies, with its info
    this.cache = new Map(); // `${t}|${z}|${c}` -> plane (most recently used last)
    this.cacheBytes = 0;
    this.pending = new Map(); // chunk key -> promise of its planes
    this.view = { scale: 1, ox: 0, oy: 0 };
    this.showRois = true;
    this.showNames = true;
    this.roiPlanes = 'adjacent'; // the ROIs of other planes drawn: 'current' (none), 'adjacent' or 'all'
    this.glomOnly = true; // only ROIs named as glomeruli (and tentative ones, if shown)
    this.showTentative = true;
    this.showUnnamed = true; // a prediction's ROIs the model could not name (numbered), which glomOnly leaves alone
    this.roiQuery = ''; // what the ROI list is searched for
    this.selectedRoi = null;
    this.hoverHandle = -1; // the point of the edited ROI chosen that the pointer is on
    this.mode = null; // 'prediction': a prediction's ROIs (`openPrediction`), else a dataset's movies
    this.showSource = { predicted: true, manual: true };
    this.canvas = this.$('.player-canvas');
    this.off = document.createElement('canvas');

    this.$('.player-close').addEventListener('click', () => el.close());
    el.addEventListener('close', () => this._unload());
    this.$('.player-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-movie]');
      if (b) this.load(this.movies[+b.dataset.movie]);
    });
    this.$('.player-play').addEventListener('click', () => this.toggle());
    this.$('.player-t').addEventListener('input', (e) => this.goTo(+e.target.value));
    this.$('.player-z').addEventListener('input', (e) => { this.z = +e.target.value; this._planeChanged(); });
    this.$('.player-max').addEventListener('change', (e) => { this.maxProj = e.target.checked; this._planeChanged(); });
    this.$('.player-c').addEventListener('change', (e) => { this.c = +e.target.value; this._planeChanged(); });
    this.$('.player-fps').addEventListener('change', (e) => { this.fps = Math.max(0.1, Math.min(200, +e.target.value || 10)); e.target.value = this.fps; });
    this.bc = { min: this.$('.player-bc-min'), max: this.$('.player-bc-max'), bright: this.$('.player-bc-bright'), contrast: this.$('.player-bc-contrast') };
    this.bc.min.addEventListener('input', (e) => { const v = this._fromSlider(+e.target.value); this._setLevels(v, Math.max(this.hi, v), 'min'); });
    this.bc.max.addEventListener('input', (e) => { const v = this._fromSlider(+e.target.value); this._setLevels(Math.min(this.lo, v), v, 'max'); });
    this.bc.bright.addEventListener('input', (e) => this._brightness(+e.target.value));
    this.bc.contrast.addEventListener('input', (e) => this._contrast(+e.target.value));
    this.$('.player-auto').addEventListener('click', () => this._autoContrast());
    this.$('.player-reset').addEventListener('click', () => this._resetContrast());
    this.showOdors = true;
    this.$('.player-odors').addEventListener('change', (e) => { this.showOdors = e.target.checked; this._paint(); });
    this.$('.player-fit').addEventListener('click', () => { this._fit(); this._paint(); });
    this.$('.player-show-rois').addEventListener('change', (e) => { this.showRois = e.target.checked; this._roiLegend(); this._paint(); });
    this.$('.player-roi-planes').addEventListener('change', (e) => { this.roiPlanes = e.target.value; this._roiLegend(); this._paint(); });
    this.$('.player-roi-glom').addEventListener('change', (e) => { this.glomOnly = e.target.checked; this._roisFiltered(); });
    this.$('.player-roi-tentative').addEventListener('change', (e) => { this.showTentative = e.target.checked; this._roisFiltered(); });
    this.$('.player-roi-unnamed').addEventListener('change', (e) => { this.showUnnamed = e.target.checked; this._roisFiltered(); });
    this.$('.player-roi-search').addEventListener('input', (e) => { this.roiQuery = e.target.value.trim().toLowerCase(); this._renderRoiList(this.roiPath); });
    this.$('.player-roi-names').addEventListener('change', (e) => { this.showNames = e.target.checked; this._paint(); });
    this.$('.player-roi-sources').addEventListener('change', (e) => {
      const s = e.target.dataset.source;
      if (!s) return;
      this.showSource[s] = e.target.checked;
      this._roisFiltered();
    });
    this.$('.player-roi-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-roi]');
      if (!b) return;
      this._selectRoi(+b.dataset.roi);
    });
    this.edit = null; // a prediction's ROIs as edited here: { id, revision, hist, name, ... } (`openPrediction`)
    this.base = null; // the ROIs read for the movie shown: { predicted, manual }
    this.$('.player-simplify').addEventListener('click', () => this._simplifyAll());
    this.$('.player-simplify-tol').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this._simplifyAll(); } });
    this.$('.player-edit-copy').addEventListener('click', () => { if (this.edit?.hist && !this.edit.hist.current) this._commit(this._copyPredicted(null)); });
    this.$('.player-edit-discard').addEventListener('click', () => this._discard());
    this.$('.player-roi-download').addEventListener('click', () => this._downloadRois());
    this.$('.player-undo').addEventListener('click', () => this._undo());
    this.$('.player-redo').addEventListener('click', () => this._undo(true));
    el.addEventListener('keydown', (e) => this._key(e));
    el.addEventListener('wheel', (e) => this._wheelSlider(e), { passive: false });
    this._wireView();
    new ResizeObserver(() => {
      this._sizeCanvas();
      if (this.fitted) this._fit(); else this._clampView();
      this._paint();
    }).observe(this.$('.player-stage'));
  }

  // Opens the player on a dataset's movies of one `kind` ('raw' or 'mocorr'), showing the first.
  async open(datasetId, title, kind) {
    this._setMode(null);
    this.datasetId = datasetId;
    this.$('.player-title').textContent = `${kind === 'raw' ? 'Raw' : 'Motion-corrected'} movies of ${title}`;
    this.$('.player-list').innerHTML = '<p class="muted">Looking for movies…</p>';
    this._clearMovie('Choose a movie on the left.');
    if (!this.el.open) this.el.showModal();
    let res;
    try { res = await this.api(`/api/datasets/${datasetId}/movies`); } catch (e) {
      this.$('.player-list').innerHTML = `<p class="player-error">${this.esc(e.message)}</p>`;
      return;
    }
    this.movies = [];
    const groups = [kind].map((k) => {
      const found = res[k];
      const item = (m) => {
        this.movies.push({ ...m, kind: k });
        return `<button type="button" class="player-item${m.unmatched ? ' unmatched' : ''}" data-movie="${this.movies.length - 1}" title="${this.esc(m.path)}">
          <span>${this.esc(m.label)}</span><span class="muted">${this.fmtBytes(m.size)}${m.rois ? ' · ROIs' : ''}${m.trials ? ' · odors' : ''}</span>
          ${m.recordings ? `<span class="player-item-recs">${this.esc(this._registered(m))}</span>` : ''}</button>`;
      };
      const items = found.movies.filter((m) => !m.unmatched).map(item).join('');
      const only = found.movies.filter((m) => m.unmatched).map(item).join('');
      const none = found.error ? `no folder: ${found.error}` : 'none found';
      // the folders of flies not in the dataset in it, left out
      const left = found.skipped?.length ? `<p class="muted small">Left out: the folders of flies not in this dataset (${
        this.esc(found.skipped.map((f) => `${f.date} fly ${f.fly_num}`).join(', '))}).</p>` : '';
      return `${left}${items || only ? items : `<p class="muted small">${this.esc(none)}</p>`}`
        + (only ? `<h4 class="player-unmatched">${ONLY[k]}</h4>${only}` : '');
    });
    this.$('.player-list').innerHTML = groups.join('');
    const first = this.movies[0];
    if (first) this.load(first);
    else this._clearMovie(`No ${kind === 'raw' ? 'raw' : 'motion-corrected'} movies were found for this dataset.`);
  }

  // Opens the player on a prediction of a fly's glomerulus ROIs (as from /api/predictions/<id>/view):
  // its images (the fly's average image, as the model saw it, and its motion-corrected movies), each
  // with the ROIs predicted and the fly's own, in colours of their own, only those of the plane shown.
  // The predicted ROIs can be edited (a copy of them: see "editing a prediction's ROIs" below), as
  // `name()` (who: the name set in Settings, if any), `onSaved` called after each save.
  openPrediction(view, title, { name, onSaved } = {}) {
    this._setMode('prediction');
    this.edit = { id: view.id, revision: view.revision || 0, saved: view.edited || null, hist: null, name, onSaved, drag: null,
      editedPath: view.edited?.path || null };
    this.datasetId = null;
    this.$('.player-title').textContent = title;
    this._clearMovie('Choose an image on the left.');
    if (!this.el.open) this.el.showModal();
    this.movies = view.movies.map((m) => ({ ...m, kind: 'mocorr' }));
    this.$('.player-list').innerHTML = this.movies.length ? this.movies.map((m, i) => `<button type="button" class="player-item"
      data-movie="${i}" title="${this.esc(m.path)}"><span>${this.esc(m.label)}</span><span class="muted">${this.fmtBytes(m.size)}${
      m.rois ? " · the fly's own ROIs" : ''}${m.trials ? ' · odors' : ''}</span>${m.recordings
      ? `<span class="player-item-recs">${this.esc(this._registered(m))}</span>` : ''}</button>`).join('')
      : '<p class="muted small">No image of this fly was found.</p>';
    if (this.movies[0]) this.load(this.movies[0]);
  }

  // a dataset's movies (null), or a prediction's ('prediction': only the ROIs of the plane shown,
  // none on a max projection; those the model could not name, numbered, under a box of their own)
  _setMode(mode) {
    this.mode = mode;
    const pred = mode === 'prediction';
    this.el.classList.toggle('player-prediction', pred);
    this.roiPlanes = pred ? 'current' : this.$('.player-roi-planes').value;
    this.glomOnly = this.showTentative = this.showUnnamed = true;
    for (const c of ['glom', 'tentative', 'unnamed']) this.$(`.player-roi-${c}`).checked = true;
    this.roiQuery = this.$('.player-roi-search').value = '';
    this.showSource = { predicted: true, manual: true, edited: true };
    if (!pred) this.edit = null;
    this.base = null;
    this.hoverHandle = -1;
  }

  // the recordings in a fly's movie of them registered together, as said in the list
  _registered(m) {
    const n = m.recordings.length;
    return n ? `${n} recording${n === 1 ? '' : 's'} registered together: ${m.recordings.join(', ')}`
      : 'its recordings registered together (which: not known)';
  }

  async load(movie) {
    this.stop();
    this._resetCache();
    const id = (this.loadId = (this.loadId || 0) + 1);
    this.el.querySelectorAll('.player-item').forEach((b) => b.classList.toggle('active', this.movies[+b.dataset.movie] === movie));
    this._clearMovie('Opening the movie…');
    let info;
    try { info = await this.api(`/api/movie?path=${encodeURIComponent(movie.path)}`); } catch (e) {
      if (id === this.loadId) this._clearMovie(`Could not open it: ${e.message}`, true);
      return;
    }
    if (id !== this.loadId) return;
    this.movie = { ...movie, info };
    Object.assign(this, { t: 0, z: 0, c: 0, maxProj: false, rois: [], roiPath: null, selectedRoi: null, current: null, lo: null, hi: null,
      range: null, autoThreshold: 0, trials: null, base: null, hoverHandle: -1 });
    if (this.edit) this.edit.drag = null;
    this._renderEdit();
    this.fps = info.fps ? Math.round(info.fps * 10) / 10 : 10;
    const s = this.$('.player-t'), zs = this.$('.player-z');
    s.max = info.t - 1; s.value = 0;
    // (one time point, e.g. a mean over time: nothing to play, nor a time to choose)
    this.$('.player-trow').hidden = info.t < 2;
    this.$('.player-speed').hidden = info.t < 2;
    zs.max = info.z - 1; zs.value = 0;
    this.$('.player-zrow').hidden = info.z < 2;
    this.$('.player-max').checked = false;
    this.$('.player-c').innerHTML = Array.from({ length: info.c }, (_, i) => `<option value="${i}">${i + 1}</option>`).join('');
    this.$('.player-crow').hidden = info.c < 2;
    this.$('.player-fps').value = this.fps;
    const rate = info.fps ? ` · ${+info.fps.toFixed(2)} ${info.z > 1 ? 'volumes' : 'frames'}/s` : '';
    this.$('.player-sub').textContent = `${movie.label}${movie.recordings ? ` (${this._registered(movie)})` : ''}: `
      + `${info.t} time points × ${info.z} plane${info.z > 1 ? 's' : ''}`
      + `${info.c > 1 ? ` × ${info.c} channels` : ''}, ${info.x} × ${info.y} px, ${info.dtype}${rate}`;
    this.$('.player-warnings').textContent = info.warnings.join(' ');
    this.$('.player-msg').textContent = '';
    this._sizeCanvas();
    if (movie.rois || movie.predicted) this._loadRois(movie, id);
    else this._renderRoiList(null);
    this.$('.player-odors-row').hidden = true;
    if (movie.trials) this._loadTrials(movie.trials, id);
    const plane = await this._plane(0);
    if (id !== this.loadId || !plane) return;
    this.current = plane;
    this._fit();
    this._resetContrast(false);
    this._autoContrast();
    this._updateLabels();
  }

  // the movie's trials (al_analysis' trial_frames_and_odors.json), to name the odor of each over it
  async _loadTrials(path, id) {
    let res;
    try { res = await this.api(`/api/movie/trials?path=${encodeURIComponent(path)}`); } catch (e) {
      if (id === this.loadId) this.$('.player-msg').textContent = `Could not read its odors: ${e.message}`;
      return;
    }
    if (id !== this.loadId) return;
    this.trials = res.trials.slice().sort((a, b) => a.start - b.start);
    this.$('.player-odors-row').hidden = !this.trials.length;
    this.$('.player-odors-row').title = `From …/${path.split('/').slice(-2).join('/')}`;
    this._paint();
  }

  // the trial (and its index) time point t is in, if any
  _trialAt(t) {
    const tr = this.trials || [];
    let lo = 0, hi = tr.length - 1, found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (tr[mid].start <= t) { found = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return found >= 0 && t <= tr[found].end ? { trial: tr[found], i: found } : null;
  }

  // a movie's ROIs (its `rois`), and those predicted (`predicted`, a prediction's), each with its `source`
  async _loadRois(movie, id) {
    this.$('.player-roi-list').innerHTML = '<p class="muted small">Reading the ROIs…</p>';
    const read = async (path) => {
      if (!path) return [];
      const q = new URLSearchParams({ path, ...(this.datasetId != null ? { dataset: this.datasetId } : {}) });
      return (await this.api(`/api/movie/rois?${q}`)).rois;
    };
    // (a prediction's ROIs as edited: read once, when it is opened; then as they are here)
    const ed = this.mode === 'prediction' ? this.edit : null, first = ed && !ed.hist;
    let manual, predicted, edited;
    try { [manual, predicted, edited] = await Promise.all([read(movie.rois), read(movie.predicted), read(first && ed.saved?.path)]); } catch (e) {
      if (id === this.loadId) this.$('.player-roi-list').innerHTML = `<p class="player-error small">${this.esc(e.message)}</p>`;
      return;
    }
    if (id !== this.loadId) return;
    this.base = { predicted: predicted.map((r) => ({ ...r, source: 'predicted' })), manual: manual.map((r) => ({ ...r, source: 'manual' })) };
    if (first && !ed.hist) {
      const state = ed.saved && edited.length ? { simplified: ed.saved.simplified ?? null, pointsEdited: Boolean(ed.saved.points_edited),
        rois: edited.filter((r) => r.paths[0]?.length >= 3).map((r) => ({ name: r.name, z: r.z ?? r.position ?? null, kind: r.kind, points: r.paths[0] })) } : null;
      ed.hist = new History(state);
      if (state) this.showSource.predicted = false; // (the edited ones are shown in their place)
    }
    this._composeRois();
    this.roiPath = movie.rois || movie.predicted;
    this._renderRoiList(this.roiPath);
    this._renderDownload();
    this._renderEdit();
    this._roiLegend();
    this._paint();
  }

  // whether an ROI is listed and drawn: those not named as glomeruli only if asked for (by
  // the server: tentative ones, a glomerulus with _t0, _t1…, apart; and a prediction's unnamed ones)
  _visible(r) {
    if (this.mode === 'prediction' && !this.showSource[r.source]) return false;
    if (r.kind === 'tentative') return this.showTentative;
    if (r.kind !== 'other') return true;
    return this._predicted(r) ? this.showUnnamed : !this.glomOnly;
  }

  // whether an ROI is predicted (or a copy of one, edited), not one drawn by hand
  _predicted(r) { return r.source === 'predicted' || r.source === 'edited'; }

  // a prediction's legend: the colour of each source of ROIs, how many there are, and whether shown
  _renderSources() {
    const box = this.$('.player-roi-sources'), pred = this.mode === 'prediction';
    box.hidden = !pred;
    if (!pred) { box.innerHTML = ''; return; }
    const n = (s) => this.rois.filter((r) => r.source === s).length;
    // (the edited ones: only once there are any)
    box.innerHTML = Object.entries(SOURCES).map(([s, { label, color }]) => (s === 'edited' && !n(s) ? '' : `<label class="check"><input type="checkbox"
      data-source="${s}"${this.showSource[s] ? ' checked' : ''}><span class="swatch-roi" style="background:${color}"></span>
      ${this.esc(label)} <span class="muted">(${n(s)})</span></label>`)).join('');
  }

  _roisFiltered() {
    if (this.selectedRoi != null && !this._visible(this.rois[this.selectedRoi])) this.selectedRoi = null;
    this._renderRoiList(this.roiPath);
    this._paint();
  }

  _renderRoiList(path) {
    this._renderSources();
    // (counting only the sources of ROIs shown, so not both those predicted and their edited copies)
    const box = this.$('.player-roi-list'), pred = this.mode === 'prediction';
    const n = (f) => this.rois.filter((r) => (!pred || this.showSource[r.source]) && f(r)).length;
    const others = n((r) => r.kind === 'other' && !this._predicted(r)), tentative = n((r) => r.kind === 'tentative');
    const unnamed = n((r) => r.kind === 'other' && this._predicted(r));
    this.$('.player-roi-glom-n').textContent = path && others ? `(${others} ${this.glomOnly ? 'hidden' : 'not'})` : '';
    this.$('.player-roi-tentative-row').hidden = !(path && tentative);
    this.$('.player-roi-tentative-n').textContent = tentative ? `(${tentative}${this.showTentative ? '' : ' hidden'})` : '';
    this.$('.player-roi-unnamed-row').hidden = !(path && pred);
    this.$('.player-roi-unnamed-n').textContent = `(${unnamed}${unnamed && !this.showUnnamed ? ' hidden' : ''})`;
    this.$('.player-roi-search').hidden = !path;
    if (!path) { box.innerHTML = '<p class="muted small">No ImageJ ROIs (RoiSet.zip) next to this movie, or in a folder above it.</p>'; return; }
    // (a prediction's: those of the plane shown only, as drawn)
    const q = this.roiQuery, onPlane = (r) => this._visible(r) && !(pred && this._roiPlane(r) != null && this._roiPlane(r) !== this.z);
    const listed = (r) => onPlane(r) && (!q || String(r.name).toLowerCase().includes(q));
    const from = path.split('/').slice(-2).join('/'), shown = this.rois.filter(listed).length, all = this.rois.filter(onPlane).length;
    const found = q ? `${shown} of ${all} ROI${all === 1 ? '' : 's'} matching “${this.esc(q)}”` : '';
    const head = pred ? `${found || `${shown} ROI${shown === 1 ? '' : 's'}`} on plane ${this.z + 1}`
      : `${found || (shown === this.rois.length ? this.rois.length : `${shown} of ${this.rois.length}`)} from …/${this.esc(from)}`;
    box.innerHTML = `<p class="muted small" title="${this.esc(path)}">${head}</p>`
      + this.rois.map((r, i) => {
        if (!listed(r)) return '';
        const plane = this._roiPlane(r);
        const name = this.esc(r.name);
        return `<button type="button" class="player-roi${i === this.selectedRoi ? ' active' : ''}" data-roi="${i}">
          <span class="swatch-roi" style="background:${this.esc(this._roiColor(r))}"></span>${name}
          <span class="muted">${plane == null ? 'all planes' : `plane ${plane + 1}`}</span></button>`;
      }).join('');
  }

  // an ROI's colour: in a prediction, its source's; else its own (as set in ImageJ), if it has one
  _roiColor(r) { return this.mode === 'prediction' ? SOURCES[r.source].color : r.color || ROI_COLOR; }

  // which planes' ROIs are drawn, and how, when they are not all of the plane shown
  _roiLegend() {
    const box = this.$('.player-roi-legend'), info = this.movie?.info;
    if (this.mode === 'prediction') { box.textContent = info?.z > 1 && this.showRois ? 'The ROIs of the plane shown.' : ''; return; }
    if (!info || info.z < 2 || !this.showRois || !this.rois.some((r) => this._roiPlane(r) != null)) { box.textContent = ''; return; }
    if (this.maxProj) box.textContent = 'On the max projection: the ROIs of every plane.';
    else if (this.roiPlanes === 'adjacent') {
      const above = this.z > 0 ? `<span class="above">red: plane ${this.z} (above)</span>` : 'no plane above';
      const below = this.z < info.z - 1 ? `<span class="below">blue: plane ${this.z + 2} (below)</span>` : 'no plane below';
      box.innerHTML = `Dashed, ${above}; ${below}.`;
    } else box.textContent = this.roiPlanes === 'all' ? 'Dashed and paler: the ROIs of the other planes.' : '';
  }

  // the Z plane an ROI was drawn on: its hyperstack Z, or its stack position on a volume (as hong2p reads them)
  _roiPlane(r) {
    if (r.z != null) return r.z;
    return this.movie?.info.z > 1 && r.position != null && r.position < this.movie.info.z ? r.position : null;
  }

  _selectRoi(i) {
    this.selectedRoi = this.selectedRoi === i ? null : i;
    const plane = this.selectedRoi == null ? null : this._roiPlane(this.rois[i]);
    if (plane != null && !this.maxProj && plane !== this.z) {
      this.z = plane;
      this.$('.player-z').value = plane;
      this._planeChanged();
    } else this._paint();
    this.el.querySelectorAll('.player-roi').forEach((b) => b.classList.toggle('active', +b.dataset.roi === this.selectedRoi));
  }

  // ------------------------------------------- editing a prediction's ROIs
  // What is edited is a copy of the predicted ROIs, made when they are first simplified (or copied
  // as they are): `this.edit.hist` keeps its states to undo and redo through, each null (no copy:
  // those predicted are the ones shown) or { rois: [{ name, z, points, kind }], simplified (by
  // how many pixels, null if not), pointsEdited }. Each change is saved at once (`_save`).

  // the ROIs listed and drawn: those predicted, the fly's own, and the edited ones as they are now
  _composeRois() {
    const state = this.edit?.hist?.current, b = this.base || { predicted: [], manual: [] };
    const edited = (state?.rois || []).map((r, k) => ({ name: r.name, z: r.z, position: null, color: null, closed: true, type: 'polygon',
      kind: r.kind, paths: [r.points], source: 'edited', k }));
    this.rois = [...b.predicted, ...b.manual, ...edited];
    this.editedFrom = this.rois.length - edited.length; // (where the edited ones start)
  }

  // the edited ROI chosen (its handles shown, its points to drag), if it is on the plane shown: { roi, state }
  _chosenEdited() {
    const r = this.selectedRoi == null ? null : this.rois[this.selectedRoi];
    if (!r || r.source !== 'edited' || !this.showRois || !this._visible(r) || this.maxProj) return null;
    return this._roiPlane(r) == null || this._roiPlane(r) === this.z ? r : null;
  }

  // a copy of the predicted ROIs to edit, each outline simplified to within `tolerance` pixels
  // (null: as it is); a composite ROI's outlines become ROIs of their own
  _copyPredicted(tolerance) {
    const rois = [];
    for (const r of this.base.predicted) {
      for (const path of r.paths) {
        const points = tolerance == null ? path.map((p) => [p[0], p[1]]) : simplify(path, tolerance);
        if (points.length >= 3) rois.push({ name: r.name, z: this._roiPlane(r), kind: r.kind, points });
      }
    }
    return { rois, simplified: tolerance, pointsEdited: false };
  }

  _simplifyAll() {
    if (!this.edit?.hist || !this.base?.predicted.length) return;
    const box = this.$('.player-simplify-tol'), tol = Math.max(0, Math.min(20, Number.isFinite(+box.value) ? +box.value : 1));
    box.value = tol;
    this._commit(this._copyPredicted(tol));
  }

  // edited ROI k with `points` in place of its own, as a new state
  _withPoints(k, points) {
    const s = this.edit.hist.current;
    return { ...s, pointsEdited: true, rois: s.rois.map((r, i) => (i === k ? { ...r, points } : r)) };
  }

  // Discards the edited ROIs, once that is confirmed: they are deleted on the server at once, for
  // everyone, and only this viewer's Undo (while it stays open) brings them back.
  async _discard() {
    const ed = this.edit;
    if (!ed?.hist?.current || ed.stale) return;
    const dialog = document.getElementById('roi-discard-confirm');
    const answer = await new Promise((resolve) => {
      dialog.returnValue = '';
      dialog.addEventListener('close', () => resolve(dialog.returnValue), { once: true });
      dialog.showModal();
    });
    if (answer === 'discard' && ed === this.edit && ed.hist.current && !ed.stale) {
      this._commit(null);
      this.$('.player-undo').focus(); // (the button pressed is gone: Ctrl+Z still reaches the viewer)
    }
  }

  // The ROIs the download button gives: a prediction's as edited here, if they are, else as predicted;
  // else the movie's own (the fly's, drawn by hand). { path (on the server), what }, or null if none.
  _roisToDownload() {
    const m = this.movie, ed = this.mode === 'prediction' ? this.edit : null;
    if (!m) return null;
    if (ed?.hist?.current && ed.editedPath) return { path: ed.editedPath, what: 'the ROIs as edited here' };
    if (m.predicted && !ed?.hist?.current) return { path: m.predicted, what: 'the ROIs as predicted' };
    if (m.predicted) return null; // (edited, but not saved yet)
    return m.rois ? { path: m.rois, what: this.mode === 'prediction' ? "the fly's own ROIs" : 'these ROIs' } : null;
  }

  _renderDownload() {
    const b = this.$('.player-roi-download'), d = this._roisToDownload();
    b.hidden = !d && !(this.mode === 'prediction' && this.edit?.hist?.current);
    if (d) b.title = `Download ${d.what}, for ImageJ: ${d.path.split('/').pop()}`;
  }

  // saves the ROIs' file under its own name, once the last change to them is saved
  async _downloadRois() {
    const ed = this.edit;
    if (this.mode === 'prediction' && ed?.saving) await ed.saving;
    const d = this._roisToDownload();
    if (!d) { this._renderEdit({ text: 'Nothing to download: the edited ROIs are not saved.', error: true }); return; }
    const a = document.createElement('a');
    a.href = `/api/movie/rois/download?path=${encodeURIComponent(d.path)}`;
    a.download = '';
    this.el.appendChild(a);
    a.click();
    a.remove();
  }

  _commit(state) {
    const prev = this.edit.hist.current;
    this.edit.hist.push(state);
    this._stateChanged(prev, state);
  }

  _undo(redo = false) {
    const h = this.edit?.hist;
    if (!h || this.edit.drag || this.edit.stale || this.mode !== 'prediction' || !(redo ? h.canRedo : h.canUndo)) return;
    const prev = h.current;
    this._stateChanged(prev, redo ? h.redo() : h.undo());
  }

  // the edited ROIs changed (to `next`, from `prev`): shown, listed and saved. Those predicted are
  // hidden when the copy is made, and shown again when it is discarded; the ROI chosen stays so
  // (a predicted one's copy in its place).
  _stateChanged(prev, next) {
    const was = this.selectedRoi == null ? null : this.rois[this.selectedRoi];
    if (!prev !== !next) this.showSource.predicted = !next;
    this._composeRois();
    this.selectedRoi = null;
    if (was && next && (was.source === 'edited' || (was.source === 'predicted' && !prev))) {
      const k = was.source === 'edited' ? was.k : this.base.predicted.indexOf(was);
      // (a predicted ROI and its copy are in the same place, unless one of several outlines)
      if (k >= 0 && k < next.rois.length && next.rois[k].name === was.name) this.selectedRoi = this.editedFrom + k;
    } else if (was && was.source !== 'edited' && this._visible(was)) this.selectedRoi = this.rois.indexOf(was);
    if (this.selectedRoi === -1) this.selectedRoi = null;
    this._renderRoiList(this.roiPath);
    this._renderEdit();
    this._paint();
    this._save();
    this._renderDownload();
  }

  // the editing box: only for a prediction's ROIs, what can be done now, and how many points there are
  _renderEdit(status) {
    const box = this.$('.player-edit'), ed = this.edit, on = Boolean(this.mode === 'prediction' && ed?.hist && this.base?.predicted.length);
    box.hidden = !on;
    if (!on) return;
    const s = ed.hist.current, busy = Boolean(ed.stale);
    this.$('.player-edit-copy').hidden = Boolean(s);
    this.$('.player-edit-discard').hidden = !s;
    this.$('.player-undo').disabled = busy || !ed.hist.canUndo;
    this.$('.player-redo').disabled = busy || !ed.hist.canRedo;
    for (const c of ['.player-simplify', '.player-edit-copy', '.player-edit-discard']) this.$(c).disabled = busy;
    if (status !== undefined) ed.status = status;
    const points = (rois) => rois.reduce((n, r) => n + r.paths.reduce((m, p) => m + p.length, 0), 0);
    const of = s ? `${points(this.rois.slice(this.editedFrom)).toLocaleString()} points in ${s.rois.length} ROIs (${
      points(this.base.predicted).toLocaleString()} as predicted)${s.simplified == null ? '' : `, simplified within ${s.simplified} px`}${
      s.pointsEdited ? ', points edited' : ''}. ` : '';
    const el = this.$('.player-edit-status');
    el.textContent = `${of}${ed.status?.text || ''}`;
    el.classList.toggle('player-error', Boolean(ed.status?.error));
  }

  // Saves the edited ROIs as they are now (or deletes them, if there are none), one save at a time, the
  // last state winning; stops for good if they were changed elsewhere meanwhile (the server says).
  _save() {
    const ed = this.edit;
    ed.dirty = true;
    if (!ed.saving) ed.saving = this._saveLoop(ed).finally(() => { ed.saving = null; });
  }

  async _saveLoop(ed) {
    const say = (text, error = false) => { ed.status = { text, error }; if (ed === this.edit) this._renderEdit(); };
    while (ed.dirty && !ed.stale) {
      ed.dirty = false;
      const s = ed.hist.current, name = ed.name?.() || '';
      say('Saving…');
      try {
        if (s) {
          const info = await this.api(`/api/predictions/${ed.id}/edited`, { method: 'PUT', json: { base: ed.revision, name,
            simplified: s.simplified, points_edited: s.pointsEdited, rois: s.rois.map((r) => ({ name: r.name, z: r.z, points: r.points })) } });
          ed.revision = info.revision;
          ed.editedPath = info.path || ed.editedPath;
        } else {
          const q = new URLSearchParams({ base: ed.revision, name });
          ed.revision = (await this.api(`/api/predictions/${ed.id}/edited?${q}`, { method: 'DELETE' })).revision;
        }
        say(`${s ? 'Saved' : 'Edited ROIs deleted'} ${new Date().toLocaleTimeString()}.`);
        if (ed === this.edit) this._renderDownload();
        ed.onSaved?.();
      } catch (e) {
        if (e.status === 409) {
          ed.stale = true;
          const by = e.body?.edited;
          say(`Not saved: these ROIs were changed elsewhere since they were opened here${by?.by ? ` (by ${by.by})` : ''}. Close this and view them again to edit those.`, true);
        } else {
          say(`Not saved (${e.message}): the next change is saved with this one.`, true);
          return;
        }
      }
    }
  }

  // --------------------------------------- what a predicted ROI's name cost
  // Whether the edited ROIs named as `r` are still the predicted ROI's outlines, point for point, on
  // every plane it has one (what a name cost is of the ROI as a 3D object, as predicted).
  _asPredicted(r) {
    const same = (a, b) => a.length === b.length && a.every((p, i) => Math.abs(p[0] - b[i][0]) < 0.011 && Math.abs(p[1] - b[i][1]) < 0.011);
    const mine = this.rois.filter((x) => x.source === 'edited' && x.name === r.name);
    const theirs = this.base.predicted.filter((x) => x.name === r.name)
      .flatMap((x) => x.paths.filter((p) => p.length >= 3).map((p) => ({ z: this._roiPlane(x) ?? null, p })));
    return mine.length === theirs.length && mine.every((m) => theirs.some((t) => t.z === (m.z ?? null) && same(m.paths[0], t.p)));
  }

  // Opens the box of what the name of ROI i (a predicted one, or its edited copy) cost (identity.js):
  // each term's value for the glomeruli that cost least, read once per prediction opened; or why
  // not: a copy no longer as predicted (the values are of the predicted outlines), or a prediction
  // made before they were kept.
  async _showIdentity(i) {
    const r = this.rois[i], ed = this.edit, dialog = document.getElementById('roi-identity');
    if (!ed || !r || r.source === 'manual' || !this.base) return;
    const copy = r.source === 'edited';
    const title = `${/^\d+$/.test(r.name) ? `Predicted ROI ${r.name}: not named` : `Predicted ROI named ${r.name}`}${copy ? ' (its copy to edit)' : ''}`;
    if (copy && !this._asPredicted(r)) {
      const s = ed.hist.current;
      showNoIdentity(dialog, title, `What this name cost cannot be shown for this ROI: it is a copy of the predicted ROI whose outline is no
        longer as predicted (${s.simplified ? `the copy was simplified within ${s.simplified} px${s.pointsEdited ? ', and points of it edited' : ''}`
    : 'a point of it was moved, added or deleted'}). The values were computed for the predicted outlines, on all the ROI's planes;
        computing them for the outlines as they are now needs the fly's odor responses and the atlas registered to the fly, which only
        predicting its ROIs again reads. The copy keeps the name the predicted ROI was given: tick “predicted” above the list and double-click
        that ROI's name
        on the image for the values it was named by.`);
      return;
    }
    ed.identity ??= this.api(`/api/predictions/${ed.id}/identity`);
    let terms;
    try { terms = await ed.identity; } catch (e) {
      ed.identity = null;
      if (ed === this.edit) showNoIdentity(dialog, title, `Could not read what this name cost: ${this.esc(e.message)}`);
      return;
    }
    if (ed !== this.edit || !this.el.open) return;
    const label = terms.rois ? terms.by_name[r.name] : null;
    if (!terms.rois) {
      showNoIdentity(dialog, title, `This prediction was made before what each name cost was kept with it. Predict this fly's ROIs again
        (<i>Predict its ROIs</i>, in the tab) to see it.`);
    } else if (label == null || !terms.rois[label]) showNoIdentity(dialog, title, 'This ROI is not among those the model named.');
    else showIdentity(dialog, `${title}${copy ? ', unchanged' : ''}`, terms, label, this.esc);
  }

  // the ROI whose name, drawn on the image, the pointer is on (a predicted one's, or its copy's): its
  // index, or null (the last drawn, if names overlap: the one on top)
  _nameAt(e) {
    const r = this.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const hit = (this.nameBoxes || []).findLast((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
    return hit ? hit.i : null;
  }

  // the pointer's place in the image, in pixels (as ROIs' points are)
  _imagePoint(e) {
    const r = this.canvas.getBoundingClientRect(), { scale, ox, oy } = this.view;
    return [(e.clientX - r.left - ox) / scale, (e.clientY - r.top - oy) / scale];
  }

  // the point of the edited ROI chosen that the pointer is on (within a handle of it): { roi, i }, or null
  _handleAt(e) {
    const roi = this.edit?.hist && !this.edit.stale ? this._chosenEdited() : null;
    if (!roi) return null;
    const [x, y] = this._imagePoint(e), i = nearestPoint(roi.paths[0], x, y, HANDLE / this.view.scale);
    return i < 0 ? null : { roi, i };
  }

  // the edited ROI of the plane shown that the pointer is in (the smallest, if in several): its index, or null
  _editedAt(e) {
    if (!this.edit?.hist?.current || !this.showRois || this.maxProj) return null;
    const [x, y] = this._imagePoint(e);
    let best = null, least = Infinity;
    this.rois.forEach((r, i) => {
      if (r.source !== 'edited' || !this._visible(r) || (this._roiPlane(r) != null && this._roiPlane(r) !== this.z)) return;
      if (inside(r.paths[0], x, y) && area(r.paths[0]) < least) { least = area(r.paths[0]); best = i; }
    });
    return best;
  }

  // whether the pointer is in the edited ROI chosen (which a drag from there moves)
  _inChosen(e) {
    const roi = this.edit?.hist && !this.edit.stale ? this._chosenEdited() : null;
    return Boolean(roi) && inside(roi.paths[0], ...this._imagePoint(e));
  }

  // A press on a point of the edited ROI chosen, as on a point of an ImageJ polygon: with Shift it
  // becomes two, with Alt (or Ctrl: window managers take Alt-click) it is deleted, else it is
  // dragged. A press inside that ROI, not on a point, takes hold of all of it, to move it as ImageJ
  // moves a selection (`whole`). Returns whether the press was on either.
  _pressHandle(e) {
    const hit = this._handleAt(e);
    if (!hit && this._inChosen(e)) {
      const roi = this._chosenEdited();
      this.edit.drag = { roi, whole: true, from: roi.paths[0], at: this._imagePoint(e), x: e.clientX, y: e.clientY, moved: false };
      this.canvas.setPointerCapture(e.pointerId);
      return true;
    }
    if (!hit) return false;
    const { roi, i } = hit, points = roi.paths[0];
    if (e.altKey || e.ctrlKey || e.metaKey) {
      const fewer = deletePoint(points, i);
      if (fewer) this._commit(this._withPoints(roi.k, fewer));
    } else if (e.shiftKey) this._commit(this._withPoints(roi.k, splitPoint(points, i).map(([x, y]) => [+x.toFixed(2), +y.toFixed(2)])));
    else {
      this.edit.drag = { roi, i, from: points[i], x: e.clientX, y: e.clientY, moved: false };
      this.canvas.setPointerCapture(e.pointerId);
    }
    return true;
  }

  // the point held (or the whole ROI) follows the pointer (within the image), the ROI redrawn as it goes
  _dragHandle(e) {
    const d = this.edit.drag, info = this.movie.info;
    if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < CLICK) return;
    d.moved = true;
    const [x, y] = this._imagePoint(e);
    if (d.whole) {
      d.roi.paths = [movePoints(d.from, x - d.at[0], y - d.at[1], info.x, info.y)];
      this._paint();
      return;
    }
    const points = d.roi.paths[0].slice();
    points[d.i] = [+Math.max(0, Math.min(info.x, x)).toFixed(2), +Math.max(0, Math.min(info.y, y)).toFixed(2)];
    d.roi.paths = [points];
    this._paint();
  }

  // the point let go: where it is now is a change (to undo, and saved), if it was moved at all
  _dropHandle() {
    const d = this.edit.drag;
    this.edit.drag = null;
    if (d.moved) this._commit(this._withPoints(d.roi.k, d.roi.paths[0]));
  }

  // the handles of the edited ROI chosen: a square on each of its points
  _paintHandles(ctx) {
    const roi = this.edit?.hist ? this._chosenEdited() : null;
    if (!roi) return;
    const { scale, ox, oy } = this.view, held = this.edit.drag?.roi === roi ? this.edit.drag.i : this.hoverHandle;
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    roi.paths[0].forEach(([x, y], i) => {
      ctx.fillStyle = i === held ? SELECTED_COLOR : '#fff';
      ctx.strokeStyle = '#000';
      ctx.fillRect(ox + x * scale - HANDLE / 2, oy + y * scale - HANDLE / 2, HANDLE, HANDLE);
      ctx.strokeRect(ox + x * scale - HANDLE / 2 + 0.5, oy + y * scale - HANDLE / 2 + 0.5, HANDLE - 1, HANDLE - 1);
    });
  }

  // --------------------------------------------------------------- planes
  _zKey() { return this.maxProj ? 'max' : this.z; }

  _resetCache() {
    this.cache.clear();
    this.cacheBytes = 0;
    this.pending.clear();
  }

  _chunk() {
    const i = this.movie.info, per = i.x * i.y * (TYPED[i.dtype]?.BYTES_PER_ELEMENT || 2) * (this.maxProj ? i.z : 1);
    return Math.max(1, Math.min(64, Math.floor(CHUNK_BYTES / per)));
  }

  // the plane at time t (of the plane / projection and channel shown), read from the server if need be
  async _plane(t, { z = this._zKey(), c = this.c } = {}) {
    const key = `${t}|${z}|${c}`;
    if (!this.cache.has(key)) {
      const n = this._chunk(), t0 = Math.floor(t / n) * n;
      await this._fetch(t0, n, z, c).catch((e) => { this.$('.player-msg').textContent = `Could not read the movie: ${e.message}`; });
      if (!this.cache.has(key)) await this._fetch(t, 1, z, c).catch(() => {});
    }
    const plane = this.cache.get(key);
    if (plane) { this.cache.delete(key); this.cache.set(key, plane); }
    return plane || null;
  }

  _fetch(t0, n, z, c) {
    const ck = `${t0}|${n}|${z}|${c}`, movie = this.movie;
    if (!this.pending.has(ck)) {
      const p = (async () => {
        const q = new URLSearchParams({ path: movie.path, t: t0, n, z, c });
        const res = await fetch(`/api/movie/planes?${q}`);
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.error || `${res.status} ${res.statusText}`);
        }
        const [k, h, w] = res.headers.get('x-shape').split(',').map(Number);
        const Ctor = TYPED[res.headers.get('x-dtype')] || Uint16Array;
        const buf = await res.arrayBuffer();
        if (movie !== this.movie) return;
        for (let i = 0; i < k; i++) this._keep(`${t0 + i}|${z}|${c}`, new Ctor(buf, i * h * w * Ctor.BYTES_PER_ELEMENT, h * w));
      })().finally(() => this.pending.delete(ck));
      this.pending.set(ck, p);
    }
    return this.pending.get(ck);
  }

  _keep(key, plane) {
    if (this.cache.has(key)) return;
    this.cache.set(key, plane);
    this.cacheBytes += plane.byteLength;
    for (const [k, p] of this.cache) {
      if (this.cacheBytes <= CACHE_BYTES) break;
      this.cache.delete(k);
      this.cacheBytes -= p.byteLength;
    }
  }

  _prefetch(t) {
    const n = this._chunk(), T = this.movie.info.t;
    for (let i = 1; i <= AHEAD; i++) {
      const t0 = (Math.floor(t / n) + i) * n % (Math.ceil(T / n) * n);
      if (t0 < T && !this.cache.has(`${t0}|${this._zKey()}|${this.c}`)) this._fetch(t0, n, this._zKey(), this.c).catch(() => {});
    }
  }

  async goTo(t) {
    if (!this.movie) return;
    this.t = Math.max(0, Math.min(this.movie.info.t - 1, t));
    this._updateLabels();
    const want = `${this.t}|${this._zKey()}|${this.c}`;
    const plane = await this._plane(this.t);
    if (plane && want === `${this.t}|${this._zKey()}|${this.c}`) { this.current = plane; this._render(); }
  }

  _planeChanged() {
    this.$('.player-z').disabled = this.maxProj;
    this._updateLabels();
    if (this.mode === 'prediction' && this.roiPath) this._renderRoiList(this.roiPath);
    this.goTo(this.t);
  }

  // -------------------------------------------------------------- playing
  toggle() { if (this.playing) this.stop(); else this.play(); }

  async play() {
    if (!this.movie || this.playing || this.movie.info.t < 2) return;
    this.playing = true;
    this.$('.player-play').textContent = '❚❚';
    this.$('.player-play').title = 'Pause (space)';
    const id = this.loadId;
    while (this.playing && id === this.loadId) {
      const started = performance.now(), next = (this.t + 1) % this.movie.info.t;
      this._prefetch(next);
      const plane = await this._plane(next);
      if (!this.playing || id !== this.loadId) break;
      if (plane) { this.t = next; this.current = plane; this._render(); this._updateLabels(); }
      const wait = 1000 / this.fps - (performance.now() - started);
      await sleep(Math.max(wait, 0));
    }
  }

  stop() {
    this.playing = false;
    this.$('.player-play').textContent = '▶';
    this.$('.player-play').title = 'Play (space)';
  }

  // ------------------------------------------------------------- drawing
  // the plane's lowest and highest values (NaN left out)
  _extent(p) {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < p.length; i++) {
      const v = p[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    return mn <= mx ? [mn, mx] : [0, 1];
  }

  // Reset, as ImageJ's: the sliders' range and the levels, the full range of the plane shown
  _resetContrast(render = true) {
    const p = this.current;
    if (!p) return;
    const [mn, mx] = this._extent(p);
    this.range = { lo: mn, hi: mx > mn ? mx : mn + 1 };
    this.autoThreshold = 0;
    if (render) this._setLevels(this.range.lo, this.range.hi);
    else { this.lo = this.range.lo; this.hi = this.range.hi; }
  }

  // Auto, as ImageJ's: from the plane's histogram (256 bins over its range), black and white
  // at the first bins from each end with more than 1 / autoThreshold of its pixels (bins
  // with over a tenth of them not counting), the threshold doubling at each press
  _autoContrast() {
    const p = this.current;
    if (!p) return;
    const [mn, mx] = this._extent(p), bin = (mx - mn) / HIST_BINS, hist = new Uint32Array(HIST_BINS);
    let n = 0;
    for (let i = 0; i < p.length; i++) {
      const v = p[i];
      if (v !== v) continue; // NaN
      hist[bin > 0 ? Math.min(HIST_BINS - 1, Math.floor((v - mn) / bin)) : 0]++;
      n++;
    }
    const limit = Math.floor(n / 10);
    this.autoThreshold = this.autoThreshold < 10 ? AUTO_THRESHOLD : this.autoThreshold / 2;
    const threshold = Math.floor(n / this.autoThreshold), over = (i) => (hist[i] > limit ? 0 : hist[i]) > threshold;
    let lo = 0, hi = HIST_BINS - 1;
    while (lo < HIST_BINS - 1 && !over(lo)) lo++;
    while (hi > 0 && !over(hi)) hi--;
    if (hi < lo) { this._resetContrast(); return; }
    let black = mn + lo * bin, white = mn + hi * bin;
    if (black === white) { black = mn; white = mx > mn ? mx : mn + 1; }
    if (!this.range) this.range = { lo: mn, hi: mx > mn ? mx : mn + 1 };
    this._setLevels(black, white);
  }

  _fromSlider(v) { return this.range.lo + (this.range.hi - this.range.lo) * v / SLIDER; }

  _toSlider(x) { return Math.round(Math.max(0, Math.min(1, (x - this.range.lo) / (this.range.hi - this.range.lo))) * SLIDER); }

  // brightness and contrast, as ImageJ's sliders: moving the middle of the levels, and
  // their width (the middle slider position keeping the full range)
  _brightness(v) {
    const r = this.range, mid = r.lo + (r.hi - r.lo) * ((SLIDER - v) / SLIDER), w = this.hi - this.lo;
    this._setLevels(mid - w / 2, mid + w / 2, 'bright');
  }

  _contrast(v) {
    const r = this.range, mid = this.lo + (this.hi - this.lo) / 2, half = SLIDER / 2;
    const slope = v <= half ? v / half : half / (SLIDER - v);
    if (slope > 0) this._setLevels(mid - (r.hi - r.lo) / 2 / slope, mid + (r.hi - r.lo) / 2 / slope, 'contrast');
  }

  // black and white at lo and hi (from the slider moved, `from`, which is left where it is)
  _setLevels(lo, hi, from = null) {
    if (!this.range) return;
    this.lo = lo;
    this.hi = hi > lo ? hi : lo + (this.range.hi - this.range.lo) / SLIDER;
    const r = this.range, w = this.hi - this.lo, half = SLIDER / 2;
    const pos = {
      min: this._toSlider(this.lo), max: this._toSlider(this.hi),
      bright: Math.round(Math.max(0, Math.min(1, 1 - ((this.lo + w / 2) - r.lo) / (r.hi - r.lo))) * SLIDER),
      contrast: (() => { let c = ((r.hi - r.lo) / w) * half; if (c > half) c = SLIDER - (w / (r.hi - r.lo)) * half; return Math.round(Math.max(0, Math.min(SLIDER, c))); })(),
    };
    for (const [k, el] of Object.entries(this.bc)) if (k !== from) el.value = pos[k];
    this.$('.player-bc-min-out').textContent = this._fmtLevel(this.lo);
    this.$('.player-bc-max-out').textContent = this._fmtLevel(this.hi);
    this._render();
  }

  _fmtLevel(v) {
    const int = /int/.test(this.movie?.info.dtype || '');
    return int ? String(Math.round(v)) : String(+v.toPrecision(4));
  }

  // the histogram of the plane shown over the sliders' range, and how its values become grey levels
  _drawHist() {
    const c = this.$('.player-hist'), p = this.current, r = this.range, dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, c.clientWidth), h = Math.max(1, c.clientHeight);
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    this.$('.player-bc-range-lo').textContent = r ? this._fmtLevel(r.lo) : '';
    this.$('.player-bc-range-hi').textContent = r ? this._fmtLevel(r.hi) : '';
    if (!p || !r) return;
    const hist = new Uint32Array(HIST_BINS), k = HIST_BINS / (r.hi - r.lo);
    for (let i = 0; i < p.length; i++) {
      const b = Math.floor((p[i] - r.lo) * k);
      if (b >= 0 && b < HIST_BINS) hist[b]++;
      else if (p[i] === r.hi) hist[HIST_BINS - 1]++;
    }
    // scaled to the second-highest bin, so a single tall one (background) does not flatten the rest
    const sorted = Array.from(hist).sort((a, b) => b - a), top = Math.max(1, Math.min(sorted[0], (sorted[1] || sorted[0]) * 1.5));
    ctx.fillStyle = getComputedStyle(c).color;
    const bw = w / HIST_BINS;
    for (let i = 0; i < HIST_BINS; i++) {
      const bh = Math.min(h, (hist[i] / top) * h);
      if (bh > 0) ctx.fillRect(i * bw, h - bh, Math.max(bw, 1), bh);
    }
    const x = (v) => ((v - r.lo) / (r.hi - r.lo)) * w;
    ctx.strokeStyle = getComputedStyle(c).getPropertyValue('--line').trim() || '#000';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, h - 0.75);
    ctx.lineTo(x(this.lo), h - 0.75);
    ctx.lineTo(x(this.hi), 0.75);
    ctx.lineTo(w, 0.75);
    ctx.stroke();
  }

  // the plane, in grey levels from lo (black) to hi (white), into the offscreen canvas
  _render() {
    const p = this.current, info = this.movie?.info;
    if (!p || !info) return;
    const { x: w, y: h } = info;
    if (this.off.width !== w || this.off.height !== h || !this.img) {
      this.off.width = w;
      this.off.height = h;
      this.img = this.off.getContext('2d').createImageData(w, h);
      this.pixels = new Uint32Array(this.img.data.buffer);
    }
    const lo = this.lo, k = 255 / ((this.hi - lo) || 1), px = this.pixels;
    for (let i = 0; i < p.length; i++) {
      let v = (p[i] - lo) * k;
      v = v < 0 ? 0 : v > 255 ? 255 : v | 0;
      px[i] = 0xff000000 | (v << 16) | (v << 8) | v;
    }
    this.off.getContext('2d').putImageData(this.img, 0, 0);
    this._drawHist();
    this._paint();
  }

  _sizeCanvas() {
    const stage = this.$('.player-stage'), dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(stage.clientWidth * dpr));
    this.canvas.height = Math.max(1, Math.round(stage.clientHeight * dpr));
  }

  // the least zoom: the image fit to the stage, filling it across or down
  _minScale() {
    const info = this.movie.info, stage = this.$('.player-stage');
    return Math.min(stage.clientWidth / info.x, stage.clientHeight / info.y) || 1;
  }

  _fit() {
    if (!this.movie?.info) return;
    this.view = { scale: this._minScale(), ox: 0, oy: 0 };
    this._clampView();
    this.fitted = true;
  }

  // the view kept from showing anything around the image: zoomed out no further than it
  // fits, centred across (or down) where it is narrower than the stage, else its edges
  // never inside the stage's
  _clampView() {
    if (!this.movie?.info) return;
    const info = this.movie.info, stage = this.$('.player-stage'), v = this.view;
    v.scale = Math.max(v.scale, this._minScale());
    const clamp = (o, size, room) => (size <= room ? (room - size) / 2 : Math.min(0, Math.max(room - size, o)));
    v.ox = clamp(v.ox, info.x * v.scale, stage.clientWidth);
    v.oy = clamp(v.oy, info.y * v.scale, stage.clientHeight);
  }

  // the plane (zoomed and moved) and the ROIs over it
  _paint() {
    const ctx = this.canvas.getContext('2d'), dpr = window.devicePixelRatio || 1, { scale, ox, oy } = this.view;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.nameBoxes = []; // where predicted ROIs' names (and their copies') are drawn, to click: `_nameAt`
    if (!this.current) return;
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
    ctx.drawImage(this.off, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._paintOdor(ctx);
    if (!this.showRois || !this.rois.length) return;
    ctx.font = '12px system-ui, sans-serif';
    this.rois.forEach((r, i) => {
      const style = this._roiStyle(r, i);
      if (!style) return;
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width;
      ctx.setLineDash(style.dash);
      ctx.globalAlpha = style.alpha;
      let cx = 0, cy = 0, n = 0;
      for (const path of r.paths) {
        ctx.beginPath();
        path.forEach(([x, y], k) => {
          const sx = ox + x * scale, sy = oy + y * scale;
          if (k) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy);
          cx += sx; cy += sy; n++;
        });
        if (r.closed) ctx.closePath();
        ctx.stroke();
      }
      if (this.showNames && n && style.named) {
        ctx.setLineDash([]);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.shadowColor = '#000';
        ctx.shadowBlur = 3;
        ctx.fillText(r.name, cx / n + 4, cy / n);
        ctx.shadowBlur = 0;
        if (this.mode === 'prediction' && r.source !== 'manual') {
          this.nameBoxes.push({ i, x: cx / n + 2, y: cy / n - 12, w: ctx.measureText(r.name).width + 4, h: 16 });
        }
      }
    });
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    this._paintHandles(ctx);
  }

  // how an ROI is drawn over the plane shown (null: not): those of the plane, or of every
  // plane, as they are; of the planes above and below (red, blue) or all others (paler),
  // dashed, as asked; the one chosen in the list as it is, wherever it is
  _roiStyle(r, i) {
    if (!this._visible(r)) return null;
    const plane = this._roiPlane(r), z = this.maxProj ? null : this.z, here = z == null || plane == null || plane === z;
    if (this.mode === 'prediction') { // (only those of the plane shown)
      if (this.maxProj || (plane != null && plane !== this.z)) return null;
      return i === this.selectedRoi ? { color: SELECTED_COLOR, width: 2.5, dash: [], alpha: 1, named: true }
        : { color: this._roiColor(r), width: 1.5, dash: [], alpha: 1, named: true };
    }
    if (i === this.selectedRoi) return { color: SELECTED_COLOR, width: 2.5, dash: here ? [] : [4, 4], alpha: 1, named: true };
    if (here) return { color: r.color || ROI_COLOR, width: 1.5, dash: [], alpha: 1, named: true };
    if (this.roiPlanes === 'all') return { color: desaturate(r.color || ROI_COLOR), width: 1.2, dash: [4, 4], alpha: 0.6, named: false };
    if (this.roiPlanes === 'adjacent' && Math.abs(plane - z) === 1) {
      return { color: plane < z ? ABOVE_COLOR : BELOW_COLOR, width: 1.2, dash: [4, 3], alpha: 0.75, named: false };
    }
    return null;
  }

  // the odor of the trial shown, at the top left: white from its first frame with odor, grey before
  _paintOdor(ctx) {
    const at = this.showOdors && this.movie ? this._trialAt(this.t) : null;
    if (!at) return;
    const on = this.t >= at.trial.onset;
    ctx.save();
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 4;
    ctx.textBaseline = 'top';
    ctx.font = '600 16px system-ui, sans-serif';
    ctx.fillStyle = on ? '#fff' : '#9a9a9a';
    ctx.fillText(at.trial.odor ?? '(odor not named)', 20, 10);
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(`trial ${at.i + 1} / ${this.trials.length} · ${on ? 'odor on' : 'before odor'}`, 20, 31);
    if (on) {
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#ff5a4f';
      ctx.beginPath();
      ctx.arc(10, 18, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  _updateLabels() {
    const info = this.movie?.info;
    if (!info) return;
    this.$('.player-t').value = this.t;
    const secs = info.fps ? ` · ${(this.t / info.fps).toFixed(2)} s` : '';
    this.$('.player-t-out').textContent = `${this.t + 1} / ${info.t}${secs}`;
    this.$('.player-z-out').textContent = this.maxProj ? 'max projection' : `${this.z + 1} / ${info.z}`;
    this._roiLegend();
  }

  _clearMovie(msg, error = false) {
    this.movie = null;
    this.current = null;
    this.rois = [];
    this.base = null;
    this._renderEdit();
    this.$('.player-sub').textContent = '';
    this.$('.player-warnings').textContent = '';
    this.$('.player-msg').textContent = msg;
    this.$('.player-msg').classList.toggle('player-error', error);
    this.$('.player-roi-list').innerHTML = '';
    this._renderSources();
    this.$('.player-roi-legend').textContent = '';
    this.$('.player-roi-glom-n').textContent = '';
    this.$('.player-roi-tentative-row').hidden = true;
    this.$('.player-roi-unnamed-row').hidden = true;
    this.$('.player-roi-search').hidden = true;
    this.$('.player-roi-download').hidden = true;
    this.$('.player-pixel').textContent = '';
    this.trials = null;
    this.range = null;
    this.$('.player-odors-row').hidden = true;
    this.$('.player-bc-min-out').textContent = '';
    this.$('.player-bc-max-out').textContent = '';
    this._drawHist();
    this._paint();
  }

  _unload() {
    this.stop();
    this.loadId = (this.loadId || 0) + 1;
    this._resetCache();
    this.movie = null;
  }

  // --------------------------------------------------- zooming and moving
  _wireView() {
    const c = this.canvas;
    c.addEventListener('wheel', (e) => {
      if (!this.movie) return;
      e.preventDefault();
      const r = c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, v = this.view;
      const f = Math.exp(-e.deltaY * 0.0015), scale = Math.max(this._minScale(), Math.min(64, v.scale * f)), k = scale / v.scale;
      this.view = { scale, ox: mx - (mx - v.ox) * k, oy: my - (my - v.oy) * k };
      this._clampView();
      this.fitted = this.view.scale <= this._minScale();
      this._paint();
    }, { passive: false });
    // a press on a point of the edited ROI chosen edits it, and one inside it moves it (`_pressHandle`);
    // elsewhere, dragging moves the view. A click chooses the edited ROI it is in (or none), or the ROI
    // whose name it is on; a double click on a name opens what the name cost (`_showIdentity`).
    let drag = null;
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !this.movie) return;
      if (this._pressHandle(e)) { e.preventDefault(); return; }
      drag = { x: e.clientX, y: e.clientY, ox: this.view.ox, oy: this.view.oy, moved: false };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', (e) => {
      if (this.edit?.drag) this._dragHandle(e);
      else if (drag) {
        if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) >= CLICK) drag.moved = true;
        this.view.ox = drag.ox + e.clientX - drag.x;
        this.view.oy = drag.oy + e.clientY - drag.y;
        this._clampView();
        this._paint();
      } else if (this.movie) { // (over a point that can be dragged, or an edited ROI that can be chosen)
        const hit = this._handleAt(e), at = hit ? hit.i : -1;
        c.style.cursor = hit ? 'crosshair' : this._nameAt(e) != null ? 'pointer' : this._inChosen(e) ? 'move' : this._editedAt(e) != null ? 'pointer' : '';
        if (at !== this.hoverHandle) { this.hoverHandle = at; this._paint(); }
      }
      this._readout(e);
    });
    c.addEventListener('pointerup', (e) => {
      // (a press inside the ROI chosen that did not move it is a click too)
      const held = this.edit?.drag, click = Boolean(this.movie) && (held ? held.whole && !held.moved : drag && !drag.moved);
      const named = click ? this._nameAt(e) : null;
      if (held) this._dropHandle();
      if (named != null) { // (a click on an ROI's name: the ROI chosen, if it is not)
        if (named !== this.selectedRoi) this._selectRoi(named);
      } else if (click && this.edit?.hist?.current) {
        const i = this._editedAt(e), chosen = this.selectedRoi != null && this.rois[this.selectedRoi]?.source === 'edited';
        if (i != null ? i !== this.selectedRoi : chosen) this._selectRoi(i ?? this.selectedRoi);
      }
      drag = null;
    });
    c.addEventListener('pointercancel', () => { drag = null; if (this.edit?.drag) this._dropHandle(); });
    c.addEventListener('pointerleave', () => { this.$('.player-pixel').textContent = ''; });
    c.addEventListener('dblclick', (e) => {
      const named = this.movie ? this._nameAt(e) : null;
      if (named != null) { this._showIdentity(named); return; }
      if (this._handleAt(e) || this._editedAt(e) != null) return;
      this._fit();
      this._paint();
    });
  }

  // the wheel over a slider steps it: a step a notch (a trackpad's scrolling added up), up
  // or right for more; ten with Shift; the brightness and contrast sliders by a hundredth
  _wheelSlider(e) {
    const el = e.target.closest?.('input[type=range]');
    if (!el || el.disabled) return;
    e.preventDefault();
    const px = (e.deltaY || e.deltaX) * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
    if (this.wheel?.el !== el || Math.sign(this.wheel.px) !== Math.sign(px)) this.wheel = { el, px: 0 };
    this.wheel.px += px;
    if (Math.abs(this.wheel.px) < WHEEL_STEP) return;
    const dir = this.wheel.px < 0 ? 1 : -1;
    this.wheel.px = 0;
    const lo = +el.min, hi = +el.max, step = (el.closest('.player-bc') ? Math.max(1, Math.round((hi - lo) / 100)) : 1) * (e.shiftKey ? 10 : 1);
    const v = Math.max(lo, Math.min(hi, +el.value + dir * step));
    if (v === +el.value) return;
    el.value = v;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  // the value of the pixel under the pointer
  _readout(e) {
    const info = this.movie?.info, p = this.current;
    if (!info || !p) return;
    const r = this.canvas.getBoundingClientRect(), { scale, ox, oy } = this.view;
    const x = Math.floor((e.clientX - r.left - ox) / scale), y = Math.floor((e.clientY - r.top - oy) / scale);
    this.$('.player-pixel').textContent = x >= 0 && y >= 0 && x < info.x && y < info.y ? `x ${x}, y ${y}: ${+p[y * info.x + x].toFixed(3)}` : '';
  }

  _key(e) {
    if (!this.movie || e.target.matches('input, select, textarea')) return;
    if (e.ctrlKey || e.metaKey) { // undo and redo of the edited ROIs
      const k = e.key.toLowerCase(), redo = k === 'y' || (k === 'z' && e.shiftKey);
      if (redo || k === 'z') { e.preventDefault(); this._undo(redo); }
      return;
    }
    const keys = {
      ' ': () => this.toggle(), '\\': () => this.toggle(),
      ArrowRight: () => this.goTo(this.t + (e.shiftKey ? 10 : 1)), ArrowLeft: () => this.goTo(this.t - (e.shiftKey ? 10 : 1)),
      '>': () => this.goTo(this.t + 1), '<': () => this.goTo(this.t - 1),
      ArrowUp: () => this._stepZ(1), ArrowDown: () => this._stepZ(-1),
      p: () => {
        if (this.mode === 'prediction') return;
        this.maxProj = !this.maxProj;
        this.$('.player-max').checked = this.maxProj;
        this._planeChanged();
      },
      a: () => this._autoContrast(),
      r: () => this._resetContrast(),
      o: () => { this.showOdors = !this.showOdors; this.$('.player-odors').checked = this.showOdors; this._paint(); },
      f: () => { this._fit(); this._paint(); },
    };
    const f = keys[e.key];
    if (f) { e.preventDefault(); f(); }
  }

  _stepZ(d) {
    const info = this.movie.info;
    if (info.z < 2 || this.maxProj) return;
    this.z = Math.max(0, Math.min(info.z - 1, this.z + d));
    this.$('.player-z').value = this.z;
    this._planeChanged();
  }
}
