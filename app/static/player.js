// Movie player for a dataset's raw and motion-corrected movies on tensor, with the ImageJ
// ROIs drawn on them; or for a prediction of a fly's glomerulus ROIs (the AL segmentation tab),
// its images with the ROIs predicted and the fly's own. Planes are read on the server a few at a
// time (app/movies.py), and kept here while there is room, so only what is shown is ever read.

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
const SOURCES = { predicted: { label: 'predicted', color: '#ff4fd8' }, manual: { label: "the fly's own (drawn by hand)", color: ROI_COLOR } };
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
    this.selectedRoi = null;
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
    this.$('.player-roi-names').addEventListener('change', (e) => { this.showNames = e.target.checked; this._paint(); });
    this.$('.player-roi-sources').addEventListener('change', (e) => {
      const s = e.target.dataset.source;
      if (!s) return;
      this.showSource[s] = e.target.checked;
      this._roisFiltered();
    });
    this.$('.player-roi-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-roi]');
      if (b) this._selectRoi(+b.dataset.roi);
    });
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
  openPrediction(view, title) {
    this._setMode('prediction');
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
  // none on a max projection, not only glomeruli, as the model numbers those it cannot name)
  _setMode(mode) {
    this.mode = mode;
    const pred = mode === 'prediction';
    this.el.classList.toggle('player-prediction', pred);
    this.roiPlanes = pred ? 'current' : this.$('.player-roi-planes').value;
    this.glomOnly = !pred;
    this.$('.player-roi-glom').checked = this.glomOnly;
    this.showSource = { predicted: true, manual: true };
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
      range: null, autoThreshold: 0, trials: null });
    this.fps = info.fps ? Math.round(info.fps * 10) / 10 : 10;
    const s = this.$('.player-t'), zs = this.$('.player-z');
    s.max = info.t - 1; s.value = 0;
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
    let manual, predicted;
    try { [manual, predicted] = await Promise.all([read(movie.rois), read(movie.predicted)]); } catch (e) {
      if (id === this.loadId) this.$('.player-roi-list').innerHTML = `<p class="player-error small">${this.esc(e.message)}</p>`;
      return;
    }
    if (id !== this.loadId) return;
    this.rois = [...predicted.map((r) => ({ ...r, source: 'predicted' })), ...manual.map((r) => ({ ...r, source: 'manual' }))];
    this.roiPath = movie.rois || movie.predicted;
    this._renderRoiList(this.roiPath);
    this._roiLegend();
    this._paint();
  }

  // whether an ROI is listed and drawn: those not named as glomeruli only if asked for (by
  // the server: tentative ones, a glomerulus with _t0, _t1…, apart)
  _visible(r) {
    if (this.mode === 'prediction' && !this.showSource[r.source]) return false;
    return r.kind === 'tentative' ? this.showTentative : !(this.glomOnly && r.kind === 'other');
  }

  // a prediction's legend: the colour of each source of ROIs, how many there are, and whether shown
  _renderSources() {
    const box = this.$('.player-roi-sources'), pred = this.mode === 'prediction';
    box.hidden = !pred;
    if (!pred) { box.innerHTML = ''; return; }
    const n = (s) => this.rois.filter((r) => r.source === s).length;
    box.innerHTML = Object.entries(SOURCES).map(([s, { label, color }]) => `<label class="check"><input type="checkbox"
      data-source="${s}"${this.showSource[s] ? ' checked' : ''}><span class="swatch-roi" style="background:${color}"></span>
      ${this.esc(label)} <span class="muted">(${n(s)})</span></label>`).join('');
  }

  _roisFiltered() {
    if (this.selectedRoi != null && !this._visible(this.rois[this.selectedRoi])) this.selectedRoi = null;
    this._renderRoiList(this.roiPath);
    this._paint();
  }

  _renderRoiList(path) {
    this._renderSources();
    const box = this.$('.player-roi-list'), n = (kind) => this.rois.filter((r) => r.kind === kind).length;
    const others = n('other'), tentative = n('tentative');
    this.$('.player-roi-glom-n').textContent = path && others ? `(${others} ${this.glomOnly ? 'hidden' : 'not'})` : '';
    this.$('.player-roi-tentative-row').hidden = !(path && tentative);
    this.$('.player-roi-tentative-n').textContent = tentative ? `(${tentative}${this.showTentative ? '' : ' hidden'})` : '';
    if (!path) { box.innerHTML = '<p class="muted small">No ImageJ ROIs (RoiSet.zip) next to this movie, or in a folder above it.</p>'; return; }
    // (a prediction's: those of the plane shown only, as drawn)
    const pred = this.mode === 'prediction', listed = (r) => this._visible(r) && !(pred && this._roiPlane(r) != null && this._roiPlane(r) !== this.z);
    const from = path.split('/').slice(-2).join('/'), shown = this.rois.filter(listed).length;
    const head = pred ? `${shown} ROI${shown === 1 ? '' : 's'} on plane ${this.z + 1}`
      : `${shown === this.rois.length ? this.rois.length : `${shown} of ${this.rois.length}`} from …/${this.esc(from)}`;
    box.innerHTML = `<p class="muted small" title="${this.esc(path)}">${head}</p>`
      + this.rois.map((r, i) => {
        if (!listed(r)) return '';
        const plane = this._roiPlane(r);
        return `<button type="button" class="player-roi${i === this.selectedRoi ? ' active' : ''}" data-roi="${i}">
          <span class="swatch-roi" style="background:${this.esc(this._roiColor(r))}"></span>${this.esc(r.name)}
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
    if (!this.movie || this.playing) return;
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
      }
    });
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
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
    this.$('.player-sub').textContent = '';
    this.$('.player-warnings').textContent = '';
    this.$('.player-msg').textContent = msg;
    this.$('.player-msg').classList.toggle('player-error', error);
    this.$('.player-roi-list').innerHTML = '';
    this._renderSources();
    this.$('.player-roi-legend').textContent = '';
    this.$('.player-roi-glom-n').textContent = '';
    this.$('.player-roi-tentative-row').hidden = true;
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
    let drag = null;
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      drag = { x: e.clientX, y: e.clientY, ox: this.view.ox, oy: this.view.oy };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', (e) => {
      if (drag) {
        this.view.ox = drag.ox + e.clientX - drag.x;
        this.view.oy = drag.oy + e.clientY - drag.y;
        this._clampView();
        this._paint();
      }
      this._readout(e);
    });
    c.addEventListener('pointerup', () => { drag = null; });
    c.addEventListener('pointerleave', () => { this.$('.player-pixel').textContent = ''; });
    c.addEventListener('dblclick', () => { this._fit(); this._paint(); });
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
