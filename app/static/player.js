// Movie player for a dataset's raw and motion-corrected movies on tensor, with the ImageJ
// ROIs drawn on them. Planes are read on the server a few at a time (app/movies.py), and
// kept here while there is room, so only what is shown is ever read.

const TYPED = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array };
const CACHE_BYTES = 256 * 1024 * 1024;
const CHUNK_BYTES = 4 * 1024 * 1024; // planes asked for at once
const AHEAD = 2; // chunks fetched ahead while playing
const ROI_COLOR = '#ffd400';
const SELECTED_COLOR = '#00e5ff';
const KINDS = { mocorr: 'Motion corrected / traces', raw: 'Raw data' };
// the movies of recordings with none in the other folder, listed apart in each
const ONLY = { raw: 'Not in the motion-corrected data', mocorr: 'Not in the raw data' };

const sleep = (ms) => new Promise((r) => { setTimeout(r, ms); });

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
    this.selectedRoi = null;
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
    for (const [sel, k] of [['.player-lo', 'lo'], ['.player-hi', 'hi']]) {
      this.$(sel).addEventListener('change', (e) => { if (e.target.value !== '') { this[k] = +e.target.value; this._render(); } });
    }
    this.$('.player-auto').addEventListener('click', () => { this._autoContrast(); this._render(); });
    this.$('.player-fit').addEventListener('click', () => { this._fit(); this._paint(); });
    this.$('.player-show-rois').addEventListener('change', (e) => { this.showRois = e.target.checked; this._paint(); });
    this.$('.player-roi-names').addEventListener('change', (e) => { this.showNames = e.target.checked; this._paint(); });
    this.$('.player-roi-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-roi]');
      if (b) this._selectRoi(+b.dataset.roi);
    });
    el.addEventListener('keydown', (e) => this._key(e));
    this._wireView();
    new ResizeObserver(() => { this._sizeCanvas(); this._paint(); }).observe(this.$('.player-stage'));
  }

  // Opens the player on a dataset's movies, showing the first of `kind` ('raw' or 'mocorr').
  async open(datasetId, title, kind) {
    this.$('.player-title').textContent = `Movies of ${title}`;
    this.$('.player-list').innerHTML = '<p class="muted">Looking for movies…</p>';
    this._clearMovie('Choose a movie on the left.');
    if (!this.el.open) this.el.showModal();
    let res;
    try { res = await this.api(`/api/datasets/${datasetId}/movies`); } catch (e) {
      this.$('.player-list').innerHTML = `<p class="player-error">${this.esc(e.message)}</p>`;
      return;
    }
    this.movies = [];
    const groups = [kind, kind === 'raw' ? 'mocorr' : 'raw'].map((k) => {
      const found = res[k];
      const item = (m) => {
        this.movies.push({ ...m, kind: k });
        return `<button type="button" class="player-item${m.unmatched ? ' unmatched' : ''}" data-movie="${this.movies.length - 1}" title="${this.esc(m.path)}">
          <span>${this.esc(m.label)}</span><span class="muted">${this.fmtBytes(m.size)}${m.rois ? ' · ROIs' : ''}</span></button>`;
      };
      const items = found.movies.filter((m) => !m.unmatched).map(item).join('');
      const only = found.movies.filter((m) => m.unmatched).map(item).join('');
      const none = found.error ? `no folder: ${found.error}` : 'none found';
      return `<h3>${KINDS[k]}</h3>${items || only ? items : `<p class="muted small">${this.esc(none)}</p>`}`
        + (only ? `<h4 class="player-unmatched">${ONLY[k]}</h4>${only}` : '');
    });
    this.$('.player-list').innerHTML = groups.join('');
    const first = this.movies.find((m) => m.kind === kind) || this.movies[0];
    if (first) this.load(first);
    else this._clearMovie('No movies were found in this dataset\'s data folders.');
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
    Object.assign(this, { t: 0, z: 0, c: 0, maxProj: false, rois: [], selectedRoi: null, current: null, lo: null, hi: null });
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
    this.$('.player-sub').textContent = `${movie.label}: ${info.t} time points × ${info.z} plane${info.z > 1 ? 's' : ''}`
      + `${info.c > 1 ? ` × ${info.c} channels` : ''}, ${info.x} × ${info.y} px, ${info.dtype}${rate}`;
    this.$('.player-warnings').textContent = info.warnings.join(' ');
    this.$('.player-msg').textContent = '';
    this._sizeCanvas();
    if (movie.rois) this._loadRois(movie.rois, id);
    else this._renderRoiList(null);
    const plane = await this._plane(0);
    if (id !== this.loadId || !plane) return;
    this.current = plane;
    this._autoContrast();
    this._fit();
    this._render();
    this._updateLabels();
  }

  async _loadRois(path, id) {
    this.$('.player-roi-list').innerHTML = '<p class="muted small">Reading the ROIs…</p>';
    let res;
    try { res = await this.api(`/api/movie/rois?path=${encodeURIComponent(path)}`); } catch (e) {
      if (id === this.loadId) this.$('.player-roi-list').innerHTML = `<p class="player-error small">${this.esc(e.message)}</p>`;
      return;
    }
    if (id !== this.loadId) return;
    this.rois = res.rois;
    this._renderRoiList(path);
    this._paint();
  }

  _renderRoiList(path) {
    const box = this.$('.player-roi-list');
    if (!path) { box.innerHTML = '<p class="muted small">No ImageJ ROIs (RoiSet.zip) next to this movie, or in a folder above it.</p>'; return; }
    const from = path.split('/').slice(-2).join('/');
    box.innerHTML = `<p class="muted small" title="${this.esc(path)}">${this.rois.length} from …/${this.esc(from)}</p>`
      + this.rois.map((r, i) => {
        const plane = this._roiPlane(r);
        return `<button type="button" class="player-roi${i === this.selectedRoi ? ' active' : ''}" data-roi="${i}">
          <span class="swatch-roi" style="background:${this.esc(r.color || ROI_COLOR)}"></span>${this.esc(r.name)}
          <span class="muted">${plane == null ? 'all planes' : `plane ${plane + 1}`}</span></button>`;
      }).join('');
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
  _autoContrast() {
    const p = this.current;
    if (!p) return;
    const step = Math.max(1, Math.floor(p.length / 65536)), sample = new Float64Array(Math.ceil(p.length / step));
    for (let i = 0, j = 0; i < p.length; i += step, j++) sample[j] = p[i];
    sample.sort();
    this.lo = sample[Math.floor(sample.length * 0.005)];
    this.hi = sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.995))];
    if (this.hi <= this.lo) this.hi = this.lo + 1;
    this.$('.player-lo').value = +this.lo.toFixed(2);
    this.$('.player-hi').value = +this.hi.toFixed(2);
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
    this._paint();
  }

  _sizeCanvas() {
    const stage = this.$('.player-stage'), dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(stage.clientWidth * dpr));
    this.canvas.height = Math.max(1, Math.round(stage.clientHeight * dpr));
  }

  _fit() {
    const info = this.movie?.info, stage = this.$('.player-stage');
    if (!info) return;
    const scale = Math.min(stage.clientWidth / info.x, stage.clientHeight / info.y) * 0.98;
    this.view = { scale, ox: (stage.clientWidth - info.x * scale) / 2, oy: (stage.clientHeight - info.y * scale) / 2 };
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
    if (!this.showRois || !this.rois.length) return;
    const z = this.maxProj ? null : this.z;
    ctx.font = '12px system-ui, sans-serif';
    this.rois.forEach((r, i) => {
      const plane = this._roiPlane(r), here = z == null || plane == null || plane === z, selected = i === this.selectedRoi;
      ctx.strokeStyle = selected ? SELECTED_COLOR : r.color || ROI_COLOR;
      ctx.lineWidth = selected ? 2.5 : 1.5;
      ctx.setLineDash(here ? [] : [4, 4]);
      ctx.globalAlpha = here || selected ? 1 : 0.5;
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
      if (this.showNames && n && (here || selected)) {
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

  _updateLabels() {
    const info = this.movie?.info;
    if (!info) return;
    this.$('.player-t').value = this.t;
    const secs = info.fps ? ` · ${(this.t / info.fps).toFixed(2)} s` : '';
    this.$('.player-t-out').textContent = `${this.t + 1} / ${info.t}${secs}`;
    this.$('.player-z-out').textContent = this.maxProj ? 'max projection' : `${this.z + 1} / ${info.z}`;
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
    this.$('.player-pixel').textContent = '';
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
      const f = Math.exp(-e.deltaY * 0.0015), scale = Math.max(0.05, Math.min(64, v.scale * f)), k = scale / v.scale;
      this.view = { scale, ox: mx - (mx - v.ox) * k, oy: my - (my - v.oy) * k };
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
        this._paint();
      }
      this._readout(e);
    });
    c.addEventListener('pointerup', () => { drag = null; });
    c.addEventListener('pointerleave', () => { this.$('.player-pixel').textContent = ''; });
    c.addEventListener('dblclick', () => { this._fit(); this._paint(); });
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
      p: () => { this.maxProj = !this.maxProj; this.$('.player-max').checked = this.maxProj; this._planeChanged(); },
      a: () => { this._autoContrast(); this._render(); },
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
