// The naming explorer of the AL segmentation tab (Naming, in its list of predictions): how a model named a fly's ROIs,
// and what it would have named them otherwise. On each plane of the fly's average image, where the model's atlas puts a
// glomerulus (its probability at each pixel: /api/predictions/<id>/naming/volumes, naming.py) under the ROIs' outlines;
// beside it, what naming each ROI as each glomerulus costs, term by term, against the most a name may cost
// (/naming/costs, which the fit worker makes the first time), and the ROIs matched to glomeruli again (naming_solve.js,
// as al-diag-seg matches them) under any other most-a-name-may-cost and weights of the terms.

import { TERMS as TERM_INFO } from './identity.js';
import { TERMS, prepare, solve, whyUnnamed } from './naming_solve.js';

const LEVELS = [0.05, 0.1, 0.25, 0.5]; // the probabilities a glomerulus's contour lines are drawn at
const RINGS = [1, 2, 4]; // SDs of a glomerulus's position its rings are drawn at (the position term is capped at 4)
const HEAT = [255, 64, 196]; // the atlas over the image
const FLOOR = 0.05; // in Every glomerulus, the probability under which a pixel is no one's
const BIN = 0.02; // the width of a column of dots, in cost
const SHOWN = 6; // the pairs listed for an ROI (or a glomerulus): those that cost least
const LABEL = Object.fromEntries(TERM_INFO.map((t) => [t.key, t.label]));
const f2 = (v) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(2));
const f3 = (v) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(3));
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
// two numbers that differ, written with as many decimals as tells them apart (2 to 5)
function apart(a, b) {
  let d = 2;
  while (d < 5 && a !== b && a.toFixed(d) === b.toFixed(d)) d += 1;
  return [a.toFixed(d), b.toFixed(d)];
}

// The line segments ([x0, y0, x1, y1, ...], in pixels, a pixel's middle at its index + 0.5) where plane `m` (X × Y values
// from `at`) crosses `level`: marching squares.
function contour(m, at, X, Y, level) {
  const out = [];
  const t = (v0, v1) => (level - v0) / (v1 - v0);
  for (let y = 0; y < Y - 1; y++) {
    for (let x = 0; x < X - 1; x++) {
      const k = at + y * X + x, a = m[k], b = m[k + 1], c = m[k + X + 1], d = m[k + X];
      const idx = (a >= level ? 1 : 0) | (b >= level ? 2 : 0) | (c >= level ? 4 : 0) | (d >= level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      // where the line crosses the square's top, right, bottom and left sides
      const tx = x + 0.5 + t(a, b), ty = y + 0.5, rx = x + 1.5, ry = y + 0.5 + t(b, c);
      const bx = x + 0.5 + t(d, c), by = y + 1.5, lx = x + 0.5, ly = y + 0.5 + t(a, d);
      switch (idx) {
        case 1: case 14: out.push(lx, ly, tx, ty); break;
        case 2: case 13: out.push(tx, ty, rx, ry); break;
        case 3: case 12: out.push(lx, ly, rx, ry); break;
        case 4: case 11: out.push(rx, ry, bx, by); break;
        case 6: case 9: out.push(tx, ty, bx, by); break;
        case 7: case 8: out.push(lx, ly, bx, by); break;
        case 5: out.push(lx, ly, tx, ty, rx, ry, bx, by); break;
        default: out.push(tx, ty, rx, ry, lx, ly, bx, by); // 10
      }
    }
  }
  return Float32Array.from(out);
}

function strokeSegments(ctx, segs, scale) {
  ctx.beginPath();
  for (let k = 0; k < segs.length; k += 4) {
    ctx.moveTo(segs[k] * scale, segs[k + 1] * scale);
    ctx.lineTo(segs[k + 2] * scale, segs[k + 3] * scale);
  }
  ctx.stroke();
}

// text with a dark edge, to read over the image
function haloText(ctx, text, x, y, color) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(0, 0, 0, .8)';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

export class NamingExplorer {
  // helpers: { api, esc } (the app's)
  constructor(dialog, { api, esc }) {
    Object.assign(this, { el: dialog, api, esc, seq: 0, raf: 0 });
    const q = (s) => dialog.querySelector(s);
    this.q = q;
    this.canvas = q('.naming-canvas');
    this.off = document.createElement('canvas');
    this.tip = q('.naming-tip');
    q('.naming-sliders').innerHTML = `<label class="naming-slider naming-slider-max" title="An ROI is named as a glomerulus only where that costs no more than this (the model's max_cost, set when it was fit); else it is numbered">
        <span>Most a name may cost</span><input type="range" data-w="max" min="0.2" max="1" step="0.005"><output></output></label>
      <p class="small muted">The weight of each term in the cost:</p>
      ${TERM_INFO.map((t) => `<label class="naming-slider" title="${esc(t.value)}"><span><i class="naming-sw naming-sw-${t.key}"></i>${esc(t.label)}</span>
        <input type="range" data-w="${t.key}" min="0" max="2" step="0.05"><output></output></label>`).join('')}`;
    q('.naming-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => { this.seq += 1; this._free(); });
    q('.naming-show').addEventListener('click', (e) => { const b = e.target.closest('[data-show]'); if (b && this.vol) this._setShow(b.dataset.show); });
    q('.naming-glom').addEventListener('change', (e) => { if (this.vol) this._chooseGlom(e.target.value, true); });
    for (const s of ['.naming-scale', '.naming-names']) q(s).addEventListener('change', () => { this._renderKey(); this._drawSoon(); });
    q('.naming-opacity').addEventListener('input', () => this._drawSoon());
    q('.naming-planes').addEventListener('click', (e) => { const b = e.target.closest('[data-z]'); if (b) this._setPlane(+b.dataset.z); });
    this.canvas.addEventListener('click', (e) => this._clickImage(e));
    this.canvas.addEventListener('mousemove', (e) => this._hover(e));
    this.canvas.addEventListener('mouseleave', () => { q('.naming-readout').textContent = ''; });
    this.canvas.addEventListener('wheel', (e) => { if (this.vol) { e.preventDefault(); this._setPlane(this.z + Math.sign(e.deltaY)); } }, { passive: false });
    dialog.addEventListener('keydown', (e) => {
      if (!this.vol || e.target.closest('input, select, textarea') || !['ArrowUp', 'ArrowDown'].includes(e.key)) return;
      e.preventDefault();
      this._setPlane(this.z + (e.key === 'ArrowDown' ? 1 : -1));
    });
    q('.naming-sliders').addEventListener('input', (e) => {
      const k = e.target.dataset.w;
      if (!k || !this.c?.full) return;
      if (k === 'max') this.max = +e.target.value; else this.w[k] = +e.target.value;
      this._solveSoon();
    });
    q('.naming-reset').addEventListener('click', () => { if (this.c) { this.w = { ...this.c.weights }; this.max = this.c.maxCost; this._solved(); } });
    const strip = q('.naming-strip');
    strip.addEventListener('pointerdown', (e) => {
      if (!e.target.closest('.naming-thr-grab') || !this.canWhatIf) return;
      this.dragging = true;
      strip.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    strip.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const svg = strip.querySelector('svg'), r = svg.getBoundingClientRect(), g = this.stripScale;
      const x = (e.clientX - r.left) / r.width * g.W;
      this.max = clamp(+(Math.round(((x - g.left) / g.width) * g.hi / 0.005) * 0.005).toFixed(3), 0.2, 1); // (0.6, not 0.6000000000000001)
      this._solveSoon();
    });
    for (const ev of ['pointerup', 'pointercancel']) strip.addEventListener(ev, () => { this.dragging = false; });
    strip.addEventListener('click', (e) => { const d = e.target.closest('[data-roi]'); if (d) this._chooseRoi(+d.dataset.roi, true); });
    q('.naming-by').addEventListener('click', (e) => {
      const b = e.target.closest('[data-by]');
      if (!b || !this.c) return;
      this.by = b.dataset.by;
      this._renderDetail();
    });
    q('.naming-detail-body').addEventListener('click', (e) => {
      const b = e.target.closest('[data-pair]');
      if (!b) return;
      const [i, j] = b.dataset.pair.split(',').map(Number);
      if (this.by === 'roi') this._chooseGlom(this.c.names[j], false); else { this.sel.roi = this.c.labels[i]; this._toRoiPlane(); this._selected(); }
    });
    q('.naming-gloms').addEventListener('click', (e) => {
      const h = e.target.closest('[data-sort]'), r = e.target.closest('[data-glom]');
      if (h) {
        this.sort = { key: h.dataset.sort, dir: this.sort.key === h.dataset.sort ? -this.sort.dir : 1 };
        this._renderGloms();
      } else if (r) this._chooseGlom(r.dataset.glom, true);
    });
    q('.naming-unnamed-only').addEventListener('change', () => this._renderGloms());
    // what a mark of the charts is, by the pointer
    const side = q('.naming-panels');
    side.addEventListener('mousemove', (e) => {
      const t = e.target.closest?.('[data-tip]');
      this.tip.hidden = !t;
      if (!t) return;
      this.tip.textContent = t.dataset.tip;
      const w = this.tip.offsetWidth, h = this.tip.offsetHeight;
      this.tip.style.left = `${clamp(e.clientX + 14, 4, window.innerWidth - w - 4)}px`;
      this.tip.style.top = `${e.clientY + 18 + h > window.innerHeight ? e.clientY - h - 10 : e.clientY + 18}px`;
    });
    side.addEventListener('mouseleave', () => { this.tip.hidden = true; });
    new ResizeObserver(() => this._drawSoon()).observe(q('.naming-canvas-box'));
  }

  // Opens the naming of prediction `p` ({ id, fly, fit_id }), at its ROI of label `roi` if one is given
  async open(p, roi = null) {
    const seq = ++this.seq, { api, esc, q } = this;
    this._free();
    Object.assign(this, { p, first: roi });
    q('#naming-title').textContent = `How fit #${p.fit_id}'s model named ${p.fly}'s ROIs`;
    if (!this.el.open) this.el.showModal();
    this._status([['info', 'Reading the prediction…']]);
    let view, vol;
    try { [view, vol] = await Promise.all([api(`/api/predictions/${p.id}/naming`), this._volumes(p.id)]); } catch (e) {
      if (seq === this.seq) this._status([['error', `Could not open it: ${esc(e.message)}`]]);
      return;
    }
    if (seq !== this.seq) return;
    Object.assign(this, { v: view, vol });
    this._setup();
    this._loadCosts(seq);
  }

  _free() {
    cancelAnimationFrame(this.raf);
    Object.assign(this, { raf: 0, v: null, vol: null, c: null, s: null, model: null, doc: null, canWhatIf: false, notes: [], dragging: false,
      sel: { roi: null, glom: null }, by: 'roi', show: 'one', z: 0, sort: { key: 'name', dir: 1 } });
    const { q } = this;
    for (const s of ['.naming-sub', '.naming-planes', '.naming-strip', '.naming-detail-body', '.naming-gloms', '.naming-outcome', '.naming-key',
      '.naming-readout', '.naming-status']) q(s).replaceChildren();
    q('.naming-glom').replaceChildren();
    for (const i of q('.naming-sliders').querySelectorAll('input')) i.disabled = true;
    q('.naming-whatif').classList.add('off');
    q('.naming-reset').disabled = true;
    this.canvas.width = 0;
    this.tip.hidden = true;
  }

  // the prediction's volumes (naming.volumes): the average image, the ROIs' labels and each glomerulus' map
  async _volumes(id) {
    let res;
    try { res = await fetch(`/api/predictions/${id}/naming/volumes`); } catch (e) { throw new Error(`cannot reach the server (${e.message})`); }
    if (!res.ok) {
      let why = `${res.status} ${res.statusText}`;
      try { why = (await res.json()).error || why; } catch { /* (as it is) */ }
      throw new Error(why);
    }
    const buf = await res.arrayBuffer(), dv = new DataView(buf);
    if (buf.byteLength < 8 || new TextDecoder().decode(new Uint8Array(buf, 0, 4)) !== 'OLFN') throw new Error('not the volumes of a prediction');
    const n = dv.getUint32(4, true), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
    const size = head.shape[0] * head.shape[1] * head.shape[2];
    let at = 8 + n;
    const avg = new Uint8Array(buf, at, size);
    at += size;
    const lab = new Uint16Array(buf.slice(at, at + 2 * size));
    at += 2 * size;
    return { avg, lab, prior: new Uint8Array(buf, at, head.maps.length * size), maps: head.maps };
  }

  _setup() {
    const v = this.v, [Z, Y, X] = v.shape;
    Object.assign(this, { Z, Y, X, size: Y * X, rois: v.rois, roiOf: new Map(v.rois.map((r) => [r.label, r])),
      mapOf: new Map(v.maps.map((m, k) => [m.name, k])), edges: new Array(Z), terr: new Array(Z), lines: new Map(),
      top: Math.max(0, ...v.maps.map((m) => m.peak)) });
    this._partial();
    this.show = v.maps.length ? 'one' : 'none';
    this.notes = [];
    if (v.final && v.kind && v.final !== v.kind) {
      this.notes.push(['info', `The prediction's ROIs are cleaned ones (${this.esc(v.final)}), which keep the names of those they were cleaned from: those
        (${this.esc(v.kind)}) are the ROIs shown here, as the costs are of them.`]);
    }
    if (!v.maps.length) {
      this.notes.push(['info', v.model.has_atlas === false ? 'Its model has no atlas: its ROIs were named by odor tuning and position alone.'
        : 'This prediction did not keep the atlas on its fly\'s planes: predict it again to see it.']);
    }
    if (!v.terms.rois) this.notes.push(['warn', 'This prediction did not keep what its names cost (it was made before they were): predict it again.']);
    // to begin with: the ROI asked for; else the one not named that came nearest to a name (else the first), with the glomerulus
    // that costs least for it
    const { c, s } = this;
    let start = null, least = Infinity;
    for (let i = 0; i < c.nL; i++) {
      const j = this._cheapest(i);
      if (s.names[i] < 0 && j >= 0 && s.cost[i * c.nN + j] < least) { least = s.cost[i * c.nN + j]; start = c.labels[i]; }
    }
    if (this.rowOf.has(this.first)) start = this.first;
    start ??= this.rois[0]?.label ?? null;
    this.z = 0;
    if (start != null) {
      this.sel.roi = start;
      this._glomOfRoi();
      this._toRoiPlane();
    }
    this._renderAll();
  }

  // The costs as far as the prediction kept them (each ROI's few best glomeruli), laid out as every pair's are: unknown ones NaN
  _partial() {
    const v = this.v, t = v.terms, rois = t.rois || {}, labels = this.rois.map((r) => r.label);
    const names = [...new Set([...v.maps.map((m) => m.name), ...Object.keys(v.glomeruli),
      ...Object.values(rois).flatMap((r) => r.candidates.map((c) => c.name))])].sort(natural);
    const nL = labels.length, nN = names.length, col = new Map(names.map((n, j) => [n, j]));
    const terms = Object.fromEntries(TERMS.map((k) => [k, new Float64Array(nL * nN).fill(NaN)]));
    const cost = new Float64Array(nL * nN).fill(NaN), assigned = new Int32Array(nL).fill(-1);
    const reference = names.map((n) => v.glomeruli[n]?.trained == null && v.glomeruli[n]?.flies != null);
    labels.forEach((l, i) => {
      for (const c of rois[l]?.candidates || []) {
        const k = i * nN + col.get(c.name);
        cost[k] = c.cost;
        for (const key of TERMS) if (c[key] != null) terms[key][k] = c[key];
        if (c.reference) reference[col.get(c.name)] = true;
      }
      if (rois[l]?.assigned != null) assigned[i] = col.get(rois[l].assigned);
    });
    const w = t.weights || Object.fromEntries(TERMS.map((k) => [k, 0]));
    this.c = { full: false, nL, nN, labels, names, terms, reference, assigned, hasAtlas: w.atlas != null, weights: { ...w, atlas: w.atlas ?? 0 },
      maxCost: t.max_cost ?? v.model.max_cost ?? NaN };
    this.model = { names: assigned, cost, layout: terms.layout, sum: t.weight_sum ?? NaN, settled: true };
    this._adopt();
  }

  _adopt() {
    const c = this.c;
    Object.assign(this, { rowOf: new Map(c.labels.map((l, i) => [l, i])), colOf: new Map(c.names.map((n, j) => [n, j])),
      w: { ...c.weights }, max: c.maxCost, s: this.model });
  }

  // every pair's cost: as kept with the prediction, else made by the fit worker now
  async _loadCosts(seq) {
    const { api, esc } = this, url = `/api/predictions/${this.p.id}/naming/costs`;
    const failed = (e) => {
      if (seq !== this.seq) return;
      this._status([...this.notes, ['warn', `${esc(e.message)}. Until it is, only each ROI's few best glomeruli are known here, and <i>What if</i> is off.`]]);
    };
    let doc;
    try { doc = await api(url); } catch (e) { failed(e); return; }
    if (seq !== this.seq) return;
    if (!doc.made) {
      this._status([...this.notes, ['info', 'The fit worker is working out what every glomerulus costs for every ROI: a few seconds, the first time a prediction is opened here…']]);
      try { doc = await api(url, { method: 'POST' }); } catch (e) { failed(e); return; }
      if (seq !== this.seq) return;
    }
    this._full(doc);
  }

  _full(doc) {
    const { esc } = this, c = { ...prepare(doc), full: true };
    const model = solve(c, c.weights, c.maxCost);
    const off = c.labels.filter((l, i) => model.names[i] !== c.assigned[i]);
    Object.assign(this, { c, model, doc, canWhatIf: doc.exact && !off.length });
    this._adopt();
    if (!doc.exact) this.notes.push(['warn', `<i>What if</i> is off: ${esc(doc.why)}.`]);
    else if (off.length) {
      this.notes.push(['warn', `<i>What if</i> is off: matched again here, ${plural(off.length, 'ROI')} (${off.join(', ')}) ${off.length === 1 ? 'comes' : 'come'} out with another
        name than al-diag-seg gave ${off.length === 1 ? 'it' : 'them'}.`]);
    }
    if (doc.predicted && !doc.same) {
      const n = doc.labels.filter((l, i) => doc.predicted[i] !== doc.assigned[i]);
      this.notes.push(['warn', `Named again from what the prediction kept, ${plural(n.length, 'ROI')} (${n.join(', ')}) ${n.length === 1 ? 'gets' : 'get'} another name than
        in the prediction: the fly's files on tensor may have changed since, or two costs are within rounding of each other. The costs here are of the names given now.`]);
    }
    if (this.sel.glom != null && !this.colOf.has(this.sel.glom)) this.sel.glom = null;
    if (this.sel.roi != null && !this.rowOf.has(this.sel.roi)) this.sel.roi = null;
    this._renderAll();
  }

  // ----------------------------------------------------------------- the names as they are now

  _whatIf() { return this.s !== this.model; }

  // a cost, written with as many decimals as tells it from the most a name may cost (3 to 6)
  _fc(v) {
    if (v == null || Number.isNaN(v)) return '—';
    let d = 3;
    while (d < 6 && v !== this.max && v.toFixed(d) === this.max.toFixed(d)) d += 1;
    return v.toFixed(d);
  }

  _name(i) { const j = this.s.names[i]; return j >= 0 ? this.c.names[j] : null; }

  _term(key, k) { return (key === 'layout' ? this.s.layout : this.c.terms[key])[k]; }

  // the glomerulus that costs least for ROI i, of those whose cost is known (-1: none)
  _cheapest(i) {
    const { nN } = this.c, cost = this.s.cost;
    let best = -1;
    for (let j = 0; j < nN; j++) if (!Number.isNaN(cost[i * nN + j]) && (best < 0 || cost[i * nN + j] < cost[i * nN + best])) best = j;
    return best;
  }

  // the ROI glomerulus j costs least for (-1: none known)
  _cheapestRoi(j) {
    const { nL, nN } = this.c, cost = this.s.cost;
    let best = -1;
    for (let i = 0; i < nL; i++) if (!Number.isNaN(cost[i * nN + j]) && (best < 0 || cost[i * nN + j] < cost[best * nN + j])) best = i;
    return best;
  }

  _solveSoon() {
    if (this.solving) return;
    this.solving = requestAnimationFrame(() => { this.solving = 0; this._solved(); });
  }

  // the ROIs named again under the weights and the most a name may cost chosen
  _solved() {
    const { c } = this;
    if (!c?.full) return;
    const same = this.max === c.maxCost && TERMS.every((k) => this.w[k] === c.weights[k]);
    const sum = TERMS.reduce((t, k) => t + (k === 'atlas' && !c.hasAtlas ? 0 : this.w[k]), 0);
    if (same || !this.canWhatIf) this.s = this.model;
    else if (sum > 0) this.s = solve(c, this.w, this.max);
    this._renderSliders();
    this._renderOutcome();
    this._renderStrip();
    this._renderDetail();
    this._renderGloms();
    this._renderGlomSelect();
    this._renderHead();
    this._renderKey();
    this._drawSoon();
  }

  // ----------------------------------------------------------------- what is chosen

  // the glomerulus shown with the ROI chosen: its name, else the one that costs least for it
  _glomOfRoi() {
    const i = this.rowOf.get(this.sel.roi);
    if (i == null) return;
    const j = this.s.names[i] >= 0 ? this.s.names[i] : this._cheapest(i);
    if (j >= 0) this.sel.glom = this.c.names[j];
  }

  // to a plane the ROI chosen is on (the one nearest its middle), unless the plane shown is one
  _toRoiPlane() {
    const r = this.roiOf.get(this.sel.roi);
    if (!r || r.planes.includes(this.z) || !r.planes.length) return;
    this.z = r.planes.reduce((a, b) => (Math.abs(b - r.centroid[0]) < Math.abs(a - r.centroid[0]) ? b : a));
  }

  _chooseRoi(label, jump) {
    this.sel.roi = label;
    this.by = 'roi';
    this._glomOfRoi();
    if (jump) this._toRoiPlane();
    this._selected();
  }

  // `listed`: chosen in the list of glomeruli (or the menu): then with the ROI it names (else the one it costs least for), on the plane
  // its map is strongest on
  _chooseGlom(name, listed) {
    if (!this.colOf.has(name)) return;
    this.sel.glom = name;
    if (listed) {
      const j = this.colOf.get(name), named = this.s.names.indexOf(j), i = named >= 0 ? named : this._cheapestRoi(j);
      this.by = 'glom';
      if (i >= 0) this.sel.roi = this.c.labels[i];
      const m = this.v.maps[this.mapOf.get(name)];
      if (m && m.peak > 0) this.z = m.at[0]; else this._toRoiPlane();
      if (this.show === 'all') this.show = 'one';
    }
    this._selected();
  }

  _selected() {
    this._renderGlomSelect();
    this._renderPlanes();
    this._renderStrip();
    this._renderDetail();
    this._renderGloms();
    this._renderKey();
    this._drawSoon();
  }

  _setShow(show) {
    this.show = show;
    this._renderGlomSelect();
    this._renderPlanes();
    this._renderKey();
    this._drawSoon();
  }

  _setPlane(z) {
    z = clamp(z, 0, this.Z - 1);
    if (z === this.z) return;
    this.z = z;
    this._renderPlanes();
    this._renderKey();
    this._drawSoon();
  }

  // ----------------------------------------------------------------- the image

  // the ROIs' edges on plane z, as line segments between pixels ({label: [x0, y0, x1, y1, ...]}), and where to write each one's name
  _roiEdges(z) {
    if (this.edges[z]) return this.edges[z];
    const { X, Y, size } = this, lab = this.vol.lab, at = z * size, segs = new Map(), sums = new Map();
    for (let y = 0; y < Y; y++) {
      for (let x = 0; x < X; x++) {
        const k = at + y * X + x, l = lab[k];
        if (!l) continue;
        let a = segs.get(l), m = sums.get(l);
        if (!a) { segs.set(l, a = []); sums.set(l, m = [0, 0, 0, y]); } // (rows are gone through from the top: its first is its top)
        m[0] += x + 0.5; m[1] += y + 0.5; m[2] += 1;
        if (x === 0 || lab[k - 1] !== l) a.push(x, y, x, y + 1);
        if (x === X - 1 || lab[k + 1] !== l) a.push(x + 1, y, x + 1, y + 1);
        if (y === 0 || lab[k - X] !== l) a.push(x, y, x + 1, y);
        if (y === Y - 1 || lab[k + X] !== l) a.push(x, y + 1, x + 1, y + 1);
      }
    }
    this.edges[z] = { segs, at: new Map([...sums].map(([l, m]) => [l, [m[0] / m[2], m[1] / m[2], m[3]]])) };
    return this.edges[z];
  }

  // Every glomerulus on plane z: at each pixel the largest probability (`max`) and whose it is (`arg`); the lines between pixels that
  // are different glomeruli's (`borders`); and where each one's name goes (`names`: where it is the most probable, weighted by it)
  _territories(z) {
    if (this.terr[z]) return this.terr[z];
    const { X, Y, Z, size } = this, prior = this.vol.prior, K = this.v.maps.length, floor = FLOOR * 255;
    const max = new Uint8Array(size), arg = new Int16Array(size).fill(-1);
    for (let k = 0; k < K; k++) {
      const at = (k * Z + z) * size;
      for (let p = 0; p < size; p++) if (prior[at + p] > max[p]) { max[p] = prior[at + p]; arg[p] = k; }
    }
    const own = (p) => (max[p] >= floor ? arg[p] : -1), borders = [], sums = new Map();
    for (let y = 0; y < Y; y++) {
      for (let x = 0; x < X; x++) {
        const p = y * X + x, k = own(p);
        if (k >= 0) {
          let m = sums.get(k);
          if (!m) sums.set(k, m = [0, 0, 0, 0]);
          m[0] += (x + 0.5) * max[p]; m[1] += (y + 0.5) * max[p]; m[2] += max[p]; m[3] += 1;
        }
        if (x < X - 1 && own(p + 1) !== k) borders.push(x + 1, y, x + 1, y + 1);
        if (y < Y - 1 && own(p + X) !== k) borders.push(x, y + 1, x + 1, y + 1);
      }
    }
    const names = [...sums].filter(([, m]) => m[3] >= 40).map(([k, m]) => ({ name: this.v.maps[k].name, x: m[0] / m[2], y: m[1] / m[2] }));
    this.terr[z] = { max, arg, borders: Float32Array.from(borders), names };
    return this.terr[z];
  }

  // what of the atlas is drawn over the plane shown: { data, at (where the plane starts in it), top (the value at full strength), name }
  _overlay() {
    const { size, Z, z } = this, peak = this.q('.naming-scale').value === 'peak';
    if (this.show === 'one') {
      const k = this.mapOf.get(this.sel.glom);
      if (k == null) return null;
      const m = this.v.maps[k];
      return { data: this.vol.prior, at: (k * Z + z) * size, top: peak ? Math.max(1, m.peak * 255) : 255, k, j: this.colOf.get(m.name) ?? -1, peak: m.peak, name: m.name };
    }
    if (this.show === 'all' && this.v.maps.length) return { data: this._territories(z).max, at: 0, top: peak ? Math.max(1, this.top * 255) : 255, peak: this.top };
    return null;
  }

  _drawSoon() { if (!this.raf) this.raf = requestAnimationFrame(() => this._draw()); }

  _draw() {
    this.raf = 0;
    if (!this.vol) return;
    const { X, Y, size, canvas, z, c, s } = this, box = this.q('.naming-canvas-box');
    const scale = Math.max(1, Math.min(box.clientWidth / X, box.clientHeight / Y));
    const dpr = window.devicePixelRatio || 1, w = Math.round(X * scale), h = Math.round(Y * scale);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }
    this.scale = scale;
    // the fly's average image, with the atlas over it
    const off = this.off;
    if (off.width !== X || off.height !== Y) { off.width = X; off.height = Y; }
    const octx = off.getContext('2d'), img = octx.createImageData(X, Y), d = img.data;
    const avg = this.vol.avg, base = z * size, ov = this._overlay(), strength = this.q('.naming-opacity').value / 100;
    for (let p = 0, k = 0; p < size; p++, k += 4) {
      const g = avg[base + p], v = ov ? ov.data[ov.at + p] : 0, a = v ? strength * Math.min(1, v / ov.top) ** 0.75 : 0;
      d[k] = g + (HEAT[0] - g) * a;
      d[k + 1] = g + (HEAT[1] - g) * a;
      d[k + 2] = g + (HEAT[2] - g) * a;
      d[k + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(off, 0, 0, w, h);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    const heat = `rgb(${HEAT.join(',')})`;
    if (ov && this.show === 'one' && strength > 0) {
      // lines of equal probability, heavier the higher
      const key = `${ov.k}|${z}`;
      if (!this.lines.has(key)) this.lines.set(key, LEVELS.map((l) => contour(this.vol.prior, ov.at, X, Y, l * 255)));
      this.lines.get(key).forEach((segs, n) => {
        if (!segs.length) return;
        ctx.strokeStyle = 'rgba(0, 0, 0, .55)';
        ctx.lineWidth = 2.2 + n * 0.7;
        strokeSegments(ctx, segs, scale);
        ctx.strokeStyle = heat;
        ctx.lineWidth = 0.8 + n * 0.7;
        strokeSegments(ctx, segs, scale);
      });
      // where its probability is what the ROI chosen would need, on average, for this name to cost no more than a name may
      const need = this._needed(this.rowOf.get(this.sel.roi) ?? -1, ov.j);
      if (need != null && need > 0 && need <= 1) {
        const segs = contour(this.vol.prior, ov.at, X, Y, need * 255);
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = 'rgba(0, 0, 0, .7)';
        ctx.lineWidth = 3.4;
        strokeSegments(ctx, segs, scale);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.6;
        strokeSegments(ctx, segs, scale);
        ctx.setLineDash([]);
      }
    } else if (ov && this.show === 'all' && strength > 0) {
      ctx.strokeStyle = 'rgba(255, 150, 225, .75)';
      ctx.lineWidth = 1;
      strokeSegments(ctx, this._territories(z).borders, scale);
    }
    // the ROIs: named ones whole, the others dashed; the one chosen heavier
    const { segs, at } = this._roiEdges(z), names = this.q('.naming-names').checked, whatIf = this._whatIf();
    const order = [...segs.keys()].sort((a, b) => (a === this.sel.roi) - (b === this.sel.roi));
    for (const l of order) {
      const i = this.rowOf.get(l), named = i != null && s.names[i] >= 0, chosen = l === this.sel.roi, e = segs.get(l);
      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(0, 0, 0, .6)';
      ctx.lineWidth = chosen ? 5 : 3;
      strokeSegments(ctx, e, scale);
      ctx.setLineDash(named ? [] : [5, 4]);
      ctx.strokeStyle = chosen ? '#00e5ff' : named ? '#ffffff' : 'rgba(255, 255, 255, .7)';
      ctx.lineWidth = chosen ? 2.5 : 1.3;
      strokeSegments(ctx, e, scale);
    }
    ctx.setLineDash([]);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (this.show === 'all' && ov && names) {
      ctx.font = 'italic 11px system-ui, sans-serif';
      for (const n of this._territories(z).names) haloText(ctx, n.name, n.x * scale, n.y * scale - 6, '#ffb3ea');
    }
    for (const l of order) {
      const i = this.rowOf.get(l), chosen = l === this.sel.roi;
      if (i == null || !(names || chosen)) continue;
      const now = this._name(i), was = whatIf && c.assigned[i] >= 0 ? c.names[c.assigned[i]] : null;
      // (under other thresholds than the model's: what changed of its name)
      const text = !whatIf || now === was ? now ?? String(l) : now == null ? `−${was}` : was == null ? `+${now}` : `${was}→${now}`;
      const [x, y, top] = at.get(l);
      ctx.font = `${now || was ? '600 12px' : '10px'} system-ui, sans-serif`;
      // (the one chosen: over it, clear of what is marked in it)
      haloText(ctx, text, x * scale, chosen ? Math.max(8, top * scale - 9) : y * scale + (this.show === 'all' ? 7 : 0),
        chosen ? '#00e5ff' : now ? '#ffffff' : 'rgba(255, 255, 255, .75)');
    }
    this._drawExpected(ctx, scale, heat);
  }

  // Where the atlas expects the glomerulus shown (its probability's centre of mass), with rings at 1, 2 and 4 SDs of its position as they
  // cut the plane shown, and the line from the middle of the ROI chosen to it: what the position term is of
  _drawExpected(ctx, scale, heat) {
    const g = this._expected();
    if (!g || this.show !== 'one') return;
    const { z } = this, x = (g.centre[2] + 0.5) * scale, y = (g.centre[1] + 0.5) * scale;
    const ring = (rx, ry, width, color) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0, 0, 2 * Math.PI); ctx.stroke(); };
    if (g.sd && this.v.spacing) {
      const [sz, sy, sx] = this.v.spacing, dz = (z - g.centre[0]) * sz;
      ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'left';
      for (const k of RINGS) {
        const r2 = (k * g.sd) ** 2 - dz * dz;
        if (r2 <= 0) continue;
        const rx = Math.sqrt(r2) / sx * scale, ry = Math.sqrt(r2) / sy * scale;
        ctx.setLineDash([2, 4]);
        ring(rx, ry, 2.4, 'rgba(0, 0, 0, .5)');
        ring(rx, ry, 1, heat);
        ctx.setLineDash([]);
        haloText(ctx, `${k} SD`, x + rx * Math.SQRT1_2 + 3, y - ry * Math.SQRT1_2 - 3, '#ffd3f3');
      }
    }
    const r = this.roiOf.get(this.sel.roi);
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    if (r) {
      const rx = (r.centroid[2] + 0.5) * scale, ry = (r.centroid[1] + 0.5) * scale;
      ctx.setLineDash([4, 3]);
      for (const [width, color] of [[3, 'rgba(0, 0, 0, .6)'], [1.2, '#00e5ff']]) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(x, y); ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.fillStyle = '#00e5ff';
      ctx.strokeStyle = 'rgba(0, 0, 0, .7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(rx, ry, 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      const d = this._apart(this.rowOf.get(r.label), this.colOf.get(g.name));
      if (d && Math.hypot(rx - x, ry - y) > 70) haloText(ctx, `${d.um.toFixed(1)} µm`, (rx + x) / 2 + 6, (ry + y) / 2, '#9ff3ff');
    }
    // (a ringed cross; said to be on another plane, where it is)
    for (const [width, color] of [[4, 'rgba(0, 0, 0, .7)'], [1.8, heat]]) {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();
    }
    if (Math.round(g.centre[0]) !== z) haloText(ctx, `at plane ${(g.centre[0] + 1).toFixed(1)}`, x + 12, y + 16, '#ffd3f3');
  }

  // The mean probability of glomerulus j that ROI i's voxels would need for the pair to cost just what a name may, its other terms as
  // they are (0 or less: it costs no more whatever the atlas says; over 1: more, whatever it says); null where the atlas term is off
  _needed(i, j) {
    if (!(i >= 0 && j >= 0)) return null;
    const { c, s } = this, k = i * c.nN + j, a = c.terms.atlas[k], wa = c.hasAtlas ? this.w.atlas : 0;
    if (!wa || Number.isNaN(a) || Number.isNaN(s.cost[k]) || !(s.sum > 0)) return null;
    return 1 - (this.max * s.sum - (s.cost[k] * s.sum - wa * a)) / wa;
  }

  // how far the middle of ROI i is from where glomerulus j is expected, as the position term has it: { um, across (of it, across
  // planes) }; null where that is not known (until every pair's cost is)
  _apart(i, j) {
    const { c } = this;
    if (!c.full || i == null || j == null) return null;
    const d = c.roiPos[i].map((v, n) => v - c.expPos[j][n]);
    return { um: Math.hypot(...d), across: Math.abs(d[0]) };
  }

  // the glomerulus shown, as the position term has it: { name, centre ([z, y, x], voxels), sd (um; null if not known) }, or null
  _expected() {
    const name = this.sel.glom, m = this.v.maps[this.mapOf.get(name)];
    if (!m?.centre) return null;
    const j = this.colOf.get(name);
    return { name, centre: m.centre, sd: (this.c.full ? this.c.posSd[j] : this.v.glomeruli[name]?.sd) ?? null };
  }

  _pixel(e) {
    const x = Math.floor(e.offsetX / this.scale), y = Math.floor(e.offsetY / this.scale);
    return x >= 0 && y >= 0 && x < this.X && y < this.Y ? y * this.X + x : -1;
  }

  _clickImage(e) {
    if (!this.vol) return;
    const p = this._pixel(e), l = p >= 0 ? this.vol.lab[this.z * this.size + p] : 0;
    if (l && this.rowOf.has(l)) this._chooseRoi(l, false);
  }

  // under the pointer: the ROI, and the glomeruli most probable there
  _hover(e) {
    if (!this.vol) return;
    const p = this._pixel(e), out = this.q('.naming-readout');
    if (p < 0) { out.textContent = ''; return; }
    const { Z, z, size } = this, l = this.vol.lab[z * size + p], i = this.rowOf.get(l), parts = [];
    if (l) parts.push(`ROI ${l}${i != null ? (this._name(i) ? `, named ${this._name(i)}` : ', not named') : ''}`);
    const here = this.v.maps.map((m, k) => [m.name, this.vol.prior[(k * Z + z) * size + p] / 255]).filter(([, v]) => v >= 0.005).sort((a, b) => b[1] - a[1]);
    if (this.v.maps.length) {
      const shown = here.slice(0, 3), own = here.find(([n]) => n === this.sel.glom);
      if (this.show === 'one' && own && !shown.includes(own)) shown.push(own);
      parts.push(here.length ? `the atlas here: ${shown.map(([n, v]) => `${n} ${f2(v)}`).join(', ')}${here.length > shown.length ? `, and ${here.length - shown.length} more under that` : ''}`
        : 'the atlas has no glomerulus here');
    }
    out.textContent = parts.join(' · ');
  }

  // ----------------------------------------------------------------- beside the image

  _status(notes) {
    this.q('.naming-status').innerHTML = notes.map(([kind, html]) => `<div class="msg ${kind}">${html}</div>`).join('');
  }

  _renderAll() {
    this._status(this.notes);
    this._renderHead();
    this._renderGlomSelect();
    this._renderPlanes();
    this._renderSliders();
    this._renderOutcome();
    this._renderStrip();
    this._renderDetail();
    this._renderGloms();
    this._renderKey();
    this._drawSoon();
  }

  _renderHead() {
    const { c, s } = this, named = s.names.filter((j) => j >= 0).length;
    this.q('.naming-sub').textContent = `${named} of ${plural(c.nL, 'ROI')} named${this._whatIf() ? ' (what if)' : ''}, among ${plural(c.names.length, 'glomerulus', 'glomeruli')}; ${
      this.Z} planes${this.v.spacing ? ` ${this.v.spacing[0]} µm apart, pixels of ${+this.v.spacing[2].toFixed(3)} µm` : ''}`;
  }

  _renderGlomSelect() {
    const sel = this.q('.naming-glom'), { c, esc } = this, named = new Set(this.s.names);
    sel.innerHTML = c.names.map((n, j) => `<option value="${esc(n)}">${esc(n)}${named.has(j) ? '' : ' (names no ROI)'}${this.mapOf.has(n) ? '' : ' (not in the atlas)'}</option>`).join('');
    if (this.sel.glom != null) sel.value = this.sel.glom;
    this.q('.naming-glom-label').hidden = this.show !== 'one';
    for (const b of this.q('.naming-show').children) {
      b.classList.toggle('active', b.dataset.show === this.show);
      b.disabled = !this.v.maps.length && b.dataset.show !== 'none';
    }
  }

  // the planes: each with how strong the map shown is on it (of its strongest), and a dot on those the ROI chosen is on
  _renderPlanes() {
    const m = this.show === 'one' ? this.v.maps[this.mapOf.get(this.sel.glom)] : null, roi = this.roiOf.get(this.sel.roi);
    this.q('.naming-planes').innerHTML = `<span class="muted small">Plane</span>${Array.from({ length: this.Z }, (_, z) => {
      const share = m && m.peak > 0 ? m.planes[z] / m.peak : 0, on = roi?.planes.includes(z);
      return `<button type="button" data-z="${z}"${z === this.z ? ' class="active"' : ''} title="${this.esc(`Plane ${z + 1}${m ? `: the largest probability of ${m.name} on it is ${f2(m.planes[z])}` : ''}${
        on ? `; ROI ${roi.label} is on it` : ''}`)}"><i class="naming-plane-bar"><i style="height:${Math.round(share * 100)}%"></i></i>${z + 1}${on ? '<b class="naming-plane-dot"></b>' : ''}</button>`;
    }).join('')}<span class="muted small naming-planes-hint">(wheel, or ↑ ↓)</span>`;
  }

  _renderSliders() {
    const { c, q } = this, box = q('.naming-sliders'), on = Boolean(this.canWhatIf);
    for (const i of box.querySelectorAll('input')) {
      const k = i.dataset.w, v = k === 'max' ? this.max : this.w[k], own = k === 'max' ? c.maxCost : c.weights[k];
      if (k === 'max' && Number.isFinite(v)) { i.min = Math.min(0.2, v); i.max = Math.max(1, v); }
      i.value = Number.isFinite(v) ? v : 0;
      i.disabled = !on || (k === 'atlas' && !c.hasAtlas);
      i.nextElementSibling.textContent = !Number.isFinite(v) ? '—' : `${k === 'max' ? v.toFixed(3) : v.toFixed(2)}${v === own ? '' : ` (the model's: ${own})`}`;
    }
    q('.naming-reset').disabled = !this._whatIf() && this.max === c.maxCost && TERMS.every((k) => this.w[k] === c.weights[k]);
    q('.naming-whatif').classList.toggle('off', !on);
  }

  // what the names are now, against the model's
  _renderOutcome() {
    const { c, s, esc } = this, out = this.q('.naming-outcome');
    const named = s.names.filter((j) => j >= 0).length;
    if (!c.full) { out.innerHTML = `${named} of ${plural(c.nL, 'ROI')} named.`; return; }
    const sum = TERMS.reduce((t, k) => t + (k === 'atlas' && !c.hasAtlas ? 0 : this.w[k]), 0);
    if (!(sum > 0)) { out.innerHTML = 'Every weight is 0: there is no cost to name by.'; return; }
    const why = { costly: 0, taken: 0, unsettled: 0 };
    for (let i = 0; i < c.nL; i++) { const y = whyUnnamed(c, s, i, this.max); if (y) why[y] += 1; }
    const rest = `Of the ${c.nL - named} not named, every glomerulus costs more than that for ${why.costly}${why.taken ? `; for ${why.taken}, each one that costs no more names another ROI` : ''}${
      why.unsettled ? `; and ${why.unsettled} ${why.unsettled === 1 ? 'has' : 'have'} one free that costs no more, as the costs are now` : ''}.`;
    const unsettled = s.settled ? '' : ` <span class="muted" title="The layout term of an ROI depends on its neighbors' names, so the ROIs are matched again with it, ${c.rounds} times at most: here the names still changed in the last round. As in al-diag-seg, the last round's names stand, and the costs shown are of them.">The matching did not settle in ${c.rounds} rounds.</span>`;
    if (!this._whatIf()) {
      out.innerHTML = `<b>${named}</b> of ${plural(c.nL, 'ROI')} named, as the model named them. ${rest}${unsettled}`;
      return;
    }
    const was = this.model.names, lab = (i) => c.labels[i], nm = (j) => esc(c.names[j]), gained = [], lost = [], other = [];
    for (let i = 0; i < c.nL; i++) {
      if (s.names[i] === was[i]) continue;
      if (was[i] < 0) gained.push(`${lab(i)} (${nm(s.names[i])})`);
      else if (s.names[i] < 0) lost.push(`${lab(i)} (${nm(was[i])})`);
      else other.push(`${lab(i)} (${nm(was[i])} → ${nm(s.names[i])})`);
    }
    const list = (what, items) => (items.length ? ` ${what}: ${items.join(', ')}.` : '');
    out.innerHTML = `<b>${named}</b> of ${plural(c.nL, 'ROI')} named; the model named ${was.filter((j) => j >= 0).length}.${
      !gained.length && !lost.length && !other.length ? ' The same ROIs, by the same names.' : ''}${list('Named now, not by the model', gained)}${
      list('No longer named', lost)}${list('Named otherwise', other)} ${rest}${unsettled}`;
  }

  // every ROI as a dot at what its name costs (or, not named, what its cheapest glomerulus would), and the most a name may cost
  _renderStrip() {
    const { c, s, esc } = this, box = this.q('.naming-strip');
    const W = 440, left = 10, width = W - 20, r = 3.6, step = 8.4;
    const dots = [];
    for (let i = 0; i < c.nL; i++) {
      const named = s.names[i] >= 0, j = named ? s.names[i] : this._cheapest(i);
      if (j >= 0) dots.push({ i, j, named, cost: s.cost[i * c.nN + j] });
    }
    if (!dots.length || !Number.isFinite(this.max)) { box.innerHTML = '<p class="muted small">No costs to show.</p>'; return; }
    const hi = Math.max(1, Math.ceil((this.max + 0.05) * 10) / 10), nb = Math.round(hi / BIN), stacks = new Map();
    dots.sort((a, b) => a.cost - b.cost);
    for (const d of dots) {
      d.bin = clamp(Math.floor(d.cost / BIN), 0, nb - 1);
      d.k = stacks.get(d.bin) || 0;
      stacks.set(d.bin, d.k + 1);
    }
    const tall = Math.max(...stacks.values()), base = 14 + tall * step, H = base + 26, x = (v) => left + (v / hi) * width;
    this.stripScale = { W, left, width, hi };
    const ticks = Array.from({ length: Math.round(hi / 0.2) + 1 }, (_, k) => k * 0.2);
    const xm = x(this.max);
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Every ROI by the cost of its name">
      <line class="naming-axis" x1="${left}" y1="${base}" x2="${left + width}" y2="${base}"/>
      ${ticks.map((t) => `<line class="naming-axis" x1="${x(t)}" y1="${base}" x2="${x(t)}" y2="${base + 4}"/><text class="naming-tick" x="${x(t)}" y="${base + 15}">${+t.toFixed(1)}</text>`).join('')}
      <text class="naming-tick naming-axis-name" x="${left + width}" y="${base + 25}">cost</text>
      ${dots.map((d) => {
    const l = c.labels[d.i], changed = this._whatIf() && s.names[d.i] !== this.model.names[d.i];
    const tip = `ROI ${l}: ${d.named ? `named ${c.names[d.j]}, at ${this._fc(d.cost)}` : `not named; ${c.names[d.j]} costs least for it, ${this._fc(d.cost)}`}${
      changed ? ` (the model: ${this.model.names[d.i] >= 0 ? c.names[this.model.names[d.i]] : 'not named'})` : ''}`;
    return `<circle class="naming-dot${d.named ? ' named' : ''}${l === this.sel.roi ? ' chosen' : ''}${changed ? ' changed' : ''}" cx="${x((d.bin + 0.5) * BIN)}"
          cy="${base - r - 2 - d.k * step}" r="${r}" data-roi="${l}" data-tip="${esc(tip)}"/>`;
  }).join('')}
      <line class="naming-thr" x1="${xm}" y1="2" x2="${xm}" y2="${base + 4}"/>
      <text class="naming-thr-text" x="${xm > W - 150 ? xm - 5 : xm + 5}" y="10" text-anchor="${xm > W - 150 ? 'end' : 'start'}">most a name may cost: ${+this.max.toFixed(3)}</text>
      <rect class="naming-thr-grab${this.canWhatIf ? '' : ' off'}" x="${xm - 7}" y="0" width="14" height="${base + 4}"/>
    </svg>
    <p class="naming-legend small muted"><span><svg viewBox="0 0 10 10"><circle class="naming-dot named" cx="5" cy="5" r="3.6"/></svg> named</span>
      <span><svg viewBox="0 0 10 10"><circle class="naming-dot" cx="5" cy="5" r="3.6"/></svg> not named: its cheapest glomerulus</span>
      ${this._whatIf() ? '<span><svg viewBox="0 0 10 10"><circle class="naming-dot named changed" cx="5" cy="5" r="3.6"/></svg> not as the model named it</span>' : ''}</p>
    ${this._means(dots)}`;
  }

  // What those costs are made of, on average: a bar of the terms' mean shares for the ROIs named, and one for those not named (of
  // the glomerulus that costs least for each): which terms keep ROIs from a name
  _means(dots) {
    const { c, s, esc } = this, w = this.w, on = TERMS.filter((k) => w[k] && !(k === 'atlas' && !c.hasAtlas));
    const groups = [['named', dots.filter((d) => d.named)], ['not named', dots.filter((d) => !d.named)]].filter(([, ds]) => ds.length);
    if (!groups.length || !on.length || !(s.sum > 0)) return '';
    const rows = groups.map(([what, ds]) => {
      const mean = Object.fromEntries(on.map((k) => [k, ds.reduce((t, d) => t + (this._term(k, d.i * c.nN + d.j) || 0), 0) / ds.length]));
      return { what, n: ds.length, mean, cost: ds.reduce((t, d) => t + d.cost, 0) / ds.length };
    });
    const hi = Math.max(this.max * 1.2, ...rows.map((r) => r.cost)) * 1.02;
    return `<div class="naming-rows naming-means" title="The mean, over those ROIs, of each term's share of the cost: of the name an ROI has, or, for one not named, of the glomerulus that costs least for it">
      ${rows.map((r) => `<div class="naming-row"><span class="naming-row-name">${r.what} <span class="muted">${r.n} ROIs, on average</span></span>
        <span class="naming-track">${on.map((k) => `<i class="naming-seg naming-sw-${k}" style="width:${(w[k] * r.mean[k] / s.sum / hi) * 100}%" data-tip="${esc(
    `${LABEL[k]}: ${f3(r.mean[k])} on average × weight ${w[k]} ÷ ${+s.sum.toFixed(2)} = ${f3(w[k] * r.mean[k] / s.sum)} of the cost`)}"></i>`).join('')}<b class="naming-track-max"
          style="left:${(this.max / hi) * 100}%"></b></span><span class="naming-row-cost">${f3(r.cost)}</span></div>`).join('')}</div>`;
  }

  // the pairs listed: of the ROI chosen with the glomeruli that cost least for it, or of the glomerulus shown with its ROIs
  _pairs() {
    const { c, s } = this, nN = c.nN, cost = s.cost;
    const i0 = this.rowOf.get(this.sel.roi), j0 = this.colOf.get(this.sel.glom);
    const known = (i, j) => !Number.isNaN(cost[i * nN + j]);
    if (this.by === 'roi') {
      if (i0 == null) return [];
      const js = Array.from({ length: nN }, (_, j) => j).filter((j) => known(i0, j)).sort((a, b) => cost[i0 * nN + a] - cost[i0 * nN + b]);
      const out = js.slice(0, SHOWN);
      for (const j of [s.names[i0], j0]) if (j != null && j >= 0 && known(i0, j) && !out.includes(j)) out.push(j);
      return out.map((j) => ({ i: i0, j }));
    }
    if (j0 == null) return [];
    const is = Array.from({ length: c.nL }, (_, i) => i).filter((i) => known(i, j0)).sort((a, b) => cost[a * nN + j0] - cost[b * nN + j0]);
    const out = is.slice(0, SHOWN);
    for (const i of [s.names.indexOf(j0), i0]) if (i != null && i >= 0 && known(i, j0) && !out.includes(i)) out.push(i);
    return out.map((i) => ({ i, j: j0 }));
  }

  _renderDetail() {
    const { c, s, esc, q } = this, body = q('.naming-detail-body'), nN = c.nN;
    for (const b of q('.naming-by').children) b.classList.toggle('active', b.dataset.by === this.by);
    const i0 = this.rowOf.get(this.sel.roi), j0 = this.colOf.get(this.sel.glom);
    const roiName = (i) => `ROI ${c.labels[i]}${this._name(i) ? ` (${esc(this._name(i))})` : ''}`;
    q('.naming-detail-title').innerHTML = this.by === 'roi' ? (i0 == null ? 'Costs' : `What naming ${roiName(i0)} costs`)
      : j0 == null ? 'Costs' : `What naming an ROI ${esc(c.names[j0])} costs`;
    const pairs = this._pairs();
    if (!pairs.length) {
      body.innerHTML = `<p class="muted small">${this.by === 'roi' ? 'Click an ROI on the image, or a dot above, for what each glomerulus costs for it.'
        : 'Choose a glomerulus, in the list below or the menu over the image, for what it costs for each ROI.'}</p>`;
      return;
    }
    const w = this.w, sum = s.sum, on = TERMS.filter((k) => w[k] && !(k === 'atlas' && !c.hasAtlas));
    const hi = Math.max(this.max * 1.2, ...pairs.map((p) => s.cost[p.i * nN + p.j])) * 1.02;
    const taken = new Map();
    s.names.forEach((j, i) => { if (j >= 0) taken.set(j, i); });
    const rows = pairs.map(({ i, j }) => {
      const k = i * nN + j, cost = s.cost[k], mine = s.names[i] === j, other = this.by === 'roi' ? taken.get(j) : (s.names[i] >= 0 ? i : undefined);
      const segs = on.map((key) => {
        const v = this._term(key, k);
        if (Number.isNaN(v)) return '';
        const more = key === 'atlas' ? ` The glomerulus's mean probability over the ROI's voxels is ${f2(1 - v)}.`
          : key === 'position' ? ` The ROI's middle is ${v >= 1 ? '4 SDs or more' : `${f2(v * 4)} SDs`} from where the glomerulus is expected.`
            : key === 'tuning' && c.reference[j] ? ' Fixed: only in the reference atlas, its tuning is not known.' : '';
        return `<i class="naming-seg naming-sw-${key}" style="width:${(w[key] * v / sum / hi) * 100}%" data-tip="${esc(`${LABEL[key]}: ${f3(v)} × weight ${w[key]} ÷ ${+sum.toFixed(2)} = ${f3(w[key] * v / sum)} of the cost.${more}`)}"></i>`;
      }).join('');
      const label = this.by === 'roi' ? `${esc(c.names[j])}${c.reference[j] ? '<sup title="Only in the reference atlas: its tuning and shape terms are fixed">ref</sup>' : ''}` : `ROI ${c.labels[i]}`;
      const note = mine ? '<span class="naming-mark">named</span>'
        : this.by === 'roi' ? (other != null ? `<span class="muted" title="Each glomerulus names one ROI: this one names ROI ${c.labels[other]}">names ROI ${c.labels[other]}</span>` : '')
          : (other != null ? `<span class="muted">named ${esc(c.names[s.names[i]])}</span>` : '');
      const active = this.by === 'roi' ? j === j0 : i === i0;
      return `<button type="button" class="naming-row${active ? ' active' : ''}${mine ? ' mine' : ''}" data-pair="${i},${j}" title="${this.by === 'roi' ? 'Show this glomerulus on the image' : 'Choose this ROI'}">
        <span class="naming-row-name">${label}${note}</span>
        <span class="naming-track">${segs}<b class="naming-track-max" style="left:${(this.max / hi) * 100}%"></b></span>
        <span class="naming-row-cost${cost > this.max ? ' over' : ''}">${this._fc(cost)}</span></button>`;
    }).join('');
    const legend = `<p class="naming-legend small muted">${TERMS.map((k) => `<span${on.includes(k) ? '' : ' class="off" title="Off (weight 0)"'}><i class="naming-sw naming-sw-${k}"></i>${esc(LABEL[k].toLowerCase())}</span>`).join('')}
      <span><b class="naming-track-max sample"></b>most a name may cost</span></p>`;
    const partial = c.full ? '' : '<p class="muted small">Only each ROI\'s few best glomeruli are known until the fit worker has worked out the rest.</p>';
    body.innerHTML = `${this._verdict(i0, j0)}${legend}<div class="naming-rows">${rows}</div>${partial}${i0 != null && j0 != null ? this._explain(i0, j0) : ''}`;
  }

  // in a sentence: whether the ROI chosen (or the glomerulus shown) has a name, and why not
  _verdict(i0, j0) {
    const { c, s, esc } = this, nN = c.nN, max = f3(this.max);
    if (this.by === 'roi' && i0 != null) {
      const j = s.names[i0];
      if (j >= 0) return `<p>Named <b>${esc(c.names[j])}</b>: that costs ${this._fc(s.cost[i0 * nN + j])}, no more than the most a name may cost (${max}).</p>`;
      const best = this._cheapest(i0);
      if (best < 0) return '<p>Not named.</p>';
      const why = c.full ? whyUnnamed(c, s, i0, this.max) : (s.cost[i0 * nN + best] > this.max ? 'costly' : 'taken');
      if (why === 'costly') {
        return `<p><b>Not named</b>: the glomerulus that costs least for it, <b>${esc(c.names[best])}</b>, costs ${this._fc(s.cost[i0 * nN + best])}, more than the most a name may cost (${max}).</p>`;
      }
      if (why === 'taken') {
        return `<p><b>Not named</b>, though some glomeruli cost no more for it than a name may (${max}): each glomerulus names one ROI, and they name the ROIs
          that, all together, cost least.</p>`;
      }
      return `<p><b>Not named</b>, though a glomerulus that names no ROI costs no more for it than a name may (${max}), as the costs are now: the layout term of
        each ROI depends on the names of its neighbors, and the matching stopped (after ${c.rounds} rounds) while names still changed.</p>`;
    }
    if (this.by === 'glom' && j0 != null) {
      const i = s.names.indexOf(j0);
      if (i >= 0) return `<p><b>${esc(c.names[j0])}</b> names ROI ${c.labels[i]}, at ${this._fc(s.cost[i * nN + j0])}.</p>`;
      const best = this._cheapestRoi(j0);
      if (best < 0) return `<p><b>${esc(c.names[j0])}</b> names no ROI.</p>`;
      const cost = s.cost[best * nN + j0];
      return `<p><b>${esc(c.names[j0])}</b> names no ROI: the ROI it costs least for, ROI ${c.labels[best]}, ${cost > this.max ? `costs ${this._fc(cost)}, more than the most a name may cost (${max})`
        : `costs ${this._fc(cost)}, but ${s.names[best] >= 0 ? `is named ${esc(c.names[s.names[best]])}, which costs less for the ROIs all together` : 'the matching left it unnamed'}`}.</p>`;
    }
    return '';
  }

  // Of one ROI and one glomerulus: what the atlas and position terms are made of, and the mean probability the ROI would need for the pair
  // to cost no more than a name may
  _explain(i, j) {
    const { c, s, esc } = this, k = i * c.nN + j, name = esc(c.names[j]), out = [];
    const a = c.terms.atlas[k], pos = c.terms.position[k], m = this.v.maps[this.mapOf.get(c.names[j])];
    if (!Number.isNaN(a)) {
      out.push(`<b>Atlas overlap.</b> Over ROI ${c.labels[i]}'s voxels, the atlas' probability of ${name} is ${f2(1 - a)} on average${m ? ` (its map reaches ${f2(m.peak)} at
        most on this fly's planes)` : ''}, so the term is ${f3(a)}.`);
    }
    if (!Number.isNaN(pos)) {
      const sd = c.full ? c.posSd[j] : this.v.glomeruli[c.names[j]]?.sd, d = this._apart(i, j), sz = this.v.spacing?.[0];
      out.push(`<b>Position.</b> Its middle is ${d ? `${d.um.toFixed(1)} µm` : pos >= 1 ? '4 SDs or more' : `${f2(pos * 4)} SDs`} from where the atlas expects ${name}${
        d && d.across >= 1 ? ` (${d.across.toFixed(1)} µm of it across planes${sz ? `, which are ${sz} µm apart` : ''})` : ''}${
        sd ? `. The SD of ${name}'s position is ${sd.toFixed(1)} µm${d ? `, so that is ${pos >= 1 ? '4 SDs or more, counted as 4' : `${f2(pos * 4)} SDs`}` : ''}` : ''}: the term is ${f3(pos)}.`);
    }
    const tune = c.terms.tuning[k], shape = c.terms.shape[k], lay = this._term('layout', k);
    if (!Number.isNaN(tune)) {
      out.push(c.reference[j] ? `<b>Odor tuning.</b> ${name} is only in the reference atlas, so its tuning (and its shape) are not known: those terms are fixed at ${f3(tune)}.`
        : `<b>Odor tuning.</b> Its tuning and ${name}'s mean tuning in the training flies correlate at ${f2(1 - tune)}, so the term is ${f3(tune)}.${this._tuningPlot(i, j)}`);
    }
    const more = [...(!Number.isNaN(shape) && this.w.shape && !c.reference[j] ? [`its volume and axis lengths are ${shape >= 1 ? '4 SDs or more' : `${f2(shape * 4)} SDs`} from ${name}'s (shape: ${f3(shape)})`] : []),
      ...(!Number.isNaN(lay) && this.w.layout ? [`its offsets to its named neighbors are ${lay >= 1 ? '4 SDs or more' : `${f2(lay * 4)} SDs`} from those the atlas expects between ${name} and their glomeruli (layout: ${f3(lay)})`] : [])];
    if (more.length) out.push(`<b>The rest.</b> ${more.join('; ').replace(/^i/, 'I')}.`);
    const need = this._needed(i, j);
    if (need != null) {
      const has = 1 - a;
      const even = need <= 0 ? `Whatever the atlas said of it, this pair would cost no more than a name may (${f3(this.max)}): its other terms leave room for an atlas term of 1.`
        : need > 1 ? `Whatever the atlas said of it, this pair would cost more than a name may (${f3(this.max)}): its other terms alone do.`
          : `With its other terms as they are, this pair costs no more than a name may (${f3(this.max)}) where that mean probability is at least <b>${apart(need, has)[0]}</b>${
            m && need > m.peak ? `: more than ${name}'s map reaches anywhere on these planes, so no ROI here could have it`
              : `${has >= need ? ' (it is)' : ` (it is ${apart(need, has)[1]})`}; the dashed white line on the image is where ${name}'s probability is that`}.`;
      out.push(`<b>The threshold.</b> ${even}${s.names[i] !== j && s.cost[k] <= this.max ? ' Costing no more is not enough for the name: each glomerulus names one ROI, and each ROI has one name.' : ''}`);
    }
    return out.length ? `<div class="naming-explain"><h4>ROI ${c.labels[i]} as ${name}</h4>${out.map((t) => `<p class="small">${t}</p>`).join('')}</div>` : '';
  }

  // ROI i's odor tuning against glomerulus j's mean one in the training flies (each z-scored over the fly's odors): a dot per odor,
  // on the same scale both ways; '' until every pair's cost is known (the tunings come with them)
  _tuningPlot(i, j) {
    const t = this.doc?.tuning, own = t?.rois[i], its = t?.glomeruli[j];
    if (!own || !its) return '';
    const { c, esc } = this, S = 150, m = 22, all = [...own, ...its], lo = Math.min(0, ...all), hi = Math.max(0, ...all), span = hi - lo || 1;
    const x = (v) => m + ((v - lo) / span) * (S - m - 6), y = (v) => S - m - ((v - lo) / span) * (S - m - 6);
    return `<span class="naming-tuning"><svg viewBox="0 0 ${S} ${S}" role="img" aria-label="The ROI's response to each odor against the glomerulus's">
      <line class="naming-axis" x1="${x(lo)}" y1="${y(0)}" x2="${x(hi)}" y2="${y(0)}"/><line class="naming-axis" x1="${x(0)}" y1="${y(lo)}" x2="${x(0)}" y2="${y(hi)}"/>
      <line class="naming-same" x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}"/>
      ${own.map((v, n) => `<circle class="naming-odor" cx="${x(its[n])}" cy="${y(v)}" r="3.4" data-tip="${esc(`${this.doc.odors[n]}: ROI ${c.labels[i]} ${f2(v)}, ${c.names[j]} ${f2(its[n])} (z-scores over the odors)`)}"/>`).join('')}
      <text class="naming-tick" x="${(x(lo) + x(hi)) / 2}" y="${S - 4}">${esc(c.names[j])}, training flies</text>
      <text class="naming-tick" transform="translate(9 ${(y(lo) + y(hi)) / 2}) rotate(-90)">ROI ${c.labels[i]}</text></svg>
      <span class="small muted">A dot per odor: its response against ${esc(c.names[j])}'s, each as z-scores over the fly's ${own.length} odors. On the line, they are the same.</span></span>`;
  }

  // every glomerulus the model can name an ROI: how many flies are behind its map, how sure the map is on this fly, and the ROI it names
  _renderGloms() {
    const { c, s, esc, v } = this, nN = c.nN, namedBy = new Map();
    s.names.forEach((j, i) => { if (j >= 0) namedBy.set(j, i); });
    let rows = c.names.map((n, j) => {
      const m = v.maps[this.mapOf.get(n)], named = namedBy.has(j), i = named ? namedBy.get(j) : this._cheapestRoi(j);
      return { n, j, i, named, flies: v.glomeruli[n]?.flies ?? null, peak: m ? m.peak : null, cost: i >= 0 ? s.cost[i * nN + j] : NaN, sd: c.full ? c.posSd[j] : v.glomeruli[n]?.sd };
    });
    if (this.q('.naming-unnamed-only').checked) rows = rows.filter((r) => !r.named);
    const { key, dir } = this.sort, val = (r) => (key === 'named' ? (r.named ? 0 : 1) : r[key]);
    rows.sort((a, b) => (key === 'name' ? dir * natural(a.n, b.n)
      : (val(a) == null || Number.isNaN(val(a))) - (val(b) == null || Number.isNaN(val(b))) || dir * (val(a) - val(b)) || natural(a.n, b.n)));
    const th = (k, label, tip) => `<th data-sort="${k}" title="${esc(tip)}"${this.sort.key === k ? ` aria-sort="${dir > 0 ? 'ascending' : 'descending'}"` : ''}>${label}${
      this.sort.key === k ? (dir > 0 ? ' ▴' : ' ▾') : ''}</th>`;
    this.q('.naming-gloms').innerHTML = `<table class="odor-table naming-gloms-table"><thead><tr>
      ${th('name', 'Glomerulus', 'Click a glomerulus to see where the atlas puts it, and what it costs for each ROI')}
      ${th('flies', 'Flies', "How many flies are behind its map in the model's atlas: the training flies that have it (their ROIs registered to each other and averaged), and 1 for the reference atlas. A map of few flies is surer of itself than it should be.")}
      ${th('peak', 'Peak', "The largest probability its map reaches on this fly's planes: no ROI's atlas term for it can be under 1 minus this")}
      ${th('sd', 'SD', 'The SD of its position among the training flies (µm), which the position term counts distances in')}
      ${th('named', 'Names', 'The ROI it names; else, in grey, the ROI it costs least for')}
      ${th('cost', 'Cost', 'What that costs')}</tr></thead><tbody>${rows.map((r) => `<tr data-glom="${esc(r.n)}"${r.n === this.sel.glom ? ' class="active"' : ''}>
        <th scope="row">${esc(r.n)}${c.reference[r.j] ? ' <sup title="Only in the reference atlas (Grabe et al. 2016): its tuning and shape are not known, and their terms are fixed">ref</sup>' : ''}</th>
        <td>${r.flies == null ? '—' : Math.round(r.flies)}</td><td>${r.peak == null ? '<span class="muted" title="The atlas has no map of it">—</span>' : f2(r.peak)}</td>
        <td>${r.sd ? r.sd.toFixed(1) : '—'}</td>
        <td>${r.i < 0 ? '—' : r.named ? `ROI ${c.labels[r.i]}` : `<span class="muted" title="It names no ROI: this is the one it costs least for">(ROI ${c.labels[r.i]})</span>`}</td>
        <td${r.cost > this.max ? ' class="muted"' : ''}>${this._fc(r.cost)}</td></tr>`).join('')}</tbody></table>`;
  }

  // what the marks on the image are
  _renderKey() {
    const { esc } = this, ov = this.vol ? this._overlay() : null, parts = [];
    if (ov) {
      const top = this.q('.naming-scale').value === 'peak' ? ov.peak : 1;
      parts.push(`<span><i class="naming-key-ramp"></i> probability 0 to ${f2(top)}${this.show === 'one' ? ` of ${esc(ov.name)}` : ' of the most probable glomerulus'}</span>`);
      if (this.show === 'one') {
        parts.push(`<span><i class="naming-key-line"></i> where it is ${LEVELS.join(', ')} (heavier: higher)</span>`);
        const need = this._needed(this.rowOf.get(this.sel.roi) ?? -1, ov.j);
        if (need != null && need > 0 && need <= 1) {
          parts.push(`<span><i class="naming-key-need"></i> where it is ${f2(need)}: what ROI ${this.sel.roi} needs on average to cost no more than a name may${
            need > ov.peak ? ' (nowhere)' : ''}</span>`);
        }
        const g = this._expected();
        if (g) parts.push(`<span>⊕ where the atlas expects ${esc(g.name)}${g.sd ? `; rings: ${RINGS.join(', ')} SDs of its position (${g.sd.toFixed(1)} µm each), as they cut this plane` : ''}</span>`);
      } else parts.push(`<span><i class="naming-key-line thin"></i> between the places of two glomeruli (none under ${FLOOR})</span>`);
    }
    parts.push('<span><i class="naming-key-roi"></i> ROI named</span>', '<span><i class="naming-key-roi dashed"></i> not named</span>', '<span><i class="naming-key-roi chosen"></i> chosen</span>');
    this.q('.naming-key').innerHTML = parts.join('');
  }
}
