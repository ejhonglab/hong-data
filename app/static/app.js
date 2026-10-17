// Odor response database — frontend (no build step, no dependencies).
import { Heatmap } from './heatmap.js';
import { COLORMAPS, colormapsOfKind, makeNorm, drawColorbar, fmtNum } from './colormaps.js';
import { MAX_CLUSTER_ITEMS, ClusterPool, layoutDendrogram } from './cluster.js';
import { exportFigure } from './figure.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode etc. */ } },
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
}

async function api(path, { method = 'GET', json, form } = {}) {
  const init = { method, headers: {} };
  if (json !== undefined) { init.headers['content-type'] = 'application/json'; init.body = JSON.stringify(json); }
  if (form) init.body = form;
  let res;
  try { res = await fetch(path, init); } catch (e) { throw new Error(`cannot reach the server (${e.message})`); }
  let body = null;
  const text = await res.text();
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }
  if (!res.ok && !(body && (body.errors || body.error))) {
    throw new Error(`${res.status} ${res.statusText}${text ? ': ' + text.slice(0, 200) : ''}`);
  }
  if (!res.ok && body && body.error && !body.errors) throw new Error(body.error);
  return body;
}

function decodeValues(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // server writes little-endian float32; nearly every client is little-endian too
  return new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}

function msgBox(kind, html) {
  const d = document.createElement('div');
  d.className = `msg ${kind}`;
  d.innerHTML = html;
  return d;
}
function listHtml(items) { return `<ul>${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`; }

function fmtBytes(n) { return n < 1e6 ? `${Math.max(1, Math.round(n / 1e3))} kB` : `${(n / 1e6).toFixed(1)} MB`; }

function saveBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
}

function setStatus(el, text, kind = '') {
  el.textContent = text;
  el.classList.toggle('error', kind === 'error');
}

// Fetched (rather than a plain link) so that the wait while the server builds a large
// zip, and any error, show up in `status`.
async function download(url, status) {
  if (status.dataset.busy) return;
  status.dataset.busy = '1';
  setStatus(status, 'Preparing the download…');
  try {
    const res = await fetch(url);
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.error || `${res.status} ${res.statusText}`);
    }
    const total = Number(res.headers.get('content-length')) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      setStatus(status, `Downloading… ${fmtBytes(got)}${total ? ` of ${fmtBytes(total)}` : ''}`);
    }
    const cd = res.headers.get('content-disposition') || '';
    const m = cd.match(/filename\*=utf-8''([^;]+)/i) || cd.match(/filename="?([^";]+)"?/i);
    saveBlob(new Blob(chunks, { type: 'application/zip' }), m ? decodeURIComponent(m[1]) : 'download.zip');
    setStatus(status, '');
  } catch (e) {
    setStatus(status, `Download failed: ${e.message}`, 'error');
  } finally {
    delete status.dataset.busy;
  }
}

// <details class="menu"> dropdowns: onPick(button.dataset) for the chosen entry
function wireMenu(menu, onPick) {
  $$('.menu-list button', menu).forEach((b) => b.addEventListener('click', () => { menu.open = false; onPick(b.dataset); }));
}
document.addEventListener('click', (e) => { $$('details.menu[open]').forEach((m) => { if (!m.contains(e.target)) m.open = false; }); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $$('details.menu[open]').forEach((m) => { m.open = false; }); });

// ------------------------------------------------------------- sorting helpers
function roiKey(roi) {
  return /^\d+$/.test(roi) ? [0, parseInt(roi, 10), ''] : [1, 0, roi.toLowerCase()];
}
function cmpRoi(a, b) {
  const ka = roiKey(a), kb = roiKey(b);
  return ka[0] - kb[0] || ka[1] - kb[1] || (ka[2] < kb[2] ? -1 : ka[2] > kb[2] ? 1 : 0);
}
function cmpFly(a, b) {
  return a.date < b.date ? -1 : a.date > b.date ? 1 : a.fly_num - b.fly_num;
}
function odorComponents(odorStr) {
  if (odorStr === 'solvent') return [];
  return odorStr.split(' + ').map((c) => {
    const m = c.match(/^(.*?) @ (-?[\d.]+(?:e-?\d+)?)$/);
    return m ? { name: m[1].toLowerCase(), conc: parseFloat(m[2]) } : { name: c.toLowerCase(), conc: 0 };
  });
}
function cmpOdor(a, b) {
  const ca = odorComponents(a), cb = odorComponents(b);
  if (ca.length !== cb.length && (ca.length === 0 || cb.length === 0)) return ca.length - cb.length; // solvent first
  for (let i = 0; i < Math.max(ca.length, cb.length); i++) {
    if (!ca[i]) return -1; if (!cb[i]) return 1;
    if (ca[i].name !== cb[i].name) return ca[i].name < cb[i].name ? -1 : 1;
    if (ca[i].conc !== cb[i].conc) return ca[i].conc - cb[i].conc;
  }
  return 0;
}

// --------------------------------------------------------- neural data model
// data: {units, trials, values: Float32Array (units x trials)}
const flyKey = (u) => `${u.date} fly ${u.fly_num}`;

// contiguous runs of equal key(item): [{key, start, end}]
function runsOf(items, key) {
  const out = [];
  items.forEach((it, i) => {
    const k = key(it);
    if (out.length && out[out.length - 1].key === k) out[out.length - 1].end = i + 1;
    else out.push({ key: k, start: i, end: i + 1 });
  });
  return out;
}

// conditions (panel, odor) grouped by panel, and units, in display order before any
// clustering ('ungrouped' rows use the fly order)
function neuralOrder(data, { rowOrder, roiSort, odorOrder }) {
  const { units, trials } = data;
  const condIndex = new Map();
  const conds = [];
  trials.forEach((t, j) => {
    const key = `${t.panel ?? ''}\u0000${t.odor}`;
    let c = condIndex.get(key);
    if (!c) { c = { panel: t.panel, odor: t.odor, trials: [] }; condIndex.set(key, c); conds.push(c); }
    c.trials.push(j);
  });
  const panelOrder = [];
  conds.forEach((c) => { if (!panelOrder.includes(c.panel)) panelOrder.push(c.panel); });
  const ordered = conds.map((c, i) => ({ c, i }));
  ordered.sort((x, y) => {
    const p = panelOrder.indexOf(x.c.panel) - panelOrder.indexOf(y.c.panel);
    if (p) return p;
    return odorOrder === 'name' ? cmpOdor(x.c.odor, y.c.odor) || x.i - y.i : x.i - y.i;
  });
  const firstRoi = new Map();
  units.forEach((u, i) => { if (!firstRoi.has(u.roi)) firstRoi.set(u.roi, i); });
  const unitIdx = units.map((u, i) => i);
  const roiCmp = (i, j) => (roiSort === 'sorted' ? cmpRoi(units[i].roi, units[j].roi) : firstRoi.get(units[i].roi) - firstRoi.get(units[j].roi));
  if (rowOrder === 'roi') unitIdx.sort((i, j) => roiCmp(i, j) || cmpFly(units[i], units[j]) || i - j);
  else unitIdx.sort((i, j) => cmpFly(units[i], units[j]) || roiCmp(i, j) || i - j);
  return { conds: ordered.map((x) => x.c), panelOrder, unitIdx };
}

// units (in unitIdx order) x columns, each column the mean over its trials
function meanOver(values, nT, unitIdx, colTrials) {
  const nR = unitIdx.length, nC = colTrials.length;
  const out = new Float32Array(nR * nC);
  for (let r = 0; r < nR; r++) {
    const base = unitIdx[r] * nT;
    for (let c = 0; c < nC; c++) {
      const ts = colTrials[c];
      let s = 0, n = 0;
      for (let k = 0; k < ts.length; k++) { const v = values[base + ts[k]]; if (v === v) { s += v; n++; } }
      out[r * nC + c] = n ? s / n : NaN;
    }
  }
  return out;
}

// Reorders ROIs and/or odors by hierarchical clustering of trial-averaged responses
// (whatever the display mode). ROIs are clustered all together, whichever fly they come
// from; odors within each panel, by their responses across all ROIs.
// -> {unitIdx, conds, rowTrees, colTrees, notes}
async function clusterNeural(pool, data, order, o, onProgress) {
  const { conds, unitIdx } = order;
  const nU = unitIdx.length, nC = conds.length;
  const means = meanOver(data.values, data.trials.length, unitIdx, conds.map((c) => c.trials));
  const jobs = [], plans = [], notes = [];
  const add = (plan, n, d, vectors) => {
    if (n < 2) return;
    if (n > MAX_CLUSTER_ITEMS) { notes.push(`${plan.what} not clustered (${n.toLocaleString()} is over the limit of ${MAX_CLUSTER_ITEMS.toLocaleString()})`); return; }
    jobs.push({ data: vectors(), n, d, metric: o.metric });
    plans.push(plan);
  };
  if (o.clusterRows) add({ kind: 'rows', group: { start: 0, end: nU }, what: 'ROIs' }, nU, nC, () => means);
  if (o.clusterCols) {
    for (const p of runsOf(conds, (c) => c.panel)) {
      const n = p.end - p.start;
      add({ kind: 'cols', group: p, what: p.key != null ? `odors of ${p.key}` : 'odors' }, n, nU, () => {
        const vecs = new Float32Array(n * nU);
        for (let c = 0; c < n; c++) for (let r = 0; r < nU; r++) vecs[c * nU + r] = means[r * nC + p.start + c];
        return vecs;
      });
    }
  }
  const results = jobs.length ? await pool.run(jobs, onProgress) : [];
  const out = { unitIdx: unitIdx.slice(), conds: conds.slice(), rowTrees: [], colTrees: [], notes };
  results.forEach((res, k) => {
    const plan = plans[k];
    const { start, end } = plan.group;
    const items = plan.kind === 'rows' ? unitIdx.slice(start, end) : conds.slice(start, end);
    const target = plan.kind === 'rows' ? out.unitIdx : out.conds;
    res.order.forEach((i, t) => { target[start + t] = items[i]; });
    (plan.kind === 'rows' ? out.rowTrees : out.colTrees).push({ merges: res.merges, leaves: items });
  });
  return out;
}

function buildNeuralModel(data, order, { mode, rowOrder }, clustering) {
  const { units, trials, values } = data;
  const conds = clustering ? clustering.conds : order.conds;
  const unitIdx = clustering ? clustering.unitIdx : order.unitIdx;
  const showGroups = order.panelOrder.length > 1 || order.panelOrder[0] != null;
  const cols = [], colTrials = [], condCenter = new Map();
  for (const c of conds) {
    const group = showGroups ? (c.panel ?? '(no panel)') : null;
    const start = cols.length;
    if (mode === 'trials') {
      const ts = c.trials.slice().sort((a, b) => trials[a].repeat - trials[b].repeat || a - b);
      for (const j of ts) { cols.push({ label: c.odor, group, sub: trials[j].repeat }); colTrials.push([j]); }
    } else {
      cols.push({ label: c.odor, group, sub: null });
      colTrials.push(c.trials);
    }
    condCenter.set(c, (start + cols.length) / 2);
  }
  const rows = unitIdx.map((i) => {
    const u = units[i];
    if (rowOrder === 'roi') return { label: flyKey(u), group: u.roi };
    if (rowOrder === 'none') return { label: `${u.roi} (${flyKey(u)})`, group: null };
    return { label: u.roi, group: flyKey(u) };
  });
  const model = {
    rows, cols, values: meanOver(values, trials.length, unitIdx, colTrials),
    rowTitle: { fly: 'ROI', roi: 'fly', none: 'ROI (fly)' }[rowOrder],
    rowGroupTitle: { fly: 'fly', roi: 'ROI', none: null }[rowOrder],
    nConditions: conds.length, nPanels: order.panelOrder.length, nUnits: units.length, nTrials: trials.length,
    rowDendro: null, colDendro: null,
  };
  if (clustering?.rowTrees.length) {
    const rowOf = new Map(unitIdx.map((u, r) => [u, r + 0.5]));
    model.rowDendro = layoutDendrogram(clustering.rowTrees.map((t) => ({ merges: t.merges, leafPos: t.leaves.map((u) => rowOf.get(u)) })));
  }
  if (clustering?.colTrees.length) {
    model.colDendro = layoutDendrogram(clustering.colTrees.map((t) => ({ merges: t.merges, leafPos: t.leaves.map((c) => condCenter.get(c)) })));
  }
  return model;
}

function dataRange(values, clip) {
  let vmin = Infinity, vmax = -Infinity, n = 0;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (v === v) { n++; if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
  if (!n) return { vmin: 0, vmax: 1 };
  if (!clip || n < 20) return { vmin, vmax };
  const finite = new Float32Array(n);
  for (let i = 0, k = 0; i < values.length; i++) { const v = values[i]; if (v === v) finite[k++] = v; }
  finite.sort();
  return { vmin: finite[Math.floor(0.005 * (n - 1))], vmax: finite[Math.ceil(0.995 * (n - 1))] };
}

// ------------------------------------------------------------- matrix viewer
class MatrixViewer {
  constructor(container, { kind = 'neural' } = {}) {
    this.container = container;
    this.kind = kind;
    container.classList.add('viewer');
    container.innerHTML = '';
    container.appendChild($('#viewer-toolbar').content.cloneNode(true));
    this.toolbar = $('.vt', container);
    this.wrap = document.createElement('div');
    this.wrap.className = 'hm-wrap';
    container.appendChild(this.wrap);
    this.heatmap = new Heatmap(this.wrap);
    this.colorbar = $('.colorbar', container);
    this.opts = Object.assign(
      {
        mode: 'average', rowOrder: 'fly', roiSort: 'file', odorOrder: 'file', clusterRows: false, clusterCols: false, metric: 'correlation',
        cellW: 16, cellH: 16, colormap: 'oklab-div', range: 'twoslope', clip: false,
      },
      LS.get('olfdb.viewer', {}),
    );
    this.data = null;
    this.title = '';
    this.fileStem = 'heatmap';
    this.clusterPool = new ClusterPool();
    this.clusterCache = new Map();
    this._seq = 0;
    this._wire();
  }

  _wire() {
    const tb = this.toolbar;
    $$('[data-mode]', tb).forEach((b) => b.addEventListener('click', () => { this.opts.mode = b.dataset.mode; this._syncToolbar(); this._rebuild(); }));
    $$('select[data-opt]', tb).forEach((s) => s.addEventListener('change', () => { this.opts[s.dataset.opt] = s.value; this._save(); this._rebuild(); }));
    for (const k of ['cellW', 'cellH']) {
      const range = $(`input[data-opt=${k}]`, tb);
      range.addEventListener('input', () => { this.opts[k] = +range.value; $(`[data-out=${k}]`, tb).textContent = range.value; this._save(); this.heatmap.setOptions({ cellW: this.opts.cellW, cellH: this.opts.cellH }); });
    }
    $$('input[type=checkbox][data-opt]', tb).forEach((cb) => cb.addEventListener('change', () => {
      this.opts[cb.dataset.opt] = cb.checked;
      this._save();
      this._syncEnabled();
      this._rebuild();
    }));
    wireMenu($('[data-role=export]', tb), ({ export: format }) => this.exportFigure(format));
  }

  _save() {
    const o = this.opts;
    LS.set('olfdb.viewer', {
      cellW: o.cellW, cellH: o.cellH, colormap: o.colormap, range: o.range, rowOrder: o.rowOrder, roiSort: o.roiSort, odorOrder: o.odorOrder,
      clusterRows: o.clusterRows, clusterCols: o.clusterCols, metric: o.metric,
    });
  }

  _syncToolbar() {
    const tb = this.toolbar, o = this.opts;
    $$('[data-mode]', tb).forEach((b) => b.classList.toggle('active', b.dataset.mode === o.mode));
    const neural = this.kind === 'neural';
    for (const role of ['mode', 'rows', 'cols', 'cluster']) $(`[data-role=${role}]`, tb).hidden = !neural;
    const mapKind = neural ? 'diverging' : (this.data?.meta?.kind || 'sequential');
    const cmSel = $('select[data-opt=colormap]', tb);
    const maps = colormapsOfKind(mapKind);
    cmSel.innerHTML = maps.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join('');
    if (!maps.some((m) => m.id === o.colormap)) o.colormap = maps[0].id;
    cmSel.value = o.colormap;
    $('[data-role=range]', tb).hidden = mapKind !== 'diverging';
    $('input[data-opt=clip]', tb).parentElement.hidden = mapKind === 'binary';
    for (const k of ['rowOrder', 'roiSort', 'odorOrder', 'metric', 'range']) { const s = $(`select[data-opt=${k}]`, tb); if (s) s.value = o[k]; }
    for (const k of ['cellW', 'cellH']) { $(`input[data-opt=${k}]`, tb).value = o[k]; $(`[data-out=${k}]`, tb).textContent = o[k]; }
    $$('input[type=checkbox][data-opt]', tb).forEach((cb) => { cb.checked = !!o[cb.dataset.opt]; });
    this._syncEnabled();
  }

  // orderings that clustering replaces are disabled while it is on (rows show as
  // ungrouped then; the chosen grouping comes back when ROI clustering is turned off)
  _syncEnabled() {
    const tb = this.toolbar, o = this.opts;
    const rows = $('select[data-opt=rowOrder]', tb), roi = $('select[data-opt=roiSort]', tb), odor = $('select[data-opt=odorOrder]', tb);
    rows.disabled = roi.disabled = o.clusterRows;
    rows.value = o.clusterRows ? 'none' : o.rowOrder;
    odor.disabled = o.clusterCols;
    rows.title = roi.title = o.clusterRows ? 'ROIs are ordered by the clustering, across all flies' : '';
    odor.title = o.clusterCols ? 'odors are ordered by the clustering' : '';
    $('select[data-opt=metric]', tb).disabled = !(o.clusterRows || o.clusterCols);
  }

  // title: for the exported figure's metadata; fileStem: its file name
  setNeuralData(payload, { title = '', fileStem = 'heatmap' } = {}) {
    this.kind = 'neural';
    this.data = { units: payload.units, trials: payload.trials, values: decodeValues(payload.values_b64) };
    this.title = title;
    this.fileStem = fileStem;
    this.clusterCache.clear();
    this.opts.colormap = this.opts.colormap in COLORMAPS && COLORMAPS[this.opts.colormap].kind === 'diverging' ? this.opts.colormap : 'oklab-div';
    this._syncToolbar();
    this._rebuild();
  }

  setGenericData(ref, { title = '', fileStem = 'heatmap' } = {}) {
    this.kind = 'generic';
    this.title = title;
    this.fileStem = fileStem;
    const values = decodeValues(ref.values_b64);
    const hasGroups = ref.rows.some((r) => r.group != null);
    let rows = ref.rows.map((r, i) => ({ label: r.label, group: hasGroups ? (r.group ?? '') : null, i }));
    let vals = values;
    if (hasGroups) {
      const first = new Map();
      rows.forEach((r) => { if (!first.has(r.group)) first.set(r.group, r.i); });
      rows = rows.slice().sort((a, b) => first.get(a.group) - first.get(b.group) || a.i - b.i);
      const nC = ref.cols.length;
      vals = new Float32Array(values.length);
      rows.forEach((r, k) => { vals.set(values.subarray(r.i * nC, (r.i + 1) * nC), k * nC); });
    }
    this.data = {
      meta: ref.meta,
      model: {
        rows: rows.map((r) => ({ label: r.label, group: r.group })),
        cols: ref.cols.map((c) => ({ label: c.label, group: null, sub: null })),
        values: vals,
        rowTitle: ref.meta.row_label || 'row',
        rowGroupTitle: hasGroups ? 'group' : null,
      },
    };
    const kind = ref.meta.kind || 'sequential';
    this.opts.colormap = kind === 'binary' ? 'binary' : kind === 'diverging' ? 'oklab-div' : (COLORMAPS[this.opts.colormapSeq]?.kind === 'sequential' ? this.opts.colormapSeq : 'viridis');
    this._syncToolbar();
    this._rebuild();
  }

  async _rebuild() {
    if (!this.data) return;
    const seq = ++this._seq;
    // ROI clustering pools the ROIs of every fly, so the rows are then not grouped
    const o = this.opts.clusterRows ? { ...this.opts, rowOrder: 'none' } : this.opts;
    if (this.kind !== 'neural') {
      if (COLORMAPS[o.colormap]?.kind === 'sequential') { this.opts.colormapSeq = o.colormap; this._save(); }
      this._show(this.data.model);
      return;
    }
    const order = neuralOrder(this.data, o);
    let clustering = null;
    const notes = [];
    if (o.clusterRows || o.clusterCols) {
      const key = [o.rowOrder, o.roiSort, o.odorOrder, o.clusterRows, o.clusterCols, o.metric].join('|');
      clustering = this.clusterCache.get(key);
      if (!clustering) {
        // show the matrix unclustered, with progress, if clustering takes more than a moment
        let shown = false, progress = 'clustering…';
        const timer = setTimeout(() => {
          if (seq !== this._seq) return;
          shown = true;
          this._show(buildNeuralModel(this.data, order, o, null), progress);
        }, 150);
        const onProgress = ({ distances, linkage }) => {
          progress = distances < 1 ? `clustering… distances ${Math.floor(distances * 100)}%` : `clustering… linkage ${Math.floor(linkage * 100)}%`;
          if (shown && seq === this._seq) this._setNote(progress);
        };
        try {
          clustering = await clusterNeural(this.clusterPool, this.data, order, o, onProgress);
          this.clusterCache.set(key, clustering);
        } catch (e) {
          if (e.cancelled) return;
          notes.push(`clustering failed: ${e.message}`);
        } finally {
          clearTimeout(timer);
        }
        if (seq !== this._seq) return;
      }
      if (clustering) {
        notes.push(...clustering.notes);
        if (clustering.rowTrees.length || clustering.colTrees.length) notes.unshift(`clustered by ${o.metric === 'euclidean' ? 'Euclidean distance' : 'correlation'}`);
      }
    }
    this._show(buildNeuralModel(this.data, order, o, clustering), notes.join(' · '));
  }

  _show(model, note = '') {
    const o = this.opts;
    const cm = COLORMAPS[o.colormap] || COLORMAPS['oklab-div'];
    const { vmin, vmax } = dataRange(model.values, o.clip);
    let norm;
    if (cm.kind === 'diverging') norm = makeNorm(o.range === 'symmetric' ? 'symmetric' : 'twoslope', vmin, vmax);
    else if (cm.kind === 'binary') norm = makeNorm('linear', 0, 1);
    else norm = makeNorm('linear', vmin, vmax);
    this.model = model;
    this.norm = norm;
    this.lut = cm.lut;
    this.heatmap.setModel(model, { cellW: o.cellW, cellH: o.cellH, lut: cm.lut, norm });
    drawColorbar(this.colorbar, cm.lut, norm);
    let text;
    if (this.kind === 'neural') {
      text = `${model.rows.length.toLocaleString()} ROIs × ${model.cols.length} ${o.mode === 'trials' ? 'trials' : 'odors'}` +
        (model.nPanels > 1 ? ` in ${model.nPanels} panels` : '') + ` · range ${fmtNum(vmin)} to ${fmtNum(vmax)}`;
    } else {
      text = `${model.rows.length} × ${model.cols.length} · range ${fmtNum(vmin)} to ${fmtNum(vmax)}` + (this.data.meta.value_label ? ` (${this.data.meta.value_label})` : '');
    }
    this.statsText = text;
    this._setNote(note);
  }

  _setNote(note) {
    $('[data-role=stats]', this.toolbar).textContent = note ? `${this.statsText} · ${note}` : this.statsText;
  }

  async exportFigure(format) {
    if (!this.model) return;
    const status = $('[data-role=export-status]', this.toolbar);
    setStatus(status, `Exporting ${format.toUpperCase()}…`);
    try {
      const { blob, note } = await exportFigure(format, this.model, {
        cellW: this.opts.cellW, cellH: this.opts.cellH, lut: this.lut, norm: this.norm, nanColor: this.heatmap.options.nanColor,
        valueLabel: this.data.meta?.value_label || '', title: this.title,
      });
      const suffix = this.kind === 'neural' && this.opts.mode === 'trials' ? '_trials' : '';
      saveBlob(blob, `${this.fileStem}${suffix}.${format}`);
      setStatus(status, note);
    } catch (e) {
      setStatus(status, `Export failed: ${e.message}`, 'error');
    }
  }
}

// ------------------------------------------------------------- tiny markdown
function renderMarkdown(md) {
  const lines = md.replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/(^|[^*])\*([^*]+)\*/g, '$1<i>$2</i>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    .replace(/(^|\s)(https?:\/\/[^\s<]+)/g, '$1<a href="$2">$2</a>');
  while (i < lines.length) {
    const l = lines[i];
    if (/^#{1,6}\s/.test(l)) { const n = l.match(/^#+/)[0].length; out.push(`<h${n}>${inline(l.slice(n).trim())}</h${n}>`); i++; continue; }
    if (/^\|/.test(l)) {
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) { rows.push(lines[i]); i++; }
      const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const body = rows.filter((r) => !/^\|\s*-{2,}/.test(r) && cells(r).some((c) => c !== '')).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('');
      out.push(`<table>${body}</table>`); continue;
    }
    if (/^\s*[-*]\s/.test(l)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*]\s/, '')); i++; }
      out.push(`<ul>${items.map((x) => `<li>${inline(x)}</li>`).join('')}</ul>`); continue;
    }
    if (l.trim() === '') { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() !== '' && !/^(#|\||\s*[-*]\s)/.test(lines[i])) { para.push(lines[i]); i++; }
    out.push(`<p>${inline(para.join(' '))}</p>`);
  }
  return out.join('\n');
}

// =================================================================== UPLOAD
class UploadFlow {
  constructor() {
    this.options = null;
    this.stage = null;
    this.data = null;
    this.generatedReadme = '';
    this.viewer = null;
    this.yamlCheck = null;
    this._wire();
  }

  async init() {
    try {
      this.options = await api('/api/options');
    } catch (e) {
      $('#file-result').replaceChildren(msgBox('error', `Could not load form options: ${esc(e.message)}`));
      this.options = { indicators: ['GCaMP6f', 'GCaMP6s'], coding_units: ['glomeruli', 'KCs'], recording_systems: [], drivers: [], panels: [], uploaders: [], genotypes: [], default_wavelength_nm: 925 };
    }
    this._fillOptions();
  }

  _fillOptions() {
    const o = this.options;
    const fill = (sel, values) => { sel.innerHTML = values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('') + '<option value="__other__">Other…</option>'; };
    fill($('#indicator-select'), o.indicators);
    fill($('#coding-unit-select'), o.coding_units);
    const sys = $('#system-select');
    sys.innerHTML = '<option value="">—</option>' + o.recording_systems.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('') + '<option value="__other__">Other…</option>';
    const dl = (id, values) => { $(id).innerHTML = values.map((v) => `<option value="${esc(v)}">`).join(''); };
    dl('#dl-uploaders', o.uploaders); dl('#dl-drivers', o.drivers); dl('#dl-genotypes', o.genotypes); dl('#dl-panels', o.panels);
    const form = $('#step-meta');
    form.wavelength_nm.value = o.default_wavelength_nm ?? 925;
    const name = LS.get('olfdb.uploader_name', '');
    if (name) form.uploader_name.value = name;
    for (const [sel, other] of [['#indicator-select', '#indicator-other'], ['#coding-unit-select', '#coding-unit-other'], ['#system-select', '#system-other']]) {
      const s = $(sel), oth = $(other);
      const sync = () => { oth.hidden = s.value !== '__other__'; if (!oth.hidden) oth.focus(); };
      s.addEventListener('change', sync);
    }
  }

  _wire() {
    const dz = $('#dropzone'), fi = $('#file-input');
    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('over'));
    dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('over'); if (e.dataTransfer.files[0]) this.stageFile(e.dataTransfer.files[0]); });
    fi.addEventListener('change', () => { if (fi.files[0]) this.stageFile(fi.files[0]); fi.value = ''; });

    const form = $('#step-meta');
    const pw = $('#power-mw'), tol = $('#power-tol');
    const syncPower = () => {
      const has = pw.value !== '';
      tol.disabled = !has;
      $$('#power-sameday input').forEach((r) => { r.disabled = !has; });
    };
    pw.addEventListener('input', syncPower);
    form.addEventListener('submit', (e) => { e.preventDefault(); this.generateReadme(); });

    $('#yaml-btn').addEventListener('click', () => $('#yaml-input').click());
    $('#yaml-input').addEventListener('change', (e) => { if (e.target.files.length) this.addYaml(e.target.files); e.target.value = ''; });
    $('#xml-btn').addEventListener('click', () => $('#xml-input').click());
    $('#xml-input').addEventListener('change', (e) => { if (e.target.files[0]) this.addXml(e.target.files[0]); e.target.value = ''; });

    $('#readme-text').addEventListener('input', () => { $('#readme-preview').innerHTML = renderMarkdown($('#readme-text').value); });
    $('#readme-regenerate').addEventListener('click', () => this.generateReadme());
    $('#commit-btn').addEventListener('click', () => this.commit());
    $('#done-another').addEventListener('click', () => this.reset());
  }

  reset() {
    this.stage = null; this.data = null; this.yamlCheck = null; this.generatedReadme = '';
    $('#file-result').innerHTML = '';
    $('#preview').hidden = true;
    $('#step-meta').hidden = true;
    $('#step-readme').hidden = true;
    $('#step-done').hidden = true;
    $('#yaml-list').innerHTML = ''; $('#yaml-check').innerHTML = ''; $('#xml-info').innerHTML = ''; $('#xml-status').textContent = '';
    $('#meta-errors').innerHTML = ''; $('#commit-errors').innerHTML = '';
    window.scrollTo(0, 0);
  }

  async stageFile(file) {
    this.reset();
    const result = $('#file-result');
    result.replaceChildren(msgBox('info', `Reading <b>${esc(file.name)}</b> (${(file.size / 1e6).toFixed(1)} MB)…`));
    const form = new FormData();
    form.append('file', file, file.name);
    let res;
    try { res = await api('/api/stage', { method: 'POST', form }); } catch (e) { result.replaceChildren(msgBox('error', `Upload failed: ${esc(e.message)}`)); return; }
    if (!res.ok) {
      result.replaceChildren(msgBox('error', `<b>${esc(file.name)} was not accepted.</b> Fix these and choose the file again:${listHtml(res.errors || [res.error])}`));
      if (res.warnings?.length) result.appendChild(msgBox('warn', `Also:${listHtml(res.warnings)}`));
      return;
    }
    this.stage = res.stage;
    this.data = res.data;
    const s = res.data.summary;
    const warn = res.data.warnings || [];
    result.innerHTML = '';
    const okBox = msgBox('ok', `<b>${esc(file.name)}</b> passed validation (${esc(s.source_format)}).`);
    okBox.insertAdjacentHTML('beforeend', `<div class="summary-grid">
      <div><span>flies</span>${s.n_flies} (${esc(s.first_date)}${s.first_date !== s.last_date ? ' to ' + esc(s.last_date) : ''})</div>
      <div><span>ROI columns</span>${s.n_units.toLocaleString()} (${esc(s.roi_kind)})</div>
      <div><span>odors</span>${s.n_odors} in ${s.panels.length} panel${s.panels.length === 1 ? '' : 's'}</div>
      <div><span>trials per odor</span>${s.repeats_min === s.repeats_max ? s.repeats_min : s.repeats_min + '–' + s.repeats_max}</div>
      <div><span>missing cells</span>${(s.nan_fraction * 100).toFixed(1)}%</div>
      <div><span>value range</span>${fmtNum(s.vmin)} to ${fmtNum(s.vmax)}</div>
    </div>`);
    result.appendChild(okBox);
    if (warn.length) result.appendChild(msgBox('warn', `Worth a look:${listHtml(warn)}`));

    $('#preview').hidden = false;
    $('#preview-caption').textContent = 'This is exactly what will be shown after saving. Hover a cell for its value.';
    if (!this.viewer) this.viewer = new MatrixViewer($('#preview-viewer'));
    this.viewer.setNeuralData(res.data, { title: file.name, fileStem: file.name.replace(/\.[^.]*$/, '') || 'preview' });

    // prefill the form from the file
    const meta = $('#step-meta');
    meta.hidden = false;
    if (s.panel_names?.length && !meta.panel_names.value) meta.panel_names.value = s.panel_names.join(', ');
    if (s.roi_kind === 'glomeruli') $('#coding-unit-select').value = 'glomeruli';
    else if (s.roi_kind === 'integers' && this.options?.coding_units.includes('KCs')) $('#coding-unit-select').value = 'KCs';
    meta.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async addYaml(files) {
    const form = new FormData();
    for (const f of files) form.append('file', f, f.name);
    const box = $('#yaml-check');
    box.replaceChildren(msgBox('info', 'Checking the odors against the YAML…'));
    let res;
    try { res = await api(`/api/stage/${this.stage.stage_id}/odor_yaml`, { method: 'POST', form }); } catch (e) { box.replaceChildren(msgBox('error', esc(e.message))); return; }
    this._renderYaml(res);
  }

  async removeYaml(name) {
    const res = await api(`/api/stage/${this.stage.stage_id}/odor_yaml/${encodeURIComponent(name)}`, { method: 'DELETE' });
    this._renderYaml(res);
  }

  _renderYaml(res) {
    this.stage = res.stage;
    this.yamlCheck = res.check;
    const list = $('#yaml-list');
    list.innerHTML = res.stage.odor_yamls.map((y) => `<div class="file"><code>${esc(y.name)}</code><span class="muted">${y.n_odors} odors</span><button type="button" class="btn small" data-remove="${esc(y.name)}">remove</button></div>`).join('');
    $$('[data-remove]', list).forEach((b) => b.addEventListener('click', () => this.removeYaml(b.dataset.remove)));
    const box = $('#yaml-check');
    box.innerHTML = '';
    if (res.errors?.length) box.appendChild(msgBox('error', `Not added:${listHtml(res.errors)}`));
    const c = res.check;
    if (!c) return;
    if (c.ok) {
      box.appendChild(msgBox('ok', `All ${c.n_matrix_odors} odors in the matrix are described in the YAML${c.unused_yaml.length ? ` (${c.unused_yaml.length} YAML odors are not in the matrix: ${esc(c.unused_yaml.join(', '))})` : ''}.`));
    } else {
      const rows = c.missing.map((m) => `<tr><td><code>${esc(m.odor)}</code></td><td class="muted">${esc(m.hint)}</td></tr>`).join('');
      box.appendChild(msgBox('error', `<b>${c.missing.length} of ${c.n_matrix_odors} odors in the matrix are not in the YAML</b> (name and concentration must both match). Add the right YAML, remove this one, or fix the file; saving is blocked until the check passes.<table class="check-table">${rows}</table>`));
    }
  }

  async addXml(file) {
    const form = new FormData();
    form.append('file', file, file.name);
    $('#xml-status').textContent = 'Reading…';
    let res;
    try { res = await api(`/api/stage/${this.stage.stage_id}/experiment_xml`, { method: 'POST', form }); } catch (e) { $('#xml-status').textContent = ''; $('#xml-info').replaceChildren(msgBox('error', esc(e.message))); return; }
    if (!res.ok) { $('#xml-status').textContent = ''; $('#xml-info').replaceChildren(msgBox('error', `Not added:${listHtml(res.errors)}`)); return; }
    this.stage = res.stage;
    $('#xml-status').innerHTML = `<code>${esc(file.name)}</code> added`;
    const info = res.info || {};
    const kv = [];
    if (info.date) kv.push(['date', info.date]);
    if (info.software_version) kv.push(['ThorImage', info.software_version]);
    if (info.lsm) kv.push(['frame', `${info.lsm.pixelX ?? '?'} × ${info.lsm.pixelY ?? '?'} px${info.lsm.frameRate ? `, ${info.lsm.frameRate} fps` : ''}${info.lsm.pixelSizeUM ? `, ${info.lsm.pixelSizeUM} µm/px` : ''}`]);
    if (info.zstage?.steps) kv.push(['z planes', `${info.zstage.steps}${info.zstage.stepSizeUM ? ` × ${info.zstage.stepSizeUM} µm` : ''}`]);
    if (info.laser_position) kv.push(['laser', `${info.laser_position} nm`]);
    if (info.pockels?.start != null) kv.push(['pockels', `${info.pockels.start}%`]);
    $('#xml-info').innerHTML = kv.length ? `<div class="kv">${kv.map(([k, v]) => `<span>${esc(k)}</span><span>${esc(v)}</span>`).join('')}</div>` : '';
    const metaForm = $('#step-meta');
    if (res.suggestions?.wavelength_nm && String(metaForm.wavelength_nm.value) === String(this.options.default_wavelength_nm)) {
      metaForm.wavelength_nm.value = res.suggestions.wavelength_nm;
    }
  }

  collectMetadata() {
    const f = $('#step-meta');
    const other = (selId, otherId) => { const s = $(selId); return s.value === '__other__' ? $(otherId).value.trim() : s.value; };
    const num = (v) => (v === '' || v == null ? null : Number(v));
    const sameday = f.querySelector('input[name=power_measured_same_day]:checked')?.value;
    return {
      uploader_name: f.uploader_name.value.trim(),
      indicator: other('#indicator-select', '#indicator-other'),
      driver: f.driver.value.trim(),
      coding_unit: other('#coding-unit-select', '#coding-unit-other'),
      genotype: f.genotype.value.trim() || null,
      age_min_days: num(f.age_min_days.value),
      age_max_days: num(f.age_max_days.value),
      power_mw: num(f.power_mw.value),
      power_tolerance_mw: f.power_mw.value === '' ? null : num(f.power_tolerance_mw.value),
      power_measured_same_day: f.power_mw.value === '' || !sameday ? null : sameday === 'true',
      wavelength_nm: num(f.wavelength_nm.value),
      source_code_url: f.source_code_url.value.trim() || null,
      raw_data_path: f.raw_data_path.value.trim() || null,
      panel_names: f.panel_names.value,
      recording_system: other('#system-select', '#system-other') || null,
      notes: f.notes.value.trim() || null,
    };
  }

  _localChecks(meta) {
    const errs = [];
    if (!meta.uploader_name) errs.push('your name is required');
    if (!meta.indicator) errs.push('calcium indicator is required');
    if (!meta.driver) errs.push('Gal4 driver is required');
    if (!meta.coding_unit) errs.push('coding unit is required');
    if (meta.age_min_days != null && meta.age_max_days != null && meta.age_max_days < meta.age_min_days) errs.push('fly age range: max is smaller than min');
    if (this.yamlCheck && !this.yamlCheck.ok) errs.push('the odor YAML check has not passed');
    return errs;
  }

  async generateReadme() {
    const meta = this.collectMetadata();
    const errs = this._localChecks(meta);
    const box = $('#meta-errors');
    box.innerHTML = '';
    if (errs.length) { box.appendChild(msgBox('error', `Before generating the README:${listHtml(errs)}`)); return; }
    LS.set('olfdb.uploader_name', meta.uploader_name);
    let res;
    try { res = await api(`/api/stage/${this.stage.stage_id}/readme`, { method: 'POST', json: { metadata: meta } }); } catch (e) { box.appendChild(msgBox('error', esc(e.message))); return; }
    if (!res.ok) { box.appendChild(msgBox('error', `The server rejected the details:${listHtml(res.errors)}`)); return; }
    this.generatedReadme = res.readme;
    $('#readme-text').value = res.readme;
    $('#readme-preview').innerHTML = renderMarkdown(res.readme);
    $('#step-readme').hidden = false;
    $('#step-readme').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async commit() {
    const meta = this.collectMetadata();
    const errs = this._localChecks(meta);
    const box = $('#commit-errors');
    box.innerHTML = '';
    if (errs.length) { box.appendChild(msgBox('error', `Not saved:${listHtml(errs)}`)); return; }
    const readme = $('#readme-text').value;
    const btn = $('#commit-btn');
    btn.disabled = true; btn.textContent = 'Saving…';
    let res;
    try {
      res = await api(`/api/stage/${this.stage.stage_id}/commit`, { method: 'POST', json: { metadata: meta, readme, readme_edited: readme !== this.generatedReadme } });
    } catch (e) { box.appendChild(msgBox('error', esc(e.message))); btn.disabled = false; btn.textContent = 'Save to database'; return; }
    btn.disabled = false; btn.textContent = 'Save to database';
    if (!res.ok) { box.appendChild(msgBox('error', `Not saved:${listHtml(res.errors)}`)); return; }
    LS.set('olfdb.uploader_name', meta.uploader_name);
    $('#step-done').hidden = false;
    $('#done-text').innerHTML = `Dataset <b>#${res.dataset_id}</b> — ${esc(res.title)} — is in the database.`;
    $('#done-link').href = `#browse/${res.dataset_id}`;
    $('#step-done').scrollIntoView({ behavior: 'smooth', block: 'start' });
    browse.invalidate();
  }
}

// =================================================================== BROWSE
class Browse {
  constructor() {
    this.datasets = null;
    this.group = LS.get('olfdb.browse_group', 'panel');
    this.selected = null;
    this.picked = new Set(); // ticked for download
    this.viewer = null;
    this.cache = new Map();
    $$('#tab-browse .seg button').forEach((b) => b.addEventListener('click', () => { this.group = b.dataset.group; LS.set('olfdb.browse_group', this.group); this.renderList(); }));
    $('#ds-readme-toggle').addEventListener('click', () => {
      const p = $('#ds-readme');
      p.hidden = !p.hidden;
      $('#ds-readme-toggle').setAttribute('aria-expanded', String(!p.hidden));
    });
    this.odorCache = new Map();
    $('#ds-odors-toggle').addEventListener('click', () => {
      const p = $('#ds-odors');
      p.hidden = !p.hidden;
      $('#ds-odors-toggle').setAttribute('aria-expanded', String(!p.hidden));
      if (!p.hidden && this.selected != null) this.showOdors(this.selected);
    });
    const downloadUrl = (ids, format) => `/api/download?ids=${ids.join(',')}&format=${format}`;
    wireMenu($('#ds-download'), ({ format }) => { if (this.selected != null) download(downloadUrl([this.selected], format), $('#ds-download-status')); });
    wireMenu($('#pick-download'), ({ format }) => download(downloadUrl([...this.picked].sort((a, b) => a - b), format), $('#pick-status')));
    $('#pick-clear').addEventListener('click', () => { this.picked.clear(); this._syncPicks(); });
    $('#dataset-list').addEventListener('change', (e) => {
      const cb = e.target;
      const ids = cb.matches('.pick') ? [+cb.dataset.id] : cb.matches('.pick-group') ? cb.dataset.ids.split(',').map(Number) : [];
      for (const id of ids) { if (cb.checked) this.picked.add(id); else this.picked.delete(id); }
      if (ids.length) this._syncPicks();
    });
  }

  invalidate() { this.datasets = null; this.cache.clear(); }

  async load() {
    if (!this.datasets) {
      try { this.datasets = (await api('/api/datasets')).datasets; } catch (e) { $('#dataset-list').replaceChildren(msgBox('error', esc(e.message))); return; }
      const ids = new Set(this.datasets.map((d) => d.id));
      for (const id of this.picked) if (!ids.has(id)) this.picked.delete(id);
    }
    this.renderList();
  }

  // checkboxes (a dataset can be listed under several panels) and the selection bar
  _syncPicks() {
    const list = $('#dataset-list');
    $$('.pick', list).forEach((cb) => { cb.checked = this.picked.has(+cb.dataset.id); });
    $$('.pick-group', list).forEach((cb) => {
      const ids = cb.dataset.ids.split(',').map(Number);
      const n = ids.filter((id) => this.picked.has(id)).length;
      cb.checked = n === ids.length;
      cb.indeterminate = n > 0 && n < ids.length;
    });
    const n = this.picked.size;
    const units = (this.datasets || []).filter((d) => this.picked.has(d.id)).reduce((s, d) => s + d.n_units, 0);
    const text = $('#pick-text');
    text.textContent = n ? `${n} selected · ${units.toLocaleString()} ROIs` : 'Tick datasets to download several together.';
    text.classList.toggle('muted', !n);
    $('#pick-download').hidden = !n;
    $('#pick-clear').hidden = !n;
  }

  renderList() {
    $$('#tab-browse .seg button').forEach((b) => b.classList.toggle('active', b.dataset.group === this.group));
    const list = $('#dataset-list');
    if (!this.datasets) return;
    if (!this.datasets.length) { list.innerHTML = '<div class="empty">Nothing uploaded yet. Use the Upload tab.</div>'; this._syncPicks(); return; }
    const item = (d) => `<div class="ds-item${this.selected === d.id ? ' active' : ''}">
        <input type="checkbox" class="pick" data-id="${d.id}" aria-label="Select dataset #${d.id} for download">
        <button type="button" class="list-item" data-id="${d.id}">
          <span class="t">${esc(d.driver)} &gt; ${esc(d.indicator)} <span class="muted">(${esc(d.coding_unit)})</span></span>
          <span class="s">${d.n_flies} ${d.n_flies === 1 ? 'fly' : 'flies'}, ${d.n_units.toLocaleString()} ROIs, ${d.n_odors} odors; last fly ${esc(d.last_fly_date)}</span>
          <span class="s">#${d.id}, ${esc(d.uploader_name)}${d.recording_system ? ', ' + esc(d.recording_system) : ''}</span>
        </button>
      </div>`;
    const groupHead = (label, ds) => `<div class="list-group"><label><input type="checkbox" class="pick-group" data-ids="${ds.map((d) => d.id).join(',')}"
        aria-label="Select all of ${esc(label)} for download"> ${esc(label)} <span class="muted">(${ds.length})</span></label></div>`;
    let html = '';
    if (this.group === 'panel') {
      const groups = new Map();
      for (const d of this.datasets) {
        const panels = (d.panels.some((p) => p.source === 'form') ? d.panels.filter((p) => p.source === 'form') : d.panels).map((p) => p.panel);
        for (const p of panels.length ? panels : ['(no panel name)']) {
          if (!groups.has(p)) groups.set(p, []);
          groups.get(p).push(d);
        }
      }
      const keys = Array.from(groups.keys()).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
      for (const k of keys) {
        const ds = groups.get(k).slice().sort((a, b) => (b.last_fly_date || '').localeCompare(a.last_fly_date || '') || b.id - a.id);
        html += groupHead(k, ds) + ds.map(item).join('');
      }
    } else {
      const ds = this.datasets.slice().sort((a, b) => (b.last_fly_date || '').localeCompare(a.last_fly_date || '') || b.id - a.id);
      const months = runsOf(ds, (d) => (d.last_fly_date || '').slice(0, 7) || 'unknown date');
      for (const m of months) html += groupHead(m.key, ds.slice(m.start, m.end)) + ds.slice(m.start, m.end).map(item).join('');
    }
    list.innerHTML = html;
    $$('.list-item', list).forEach((b) => b.addEventListener('click', () => { location.hash = `#browse/${b.dataset.id}`; }));
    this._syncPicks();
  }

  async show(id) {
    this.selected = id;
    if (this.datasets) this.renderList();
    $('#browse-empty').hidden = true;
    $('#dataset-view').hidden = false;
    $('#ds-title').textContent = 'Loading…';
    $('#ds-sub').textContent = '';
    let d, m;
    try {
      if (!this.cache.has(id)) {
        const [dd, mm] = await Promise.all([api(`/api/datasets/${id}`), api(`/api/datasets/${id}/matrix`)]);
        this.cache.set(id, { d: dd, m: mm });
      }
      ({ d, m } = this.cache.get(id));
    } catch (e) { $('#ds-title').textContent = 'Could not load this dataset'; $('#ds-sub').textContent = e.message; return; }
    if (this.selected !== id) return;
    $('#ds-title').textContent = d.title;
    const md = d.metadata;
    const bits = [`uploaded by ${d.uploader_name} on ${d.created_at.slice(0, 10)}`];
    if (md.genotype) bits.push(md.genotype);
    if (md.wavelength_nm) bits.push(`${md.wavelength_nm} nm`);
    if (md.power_mw) bits.push(`${md.power_mw} mW${md.power_tolerance_mw != null ? ' ±' + md.power_tolerance_mw : ''}`);
    if (md.recording_system) bits.push(md.recording_system);
    $('#ds-sub').textContent = bits.join('; ');
    $('#ds-readme-html').innerHTML = d.readme_html;
    $('#ds-files').innerHTML = d.files.map((f) => `<a href="/api/datasets/${id}/files/${encodeURI(f)}" download>${esc(f)}</a>`).join('');
    if (!this.viewer) this.viewer = new MatrixViewer($('#dataset-viewer'));
    this.viewer.setNeuralData(m, { title: d.title, fileStem: `olfdb_dataset-${id}` });
    if (!$('#ds-odors').hidden) this.showOdors(id);
  }

  async showOdors(id) {
    const panel = $('#ds-odors');
    if (!this.odorCache.has(id)) {
      panel.innerHTML = '<p class="muted">Loading…</p>';
      try { this.odorCache.set(id, await api(`/api/datasets/${id}/odors`)); } catch (e) { panel.replaceChildren(msgBox('error', esc(e.message))); return; }
    }
    if (this.selected === id) panel.innerHTML = datasetOdorTable(this.odorCache.get(id));
  }
}

// ==================================================================== ODORS
const ODOR_FIELDS = {
  name: 'name', abbrev: 'abbreviation', aliases: 'aliases', cid: 'CID', cas: 'CAS', solvent: 'solvent', supplier: 'supplier',
  catalog_no: 'catalog no.', catalog_url: 'catalog URL', metacyc: 'MetaCyc', lotus: 'LOTUS', inventory: 'inventory', notes: 'notes',
};
const extLink = (url, text) => (url ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(text)}</a>` : esc(text ?? ''));
const structureImg = (o) => (o.links?.structure
  ? `<img class="structure" src="${esc(o.links.structure)}" alt="structure of ${esc(o.name)}" loading="lazy" onerror="this.remove()">` : '');
const fmtValue = (v) => (Array.isArray(v) ? v.join('; ') : String(v ?? ''));
const fmtWhen = (iso) => iso.replace('T', ' ').slice(0, 16);
const usedBy = (n) => (n ? `used by ${n} dataset${n === 1 ? '' : 's'}` : 'not used by any dataset');
const syncCounts = (s) => `${s.n_added} new, ${s.n_changed} changed, ${s.n_deleted} deleted`;
// sort names ignoring leading stereo descriptors / locants: (+)-pulegone, 1,4-diaminobutane, p-cresol
const odorSortKey = (name) => name.toLowerCase().replace(/^(\([^)]*\)-|[\d,]+-|[a-z]-(?=[a-z]))+/, '');
function setOdorBadge(changed) { $('#odors-badge').hidden = !changed; }

// linked identifier cells, shared by the Odors tab and the dataset odor tables
function idCells(o) {
  const l = o.links || {};
  const catalog = o.catalog_no || l.catalog ? extLink(l.catalog, o.catalog_no || 'product page') : '';
  return {
    cid: extLink(l.pubchem, o.cid), cas: extLink(l.cas, o.cas), metacyc: extLink(l.metacyc, o.metacyc), lotus: extLink(l.lotus, o.lotus),
    catalog: [o.supplier ? `<span class="muted">${esc(o.supplier)}</span>` : '', catalog].filter(Boolean).join(' '),
  };
}

// a dataset's odors (GET /api/datasets/<id>/odors), per panel, like the thesis odor tables
function datasetOdorTable({ rows, odors }) {
  const hasGlom = rows.some((r) => r.yaml?.glomerulus);
  const nCols = hasGlom ? 11 : 10;
  const body = runsOf(rows, (r) => r.panel).map((g) => {
    const title = g.key != null ? `<tr class="group"><td colspan="${nCols}">${esc(g.key)}</td></tr>` : '';
    return title + rows.slice(g.start, g.end).map((r) => {
      const o = r.odor_id != null ? odors[r.odor_id] : null;
      const c = o ? idCells(o) : {};
      let name;
      if (o) name = `<a href="#odors/${o.id}">${esc(o.name)}</a>`;
      else if (r.match === 'ambiguous') name = `<span class="muted">ambiguous: ${r.candidates.map((i) => `<a href="#odors/${i}">${esc(odors[i].name)}</a>`).join(' or ')}</span>`;
      else name = `<span class="muted">${r.yaml?.name ? `${esc(r.yaml.name)}, ` : ''}not in the odor table</span>`;
      return `<tr><td><code>${esc(r.odor)}</code></td><td>${name}</td><td>${r.log10_conc ?? ''}</td><td>${esc(r.yaml?.solvent || o?.solvent || '')}</td>
        ${hasGlom ? `<td>${esc(r.yaml?.glomerulus || '')}</td>` : ''}<td>${c.cid || ''}</td><td>${c.cas || ''}</td><td>${c.catalog || ''}</td>
        <td>${c.metacyc || ''}</td><td>${c.lotus || ''}</td><td class="structure-cell">${o ? structureImg(o) : ''}</td></tr>`;
    }).join('');
  }).join('');
  const n = rows.filter((r) => r.odor_id != null).length;
  return `<p class="muted">${n} of ${rows.length} odors are in the <a href="#odors">odor table</a> (matched through the dataset's odor YAML where it has one, otherwise by abbreviation).</p>
    <table class="odor-table"><thead><tr><th>Odor</th><th>Name</th><th>log<sub>10</sub> conc.</th><th>Solvent</th>${hasGlom ? '<th>Target glomerulus</th>' : ''}
    <th>CID</th><th>CAS</th><th>Catalog no.</th><th>MetaCyc</th><th>LOTUS</th><th>Structure</th></tr></thead><tbody>${body}</tbody></table>`;
}

class OdorsPage {
  constructor() {
    this.odors = null;
    this.syncs = [];
    this.preview = null;
    this.highlightId = null;
    $('#sheet-check').addEventListener('click', () => this.runPreview());
    $('#odor-csv-btn').addEventListener('click', () => $('#odor-csv-input').click());
    $('#odor-csv-input').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) this.runPreview(f); });
    $('#odor-filter').addEventListener('input', () => this.renderTable());
  }

  async load({ force = false } = {}) {
    this.loadStatus();
    if (this.odors && !force) { this.renderTable(); return; }
    try {
      ({ odors: this.odors, syncs: this.syncs } = await api('/api/odors'));
    } catch (e) { $('#odor-table').replaceChildren(msgBox('error', esc(e.message))); return; }
    this.odors.sort((a, b) => odorSortKey(a.name).localeCompare(odorSortKey(b.name)) || a.name.localeCompare(b.name));
    this.renderTable();
    this.renderSyncs();
  }

  async loadStatus() {
    try { this.status = await api('/api/odors/status'); } catch (e) { $('#sheet-status').replaceChildren(msgBox('error', esc(e.message))); return; }
    setOdorBadge(this.status.changed);
    const s = this.status, box = $('#sheet-status');
    $('#sheet-check').hidden = !s.configured;
    if (!s.configured) {
      box.innerHTML = `<p class="muted">No odor sheet is set up on the server (<code>OLFDB_ODOR_SHEET_URL</code>, see DEPLOY.md). You can still import a CSV with the sheet's columns;
        the template has the odors from the thesis tables, and can also be imported into a new Google sheet to start it.</p>`;
      return;
    }
    let state = '';
    if (s.error) state = `<span class="pill error">can't read it</span> ${esc(s.error)}`;
    else if (s.changed) state = `<span class="pill warn">${s.last_sync ? 'changed since the last sync' : 'never synced'}</span>`;
    else if (s.fingerprint) state = '<span class="pill ok">in sync</span>';
    const last = s.last_sync;
    box.innerHTML = `<p><b>Odor sheet:</b> <a href="${esc(s.url)}" target="_blank" rel="noopener">Google sheet</a> ${state}</p>
      <p class="muted">Read ${s.checked_at ? `${esc(fmtWhen(s.checked_at))}${s.n_rows != null ? ` (${s.n_rows} odors)` : ''}` : 'not yet'};
      last synced ${last ? `${esc(fmtWhen(last.applied_at))} by ${esc(last.applied_by)} (${syncCounts(last)})` : 'never'}.
      The sheet is looked at again every few minutes while the app is in use, to tell whether it changed.</p>`;
  }

  async runPreview(file) {
    const box = $('#odor-preview');
    box.replaceChildren(msgBox('info', file ? `Reading <b>${esc(file.name)}</b>…` : 'Reading the sheet…'));
    try {
      let form;
      if (file) { form = new FormData(); form.append('file', file, file.name); }
      this.preview = await api('/api/odors/preview', file ? { method: 'POST', form } : { method: 'POST', json: {} });
    } catch (e) { box.replaceChildren(msgBox('error', esc(e.message))); return; }
    this.renderPreview();
    if (!file) this.loadStatus();
  }

  renderPreview() {
    const p = this.preview, box = $('#odor-preview');
    const where = p.source === 'Google sheet' ? 'sheet' : 'file';
    const nothing = !p.added.length && !p.changed.length && !p.missing.length;
    const empty = '<span class="muted">(empty)</span>';
    const parts = [`<h3 class="subhead">${esc(p.source)}: ${p.n_rows} odors</h3>`];
    if (p.errors.length) parts.push(`<div class="msg error"><b>Fix these in the ${where} first</b> (nothing is synced until they are):${listHtml(p.errors)}</div>`);
    if (p.warnings.length) parts.push(`<div class="msg warn">Worth a look:${listHtml(p.warnings)}</div>`);
    if (nothing && !p.errors.length) parts.push(`<div class="msg ok">The odor table already matches the ${where} (${p.unchanged} odors).</div>`);
    if (p.added.length) {
      parts.push(`<h4>New odors (${p.added.length}) <span class="muted">added when you apply</span></h4>
        <table class="odor-table"><thead><tr><th>Name</th><th>Abbrev.</th><th>Aliases</th><th>CID</th><th>CAS</th><th>Solvent</th><th>Supplier</th><th>Catalog no.</th></tr></thead><tbody>
        ${p.added.map((r) => `<tr><td><b>${esc(r.name)}</b></td>${['abbrev', 'aliases', 'cid', 'cas', 'solvent', 'supplier', 'catalog_no'].map((f) => `<td>${esc(fmtValue(r[f]))}</td>`).join('')}</tr>`).join('')}
        </tbody></table>`);
    }
    if (p.changed.length) {
      parts.push(`<h4>Changes to odors already in the table (${p.changed.length})</h4>
        <div class="msg warn">Only the changes you tick are applied. Renaming an odor or changing its abbreviation can unlink datasets that refer to it
        by the old name or abbreviation; keeping the old one in its aliases keeps them linked.</div>
        <label class="check"><input type="checkbox" data-all-changes> tick all</label>`);
      parts.push(p.changed.map((c) => `<div class="change">
        <label class="check"><input type="checkbox" data-change="${c.id}"> <b>${esc(c.rename ? `${c.old_name} → ${c.name}` : c.name)}</b></label>
        <span class="muted">${usedBy(c.datasets)}</span>${c.unlinked ? ` <span class="pill warn">${c.unlinked} would no longer be linked to it</span>` : ''}
        <table class="diff">${c.fields.map((f) => `<tr><td>${esc(ODOR_FIELDS[f.field] || f.field)}</td><td class="old">${esc(fmtValue(f.old)) || empty}</td>
          <td>→</td><td class="new">${esc(fmtValue(f.new)) || empty}</td></tr>`).join('')}</table></div>`).join(''));
    }
    if (p.missing.length) {
      parts.push(`<h4>In the table but not in the ${where} (${p.missing.length}) <span class="muted">kept unless you tick them</span></h4>`);
      parts.push(p.missing.map((o) => `<div class="change"><label class="check"><input type="checkbox" data-delete="${o.id}"> delete <b>${esc(o.name)}</b>${o.abbrev ? ` (${esc(o.abbrev)})` : ''}</label>
        <span class="muted">${usedBy(o.datasets)}</span>${o.datasets ? ' <span class="pill warn">deleting it unlinks them</span>' : ''}</div>`).join(''));
    }
    if (p.unchanged && !nothing) parts.push(`<p class="muted">${p.unchanged} odors unchanged.</p>`);
    if (!nothing && !p.errors.length) {
      parts.push(`<div class="actions"><label class="check">Your name <input id="odor-sync-name" value="${esc(LS.get('olfdb.uploader_name', ''))}"></label>
        <button type="button" class="btn primary" id="odor-apply"></button><button type="button" class="btn" id="odor-cancel">Cancel</button></div>
        <div id="odor-apply-result"></div>`);
    }
    box.innerHTML = parts.join('');
    $('[data-all-changes]', box)?.addEventListener('change', (e) => { $$('[data-change]', box).forEach((x) => { x.checked = e.target.checked; }); this._syncApply(); });
    $$('[data-change], [data-delete]', box).forEach((x) => x.addEventListener('change', () => this._syncApply()));
    $('#odor-apply', box)?.addEventListener('click', () => this.apply());
    $('#odor-cancel', box)?.addEventListener('click', () => { this.preview = null; box.innerHTML = ''; });
    this._syncApply();
  }

  _syncApply() {
    const btn = $('#odor-apply');
    if (!btn) return;
    const p = this.preview, box = $('#odor-preview');
    const nc = $$('[data-change]:checked', box).length, nd = $$('[data-delete]:checked', box).length;
    btn.textContent = `Apply: ${p.added.length} new, ${nc} of ${p.changed.length} changes${p.missing.length ? `, ${nd} deletions` : ''}`;
    btn.disabled = !p.added.length && !nc && !nd;
  }

  async apply() {
    const box = $('#odor-preview'), out = $('#odor-apply-result');
    const name = $('#odor-sync-name').value.trim();
    if (!name) { out.replaceChildren(msgBox('error', 'Your name is required (it goes in the sync history).')); return; }
    LS.set('olfdb.uploader_name', name);
    const json = {
      fingerprint: this.preview.fingerprint, applied_by: name,
      changes: $$('[data-change]:checked', box).map((x) => +x.dataset.change),
      deletions: $$('[data-delete]:checked', box).map((x) => +x.dataset.delete),
    };
    let res;
    try { res = await api('/api/odors/apply', { method: 'POST', json }); } catch (e) { out.replaceChildren(msgBox('error', esc(e.message))); return; }
    if (!res.ok) { out.replaceChildren(msgBox('error', `Not applied:${listHtml(res.errors)}`)); return; }
    this.preview = null;
    box.replaceChildren(msgBox('ok', `Synced: ${res.added} new, ${res.changed} changed, ${res.deleted} deleted.`));
    browse.odorCache.clear();
    await this.load({ force: true });
  }

  renderTable() {
    const all = this.odors || [];
    const q = $('#odor-filter').value.trim().toLowerCase();
    const shown = q ? all.filter((o) => [o.name, o.abbrev, ...o.aliases, o.cid, o.cas, o.catalog_no, o.supplier, o.inventory]
      .some((v) => v != null && String(v).toLowerCase().includes(q))) : all;
    $('#odor-count').textContent = all.length ? `(${q ? `${shown.length} of ` : ''}${all.length})` : '';
    if (!all.length) {
      $('#odor-table').innerHTML = '<div class="empty">The odor table is empty. Sync it from the odor sheet, or import a CSV (the template has the odors from the thesis).</div>';
      return;
    }
    $('#odor-table').innerHTML = `<table class="odor-table"><thead><tr><th>Structure</th><th>Name</th><th>Abbrev.</th><th>CID</th><th>CAS</th><th>Solvent</th>
      <th>Catalog no.</th><th>MetaCyc</th><th>LOTUS</th><th>Inventory</th><th>Datasets</th><th>Notes</th></tr></thead><tbody>
      ${shown.map((o) => {
        const c = idCells(o);
        return `<tr id="odor-${o.id}"><td class="structure-cell">${structureImg(o)}</td>
          <td><b>${esc(o.name)}</b>${o.aliases.length ? `<div class="muted small">${esc(o.aliases.join('; '))}</div>` : ''}</td>
          <td>${esc(o.abbrev || '')}</td><td>${c.cid}</td><td>${c.cas}</td><td>${esc(o.solvent || '')}</td><td>${c.catalog}</td>
          <td>${c.metacyc}</td><td>${c.lotus}</td><td>${esc(o.inventory || '')}</td>
          <td>${o.datasets.map((i) => `<a href="#browse/${i}">#${i}</a>`).join(' ')}</td><td class="notes">${esc(o.notes || '')}</td></tr>`;
      }).join('')}</tbody></table>`;
    if (this.highlightId != null) this.highlight(this.highlightId);
  }

  highlight(id) {
    this.highlightId = id;
    const tr = $(`#odor-${id}`);
    if (!tr) return;
    $$('.odor-table tr.hl').forEach((x) => x.classList.remove('hl'));
    tr.classList.add('hl');
    tr.scrollIntoView({ block: 'center' });
  }

  renderSyncs() {
    const names = (xs) => esc(xs.join(', '));
    $('#odor-syncs').innerHTML = this.syncs.length ? `<table class="odor-table"><thead><tr><th>When</th><th>By</th><th>From</th><th>What</th></tr></thead><tbody>
      ${this.syncs.map((s) => {
        const d = s.details || {};
        const what = [
          d.added?.length ? `<div><b>new:</b> ${names(d.added)}</div>` : '',
          d.changed?.length ? `<div><b>changed:</b> ${d.changed.map((c) => `${esc(c.name)} <span class="muted">(${c.fields.map((f) => esc(ODOR_FIELDS[f.field] || f.field)).join(', ')})</span>`).join(', ')}</div>` : '',
          d.deleted?.length ? `<div><b>deleted:</b> ${names(d.deleted)}</div>` : '',
          d.not_applied?.length ? `<div class="muted">changes not applied: ${names(d.not_applied)}</div>` : '',
        ].join('');
        return `<tr><td>${esc(fmtWhen(s.applied_at))}</td><td>${esc(s.applied_by)}</td><td>${esc(s.source)}</td><td>${what || syncCounts(s)}</td></tr>`;
      }).join('')}</tbody></table>` : '<p class="muted">No syncs yet.</p>';
  }
}

// ================================================================ REFERENCE
class Reference {
  constructor() { this.items = null; this.viewer = null; this.selected = null; }

  async load() {
    if (!this.items) {
      try { this.items = (await api('/api/reference')).datasets; } catch (e) { $('#reference-list').replaceChildren(msgBox('error', esc(e.message))); return; }
    }
    const list = $('#reference-list');
    if (!this.items.length) { list.innerHTML = '<div class="empty">No reference matrices are configured on the server (see reference_data/manifest.json).</div>'; return; }
    list.innerHTML = this.items.map((it) => `<button type="button" class="list-item${this.selected === it.id ? ' active' : ''}" data-id="${esc(it.id)}" ${it.available ? '' : 'disabled'}>
        <span class="t">${esc(it.title)}</span><span class="s">${esc(it.description || '')}${it.available ? '' : ' — file missing'}</span></button>`).join('');
    $$('.list-item', list).forEach((b) => b.addEventListener('click', () => { location.hash = `#reference/${b.dataset.id}`; }));
  }

  async show(id) {
    this.selected = id;
    if (this.items) this.load();
    $('#reference-empty').hidden = true;
    $('#reference-view').hidden = false;
    $('#ref-title').textContent = 'Loading…';
    let ref;
    try { ref = await api(`/api/reference/${encodeURIComponent(id)}`); } catch (e) { $('#ref-title').textContent = 'Could not load'; $('#ref-desc').textContent = e.message; return; }
    if (this.selected !== id) return;
    $('#ref-title').textContent = ref.meta.title;
    $('#ref-desc').textContent = [ref.meta.description, ref.meta.source ? `Source: ${ref.meta.source}` : ''].filter(Boolean).join(' ');
    if (!this.viewer) this.viewer = new MatrixViewer($('#reference-viewer'), { kind: 'generic' });
    this.viewer.setGenericData(ref, { title: ref.meta.title, fileStem: `olfdb_reference-${id}` });
  }
}

// ==================================================================== router
const upload = new UploadFlow();
const browse = new Browse();
const odorsPage = new OdorsPage();
const reference = new Reference();

function route() {
  const [tab, id] = location.hash.replace(/^#/, '').split('/');
  const name = ['upload', 'browse', 'odors', 'reference'].includes(tab) ? tab : 'browse';
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  $$('.tabs a').forEach((a) => a.classList.toggle('active', a.dataset.tab === name));
  if (name === 'browse') { browse.load().then(() => { if (id) browse.show(+id); }); }
  if (name === 'odors') { odorsPage.highlightId = id ? +id : null; odorsPage.load().then(() => { if (id) odorsPage.highlight(+id); }); }
  if (name === 'reference') { reference.load().then(() => { if (id) reference.show(id); }); }
  if (name === 'browse' && !id) { $('#browse-empty').hidden = false; $('#dataset-view').hidden = true; browse.selected = null; }
  if (name === 'reference' && !id) { $('#reference-empty').hidden = false; $('#reference-view').hidden = true; reference.selected = null; }
}

window.addEventListener('hashchange', route);
upload.init().then(route);
api('/api/health').then(() => { $('#topbar-status').textContent = ''; }).catch(() => { $('#topbar-status').textContent = 'server unreachable'; });
api('/api/odors/status').then((s) => setOdorBadge(s.changed)).catch(() => {});
