// Odor response database — frontend (no build step, no dependencies).
import { Heatmap } from './heatmap.js';
import { COLORMAPS, colormapsOfKind, makeNorm, drawColorbar, fmtNum } from './colormaps.js';
import { MAX_CLUSTER_ITEMS, ClusterPool, layoutDendrogram } from './cluster.js';
import { exportFigure } from './figure.js';
import { MoviePlayer } from './player.js';

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
  el.title = text;
  el.classList.toggle('error', kind === 'error');
}

// Fetched (rather than a plain link) so that the wait while the server builds a large
// zip (or CSV), and any error, show up in `status`.
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
    const type = res.headers.get('content-type') || 'application/zip';
    saveBlob(new Blob(chunks, { type }), m ? decodeURIComponent(m[1]) : `download.${type.includes('csv') ? 'csv' : 'zip'}`);
    setStatus(status, '');
  } catch (e) {
    setStatus(status, `Download failed: ${e.message}`, 'error');
  } finally {
    delete status.dataset.busy;
  }
}

// navigator.clipboard needs HTTPS (or localhost); the site is usually plain HTTP
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); return; }
  const t = document.createElement('textarea');
  t.value = text;
  t.style.cssText = 'position: fixed; opacity: 0;';
  document.body.appendChild(t);
  t.select();
  const ok = document.execCommand('copy');
  t.remove();
  if (!ok) throw new Error('the browser did not allow it');
}

// shows a check of a dataset's data path by its input (as from /api/tensor/check, or a
// local one; null to clear it); one with a warning is still ok
function showPathCheck(input, msg, c) {
  const warn = !!c?.ok && !!c.warning;
  input.classList.toggle('invalid', !!c && !c.ok);
  input.classList.toggle('path-warn', warn);
  msg.className = `path-msg${c ? (c.ok ? (warn ? ' warn' : ' ok') : ' bad') : ''}`;
  msg.textContent = c ? (c.ok ? c.warning || '✓ found on tensor' : c.error) : '';
}

// Checks a dataset's data path as typed, then on the server (as the Settings tab checks
// folders), writing it as normalized there if it is found and no longer being typed in.
async function checkPathInput(input, msg) {
  const value = input.value.trim();
  const problem = value ? localDirProblem(value) : null;
  showPathCheck(input, msg, problem ? { ok: false, error: problem } : null);
  if (!value || problem) return;
  let c;
  try { c = await api('/api/tensor/check', { method: 'POST', json: { path: value } }); } catch { return; }
  if (input.value.trim() !== c.input) return; // changed since
  if (c.ok && document.activeElement !== input) input.value = c.path;
  showPathCheck(input, msg, c);
}

// shows a <dialog> of buttons; resolves to the value of the one chosen ('' if dismissed with Escape)
function askDialog(dialog) {
  return new Promise((resolve) => {
    dialog.returnValue = '';
    dialog.addEventListener('close', () => resolve(dialog.returnValue), { once: true });
    dialog.showModal();
  });
}

// <details class="menu"> dropdowns: onPick(button.dataset) for the chosen entry
function wireMenu(menu, onPick) {
  $$('.menu-list button', menu).forEach((b) => b.addEventListener('click', () => { menu.open = false; onPick(b.dataset); }));
}
document.addEventListener('click', (e) => { $$('details.menu[open]').forEach((m) => { if (!m.contains(e.target)) m.open = false; }); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $$('details.menu[open]').forEach((m) => { m.open = false; }); });

// A menu at the pointer, for a right click: entries [{html, title, disabled, onPick}].
// A pick, a press elsewhere, Escape, scrolling or resizing closes it.
let closeContextMenu = () => {};
function showContextMenu(x, y, entries) {
  closeContextMenu();
  const menu = document.createElement('div');
  menu.className = 'menu-list ctx-menu';
  for (const e of entries) {
    const b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = e.html;
    b.disabled = !!e.disabled;
    b.title = e.title || '';
    b.addEventListener('click', () => { closeContextMenu(); e.onPick(); });
    menu.appendChild(b);
  }
  document.body.appendChild(menu);
  const r = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(4, Math.min(x, innerWidth - r.width - 4))}px`;
  menu.style.top = `${Math.max(4, Math.min(y, innerHeight - r.height - 4))}px`;
  const away = (e) => { if (!menu.contains(e.target)) closeContextMenu(); };
  const key = (e) => { if (e.key === 'Escape') closeContextMenu(); };
  const close = () => closeContextMenu();
  document.addEventListener('pointerdown', away, true);
  document.addEventListener('keydown', key);
  document.addEventListener('scroll', close, true);
  window.addEventListener('resize', close);
  window.addEventListener('blur', close);
  closeContextMenu = () => {
    menu.remove();
    document.removeEventListener('pointerdown', away, true);
    document.removeEventListener('keydown', key);
    document.removeEventListener('scroll', close, true);
    window.removeEventListener('resize', close);
    window.removeEventListener('blur', close);
    closeContextMenu = () => {};
  };
  menu.querySelector('button:not(:disabled)')?.focus();
}

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
    if (!c) { c = { key, panel: t.panel, odor: t.odor, trials: [] }; condIndex.set(key, c); conds.push(c); }
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

// The order clustering starts from (ties keep it), whatever order is shown
const CLUSTER_INPUT_ORDER = { rowOrder: 'fly', roiSort: 'file', odorOrder: 'file' };

// The items of one axis less those removed, in the order they were dragged into if they
// were. edit: {removed: Set of keys, order: [keys] or null}
function arrange(items, keyOf, { removed, order }) {
  const kept = removed.size ? items.filter((it) => !removed.has(keyOf(it))) : items.slice();
  if (!order) return kept;
  const pos = new Map(order.map((k, i) => [k, i]));
  const at = (it) => pos.get(keyOf(it)) ?? Infinity;
  return kept.map((it, i) => ({ it, i })).sort((a, b) => at(a.it) - at(b.it) || a.i - b.i).map((x) => x.it);
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

// Orders ROIs and/or odors by hierarchical clustering of trial-averaged responses
// (whatever the display mode). ROIs are clustered all together, whichever fly they come
// from; odors within each panel, by their responses across all ROIs.
// input: {unitIdx, conds} (those not removed, in CLUSTER_INPUT_ORDER)
// -> {unitIdx, conds (each null unless clustered), rowTrees, colTrees, notes}
async function clusterNeural(pool, data, input, o, onProgress) {
  const { conds, unitIdx } = input;
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
  const out = { unitIdx: o.clusterRows ? unitIdx.slice() : null, conds: o.clusterCols ? conds.slice() : null, rowTrees: [], colTrees: [], notes };
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

// order: {unitIdx, conds, panelOrder} as shown unless clustered
function buildNeuralModel(data, order, { mode, rowOrder }, clustering) {
  const { units, trials, values } = data;
  const conds = clustering?.conds ?? order.conds;
  const unitIdx = clustering?.unitIdx ?? order.unitIdx;
  const showGroups = order.panelOrder.length > 1 || order.panelOrder[0] != null;
  const cols = [], colTrials = [], condCenter = new Map();
  for (const c of conds) {
    const group = showGroups ? (c.panel ?? '(no panel)') : null;
    const start = cols.length;
    if (mode === 'trials') {
      const ts = c.trials.slice().sort((a, b) => trials[a].repeat - trials[b].repeat || a - b);
      for (const j of ts) { cols.push({ label: c.odor, group, sub: trials[j].repeat, key: c.key }); colTrials.push([j]); }
    } else {
      cols.push({ label: c.odor, group, sub: null, key: c.key });
      colTrials.push(c.trials);
    }
    condCenter.set(c, (start + cols.length) / 2);
  }
  const rows = unitIdx.map((i) => {
    const u = units[i];
    if (rowOrder === 'roi') return { label: flyKey(u), group: u.roi, key: i };
    if (rowOrder === 'none') return { label: `${u.roi} (${flyKey(u)})`, group: null, key: i };
    return { label: u.roi, group: flyKey(u), key: i };
  });
  const model = {
    rows, cols, values: meanOver(values, trials.length, unitIdx, colTrials),
    rowTitle: { fly: 'ROI', roi: 'fly', none: 'ROI (fly)' }[rowOrder],
    rowGroupTitle: { fly: 'fly', roi: 'ROI', none: null }[rowOrder],
    colTitle: 'odor', colGroupTitle: showGroups ? 'panel' : null,
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

// rows as columns and columns as rows, with their labels, titles and dendrograms
function transposeModel(m) {
  const nR = m.rows.length, nC = m.cols.length;
  const values = new Float32Array(nR * nC);
  for (let r = 0; r < nR; r++) for (let c = 0; c < nC; c++) values[c * nR + r] = m.values[r * nC + c];
  return {
    ...m, rows: m.cols, cols: m.rows, values, rowDendro: m.colDendro, colDendro: m.rowDendro, transposed: true,
    rowTitle: m.colTitle, rowGroupTitle: m.colGroupTitle, colTitle: m.rowTitle, colGroupTitle: m.rowGroupTitle,
  };
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
// Edits of a viewer's rows and columns, kept for each matrix shown until reset: keys
// removed, and the order they were dragged into. For neural data, rows are the units and
// columns the odors, however they are shown.
const newEdits = () => ({ rows: { removed: new Set(), order: null }, cols: { removed: new Set(), order: null } });

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
    this.hint = document.createElement('div');
    this.hint.className = 'vt-hint';
    container.appendChild(this.hint);
    this.heatmap = new Heatmap(this.wrap, {
      onReorder: (axis, keys) => this._reorder(axis, keys),
      onLabelMenu: (info) => this._labelMenu(info),
      onLabelRemove: ({ axis, keys }) => this._remove(axis, keys),
    });
    this.colorbar = $('.colorbar', container);
    const saved = LS.get('olfdb.viewer', {});
    this.opts = Object.assign(
      {
        mode: 'average', rowOrder: 'fly', roiSort: 'file', odorOrder: 'file', clusterRows: false, clusterCols: false, metric: 'correlation',
        cellW: 16, cellH: 16, colormap: 'oklab-div', range: 'twoslope', clip: false,
      },
      saved,
      { transpose: !!saved.transposed?.[kind] },
    );
    this.data = null;
    this.source = null; // the payload shown
    this.edits = newEdits();
    this.editsOf = new WeakMap(); // by payload, for when it is shown again
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
    $$('select[data-opt]', tb).forEach((s) => s.addEventListener('change', () => {
      this.opts[s.dataset.opt] = s.value;
      // an order chosen here replaces the one dragged into
      if (s.dataset.opt === 'rowOrder' || s.dataset.opt === 'roiSort') this.edits.rows.order = null;
      if (s.dataset.opt === 'odorOrder') this.edits.cols.order = null;
      this._save();
      this._rebuild();
    }));
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
    $('[data-role=reset]', tb).addEventListener('click', () => { Object.assign(this.edits, newEdits()); this._rebuild(); });
    wireMenu($('[data-role=export]', tb), ({ export: format }) => this.exportFigure(format));
  }

  _save() {
    const o = this.opts;
    LS.set('olfdb.viewer', {
      cellW: o.cellW, cellH: o.cellH, colormap: o.colormap, range: o.range, rowOrder: o.rowOrder, roiSort: o.roiSort, odorOrder: o.odorOrder,
      clusterRows: o.clusterRows, clusterCols: o.clusterCols, metric: o.metric,
      transposed: { ...LS.get('olfdb.viewer', {}).transposed, [this.kind]: o.transpose },
    });
  }

  _setKind(kind) {
    if (kind === this.kind) return;
    this.kind = kind;
    this.opts.transpose = !!LS.get('olfdb.viewer', {}).transposed?.[kind];
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
    $('input[data-opt=transpose]', tb).parentElement.title = neural ? 'Odors as rows and ROIs as columns' : 'Rows as columns and columns as rows';
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
    $('[data-role=rows-label]', tb).textContent = o.transpose ? 'Columns' : 'Rows';
  }

  // title: for the exported figure's metadata; fileStem: its file name
  setNeuralData(payload, { title = '', fileStem = 'heatmap' } = {}) {
    this._setKind('neural');
    if (payload !== this.source) {
      this.source = payload;
      this.data = { units: payload.units, trials: payload.trials, values: decodeValues(payload.values_b64) };
      this._editsFor(payload);
      this.clusterCache.clear();
    }
    this.title = title;
    this.fileStem = fileStem;
    this.opts.colormap = this.opts.colormap in COLORMAPS && COLORMAPS[this.opts.colormap].kind === 'diverging' ? this.opts.colormap : 'oklab-div';
    this._syncToolbar();
    this._rebuild();
  }

  setGenericData(ref, { title = '', fileStem = 'heatmap' } = {}) {
    this._setKind('generic');
    this.title = title;
    this.fileStem = fileStem;
    if (ref !== this.source) {
      this.source = ref;
      this._editsFor(ref);
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
          rows: rows.map((r) => ({ label: r.label, group: r.group, key: r.i })),
          cols: ref.cols.map((c, j) => ({ label: c.label, group: null, sub: null, key: j })),
          values: vals,
          rowTitle: ref.meta.row_label || 'row',
          rowGroupTitle: hasGroups ? 'group' : null,
          colTitle: ref.meta.col_label || 'column',
          colGroupTitle: null,
        },
      };
    }
    const kind = ref.meta.kind || 'sequential';
    this.opts.colormap = kind === 'binary' ? 'binary' : kind === 'diverging' ? 'oklab-div' : (COLORMAPS[this.opts.colormapSeq]?.kind === 'sequential' ? this.opts.colormapSeq : 'viridis');
    this._syncToolbar();
    this._rebuild();
  }

  _editsFor(payload) {
    if (!this.editsOf.has(payload)) this.editsOf.set(payload, newEdits());
    this.edits = this.editsOf.get(payload);
  }

  // the matrix less the rows and columns removed, in the order they were dragged into
  _genericModel() {
    const b = this.data.model, nC = b.cols.length;
    const rows = arrange(b.rows.map((_, i) => i), (i) => b.rows[i].key, this.edits.rows);
    const cols = arrange(b.cols.map((_, j) => j), (j) => b.cols[j].key, this.edits.cols);
    const values = new Float32Array(rows.length * cols.length);
    rows.forEach((i, r) => { cols.forEach((j, c) => { values[r * cols.length + c] = b.values[i * nC + j]; }); });
    return { ...b, rows: rows.map((i) => b.rows[i]), cols: cols.map((j) => b.cols[j]), values };
  }

  async _rebuild() {
    if (!this.data) return;
    const seq = ++this._seq;
    this._syncEdits();
    // ROI clustering pools the ROIs of every fly, so the rows are then not grouped
    const o = this.opts.clusterRows ? { ...this.opts, rowOrder: 'none' } : this.opts;
    if (this.kind !== 'neural') {
      if (COLORMAPS[o.colormap]?.kind === 'sequential') { this.opts.colormapSeq = o.colormap; this._save(); }
      this._show(this._genericModel());
      return;
    }
    const { rows, cols } = this.edits;
    // an order dragged into is not used while its axis is clustered, but comes back after
    const order = neuralOrder(this.data, o);
    order.unitIdx = arrange(order.unitIdx, (u) => u, { removed: rows.removed, order: o.clusterRows ? null : rows.order });
    order.conds = arrange(order.conds, (c) => c.key, { removed: cols.removed, order: o.clusterCols ? null : cols.order });
    let clustering = null;
    const notes = [];
    if (o.clusterRows || o.clusterCols) {
      const removed = (e) => [...e.removed].sort().join('\u0001');
      const key = [o.clusterRows, o.clusterCols, o.metric, removed(rows), removed(cols)].join('|');
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
          // from the same order whatever is shown, as that decides ties
          const input = neuralOrder(this.data, CLUSTER_INPUT_ORDER);
          clustering = await clusterNeural(this.clusterPool, this.data, {
            unitIdx: arrange(input.unitIdx, (u) => u, { removed: rows.removed, order: null }),
            conds: arrange(input.conds, (c) => c.key, { removed: cols.removed, order: null }),
          }, o, onProgress);
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
    const neural = this.kind === 'neural';
    const cm = COLORMAPS[o.colormap] || COLORMAPS['oklab-div'];
    const { vmin, vmax } = dataRange(model.values, o.clip);
    let norm;
    if (cm.kind === 'diverging') norm = makeNorm(o.range === 'symmetric' ? 'symmetric' : 'twoslope', vmin, vmax);
    else if (cm.kind === 'binary') norm = makeNorm('linear', 0, 1);
    else norm = makeNorm('linear', vmin, vmax);
    const dims = neural
      ? [`${model.rows.length.toLocaleString()} ROIs`, `${model.cols.length} ${o.mode === 'trials' ? 'trials' : 'odors'}`]
      : [model.rows.length, model.cols.length];
    // labels can be dragged along axes not clustered
    const drag = { rows: !(neural && o.clusterRows), cols: !(neural && o.clusterCols) };
    if (o.transpose) {
      model = transposeModel(model);
      dims.reverse();
      Object.assign(drag, { rows: drag.cols, cols: drag.rows });
    }
    this.model = model;
    this.norm = norm;
    this.lut = cm.lut;
    this.heatmap.setModel(model, { cellW: o.cellW, cellH: o.cellH, lut: cm.lut, norm, reorder: drag });
    drawColorbar(this.colorbar, cm.lut, norm);
    this.statsText = `${dims.join(' × ')}` + (neural && model.nPanels > 1 ? ` in ${model.nPanels} panels` : '') + ` · range ${fmtNum(vmin)} to ${fmtNum(vmax)}` +
      (!neural && this.data.meta.value_label ? ` (${this.data.meta.value_label})` : '');
    this._setNote(note);
    const grouped = (drag.rows && model.rows.some((r) => r.group != null)) || (drag.cols && model.cols.some((c) => c.group != null));
    const which = drag.rows && drag.cols ? 'a label' : drag.rows ? 'a row label' : 'a column label';
    this.hint.textContent = [
      drag.rows || drag.cols ? `Drag ${which} to move it${grouped ? ' within its group (drag a group’s name to move the group)' : ''}` : '',
      'Ctrl-click a label to remove it, or right-click it for a menu.',
    ].filter(Boolean).join('; ');
  }

  _setNote(note) {
    const el = $('[data-role=stats]', this.toolbar);
    el.textContent = note ? `${this.statsText} · ${note}` : this.statsText;
    el.title = el.textContent; // in full, as it is cut short when it does not fit
  }

  // the reset button, shown while anything is removed or dragged
  _syncEdits() {
    const { rows, cols } = this.edits;
    const nouns = this.kind === 'neural' ? ['ROI', 'odor'] : ['row', 'column'];
    const bits = [rows, cols].flatMap((e, k) => (e.removed.size ? [`${e.removed.size} ${nouns[k]}${e.removed.size === 1 ? '' : 's'} removed`] : []));
    if (rows.order || cols.order) bits.push('reordered');
    const btn = $('[data-role=reset]', this.toolbar);
    btn.hidden = !bits.length;
    btn.title = `Show every ${nouns.join(' and ')} again, in the order chosen above (now: ${bits.join(', ')})`;
  }

  // axis of the heatmap -> axis of the edits
  _editAxis(axis) { return this.opts.transpose ? (axis === 'rows' ? 'cols' : 'rows') : axis; }

  _reorder(axis, keys) {
    this.edits[this._editAxis(axis)].order = keys;
    this._rebuild();
  }

  // how many items of the heatmap's axis are shown (the trials of an odor being one)
  _nShown(axis) { return new Set(this.model[axis].map((it) => it.key)).size; }

  // removes keys of the heatmap's axis, unless none would be left (ctrl-click on a label)
  _remove(axis, keys) {
    if (keys.length >= this._nShown(axis)) return;
    const { removed } = this.edits[this._editAxis(axis)];
    keys.forEach((k) => removed.add(k));
    this._rebuild();
  }

  // right-click on a label: remove what it labels (a glomerulus also from every fly)
  _labelMenu({ axis, level, items, keys, x, y }) {
    const ax = this._editAxis(axis);
    const [one, many] = { neural: { rows: ['ROI', 'ROIs'], cols: ['odor', 'odors'] }, generic: { rows: ['row', 'rows'], cols: ['column', 'columns'] } }[this.kind][ax];
    const shown = this._nShown(axis);
    const it = items[0];
    const muted = (s) => ` <span class="muted">${esc(s)}</span>`;
    const entries = [level === 'group'
      ? { html: `Remove ${esc(it.group)}${muted(`${keys.length} ${keys.length === 1 ? one : many}`)}`, keys }
      : { html: `Remove ${esc(it.label)}${it.group != null ? muted(it.group) : ''}`, keys }];
    if (this.kind === 'neural' && ax === 'rows' && level === 'item') {
      const roi = this.data.units[it.key].roi;
      const same = this.model[axis].filter((r) => this.data.units[r.key].roi === roi).map((r) => r.key);
      if (!/^\d+$/.test(roi) && same.length > 1) entries.push({ html: `Remove ${esc(roi)} from every fly${muted(`${same.length} ROIs`)}`, keys: same });
    }
    showContextMenu(x, y, entries.map((e) => ({
      html: e.html,
      disabled: e.keys.length >= shown,
      title: e.keys.length >= shown ? `At least one ${one} has to stay` : '',
      onPick: () => this._remove(axis, e.keys),
    })));
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
// where a value in the upload form came from, when it was not typed in (shown as its color)
const SOURCES = { settings: 'from your settings', nas: 'found on tensor', file: 'read from the files you added' };
// fields that can be filled in automatically; once the user changes one, it is left alone
const AUTO_FIELDS = ['uploader_name', 'indicator', 'driver', 'coding_unit', 'genotype', 'age_min_days', 'age_max_days', 'wavelength_nm',
  'power_mw', 'power_tolerance_mw', 'power_measured_same_day', 'recording_system', 'panel_names', 'source_code_url', 'raw_data_path',
  'mocorr_data_path', 'notes'];
// fields about one dataset only, emptied for each new file
const DATASET_FIELDS = ['raw_data_path', 'mocorr_data_path'];
// <select> + "Other…" input pairs
const COMBOS = { indicator: ['#indicator-select', '#indicator-other'], coding_unit: ['#coding-unit-select', '#coding-unit-other'], recording_system: ['#system-select', '#system-other'] };
// the field a form element belongs to
const fieldKey = (el) => (el.name || '').replace(/_(select|other)$/, '');

function markSource(el, src, detail = '') {
  if (src) { el.dataset.src = src; el.title = detail ? `${SOURCES[src]}: ${detail}` : SOURCES[src]; } else { delete el.dataset.src; el.removeAttribute('title'); }
}

class UploadFlow {
  constructor() {
    this.options = null;
    this.stage = null;
    this.data = null;
    this.generatedReadme = '';
    this.viewer = null;
    this.yamlCheck = null;
    this.nas = null; // what the last lookup found on tensor
    this.manual = new Set(); // fields the user changed
    this.auto = new Set(); // fields filled in automatically
    this._lookupSeq = 0;
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
    dl('#dl-indicators', o.indicators); dl('#dl-coding-units', o.coding_units); dl('#dl-systems', o.recording_systems);
    $('#step-meta').wavelength_nm.value = o.default_wavelength_nm ?? 925;
    for (const [sel, other] of Object.values(COMBOS)) {
      const s = $(sel), oth = $(other);
      const sync = () => { oth.hidden = s.value !== '__other__'; if (!oth.hidden) oth.focus(); };
      s.addEventListener('change', sync);
    }
    this._autofill();
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
    $('#power-mw').addEventListener('input', () => this._syncPower());
    // capturing, to know a field was changed by the user before its own listeners run
    form.addEventListener('input', (e) => this._changedByUser(e.target), true);
    form.addEventListener('change', (e) => this._changedByUser(e.target), true);
    form.uploader_name.addEventListener('change', () => { if (this.stage) this.lookup(); });
    // checked on the server when left, as folders are in the Settings tab
    for (const key of DATASET_FIELDS) form[key].addEventListener('change', () => checkPathInput(form[key], $(`[data-msg-for=${key}]`)));
    form.addEventListener('submit', (e) => { e.preventDefault(); this.generateReadme(); });

    $('#yaml-btn').addEventListener('click', () => $('#yaml-input').click());
    $('#yaml-input').addEventListener('change', (e) => { if (e.target.files.length) this.addYaml([...e.target.files]); e.target.value = ''; });
    $('#xml-btn').addEventListener('click', () => $('#xml-input').click());
    $('#xml-input').addEventListener('change', (e) => { if (e.target.files[0]) this.addXml(e.target.files[0]); e.target.value = ''; });

    $('#readme-text').addEventListener('input', () => { $('#readme-preview').innerHTML = renderMarkdown($('#readme-text').value); });
    $('#readme-regenerate').addEventListener('click', () => this.generateReadme());
    $('#commit-btn').addEventListener('click', () => this.commit());
    $('#done-another').addEventListener('click', () => this.reset());
  }

  _syncPower() {
    const has = $('#power-mw').value !== '';
    $('#power-tol').disabled = !has;
    $$('#power-sameday input').forEach((r) => { r.disabled = !has; });
  }

  // ------------------------------------------------- automatically filled fields
  _changedByUser(el) {
    const key = fieldKey(el);
    if (!AUTO_FIELDS.includes(key) || this.manual.has(key)) return;
    this.manual.add(key);
    this.auto.delete(key);
    this._mark(key, null);
  }

  _setField(key, value) {
    const v = value == null ? '' : String(value);
    if (COMBOS[key]) {
      const [s, other] = COMBOS[key].map((sel) => $(sel));
      const known = v !== '__other__' && [...s.options].some((o) => o.value === v);
      s.value = known ? v : v ? '__other__' : s.options[0].value;
      other.value = known ? '' : v;
      other.hidden = s.value !== '__other__';
    } else if (key === 'power_measured_same_day') {
      $$('#power-sameday input').forEach((r) => { r.checked = r.value === v; });
    } else {
      $('#step-meta')[key].value = v;
    }
    if (key === 'power_mw') this._syncPower();
    if (DATASET_FIELDS.includes(key)) showPathCheck($('#step-meta')[key], $(`[data-msg-for=${key}]`), null);
  }

  _fieldEls(key) {
    if (COMBOS[key]) return COMBOS[key].map((sel) => $(sel));
    if (key === 'power_measured_same_day') return [$('#power-sameday')];
    return [$('#step-meta')[key]];
  }

  _mark(key, src, detail = '') {
    for (const el of this._fieldEls(key)) markSource(el, src, detail);
  }

  // Fills each field the user has not changed with the most specific value there is for
  // it: found on tensor, then read from the dataset's files, then the defaults in
  // Settings (a guess from the files comes last). A field filled in before that has
  // nothing now goes back to how it started.
  _autofill() {
    const best = new Map();
    const offer = (key, value, src, rank, detail = '') => {
      if (value == null || value === '' || (best.get(key)?.rank ?? 0) >= rank) return;
      best.set(key, { value, src, rank, detail });
    };
    const s = settings.current();
    offer('uploader_name', s.name, 'settings', 2);
    for (const [k, v] of Object.entries(s.defaults)) offer(k, v, 'settings', 2);
    const sum = this.data?.summary;
    if (sum?.panel_names?.length) offer('panel_names', sum.panel_names.join(', '), 'file', 3, 'the panel level of the data file');
    if (sum?.roi_kind === 'glomeruli') offer('coding_unit', 'glomeruli', 'file', 3, 'the ROIs are glomeruli');
    if (sum?.roi_kind === 'integers' && this.options?.coding_units.includes('KCs')) offer('coding_unit', 'KCs', 'file', 1, 'numbered ROIs');
    if (this.nas) {
      offer('raw_data_path', this.nas.raw_data_path, 'nas', 4, 'the folder with the flies found');
      offer('mocorr_data_path', this.nas.mocorr_data_path, 'nas', 4, 'the folder with the flies found');
    }
    for (const key of AUTO_FIELDS) {
      if (this.manual.has(key)) continue;
      const b = best.get(key);
      if (b) {
        this._setField(key, b.value);
        this._mark(key, b.src, b.detail);
        this.auto.add(key);
      } else if (this.auto.has(key)) {
        this._setField(key, key === 'wavelength_nm' ? this.options?.default_wavelength_nm ?? 925 : null);
        this._mark(key, null);
        this.auto.delete(key);
      }
    }
  }

  // the name or the defaults in Settings changed
  settingsChanged() {
    if (!this.options) return;
    const name = $('#step-meta').uploader_name.value;
    this._autofill();
    if (this.stage && $('#step-meta').uploader_name.value !== name) this.lookup();
  }

  // Looks for the dataset's flies in the uploader's folders on tensor, and attaches the
  // Experiment.xml and odor YAMLs found there, unless some are attached already.
  async lookup() {
    const seq = ++this._lookupSeq;
    const box = $('#nas-lookup');
    const name = $('#step-meta').uploader_name.value.trim();
    const stageId = this.stage.stage_id;
    const done = (html) => { this.nas = null; box.replaceChildren(msgBox('info', html)); this._autofill(); };
    if (!name) { done('Set your name and data folders in <a href="#settings">Settings</a> to have the files of this dataset found on tensor.'); return; }
    box.replaceChildren(msgBox('info', `Looking for these flies in the folders of ${esc(name)} on tensor…`));
    let res;
    try {
      res = await api(`/api/stage/${stageId}/lookup`, { method: 'POST', json: { name } });
    } catch (e) {
      if (seq === this._lookupSeq) box.replaceChildren(msgBox('error', `Could not look on tensor: ${esc(e.message)}`));
      return;
    }
    if (seq !== this._lookupSeq || this.stage?.stage_id !== stageId) return;
    if (!res.user) {
      done(`No data folders are saved for <b>${esc(name)}</b>. Add them in <a href="#settings">Settings</a> to have the raw data, Experiment.xml and odor YAML of your uploads found on tensor.`);
      return;
    }
    this.nas = res;
    if (res.experiment_xmls.length && !this.stage.experiment_xml) await this.addXml({ tensorPath: res.experiment_xmls[0] });
    if (res.odor_yamls.selected.length && !this.stage?.odor_yamls.length) await this.addYaml({ tensorPaths: res.odor_yamls.selected });
    if (seq !== this._lookupSeq || this.stage?.stage_id !== stageId) return;
    this._renderLookup();
    this._autofill();
  }

  _renderLookup() {
    const n = this.nas, st = this.stage, box = $('#nas-lookup');
    const flyName = (f) => `${f.date} fly ${f.fly_num}`;
    const plural = (k, word) => `${k} ${word}${k === 1 ? '' : 's'}`;
    const flies = (key, folders, path) => {
      if (!n.user[folders].length) return '<span class="muted">no such folders in your settings</span>';
      const missing = n.flies.filter((f) => !f[key].length);
      const found = n.flies.length - missing.length;
      let s = `${found} of ${n.flies.length} ${n.flies.length === 1 ? 'fly' : 'flies'}`;
      if (path) s += ` in <code>${esc(path)}</code>`;
      if (found && missing.length) s += ` <span class="muted">(not ${esc(missing.slice(0, 4).map(flyName).join(', '))}${missing.length > 4 ? ', …' : ''})</span>`;
      return s;
    };
    const short = (p) => p.replace(/\/Experiment\.xml$/, '').replace(n.raw_data_path ? `${n.raw_data_path}/` : '', '');
    let xml = n.experiment_xmls.length ? plural(n.experiment_xmls.length, 'recording') : '<span class="muted">none</span>';
    if (n.experiment_xmls.length) {
      const current = st.experiment_xml_tensor_path;
      const opts = (n.experiment_xmls.includes(current) ? [] : [`<option value="">${st.experiment_xml ? 'the one you added' : 'none'}</option>`])
        .concat(n.experiment_xmls.map((p) => `<option value="${esc(p)}"${p === current ? ' selected' : ''}>${esc(short(p))}</option>`));
      xml += `; <label>Experiment.xml attached: <select data-xml>${opts.join('')}</select></label>`;
    }
    const y = n.odor_yamls;
    const attached = new Set(st.odor_yamls.map((a) => a.tensor_path).filter(Boolean));
    let yaml;
    if (!n.user.odor_dirs.length) yaml = '<span class="muted">no such folders in your settings</span>';
    else if (y.selected.length) {
      yaml = `${plural(y.selected.length, 'file')} describing all ${y.n_matrix_odors} odors`;
      yaml += y.selected.every((p) => attached.has(p)) ? ', attached' : ' <span class="muted">(not attached, as other YAML was attached already)</span>';
    } else if (y.candidates.length) {
      yaml = `${plural(y.candidates.length, 'file')} describing some of the odors, but none for ${esc(y.missing.slice(0, 6).join(', '))}${y.missing.length > 6 ? ', …' : ''}; attach any of them below`;
    } else yaml = '<span class="muted">none describing these odors</span>';
    const flyRows = n.flies.map((f) => `<tr><td>${esc(flyName(f))}</td><td>${f.raw.map((p) => `<code>${esc(p)}</code>`).join('<br>') || '—'}</td>
      <td>${f.mocorr.map((p) => `<code>${esc(p)}</code>`).join('<br>') || '—'}</td><td>${f.recordings.length}</td></tr>`).join('');
    const yamlRows = y.candidates.map((c) => `<tr><td><code>${esc(c.path)}</code></td><td class="muted">${esc(c.reason)}</td>
      <td>${c.n_covered} of ${y.n_matrix_odors} odors</td><td>${attached.has(c.path) ? '<span class="muted">attached</span>'
        : `<button type="button" class="btn small" data-attach-yaml="${esc(c.path)}">attach</button>`}</td></tr>`).join('');
    box.innerHTML = `<div class="msg info lookup">
      <div class="lookup-head"><b>Found on tensor</b><span>in the folders of ${esc(n.user.name)} <span class="muted">(<a href="#settings">settings</a>)</span></span>
        <button type="button" class="btn small" data-again>Look again</button></div>
      <ul><li>Raw data: ${flies('raw', 'raw_dirs', n.raw_data_path)}</li>
        <li>Motion-corrected data: ${flies('mocorr', 'mocorr_dirs', n.mocorr_data_path)}</li>
        <li>ThorImage recordings: ${xml}</li>
        <li>Odor YAML: ${yaml}</li></ul>
      <details><summary>Each fly${y.candidates.length ? ', and the odor YAMLs tried' : ''}</summary>
        <table class="odor-table"><thead><tr><th>Fly</th><th>Raw data</th><th>Motion-corrected</th><th>Recordings</th></tr></thead><tbody>${flyRows}</tbody></table>
        ${yamlRows ? `<table class="odor-table"><thead><tr><th>Odor YAML</th><th>Found by</th><th>Describes</th><th></th></tr></thead><tbody>${yamlRows}</tbody></table>` : ''}
      </details></div>`;
    $('[data-again]', box).addEventListener('click', () => this.lookup());
    $('[data-xml]', box)?.addEventListener('change', (e) => { if (e.target.value) this.addXml({ tensorPath: e.target.value }); });
    $$('[data-attach-yaml]', box).forEach((b) => b.addEventListener('click', () => this.addYaml({ tensorPaths: [b.dataset.attachYaml] })));
  }

  // ------------------------------------------------------------------- staging
  reset() {
    this.stage = null; this.data = null; this.yamlCheck = null; this.generatedReadme = '';
    this.nas = null; this._lookupSeq++;
    for (const key of DATASET_FIELDS) {
      this._setField(key, null);
      this._mark(key, null);
      this.manual.delete(key);
      this.auto.delete(key);
    }
    $('#nas-lookup').innerHTML = '';
    $('#file-result').innerHTML = '';
    $('#preview').hidden = true;
    $('#step-meta').hidden = true;
    $('#step-readme').hidden = true;
    $('#step-done').hidden = true;
    $('#yaml-list').innerHTML = ''; $('#yaml-check').innerHTML = ''; $('#xml-info').innerHTML = ''; this._renderXml();
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

    const meta = $('#step-meta');
    meta.hidden = false;
    this._autofill();
    this.lookup();
    meta.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // files: File objects, or {tensorPaths} for files on tensor
  async addYaml(files) {
    const stageId = this.stage.stage_id;
    let req;
    if (files.tensorPaths) req = { method: 'POST', json: { tensor_paths: files.tensorPaths } };
    else {
      const form = new FormData();
      for (const f of files) form.append('file', f, f.name);
      req = { method: 'POST', form };
    }
    const box = $('#yaml-check');
    box.replaceChildren(msgBox('info', 'Checking the odors against the YAML…'));
    let res;
    try { res = await api(`/api/stage/${stageId}/odor_yaml`, req); } catch (e) { box.replaceChildren(msgBox('error', esc(e.message))); return; }
    if (this.stage?.stage_id === stageId) this._renderYaml(res);
  }

  async removeYaml(name) {
    const res = await api(`/api/stage/${this.stage.stage_id}/odor_yaml/${encodeURIComponent(name)}`, { method: 'DELETE' });
    this._renderYaml(res);
  }

  _renderYaml(res) {
    this.stage = res.stage;
    this.yamlCheck = res.check;
    const list = $('#yaml-list');
    list.innerHTML = res.stage.odor_yamls.map((y) => `<div class="file"${y.tensor_path ? ` data-src="nas" title="${esc(`${SOURCES.nas}: ${y.tensor_path}`)}"` : ''}>
      <code>${esc(y.name)}</code><span class="muted">${y.n_odors} odors${y.tensor_path ? `, from <code>${esc(y.tensor_path)}</code>` : ''}</span>
      <button type="button" class="btn small" data-remove="${esc(y.name)}">remove</button></div>`).join('');
    $$('[data-remove]', list).forEach((b) => b.addEventListener('click', () => this.removeYaml(b.dataset.remove)));
    const box = $('#yaml-check');
    box.innerHTML = '';
    if (res.errors?.length) box.appendChild(msgBox('error', `Not added:${listHtml(res.errors)}`));
    if (this.nas) this._renderLookup();
    const c = res.check;
    if (!c) return;
    if (c.ok) {
      box.appendChild(msgBox('ok', `All ${c.n_matrix_odors} odors in the matrix are described in the YAML${c.unused_yaml.length ? ` (${c.unused_yaml.length} YAML odors are not in the matrix: ${esc(c.unused_yaml.join(', '))})` : ''}.`));
    } else {
      const rows = c.missing.map((m) => `<tr><td><code>${esc(m.odor)}</code></td><td class="muted">${esc(m.hint)}</td></tr>`).join('');
      box.appendChild(msgBox('error', `<b>${c.missing.length} of ${c.n_matrix_odors} odors in the matrix are not in the YAML</b> (name and concentration must both match). Add the right YAML, remove this one, or fix the file; saving is blocked until the check passes.<table class="check-table">${rows}</table>`));
    }
  }

  // file: a File, or {tensorPath} for one on tensor
  async addXml(file) {
    const stageId = this.stage.stage_id;
    let req;
    if (!(file instanceof File)) req = { method: 'POST', json: { tensor_path: file.tensorPath } };
    else {
      const form = new FormData();
      form.append('file', file, file.name);
      req = { method: 'POST', form };
    }
    $('#xml-status').textContent = 'Reading…';
    let res;
    try { res = await api(`/api/stage/${stageId}/experiment_xml`, req); } catch (e) { this._renderXml(); $('#xml-info').replaceChildren(msgBox('error', esc(e.message))); return; }
    if (this.stage?.stage_id !== stageId) return;
    if (!res.ok) { this._renderXml(); $('#xml-info').replaceChildren(msgBox('error', `Not added:${listHtml(res.errors)}`)); return; }
    this.stage = res.stage;
    this._renderXml();
    if (this.nas) this._renderLookup();
  }

  _renderXml() {
    const st = this.stage, status = $('#xml-status');
    markSource(status, null);
    if (!st?.experiment_xml) { status.textContent = ''; return; }
    const from = st.experiment_xml_tensor_path;
    status.innerHTML = `<code>${esc(st.experiment_xml)}</code> added${from ? ` from <code>${esc(from)}</code>` : ''}`;
    if (from) markSource(status, 'nas');
    const info = st.experiment_xml_info || {};
    const kv = [];
    if (info.date) kv.push(['date', info.date]);
    if (info.software_version) kv.push(['ThorImage', info.software_version]);
    if (info.lsm) kv.push(['frame', `${info.lsm.pixelX ?? '?'} × ${info.lsm.pixelY ?? '?'} px${info.lsm.frameRate ? `, ${info.lsm.frameRate} fps` : ''}${info.lsm.pixelSizeUM ? `, ${info.lsm.pixelSizeUM} µm/px` : ''}`]);
    if (info.zstage?.steps) kv.push(['z planes', `${info.zstage.steps}${info.zstage.stepSizeUM ? ` × ${info.zstage.stepSizeUM} µm` : ''}`]);
    if (info.pockels?.start != null) kv.push(['pockels', `${info.pockels.start}%`]);
    $('#xml-info').innerHTML = kv.length ? `<div class="kv">${kv.map(([k, v]) => `<span>${esc(k)}</span><span>${esc(v)}</span>`).join('')}</div>` : '';
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
      mocorr_data_path: f.mocorr_data_path.value.trim() || null,
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
    settings.rememberName(meta.uploader_name);
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
    settings.rememberName(meta.uploader_name);
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
    this.readmeDrafts = new Map(); // dataset id -> its README as being edited (kept while another is shown)
    $('#ds-readme-edit').addEventListener('click', () => this.editReadme());
    $('#ds-readme-save').addEventListener('click', () => this.saveReadme());
    $('#ds-readme-text').addEventListener('input', (e) => {
      if (this.selected != null) this.readmeDrafts.set(this.selected, e.target.value);
      $('#ds-readme-preview').innerHTML = renderMarkdown(e.target.value);
    });
    window.addEventListener('beforeunload', (e) => {
      if ([...this.readmeDrafts].some(([id, text]) => text !== this.cache.get(id)?.d.readme_md)) { e.preventDefault(); e.returnValue = ''; }
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
    $$('.data-link').forEach((menu) => {
      menu.addEventListener('click', async (e) => {
        if (e.target.closest('[data-save-path]')) { this.saveDataPath(menu); return; }
        if (e.target.closest('[data-movies]')) {
          menu.open = false;
          player.open(this.selected, this.cache.get(this.selected).d.title, menu.dataset.kind);
          return;
        }
        const b = e.target.closest('[data-copy]');
        if (!b) return;
        try { await copyText(b.dataset.copy); b.textContent = 'Copied'; } catch (err) { b.textContent = `Could not copy: ${err.message}`; }
        setTimeout(() => { b.textContent = 'Copy'; }, 1500);
      });
      // a path entered here is checked as in the upload form
      menu.addEventListener('input', (e) => {
        if (!e.target.matches('.data-path-input')) return;
        const problem = e.target.value.trim() ? localDirProblem(e.target.value) : null;
        showPathCheck(e.target, $('.path-msg', menu), problem ? { ok: false, error: problem } : null);
      });
      menu.addEventListener('change', (e) => { if (e.target.matches('.data-path-input')) checkPathInput(e.target, $('.path-msg', menu)); });
      menu.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.matches('.data-path-input')) { e.preventDefault(); this.saveDataPath(menu); }
      });
    });
    $('#pick-clear').addEventListener('click', () => { this.picked.clear(); this._syncPicks(); });
    $('#dataset-list').addEventListener('change', (e) => {
      const cb = e.target;
      const ids = cb.matches('.pick') ? [+cb.dataset.id] : cb.matches('.pick-group') ? cb.dataset.ids.split(',').map(Number) : [];
      for (const id of ids) { if (cb.checked) this.picked.add(id); else this.picked.delete(id); }
      if (ids.length) this._syncPicks();
    });
  }

  invalidate() { this.datasets = null; this.cache.clear(); }

  // a dataset deleted (in the Admin tab)
  forget(id) {
    this.invalidate();
    this.picked.delete(id);
    this.readmeDrafts.delete(id);
  }

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
    $$('#dataset-view details.menu').forEach((menu) => { menu.open = false; }); // they were about another dataset
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
    setStatus($('#ds-readme-status'), '');
    this._showReadme(id);
    $('#ds-files').innerHTML = d.files.map((f) => `<a href="/api/datasets/${id}/files/${encodeURI(f)}" download>${esc(f)}</a>`).join('');
    if (!this.viewer) this.viewer = new MatrixViewer($('#dataset-viewer'));
    this.viewer.setNeuralData(m, { title: d.title, fileStem: `olfdb_dataset-${id}` });
    if (!$('#ds-odors').hidden) this.showOdors(id);
    this.showDataPaths(id);
  }

  // The links to the dataset's raw and motion-corrected data on tensor, red with the
  // reason when there is none to follow: looked up afresh each time (tensor may come and
  // go, and so may the folders set in the Admin tab), after the rest is shown.
  async showDataPaths(id) {
    const menus = $$('.data-link');
    menus.forEach((m) => this._dataLink(m, null));
    let res;
    try { res = await api(`/api/datasets/${id}/data_paths`); } catch (e) {
      if (this.selected === id) menus.forEach((m) => this._dataLink(m, { status: 'unchecked', error: `could not ask the server (${e.message})` }));
      return;
    }
    if (this.selected === id) menus.forEach((m) => this._dataLink(m, res[m.dataset.kind]));
  }

  // r: as from tensor.find_data, or null while it is looked up
  _dataLink(menu, r) {
    const label = DATA_LINKS[menu.dataset.kind], list = $('.menu-list', menu);
    menu.classList.toggle('missing', !!r && r.status !== 'ok');
    const why = r ? DATA_WHY[r.status] : 'checking…';
    $('summary', menu).innerHTML = `${label}${why ? ` <span class="why">${why}</span>` : ''}`;
    $('summary', menu).title = r && r.status !== 'ok' ? `${label}: ${r.error}` : '';
    if (!r) { list.innerHTML = '<p class="menu-note">Looking for it on the server…</p>'; return; }
    const kind = menu.dataset.kind;
    const edit = `<div class="data-edit">
        <label class="data-edit-label" for="data-path-${kind}">${r.status === 'none' ? 'Enter its path' : 'Its path'} on tensor, from the top (as at upload)</label>
        <div class="data-edit-row"><input id="data-path-${kind}" class="data-path-input" spellcheck="false" autocomplete="off"
          placeholder="e.g. tom/${kind === 'raw' ? 'raw_data' : 'analysis'}/2023-11-19" value="${esc(r.input || '')}">
          <button type="button" class="btn small primary" data-save-path>Save</button></div>
        <span class="path-msg"></span>
        <p class="menu-note">Saving changes the dataset's details, not its README: edit that too if it gives the path.</p>
      </div>`;
    if (r.status !== 'ok') {
      list.innerHTML = (r.status === 'none'
        ? `<p class="menu-note">No ${label.toLowerCase()} path was entered for this dataset.</p>`
        : `<p class="menu-note">Entered: <code>${esc(r.input || '')}</code></p><p class="menu-note data-error">${esc(r.error)}</p>`) + edit;
      this._showDataSaved(menu);
      return;
    }
    const shown = r.tensor_path ?? r.path;
    const entries = r.entries.map((x) => `<li${x.dir ? ' class="dir"' : ''}>${esc(x.name)}${x.dir ? '/' : ''}${x.size != null ? ` <span class="muted">${fmtBytes(x.size)}</span>` : ''}</li>`).join('');
    list.innerHTML = `<div class="kv data-where">
        <span>${r.tensor_path != null ? 'On tensor' : 'On the server'}</span><span><code>${esc(shown)}</code> <button type="button" class="btn small" data-copy="${esc(shown)}">Copy</button></span>
        ${r.tensor_path != null ? `<span>On the server</span><span><code>${esc(r.path)}</code></span>` : ''}
      </div>
      ${r.is_dir ? `<div class="data-movies"><button type="button" class="btn small" data-movies>Play movies…</button>
        <span class="muted">${kind === 'mocorr' ? 'the TIFFs in it, with their ImageJ ROIs' : 'the ThorImage .raw files in it'}</span></div>` : ''}
      ${r.is_dir ? (r.n_entries ? `<ul class="data-entries">${entries}</ul>${r.n_entries > r.entries.length ? `<p class="menu-note">… and ${r.n_entries - r.entries.length} more</p>` : ''}` : '<p class="menu-note">The folder is empty.</p>') : '<p class="menu-note">A file, not a folder.</p>'}
      <p class="menu-note">Browsers cannot open folders on your computer from a web page: open it from where tensor is mounted on yours.</p>
      ${edit}`;
    this._showDataSaved(menu);
  }

  _showDataSaved(menu) {
    if (this._dataSaved !== menu.dataset.kind) return;
    this._dataSaved = null;
    const msg = $('.path-msg', menu);
    msg.className = 'path-msg ok';
    msg.textContent = 'Saved.';
  }

  // Saves the path entered in a data menu as the dataset's (checked by the server as at upload).
  async saveDataPath(menu) {
    const id = this.selected, key = `${menu.dataset.kind}_data_path`;
    const input = $('.data-path-input', menu), msg = $('.path-msg', menu);
    const value = input.value.trim();
    const problem = value ? localDirProblem(value) : null;
    if (problem) { showPathCheck(input, msg, { ok: false, error: problem }); return; }
    msg.className = 'path-msg';
    msg.textContent = 'Saving…';
    let res;
    try {
      res = await api(`/api/datasets/${id}/data_paths`, { method: 'PUT', json: { [key]: value } });
    } catch (e) {
      showPathCheck(input, msg, { ok: false, error: `not saved: ${e.message}` });
      return;
    }
    if (!res.ok) { showPathCheck(input, msg, res.checks[key] || { ok: false, error: res.errors.join('; ') }); return; }
    const entry = this.cache.get(id);
    if (entry) entry.d.metadata = res.metadata;
    this._dataSaved = menu.dataset.kind;
    if (this.selected === id) await this.showDataPaths(id);
  }

  // the README of the dataset shown, or the edit of it under way
  _showReadme(id) {
    const { d } = this.cache.get(id), draft = this.readmeDrafts.get(id), editing = draft != null;
    $('#ds-readme').classList.toggle('editing', editing);
    $('#ds-readme-html').hidden = editing;
    $('#ds-readme-editor').hidden = !editing;
    $('#ds-readme-edit').hidden = editing;
    $('#ds-readme-save').hidden = !editing;
    $('#ds-readme-html').innerHTML = d.readme_html;
    if (editing) {
      $('#ds-readme-text').value = draft;
      $('#ds-readme-preview').innerHTML = renderMarkdown(draft);
    }
  }

  editReadme() {
    const id = this.selected;
    if (id == null) return;
    this.readmeDrafts.set(id, this.cache.get(id).d.readme_md);
    setStatus($('#ds-readme-status'), '');
    this._showReadme(id);
    $('#ds-readme-text').focus();
  }

  // Asks whether to overwrite the README on the server with the edit, or to discard the edit.
  async saveReadme() {
    const id = this.selected, status = $('#ds-readme-status');
    const { d } = this.cache.get(id);
    const text = $('#ds-readme-text').value;
    const done = (msg) => { this.readmeDrafts.delete(id); this._showReadme(id); setStatus(status, msg); };
    if (text === d.readme_md) { done('Nothing was changed.'); return; }
    $('#readme-confirm-path').textContent = d.readme_path;
    const answer = await askDialog($('#readme-confirm'));
    if (answer === 'discard') { done('Your edits were discarded.'); return; }
    if (answer !== 'overwrite') return;
    setStatus(status, 'Saving…');
    let res;
    try {
      res = await api(`/api/datasets/${id}/readme`, { method: 'PUT', json: { readme: text, base: d.readme_md } });
    } catch (e) {
      setStatus(status, `Not saved: ${e.message}`, 'error');
      return;
    }
    Object.assign(d, { readme_md: res.readme_md, readme_html: res.readme_html });
    if (this.selected === id) done(`Saved to ${res.readme_path}.`);
    else this.readmeDrafts.delete(id);
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

const DATA_LINKS = { raw: 'Raw data', mocorr: 'Motion corrected / traces' };
const DATA_WHY = { ok: '', none: 'not entered', not_found: 'not found', not_allowed: 'not allowed', unreadable: 'not readable', unchecked: 'not checked' };

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
  const nCols = hasGlom ? 12 : 11;
  const body = runsOf(rows, (r) => r.panel).map((g) => {
    const title = g.key != null ? `<tr class="group"><td colspan="${nCols}">${esc(g.key)}</td></tr>` : '';
    return title + rows.slice(g.start, g.end).map((r) => {
      const o = r.odor_id != null ? odors[r.odor_id] : null;
      const c = o ? idCells(o) : {};
      let name;
      if (o) name = `<a href="#odors/${o.id}">${esc(o.name)}</a>`;
      else if (r.match === 'ambiguous') name = `<span class="muted">ambiguous: ${r.candidates.map((i) => `<a href="#odors/${i}">${esc(odors[i].name)}</a>`).join(' or ')}</span>`;
      else name = `<span class="muted">${r.yaml?.name ? `${esc(r.yaml.name)}, ` : ''}not in the odor table</span>`;
      return `<tr><td><code>${esc(r.odor)}</code></td><td>${name}</td><td>${r.log10_conc ?? ''}</td><td>${c.cid || ''}</td>
        <td class="inchi">${esc(o?.inchi || '')}</td><td>${esc(r.yaml?.solvent || o?.solvent || '')}</td>
        ${hasGlom ? `<td>${esc(r.yaml?.glomerulus || '')}</td>` : ''}<td>${c.catalog || ''}</td><td>${c.metacyc || ''}</td><td>${c.lotus || ''}</td>
        <td class="structure-cell">${o ? structureImg(o) : ''}</td><td>${c.cas || ''}</td></tr>`;
    }).join('');
  }).join('');
  const n = rows.filter((r) => r.odor_id != null).length;
  return `<p class="muted">${n} of ${rows.length} odors are in the <a href="#odors">odor table</a> (matched through the dataset's odor YAML where it has one, otherwise by abbreviation).</p>
    <table class="odor-table"><thead><tr><th>Odor</th><th>Name</th><th>log<sub>10</sub> conc.</th><th>CID</th><th>InChI</th><th>Solvent</th>
    ${hasGlom ? '<th>Target glomerulus</th>' : ''}<th>Catalog no.</th><th>MetaCyc</th><th>LOTUS</th><th>Structure</th><th>CAS</th></tr></thead><tbody>${body}</tbody></table>`;
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
    $('#sheet-check').parentElement.hidden = !s.configured;
    if (!s.configured) {
      box.innerHTML = `<p class="muted">No odor sheet is set up yet: its link goes in the <a href="#admin">Admin tab</a>, which can also import a CSV with the sheet's columns
        (its template has the odors from the thesis tables, and can start a new Google sheet).</p>`;
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

  // a CSV is imported from the Admin tab, the sheet synced from the Odors tab; one preview at a time
  async runPreview(file) {
    const box = $(file ? '#admin-odor-preview' : '#odor-preview');
    $(file ? '#odor-preview' : '#admin-odor-preview').replaceChildren();
    this.previewBox = box;
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
    const p = this.preview, box = this.previewBox;
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
      // a sync from the sheet goes in its history with a name; a CSV import with none
      const name = where === 'sheet' ? `<label class="check">Your name <input id="odor-sync-name" value="${esc(LS.get('olfdb.uploader_name', ''))}"></label>` : '';
      parts.push(`<div class="actions">${name}
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
    const p = this.preview, box = this.previewBox;
    const nc = $$('[data-change]:checked', box).length, nd = $$('[data-delete]:checked', box).length;
    btn.textContent = `Apply: ${p.added.length} new, ${nc} of ${p.changed.length} changes${p.missing.length ? `, ${nd} deletions` : ''}`;
    btn.disabled = !p.added.length && !nc && !nd;
  }

  async apply() {
    const box = this.previewBox, out = $('#odor-apply-result');
    const sheet = this.preview.source === 'Google sheet';
    const name = sheet ? $('#odor-sync-name').value.trim() : '';
    if (sheet && !name) { out.replaceChildren(msgBox('error', 'Your name is required (it goes in the sync history).')); return; }
    if (sheet) settings.rememberName(name);
    const json = {
      fingerprint: this.preview.fingerprint, ...(sheet ? { applied_by: name } : {}),
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
      $('#odor-table').innerHTML = '<div class="empty">The odor table is empty. Sync it from the odor sheet, or import a CSV in the Admin tab (its template has the odors from the thesis).</div>';
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
        return `<tr><td>${esc(fmtWhen(s.applied_at))}</td><td>${s.applied_by ? esc(s.applied_by) : '<span class="muted">—</span>'}</td><td>${esc(s.source)}</td><td>${what || syncCounts(s)}</td></tr>`;
      }).join('')}</tbody></table>` : '<p class="muted">No syncs yet.</p>';
  }
}

// ================================================================ REFERENCE
class Reference {
  constructor() { this.items = null; this.viewer = null; this.selected = null; this.cache = new Map(); }

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
    try {
      // kept, so that rows and columns removed or moved stay so when it is shown again
      if (!this.cache.has(id)) this.cache.set(id, await api(`/api/reference/${encodeURIComponent(id)}`));
      ref = this.cache.get(id);
    } catch (e) { $('#ref-title').textContent = 'Could not load'; $('#ref-desc').textContent = e.message; return; }
    if (this.selected !== id) return;
    $('#ref-title').textContent = ref.meta.title;
    $('#ref-desc').textContent = [ref.meta.description, ref.meta.source ? `Source: ${ref.meta.source}` : ''].filter(Boolean).join(' ');
    if (!this.viewer) this.viewer = new MatrixViewer($('#reference-viewer'), { kind: 'generic' });
    this.viewer.setGenericData(ref, { title: ref.meta.title, fileStem: `olfdb_reference-${id}` });
  }
}

// ================================================================= SETTINGS
// The user's name and data folders on tensor (saved on the server, which looks for the
// files of their uploads there), and their defaults for the upload form (kept in this browser).
const DIR_KINDS = ['raw_dirs', 'mocorr_dirs', 'odor_dirs'];
const DEFAULT_FIELDS = ['indicator', 'driver', 'coding_unit', 'recording_system', 'genotype', 'age_min_days', 'age_max_days',
  'wavelength_nm', 'power_mw', 'power_tolerance_mw', 'power_measured_same_day', 'panel_names', 'source_code_url', 'notes'];
const NUMBER_FIELDS = ['age_min_days', 'age_max_days', 'wavelength_nm', 'power_mw', 'power_tolerance_mw'];

// what is wrong with a folder as typed, as far as can be told without the server
function localDirProblem(path) {
  if (/^\s*([/\\~]|[A-Za-z]:)/.test(path)) return 'absolute paths are not accepted: give the folder relative to the top of tensor';
  if (path.split(/[/\\]/).includes('..')) return "'..' is not allowed: give the folder from the top of tensor";
  return null;
}

// a row of a .dirlist (folders, one per input, with a last empty one to add another)
function dirRow(list, value) {
  const row = document.createElement('div');
  row.className = 'dir-row';
  row.innerHTML = `<input class="dir" spellcheck="false" autocomplete="off">
    <span class="dir-mark" aria-hidden="true"></span><button type="button" class="btn small" data-remove title="Remove this folder">×</button>
    <div class="dir-msg"></div>`;
  const input = $('input', row);
  input.placeholder = list.dataset.placeholder;
  input.setAttribute('aria-label', list.dataset.label || 'Folder on tensor');
  input.value = value;
  return row;
}

// c: a check from the server (or a local one), or null to clear it; one with a warning is still ok
function showDirCheck(row, c) {
  const warn = !!c?.ok && !!c.warning;
  row.classList.toggle('ok', !!c?.ok && !warn);
  row.classList.toggle('warn', warn);
  row.classList.toggle('bad', !!c && !c.ok);
  $('.dir-mark', row).textContent = c ? (c.ok ? (warn ? '!' : '✓') : '✗') : '';
  $('.dir-msg', row).textContent = c ? (c.ok ? c.warning || '' : c.error) : '';
}

class SettingsPage {
  constructor() {
    this.name = LS.get('olfdb.uploader_name', '');
    this.defaults = LS.get('olfdb.defaults', {});
    this.user = null; // the folders as saved on the server
    this.owner = ''; // whose folders are shown ('' if not saved under any name yet)
    this.loaded = false;
    this._checking = false;
    this._defaultsSeq = 0;
    const box = $('#settings-folders');
    $('#settings-name').addEventListener('change', (e) => this.setName(e.target.value));
    box.addEventListener('input', (e) => { if (e.target.matches('input.dir')) this._dirEdited(e.target); });
    box.addEventListener('change', (e) => { if (e.target.matches('input.dir')) this.check(); });
    box.addEventListener('click', (e) => {
      const row = e.target.closest('[data-remove]')?.closest('.dir-row');
      if (!row) return;
      if (row.nextElementSibling) row.remove(); else { $('input', row).value = ''; showDirCheck(row, null); }
      this.check();
    });
    const f = $('#settings-defaults');
    f.power_mw.addEventListener('input', () => this._syncPower());
    f.addEventListener('change', () => this.checkDefaults());
    f.addEventListener('submit', (e) => e.preventDefault());
    $('#defaults-clear').addEventListener('click', () => { this._fillDefaults({}); this.checkDefaults(); });
  }

  // what the upload form is filled in from
  current() { return { name: this.name, defaults: this.defaults }; }

  // a name entered elsewhere (upload form, odor sync) is kept only while none is set here
  rememberName(name) {
    if (this.name || !name) return;
    this.name = name;
    LS.set('olfdb.uploader_name', name);
    $('#settings-name').value = name;
  }

  async load() {
    if (this.loaded) return;
    this.loaded = true;
    $('#settings-name').value = this.name;
    $('#tensor-root').textContent = upload.options?.tensor_root || '?';
    this._fillDefaults(this.defaults);
    if (this.name) {
      try { this.user = (await api(`/api/user?name=${encodeURIComponent(this.name)}`)).user; } catch { /* check() says so */ }
    }
    // folders that could not be saved yet are kept in this browser until they can
    const draft = LS.get('olfdb.dirs_draft', null);
    const dirs = draft && draft.name === this.name ? draft.dirs : this.user || {};
    this.owner = draft && draft.name === this.name ? draft.name : this.user?.name || '';
    this._setDirs(dirs);
    if (this.name || DIR_KINDS.some((k) => dirs[k]?.length)) this.check();
    else this._status('none', 'No Folders Yet', 'Enter your name and your folders above.');
  }

  async setName(value) {
    const name = value.trim().replace(/\s+/g, ' ');
    $('#settings-name').value = name;
    if (name === this.name) return;
    this.name = name;
    LS.set('olfdb.uploader_name', name);
    upload.settingsChanged();
    if (name) {
      let user = null;
      try { user = (await api(`/api/user?name=${encodeURIComponent(name)}`)).user; } catch { /* check() says so */ }
      if (this.name !== name) return;
      // the folders saved under this name before (perhaps on another computer); someone
      // else's are not carried over to a new name, but ones entered before any name are
      this.user = user;
      if (user || this.owner) {
        this._setDirs(user || {});
        this.owner = user?.name || '';
      }
    }
    this.check();
  }

  _setDirs(dirs) {
    for (const kind of DIR_KINDS) {
      const list = $(`.dirlist[data-kind=${kind}]`);
      list.replaceChildren(...[...(dirs[kind] || []), ''].map((p) => dirRow(list, p)));
    }
  }

  _dirEdited(input) {
    const row = input.closest('.dir-row');
    const problem = localDirProblem(input.value);
    showDirCheck(row, problem ? { ok: false, error: problem } : null);
    if (!row.nextElementSibling && input.value.trim()) row.parentElement.appendChild(dirRow(row.parentElement, ''));
    this._status('pending', 'Not Checked Yet', 'Folders are checked on the server when you leave their field.');
  }

  _status(kind, label, detail = '') {
    $('#paths-status').innerHTML = `<span class="indicator ${kind}">${esc(label)}</span>${detail ? ` <span class="muted">${detail}</span>` : ''}`;
  }

  // Checks every folder on the server, which saves them under the name once all are found
  // there. One check at a time, so that saves reach the server in order; edits made during
  // one are checked right after it.
  async check() {
    if (this._checking) { this._checkAgain = true; return; }
    this._checking = true;
    try {
      do {
        this._checkAgain = false;
        await this._check();
      } while (this._checkAgain);
    } finally {
      this._checking = false;
    }
  }

  async _check() {
    const rows = {}, dirs = {};
    for (const kind of DIR_KINDS) {
      rows[kind] = $$(`.dirlist[data-kind=${kind}] .dir-row`).filter((r) => $('input', r).value.trim());
      dirs[kind] = rows[kind].map((r) => $('input', r).value.trim());
    }
    this._status('pending', 'Checking…');
    let res;
    try {
      res = await api('/api/user', { method: 'POST', json: { name: this.name, ...dirs } });
    } catch (e) {
      if (!this._checkAgain) this._status('bad', 'Could Not Check', esc(e.message));
      return;
    }
    if (this._checkAgain) return;
    for (const kind of DIR_KINDS) {
      res.dirs[kind].forEach((c, i) => {
        const input = $('input', rows[kind][i]);
        if (input.value.trim() !== c.input) return; // changed since
        if (c.ok && document.activeElement !== input) input.value = c.path;
        showDirCheck(rows[kind][i], c);
      });
    }
    if (res.saved) {
      this.user = res.user;
      this.owner = res.user.name;
      LS.set('olfdb.dirs_draft', null);
    } else LS.set('olfdb.dirs_draft', { name: this.name, dirs });
    const all = DIR_KINDS.flatMap((k) => res.dirs[k]);
    const bad = all.filter((c) => !c.ok).length;
    const saved = res.saved ? `Saved on the server for <b>${esc(res.user.name)}</b>.` : 'Enter your name above to save them on the server.';
    if (!res.root.ok) this._status('bad', 'Tensor Not Accessible', `${esc(res.root.error)}. Nothing is saved until the server can read tensor.`);
    else if (bad) this._status('bad', `${bad} of ${all.length} Paths Not Accessible`, 'Fix or remove the folders marked above; they are saved on the server once all of them are found there.');
    else if (!all.length) this._status('none', 'No Folders Yet', this.name ? '' : 'Enter your name and your folders above.');
    else this._status('ok', 'All Paths Accessible', saved);
  }

  // ------------------------------------------------------------------ defaults
  _fillDefaults(values) {
    const f = $('#settings-defaults');
    for (const key of DEFAULT_FIELDS) {
      const v = values[key] == null ? '' : String(values[key]);
      if (key === 'power_measured_same_day') $$(`input[name=${key}]`, f).forEach((r) => { r.checked = r.value === v; });
      else f[key].value = v;
    }
    this._syncPower();
  }

  _syncPower() {
    const f = $('#settings-defaults'), has = f.power_mw.value !== '';
    f.power_tolerance_mw.disabled = !has;
    $$('input[name=power_measured_same_day]', f).forEach((r) => { r.disabled = !has; });
  }

  // the defaults entered, as the upload form would send them (empty ones left out)
  _collectDefaults() {
    const f = $('#settings-defaults'), out = {};
    const hasPower = f.power_mw.value !== '';
    for (const key of DEFAULT_FIELDS) {
      let v;
      if (key === 'power_measured_same_day') {
        const r = $(`input[name=${key}]:checked`, f)?.value;
        v = hasPower && r ? r === 'true' : null;
      } else if (key === 'power_tolerance_mw' && !hasPower) v = null;
      else if (NUMBER_FIELDS.includes(key)) v = f[key].value === '' ? null : Number(f[key].value);
      else v = f[key].value.trim() || null;
      if (v != null) out[key] = v;
    }
    return out;
  }

  // Checks the defaults as an upload is checked (by the browser for the same kinds of
  // inputs, then by the server); they are kept, and used, once they pass.
  async checkDefaults() {
    const seq = ++this._defaultsSeq;
    const f = $('#settings-defaults'), box = $('#defaults-status');
    const values = this._collectDefaults();
    const problems = new Map();
    for (const el of $$('input[name], textarea[name]', f)) {
      if (!el.disabled && !el.checkValidity()) problems.set(el.name, `${el.closest('.field').firstElementChild.textContent}: ${el.validationMessage}`);
    }
    let res;
    try {
      const checkable = Object.fromEntries(Object.entries(values).filter(([k]) => !problems.has(k)));
      res = await api('/api/metadata/check', { method: 'POST', json: { metadata: checkable } });
    } catch (e) {
      if (seq === this._defaultsSeq) box.replaceChildren(msgBox('error', `Could not check the defaults: ${esc(e.message)}`));
      return;
    }
    if (seq !== this._defaultsSeq) return;
    for (const e of res.errors) if (!problems.has(e.field)) problems.set(e.field, e.message);
    $$('[name]', f).forEach((el) => el.classList.toggle('invalid', problems.has(el.name)));
    if (problems.size) { box.replaceChildren(msgBox('error', `Not kept until fixed:${listHtml([...problems.values()])}`)); return; }
    this.defaults = values;
    LS.set('olfdb.defaults', values);
    const n = Object.keys(values).length;
    box.replaceChildren(msgBox('ok', n ? `${n} default${n === 1 ? '' : 's'} kept on this computer.` : 'No defaults set.'));
    upload.settingsChanged();
  }
}

// ===================================================================== MERGE
// Every dataset of a coding unit in one download, groups of them optionally scaled to each
// other first (app/merge.py): how they would be scaled is shown as the options change.
class MergePage {
  constructor() {
    this._seq = 0;
    $('#tab-merge').addEventListener('change', (e) => {
      if (e.target.matches('#merge-unit, [name=merge-mode], [name=merge-by], #merge-target, #merge-tol, #merge-power-tol')) this.preview();
    });
    $('#merge-plan').addEventListener('click', (e) => {
      if (e.target.closest('[data-scales]')) download(`/api/merge/scales?${this._query()}`, $('#merge-status'));
    });
    $$('#merge-download button').forEach((b) => b.addEventListener('click', () => {
      if ($('#merge-unit').value) download(`/api/merge/download?${this._query(b.dataset.format)}`, $('#merge-status'));
    }));
  }

  // each time the tab is opened (datasets may have been uploaded since)
  async load() {
    let res;
    try { res = await api('/api/merge'); } catch (e) { $('#merge-plan').replaceChildren(msgBox('error', esc(e.message))); return; }
    const sel = $('#merge-unit'), keep = sel.value || LS.get('olfdb.merge_unit', '');
    sel.innerHTML = res.coding_units.map((u) => `<option value="${esc(u.coding_unit)}">${esc(u.coding_unit)} (${u.ids.length} dataset${u.ids.length === 1 ? '' : 's'})</option>`).join('');
    $$('#merge-download button').forEach((b) => { b.disabled = !res.coding_units.length; });
    if (!res.coding_units.length) { $('#merge-plan').replaceChildren(msgBox('info', 'Nothing has been uploaded yet.')); return; }
    if (res.coding_units.some((u) => u.coding_unit === keep)) sel.value = keep;
    this.preview();
  }

  _options() {
    return {
      coding_unit: $('#merge-unit').value, mode: $('[name=merge-mode]:checked').value,
      by: $$('[name=merge-by]:checked').map((c) => c.value), target: $('#merge-target').value || 'average',
      tolerance: Number($('#merge-tol').value), power_tolerance: Number($('#merge-power-tol').value),
    };
  }

  _query(format) {
    const o = this._options();
    return new URLSearchParams({ ...o, by: o.by.join(','), ...(format ? { format } : {}) }).toString();
  }

  async preview() {
    const seq = ++this._seq, o = this._options(), box = $('#merge-plan');
    LS.set('olfdb.merge_unit', o.coding_unit);
    $('#merge-options').disabled = o.mode === 'none';
    if (!o.coding_unit) return;
    $('#merge-power-tol').disabled = !o.by.includes('power');
    if (!(o.tolerance >= 0 && o.tolerance <= 10)) { box.replaceChildren(msgBox('error', 'The distance between concentrations must be between 0 and 10 log10 units.')); return; }
    if (!(o.power_tolerance >= 0 && o.power_tolerance <= 1000)) { box.replaceChildren(msgBox('error', 'The distance between power ranges must be between 0 and 1000 mW.')); return; }
    let p;
    try { p = await api('/api/merge/preview', { method: 'POST', json: o }); } catch (e) {
      if (seq === this._seq) box.replaceChildren(msgBox('error', esc(e.message)));
      return;
    }
    if (seq !== this._seq) return;
    const sel = $('#merge-target');
    sel.innerHTML = [['average', 'the average of the groups'], ...p.groups.map((g) => [g.key, `the group ${g.key}`])]
      .map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join('');
    sel.value = p.target || 'average';
    box.innerHTML = this._planHtml(p);
  }

  _planHtml(p) {
    const link = (id) => `<a href="#browse/${id}">#${id}</a>`;
    const factor = (s) => `× ${+s.toPrecision(3)}`;
    const groupOf = new Map(p.groups.flatMap((g) => g.ids.map((i) => [i, g])));
    const why = new Map(p.unscaled.map((u) => [u.id, u.reason]));
    const power = (d) => (d.power_mw == null ? '<span class="muted">not measured</span>'
      : `${d.power_mw} ± ${d.power_tolerance_mw ?? '0.5 <span class="muted">(assumed)</span>'} mW`);
    const parts = [];
    if (p.mode === 'groups' && !p.groups.length) {
      parts.push(msgBox('warn', 'None of these datasets can be grouped this way (why is in the table below), so none is scaled.').outerHTML);
    } else if (p.mode === 'groups') {
      const target = p.target === 'average' ? 'the average of the groups' : `the group <b>${esc(p.target)}</b>`;
      parts.push(`<p>${p.groups.length} group${p.groups.length === 1 ? '' : 's'}, each scaled to ${target}; responses compared
        ${p.by_roi ? 'per glomerulus' : 'averaged over ROIs (they are not all glomeruli)'}.</p>`);
      if (p.groups.length < 2) parts.push(msgBox('warn', 'With a single group, nothing is scaled: group the datasets by something that tells them apart.').outerHTML);
      parts.push(`<table class="odor-table"><thead><tr><th>Group</th><th>Datasets</th><th>Scale</th><th>Compared on</th><th>r</th><th></th></tr></thead><tbody>
        ${p.groups.map((g) => `<tr><td><b>${esc(g.key)}</b></td><td>${g.ids.map(link).join(' ')}</td>
          <td class="scale">${g.scale == null ? '<span class="muted">× 1</span>' : factor(g.scale)}</td>
          <td>${g.n_points ? `${g.n_stimuli} stimul${g.n_stimuli === 1 ? 'us' : 'i'}, ${g.n_points} responses` : ''}</td>
          <td class="scale">${g.r == null ? '' : g.r.toFixed(2)}</td><td class="muted">${esc(g.note || '')}</td></tr>`).join('')}
        </tbody></table>`);
    } else parts.push('<p>Every dataset stays at its own scale.</p>');
    // with scaling, the factor each dataset is multiplied by (as in scaling.csv, which can be downloaded on its own)
    const scaled = p.mode === 'groups';
    parts.push(`<div class="merge-scales-head"><h3>${scaled ? 'Scale factor of each dataset' : `Datasets (${p.datasets.length})`}</h3>
      ${scaled ? '<button type="button" class="btn small" data-scales title="scaling.csv: one row per dataset, by dataset_id as in the matrices">Download as CSV</button>' : ''}</div>
      <table class="odor-table merge-scales"><thead><tr><th>Dataset</th><th>Experimenter</th><th>Recording system</th><th>Power</th>
      ${scaled ? '<th>Group</th><th>Scale</th>' : ''}</tr></thead><tbody>
      ${p.datasets.map((d) => {
        const g = groupOf.get(d.id);
        return `<tr><td>${link(d.id)} ${esc(d.title)}</td><td>${esc(d.uploader || '')}</td><td>${esc(d.system || '')}</td><td>${power(d)}</td>
          ${scaled ? `<td>${g ? esc(g.key) : `<span class="muted">${esc(why.get(d.id) || '')}</span>`}</td>
          <td class="scale">${factor(p.scales[d.id])}</td>` : ''}</tr>`;
      }).join('')}</tbody></table>`);
    return parts.join('');
  }
}

// ===================================================================== ADMIN
// Site-wide settings: the folders on the server the app may read and write (fixed when
// the server starts, so saved changes wait for a restart, which can be asked for here
// too) and the odor sheet's link; the server's name and addresses; deleting datasets.
// Shown each time the tab is opened, once the user says they may change these settings.
const ADMIN_KINDS = ['read_dirs', 'write_dirs'];

// what is wrong with a folder as typed, as far as can be told without the server
function localServerDirProblem(path) {
  if (!/^\s*\//.test(path)) return 'give the absolute path on the server, starting with /';
  if (path.split('/').includes('..')) return "'..' is not allowed: give the folder itself";
  return null;
}

class AdminPage {
  constructor() {
    this.state = null; // as from GET /api/admin
    this.datasets = null; // as from GET /api/datasets
    this.back = 'browse'; // where to go if the user may not change these settings
    this._checkSeq = 0;
    const box = $('#admin-folders');
    box.addEventListener('input', (e) => { if (e.target.matches('input.dir')) this._dirEdited(e.target); });
    box.addEventListener('change', (e) => { if (e.target.matches('input.dir')) this.check(); });
    box.addEventListener('click', (e) => {
      const row = e.target.closest('[data-remove]')?.closest('.dir-row');
      if (!row) return;
      if (row.nextElementSibling) row.remove(); else { $('input', row).value = ''; showDirCheck(row, null); }
      this.check();
    });
    $('#admin-save').addEventListener('click', () => this.save());
    $('#admin-sheet-save').addEventListener('click', () => this.saveSheet());
    $('#admin-sheet-url').addEventListener('keydown', (e) => { if (e.key === 'Enter') this.saveSheet(); });
    $('#admin-restart').addEventListener('click', () => this.restart());
    $('#admin-confirm').addEventListener('close', () => this._answered());
    $('#admin-ds-filter').addEventListener('input', () => this._showDatasets());
    $('#admin-ds-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-delete]');
      if (b) this.deleteDataset(+b.dataset.delete);
    });
  }

  // the tab was opened; `back`: the place to return to if the answer is not yes
  enter(back) {
    this.back = back;
    $('#admin-page').hidden = true;
    const dialog = $('#admin-confirm');
    if (dialog.open) return;
    dialog.returnValue = '';
    dialog.showModal();
  }

  // another tab was opened (with the browser's back button, while the question was asked)
  leave() {
    const dialog = $('#admin-confirm');
    if (dialog.open) dialog.close('left');
  }

  _answered() {
    const answer = $('#admin-confirm').returnValue; // '' if dismissed with Escape
    if (answer === 'left') return;
    if (answer === 'yes') { $('#admin-page').hidden = false; this.load(); } else location.replace(`#${this.back}`);
  }

  async load() {
    this.loadHost();
    this.loadDatasets();
    $('#admin-result').replaceChildren();
    $('#admin-sheet-result').replaceChildren();
    $('#admin-ds-result').replaceChildren();
    let s;
    try { s = await api('/api/admin'); } catch (e) { $('#admin-result').replaceChildren(msgBox('error', `Could not load the settings: ${esc(e.message)}`)); return; }
    this._show(s);
  }

  _show(s) {
    this.state = s;
    $('#admin-data-dir').textContent = s.data_dir;
    $('#admin-data-dir-where').textContent = s.restart.by === 'Docker' ? ' in the container; on the server, the folder docker-compose.yml mounts there' : '';
    $('#admin-tensor-root').textContent = s.tensor_root;
    $('#admin-deleted-dir').textContent = s.deleted_dir;
    for (const kind of ADMIN_KINDS) {
      const list = $(`#admin-folders .dirlist[data-kind=${kind}]`);
      list.dataset.label = 'Folder on the server';
      list.replaceChildren(...[...s.saved[kind], ''].map((p) => dirRow(list, p)));
      s.checks[kind].forEach((c, i) => showDirCheck(list.children[i], c));
    }
    this._showSheet();
    this._showServer();
  }

  _rows(kind) { return $$(`#admin-folders .dirlist[data-kind=${kind}] .dir-row`).filter((r) => $('input', r).value.trim()); }

  _dirEdited(input) {
    const row = input.closest('.dir-row');
    const problem = input.value.trim() ? localServerDirProblem(input.value) : null;
    showDirCheck(row, problem ? { ok: false, error: problem } : null);
    if (!row.nextElementSibling && input.value.trim()) row.parentElement.appendChild(dirRow(row.parentElement, ''));
    $('#admin-result').replaceChildren();
  }

  // checks of the server (rows: by kind, those with a folder in them, as sent)
  _showChecks(rows, checks) {
    for (const kind of ADMIN_KINDS) {
      checks[kind].forEach((c, i) => {
        const input = $('input', rows[kind][i]);
        if (input.value.trim() !== c.input) return; // changed since
        if (c.ok && document.activeElement !== input) input.value = c.path;
        showDirCheck(rows[kind][i], c);
      });
    }
  }

  _collect() {
    const rows = Object.fromEntries(ADMIN_KINDS.map((k) => [k, this._rows(k)]));
    return { rows, dirs: Object.fromEntries(ADMIN_KINDS.map((k) => [k, rows[k].map((r) => $('input', r).value.trim())])) };
  }

  async check() {
    const seq = ++this._checkSeq;
    const { rows, dirs } = this._collect();
    let res;
    try { res = await api('/api/admin/check', { method: 'POST', json: dirs }); } catch { return; }
    if (seq === this._checkSeq) this._showChecks(rows, res.checks);
  }

  async save() {
    const out = $('#admin-result');
    const { rows, dirs } = this._collect();
    let res;
    try { res = await api('/api/admin', { method: 'POST', json: dirs }); } catch (e) { out.replaceChildren(msgBox('error', `Not saved: ${esc(e.message)}`)); return; }
    if (!res.ok) {
      this._showChecks(rows, res.checks);
      out.replaceChildren(msgBox('error', `Not saved:${listHtml(res.errors)}`));
      return;
    }
    this._show(res);
    out.replaceChildren(msgBox('ok', res.pending ? 'Saved. These folders take effect when the server restarts (below).' : 'Saved: these are the folders in effect now.'));
  }

  _showSheet() {
    const o = this.state.odor_sheet, input = $('#admin-sheet-url');
    if (document.activeElement !== input) input.value = o.url || '';
    $('#admin-sheet-source').innerHTML = o.saved ? `Set here ${esc(fmtWhen(o.updated_at))}.`
      : o.url ? 'Set on the server (<code>OLFDB_ODOR_SHEET_URL</code>); saving a link here replaces it.'
        : 'None yet: until there is one, the odor table can only be synced from CSVs imported below.';
  }

  async saveSheet() {
    const out = $('#admin-sheet-result'), url = $('#admin-sheet-url').value.trim();
    out.replaceChildren(msgBox('info', url ? 'Saving, and reading the sheet…' : 'Clearing…'));
    let res;
    try { res = await api('/api/admin/odor_sheet', { method: 'POST', json: { url } }); } catch (e) { out.replaceChildren(msgBox('error', `Not saved: ${esc(e.message)}`)); return; }
    if (!res.ok) { out.replaceChildren(msgBox('error', `Not saved:${listHtml(res.errors)}`)); return; }
    this.state.odor_sheet = res.odor_sheet;
    this._showSheet();
    const r = res.read;
    if (!url) out.replaceChildren(msgBox('ok', res.odor_sheet.url ? 'Cleared here: the sheet set on the server (OLFDB_ODOR_SHEET_URL) is used again.' : 'Cleared: there is no odor sheet now.'));
    else if (r.error) out.replaceChildren(msgBox('warn', `Saved, but it cannot be read: ${esc(r.error)}`));
    else out.replaceChildren(msgBox('ok', `Saved: the Odors tab syncs from this sheet, which has ${r.n_rows} odors.`));
    odorsPage.loadStatus();
  }

  async loadHost() {
    const box = $('#admin-host-info');
    let h;
    try { h = await api('/api/admin/host'); } catch (e) { box.innerHTML = `<span>Name</span><span class="muted">could not be found (${esc(e.message)})</span>`; return; }
    const name = h.source === 'container'
      ? `${esc(h.name)} <span class="muted">(the Docker container's; for the server's, mount its /etc/hostname as docker-compose.yml does)</span>`
      : esc(h.name);
    const none = h.source === 'mounted' ? `none: the network's DNS does not know ${esc(h.name)}` : 'none found';
    const addresses = h.addresses.length ? h.addresses.map((a) => `<code>${esc(a)}</code>`).join(' ') : `<span class="muted">${none}</span>`;
    box.innerHTML = [
      ['Name', name],
      [h.source === 'mounted' ? 'IP addresses (by DNS)' : 'IP addresses', addresses],
      ['Reached here at', `<code>${esc(location.origin)}</code>`],
    ].map(([a, b]) => `<span>${a}</span><span>${b}</span>`).join('');
  }

  async loadDatasets() {
    try { this.datasets = (await api('/api/datasets')).datasets; } catch (e) {
      $('#admin-ds-list').replaceChildren(msgBox('error', `Could not load the datasets: ${esc(e.message)}`));
      return;
    }
    this._showDatasets();
  }

  // newest upload first, those with every word of the filter somewhere
  _showDatasets() {
    if (!this.datasets) return;
    const box = $('#admin-ds-list');
    if (!this.datasets.length) { box.innerHTML = '<p class="muted">No datasets have been uploaded.</p>'; return; }
    const words = $('#admin-ds-filter').value.toLowerCase().split(/\s+/).filter(Boolean);
    const text = (d) => `#${d.id} ${d.title} ${d.uploader_name} ${d.panel_names || ''} ${d.coding_unit}`.toLowerCase();
    const shown = this.datasets.filter((d) => words.every((w) => text(d).includes(w))).sort((a, b) => b.id - a.id);
    if (!shown.length) { box.innerHTML = '<p class="muted">No dataset matches.</p>'; return; }
    box.innerHTML = `<table class="odor-table"><thead><tr><th>#</th><th>Dataset</th><th>Uploaded</th><th>Last fly</th><th>Flies</th><th>ROIs</th><th>Odors</th><th></th></tr></thead><tbody>
      ${shown.map((d) => `<tr><td class="num">${d.id}</td><td><a href="#browse/${d.id}">${esc(d.title)}</a></td>
        <td>${esc(d.uploader_name)}, ${esc(d.created_at.slice(0, 10))}</td><td>${esc(d.last_fly_date || '')}</td>
        <td class="num">${d.n_flies}</td><td class="num">${d.n_units.toLocaleString()}</td><td class="num">${d.n_odors}</td>
        <td><button type="button" class="btn small" data-delete="${d.id}">Delete…</button></td></tr>`).join('')}</tbody></table>`;
  }

  async deleteDataset(id) {
    const d = this.datasets.find((x) => x.id === id), out = $('#admin-ds-result');
    if (!d) return;
    const where = `${this.state?.deleted_dir ?? 'data/deleted'}/${id}`;
    $('#delete-confirm-q').textContent = `Delete dataset #${id}, “${d.title}”?`;
    $('#delete-confirm-detail').innerHTML = `Uploaded by ${esc(d.uploader_name)} on ${esc(d.created_at.slice(0, 10))}: ${d.n_flies} flies, `
      + `${d.n_units.toLocaleString()} ROIs, ${d.n_odors} odors. It is gone from the site at once; its files are moved to <code>${esc(where)}</code> on the server.`;
    if (await askDialog($('#delete-confirm')) !== 'delete') return;
    let res;
    try { res = await api(`/api/admin/datasets/${id}`, { method: 'DELETE' }); } catch (e) {
      out.replaceChildren(msgBox('error', `#${id} was not deleted: ${esc(e.message)}`));
      this.loadDatasets();
      return;
    }
    out.replaceChildren(msgBox('ok', `Deleted #${id}, ${esc(d.title)}. ${res.files ? `Its files are in <code>${esc(res.files)}</code> on the server.` : 'It had no files on the server.'}`));
    this.datasets = this.datasets.filter((x) => x.id !== id);
    this._showDatasets();
    browse.forget(id);
  }

  _showServer() {
    const s = this.state, k = s.kernel;
    const [kind, label, detail] = s.pending
      ? ['warn', 'Restart Needed', 'The folders saved above take effect when the server restarts.']
      : ['ok', 'In Effect', 'The server has been held to the folders above since it started.'];
    $('#admin-status').innerHTML = `<span class="indicator ${kind}">${label}</span> <span class="muted">${detail}</span>`;
    const folders = (dirs) => (dirs.length ? dirs.map((d) => `<code>${esc(d)}</code>`).join(' ') : 'none');
    const rows = [['Started', s.started_at ? esc(fmtWhen(s.started_at)) : '?']];
    if (s.pending && s.active) rows.push(['Reading now', folders(s.active.read_dirs)], ['Writing now', `${folders(s.active.write_dirs)} (and its data folder)`]);
    rows.push(
      ['Held to them by', k?.enforced ? `the kernel (Landlock, ABI ${k.abi}) and the app's own checks` : `the app's own checks only (Landlock: ${esc(k?.reason || 'not in use')})`],
      ['Last changed', s.saved.updated_at ? esc(fmtWhen(s.saved.updated_at)) : 'never (these are the defaults)'],
      ['Started again by', s.restart.possible ? esc(s.restart.by) : 'nothing: restart it by hand on the server (DEPLOY.md)'],
    );
    $('#admin-info').innerHTML = rows.map(([a, b]) => `<span>${a}</span><span>${b}</span>`).join('');
    const btn = $('#admin-restart');
    btn.disabled = !s.restart.possible;
    btn.title = s.restart.possible ? '' : 'The server is not run by Docker or systemd, which would start it again';
    btn.classList.toggle('primary', s.pending);
  }

  // Asks the server to restart, then waits until it answers again, started anew.
  async restart() {
    const btn = $('#admin-restart'), status = $('#admin-restart-status');
    btn.disabled = true;
    setStatus(status, 'Asking the server to restart…');
    let res;
    try {
      res = await api('/api/admin/restart', { method: 'POST', json: {} });
    } catch (e) {
      setStatus(status, `Could not restart: ${e.message}`, 'error');
      btn.disabled = false;
      return;
    }
    const t0 = Date.now();
    for (;;) {
      await new Promise((r) => { setTimeout(r, 1000); });
      const health = await api('/api/health').catch(() => null);
      if (health && health.started !== res.started_at) break;
      const secs = Math.round((Date.now() - t0) / 1000);
      if (secs > 90) {
        setStatus(status, `The server has not come back after ${secs} s: see its logs (DEPLOY.md, section 7).`, 'error');
        btn.disabled = false;
        return;
      }
      setStatus(status, `Restarting… ${secs} s`);
    }
    setStatus(status, 'Restarted.');
    $('#topbar-status').textContent = '';
    await this.load();
  }
}

// ==================================================================== router
const upload = new UploadFlow();
const browse = new Browse();
const odorsPage = new OdorsPage();
const reference = new Reference();
const mergePage = new MergePage();
const settings = new SettingsPage();
const admin = new AdminPage();
const player = new MoviePlayer($('#player'), { api, esc, fmtBytes });

let currentTab = null;
let lastPlace = 'browse'; // the hash of the last place shown other than the Admin tab

function route() {
  const [tab, id] = location.hash.replace(/^#/, '').split('/');
  const name = ['upload', 'browse', 'merge', 'odors', 'reference', 'settings', 'admin'].includes(tab) ? tab : 'browse';
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  $$('.tabs a').forEach((a) => a.classList.toggle('active', a.dataset.tab === name));
  if (name === 'admin' && currentTab !== 'admin') admin.enter(lastPlace);
  if (name !== 'admin') { admin.leave(); lastPlace = location.hash.replace(/^#/, '') || 'browse'; }
  currentTab = name;
  if (name === 'browse') { browse.load().then(() => { if (id) browse.show(+id); }); }
  if (name === 'odors') { odorsPage.highlightId = id ? +id : null; odorsPage.load().then(() => { if (id) odorsPage.highlight(+id); }); }
  if (name === 'reference') { reference.load().then(() => { if (id) reference.show(id); }); }
  if (name === 'settings') settings.load();
  if (name === 'merge') mergePage.load();
  if (name === 'browse' && !id) { $('#browse-empty').hidden = false; $('#dataset-view').hidden = true; browse.selected = null; }
  if (name === 'reference' && !id) { $('#reference-empty').hidden = false; $('#reference-view').hidden = true; reference.selected = null; }
}

window.addEventListener('hashchange', route);
upload.init().then(route);
api('/api/health').then(() => { $('#topbar-status').textContent = ''; }).catch(() => { $('#topbar-status').textContent = 'server unreachable'; });
api('/api/odors/status').then((s) => setOdorBadge(s.changed)).catch(() => {});
