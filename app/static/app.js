// Odor response database — frontend (no build step, no dependencies).
import { Heatmap } from './heatmap.js';
import { COLORMAPS, colormapsOfKind, makeNorm, drawColorbar, fmtNum } from './colormaps.js';
import { MAX_CLUSTER_ITEMS, ClusterPool, layoutDendrogram } from './cluster.js';
import { exportFigure } from './figure.js';
import { MoviePlayer } from './player.js';
import { SIMILARITIES, similarityMatrix, tanimoto, correlateSimilarities, sharedSimilarities, holm } from './similarity.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode etc. */ } },
};

// How odors are shown, as chosen in Settings (kept in this browser): lowerOdors, their
// names (abbreviations) in lowercase, in downloads too but of the original files
const DISPLAY_KEY = 'olfdb.display';
const display = { lowerOdors: false, ...LS.get(DISPLAY_KEY, {}) };

// odor names (abbreviations) as shown in lowercase: each lowercased, but for those that
// would then be the same as another (differing from it only in case), which are left as
// they are, so that no two odors become one (as odors.lowercase_map on the server)
function lowercaseMap(names) {
  const groups = new Map();
  for (const n of new Set(names)) groups.set(n.toLowerCase(), [...(groups.get(n.toLowerCase()) || []), n]);
  const out = new Map();
  for (const [low, same] of groups) for (const n of same) out.set(n, same.length === 1 ? low : n);
  return out;
}

// the names in an odor string ('A @ -3 + b @ -2': A and b), but solvent
const odorNames = (s) => String(s).split(' + ').filter((p) => p !== 'solvent').map((p) => p.split(' @ ')[0]);

// shows odor strings (one odor, or a mixture) of `odorStrs`, all shown together, as
// Settings say: each odor's name lowercased (lowercaseMap, over them all), or as it is
function odorShower(odorStrs) {
  let map = null;
  return (s) => {
    if (!display.lowerOdors || s == null) return s;
    map ??= lowercaseMap(odorStrs.flatMap(odorNames));
    return String(s).split(' + ').map((p) => {
      const at = p.indexOf(' @ '), name = at < 0 ? p : p.slice(0, at);
      return (map.get(name) ?? name) + (at < 0 ? '' : p.slice(at));
    }).join(' + ');
  };
}

// shows the odor table's abbreviations of `abbrevs`, all shown together, as Settings say
function abbrevShower(abbrevs) {
  let map = null;
  return (a) => {
    if (!display.lowerOdors || !a) return a;
    map ??= lowercaseMap(abbrevs.filter(Boolean));
    return map.get(a) ?? a;
  };
}

// a download's query parameter for odor names in lowercase, if Settings say so
const lowerQuery = (sep = '&') => (display.lowerOdors ? `${sep}lower=1` : '');

// what is shown already, shown again as Settings now say odors are
function displayChanged() {
  LS.set(DISPLAY_KEY, display);
  browse.viewer?._rebuild();
  if (browse.selected != null && !$('#ds-odors').hidden) browse.showOdors(browse.selected);
  reference.viewer?._rebuild();
}

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
// A warning of a reference dataset's chemicals not identified (no CID or InChI), naming
// the first `n` of them (`labels`, as in its file); when there are more, all of them can
// be shown (listed once asked for) and copied, one per line. after: what to do, after it
function unidentifiedBox(warning, labels, after, n = 20) {
  const box = msgBox('warn', `<b>${esc(warning)}</b>${labels.length ? `: ${labels.slice(0, n).map(esc).join(', ')}` : ''}${labels.length > n ? ', …' : '.'} ${after}`);
  if (labels.length <= n) return box;
  const all = document.createElement('details');
  all.className = 'unidentified';
  all.innerHTML = `<summary>Show all ${labels.length.toLocaleString()}</summary>
    <div class="actions"><button type="button" class="btn small">Copy them, one per line</button><span class="status"></span></div><ol></ol>`;
  all.addEventListener('toggle', () => {
    const ol = $('ol', all);
    if (all.open && !ol.childElementCount) ol.innerHTML = labels.map((l) => `<li>${l ? esc(l) : '<span class="muted">(no label)</span>'}</li>`).join('');
  });
  $('button', all).addEventListener('click', async (e) => {
    const status = e.target.nextElementSibling;
    try { await copyText(labels.join('\n')); setStatus(status, `✓ Copied ${labels.length.toLocaleString()}`); } catch (err) {
      setStatus(status, `Could not copy them: ${err.message}`, 'error');
    }
  });
  box.appendChild(all);
  return box;
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
      for (const j of ts) { cols.push({ label: data.showOdor(c.odor), group, sub: trials[j].repeat, key: c.key }); colTrials.push([j]); }
    } else {
      cols.push({ label: data.showOdor(c.odor), group, sub: null, key: c.key });
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

// A chemical of a reference dataset labelled by its name or abbreviation in the odor table
// (as matched there; the abbreviation as `showAbbrev` shows it), or as in the file:
// {label, detail (the rest, for the tooltip)}
function chemLabel(it, by, showAbbrev = (a) => a) {
  const c = it.chem, abbrev = showAbbrev(c.abbrev);
  const label = (by === 'abbrev' && (abbrev || c.name)) || (by === 'name' && c.name) || it.file;
  const others = [c.name, abbrev].filter((n) => n && n !== label);
  if (it.file.toLowerCase() !== label.toLowerCase() && !others.some((n) => n.toLowerCase() === it.file.toLowerCase())) others.push(`“${it.file}” in the file`);
  if (c.cid) others.push(`CID ${c.cid}`);
  if (!c.name) others.push('not in the Odors table');
  return { label, detail: others.join(' · ') };
}

// rows as columns and columns as rows, with their labels, titles and dendrograms
function transposeModel(m) {
  const nR = m.rows.length, nC = m.cols.length;
  const values = new Float32Array(nR * nC);
  for (let r = 0; r < nR; r++) for (let c = 0; c < nC; c++) values[c * nR + r] = m.values[r * nC + c];
  return {
    ...m, rows: m.cols, cols: m.rows, values, rowDendro: m.colDendro, colDendro: m.rowDendro, transposed: true,
    cellText: m.cellText && ((r, c) => m.cellText(c, r)),
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
  // cells: [width, height] of this viewer's cells to start with, not those of every viewer;
  // asUploaded: odors as named in the data, whatever Settings say (the upload preview)
  constructor(container, { kind = 'neural', cells = null, asUploaded = false } = {}) {
    this.container = container;
    this.kind = kind;
    this.asUploaded = asUploaded;
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
      onHover: (info) => this._linkHover(info),
    });
    this.heatmap.scroll.addEventListener('scroll', () => this._linkScroll(), { passive: true });
    this.colorbar = $('.colorbar', container);
    const saved = LS.get('olfdb.viewer', {});
    this.opts = Object.assign(
      {
        mode: 'average', rowOrder: 'fly', roiSort: 'file', odorOrder: 'file', clusterRows: false, clusterCols: false, metric: 'correlation',
        cellW: 16, cellH: 16, colormap: 'oklab-div', range: 'twoslope', clip: false, chemLabels: 'name',
      },
      saved,
      { transpose: !!saved.transposed?.[kind] },
      cells ? { cellW: cells[0], cellH: cells[1] } : {},
    );
    this.ownCells = !!cells;
    this.tableExport = null; // () => {text, name}: the CSV the export menu offers, if any
    this.data = null;
    this.source = null; // the payload shown
    this.edits = newEdits();
    this.editsOf = new WeakMap(); // by payload, for when it is shown again
    this.title = '';
    this.fileStem = 'heatmap';
    this.clusterPool = new ClusterPool();
    this.clusterCache = new Map();
    this._seq = 0;
    this.onShow = null; // called once the matrix is drawn (as shown, or with cells of another size)
    // {other, cols}: a viewer of the same rows in the same order (and with cols, the same
    // columns), shown beside this one: rows moved, removed or reset, cell sizes, transposing,
    // chemical labels and scrolling are kept the same in both, and what is hovered is marked
    this.linked = null;
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
      if (s.dataset.opt === 'chemLabels') this.linked?.other._setLinkedOpts({ chemLabels: s.value });
    }));
    for (const k of ['cellW', 'cellH']) {
      const range = $(`input[data-opt=${k}]`, tb);
      range.addEventListener('input', () => {
        this.opts[k] = +range.value;
        $(`[data-out=${k}]`, tb).textContent = range.value;
        this._save();
        this.heatmap.setOptions({ cellW: this.opts.cellW, cellH: this.opts.cellH });
        this.onShow?.();
        this.linked?.other._setLinkedOpts({ [k]: this.opts[k] });
      });
    }
    $$('input[type=checkbox][data-opt]', tb).forEach((cb) => cb.addEventListener('change', () => {
      this.opts[cb.dataset.opt] = cb.checked;
      this._save();
      this._syncEnabled();
      this._rebuild();
      if (cb.dataset.opt === 'transpose') this.linked?.other._setLinkedOpts({ transpose: cb.checked });
    }));
    $('[data-role=reset]', tb).addEventListener('click', () => { Object.assign(this.edits, newEdits()); this._rebuild(); this._linkEdits(); });
    wireMenu($('[data-role=export]', tb), ({ export: format }) => this.exportFigure(format));
  }

  _save() {
    const o = this.opts, saved = LS.get('olfdb.viewer', {});
    LS.set('olfdb.viewer', {
      cellW: this.ownCells ? saved.cellW : o.cellW, cellH: this.ownCells ? saved.cellH : o.cellH, colormap: o.colormap, range: o.range, rowOrder: o.rowOrder, roiSort: o.roiSort, odorOrder: o.odorOrder,
      clusterRows: o.clusterRows, clusterCols: o.clusterCols, metric: o.metric, chemLabels: o.chemLabels,
      transposed: { ...saved.transposed, [this.kind]: o.transpose },
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
    $('[data-role=chem]', tb).hidden = neural || !this.data?.chemAxis;
    const mapKind = neural ? 'diverging' : (this.data?.meta?.kind || 'sequential');
    const cmSel = $('select[data-opt=colormap]', tb);
    const maps = colormapsOfKind(mapKind);
    cmSel.innerHTML = maps.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join('');
    if (!maps.some((m) => m.id === o.colormap)) o.colormap = maps[0].id;
    cmSel.value = o.colormap;
    $('[data-role=range]', tb).hidden = mapKind !== 'diverging';
    $('input[data-opt=clip]', tb).parentElement.hidden = mapKind === 'binary';
    $('input[data-opt=transpose]', tb).parentElement.title = neural ? 'Odors as rows and ROIs as columns' : 'Rows as columns and columns as rows';
    for (const k of ['rowOrder', 'roiSort', 'odorOrder', 'metric', 'range', 'chemLabels']) { const s = $(`select[data-opt=${k}]`, tb); if (s) s.value = o[k]; }
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
      this.data = { units: payload.units, trials: payload.trials, values: decodeValues(payload.values_b64),
        showOdor: this.asUploaded ? (s) => s : odorShower(payload.trials.map((t) => t.odor)) };
      this._editsFor(payload);
      this.clusterCache.clear();
    }
    this.title = title;
    this.fileStem = fileStem;
    this.opts.colormap = this.opts.colormap in COLORMAPS && COLORMAPS[this.opts.colormap].kind === 'diverging' ? this.opts.colormap : 'oklab-div';
    this._syncToolbar();
    this._rebuild();
  }

  // ref: a reference dataset (as from /api/reference/<id>), or a matrix like one with its
  // `values` (Float32Array) given; kind: of the viewer, for its transposing kept apart
  setGenericData(ref, { title = '', fileStem = 'heatmap', kind: viewerKind = 'generic' } = {}) {
    this._setKind(viewerKind);
    this.title = title;
    this.fileStem = fileStem;
    if (ref !== this.source) {
      this.source = ref;
      this._editsFor(ref);
      const values = ref.values || decodeValues(ref.values_b64);
      const hasGroups = ref.rows.some((r) => r.group != null);
      let rows = ref.rows.map((r, i) => ({ label: r.label, group: hasGroups ? (r.group ?? '') : null, chem: r.chem, detail: r.detail, i }));
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
        chemAxis: ref.chemicals?.matched ? ref.chemicals.axis : null,
        model: {
          rows: rows.map((r) => ({ label: r.label, group: r.group, key: r.i, file: r.label, chem: r.chem, detail: r.detail })),
          cols: ref.cols.map((c, j) => ({ label: c.label, group: null, sub: null, key: j, file: c.label, chem: c.chem, detail: c.detail })),
          values: vals,
          cellText: ref.cellText, // (row, col) of the matrix as given: text to write in that cell
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

  // the matrix less the rows and columns removed, in the order they were dragged into,
  // its chemicals labelled as chosen
  _genericModel() {
    const b = this.data.model, nC = b.cols.length;
    const rows = arrange(b.rows.map((_, i) => i), (i) => b.rows[i].key, this.edits.rows);
    const cols = arrange(b.cols.map((_, j) => j), (j) => b.cols[j].key, this.edits.cols);
    const values = new Float32Array(rows.length * cols.length);
    rows.forEach((i, r) => { cols.forEach((j, c) => { values[r * cols.length + c] = b.values[i * nC + j]; }); });
    const showAbbrev = abbrevShower([...b.rows, ...b.cols].map((it) => it.chem?.abbrev));
    const label = (it) => (it.chem ? { ...it, ...chemLabel(it, this.opts.chemLabels, showAbbrev) } : it);
    return { ...b, rows: rows.map((i) => label(b.rows[i])), cols: cols.map((j) => label(b.cols[j])), values,
      cellText: b.cellText && ((r, c) => b.cellText(rows[r], cols[c])) };
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
    const fixed = !neural && this.data.meta.value_range; // [vmin, vmax] whatever the values
    const { vmin, vmax } = fixed ? { vmin: fixed[0], vmax: fixed[1] } : dataRange(model.values, o.clip);
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
    this.onShow?.();
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
    this._linkEdits();
  }

  // how many items of the heatmap's axis are shown (the trials of an odor being one)
  _nShown(axis) { return new Set(this.model[axis].map((it) => it.key)).size; }

  // removes keys of the heatmap's axis, unless none would be left (ctrl-click on a label)
  _remove(axis, keys) {
    if (keys.length >= this._nShown(axis)) return;
    const { removed } = this.edits[this._editAxis(axis)];
    keys.forEach((k) => removed.add(k));
    this._rebuild();
    this._linkEdits();
  }

  // ------------------------------------------------------- linked viewers
  // this viewer's edits of its rows (and columns, if they are the same too) made the other's
  _linkEdits() {
    const l = this.linked;
    if (!l) return;
    for (const ax of l.cols ? ['rows', 'cols'] : ['rows']) {
      const e = this.edits[ax];
      l.other.edits[ax] = { removed: new Set(e.removed), order: e.order && [...e.order] };
    }
    l.other._rebuild();
  }

  // options as the linked viewer's were just set: cell size, transposing, chemical labels
  _setLinkedOpts(patch) {
    Object.assign(this.opts, patch);
    this._syncToolbar();
    if ('transpose' in patch || 'chemLabels' in patch) this._rebuild();
    else { this.heatmap.setOptions({ cellW: this.opts.cellW, cellH: this.opts.cellH }); this.onShow?.(); }
  }

  // scrolled along the shared axis (both, when rows and columns are shared), so is the other
  _linkScroll() {
    const l = this.linked;
    if (!l) return;
    const from = this.heatmap.scroll, to = l.other.heatmap.scroll;
    if (l.cols || !this.opts.transpose) to.scrollTop = from.scrollTop;
    if (l.cols || this.opts.transpose) to.scrollLeft = from.scrollLeft;
  }

  // what is hovered here marked there: the row (and with shared columns, the column)
  _linkHover(info) {
    const l = this.linked;
    if (!l) return;
    const at = (i) => (i >= 0 ? i : null);
    let mark = null;
    if (info && l.cols) mark = { rows: at(info.rowIndex), cols: at(info.colIndex) };
    else if (info) mark = this.opts.transpose ? { rows: null, cols: at(info.colIndex) } : { rows: at(info.rowIndex), cols: null };
    l.other.heatmap.setMark(mark);
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

  // the matrix as shown, as a CSV table: its row labels (and groups) then a column each
  tableCsv() {
    const m = this.model, groups = m.rows.some((r) => r.group != null);
    const head = [...(groups ? [m.rowGroupTitle || 'group'] : []), m.rowTitle || 'row', ...m.cols.map((c) => c.label)];
    const lines = [head, ...m.rows.map((r, i) => [...(groups ? [r.group ?? ''] : []), r.label,
      ...Array.from(m.values.subarray(i * m.cols.length, (i + 1) * m.cols.length), (v) => (v === v ? String(v) : ''))])];
    return lines.map((cells) => cells.map(csvCell).join(',')).join('\n') + '\n';
  }

  async exportFigure(format) {
    if (!this.model) return;
    const status = $('[data-role=export-status]', this.toolbar);
    if (format === 'csv') {
      const t = this.tableExport?.() ?? { text: this.tableCsv(), name: `${this.fileStem}.csv` };
      saveBlob(new Blob([t.text], { type: 'text/csv' }), t.name);
      setStatus(status, '');
      return;
    }
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

// a cell of a CSV table, quoted if need be
const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

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

// An attached odor YAML's own check (`_yaml_files` on the server), shown under it: its odors
// against the odor table, and what looks wrong in the file. A warning only, unlike the check
// of the matrix's odors against the YAMLs, which saving waits for.
function yamlFileCheck(f) {
  const n = f.odors.length;
  const odd = f.table_empty ? [] : f.odors.filter((o) => o.problem);
  const notIn = odd.filter((o) => o.odor_id == null).length;
  const items = [...f.problems.map(esc), ...odd.map((o) => `<b>${esc(o.name)}</b>${o.abbrev ? ` (${esc(o.abbrev)})` : ''}: ${esc(o.problem)}`)];
  const table = '<a href="#odors">odor table</a>';
  const empty = f.table_empty ? `<div class="yaml-file-check muted">The ${table} is empty: its odors were not checked against it.</div>` : '';
  if (!items.length) return empty || `<div class="yaml-file-check ok">✓ All ${n} odors are in the ${table}, named and abbreviated as there.</div>`;
  const head = [notIn ? `${notIn} of ${n} odors not in the ${table}` : '', odd.length - notIn ? `${odd.length - notIn} unlike the ${table}` : '',
    f.problems.length ? `${f.problems.length} problem${f.problems.length === 1 ? '' : 's'} in the file` : ''].filter(Boolean).join('; ');
  return `<details class="yaml-file-check warn"${items.length <= 5 ? ' open' : ''}><summary>⚠ ${head}</summary>
    <ul>${items.map((x) => `<li>${x}</li>`).join('')}</ul></details>${empty}`;
}

// the upload under way, kept in this browser to be continued (see UploadFlow.resume)
const UPLOAD_KEY = 'olfdb.upload';

// The ROI names of a data file not accepted (`rois` of a rejected upload: each with the
// accepted glomerulus name it is but for case), and the glomerulus names accepted.
function roiNamesBox({ not_accepted: bad, accepted }) {
  const items = bad.map((b) => `<code>${esc(b.roi)}</code>${b.same_but_case ? ` <span class="muted">(write <code>${esc(b.same_but_case)}</code>)</span>` : ''}`);
  return msgBox('info', `<b>${bad.length} ROI name${bad.length === 1 ? ' is' : 's are'} not accepted:</b> ${items.join(', ')}.
    <p>Each ROI must be one of the ${accepted.length} glomerulus names accepted (as written: upper and lower case matter), or a
    non-negative integer ROI id (<code>0</code>, <code>1</code>, …):</p>
    <p class="roi-names">${accepted.map((g) => `<code>${esc(g)}</code>`).join(' ')}</p>
    <p class="small">The list is <code>config/glomeruli.txt</code> on the server, which can be edited to add names.</p>`);
}

class UploadFlow {
  constructor() {
    this.options = null;
    this.stage = null;
    this.data = null;
    this.fileName = '';
    this.generatedReadme = '';
    this.viewer = null;
    this.yamlCheck = null;
    this.odorCheck = null; // its odors against the odor table (GET /api/stage/<id>/odors)
    this.syncNote = ''; // what syncing the odor sheet from here last did
    this._odorSeq = 0;
    this._saveTimer = 0;
    this.nas = null; // what the last lookup found on tensor
    this.user = null; // the uploader's folders on tensor, as saved in Settings (from the last lookup)
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
    await this.resume();
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
    // the example file saved from GitHub (which allows that from any site), rather than shown
    // there: browsers ignore `download` on a link to another site
    const example = $('.example-file a');
    example.addEventListener('click', async (e) => {
      e.preventDefault();
      const out = example.nextElementSibling;
      setStatus(out, 'Downloading…');
      try {
        const res = await fetch(example.href);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        saveBlob(new Blob([await res.blob()], { type: 'text/csv' }), example.textContent.trim());
        setStatus(out, '');
      } catch (err) {
        setStatus(out, `Could not download it (${err.message}): opened instead`, 'error');
        window.open(example.href, '_blank', 'noopener');
      }
    });

    const form = $('#step-meta');
    $('#power-mw').addEventListener('input', () => this._syncPower());
    // capturing, to know a field was changed by the user before its own listeners run
    form.addEventListener('input', (e) => this._changedByUser(e.target), true);
    form.addEventListener('change', (e) => this._changedByUser(e.target), true);
    form.uploader_name.addEventListener('change', () => { if (this.stage) this.lookup(); });
    // checked on the server when left, as folders are in the Settings tab
    for (const key of DATASET_FIELDS) {
      form[key].addEventListener('change', async () => {
        await checkPathInput(form[key], $(`[data-msg-for=${key}]`));
        if (key === 'raw_data_path' && this.stage) this.lookup(); // its flies are looked for there
      });
    }
    form.addEventListener('submit', (e) => { e.preventDefault(); this.generateReadme(); });
    form.addEventListener('input', () => this._saveSoon());
    form.addEventListener('change', () => this._saveSoon());
    $('#odor-table-check').addEventListener('click', async (e) => {
      const sync = e.target.closest('[data-sync-sheet]');
      if (sync) { this.syncSheet(sync); return; }
      const b = e.target.closest('[data-copy-sheet]');
      if (!b) return;
      try { await copyText(this.odorCheck.sheet.paste); setStatus(b.nextElementSibling, '✓ Copied'); } catch (err) {
        setStatus(b.nextElementSibling, `Could not copy: ${err.message}`, 'error');
      }
    });

    $('#yaml-btn').addEventListener('click', () => $('#yaml-input').click());
    $('#yaml-input').addEventListener('change', (e) => { if (e.target.files.length) this.addYaml([...e.target.files]); e.target.value = ''; });
    $('#xml-btn').addEventListener('click', () => $('#xml-input').click());
    $('#xml-input').addEventListener('change', (e) => { if (e.target.files[0]) this.addXml(e.target.files[0]); e.target.value = ''; });

    $('#readme-text').addEventListener('input', () => { $('#readme-preview').innerHTML = renderMarkdown($('#readme-text').value); this._saveSoon(); });
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
  // it: the uploader's data folders in Settings (for the data paths), then found on
  // tensor, then read from the dataset's files, then the defaults in Settings (a guess from
  // the files comes last). A field filled in before that has nothing now goes back to how
  // it started.
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
    for (const [key, kind, flies, what] of [['raw_data_path', 'raw_dirs', 'raw', 'raw data'], ['mocorr_data_path', 'mocorr_dirs', 'mocorr', 'motion-corrected data']]) {
      const dirs = this.user?.[kind] || [];
      if (!dirs.length) continue;
      // of several, the one with the most of these flies (as found on tensor), else the first
      const inside = (d) => (this.nas?.flies || []).filter((f) => f[flies].some((p) => p === d || p.startsWith(`${d}/`))).length;
      const best = dirs.reduce((a, d) => (inside(d) > inside(a) ? d : a), dirs[0]);
      offer(key, best, 'settings', 5, dirs.length === 1 ? `your ${what} folder`
        : inside(best) ? `the one of your ${what} folders with these flies` : `the first of your ${what} folders`);
    }
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

  // the uploader's folders were saved in Settings
  foldersChanged() {
    if (this.stage && $('#step-meta').uploader_name.value.trim().toLowerCase() === settings.name.toLowerCase()) this.lookup();
  }

  // the name or the defaults in Settings changed
  settingsChanged() {
    if (!this.options) return;
    const name = $('#step-meta').uploader_name.value;
    this._autofill();
    if (this.stage && $('#step-meta').uploader_name.value !== name) this.lookup();
  }

  // Looks for the dataset's flies on tensor: in the raw data path as entered, else in the
  // uploader's raw data folders (from Settings), and for odor YAMLs in their odor YAML
  // folders. Attaches the most recent Experiment.xml found (unless one was added or chosen
  // by hand), and the odor YAMLs describing every odor (unless some are attached already).
  async lookup() {
    const seq = ++this._lookupSeq;
    const box = $('#nas-lookup'), form = $('#step-meta');
    const name = form.uploader_name.value.trim();
    // as typed in, not as filled in from an earlier lookup
    const raw = this.manual.has('raw_data_path') ? form.raw_data_path.value.trim() : '';
    const stageId = this.stage.stage_id;
    const done = (html, kind = 'info') => {
      this.nas = null; this.looking = false;
      box.replaceChildren(msgBox(kind, html)); this._autofill(); this._renderYamlStatus();
    };
    if (!name && !raw) {
      this.user = null;
      done('Set your name and data folders in <a href="#settings">Settings</a>, or enter the raw data path below, to have the files of this dataset found on tensor.');
      return;
    }
    this.looking = true;
    this._renderYamlStatus();
    box.replaceChildren(msgBox('info', raw ? `Looking for these flies in <code>${esc(raw)}</code> on tensor…` : `Looking for these flies in the folders of ${esc(name)} on tensor…`));
    let res;
    try {
      res = await api(`/api/stage/${stageId}/lookup`, { method: 'POST', json: { name, raw_data_path: raw } });
    } catch (e) {
      if (seq === this._lookupSeq) done(`Could not look on tensor: ${esc(e.message)}`, 'error');
      return;
    }
    if (seq !== this._lookupSeq || this.stage?.stage_id !== stageId) return;
    this.user = res.user || null;
    if (res.raw_error) { done(`Could not look in the raw data path: ${esc(res.raw_error)}`, 'error'); return; }
    if (!res.user && !raw) {
      done(`No data folders are saved for <b>${esc(name)}</b>. Add them in <a href="#settings">Settings</a>, or enter the raw data path below, to have the raw data, Experiment.xml and odor YAML of your uploads found on tensor.`);
      return;
    }
    this.nas = res;
    const xml = res.experiment_xmls.at(-1); // the most recent recording
    if (xml && (!this.stage.experiment_xml || (this.xmlAuto && this.stage.experiment_xml_tensor_path !== xml))) await this.addXml({ tensorPath: xml }, { auto: true });
    if (res.odor_yamls.selected.length && !this.stage?.odor_yamls.length) await this.addYaml({ tensorPaths: res.odor_yamls.selected });
    if (seq !== this._lookupSeq || this.stage?.stage_id !== stageId) return;
    this.looking = false;
    this._renderLookup();
    this._autofill();
    this._renderYamlStatus();
  }

  // Next to the YAML button: the YAMLs attached and whether they describe every odor, or
  // why none is (found on tensor).
  _renderYamlStatus() {
    const status = $('#yaml-status'), yamls = this.stage?.odor_yamls || [], c = this.yamlCheck;
    status.className = 'xml-status';
    if (!this.stage) { status.textContent = ''; return; }
    if (yamls.length) {
      status.classList.add(c?.ok ? 'ok' : 'bad');
      status.innerHTML = `${c?.ok ? '✓' : '✗'} ${yamls.map((y) => `<code>${esc(y.name)}</code>`).join(', ')} attached`
        + (c ? (c.ok ? `: all ${c.n_matrix_odors} odors described` : `: ${c.missing.length} of ${c.n_matrix_odors} odors not described (below)`) : '');
      return;
    }
    if (this.looking) { status.classList.add('muted'); status.textContent = 'None attached yet: looking on tensor…'; return; }
    const n = this.nas, y = n?.odor_yamls;
    let why;
    if (!n?.user?.odor_dirs.length) why = 'no odor YAML folders in your <a href="#settings">Settings</a> to find one in on tensor; add one here';
    else if (y.candidates.length) {
      why = `the ${y.candidates.length} found on tensor do not describe ${esc(y.missing.slice(0, 4).join(', '))}${y.missing.length > 4 ? ', …' : ''}`
        + ' (they are listed above, to attach); add one here';
    } else why = 'none in your odor YAML folders on tensor describes these odors; add one here';
    status.classList.add('warn');
    status.innerHTML = `⚠ No odor YAML attached: ${why}.`;
  }

  _renderLookup() {
    const n = this.nas, st = this.stage, box = $('#nas-lookup');
    const flyName = (f) => `${f.date} fly ${f.fly_num}`;
    const plural = (k, word) => `${k} ${word}${k === 1 ? '' : 's'}`;
    const dirsOf = (k) => (k === 'raw_dirs' && n.raw_data_path_given ? [n.raw_data_path_given] : n.user?.[k] || []);
    const flies = (key, folders, path) => {
      if (!dirsOf(folders).length) return '<span class="muted">no such folders in your settings</span>';
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
    if (!dirsOf('odor_dirs').length) yaml = '<span class="warn">not looked for: no odor YAML folders in your settings</span>';
    else if (y.selected.length) {
      yaml = `${plural(y.selected.length, 'file')} describing all ${y.n_matrix_odors} odors`;
      yaml += y.selected.every((p) => attached.has(p)) ? ', attached' : ' <span class="muted">(not attached, as other YAML was attached already)</span>';
    } else if (y.candidates.length) {
      yaml = `${plural(y.candidates.length, 'file')} describing some of the odors, but none for ${esc(y.missing.slice(0, 6).join(', '))}${y.missing.length > 6 ? ', …' : ''}; attach any of them below`;
    } else yaml = '<span class="muted">none describing these odors</span>';
    // what was found for each panel (YAMLs are looked for panel by panel)
    const file = (path) => `<code title="${esc(path)}">${esc(path.split('/').pop())}</code>`;
    if (dirsOf('odor_dirs').length && y.panels?.length && (y.panels.length > 1 || y.panels[0].panel != null)) {
      yaml += `<ul>${y.panels.map((p) => `<li><b>${esc(p.panel ?? 'its odors')}</b> (${p.n_odors} odors): ${p.complete
        ? `${p.selected.map(file).join(', ')} <span class="muted">(${esc(p.how)})</span>`
        : `<span class="warn">${p.missing.length} described by no YAML found</span> (${esc(p.missing.slice(0, 4).join(', '))}${p.missing.length > 4 ? ', …' : ''})${
          p.selected.length ? `; the rest by ${p.selected.map(file).join(', ')}` : ''}`}${p.remarks?.length
        ? `<ul class="muted small">${p.remarks.slice(0, 5).map((r) => `<li>${esc(r)}</li>`).join('')}${p.remarks.length > 5 ? `<li>and ${p.remarks.length - 5} more like these</li>` : ''}</ul>`
        : ''}</li>`).join('')}</ul>`;
    }
    const flyRows = n.flies.map((f) => `<tr><td>${esc(flyName(f))}</td><td>${f.raw.map((p) => `<code>${esc(p)}</code>`).join('<br>') || '—'}</td>
      <td>${f.mocorr.map((p) => `<code>${esc(p)}</code>`).join('<br>') || '—'}</td><td>${f.recordings.length}</td></tr>`).join('');
    const yamlRows = y.candidates.map((c) => `<tr><td><code>${esc(c.path)}</code></td><td class="muted">${esc(c.reason)}</td>
      <td>${c.n_covered} of ${y.n_matrix_odors} odors</td><td>${attached.has(c.path) ? '<span class="muted">attached</span>'
        : `<button type="button" class="btn small" data-attach-yaml="${esc(c.path)}">attach</button>`}</td></tr>`).join('');
    box.innerHTML = `<div class="msg info lookup">
      <div class="lookup-head"><b>Found on tensor</b><span>${n.raw_data_path_given ? `in <code>${esc(n.raw_data_path_given)}</code> (the raw data path)${n.user ? `, and the folders of ${esc(n.user.name)}` : ''}`
        : `in the folders of ${esc(n.user.name)}`} <span class="muted">(<a href="#settings">settings</a>)</span></span>
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
    $('[data-xml]', box)?.addEventListener('change', (e) => { if (e.target.value) this.addXml({ tensorPath: e.target.value }); }); // chosen
    $$('[data-attach-yaml]', box).forEach((b) => b.addEventListener('click', () => this.addYaml({ tensorPaths: [b.dataset.attachYaml] })));
  }

  // ------------------------------------------------------------------- staging
  reset() {
    this.stage = null; this.data = null; this.yamlCheck = null; this.generatedReadme = ''; this.fileName = '';
    this.nas = null; this.user = null; this._lookupSeq++; this.looking = false; this.xmlAuto = false;
    this.odorCheck = null; this.syncNote = ''; this._odorSeq++;
    clearTimeout(this._saveTimer);
    LS.set(UPLOAD_KEY, null);
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
    $('#yaml-list').innerHTML = ''; $('#yaml-check').innerHTML = ''; $('#xml-info').innerHTML = ''; this._renderXml(); this._renderYamlStatus();
    $('#odor-table-check').innerHTML = '';
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
      if (res.rois) result.appendChild(roiNamesBox(res.rois));
      if (res.warnings?.length) result.appendChild(msgBox('warn', `Also:${listHtml(res.warnings)}`));
      return;
    }
    this.stage = res.stage;
    this.data = res.data;
    this.fileName = file.name;
    this._showStaged();
    this._autofill();
    this.lookup();
    this.checkOdors();
    this._save();
    $('#step-meta').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // the data file staged: its summary and preview, then the form
  _showStaged() {
    const result = $('#file-result'), name = this.fileName;
    const s = this.data.summary;
    const warn = this.data.warnings || [];
    result.innerHTML = '';
    const okBox = msgBox('ok', `<b>${esc(name)}</b> passed validation (${esc(s.source_format)}).`);
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
    if (!this.viewer) this.viewer = new MatrixViewer($('#preview-viewer'), { asUploaded: true });
    this.viewer.setNeuralData(this.data, { title: name, fileStem: name.replace(/\.[^.]*$/, '') || 'preview' });
    $('#step-meta').hidden = false;
  }

  // ------------------------------------------------------- continuing it later
  _saveSoon() {
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => this._save(), 300);
  }

  // the upload under way (its stage on the server, the form as filled in) kept in this
  // browser, to be continued when the tab is opened again (`resume`)
  _save() {
    if (!this.stage) return;
    const fields = {};
    for (const el of $('#step-meta').elements) {
      if (el.name && el.type !== 'file' && (el.type !== 'radio' || el.checked)) fields[el.name] = el.value;
    }
    const readme = $('#step-readme').hidden ? null : { text: $('#readme-text').value, generated: this.generatedReadme };
    LS.set(UPLOAD_KEY, { stage_id: this.stage.stage_id, file: this.fileName, saved_at: Date.now() / 1000, fields,
      manual: [...this.manual], xml_auto: this.xmlAuto, readme });
  }

  // Continues the upload kept in this browser (`_save`), while the server still has it.
  async resume() {
    const saved = LS.get(UPLOAD_KEY, null);
    if (!saved?.stage_id || this.stage) return;
    let res;
    try { res = await api(`/api/stage/${encodeURIComponent(saved.stage_id)}`); } catch (e) {
      if (!e.message.startsWith('cannot reach')) LS.set(UPLOAD_KEY, null);
      $('#file-result').replaceChildren(msgBox('warn', `Your upload of <b>${esc(saved.file)}</b> could not be continued: ${esc(e.message)}.`));
      return;
    }
    if (this.stage) return; // a file was dropped meanwhile
    this.stage = res.stage;
    this.data = res.data;
    this.fileName = saved.file;
    this._showStaged();
    const form = $('#step-meta');
    for (const el of form.elements) {
      if (!el.name || !(el.name in saved.fields)) continue;
      if (el.type === 'radio') el.checked = el.value === saved.fields[el.name]; else el.value = saved.fields[el.name];
    }
    for (const [sel, other] of Object.values(COMBOS)) $(other).hidden = $(sel).value !== '__other__';
    this.manual = new Set(saved.manual);
    this.xmlAuto = !!saved.xml_auto;
    this._syncPower();
    for (const key of DATASET_FIELDS) if (form[key].value.trim()) checkPathInput(form[key], $(`[data-msg-for=${key}]`));
    this._renderYaml(res);
    this._renderXml();
    if (saved.readme) {
      this.generatedReadme = saved.readme.generated;
      $('#readme-text').value = saved.readme.text;
      $('#readme-preview').innerHTML = renderMarkdown(saved.readme.text);
      $('#step-readme').hidden = false;
    }
    const note = msgBox('info', `Continuing your upload of <b>${esc(saved.file)}</b>, as you left it ${esc(fmtEpoch(saved.saved_at))}.
      <button type="button" class="btn small" data-start-over>Start over</button>`);
    $('[data-start-over]', note).addEventListener('click', () => this.reset());
    $('#file-result').prepend(note);
    this._autofill();
    this.lookup();
  }

  // ------------------------------------------------- its odors in the odor table
  async checkOdors() {
    const stageId = this.stage?.stage_id, seq = ++this._odorSeq;
    if (!stageId) return;
    if (!this.odorCheck) this._renderOdorCheck();
    let res;
    try { res = await api(`/api/stage/${stageId}/odors`); } catch (e) {
      if (seq === this._odorSeq) $('#odor-table-check').replaceChildren(msgBox('error', `Could not check its odors against the odor table: ${esc(e.message)}`));
      return;
    }
    if (seq !== this._odorSeq || this.stage?.stage_id !== stageId) return;
    this.odorCheck = res;
    this._renderOdorCheck();
  }

  // the odors of the data not one odor in the odor table (as in the data)
  _unmatchedOdors() { return [...new Set((this.odorCheck?.rows || []).filter((r) => r.odor_id == null).map((r) => r.odor))]; }

  // Under the odor YAMLs: whether each of the dataset's odors is one in the odor table
  // (matched as in Browse, through the YAMLs attached), which saving waits for; if not,
  // their rows for the odor sheet to copy, and a button syncing the odor table from it.
  _renderOdorCheck() {
    const box = $('#odor-table-check'), c = this.odorCheck;
    if (!this.stage) { box.replaceChildren(); return; }
    if (!c) { box.innerHTML = '<p class="muted small">Checking its odors against the odor table…</p>'; return; }
    const table = '<a href="#odors">odor table</a>';
    const n = new Set(c.rows.map((r) => r.odor)).size, k = this._unmatchedOdors().length;
    const note = this.syncNote ? ` <span class="muted">(${esc(this.syncNote)})</span>` : '';
    const upper = c.case?.uppercase || [], codes = (names) => names.map((x) => `<code>${esc(x)}</code>`).join(', ');
    const caps = upper.length ? `<p><b>${upper.length} of its odor names ${upper.length === 1 ? 'has' : 'have'} capital letters</b> (${codes(upper)}),
      and new odor names must be lowercase (set in the Admin tab): rename ${upper.length === 1 ? 'it' : 'them'} in the data file, and add it again.</p>` : '';
    const clashes = (c.case?.clashes || []).length ? `<p class="small">${c.case.clashes.map((g) => codes(g)).join('; ')} differ only in case: where
      odor names are shown in lowercase (Settings), they are shown as they are, so as not to look the same.</p>` : '';
    if (!k) {
      box.innerHTML = upper.length ? `<div class="msg error">${caps}${clashes}</div>`
        : `<div class="yaml-file-check ok">✓ Each of its ${n} odors is in the ${table}.${note}</div>${clashes ? `<div class="msg info">${clashes}</div>` : ''}`;
      return;
    }
    const parts = [`<p><b>${k} of its ${n} odors ${k === 1 ? 'is' : 'are'} not in the ${table}</b>: the dataset can be saved once ${k === 1 ? 'it is' : 'they are'}.</p>`, caps, clashes];
    if (c.missing.length) parts.push(`<p>Not in it:</p>${missingRowsHtml(c.missing, c.sheet, 'sync it here (below)')}`);
    const ambiguous = ambiguousOdors(c.rows);
    if (ambiguous.length) {
      parts.push(`<p>More than one odor in it goes by ${odorLabels(ambiguous)}: attach an odor YAML naming the dataset's (above), or give
        them abbreviations of their own in the sheet and sync it here.</p>`);
    }
    const hours = this.options?.stage_ttl_hours ?? 48;
    parts.push(`<div class="actions"><button type="button" class="btn" data-sync-sheet>Sync from the odor sheet</button><span class="status">${esc(this.syncNote)}</span></div>
      <p class="small">Adds the odors new in the sheet to the odor table, then checks these again (changes to odors already in it are
      left for the <a href="#odors">Odors tab</a>). You can leave this page meanwhile: this upload is kept for ${hours} hours after you
      last worked on it, and continues where you left it when you open the Upload tab again in this browser.</p>`);
    box.innerHTML = `<div class="msg error">${parts.join('')}</div>`;
  }

  // Syncs the odor table from the odor sheet as far as needs no one to confirm it: the
  // odors new in the sheet (changes and deletions wait for the Odors tab).
  async syncSheet(btn) {
    const out = btn.nextElementSibling;
    btn.disabled = true;
    setStatus(out, 'Reading the odor sheet…');
    try {
      const p = await api('/api/odors/preview', { method: 'POST', json: {} });
      odorsPage.loadStatus();
      if (p.errors.length) {
        setStatus(out, `Not synced: the sheet has ${p.errors.length} error${p.errors.length === 1 ? '' : 's'} to fix first (${p.errors.slice(0, 2).join('; ')}${p.errors.length > 2 ? '; …' : ''})`, 'error');
        return;
      }
      let added = 0;
      if (p.added.length) {
        setStatus(out, `Adding ${p.added.length} new odor${p.added.length === 1 ? '' : 's'} to the odor table…`);
        const r = await api('/api/odors/apply', { method: 'POST', json: { fingerprint: p.fingerprint, changes: [], deletions: [] } });
        if (!r.ok) { setStatus(out, `Not synced: ${r.errors.join('; ')}`, 'error'); return; }
        added = r.added;
        browse.odorCache.clear();
        comparePage.invalidate(); // its odors' matches to the table
        reference.invalidate();
        odorsPage.odors = null; // read again when the Odors tab is opened
      }
      const waiting = p.changed.length + p.missing.length;
      this.syncNote = `${added ? `${added} new odor${added === 1 ? '' : 's'} added from the odor sheet` : 'nothing new in the odor sheet'}`
        + `${waiting ? `; ${waiting} other change${waiting === 1 ? '' : 's'} to the odor table wait${waiting === 1 ? 's' : ''} for the Odors tab` : ''}`;
      setStatus(out, 'Checking its odors again…');
      await this.checkOdors();
    } catch (e) { setStatus(out, `Not synced: ${e.message}`, 'error'); } finally { btn.disabled = false; }
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
    const fresh = new Set((res.added || []).map((a) => a.name));
    const files = new Map((res.files || []).map((f) => [f.name, f]));
    list.innerHTML = res.stage.odor_yamls.map((y) => `<div class="yaml-file${fresh.has(y.name) ? ' fresh' : ''}">
      <div class="file"${y.tensor_path ? ` data-src="nas" title="${esc(`${SOURCES.nas}: ${y.tensor_path}`)}"` : ''}>
      <code>${esc(y.name)}</code><span class="muted">${y.n_odors} odors${y.tensor_path ? `, from <code>${esc(y.tensor_path)}</code>` : ''}</span>
      <button type="button" class="btn small" data-remove="${esc(y.name)}">remove</button></div>
      ${files.has(y.name) ? yamlFileCheck(files.get(y.name)) : ''}</div>`).join('');
    $$('[data-remove]', list).forEach((b) => b.addEventListener('click', () => this.removeYaml(b.dataset.remove)));
    const box = $('#yaml-check');
    box.innerHTML = '';
    // what this request did, as a file of the same name replaces one looks the same in the list
    const done = (res.added || []).map((a) => `${a.replaced ? 'Replaced' : 'Added'} <code>${esc(a.name)}</code>`
      + `${a.tensor_path ? `, from tensor (<code>${esc(a.tensor_path)}</code>),` : a.replaced ? ' with the file uploaded' : ''} (${a.n_odors} odors).`);
    if (res.removed) done.push(`Removed <code>${esc(res.removed)}</code>.`);
    if (done.length) box.appendChild(msgBox('info', done.join('<br>')));
    if (res.errors?.length) box.appendChild(msgBox('error', `Not added:${listHtml(res.errors)}`));
    if (this.nas) this._renderLookup();
    this._renderYamlStatus();
    this.checkOdors(); // matched through the YAMLs
    this._saveSoon();
    const c = res.check;
    if (!c) return;
    if (c.ok) {
      box.appendChild(msgBox('ok', `All ${c.n_matrix_odors} odors in the matrix are described in the YAML${c.unused_yaml.length ? ` (${c.unused_yaml.length} YAML odors are not in the matrix: ${esc(c.unused_yaml.join(', '))})` : ''}.`));
    } else {
      const rows = c.missing.map((m) => `<tr><td><code>${esc(m.odor)}</code></td><td class="muted">${esc(m.hint)}</td></tr>`).join('');
      box.appendChild(msgBox('error', `<b>${c.missing.length} of ${c.n_matrix_odors} odors in the matrix are not in the YAML</b> (name and concentration must both match). Add the right YAML, remove this one, or fix the file; saving is blocked until the check passes.<table class="check-table">${rows}</table>`));
    }
  }

  // file: a File, or {tensorPath} for one on tensor; auto: attached by a lookup, not by hand
  async addXml(file, { auto = false } = {}) {
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
    this.xmlAuto = auto;
    this._saveSoon();
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
    if (!meta.driver) errs.push('driver is required');
    if (!meta.coding_unit) errs.push('coding unit is required');
    if (meta.age_min_days != null && meta.age_max_days != null && meta.age_max_days < meta.age_min_days) errs.push('fly age range: max is smaller than min');
    if (this.yamlCheck && !this.yamlCheck.ok) errs.push('the odor YAML check has not passed');
    const upper = this.odorCheck?.case?.uppercase || [];
    if (upper.length) errs.push(`${upper.length} of its odor names ${upper.length === 1 ? 'has' : 'have'} capital letters, and new ones must be lowercase (see under the odor YAML)`);
    const unmatched = this._unmatchedOdors();
    if (unmatched.length) {
      errs.push(`${unmatched.length} of its odors ${unmatched.length === 1 ? 'is' : 'are'} not in the odor table (see under the odor YAML): `
        + `add ${unmatched.length === 1 ? 'it' : 'them'} to the odor sheet and sync it there`);
    }
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
    this._save();
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
    LS.set(UPLOAD_KEY, null);
    $('#step-done').hidden = false;
    $('#done-text').innerHTML = `Dataset <b>#${res.dataset_id}</b> — ${esc(res.title)} — is in the database.`;
    $('#done-link').href = `#browse/${res.dataset_id}`;
    $('#step-done').scrollIntoView({ behavior: 'smooth', block: 'start' });
    browse.invalidate();
  }
}

// =================================================================== BROWSE
// a coding unit matched as the Merge tab matches it: in any case, spaces collapsed
const unitKey = (u) => (u || '').trim().replace(/\s+/g, ' ').toLowerCase();

// The coding unit menu `sel` (each unit spelled as first uploaded, the most common first,
// as in the Merge tab), with `unit` (a unitKey, '' for all) chosen; returns the datasets to
// list: all of them if that unit is no longer uploaded.
function filterByUnit(datasets, sel, unit) {
  const units = new Map();
  for (const d of datasets.slice().sort((a, b) => a.id - b.id)) {
    const k = unitKey(d.coding_unit);
    if (!units.has(k)) units.set(k, { label: d.coding_unit.trim().replace(/\s+/g, ' '), n: 0 });
    units.get(k).n++;
  }
  const keys = [...units.keys()].sort((a, b) => units.get(b).n - units.get(a).n || a.localeCompare(b));
  const html = `<option value="">All (${datasets.length})</option>`
    + keys.map((k) => `<option value="${esc(k)}">${esc(units.get(k).label)} (${units.get(k).n})</option>`).join('');
  if (sel.innerHTML !== html) sel.innerHTML = html;
  const chosen = units.has(unit) ? unit : '';
  sel.value = chosen;
  return chosen ? datasets.filter((d) => unitKey(d.coding_unit) === chosen) : datasets;
}

// Datasets in the groups of a list (`group`: 'panel', by the panels the form gave, else
// the file's; or 'date', by the month of their last fly), each newest first: [{label, datasets}]
function datasetGroups(datasets, group) {
  const newest = (ds) => ds.slice().sort((a, b) => (b.last_fly_date || '').localeCompare(a.last_fly_date || '') || b.id - a.id);
  if (group !== 'panel') {
    const ds = newest(datasets);
    return runsOf(ds, (d) => (d.last_fly_date || '').slice(0, 7) || 'unknown date').map((m) => ({ label: m.key, datasets: ds.slice(m.start, m.end) }));
  }
  const groups = new Map();
  for (const d of datasets) {
    const panels = (d.panels.some((p) => p.source === 'form') ? d.panels.filter((p) => p.source === 'form') : d.panels).map((p) => p.panel);
    for (const p of panels.length ? panels : ['(no panel name)']) {
      if (!groups.has(p)) groups.set(p, []);
      groups.get(p).push(d);
    }
  }
  return [...groups.keys()].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
    .map((k) => ({ label: k, datasets: newest(groups.get(k)) }));
}

class Browse {
  constructor() {
    this.datasets = null;
    this.group = LS.get('olfdb.browse_group', 'panel');
    this.unit = LS.get('olfdb.browse_unit', ''); // unitKey of the one coding unit listed, '' for all
    this.selected = null;
    this.picked = new Set(); // ticked for download
    this.viewer = null;
    this.cache = new Map();
    $$('#tab-browse .seg button').forEach((b) => b.addEventListener('click', () => { this.group = b.dataset.group; LS.set('olfdb.browse_group', this.group); this.renderList(); }));
    $('#browse-unit').addEventListener('change', (e) => { this.unit = e.target.value; LS.set('olfdb.browse_unit', this.unit); this.renderList(); });
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
    $('#ds-mismatches').addEventListener('click', () => this.showMismatches());
    $('#ds-odors').addEventListener('click', async (e) => {
      const b = e.target.closest('[data-copy-sheet]');
      if (!b) return;
      const out = b.nextElementSibling;
      try {
        await copyText(this.odorCache.get(this.selected).sheet.paste);
        setStatus(out, '✓ Copied');
      } catch (err) { setStatus(out, `Could not copy: ${err.message}`, 'error'); }
    });
    const downloadUrl = (ids, format) => `/api/download?ids=${ids.join(',')}&format=${format}${format === 'files' ? '' : lowerQuery()}`;
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
    const listed = new Set($$('.pick', list).map((cb) => +cb.dataset.id));
    const unlisted = [...this.picked].filter((id) => !listed.has(id)).length; // ticked under another coding unit
    const text = $('#pick-text');
    text.textContent = n ? `${n} selected${unlisted ? ` (${unlisted} not listed)` : ''} · ${units.toLocaleString()} ROIs` : 'Tick datasets to download several together.';
    text.classList.toggle('muted', !n);
    $('#pick-download').hidden = !n;
    $('#pick-clear').hidden = !n;
  }

  renderList() {
    $$('#tab-browse .seg button').forEach((b) => b.classList.toggle('active', b.dataset.group === this.group));
    const list = $('#dataset-list');
    if (!this.datasets) return;
    $('.unit-filter').hidden = !this.datasets.length;
    if (!this.datasets.length) { list.innerHTML = '<div class="empty">Nothing uploaded yet. Use the Upload tab.</div>'; this._syncPicks(); return; }
    const datasets = this._unitFilter();
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
    list.innerHTML = datasetGroups(datasets, this.group).map((g) => groupHead(g.label, g.datasets) + g.datasets.map(item).join('')).join('');
    $$('.list-item', list).forEach((b) => b.addEventListener('click', () => { location.hash = `#browse/${b.dataset.id}`; }));
    this._syncPicks();
  }

  // The coding unit menu, each unit spelled as first uploaded, the most common first (as in
  // the Merge tab); returns the datasets to list. A unit no longer uploaded lists them all.
  _unitFilter() { return filterByUnit(this.datasets, $('#browse-unit'), this.unit); }

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
    this.recordings = null;
    $('#ds-mismatches').hidden = true;
    $('#ds-all-data').hidden = true;
    let res;
    try { res = await api(`/api/datasets/${id}/data_paths`); } catch (e) {
      if (this.selected === id) menus.forEach((m) => this._dataLink(m, { status: 'unchecked', error: `could not ask the server (${e.message})` }));
      return;
    }
    if (this.selected !== id) return;
    menus.forEach((m) => this._dataLink(m, res[m.dataset.kind]));
    // recordings in one of the folders and not the other: all of them side by side, on asking
    this.recordings = res.recordings;
    this.dataFound = { raw: res.raw.found, mocorr: res.mocorr.found };
    this.dataSkipped = { raw: res.raw.skipped, mocorr: res.mocorr.skipped };
    const n = (kind) => res[kind].unmatched?.length || 0;
    const btn = $('#ds-mismatches');
    btn.hidden = !(n('raw') + n('mocorr'));
    btn.title = `${n('raw')} recording${n('raw') === 1 ? '' : 's'} only in the raw data, ${n('mocorr')} only in the motion-corrected data`;
    // both looked in, and every recording in both (but anatomical ones)
    const all = $('#ds-all-data');
    all.hidden = !(res.raw.compared && btn.hidden);
    all.title = 'Every recording of its flies is in both the raw and the motion-corrected data (anatomical ones need not be)';
  }

  // every recording of the dataset's flies in its raw and motion-corrected data folders,
  // side by side in the order they were recorded, those in one only highlighted
  showMismatches() {
    const rows = this.recordings || [], d = this.cache.get(this.selected)?.d;
    const only = (kind) => rows.filter((r) => !r.matched && !r.anatomical && r[kind]).length;
    $('#mismatches-title').textContent = `Raw and motion-corrected data${d ? ` of ${d.title}` : ''}`;
    // how each folder's movies were found: in its flies' folders, or by the folders in their paths
    const how = Object.entries(this.dataFound || {}).filter(([, f]) => f && (f.path || f.none)).map(([kind, f]) => ` ${DATA_LINKS[kind]}: `
      + `none of this dataset's flies has a folder (<date>/<fly>) in its folder, so its movies were matched to them by the `
      + `folders in their paths${f.none ? `; ${f.none} are not in a folder of one of its flies (listed after them)` : ''}.`).join('');
    const left = Object.entries(this.dataSkipped || {}).filter(([, f]) => f?.length).map(([kind, f]) => ` ${DATA_LINKS[kind]}: `
      + `left out, the folders of flies not in this dataset: ${f.map((x) => `${x.date} fly ${x.fly_num}`).join(', ')}.`).join('');
    // the dataset's own flies, to tell those of other flies (or numbered otherwise) from them
    const theirs = (rows.some((r) => !r.of_dataset) || left) && d?.summary?.flies
      ? ` This dataset's flies (by its data's date and fly_num): ${d.summary.flies.map((f) => `${f.date} fly ${f.fly_num}`).join(', ')}.` : '';
    $('#mismatches-note').textContent = `${rows.length} recordings: ${only('raw')} only in the raw data, ${only('mocorr')} only in the `
      + 'motion-corrected data (highlighted). In the order they were recorded, as their Experiment.xml says (else by when it was '
      + 'written: "file time"); one with no raw data after the one before it by name. Anatomical recordings are not expected to '
      + `be motion corrected.${how}${left}${theirs}`;
    const cell = (r, kind) => (r[kind]
      ? `<code>${esc(r.recording ?? "(the fly's folder)")}</code> <span class="muted">${esc(r[kind].slice(0, 3).join(', '))}${r[kind].length > 3 ? `, and ${r[kind].length - 3} more` : ''}</span>`
      : `<span class="absent">${r.anatomical ? '— (anatomical)' : '— not here'}</span>`);
    let fly = null;
    $('#mismatches tbody').innerHTML = rows.map((r) => {
      const group = !r.date ? 'In no fly\'s folder (by the folder they are in)'
        : `${r.date} fly ${r.fly_num}${r.of_dataset ? '' : ' (by its folders: not one of this dataset\'s flies)'}`;
      const head = group !== fly ? `<tr class="group"><td colspan="3">${esc((fly = group))}</td></tr>` : '';
      const cls = r.matched ? '' : r.anatomical ? ' class="anatomical"' : ' class="mismatch"';
      const when = r.recorded ? esc(r.recorded.slice(0, 16)) : r.time != null ? `${esc(fmtEpoch(r.time))} <span class="muted">(file time)</span>` : '—';
      return `${head}<tr${cls}><td class="when">${when}</td>
        <td>${cell(r, 'raw')}</td><td>${cell(r, 'mocorr')}</td></tr>`;
    }).join('');
    $('#mismatches').showModal();
  }

  // r: as from tensor.find_data, or null while it is looked up
  _dataLink(menu, r) {
    const label = DATA_LINKS[menu.dataset.kind], list = $('.menu-list', menu);
    const only = r?.status === 'ok' ? r.unmatched || [] : [], [onlyWhy, onlyHead] = DATA_ONLY[menu.dataset.kind];
    menu.classList.toggle('missing', !!r && r.status !== 'ok');
    menu.classList.toggle('warn', only.length > 0);
    const recordings = (n) => `${n} recording${n === 1 ? '' : 's'}`;
    const why = r ? (only.length ? `${recordings(only.length)} ${onlyWhy}` : DATA_WHY[r.status]) : 'checking…';
    $('summary', menu).innerHTML = `<span class="data-label"><span>${label}</span>${why ? `<span class="why">${why}</span>` : ''}</span>`;
    $('summary', menu).title = r && r.status !== 'ok' ? `${label}: ${r.error}` : only.length ? `${label}: ${recordings(only.length)} ${onlyWhy}` : '';
    // those recordings, by fly
    const byFly = new Map();
    for (const u of only) {
      const fly = `${u.date} fly ${u.fly_num}`;
      byFly.set(fly, [...(byFly.get(fly) || []), u.recording == null ? "(in the fly's folder)" : u.recording]);
    }
    const onlyHtml = only.length ? `<div class="msg warn data-only"><b>${onlyHead}</b> (${recordings(only.length)}, listed apart under Play movies…):
      <ul>${[...byFly].map(([fly, recs]) => `<li>${esc(fly)}: ${recs.map((x) => `<code>${esc(x)}</code>`).join(', ')}</li>`).join('')}</ul></div>` : '';
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
    list.innerHTML = `${onlyHtml}<div class="kv data-where">
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
    if (this.selected !== id) return;
    panel.innerHTML = datasetOdorTable(this.odorCache.get(id));
    wireMenu($('#ds-odors-download', panel), ({ odorFormat }) => download(`/api/datasets/${id}/odors.${odorFormat}${lowerQuery('?')}`,
      $('#ds-odors-status', panel)));
  }
}

const DATA_LINKS = { raw: 'Raw data', mocorr: 'Motion corrected / traces' };
// recordings with movies in one of a dataset's data folders and none in the other: said so on that one
const DATA_ONLY = { raw: ['not motion corrected', 'Not in the motion-corrected data'], mocorr: ['not in the raw data', 'Not in the raw data'] };
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
// a time as 'YYYY-MM-DD HH:MM' in the browser's time zone
const fmtDate = (d) => {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};
// a time from the server (ISO) so: in the browser's time zone if it says its own (…Z,
// …-07:00), else as it is (the server's)
const fmtWhen = (iso) => {
  if (!/(Z|[+-]\d\d:?\d\d)$/.test(iso)) return iso.replace('T', ' ').slice(0, 16);
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso.replace('T', ' ').slice(0, 16) : fmtDate(d);
};
// a time in seconds since 1970 as fmtWhen gives one, in the browser's time zone
const fmtEpoch = (s) => fmtDate(new Date(s * 1000));
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

// a dataset's odors not in the odor table (`missing`, `sheet`: GET /api/datasets/<id>/odors or
// /api/stage/<id>/odors), with a button copying their rows for the odor sheet and how to paste
// them; `sync`: how to sync the sheet then
function missingRowsHtml(missing, sheet, sync) {
  const items = missing.map((m) => `<li>${m.name ? `<b>${esc(m.name)}</b>` : '<b>no name</b> <span class="muted">(no odor YAML names it)</span>'}:
    ${m.odors.map((o) => `<code>${esc(o)}</code>`).join(', ')}</li>`).join('');
  const k = missing.length, rowsText = `${k} row${k === 1 ? '' : 's'}`;
  const order = sheet.columns_from === 'sheet' ? `in the order of the odor sheet's columns (as read ${esc(fmtWhen(sheet.checked_at))})`
    : `in the order of the template's columns (<code>${esc(Object.keys(ODOR_FIELDS).join(', '))}</code>), as ${sheet.url
      ? `the odor sheet could not be read (${esc(sheet.error || 'not yet')})` : 'no odor sheet is set (Admin tab)'}: check that the sheet's are in this order`;
  const unnamed = missing.filter((m) => !m.name).length;
  const naming = !unnamed ? '' : unnamed < k ? `, give the ${unnamed} without a name one (the sheet needs it)`
    : `, give ${k === 1 ? 'it a name' : 'each a name'} (the sheet needs one)`;
  return `<ul class="missing-odors">${items}</ul>
    <div class="actions"><button type="button" class="btn small" data-copy-sheet>Copy ${rowsText} for the odor sheet</button><span class="status"></span></div>
    <p class="small">Tab-separated, ${order}. Paste them into the first column of the first empty row of
    ${sheet.url ? extLink(sheet.url, 'the odor sheet') : 'the odor sheet'}${naming}, fill in what else you know, then ${sync}.</p>`;
}

// the odors (`dataset_odors` rows, one per label) more than one odor in the odor table goes by
const ambiguousOdors = (rows) => [...new Map(rows.filter((r) => r.match === 'ambiguous').map((r) => [r.label, r])).values()];
const odorLabels = (rows) => rows.map((r) => `<code>${esc(r.label)}</code>`).join(', ');

// what a dataset's odors not in the odor table would need in the odor sheet (Browse)
function missingOdorsHtml({ rows, missing, sheet }) {
  const parts = [];
  if (missing.length) {
    parts.push(`<p>Not in the odor table (highlighted below):</p>${missingRowsHtml(missing, sheet, 'sync the sheet in the <a href="#odors">Odors tab</a>')}`);
  }
  const ambiguous = ambiguousOdors(rows);
  if (ambiguous.length) {
    parts.push(`<p>More than one odor in the odor table goes by ${odorLabels(ambiguous)} (highlighted below):
      an odor YAML naming the dataset's, or abbreviations of their own in the sheet, would tell which.</p>`);
  }
  return parts.length ? `<div class="msg warn">${parts.join('')}</div>` : '';
}

// a dataset's odors (GET /api/datasets/<id>/odors), per panel, like the thesis odor tables
function datasetOdorTable(data) {
  const { rows, odors } = data;
  const showOdor = odorShower(rows.map((r) => r.odor));
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
      const cls = o ? '' : r.match === 'ambiguous' ? ' class="ambiguous"' : ' class="missing"';
      return `<tr${cls}><td><code>${esc(showOdor(r.odor))}</code></td><td>${name}</td><td>${r.log10_conc ?? ''}</td><td>${c.cid || ''}</td>
        <td class="inchi">${esc(o?.inchi || '')}</td><td>${esc(r.yaml?.solvent || o?.solvent || '')}</td>
        ${hasGlom ? `<td>${esc(r.yaml?.glomerulus || '')}</td>` : ''}<td>${c.catalog || ''}</td><td>${c.metacyc || ''}</td><td>${c.lotus || ''}</td>
        <td class="structure-cell">${o ? structureImg(o) : ''}</td><td>${c.cas || ''}</td></tr>`;
    }).join('');
  }).join('');
  const n = rows.filter((r) => r.odor_id != null).length;
  return `<div class="odors-head"><p class="muted">${n === rows.length ? `All ${n}` : `${n} of ${rows.length}`} odors are in the <a href="#odors">odor table</a> (matched through the dataset's odor YAML where it has one, otherwise by abbreviation).</p>
    <span class="dl-status" id="ds-odors-status"></span>
    <details class="menu menu-right" id="ds-odors-download">
      <summary class="btn small">Download table</summary>
      <div class="menu-list">
        <button type="button" data-odor-format="csv">CSV</button>
        <button type="button" data-odor-format="tex">LaTeX (.tex)</button>
        <p class="menu-note">CSV: every odor with what the odor table has on it (IDs, InChI, supplier, links), as in downloads of several datasets.<br><br>LaTeX: a booktabs longtable to <code>\\input</code>, by panel, linked as here; InChI and structures left out.</p>
      </div>
    </details></div>
    ${missingOdorsHtml(data)}<table class="odor-table"><thead><tr><th>Odor</th><th>Name</th><th>log<sub>10</sub> conc.</th><th>CID</th><th>InChI</th><th>Solvent</th>
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
      last synced ${last ? `${esc(fmtWhen(last.applied_at))} (${syncCounts(last)})` : 'never'}.
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
      parts.push(`<div class="actions">
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
    const json = {
      fingerprint: this.preview.fingerprint,
      changes: $$('[data-change]:checked', box).map((x) => +x.dataset.change),
      deletions: $$('[data-delete]:checked', box).map((x) => +x.dataset.delete),
    };
    let res;
    try { res = await api('/api/odors/apply', { method: 'POST', json }); } catch (e) { out.replaceChildren(msgBox('error', esc(e.message))); return; }
    if (!res.ok) { out.replaceChildren(msgBox('error', `Not applied:${listHtml(res.errors)}`)); return; }
    this.preview = null;
    box.replaceChildren(msgBox('ok', `Synced: ${res.added} new, ${res.changed} changed, ${res.deleted} deleted.`));
    browse.odorCache.clear();
    comparePage.invalidate(); // its odors' matches to the table
    reference.invalidate(); // their chemicals' names
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
    const showAbbrev = abbrevShower(all.map((o) => o.abbrev));
    $('#odor-table').innerHTML = `<table class="odor-table"><thead><tr><th>Structure</th><th>Name</th><th>Abbrev.</th><th>CID</th><th>CAS</th><th>Solvent</th>
      <th>Catalog no.</th><th>MetaCyc</th><th>LOTUS</th><th>Inventory</th><th>Datasets</th><th>Notes</th></tr></thead><tbody>
      ${shown.map((o) => {
        const c = idCells(o);
        return `<tr id="odor-${o.id}"><td class="structure-cell">${structureImg(o)}</td>
          <td><b>${esc(o.name)}</b>${o.aliases.length ? `<div class="muted small">${esc(o.aliases.join('; '))}</div>` : ''}</td>
          <td>${esc(showAbbrev(o.abbrev) || '')}</td><td>${c.cid}</td><td>${c.cas}</td><td>${esc(o.solvent || '')}</td><td>${c.catalog}</td>
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
    $('#odor-syncs').innerHTML = this.syncs.length ? `<table class="odor-table"><thead><tr><th>When</th><th>From</th><th>What</th></tr></thead><tbody>
      ${this.syncs.map((s) => {
        const d = s.details || {};
        const what = [
          d.added?.length ? `<div><b>new:</b> ${names(d.added)}</div>` : '',
          d.changed?.length ? `<div><b>changed:</b> ${d.changed.map((c) => `${esc(c.name)} <span class="muted">(${c.fields.map((f) => esc(ODOR_FIELDS[f.field] || f.field)).join(', ')})</span>`).join(', ')}</div>` : '',
          d.deleted?.length ? `<div><b>deleted:</b> ${names(d.deleted)}</div>` : '',
          d.not_applied?.length ? `<div class="muted">changes not applied: ${names(d.not_applied)}</div>` : '',
        ].join('');
        return `<tr><td>${esc(fmtWhen(s.applied_at))}</td><td>${esc(s.source)}</td><td>${what || syncCounts(s)}</td></tr>`;
      }).join('')}</tbody></table>` : '<p class="muted">No syncs yet.</p>';
  }
}

// ================================================================ REFERENCE
// Reference datasets that cannot be read (or not yet) are listed by their names alone, with why;
// those with chemicals not identified by a CID or InChI, with a warning.
function setAdminBadge(items) {
  const broken = items.filter((it) => it.state === 'error');
  const badge = $('#admin-badge');
  badge.hidden = !broken.length;
  badge.title = `${broken.length} reference dataset${broken.length === 1 ? '' : 's'} cannot be read (see the Admin tab)`;
}

class Reference {
  constructor() { this.items = null; this.viewer = null; this.selected = null; this.cache = new Map(); }

  // after reference datasets were added or deleted, or the odor table changed
  invalidate() { this.items = null; this.cache.clear(); }

  async load() {
    if (!this.items) {
      try { this.items = (await api('/api/reference')).datasets; } catch (e) { $('#reference-list').replaceChildren(msgBox('error', esc(e.message))); return; }
      setAdminBadge(this.items);
    }
    const list = $('#reference-list');
    if (!this.items.length) { list.innerHTML = '<div class="empty">No reference data yet: it is added in the Admin tab.</div>'; return; }
    list.innerHTML = this.items.map((it) => {
      const ok = it.state === 'ok';
      const why = ok ? (it.warning ? `<span class="s warn">⚠ ${esc(it.warning)}</span>` : '')
        : `<span class="s ${it.state === 'error' ? 'bad' : 'muted'}">${esc(it.state === 'error' ? `Cannot be read: ${it.error}` : 'Read once the server restarts')}</span>`;
      return `<button type="button" class="list-item${this.selected === it.id ? ' active' : ''}" data-id="${esc(it.id)}">
        <span class="t">${esc(it.title)}</span>${ok && it.description ? `<span class="s">${esc(it.description)}</span>` : ''}${why}</button>`;
    }).join('');
    $$('.list-item', list).forEach((b) => b.addEventListener('click', () => { location.hash = `#reference/${b.dataset.id}`; }));
  }

  async show(id) {
    this.selected = id;
    if (this.items) this.load();
    $('#reference-empty').hidden = true;
    $('#reference-view').hidden = false;
    $('#ref-view-title').textContent = 'Loading…';
    $('#ref-desc').textContent = '';
    $('#ref-msgs').replaceChildren();
    let ref;
    try {
      // kept, so that rows and columns removed or moved stay so when it is shown again
      if (!this.cache.has(id)) this.cache.set(id, await api(`/api/reference/${encodeURIComponent(id)}`));
      ref = this.cache.get(id);
    } catch (e) { $('#ref-view-title').textContent = 'Could not load'; $('#ref-desc').textContent = e.message; return; }
    if (this.selected !== id) return;
    const m = ref.meta;
    $('#ref-view-title').textContent = m.title;
    $('#reference-viewer').hidden = ref.state !== 'ok';
    if (ref.state !== 'ok') {
      this.cache.delete(id); // looked at again next time
      $('#ref-msgs').replaceChildren(ref.state === 'error'
        ? msgBox('error', `<b>It cannot be read:</b> ${esc(ref.error)}.${m.origin === 'admin' ? ` <span class="small">(<code>${esc(m.path)}</code>; see the Admin tab)</span>` : ''}`)
        : msgBox('info', 'It is read once the server restarts (Admin tab).'));
      return;
    }
    $('#ref-desc').textContent = [m.description, m.source ? `Source: ${m.source}` : ''].filter(Boolean).join(' ');
    const c = ref.chemicals, msgs = [];
    if (ref.warning) {
      msgs.push(unidentifiedBox(ref.warning, c.unidentified, 'Add them to the Odors table (with their CID), or give their CIDs or InChIs in the file.', 30));
    }
    const notes = [];
    if (c.axis) notes.push(`Its ${c.axis === 'rows' ? 'rows' : 'columns'} are chemicals: ${c.matched} of ${c.n} matched to the Odors table, labelled as there.`);
    const g = ref.glomeruli;
    if (g?.axis) notes.push(`Its ${g.axis === 'rows' ? 'rows' : 'columns'} are glomeruli (${g.matched} of ${g.n}): it can be compared with datasets on their glomeruli (Compare tab).`);
    if (c.ambiguous.length) notes.push(`More than one odor in the Odors table matches ${c.ambiguous.map(esc).join(', ')}.`);
    notes.push(...ref.notes.map((n) => `${esc(n[0].toUpperCase() + n.slice(1))}.`));
    if (m.origin === 'admin') notes.push(`Read from <code>${esc(m.path)}</code> on ${esc(fmtWhen(ref.imported_at))}.`);
    if (notes.length) msgs.push(Object.assign(document.createElement('p'), { className: 'muted small ref-notes', innerHTML: notes.join(' ') }));
    $('#ref-msgs').replaceChildren(...msgs);
    if (!this.viewer) this.viewer = new MatrixViewer($('#reference-viewer'), { kind: 'generic' });
    this.viewer.setGenericData(ref, { title: m.title, fileStem: `olfdb_reference-${id}` });
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
  const msg = $('.dir-msg', row);
  msg.textContent = c ? (c.ok ? c.warning || '' : c.error) : '';
  // what to change on the server for it to see the folder (Admin tab: HTML from the server)
  if (c?.fix) msg.insertAdjacentHTML('beforeend', `<div class="dir-fix">${c.fix}</div>`);
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
    const lower = $('#settings-lower-odors');
    lower.checked = display.lowerOdors;
    lower.addEventListener('change', () => { display.lowerOdors = lower.checked; displayChanged(); });
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
      upload.foldersChanged();
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
    return new URLSearchParams({ ...o, by: o.by.join(','), ...(format ? { format } : {}), ...(format && display.lowerOdors ? { lower: '1' } : {}) }).toString();
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
// too), the reference data; deleting datasets; the odor sheet's link; the server's name
// and addresses.
// Shown each time the tab is opened, once the user says they may change these settings.
const ADMIN_KINDS = ['read_dirs', 'write_dirs'];

// A reference dataset's file as checked (see /api/admin/references/check), under its field
function showRefCheck(c) {
  const input = $('#ref-path'), msg = $('#ref-path-msg'), box = $('#ref-check');
  input.classList.toggle('invalid', !!c && !c.ok);
  input.classList.toggle('path-warn', !!c?.restart);
  msg.className = `path-msg${c && !c.checking ? (c.ok ? (c.restart ? ' warn' : ' ok') : ' bad') : ''}`;
  box.replaceChildren();
  if (!c) { msg.textContent = ''; return; }
  if (c.checking) { msg.textContent = 'Checking…'; return; }
  if (!c.ok) { msg.textContent = c.error; return; }
  if (c.restart) {
    msg.textContent = c.grant ? 'outside the folders the server may read now: adding it adds this file to them (to read only)'
      : 'in the folders the server may read once it restarts';
    box.appendChild(msgBox('warn', `The server reads it once it restarts, after adding it:<ol class="steps">${c.steps.map((s) => `<li>${s}</li>`).join('')}</ol>`));
    return;
  }
  const p = c.preview, ch = p.chemicals;
  msg.textContent = `✓ read: ${p.shape[0].toLocaleString()} rows × ${p.shape[1].toLocaleString()} columns, ${p.kind}`;
  const lines = [];
  if (ch.axis) lines.push(`Its ${ch.axis === 'rows' ? 'rows' : 'columns'} are chemicals: ${ch.matched} of ${ch.n} matched to the Odors table, ${ch.identified} identified by CID or InChI.`);
  const g = p.glomeruli;
  if (g?.axis) lines.push(`Its ${g.axis === 'rows' ? 'rows' : 'columns'} are glomeruli (${g.matched} of ${g.n}, by name): it can be compared with datasets on their glomeruli.`);
  lines.push(...p.notes.map((n) => `${esc(n[0].toUpperCase() + n.slice(1))}.`));
  if (lines.length) box.appendChild(Object.assign(document.createElement('p'), { className: 'muted small', innerHTML: lines.join(' ') }));
  if (p.warning) {
    box.appendChild(unidentifiedBox(p.warning, ch.unidentified || [], 'It can be added anyway, with a warning by it in the Reference data tab, where they are listed too.'));
  }
}

// what is wrong with a folder as typed, as far as can be told without the server
function localServerDirProblem(path) {
  if (!/^\s*\//.test(path)) return 'give the absolute path on the server, starting with /';
  if (path.split('/').includes('..')) return "'..' is not allowed: give the folder itself";
  return null;
}

// The files the server has read or written in its own folders (as from /api/admin/files)
// as a tree: {dirs: Map(name: node), files}
function fileTree(files) {
  const top = { dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = f.path.split('/');
    let node = top;
    for (const part of parts.slice(0, -1)) {
      if (!node.dirs.has(part)) node.dirs.set(part, { dirs: new Map(), files: [] });
      node = node.dirs.get(part);
    }
    node.files.push({ ...f, name: parts.at(-1) });
  }
  return top;
}

function fileTreeTotals(node) {
  const t = { n: node.files.length, gone: 0, size: 0 };
  for (const f of node.files) { if (f.size === null) t.gone++; else t.size += f.size; }
  for (const child of node.dirs.values()) {
    const c = fileTreeTotals(child);
    t.n += c.n; t.gone += c.gone; t.size += c.size;
  }
  return t;
}

// a folder of `fileTree` (named `label`), folders first, each closed but for this one if `open`
function fileTreeHtml(node, label, open) {
  while (!node.files.length && node.dirs.size === 1) { // a folder holding just a folder: on one line
    const [[name, child]] = node.dirs;
    label += `${name}/`;
    node = child;
  }
  const t = fileTreeTotals(node);
  const counts = [`${t.n.toLocaleString()} file${t.n === 1 ? '' : 's'}`, t.gone && `${t.gone.toLocaleString()} deleted since`,
    t.size && fmtBytes(t.size)].filter(Boolean).join(', ');
  const dirs = [...node.dirs].sort(([a], [b]) => a.localeCompare(b)).map(([name, child]) => `<li>${fileTreeHtml(child, `${name}/`, false)}</li>`);
  const files = node.files.map((f) => {
    const did = [f.read && 'read', f.wrote && 'written'].filter(Boolean).join(', ');
    const state = f.size === null ? 'deleted since' : `${f.size ? fmtBytes(f.size) : 'empty'} · modified ${fmtEpoch(f.mtime)}`;
    const when = f.first_at ? `first read or written ${fmtWhen(f.first_at)}` : 'in the data folder when this record began';
    return `<li class="file${f.size === null ? ' gone' : ''}" title="${esc(when)}"><code>${esc(f.name)}</code> <span class="muted">${state} · ${did}</span></li>`;
  });
  return `<details${open ? ' open' : ''}><summary><code>${esc(label)}</code> <span class="muted">${counts}</span></summary><ul>${[...dirs, ...files].join('')}</ul></details>`;
}

class AdminPage {
  constructor() {
    this.state = null; // as from GET /api/admin
    this.datasets = null; // as from GET /api/datasets
    this.refs = null; // as from GET /api/reference
    this._refSeq = 0;
    this._refTimer = 0;
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
    $('#admin-backup-btn').addEventListener('click', () => this.backup());
    this._backupTimer = 0;
    this._watchingBackup = false; // a backup started here is being made
    $('#admin-sheet-save').addEventListener('click', () => this.saveSheet());
    $('#admin-lowercase').addEventListener('change', (e) => this.saveCase(e.target.checked));
    $('#admin-sheet-url').addEventListener('keydown', (e) => { if (e.key === 'Enter') this.saveSheet(); });
    $('#admin-restart').addEventListener('click', () => this.restart());
    $('#admin-confirm').addEventListener('close', () => this._answered());
    $('#ref-path').addEventListener('input', () => {
      clearTimeout(this._refTimer);
      const problem = $('#ref-path').value.trim() ? localServerDirProblem($('#ref-path').value) : null;
      showRefCheck(problem ? { ok: false, error: problem } : null);
      if (!problem && $('#ref-path').value.trim()) this._refTimer = setTimeout(() => this.checkRef(), 500);
    });
    $('#ref-path').addEventListener('change', () => { clearTimeout(this._refTimer); this.checkRef(); });
    for (const sel of ['#ref-axis', '#ref-kind']) $(sel).addEventListener('change', () => this.checkRef());
    $('#ref-add').addEventListener('click', () => this.addRef());
    $('#admin-refs-list').addEventListener('click', (e) => {
      const b = e.target.closest('[data-delete-ref]');
      if (b) this.deleteRef(b.dataset.deleteRef);
    });
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
    this.loadFiles();
    this.loadBackups();
    this.loadDatasets();
    this.loadRefs();
    $('#ref-result').replaceChildren();
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
    $('#admin-data-dir-where').textContent = s.in_container ? ' in the container; on the server, the folder docker-compose.yml mounts there' : '';
    $('#admin-tensor-root').textContent = s.tensor_root;
    $('#admin-deleted-dir').textContent = s.deleted_dir;
    for (const kind of ADMIN_KINDS) {
      const list = $(`#admin-folders .dirlist[data-kind=${kind}]`);
      list.dataset.label = 'Folder on the server';
      list.replaceChildren(...[...s.saved[kind], ''].map((p) => dirRow(list, p)));
      s.checks[kind].forEach((c, i) => showDirCheck(list.children[i], c));
    }
    this._showSheet();
    $('#admin-lowercase').checked = !!s.lowercase_only;
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
    if (!res.pending) { out.replaceChildren(msgBox('ok', 'Saved: these are the folders in effect now.')); return; }
    // restarting from here, rather than from Restart below
    const box = msgBox('ok', `Saved. These folders take effect when the server restarts${res.restart.possible
      ? ': <button type="button" class="btn small primary" data-restart>Restart now</button> <span class="dl-status" aria-live="polite"></span>'
      : '. It is not run by Docker or systemd, which would start it again: restart it by hand on the server (DEPLOY.md).'}`);
    $('[data-restart]', box)?.addEventListener('click', (e) => this.restart(e.target, e.target.nextElementSibling, { here: true }));
    out.replaceChildren(box);
  }

  _showSheet() {
    const o = this.state.odor_sheet, input = $('#admin-sheet-url');
    if (document.activeElement !== input) input.value = o.url || '';
    $('#admin-sheet-source').innerHTML = o.saved ? `Set here ${esc(fmtWhen(o.updated_at))}.`
      : o.url ? 'Set on the server (<code>OLFDB_ODOR_SHEET_URL</code>); saving a link here replaces it.'
        : 'None yet: until there is one, the odor table can only be synced from CSVs imported below.';
  }

  async saveCase(only) {
    const out = $('#admin-lowercase-result');
    let res;
    try { res = await api('/api/admin/odor_case', { method: 'POST', json: { lowercase_only: only } }); } catch (e) {
      out.replaceChildren(msgBox('error', `Not saved: ${esc(e.message)}`));
      $('#admin-lowercase').checked = !only;
      return;
    }
    this.state.lowercase_only = res.lowercase_only;
    out.replaceChildren(msgBox('ok', res.lowercase_only ? 'Saved: new odor names with capital letters are refused from now on.' : 'Saved: odor names may have capital letters.'));
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

  async loadFiles() {
    const box = $('#admin-files');
    box.replaceChildren(Object.assign(document.createElement('p'), { className: 'muted', textContent: 'Listing…' }));
    let res;
    try { res = await api('/api/admin/files'); } catch (e) { box.replaceChildren(msgBox('error', `Could not list them: ${esc(e.message)}`)); return; }
    $('#admin-files-root').textContent = res.root;
    $('#admin-files-since').textContent = res.since ? ` since ${fmtWhen(res.since)}` : '';
    const tree = document.createElement('div');
    tree.innerHTML = res.files.length ? fileTreeHtml(fileTree(res.files), `${res.root.replace(/\/$/, '')}/`, true) : '<p class="muted">None yet.</p>';
    box.replaceChildren(...(res.note ? [msgBox('warn', `Paths as in the Docker container: ${esc(res.note)}.`)] : []), ...tree.childNodes);
  }

  async loadBackups() {
    clearTimeout(this._backupTimer);
    let s;
    try { s = await api('/api/admin/backups'); } catch (e) {
      $('#admin-backup-status').replaceChildren(msgBox('error', `Could not read the backups: ${esc(e.message)}`));
      return;
    }
    this._showBackups(s);
  }

  // the backups (as from /api/admin/backups): whether the last is current, the folders to
  // make one into, and every backup in them; checked again every second while one is made
  _showBackups(s) {
    const { job, last, current: c } = s, out = $('#admin-backup-msg');
    const select = $('#admin-backup-folder'), chosen = select.value;
    select.innerHTML = s.folders.map((f) => `<option value="${esc(f.path)}"${f.ok ? '' : ' disabled'}>${esc(f.path)}${f.ok ? '' : ` (${esc(f.error)})`}</option>`).join('');
    const ok = s.folders.filter((f) => f.ok);
    if (ok.some((f) => f.path === chosen)) select.value = chosen; else if (ok.length) select.value = ok[0].path;
    $('.backup-folder').hidden = !s.folders.length;
    select.disabled = job.running;
    $('#admin-backup-btn').disabled = job.running || !ok.length;

    let kind, label, detail;
    if (job.running) {
      [kind, label] = ['pending', 'Backing Up…'];
      detail = `into <code>${esc(job.folder)}</code>: ${job.total ? `${job.done.toLocaleString()} of ${job.total.toLocaleString()} files` : 'the database'}`;
    } else if (!last) [kind, label, detail] = ['none', 'No Backup Yet', ''];
    else if (c.ok) [kind, label, detail] = ['ok', 'Backup Current', 'nothing has changed since the last backup'];
    else {
      const files = (n, what) => n && `${n.toLocaleString()} file${n === 1 ? '' : 's'} ${what}`;
      [kind, label] = ['bad', 'Backup Not Current'];
      detail = `changed since the last backup: ${[c.database && 'the database', files(c.added, 'new'), files(c.changed, 'changed'),
        files(c.removed, 'removed')].filter(Boolean).join(', ')}`;
    }
    const parts = [`<p><span class="indicator ${kind}">${label}</span>${detail ? ` <span class="muted">${detail}</span>` : ''}</p>`];
    if (last) {
      parts.push(`<p>Last backup: <b>${esc(fmtWhen(last.created_at))}</b>, into <code>${esc(last.folder)}</code>
        (${last.database.datasets} dataset${last.database.datasets === 1 ? '' : 's'}, ${last.n_files.toLocaleString()} files, ${fmtBytes(last.bytes)}).</p>`);
    }
    if (!s.folders.length) {
      const pending = s.pending.length ? ` <code>${s.pending.map(esc).join('</code>, <code>')}</code> ${s.pending.length === 1 ? 'is' : 'are'} saved there: in effect once the server restarts (below).` : '';
      parts.push(msgBox('info', `No folder to back up into: add one above, under <i>May also write</i> (in Docker, also mounted writable in
        <code>docker-compose.yml</code>), save, then restart the server.${pending}`).outerHTML);
    } else if (!ok.length) {
      parts.push(msgBox('error', 'None of the folders it may write can be written now: see under <i>May also write</i>, above, for what it takes.').outerHTML);
    }
    if (!job.running && job.error && !this._watchingBackup) parts.push(msgBox('error', `The last backup failed: ${esc(job.error)}`).outerHTML);
    $('#admin-backup-status').innerHTML = parts.join('');

    if (this._watchingBackup && !job.running) {
      this._watchingBackup = false;
      setStatus(out, job.error ? `Failed: ${job.error}` : `✓ Backed up as ${job.name}`, job.error ? 'error' : '');
    }
    if (job.running) this._backupTimer = setTimeout(() => this.loadBackups(), 1000);

    const rows = s.folders.flatMap((f) => f.backups.map((b) => ({ ...b, folder: f.path }))).sort((a, b) => b.name.localeCompare(a.name));
    const many = s.folders.length > 1;
    $('#admin-backup-list').innerHTML = !rows.length ? '' : `<table class="odor-table backups"><thead><tr><th>Backup</th><th>Made</th>
      <th>Datasets</th><th>Files</th><th>Size</th><th title="what it does not share with the backup before it">New in it</th>${many ? '<th>Folder</th>' : ''}</tr></thead>
      <tbody>${rows.map((b) => `<tr${b === rows.find((r) => r.complete) ? ' class="hl"' : ''}><td><code>${esc(b.name)}</code></td>${b.complete
        ? `<td>${esc(fmtWhen(b.created_at))}</td><td>${b.database.datasets}</td><td>${b.n_files.toLocaleString()}</td><td>${fmtBytes(b.bytes)}</td>
          <td>${fmtBytes(b.copied_bytes)}</td>`
        : `<td colspan="5" class="muted">${job.running && job.folder === b.folder ? 'being made' : 'unfinished (interrupted): removed by the next backup into this folder'}</td>`}
        ${many ? `<td><code>${esc(b.folder)}</code></td>` : ''}</tr>`).join('')}</tbody></table>`;
  }

  async backup() {
    const out = $('#admin-backup-msg');
    setStatus(out, '');
    let s;
    try { s = await api('/api/admin/backups', { method: 'POST', json: { folder: $('#admin-backup-folder').value } }); } catch (e) {
      setStatus(out, `Not started: ${e.message}`, 'error');
      return;
    }
    this._watchingBackup = true;
    this._showBackups(s);
  }

  // the folders and restart state again (after a reference dataset added a folder, say),
  // leaving messages shown
  async _reloadState() {
    try { this._show(await api('/api/admin')); } catch { /* shown when the tab is opened again */ }
  }

  async loadRefs() {
    try { this.refs = (await api('/api/reference')).datasets; } catch (e) {
      $('#admin-refs-list').replaceChildren(msgBox('error', `Could not load the reference data: ${esc(e.message)}`));
      return;
    }
    setAdminBadge(this.refs);
    const broken = this.refs.filter((r) => r.state === 'error');
    $('#admin-refs-status').replaceChildren(...(broken.length ? [msgBox('error',
      `<b>${broken.length} reference dataset${broken.length === 1 ? '' : 's'} cannot be read</b>, and ${broken.length === 1 ? 'is' : 'are'} listed by name alone in the Reference data tab: ${broken.map((r) => esc(r.title)).join(', ')} (why: below).`)] : []));
    // those added here (the built-in examples are only in the Reference data tab), and any that cannot be read
    const listed = this.refs.filter((r) => r.origin === 'admin' || r.state === 'error');
    if (!listed.length) { $('#admin-refs-list').innerHTML = '<p class="muted">None added yet.</p>'; return; }
    const stateHtml = (r) => {
      if (r.state === 'error') return `<span class="bad">Cannot be read: ${esc(r.error)}</span>`;
      if (r.state === 'pending') return '<span class="warn">Read once the server restarts (below)</span>';
      const [nr, nc] = r.shape;
      return `${nr.toLocaleString()} × ${nc.toLocaleString()}, ${esc(r.kind)}${r.imported_at ? `, read ${esc(fmtWhen(r.imported_at))}` : ''}`
        + (r.warning ? `<br><span class="warn">⚠ ${esc(r.warning)}</span>${r.chemicals?.axis && r.chemicals.identified < r.chemicals.n
          ? ` <a href="#reference/${esc(r.id)}">(which)</a>` : ''}` : '');
    };
    $('#admin-refs-list').innerHTML = `<table class="odor-table"><thead><tr><th>Title</th><th>File</th><th>State</th><th></th></tr></thead><tbody>
      ${listed.map((r) => `<tr><td><a href="#reference/${esc(r.id)}">${esc(r.title)}</a></td>
        <td><code>${esc(r.path)}</code>${r.origin === 'builtin' ? ' <span class="muted">(built in)</span>' : ''}</td><td>${stateHtml(r)}</td>
        <td>${r.origin === 'admin' ? `<button type="button" class="btn small" data-delete-ref="${esc(r.id)}">Delete…</button>`
          : '<span class="muted small" title="Remove it from reference_data/manifest.json on the server">built in</span>'}</td></tr>`).join('')}</tbody></table>`;
  }

  _refBody() {
    const v = (id) => $(id).value.trim();
    return {
      path: v('#ref-path'), title: v('#ref-title'), description: v('#ref-description'), source: v('#ref-source'),
      chemical_axis: $('#ref-axis').value, kind: $('#ref-kind').value,
      value_label: v('#ref-value-label'), row_label: v('#ref-row-label'), col_label: v('#ref-col-label'),
    };
  }

  // the file, checked on the server: read now if it may be, else what adding it would take
  async checkRef() {
    const seq = ++this._refSeq, body = this._refBody();
    if (!body.path || localServerDirProblem(body.path)) return;
    showRefCheck({ ok: true, checking: true });
    let c;
    try { c = await api('/api/admin/references/check', { method: 'POST', json: body }); } catch (e) { c = { ok: false, error: e.message }; }
    if (seq === this._refSeq && $('#ref-path').value.trim() === body.path) showRefCheck(c);
  }

  async addRef() {
    const out = $('#ref-result');
    let res;
    try { res = await api('/api/admin/references', { method: 'POST', json: this._refBody() }); } catch (e) {
      out.replaceChildren(msgBox('error', `Not added: ${esc(e.message)}`));
      return;
    }
    if (!res.ok) {
      if (res.check) showRefCheck(res.check);
      out.replaceChildren(msgBox('error', `Not added:${listHtml(res.errors)}`));
      return;
    }
    const title = esc(this._refBody().title);
    if (res.restart) {
      const added = res.grant ? ` Its file was added to the folders it may read (above), to read only.` : '';
      out.replaceChildren(msgBox('warn', `<b>Added “${title}”. The server reads it once it restarts.</b>${added}<ol class="steps">${res.steps.map((s) => `<li>${s}</li>`).join('')}</ol>`
        + 'After the restart, it is listed above as read, or with why it cannot be.'));
    } else if (res.state === 'ok') {
      out.replaceChildren(msgBox('ok', `Added “${title}”: it is in the Reference data tab.`));
    } else {
      out.replaceChildren(msgBox('error', `Added “${title}”, but it cannot be read: ${esc(res.error)}`));
    }
    for (const id of ['#ref-path', '#ref-title', '#ref-description', '#ref-source', '#ref-value-label', '#ref-row-label', '#ref-col-label']) $(id).value = '';
    showRefCheck(null);
    reference.invalidate();
    this._reloadState();
    this.loadRefs();
  }

  async deleteRef(id) {
    const r = this.refs.find((x) => x.id === id), out = $('#ref-result');
    if (!r) return;
    $('#delete-confirm-q').textContent = `Delete the reference dataset “${r.title}”?`;
    $('#delete-confirm-detail').innerHTML = `Only the app's copy of it is deleted: its file, <code>${esc(r.path)}</code>, is left as it is.`
      + (r.read_grant ? ' It was added to the folders the server may read for it; it is taken off them again, unless another reference dataset needs it.' : '');
    if (await askDialog($('#delete-confirm')) !== 'delete') return;
    let res;
    try { res = await api(`/api/admin/references/${encodeURIComponent(id)}`, { method: 'DELETE' }); } catch (e) {
      out.replaceChildren(msgBox('error', `Not deleted: ${esc(e.message)}`));
      this.loadRefs();
      return;
    }
    out.replaceChildren(msgBox('ok', `Deleted “${esc(r.title)}”; its file was left as it is.`
      + (res.revoked ? ` <code>${esc(res.revoked)}</code> was taken off the folders the server may read: it can read it until it restarts.` : '')));
    reference.invalidate();
    if (res.revoked) this._reloadState();
    this.loadRefs();
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
  // `btn`, `status`: the button clicked and where to say how it goes (that of Restart, or
  // one by the folders just saved: `here`)
  async restart(btn = $('#admin-restart'), status = $('#admin-restart-status'), { here = false } = {}) {
    const buttons = [...new Set([$('#admin-restart'), btn])];
    const enable = () => buttons.forEach((b) => { b.disabled = false; });
    buttons.forEach((b) => { b.disabled = true; });
    setStatus(status, 'Asking the server to restart…');
    let res;
    try {
      res = await api('/api/admin/restart', { method: 'POST', json: {} });
    } catch (e) {
      setStatus(status, `Could not restart: ${e.message}`, 'error');
      enable();
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
        enable();
        return;
      }
      setStatus(status, `Restarting… ${secs} s`);
    }
    setStatus(status, 'Restarted.');
    $('#topbar-status').textContent = '';
    await this.load();
    if (here && this.state) {
      $('#admin-result').replaceChildren(this.state.pending
        ? msgBox('warn', 'Restarted, but the folders saved are still not those in effect: see Restart, below.')
        : msgBox('ok', 'Restarted: the folders saved are in effect now.'));
    }
  }
}

// =================================================================== COMPARE
// Datasets side by side, each its odors (rows) by its glomeruli or ROIs, every trial of an
// odor averaged (and every fly, for glomeruli: /api/compare/datasets/<id>); how alike
// their odors are within each, and in other spaces of chemicals (the ECFP fingerprints of
// their structures, reference datasets); and how alike those likenesses are between any two.
const COMPARE_KEY = 'olfdb.compare';

// coding units spelled as they are shown (by unitKey)
const UNIT_NAMES = { kc: 'KCs', kcs: 'KCs', orn: 'ORNs', orns: 'ORNs', pn: 'PNs', pns: 'PNs' };

// a dataset's name as a space: what was recorded (its coding unit; for glomeruli, ORNs if
// its driver is pebbled, PNs if GH146)
function neuralSpaceName(d) {
  const unit = (d.coding_unit || '').trim().replace(/\s+/g, ' ');
  if (unitKey(unit) !== 'glomeruli') return UNIT_NAMES[unitKey(unit)] || unit || `#${d.id}`;
  const driver = (d.driver || '').toLowerCase();
  return driver.includes('pebbled') ? 'ORNs' : driver.includes('gh146') ? 'PNs' : 'glomeruli';
}

// the Holm–Bonferroni method, by which the p-values of correlations between similarities are adjusted
const HOLM_URL = 'https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method';

// what spaces are compared on (ComparePage's o.on), as the page says it: and its options panel's notes
const ON = {
  odors: {
    one: 'odor', many: 'odors', sim: 'odor–odor similarity',
    show: 'Datasets, reference data (over the odors shown that it has) and the two side by side alike; chemical structure only ever as similarity.',
    simNote: "Of two odors: by their responses across a dataset's glomeruli or ROIs (those both have), or by their values across a "
      + "reference dataset's other axis. The correlations between similarities below are of these.",
    match: "Odors are matched by the odor table, mixtures left out. Each space's similarities are shown over the odors shown that it has.",
    none: 'No reference datasets with chemicals (Reference data tab).',
  },
  glomeruli: {
    one: 'glomerulus', many: 'glomeruli', sim: 'glomerulus–glomerulus similarity',
    show: 'Datasets of glomeruli (as glomeruli × odors, every fly averaged) and reference data of glomeruli (over the glomeruli shown that it has), '
      + 'and the two side by side alike. No odor–odor similarity is shown, nor chemical structure.',
    simNote: "Of two glomeruli: by their responses across a dataset's odors (every one, mixtures and concentrations as they are), or "
      + "by their values across a reference dataset's other axis. The correlations between similarities below are of these.",
    match: "Glomeruli are matched by name (but for case), as datasets' ROIs are named. Each space's similarities are shown over the "
      + 'glomeruli shown that it has.',
    none: 'No reference datasets with glomeruli: rows or columns labelled by glomerulus names (Reference data tab).',
  },
};

// how those p-values can be had (pvalue-worker.js), as their CSV says, of odors or glomeruli
const pBy = (unit, many) => ({ pairs: `bootstrap of the pairs of ${many}`, odors: `bootstrap of the ${many}`,
  permute: `permutation of the ${many} (Mantel test)` }[unit]);

// a p-value in a few characters
const fmtP = (p) => (p < 0.001 ? p.toExponential(1) : p < 0.01 ? p.toFixed(4) : p.toFixed(3));

// a 32-bit hash of a string (FNV-1a), as a seed
function hashString(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

// so the viewer offers chemicals' names or abbreviations as row labels
const CHEM_ROWS = { matched: true, axis: 'rows' };

// a space (as in ComparePage's `all`) over some of its odors (or glomeruli: `noun`), in
// their order, as a viewer shows a matrix: its data (by its glomeruli, ROIs, odors or other
// axis) or its similarities. rowOf: an odor's (or glomerulus's) row (label, chemical)
function spaceData(s, ids, rowOf, noun = 'odor') {
  const at = new Map(s.space.ids.map((id, i) => [id, i])), { vectors, cols, meta } = s.space.data, nC = cols.length;
  const values = new Float32Array(ids.length * nC);
  ids.forEach((id, r) => values.set(vectors[at.get(id)], r * nC));
  return { meta: { ...meta, row_label: noun }, rows: ids.map(rowOf), cols, values, chemicals: noun === 'odor' ? CHEM_ROWS : null };
}

function spaceSims(s, ids, rowOf, noun = 'odor') {
  const at = new Map(s.space.ids.map((id, i) => [id, i])), n = s.space.ids.length, m = ids.length;
  const values = new Float32Array(m * m);
  ids.forEach((a, i) => ids.forEach((b, j) => { values[i * m + j] = s.space.sim[at.get(a) * n + at.get(b)]; }));
  const rows = ids.map(rowOf);
  return { meta: { kind: 'diverging', row_label: noun, col_label: noun, value_label: s.simLabel }, rows, cols: rows, values,
    chemicals: noun === 'odor' ? CHEM_ROWS : null };
}

// two linked viewers ({el, viewer}) side by side made level, the same height above their
// cells (header and toolbar), then each fitted to what it shows
function levelViewers(x, y) {
  const vs = [x.viewer, y.viewer];
  if (vs.some((v) => !v.model)) return;
  const h = Math.max(...vs.map((v) => v.heatmap.naturalHeaderH));
  for (const v of vs) if (v.heatmap.options.minHeaderH !== h) v.heatmap.setOptions({ minHeaderH: h });
  for (const v of vs) v.toolbar.style.minHeight = '';
  const t = Math.ceil(Math.max(...vs.map((v) => v.toolbar.getBoundingClientRect().height)));
  for (const v of vs) v.toolbar.style.minHeight = `${t}px`;
  fitViewer(x.el, x.viewer);
  fitViewer(y.el, y.viewer);
}

// a viewer's height fitted to what it shows, up to most of the window's
function fitViewer(el, v) {
  if (!v.model) return;
  const want = v.toolbar.offsetHeight + v.heatmap.headerH + v.model.rows.length * v.opts.cellH + v.hint.offsetHeight + 22;
  el.style.height = `${Math.max(160, Math.min(want, Math.round(window.innerHeight * 0.8)))}px`;
}

class ComparePage {
  constructor() {
    // on: what spaces are compared on, 'odors' or 'glomeruli' (a key of ON);
    // metric, refMetric: similarity (of odors, or of glomeruli) within datasets, and within reference data;
    // show: each space as its 'data' or its similarities ('sim');
    // bootN, pBy: the resamplings (or permutations) for p-values, and how (a unit pBy names);
    // pair: the keys of two spaces shown side by side (null: not chosen)
    this.o = {
      group: 'panel', unit: '', picked: [], on: 'odors', metric: 'pearson', refMetric: 'pearson', show: 'data', spaces: [], ecfpD: 6, conc: 'max',
      rsa: false, rsaMethod: 'spearman', bootN: 5000, pBy: 'permute', pair: [null, null], ...LS.get(COMPARE_KEY, {}),
    };
    // `place` once, for datasets alone: under their data ('below'), instead of it, or 'none'
    if (this.o.place) this.o.show = this.o.place === 'instead' ? 'sim' : 'data';
    delete this.o.place;
    if (this.o.show !== 'sim') this.o.show = 'data'; // 'both' once
    delete this.o.sim; // a checkbox once, besides `place`
    delete this.o.bootUnit; // pBy once, when bootstrapping pairs of odors was the default
    this.datasets = null;
    this.refs = []; // reference datasets with chemicals or glomeruli, as listed
    this.data = new Map(); // dataset id -> as from /api/compare/datasets/<id>, with its values
    this.refData = new Map(); // reference id -> as from /api/reference/<id>
    this.ecfp = new Map(); // diameter -> fingerprints
    this.memo = new Map(); // what is computed, by what it is of
    this.blocks = new Map(); // key -> {el, viewer}, kept while shown
    this.job = null; // the bootstrap of p-values running (_rsaP)
    this.all = []; // the spaces shown, as last rendered
    this.pairs = []; // linked viewers side by side, as last rendered: [[{el, viewer}, {el, viewer}]]
    this.pairWidth = 0;
    this.pairRO = new ResizeObserver(([e]) => {
      // their toolbars wrap anew
      if (Math.round(e.contentRect.width) === this.pairWidth) return;
      this.pairWidth = Math.round(e.contentRect.width);
      this.pairs.forEach(([x, y]) => levelViewers(x, y));
    });
    this._seq = 0;
    const tab = $('#tab-compare');
    $$('.seg button', tab).forEach((b) => b.addEventListener('click', () => { this.o.group = b.dataset.group; this._save(); this.renderList(); }));
    $('#compare-unit').addEventListener('change', (e) => { this.o.unit = e.target.value; this._save(); this.renderList(); });
    $('#compare-list').addEventListener('change', (e) => {
      const ids = e.target.matches('.pick') ? [+e.target.dataset.id] : e.target.matches('.pick-group') ? e.target.dataset.ids.split(',').map(Number) : null;
      if (!ids) return;
      const picked = new Set(this.o.picked);
      for (const id of ids) { if (e.target.checked) picked.add(id); else picked.delete(id); }
      this.o.picked = this.datasets.map((d) => d.id).filter((id) => picked.has(id)).reverse(); // oldest first
      this._changed();
    });
    $('#compare-clear').addEventListener('click', () => { this.o.picked = []; this._changed(); });
    $('#cmp-metric').innerHTML = Object.entries(SIMILARITIES).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join('');
    for (const [id, key, read] of [['#cmp-on', 'on'], ['#cmp-metric', 'metric'], ['#cmp-ref-metric', 'refMetric'], ['#cmp-show', 'show'], ['#cmp-conc', 'conc'],
      ['#cmp-rsa', 'rsa', (el) => el.checked], ['#cmp-rsa-method', 'rsaMethod'], ['#cmp-p-by', 'pBy'],
      ['#cmp-boot-n', 'bootN', (el) => { const n = Math.round(+el.value); return el.value && n > 0 ? Math.min(100000, Math.max(100, n)) : this.o.bootN; }]]) {
      $(id).addEventListener('change', (e) => { this.o[key] = read ? read(e.target) : e.target.value; this._changed(); });
    }
    $('#cmp-pair').addEventListener('change', (e) => {
      if (!e.target.dataset.side) return;
      this.o.pair = [...this.o.pair];
      this.o.pair[+e.target.dataset.side] = e.target.value || null;
      this._changed();
    });
    $('#cmp-spaces').addEventListener('change', (e) => {
      if (e.target.matches('[data-space]')) {
        const key = e.target.dataset.space;
        this.o.spaces = e.target.checked ? [...new Set([...this.o.spaces, key])] : this.o.spaces.filter((k) => k !== key);
      } else if (e.target.matches('#cmp-ecfp-d')) {
        const d = Math.round(+e.target.value / 2) * 2;
        this.o.ecfpD = Math.min(12, Math.max(0, Number.isFinite(d) ? d : 6));
      } else return;
      this._changed();
    });
  }

  _save() { LS.set(COMPARE_KEY, this.o); }

  _changed() { this._save(); this.renderList(); this.render(); }

  // each time the tab is opened: datasets and reference datasets may have come or gone
  async load() {
    let ds, refs;
    try { [ds, refs] = await Promise.all([api('/api/datasets'), api('/api/reference')]); } catch (e) {
      $('#compare-view').replaceChildren(msgBox('error', `Could not load the datasets: ${esc(e.message)}`));
      return;
    }
    this.datasets = ds.datasets;
    this.refs = refs.datasets.filter((r) => r.state === 'ok' && ((r.chemicals?.axis && r.chemicals.matched) || r.glomeruli?.axis));
    const ids = new Set(this.datasets.map((d) => d.id));
    this.o.picked = this.o.picked.filter((id) => ids.has(id));
    this.o.spaces = this.o.spaces.filter((k) => k === 'ecfp' || this.refs.some((r) => `ref:${r.id}` === k));
    this.renderList();
    await this.render();
  }

  invalidate() { this.data.clear(); this.memo.clear(); }

  renderList() {
    $$('#tab-compare .seg button').forEach((b) => b.classList.toggle('active', b.dataset.group === this.o.group));
    const list = $('#compare-list');
    if (!this.datasets) return;
    if (!this.datasets.length) { list.innerHTML = '<div class="empty">Nothing uploaded yet. Use the Upload tab.</div>'; return; }
    const picked = new Set(this.o.picked);
    const item = (d) => `<label class="ds-item${picked.has(d.id) ? ' active' : ''}">
        <input type="checkbox" class="pick" data-id="${d.id}"${picked.has(d.id) ? ' checked' : ''}>
        <span class="list-item">
          <span class="t">${esc(d.driver)} &gt; ${esc(d.indicator)} <span class="muted">(${esc(d.coding_unit)})</span></span>
          <span class="s">${d.n_flies} ${d.n_flies === 1 ? 'fly' : 'flies'}, ${d.n_units.toLocaleString()} ROIs, ${d.n_odors} odors; last fly ${esc(d.last_fly_date)}</span>
          <span class="s">#${d.id}, ${esc(d.uploader_name)}</span>
        </span></label>`;
    const head = (g) => {
      const all = g.datasets.every((d) => picked.has(d.id));
      return `<div class="list-group"><label><input type="checkbox" class="pick-group" data-ids="${g.datasets.map((d) => d.id).join(',')}"${all ? ' checked' : ''}>
        ${esc(g.label)} <span class="muted">(${g.datasets.length})</span></label></div>`;
    };
    list.innerHTML = datasetGroups(filterByUnit(this.datasets, $('#compare-unit'), this.o.unit), this.o.group)
      .map((g) => head(g) + g.datasets.map(item).join('')).join('');
    const n = this.o.picked.length;
    $('#compare-count').textContent = n ? `${n} dataset${n === 1 ? '' : 's'} shown` : '';
    $('#compare-clear').hidden = !n;
  }

  // ------------------------------------------------------------------- spaces
  _memo(key, make) {
    if (!this.memo.has(key)) this.memo.set(key, make());
    return this.memo.get(key);
  }

  // the chemicals (of the odor table) among a dataset's odors, each as the row (or rows)
  // of the concentration chosen: {ids, rows: [[row index]]}
  _chemicals(d) {
    const byId = new Map();
    d.rows.forEach((r, i) => { if (r.odor_id != null) byId.set(r.odor_id, [...(byId.get(r.odor_id) || []), i]); });
    const ids = [...byId.keys()];
    return {
      ids,
      rows: ids.map((id) => {
        const rows = byId.get(id);
        if (this.o.conc === 'mean') return rows;
        const concs = rows.map((i) => d.rows[i].log10_conc ?? -Infinity);
        const want = (this.o.conc === 'max' ? Math.max : Math.min)(...concs);
        return rows.filter((_, k) => concs[k] === want); // in more than one panel: all of them
      }),
    };
  }

  // A space of chemicals: {ids (of the odor table), sim (their similarities, ids × ids),
  // data: {vectors (each chemical's values), cols ({label} of each value), meta (as a
  // reference dataset's: kind, value_label, col_label)}, or null for none worth showing}.
  // A dataset's, by their responses (the mean of their rows, if more than one)
  _datasetSpace(d) {
    return this._memo(`ds:${d.id}:${this.o.metric}:${this.o.conc}`, () => {
      const { ids, rows } = this._chemicals(d), nC = d.cols.length;
      const vectors = rows.map((rs) => Float32Array.from({ length: nC }, (_, c) => {
        let sum = 0, n = 0;
        for (const i of rs) { const v = d.values[i * nC + c]; if (v === v) { sum += v; n++; } }
        return n ? sum / n : NaN;
      }));
      return {
        ids, sim: similarityMatrix(vectors, this.o.metric),
        data: { vectors, cols: d.cols.map((c) => ({ label: c.label })), meta: { kind: 'diverging', col_label: d.kind === 'glomeruli' ? 'glomerulus' : 'ROI' } },
      };
    });
  }

  // a dataset's of glomeruli, by their responses (the mean of the flies with each) to every
  // odor of it, its data its glomeruli × its odors
  _datasetGlomSpace(d) {
    return this._memo(`dsglom:${d.id}:${this.o.metric}:${display.lowerOdors}`, () => {
      const nR = d.rows.length, nC = d.cols.length, showOdor = odorShower(d.rows.map((r) => r.odor));
      const vectors = d.cols.map((_, c) => Float32Array.from({ length: nR }, (_, i) => d.values[i * nC + c]));
      return {
        ids: d.cols.map((c) => c.label), sim: similarityMatrix(vectors, this.o.metric),
        data: { vectors, cols: d.rows.map((r) => ({ label: showOdor(r.odor) })), meta: { kind: 'diverging', col_label: 'odor' } },
      };
    });
  }

  // a reference dataset's, of its chemicals (on 'odors') or its glomeruli (on 'glomeruli'),
  // by their values along its other axis (each once, the first of its rows or columns)
  _refSpace(ref, on = 'odors') {
    return this._memo(`ref:${ref.meta.id}:${on}:${this.o.refMetric}`, () => {
      const axis = on === 'odors' ? ref.chemicals.axis : ref.glomeruli.axis;
      const values = decodeValues(ref.values_b64), nR = ref.rows.length, nC = ref.cols.length, onRows = axis === 'rows';
      const ids = [], vectors = [];
      (onRows ? ref.rows : ref.cols).forEach((it, k) => {
        const id = on === 'odors' ? it.chem?.odor_id : it.glomerulus;
        if (id == null || ids.includes(id)) return;
        ids.push(id);
        vectors.push(onRows ? values.subarray(k * nC, (k + 1) * nC) : Float32Array.from({ length: nR }, (_, r) => values[r * nC + k]));
      });
      const m = ref.meta;
      return {
        ids, sim: similarityMatrix(vectors, this.o.refMetric),
        data: { vectors, cols: (onRows ? ref.cols : ref.rows).map((it) => ({ label: it.label })),
          meta: { kind: m.kind, value_label: m.value_label, col_label: (onRows ? m.col_label : m.row_label) || (onRows ? 'column' : 'row') } },
      };
    });
  }

  // chemical structure's: their fingerprints' bits, not worth showing as data
  _ecfpSpace(fp) {
    return this._memo(`ecfp:${fp.diameter}:${Object.keys(fp.fingerprints).length}`, () => {
      const ids = Object.keys(fp.fingerprints).map(Number);
      return { ids, sim: similarityMatrix(ids.map((id) => fp.fingerprints[id]), null, tanimoto), data: null };
    });
  }

  // an odor of the odor table as a row of a matrix, labelled by its name (the viewer can
  // show its abbreviation instead), from the datasets and reference datasets loaded
  _chemRows(datasets) {
    const chems = new Map();
    for (const d of datasets) for (const r of d.rows || []) if (r.odor_id != null && r.chem) chems.set(r.odor_id, r.chem);
    for (const ref of this.refData.values()) {
      for (const it of [...ref.rows, ...ref.cols]) if (it.chem?.name && !chems.has(it.chem.odor_id)) chems.set(it.chem.odor_id, it.chem);
    }
    return (id) => {
      const c = chems.get(id);
      return c?.name ? { label: c.name, chem: { name: c.name, abbrev: c.abbrev, cid: c.cid } } : { label: `odor ${id}` };
    };
  }

  _spaceTitle(key) {
    if (key === 'ecfp') return `ECFP${this.o.ecfpD} fingerprints`;
    return this.refs.find((r) => `ref:${r.id}` === key)?.title || key;
  }

  // the reference datasets that can be compared to on what the page compares on
  _refsOn() {
    return this.refs.filter((r) => (this.o.on === 'odors' ? r.chemicals?.axis && r.chemicals.matched : r.glomeruli?.axis));
  }

  // the spaces compared to (of those chosen, those of what the page compares on), loaded:
  // [{key, title, space, note, simLabel}]
  async _spaces() {
    const out = [], on = this.o.on, refs = new Set(this._refsOn().map((r) => `ref:${r.id}`));
    for (const key of this.o.spaces.filter((k) => (k === 'ecfp' ? on === 'odors' : refs.has(k)))) {
      try {
        if (key === 'ecfp') {
          const fp = await this._fingerprints();
          out.push({ key, title: this._spaceTitle(key), space: this._ecfpSpace(fp), note: 'their Jaccard similarity (1 − Jaccard distance)',
            simLabel: 'Jaccard similarity' });
        } else {
          const id = key.slice(4);
          if (!this.refData.has(id)) this.refData.set(id, await api(`/api/reference/${encodeURIComponent(id)}`));
          const ref = this.refData.get(id);
          const axis = on === 'odors' ? ref.chemicals.axis : ref.glomeruli.axis;
          const along = axis === 'rows' ? ref.meta.col_label || 'columns' : ref.meta.row_label || 'rows';
          const simLabel = SIMILARITIES[this.o.refMetric].label;
          out.push({ key, title: ref.meta.title, space: this._refSpace(ref, on), note: `by their values across its ${along}, ${simLabel}`, simLabel });
        }
      } catch (e) {
        out.push({ key, title: this._spaceTitle(key), error: e.message });
      }
    }
    return out;
  }

  async _fingerprints() {
    const d = this.o.ecfpD;
    if (!this.ecfp.has(d)) this.ecfp.set(d, await api(`/api/compare/ecfp?diameter=${d}`));
    return this.ecfp.get(d);
  }

  // ------------------------------------------------------------------- render
  // the options: the spaces to compare to, each with how many of the odors (or glomeruli) shown it has
  _renderOptions(shown) {
    const on = this.o.on, N = ON[on];
    $('#cmp-rsa').checked = !!this.o.rsa;
    for (const [id, key] of [['#cmp-on', 'on'], ['#cmp-metric', 'metric'], ['#cmp-ref-metric', 'refMetric'], ['#cmp-show', 'show'], ['#cmp-conc', 'conc'],
      ['#cmp-rsa-method', 'rsaMethod'], ['#cmp-boot-n', 'bootN'], ['#cmp-p-by', 'pBy']]) $(id).value = this.o[key];
    $('#cmp-show option[value=sim]').textContent = `its ${N.sim}`;
    $('#cmp-sim-head').textContent = N.sim[0].toUpperCase() + N.sim.slice(1);
    for (const [id, note] of [['#cmp-show-note', N.show], ['#cmp-sim-note', N.simNote], ['#cmp-match-note', N.match],
      ['#cmp-pair-note', `Two of the spaces shown next to each other, over the ${N.many} both have, in one order, each row level with its other; `
        + 'or click a cell of the correlations between similarities.']]) $(id).textContent = note;
    $('#cmp-conc-opt').hidden = on !== 'odors';
    if (on === 'odors') {
      $('#cmp-shown').textContent = shown.size
        ? `The datasets shown have ${shown.size} odor${shown.size === 1 ? '' : 's'} of the odor table (by themselves, not mixed).`
        : 'No odors of the odor table shown yet.';
    } else {
      $('#cmp-shown').textContent = shown.size ? `The datasets of glomeruli shown have ${shown.size} glomerul${shown.size === 1 ? 'us' : 'i'}.`
        : 'No glomeruli shown yet: tick datasets of glomeruli on the left.';
    }
    const overlap = (ids) => {
      if (!shown.size) return `<span class="overlap">it has ${ids.length} ${N.many}</span>`;
      const n = ids.filter((id) => shown.has(id)).length;
      return `<span class="overlap${n < 3 && this.o.picked.length ? ' few' : ''}">${n} of the ${shown.size} ${N.many} shown${ids.length ? ` (it has ${ids.length})` : ''}</span>`;
    };
    const fp = this.ecfp.get(this.o.ecfpD);
    const checked = (key) => (this.o.spaces.includes(key) ? ' checked' : '');
    const html = on !== 'odors' ? [] : [`<div class="space"><label class="check"><input type="checkbox" data-space="ecfp"${checked('ecfp')}> Chemical structure:
        ECFP fingerprints, by their Jaccard similarity (1 − Jaccard distance)</label>
        <label class="opt">Diameter <input type="number" id="cmp-ecfp-d" min="0" max="12" step="2" value="${this.o.ecfpD}"> (ECFP${this.o.ecfpD})</label>
        ${fp ? overlap(Object.keys(fp.fingerprints).map(Number)) : '<span class="overlap">…</span>'}</div>`];
    const refs = this._refsOn();
    for (const r of refs) {
      html.push(`<div class="space"><label class="check"><input type="checkbox" data-space="ref:${esc(r.id)}"${checked(`ref:${r.id}`)}> ${esc(r.title)}</label>
        ${overlap((on === 'odors' ? r.chemicals.odor_ids : r.glomeruli.names) || [])}</div>`);
    }
    if (!refs.length) html.push(`<p class="muted small">${esc(N.none)}</p>`);
    const active = document.activeElement?.id === 'cmp-ecfp-d';
    $('#cmp-spaces').innerHTML = html.join('');
    if (active) $('#cmp-ecfp-d').focus();
  }

  // the menus of the two spaces side by side, of those shown
  _renderPairOptions(all) {
    const menu = (k) => `<select data-side="${k}" aria-label="${k ? 'Second' : 'First'} space"><option value="">—</option>${all.map((s) =>
      `<option value="${esc(s.key)}"${this.o.pair[k] === s.key ? ' selected' : ''}>${esc(s.title)}</option>`).join('')}</select>`;
    $('#cmp-pair').innerHTML = `<label class="opt">${menu(0)}</label><label class="opt">and ${menu(1)}</label>`;
  }

  // a viewer of the page, kept while it is shown again (by key): on its own (fitted to what
  // it shows) until linked again. cells: its own cell size to start with, table: the CSV its
  // export menu offers (else the matrix as shown), onCellClick: as the heatmap's
  _viewer(key, payload, title, fileStem, { cells = null, table = null, onCellClick = null } = {}) {
    let b = this.blocks.get(key);
    if (!b) {
      const el = document.createElement('div');
      el.className = 'cmp-viewer';
      $('#compare-view').appendChild(el); // attached, to be measured
      b = { el, viewer: new MatrixViewer(el, { kind: 'compare', cells }) };
      $('[data-export=csv]', b.viewer.toolbar).hidden = false;
      this.blocks.set(key, b);
    }
    const v = b.viewer;
    v.tableExport = table;
    v.heatmap.options.onCellClick = onCellClick;
    v.linked = null;
    v.heatmap.mark = null;
    v.heatmap.options.minHeaderH = 0;
    v.toolbar.style.minHeight = '';
    v.onShow = () => fitViewer(b.el, v);
    return { el: b.el, viewer: v, show: () => v.setGenericData(payload, { title, fileStem, kind: 'compare' }) };
  }

  // a block of the page: a viewer under its title
  _block(key, title, sub, payload, fileStem, opts = {}) {
    const b = this._viewer(key, payload, title, fileStem, opts);
    const section = document.createElement('section');
    section.className = 'cmp-block';
    section.innerHTML = `<div class="cmp-head"><h3>${esc(title)}</h3>${sub ? `<span class="muted">${sub}</span>` : ''}</div>`;
    section.appendChild(b.el);
    return { section, show: b.show };
  }

  async render() {
    const seq = ++this._seq, view = $('#compare-view');
    $('#compare-empty').hidden = !!this.o.picked.length;
    await Promise.all(this.o.picked.filter((id) => !this.data.has(id)).map(async (id) => {
      try {
        const d = await api(`/api/compare/datasets/${id}`);
        this.data.set(id, { ...d, values: decodeValues(d.values_b64) });
      } catch (e) { this.data.set(id, { id, error: e.message }); }
    }));
    const on = this.o.on, N = ON[on];
    if (on === 'odors') try { await this._fingerprints(); } catch { /* said so when compared to */ }
    const spaces = await this._spaces();
    if (seq !== this._seq) return;
    const datasets = this.o.picked.map((id) => this.data.get(id));
    // what is compared on, of the datasets shown: their odors of the odor table, or their glomeruli
    const shown = new Set(on === 'odors' ? datasets.flatMap((d) => (d.rows || []).filter((r) => r.odor_id != null).map((r) => r.odor_id))
      : datasets.flatMap((d) => (d.kind === 'glomeruli' ? d.cols.map((c) => c.label) : [])));
    this._renderOptions(shown);
    // every space shown: the datasets by what was recorded (ORNs, PNs, KCs...), by their
    // full names those that would share one, then the spaces compared to
    const metric = SIMILARITIES[this.o.metric].label, show = this.o.show;
    const loaded = datasets.filter((d) => !d.error && (on === 'odors' || d.kind === 'glomeruli'));
    const short = loaded.map(neuralSpaceName);
    const all = [
      ...loaded.map((d, i) => {
        const full = `#${d.id} ${d.title}`;
        return { key: `ds:${d.id}`, title: short.filter((n) => n === short[i]).length > 1 ? full : short[i], fullTitle: full,
          space: on === 'odors' ? this._datasetSpace(d) : this._datasetGlomSpace(d), simLabel: metric };
      }),
      ...spaces.filter((s) => !s.error),
    ];
    this.all = all;
    this._renderPairOptions(all);
    const rowOf = on === 'odors' ? this._chemRows(datasets) : (g) => ({ label: g }), state = this._stateKey();

    const parts = [], shows = [], used = new Set();
    const block = (key, ...args) => { used.add(key); const b = this._block(key, ...args); parts.push(b.section); shows.push(b.show); return b.section; };
    const info = (html) => parts.push(msgBox('info', html));
    // how alike the similarities of each two spaces are
    if (this.o.rsa && all.length < 2) {
      info('Show two datasets, or one and a space to compare it to (on the right), to correlate their similarities.');
    } else if (this.o.rsa) {
      const method = this.o.rsaMethod === 'spearman' ? 'Spearman' : 'Pearson', job = this._rsaP(all);
      const table = () => this._rsaCsv(all, method, job), onCellClick = (cell) => this._openPair(cell);
      // the correlations, and their p-values to the right
      const row = document.createElement('div');
      row.className = 'cmp-row';
      row.append(
        block('rsa', 'Between similarities', `${method} correlation of each two spaces' ${N.one}–${N.one} similarities (${esc(metric)} within datasets,
          ${esc(SIMILARITIES[this.o.refMetric].label)} within reference data), over the pairs of ${N.many} both have. Click a cell to see its two
          spaces side by side.`, ...this._rsa(all, method), { cells: [84, 40], table, onCellClick }),
        block('rsa-p', 'Their p-values', `Two-sided ${job.unit === 'permute'
          ? `permutation p of each correlation against 0 (${job.resamplings.toLocaleString()} permutations of one space's ${N.many}:
            a <a href="https://en.wikipedia.org/wiki/Mantel_test" target="_blank" rel="noopener">Mantel test</a>)`
          : `bootstrap p of each correlation against 0 (${job.resamplings.toLocaleString()} resamplings of the
            ${job.unit === 'odors' ? `${N.many}, with replacement)` : `pairs of ${N.many}, with replacement: these can be far too small, see on the right)`}`},
          <a href="${HOLM_URL}" target="_blank" rel="noopener">Holm–Bonferroni</a> adjusted for the ${job.pairs.length} pair${job.pairs.length === 1 ? '' : 's'}
          of spaces (more on the right); * below 0.05. <span class="cmp-status"></span>`,
        job.payload, 'similarities_correlated_p', { cells: [84, 40], table, onCellClick }));
      parts.splice(-2, 2, row);
      job.status = $('.cmp-status', row);
      this._pStatus(job);
    }
    if (!this.o.rsa || all.length < 2) this._stopP();
    // two of them side by side
    this.pairs = [];
    this.pairRO.disconnect();
    const pair = this._pairSection(all, rowOf, used, shows);
    if (pair) parts.push(pair);
    for (const s of spaces.filter((x) => x.error)) parts.push(msgBox('error', `${esc(s.title)} could not be compared to: ${esc(s.error)}`));
    for (const d of datasets) {
      if (d.error) { parts.push(msgBox('error', `Dataset #${d.id} could not be shown: ${esc(d.error)}`)); continue; }
      if (on === 'glomeruli') {
        if (d.kind !== 'glomeruli') { info(`#${d.id} ${esc(d.title)}: its ROIs are not glomeruli, so it is left out when comparing on glomeruli.`); continue; }
        // its glomeruli (rows) by its odors, or their similarities
        const s = { space: this._datasetGlomSpace(d), simLabel: metric }, ids = s.space.ids;
        if (show !== 'sim') {
          block(`dsg:${d.id}`, `#${d.id} ${d.title}`, `${esc(d.coding_unit || '')}: ${ids.length} glomeruli, each the mean of the flies with it, × ${d.rows.length} odors;
            every trial averaged`, this._memo(`glomdata:${d.id}:${display.lowerOdors}`, () => spaceData(s, ids, rowOf, 'glomerulus')),
          `dataset_${d.id}_mean_by_glomerulus`);
        } else {
          block(`simg:${d.id}`, `#${d.id} ${d.title}: glomerulus–glomerulus similarity`, `${metric} of their responses across its ${d.rows.length} odors`,
            this._memo(`glomsim:${d.id}:${this.o.metric}`, () => spaceSims(s, ids, rowOf, 'glomerulus')), `dataset_${d.id}_glomerulus_similarity`);
        }
        continue;
      }
      const what = d.kind === 'glomeruli' ? `${d.cols.length} glomeruli, each the mean of the flies with it` : `${d.cols.length.toLocaleString()} ROIs`;
      const sub = `${esc(d.coding_unit || '')}: ${d.rows.length} odors × ${what}; every trial averaged`;
      const showOdor = odorShower(d.rows.map((r) => r.odor));
      if (show !== 'sim') {
        block(`ds:${d.id}`, `#${d.id} ${d.title}`, sub, this._memo(`data:${d.id}:${display.lowerOdors}`, () => ({
          meta: { kind: 'diverging', row_label: 'odor', col_label: d.kind === 'glomeruli' ? 'glomerulus' : 'ROI' },
          rows: d.rows.map((r) => ({ label: showOdor(r.odor), group: r.panel })), cols: d.cols.map((c) => ({ label: c.label })), values: d.values,
        })), `dataset_${d.id}_mean`);
      }
      if (show !== 'data') {
        block(`sim:${d.id}`, `#${d.id} ${d.title}: odor–odor similarity`, `${metric} of their responses across its ${d.kind === 'glomeruli' ? 'glomeruli' : 'ROIs'}`,
          this._memo(`simdata:${d.id}:${this.o.metric}:${display.lowerOdors}`, () => {
            const nC = d.cols.length, rows = d.rows.map((_, i) => d.values.subarray(i * nC, (i + 1) * nC));
            const labels = d.rows.map((r) => ({ label: showOdor(r.odor), group: r.panel }));
            return { meta: { kind: 'diverging', row_label: 'odor', col_label: 'odor' }, rows: labels, cols: labels.map((l) => ({ label: l.label })),
              values: similarityMatrix(rows, this.o.metric) };
          }), `dataset_${d.id}_similarity`);
      }
    }
    // each space compared to, over the odors (or glomeruli) shown it has: its data (if it
    // has any worth showing), its similarities
    for (const s of spaces.filter((x) => !x.error)) {
      const has = new Set(s.space.ids), ids = [...shown].filter((id) => has.has(id)), stem = s.key.replace(/\W+/g, '_');
      if (ids.length < 2) { parts.push(msgBox('info', `${esc(s.title)}: ${ids.length ? 'only one' : 'none'} of the ${N.many} shown.`)); continue; }
      const data = s.space.data;
      if (show !== 'sim' && data) {
        const sub = `the ${ids.length} of the ${N.many} shown it has, by its ${data.meta.col_label}${data.meta.value_label ? ` (${data.meta.value_label})` : ''}`;
        block(`spacedata:${s.key}`, s.title, esc(sub), this._memo(`spacevals:${s.key}:${state}:${ids.join(',')}`, () => spaceData(s, ids, rowOf, N.one)),
          `${stem}_data`);
      }
      if (show !== 'data' || !data) {
        block(`space:${s.key}`, `${s.title}: ${N.sim}`, esc(`${s.note}; the ${ids.length} of the ${N.many} shown it has`),
          this._memo(`spacesims:${s.key}:${state}:${ids.join(',')}`, () => spaceSims(s, ids, rowOf, N.one)), `${stem}_similarity`);
      }
    }
    for (const key of [...this.blocks.keys()]) if (!used.has(key)) this.blocks.delete(key);
    view.replaceChildren(...parts);
    shows.forEach((show) => show());
  }

  // what every space's similarities (and so the correlations between them) depend on, but which spaces
  _stateKey() {
    const o = this.o;
    return `${o.on}:${o.metric}:${o.refMetric}:${o.conc}:${o.ecfpD}`;
  }

  // Two spaces side by side (o.pair): the odors (or glomeruli) both have, in one order (that
  // of the first dataset of the two, else of the first), each as its data or its
  // similarities (o.show), each two viewers next to each other linked, so their rows stay
  // level. A section, or null if none is chosen; its viewers' keys added to `used`, their
  // shows to `shows`. rowOf: an odor's (or glomerulus's) row
  _pairSection(all, rowOf, used, shows) {
    const N = ON[this.o.on];
    const [ka, kb] = this.o.pair;
    if (!ka && !kb) return null;
    const section = document.createElement('section');
    section.className = 'cmp-block cmp-pair';
    section.id = 'cmp-pair-view';
    section.innerHTML = `<div class="cmp-head"><h3>Side by side</h3><span class="muted"></span>
      <button type="button" class="btn small" data-role="close">Close</button></div>`;
    $('[data-role=close]', section).addEventListener('click', () => { this.o.pair = [null, null]; this._changed(); });
    const sub = $('.cmp-head .muted', section);
    const a = all.find((s) => s.key === ka), b = all.find((s) => s.key === kb);
    if (!a || !b || a === b) {
      sub.textContent = 'Choose two of the spaces shown (on the right), or click a cell of the correlations between similarities.';
      return section;
    }
    const first = [a, b].find((s) => s.key.startsWith('ds:')) || a, has = new Set((first === a ? b : a).space.ids);
    const ids = first.space.ids.filter((id) => has.has(id));
    if (ids.length < 2) {
      sub.textContent = `${a.title} and ${b.title} have ${ids.length ? `only one ${N.one}` : `no ${N.many}`} in common.`;
      return section;
    }
    sub.textContent = `${a.title} and ${b.title}: the ${ids.length} ${N.many} both have, in the order of ${first.title}. Rows moved or removed,
      cell sizes, transposing and scrolling stay the same in both.`;
    // chemical structure, with no data worth showing, as its similarities either way
    const state = this._stateKey(), row = document.createElement('div');
    row.className = 'cmp-pair-row';
    const [x, y] = [a, b].map((s, k) => {
      const asData = this.o.show === 'data' && !!s.space.data, what = asData ? 'data' : 'similarity', key = `pair:${k}`;
      used.add(key);
      const payload = this._memo(`pair:${what}:${s.key}:${state}:${ids.join(',')}`,
        () => (asData ? spaceData(s, ids, rowOf, N.one) : spaceSims(s, ids, rowOf, N.one)));
      const v = this._viewer(key, payload, `${s.fullTitle || s.title}: ${what}`, `side_by_side_${(s.fullTitle || s.title).replace(/\W+/g, '_')}_${what}`);
      const cap = asData ? `its data${payload.meta.value_label ? ` (${payload.meta.value_label})` : ''}` : `${N.one}–${N.one} ${s.simLabel}`;
      const side = document.createElement('div');
      side.className = 'cmp-pair-side';
      side.innerHTML = `<div class="cmp-pair-cap" title="${esc(`${s.fullTitle || s.title}: ${cap}`)}">${esc(s.title)}: <span class="muted">${esc(cap)}</span></div>`;
      side.appendChild(v.el);
      row.appendChild(side);
      shows.push(v.show);
      return { ...v, what };
    });
    const cols = x.what === 'similarity' && y.what === 'similarity';
    x.viewer.linked = { other: y.viewer, cols };
    y.viewer.linked = { other: x.viewer, cols };
    x.viewer.onShow = y.viewer.onShow = () => levelViewers(x, y);
    this.pairs = [[x, y]];
    section.appendChild(row);
    this.pairWidth = 0;
    this.pairRO.observe(section);
    return section;
  }

  // a cell of the correlations between similarities (or their p-values) clicked: its two spaces side by side
  _openPair({ row, col }) {
    const a = this.all[row.key], b = this.all[col.key];
    if (!a || !b || a === b) return;
    this.o.pair = [a.key, b.key];
    this._save();
    this.render().then(() => $('#cmp-pair-view')?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }

  // how alike the similarities of each two spaces are: [the matrix for its viewer (r, each
  // cell saying it and of how many odors), its file name]
  _rsa(all, method) {
    return [this._memo(`rsa:${this._rsaKey(all)}`, () => {
      const n = all.length, cells = this._rsaCells(all), many = ON[this.o.on].many;
      const values = new Float32Array(n * n);
      cells.forEach((row, i) => row.forEach((c, j) => { values[i * n + j] = c ? c.r : 1; }));
      const labels = all.map((s) => ({ label: s.title }));
      return {
        meta: { kind: 'diverging', row_label: 'space', col_label: 'space', value_label: `${method} r of their similarities` },
        rows: labels, cols: labels, values,
        cellText: (i, j) => { const c = cells[i][j]; return c ? `${c.r === c.r ? c.r.toFixed(2) : '–'}\n${c.nOdors} ${many}` : '1'; },
      };
    }), 'similarities_correlated'];
  }

  // what the correlations between the similarities of spaces `all` are of
  _rsaKey(all) {
    return `${this.o.rsaMethod}:${this._stateKey()}:${all.map((s) => s.key).join(',')}`;
  }

  _rsaCells(all) {
    return this._memo(`rsacells:${this._rsaKey(all)}`,
      () => all.map((a, i) => all.map((b, j) => (i === j ? null : correlateSimilarities(this.o.rsaMethod, a.space, b.space)))));
  }

  // p-values of the correlations of _rsaCells (by bootstrap or permutation, as `pBy`),
  // each two spaces once (row by row, above the diagonal), from a worker
  // (pvalue-worker.js): a job with its matrix
  // (`payload`: −log10 of the adjusted p, filled in and shown again once done), `ps`,
  // `adjusted` (by Holm's method) and `valid` then, and its `progress` until
  _rsaP(all) {
    const o = this.o, key = `rsap:${o.bootN}:${o.pBy}:${this._rsaKey(all)}`;
    if (this.memo.has(key)) return this.memo.get(key);
    this._stopP();
    const n = all.length, pairs = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
    const at = new Map(pairs.map(([i, j], k) => [i * n + j, k]));
    const job = { key, pairs, resamplings: o.bootN, unit: o.pBy, many: ON[o.on].many, progress: 0, ps: null, adjusted: null, valid: null,
      error: null, status: null };
    const values = new Float32Array(n * n).fill(NaN), labels = all.map((s) => ({ label: s.title }));
    job.payload = {
      meta: { kind: 'sequential', row_label: 'space', col_label: 'space', value_label: '−log10 of the Holm-adjusted p',
        value_range: [0, -Math.log10((o.pBy === 'permute' ? 1 : 2) / (o.bootN + 1))] }, // to the smallest p there can be
      rows: labels, cols: labels, values,
      cellText: (i, j) => {
        if (i === j) return '';
        if (!job.ps) return job.error ? 'failed' : '…';
        const k = at.get(Math.min(i, j) * n + Math.max(i, j)), p = job.adjusted[k];
        return p === p ? `p ${fmtP(p)}${p < 0.05 ? ' *' : ''}\nraw ${fmtP(job.ps[k])}` : '–';
      },
    };
    // each test seeded by its spaces, so its p-value is the same whatever else is shown
    const tests = pairs.map(([i, j]) => {
      const t = sharedSimilarities(all[i].space, all[j].space), seed = hashString(`${all[i].key}|${all[j].key}`);
      return o.pBy === 'pairs' ? { seed, xs: t.xs, ys: t.ys } : { seed, n: t.n, subA: t.subA, subB: t.subB };
    });
    const worker = new Worker(new URL('./pvalue-worker.js', import.meta.url), { type: 'module' });
    job.stop = () => worker.terminate();
    worker.onmessage = ({ data }) => {
      if (!data.ps) { job.progress = data.progress; this._pStatus(job); return; }
      Object.assign(job, { ps: data.ps, valid: data.valid, adjusted: holm(data.ps) });
      pairs.forEach(([i, j], k) => { values[i * n + j] = values[j * n + i] = -Math.log10(job.adjusted[k]); });
      worker.terminate();
      this._pDone(job);
    };
    worker.onerror = (e) => { e.preventDefault(); job.error = e.message || 'the worker failed'; worker.terminate(); this._pDone(job); };
    worker.postMessage({ tests, method: o.rsaMethod, resamplings: o.bootN, unit: o.pBy },
      tests.flatMap((t) => (t.xs ? [t.xs.buffer, t.ys.buffer] : [t.subA.buffer, t.subB.buffer])));
    this.memo.set(key, job);
    this.job = job;
    return job;
  }

  // the p-values being computed stopped (as others are wanted, or none), and forgotten
  _stopP() {
    const job = this.job;
    this.job = null;
    if (job && !job.ps && !job.error) { job.stop(); this.memo.delete(job.key); }
  }

  _pDone(job) {
    if (this.job === job) this.job = null;
    const b = this.blocks.get('rsa-p');
    if (b?.viewer.source === job.payload) b.viewer._rebuild();
    this._pStatus(job);
  }

  _pStatus(job) {
    if (!job.status?.isConnected) return;
    job.status.textContent = job.error ? `Could not compute them: ${job.error}` : job.ps ? ''
      : `${job.unit === 'permute' ? 'Permuting' : 'Resampling'}… ${Math.floor(job.progress * 100)}%`;
    job.status.classList.toggle('err', !!job.error);
  }

  // each two spaces once, with the correlation of their similarities and its p-values
  // (once computed), as a CSV table
  _rsaCsv(all, method, job) {
    const cells = this._rsaCells(all), num = (v) => (v != null && v === v ? String(v) : '');
    const lines = [['space_a', 'space_b', 'correlation', 'similarity_a', 'similarity_b', 'r', `n_${job.many}`, 'n_pairs', 'p', 'p_holm', 'p_by', 'resamplings']];
    let k = 0;
    all.forEach((a, i) => all.forEach((b, j) => {
      if (j <= i) return;
      const c = cells[i][j], t = k++;
      lines.push([a.fullTitle || a.title, b.fullTitle || b.title, method, a.simLabel, b.simLabel, num(c.r), c.nOdors, c.nPairs,
        num(job.ps?.[t]), num(job.adjusted?.[t]), pBy(job.unit, job.many), job.valid ? job.valid[t] : '']);
    }));
    return { name: 'similarities_correlated.csv', text: `${lines.map((l) => l.map(csvCell).join(',')).join('\n')}\n` };
  }
}

// ==================================================================== router
const upload = new UploadFlow();
const browse = new Browse();
const odorsPage = new OdorsPage();
const reference = new Reference();
const mergePage = new MergePage();
const comparePage = new ComparePage();
const settings = new SettingsPage();
const admin = new AdminPage();
const player = new MoviePlayer($('#player'), { api, esc, fmtBytes });

let currentTab = null;
let lastPlace = 'browse'; // the hash of the last place shown other than the Admin tab

function route() {
  const [tab, id] = location.hash.replace(/^#/, '').split('/');
  const name = ['upload', 'browse', 'merge', 'compare', 'odors', 'reference', 'settings', 'admin'].includes(tab) ? tab : 'browse';
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
  if (name === 'compare') comparePage.load();
  if (name === 'browse' && !id) { $('#browse-empty').hidden = false; $('#dataset-view').hidden = true; browse.selected = null; }
  if (name === 'reference' && !id) { $('#reference-empty').hidden = false; $('#reference-view').hidden = true; reference.selected = null; }
}

window.addEventListener('hashchange', route);
upload.init().then(route);
api('/api/health').then(() => { $('#topbar-status').textContent = ''; }).catch(() => { $('#topbar-status').textContent = 'server unreachable'; });
api('/api/odors/status').then((s) => setOdorBadge(s.changed)).catch(() => {});
api('/api/reference').then((r) => setAdminBadge(r.datasets)).catch(() => {});
