// Odor response database — frontend (no build step, no dependencies).
import { Heatmap } from './heatmap.js';
import { COLORMAPS, colormapsOfKind, makeNorm, drawColorbar, fmtNum } from './colormaps.js';

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
function buildNeuralModel(data, { mode, rowOrder, roiSort, odorOrder }) {
  const { units, trials, values } = data;
  const nT = trials.length;
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
  const showGroups = panelOrder.length > 1 || panelOrder[0] != null;
  const cols = [], colTrials = [];
  for (const { c } of ordered) {
    const group = showGroups ? (c.panel ?? '(no panel)') : null;
    if (mode === 'trials') {
      const ts = c.trials.slice().sort((a, b) => trials[a].repeat - trials[b].repeat || a - b);
      for (const j of ts) { cols.push({ label: c.odor, group, sub: trials[j].repeat }); colTrials.push([j]); }
    } else {
      cols.push({ label: c.odor, group, sub: null });
      colTrials.push(c.trials);
    }
  }
  const flyKey = (u) => `${u.date} fly ${u.fly_num}`;
  const firstRoi = new Map();
  units.forEach((u, i) => { if (!firstRoi.has(u.roi)) firstRoi.set(u.roi, i); });
  const unitIdx = units.map((u, i) => i);
  const roiCmp = (i, j) => (roiSort === 'sorted' ? cmpRoi(units[i].roi, units[j].roi) : firstRoi.get(units[i].roi) - firstRoi.get(units[j].roi));
  if (rowOrder === 'fly') unitIdx.sort((i, j) => cmpFly(units[i], units[j]) || roiCmp(i, j) || i - j);
  else unitIdx.sort((i, j) => roiCmp(i, j) || cmpFly(units[i], units[j]) || i - j);
  const rows = unitIdx.map((i) => (rowOrder === 'fly'
    ? { label: units[i].roi, group: flyKey(units[i]) }
    : { label: flyKey(units[i]), group: units[i].roi }));
  const nR = rows.length, nC = cols.length;
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
  return {
    rows, cols, values: out,
    rowTitle: rowOrder === 'fly' ? 'ROI' : 'fly',
    rowGroupTitle: rowOrder === 'fly' ? 'fly' : 'ROI',
    nConditions: conds.length, nPanels: panelOrder.length, nUnits: units.length, nTrials: nT,
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
      { mode: 'average', rowOrder: 'fly', roiSort: 'file', odorOrder: 'file', cellW: 16, cellH: 16, colormap: 'oklab-div', range: 'twoslope', clip: false },
      LS.get('olfdb.viewer', {}),
    );
    this.data = null;
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
    $('input[data-opt=clip]', tb).addEventListener('change', (e) => { this.opts.clip = e.target.checked; this._save(); this._rebuild(); });
  }

  _save() { LS.set('olfdb.viewer', { cellW: this.opts.cellW, cellH: this.opts.cellH, colormap: this.opts.colormap, range: this.opts.range, rowOrder: this.opts.rowOrder, roiSort: this.opts.roiSort, odorOrder: this.opts.odorOrder }); }

  _syncToolbar() {
    const tb = this.toolbar, o = this.opts;
    $$('[data-mode]', tb).forEach((b) => b.classList.toggle('active', b.dataset.mode === o.mode));
    const neural = this.kind === 'neural';
    $('[data-role=mode]', tb).hidden = !neural;
    $('[data-role=rows]', tb).hidden = !neural;
    $('[data-role=cols]', tb).hidden = !neural;
    const mapKind = neural ? 'diverging' : (this.data?.meta?.kind || 'sequential');
    const cmSel = $('select[data-opt=colormap]', tb);
    const maps = colormapsOfKind(mapKind);
    cmSel.innerHTML = maps.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join('');
    if (!maps.some((m) => m.id === o.colormap)) o.colormap = maps[0].id;
    cmSel.value = o.colormap;
    $('[data-role=range]', tb).hidden = mapKind !== 'diverging';
    $('input[data-opt=clip]', tb).parentElement.hidden = mapKind === 'binary';
    for (const k of ['rowOrder', 'roiSort', 'odorOrder', 'range']) { const s = $(`select[data-opt=${k}]`, tb); if (s) s.value = o[k]; }
    for (const k of ['cellW', 'cellH']) { $(`input[data-opt=${k}]`, tb).value = o[k]; $(`[data-out=${k}]`, tb).textContent = o[k]; }
    $('input[data-opt=clip]', tb).checked = !!o.clip;
  }

  setNeuralData(payload) {
    this.kind = 'neural';
    this.data = { units: payload.units, trials: payload.trials, values: decodeValues(payload.values_b64) };
    this.opts.colormap = this.opts.colormap in COLORMAPS && COLORMAPS[this.opts.colormap].kind === 'diverging' ? this.opts.colormap : 'oklab-div';
    this._syncToolbar();
    this._rebuild();
  }

  setGenericData(ref) {
    this.kind = 'generic';
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

  _rebuild() {
    if (!this.data) return;
    const o = this.opts;
    let model;
    if (this.kind === 'neural') {
      model = buildNeuralModel(this.data, o);
    } else {
      model = this.data.model;
      if (COLORMAPS[o.colormap]?.kind === 'sequential') { this.opts.colormapSeq = o.colormap; this._save(); }
    }
    const cm = COLORMAPS[o.colormap] || COLORMAPS['oklab-div'];
    const { vmin, vmax } = dataRange(model.values, o.clip);
    let norm;
    if (cm.kind === 'diverging') norm = makeNorm(o.range === 'symmetric' ? 'symmetric' : 'twoslope', vmin, vmax);
    else if (cm.kind === 'binary') norm = makeNorm('linear', 0, 1);
    else norm = makeNorm('linear', vmin, vmax);
    this.model = model;
    this.norm = norm;
    this.heatmap.setModel(model, { cellW: o.cellW, cellH: o.cellH, lut: cm.lut, norm });
    drawColorbar(this.colorbar, cm.lut, norm);
    const stats = $('[data-role=stats]', this.toolbar);
    if (this.kind === 'neural') {
      const m = model;
      stats.textContent = `${m.rows.length.toLocaleString()} ROIs × ${m.cols.length} ${o.mode === 'trials' ? 'trials' : 'odors'}` +
        (m.nPanels > 1 ? ` in ${m.nPanels} panels` : '') + ` · range ${fmtNum(vmin)} to ${fmtNum(vmax)}`;
    } else {
      stats.textContent = `${model.rows.length} × ${model.cols.length} · range ${fmtNum(vmin)} to ${fmtNum(vmax)}` + (this.data.meta.value_label ? ` (${this.data.meta.value_label})` : '');
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
    this.viewer.setNeuralData(res.data);

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
    this.viewer = null;
    this.cache = new Map();
    $$('#tab-browse .seg button').forEach((b) => b.addEventListener('click', () => { this.group = b.dataset.group; LS.set('olfdb.browse_group', this.group); this.renderList(); }));
    $('#ds-readme-toggle').addEventListener('click', () => {
      const p = $('#ds-readme');
      p.hidden = !p.hidden;
      $('#ds-readme-toggle').setAttribute('aria-expanded', String(!p.hidden));
    });
  }

  invalidate() { this.datasets = null; this.cache.clear(); }

  async load() {
    if (!this.datasets) {
      try { this.datasets = (await api('/api/datasets')).datasets; } catch (e) { $('#dataset-list').replaceChildren(msgBox('error', esc(e.message))); return; }
    }
    this.renderList();
  }

  renderList() {
    $$('#tab-browse .seg button').forEach((b) => b.classList.toggle('active', b.dataset.group === this.group));
    const list = $('#dataset-list');
    if (!this.datasets) return;
    if (!this.datasets.length) { list.innerHTML = '<div class="empty">Nothing uploaded yet. Use the Upload tab.</div>'; return; }
    const item = (d) => `<button type="button" class="list-item${this.selected === d.id ? ' active' : ''}" data-id="${d.id}">
        <span class="t">${esc(d.driver)} &gt; ${esc(d.indicator)} <span class="muted">(${esc(d.coding_unit)})</span></span>
        <span class="s">${d.n_flies} ${d.n_flies === 1 ? 'fly' : 'flies'}, ${d.n_units.toLocaleString()} ROIs, ${d.n_odors} odors; last fly ${esc(d.last_fly_date)}</span>
        <span class="s">#${d.id}, ${esc(d.uploader_name)}${d.recording_system ? ', ' + esc(d.recording_system) : ''}</span>
      </button>`;
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
        html += `<div class="list-group">${esc(k)} <span class="muted">(${ds.length})</span></div>` + ds.map(item).join('');
      }
    } else {
      const ds = this.datasets.slice().sort((a, b) => (b.last_fly_date || '').localeCompare(a.last_fly_date || '') || b.id - a.id);
      let last = null;
      for (const d of ds) {
        const month = (d.last_fly_date || '').slice(0, 7) || 'unknown date';
        if (month !== last) { html += `<div class="list-group">${esc(month)}</div>`; last = month; }
        html += item(d);
      }
    }
    list.innerHTML = html;
    $$('.list-item', list).forEach((b) => b.addEventListener('click', () => { location.hash = `#browse/${b.dataset.id}`; }));
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
    this.viewer.setNeuralData(m);
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
    this.viewer.setGenericData(ref);
  }
}

// ==================================================================== router
const upload = new UploadFlow();
const browse = new Browse();
const reference = new Reference();

function route() {
  const [tab, id] = location.hash.replace(/^#/, '').split('/');
  const name = ['upload', 'browse', 'reference'].includes(tab) ? tab : 'browse';
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  $$('.tabs a').forEach((a) => a.classList.toggle('active', a.dataset.tab === name));
  if (name === 'browse') { browse.load().then(() => { if (id) browse.show(+id); }); }
  if (name === 'reference') { reference.load().then(() => { if (id) reference.show(id); }); }
  if (name === 'browse' && !id) { $('#browse-empty').hidden = false; $('#dataset-view').hidden = true; browse.selected = null; }
  if (name === 'reference' && !id) { $('#reference-empty').hidden = false; $('#reference-view').hidden = true; reference.selected = null; }
}

window.addEventListener('hashchange', route);
upload.init().then(route);
api('/api/health').then(() => { $('#topbar-status').textContent = ''; }).catch(() => { $('#topbar-status').textContent = 'server unreachable'; });
