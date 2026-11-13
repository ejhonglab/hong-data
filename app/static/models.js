// The Models tab (model_data.py): the models chosen as baselines in the Admin tab first, in a table
// of their own (quick to load: only their experiments' folders are read), then every model of the
// folders of model data added there, by its parameters, in a table sorted by any of them, whose
// rows and columns are drawn as they come into view (hundreds of models by hundreds of
// parameters, the archive's, would be a million cells); and a model run opened below them, its KCs' spike counts
// to odors in a viewer (KCs that never spike left out, unless chosen), to compare with datasets
// in the Compare tab.

const KEY = 'olfdb.models'; // {hideSame, allKeys, sort: {col, dir}}: kept in this browser
const EXP = ':experiment'; // the column of a model's experiment (its folder), before its parameters
const ROW_H = 29; // a row's height (px) until it has been drawn, with its runs on one line
const OVERSCAN = 20; // rows drawn above and below those in view
const COL_OVERSCAN = 8; // columns drawn left and right of those in view
const MAX_CH = 26; // a column's width at most (characters): longer values are cut short (all of them on hover)

// a cell's value as shown: numbers to 6 significant digits (all of them on hover)
function fmtValue(v) {
  if (v == null) return '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return String(Number(v.toPrecision(6)));
  return String(v);
}

// two cells' values in order: missing last, then numbers, booleans and text
function cmpValues(a, b) {
  if (a == null || b == null) return a == null && b == null ? 0 : a == null ? 1 : -1;
  const rank = (v) => (typeof v === 'number' ? 0 : typeof v === 'boolean' ? 1 : 2);
  if (rank(a) !== rank(b)) return rank(a) - rank(b);
  if (typeof a === 'string') return a.localeCompare(b, undefined, { numeric: true });
  return a < b ? -1 : a > b ? 1 : 0;
}

// the characters a value takes as shown, about
const valueChars = (v) => (typeof v === 'number' ? 9 : typeof v === 'boolean' ? 5 : String(v).length);

// a listing's experiments' params.csv columns, for looking them up (once per listing)
function prep(d) {
  if (d.paramCols) return;
  d.paramCols = new Set();
  for (const e of Object.values(d.experiments)) {
    e.paramSet = new Set(e.params);
    e.params.forEach((c) => d.paramCols.add(c));
  }
}

export class ModelsPage {
  // helpers: { $, esc, api, msgBox, fmtWhen, LS, decodeValues, fitViewer, MatrixViewer (class),
  // download(url, status, {json}), wireMenu(menu, onPick), onCompare(model: {id, title, filter, nSpikes, run}) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/models
    this.base = null; // as from /api/models/baselines
    this.jobs = null; // the models of baseline runs on datasets, done (model_runs.js: /api/models/runs' view), listed as base's
    this.loadedAt = 0;
    this._loading = null; // the load under way
    this.runs = new Map(); // run id -> as from /api/models/run
    this.shown = new Map(); // `${run id}|${filter}` -> the matrix shown (kept, for its edits to stay)
    this.current = null; // the run shown: {id, model, run, label}
    this.openId = null; // the run opened (its row marked in the tables), shown once loaded
    this._seq = 0;
    this.viewer = null;
    this.o = { hideSame: false, allKeys: false, sort: { col: null, dir: 1 }, ...h.LS.get(KEY, {}) };
    this.find = ''; // what the rows shown have, as typed
    this.heights = new Map(); // a model's row height as drawn (px), by its index in data.models
    this.widths = new Map(); // a column's width (characters), of data
    this.view = null; // the table of every model as laid out: {cols, order, lefts, first, last, c0, c1} (the rows and columns drawn)
    this.filter = { spiking: true, responding: false };
    const { $ } = h;
    $('#models-hide-same').checked = !!this.o.hideSame;
    $('#models-hide-same').addEventListener('change', (e) => { this.o.hideSame = e.target.checked; this._save(); this.render(); });
    $('#models-all-keys').checked = !!this.o.allKeys;
    $('#models-all-keys').addEventListener('change', (e) => { this.o.allKeys = e.target.checked; this._save(); this.render(); });
    let findTimer = null;
    $('#models-find').addEventListener('input', (e) => {
      clearTimeout(findTimer);
      findTimer = setTimeout(() => { this.find = e.target.value; this.render(); }, 200);
    });
    for (const id of ['#models-table', '#models-base-table']) {
      $(id).addEventListener('click', (e) => {
        const th = e.target.closest('[data-sort]');
        if (th) {
          const col = th.dataset.sort, s = this.o.sort;
          this.o.sort = { col, dir: s.col === col ? -s.dir : 1 };
          this._save();
          this.render();
          return;
        }
        const b = e.target.closest('[data-run]');
        if (b) location.hash = `#models/${encodeURIComponent(b.dataset.run)}`;
        const x = e.target.closest('[data-exp]');
        if (x) this.showExperiment(id === '#models-table' ? this.data : this.base, x.dataset.exp);
      });
    }
    let frame = 0;
    const redraw = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; this._draw(); }); };
    $('#models-table').addEventListener('scroll', redraw, { passive: true });
    window.addEventListener('resize', redraw);
    $('#model-spiking').addEventListener('change', (e) => { this.filter.spiking = e.target.checked; this._showRun(); });
    $('#model-responding').addEventListener('change', (e) => { this.filter.responding = e.target.checked; this._showRun(); });
    $('#model-compare').addEventListener('click', () => this._compare());
    h.wireMenu($('#model-download'), ({ format }) => this._download(format));
  }

  _save() { this.h.LS.set(KEY, { hideSame: this.o.hideSame, allKeys: this.o.allKeys, sort: this.o.sort }); }

  // after folders of model data, or baselines, were added, changed or deleted (Admin tab): read again when next shown
  invalidate() { this.data = null; this.base = null; this.runs.clear(); this.shown.clear(); }

  // the tab opened with no run: none shown
  close() {
    this._seq++;
    this.openId = null;
    this.current = null;
    this.h.$('#model-view').hidden = true;
    this._mark();
  }

  // each time the tab is opened (models may have been added to the folders since), but not
  // again within a few seconds (as when a run is opened); the baselines shown as soon as they come
  load() {
    if (this.data && Date.now() - this.loadedAt < 10e3) return Promise.resolve();
    if (!this._loading) this._loading = this._load().finally(() => { this._loading = null; });
    return this._loading;
  }

  async _load() {
    const { $, api, msgBox, esc } = this.h, t0 = Date.now(), what = $('#models-loading-what');
    const tell = () => {
      const s = Math.round((Date.now() - t0) / 1000);
      what.textContent = `Reading the folders of model data on tensor${this.base ? ' for every model' : ''}…${s >= 2 ? ` ${s} s` : ''}`;
    };
    $('#models-loading').hidden = false;
    tell();
    const timer = setInterval(tell, 1000);
    const base = api('/api/models/baselines').then((b) => { this.base = b; this.renderBase(); tell(); }, (e) => {
      $('#models-base-step').hidden = false;
      $('#models-base-status').replaceChildren(msgBox('error', `Could not load the baseline models: ${esc(e.message)}`));
    });
    try {
      const data = await api('/api/models');
      this.data = data;
      this.heights.clear();
      this.widths.clear();
      this.loadedAt = Date.now();
      this.render();
    } catch (e) {
      $('#models-status').replaceChildren(msgBox('error', `Could not load the models: ${esc(e.message)}`));
    }
    await base;
    clearInterval(timer);
    $('#models-loading').hidden = true;
  }

  // a model's value of a parameter: its row's, if its params.csv has the column, else its
  // experiment's provenance.json's (of it alone, or of all its experiment's models)
  _value(d, m, col) {
    const e = d.experiments[m.experiment];
    if (col === EXP) return e.folder || '(top)';
    return (e.paramSet.has(col) ? m.params[col] : m.own?.[col] ?? e.flat[col]) ?? null;
  }

  // what a run of a model is called: its experiment's folder, its tag (or variant) after what its
  // file's name has before that (e.g. its base: models of two bases may share a tag), and its condition
  _title(d, m, r) {
    const e = d.experiments[m.experiment], name = [r.prefix, m.params.tag ?? m.params.variant].filter((x) => x != null && x !== '');
    return `${e.folder || this._source(e.source)?.path || 'model'}: ${name.join('_')}, ${r.label || 'spike counts'}`;
  }

  _source(id) { return this.data?.sources.find((s) => s.id === id); }

  // whether the models that have a parameter differ in it
  _differ(d, models, c) {
    let first;
    for (const m of models) {
      const v = this._value(d, m, c);
      if (v == null) continue;
      if (first === undefined) first = v;
      else if (v !== first) return true;
    }
    return false;
  }

  // the parameters shown: params.csv's (and provenance.json's keys of the same names), or (as
  // chosen) all of provenance.json's too; all, or (as chosen) those whose models that have them differ
  _columns() {
    const d = this.data;
    let cols = this.o.allKeys ? d.columns : d.columns.filter((c) => c === 'generated' || d.paramCols.has(c));
    if (this.o.hideSame) cols = cols.filter((c) => this._differ(d, d.models, c));
    return cols;
  }

  _cell(d, m, c) {
    const { esc } = this.h, v = this._value(d, m, c);
    if (c === EXP) {
      return `<td><button type="button" class="linkish" data-exp="${esc(m.experiment)}" title="Show ${esc(v)}'s provenance.json (what made its models)">${esc(v)}</button></td>`;
    }
    if (v == null) return '<td></td>';
    const text = c === 'generated' && typeof v === 'string' ? this.h.fmtWhen(v) : fmtValue(v);
    return `<td${typeof v === 'number' ? ' class="num"' : ''} title="${esc(`${c === EXP ? 'experiment' : c}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)}">${esc(text)}</td>`;
  }

  // a model's row: its runs, and its values of cols (of those from c0 to c1 alone, if given: the
  // others stood in for by a cell as wide)
  _row(d, m, cols, i = null, c0 = 0, c1 = cols.length - 1) {
    const { esc } = this.h, open = this.openId;
    const runs = m.runs.length ? m.runs.map((r) => `<button type="button" class="btn small${r.id === open ? ' primary' : ''}" data-run="${esc(r.id)}"
      title="${esc(`Open ${r.file}`)}">${esc(r.label || 'spike counts')}</button>`).join('')
      : '<span class="muted small" title="No spike count file of it was found">none</span>';
    const cells = cols.slice(c0, c1 + 1).map((c) => this._cell(d, m, c)).join('');
    return `<tr${i != null ? ` data-i="${i}"` : ''}${m.runs.some((r) => r.id === open) ? ' class="hl"' : ''}><td class="model-runs">${runs}</td>`
      + `${c0 > 0 ? `<td colspan="${c0}"></td>` : ''}${cells}${c1 < cols.length - 1 ? `<td colspan="${cols.length - 1 - c1}"></td>` : ''}</tr>`;
  }

  _head(cols, sortable) {
    const { esc } = this.h, s = this.o.sort;
    return cols.map((c) => {
      const name = c === EXP ? 'experiment' : c;
      if (!sortable) return `<th title="${esc(name)}"><span class="th-plain">${esc(name)}</span></th>`;
      const on = s.col === c;
      return `<th aria-sort="${on ? (s.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" class="th-sort${on ? ' on' : ''}" data-sort="${esc(c)}"
        title="Sort by ${esc(name)}${on ? (s.dir > 0 ? ' (again: descending)' : ' (again: ascending)') : ''}">${esc(name)}${on ? (s.dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`;
    }).join('');
  }

  // the baselines' table: their params.csv's parameters that they have (few, and few differ)
  renderBase() {
    const { $, esc, msgBox } = this.h, b = this.base;
    if (!b) return;
    prep(b);
    $('#models-base-step').hidden = !b.baselines.length;
    const named = (x) => `<code>${esc(x.name ?? `row ${x.row + 1}`)}</code> of <code>${esc(x.experiment)}</code>`;
    $('#models-base-status').replaceChildren(...b.baselines.filter((x) => !x.ok).map((x) => msgBox('warn', `${named(x)} is not shown: ${esc(x.error)}.`)),
      ...b.baselines.filter((x) => x.ok && x.drift?.length).map((x) => msgBox('warn', `${named(x)} is shown as imported into olfdb
        (${esc(this.h.fmtWhen(x.imported_at))}). Its files on tensor differ since: ${esc(x.drift.join('; '))}. To show them instead, import it
        again in the <a href="#admin">Admin tab</a>.`)),
      ...b.baselines.filter((x) => x.ok && !x.imported_at).map((x) => msgBox('info', `${named(x)} is read on tensor: its files are not imported
        into olfdb, so not backed up (<a href="#admin">Admin tab</a>: Import).`)));
    $('#models-base-count').textContent = b.models.length > 1 ? `${b.models.length}` : '';
    if (!b.models.length) { $('#models-base-table').replaceChildren(); return; }
    const cols = [EXP, ...b.columns.filter((c) => (c === 'generated' || b.paramCols.has(c)) && b.models.some((m) => this._value(b, m, c) != null))];
    $('#models-base-table').innerHTML = `<table class="odor-table models-table"><thead><tr><th>Open</th>${this._head(cols, false)}</tr></thead>
      <tbody>${b.models.map((m) => this._row(b, m, cols)).join('')}</tbody></table>`;
  }

  // a column's width in characters: its name's, or its longest value's, up to MAX_CH (each value
  // seen once, for every column, when first needed)
  _width(c) {
    const d = this.data, w = this.widths;
    if (!w.size) {
      const see = (k, v) => { if (v != null) { const n = valueChars(v); if (!(w.get(k) >= n)) w.set(k, n); } };
      for (const e of Object.values(d.experiments)) {
        for (const k in e.flat) see(k, e.flat[k]);
        see(EXP, e.folder || '(top)');
      }
      for (const m of d.models) {
        for (const k in m.params) see(k, m.params[k]);
        for (const k in m.own || {}) see(k, m.own[k]);
      }
      w.set('generated', 16);
    }
    return Math.min(Math.max((c === EXP ? 10 : c.length) + 2, w.get(c) ?? 0), MAX_CH);
  }

  // the table of every model: laid out with its columns' widths set, its rows drawn as they come into view (`_draw`)
  render() {
    const { $, esc, msgBox } = this.h, d = this.data;
    if (!d) return;
    prep(d);
    const status = [];
    if (!d.sources.length) {
      status.push(msgBox('info', 'No model data yet: folders of it on tensor are added in the <a href="#admin">Admin tab</a> (under <i>Model data</i>).'));
    }
    for (const s of d.sources.filter((x) => !x.ok)) status.push(msgBox('error', `<code>${esc(s.path)}</code> cannot be read: ${esc(s.error)}`));
    const ok = d.sources.filter((s) => s.ok);
    const nExp = Object.keys(d.experiments).length, nRuns = d.models.reduce((n, m) => n + m.runs.length, 0);
    $('#models-count').textContent = ok.length ? `${d.models.length.toLocaleString()} models of ${nExp} experiments, ${nRuns.toLocaleString()} runs` : '';
    const notes = [];
    if (ok.length) notes.push(`From ${ok.map((s) => `<code>${esc(s.path)}</code>`).join(', ')} on tensor (read there each time this tab is opened).`);
    const broken = Object.values(d.experiments).filter((e) => e.error);
    if (broken.length) notes.push(`<span class="warn">${broken.map((e) => `${esc(e.folder || '(top)')}: ${esc(e.error)}`).join('; ')}.</span>`);
    if (d.unmatched.length) {
      notes.push(`${d.unmatched.length.toLocaleString()} spike count file${d.unmatched.length === 1 ? ' is' : 's are'} of no model of a params.csv
        <span class="muted" title="${esc(d.unmatched.join('\n'))}">(hover to see which)</span>.`);
    }
    const skipped = ok.reduce((n, s) => n + (s.scan.skipped || 0), 0);
    if (skipped) notes.push(`${skipped.toLocaleString()} other spike count file${skipped === 1 ? ' is' : 's are'} not models, as the patterns set for their folder in the Admin tab say.`);
    if (notes.length) status.push(Object.assign(document.createElement('p'), { className: 'muted small', innerHTML: notes.join(' ') }));
    $('#models-status').replaceChildren(...status);
    $('#models-table-step').hidden = !d.models.length;
    if (!d.models.length) return;
    const params = this._columns(), cols = [EXP, ...params], s = this.o.sort;
    const nParams = d.columns.filter((c) => c === 'generated' || d.paramCols.has(c)).length;
    $('#models-all-n').textContent = `(${(d.columns.length - nParams).toLocaleString()} more)`;
    let order = d.models.map((m, i) => i);
    const q = this.find.trim().toLowerCase();
    if (q) {
      order = order.filter((i) => {
        const m = d.models[i];
        return cols.some((c) => { const v = this._value(d, m, c); return v != null && fmtValue(v).toLowerCase().includes(q); })
          || m.runs.some((r) => (r.label || '').toLowerCase().includes(q));
      });
    }
    if (s.col && cols.includes(s.col)) {
      const v = d.models.map((m) => this._value(d, m, s.col));
      order.sort((i, j) => (v[i] == null || v[j] == null ? cmpValues(v[i], v[j]) : s.dir * cmpValues(v[i], v[j])) || i - j);
    }
    const all = this.o.allKeys ? d.columns.length : nParams;
    $('#models-shown').textContent = `${params.length < all ? `${params.length.toLocaleString()} of ` : ''}${all.toLocaleString()} parameters`
      + (q ? ` · ${order.length.toLocaleString()} of ${d.models.length.toLocaleString()} models found` : '');
    const widths = cols.map((c) => this._width(c));
    const colTags = widths.map((w) => `<col style="width: calc(${w}ch + 16px)">`).join('');
    const width = `calc(15em + ${widths.reduce((a, b) => a + b, 0)}ch + ${16 * cols.length}px)`;
    const wrap = $('#models-table');
    wrap.innerHTML = `<table class="odor-table models-table models-virtual" style="width: ${width}">
      <colgroup><col style="width: 15em">${colTags}</colgroup>
      <thead><tr><th>Open</th>${this._head(cols, true)}</tr></thead><tbody></tbody></table>`
      + (order.length ? '' : '<p class="muted models-none">No model has that.</p>');
    // where each column starts (px from the table's left; and where the last ends)
    const ths = [...wrap.querySelectorAll('thead th')].slice(1);
    const lefts = Float64Array.from([...ths.map((th) => th.offsetLeft), wrap.querySelector('table').offsetWidth]);
    this.view = { cols, order, lefts, first: -1, last: -1, c0: -1, c1: -1 };
    this._draw(true);
  }

  // a model's row height (px) until drawn: a line of its runs per two of them
  _guess(i) { return ROW_H + 24 * Math.max(0, Math.ceil(this.data.models[i].runs.length / 2) - 1); }

  // the rows and columns in view drawn (and some either side), the others stood in for by the
  // space they take; drawn again only once those in view come near the edge of those drawn
  _draw(force = false) {
    const v = this.view, wrap = this.h.$('#models-table'), d = this.data;
    const tbody = wrap.querySelector('tbody');
    if (!v || !tbody) return;
    const n = v.order.length, tops = new Float64Array(n + 1);
    for (let k = 0; k < n; k++) tops[k + 1] = tops[k] + (this.heights.get(v.order[k]) ?? this._guess(v.order[k]));
    const head = wrap.querySelector('thead')?.offsetHeight || 0, y = Math.max(0, wrap.scrollTop - head);
    let r0 = 0, hi = n; // the first row whose bottom is below the top of the view
    while (r0 < hi) { const mid = (r0 + hi) >> 1; if (tops[mid + 1] <= y) r0 = mid + 1; else hi = mid; }
    let r1 = r0;
    while (r1 < n - 1 && tops[r1 + 1] < y + wrap.clientHeight) r1++;
    // (the columns of parameters, right of the runs' kept in view at the left)
    const L = v.lefts, nc = v.cols.length, x = wrap.scrollLeft;
    let c0 = 0;
    while (c0 < nc - 1 && L[c0 + 1] <= x + L[0]) c0++;
    let c1 = c0;
    while (c1 < nc - 1 && L[c1 + 1] < x + wrap.clientWidth) c1++;
    if (!force && v.first <= Math.max(0, r0 - 5) && v.last >= Math.min(n - 1, r1 + 5)
        && v.c0 <= Math.max(0, c0 - 2) && v.c1 >= Math.min(nc - 1, c1 + 2)) return;
    const first = Math.max(0, r0 - OVERSCAN), last = Math.min(n - 1, r1 + OVERSCAN);
    Object.assign(v, { first, last, c0: Math.max(0, c0 - COL_OVERSCAN), c1: Math.min(nc - 1, c1 + COL_OVERSCAN) });
    const span = nc + 1, rows = [];
    for (let k = first; k <= last; k++) rows.push(this._row(d, d.models[v.order[k]], v.cols, v.order[k], v.c0, v.c1));
    const pad = (h) => `<tr class="pad" aria-hidden="true"><td colspan="${span}" style="height: ${h}px"></td></tr>`;
    tbody.innerHTML = n ? `${pad(tops[first])}${rows.join('')}${pad(tops[n] - tops[last + 1])}` : '';
    for (const tr of tbody.querySelectorAll('tr[data-i]')) this.heights.set(+tr.dataset.i, tr.getBoundingClientRect().height);
  }

  // the run opened marked in the tables
  _mark() {
    if (this.view) this._draw(true);
    this.renderBase();
  }

  // an experiment's provenance.json, shown once (its keys, but those of one of its models alone,
  // are of all its models: by default not shown as columns), and its fits' own provenance files
  showExperiment(d, key) {
    const { $, esc, fmtWhen } = this.h, e = d?.experiments[key];
    if (!e) return;
    const models = (this.data?.experiments[key] ? this.data : d).models.filter((m) => m.experiment === key);
    const src = this._source(e.source), name = e.folder || '(top)';
    const attached = new Set(models.map((m) => m.fit_provenance).filter(Boolean));
    const others = (e.fit_provenance || []).filter((f) => !attached.has(f));
    const keys = Object.entries(e.flat);
    $('#models-exp-title').textContent = `Experiment ${name}`;
    $('#models-exp-body').innerHTML = `<p class="muted small">${src ? `<code>${esc(src.path)}${e.folder ? `/${esc(e.folder)}` : ''}</code> on tensor: ` : ''}
        ${models.length.toLocaleString()} model${models.length === 1 ? '' : 's'}. Below, the keys of its <code>provenance.json</code>, which are of all its models
        (those under one model's tag, variant or base are of that model alone: with it, as <code>…-&gt;*-&gt;…</code>, when the table shows
        <code>provenance.json</code>'s other keys, and when one of its runs is opened).${attached.size ? ` ${attached.size} of its models have
        a <code>provenance_&lt;tag&gt;.json</code> of their own (their fit's), shown when one of their runs is opened.` : ''}${others.length
        ? ` Of no tag of its: ${others.map((f) => `<code>${esc(f)}</code>`).join(', ')}.` : ''}</p>`
      + (keys.length ? `<div class="models-exp-wrap"><table class="odor-table models-exp-table"><tbody>${keys.map(([k, v]) => `<tr><th scope="row"><code>${esc(k)}</code></th>
          <td>${esc(k === 'generated' && typeof v === 'string' ? fmtWhen(v) : fmtValue(v))}</td></tr>`).join('')}</tbody></table></div>`
        : `<p class="muted">${e.error ? esc(e.error) : 'It has no <code>provenance.json</code>.'}</p>`);
    $('#models-exp').hidden = false;
    $('#models-exp').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // the done baseline runs on datasets, as model_runs.js has them (their spike counts opened as any model's)
  setJobs(view) {
    prep(view);
    this.jobs = view;
  }

  // a run's spike counts read again when next opened (a baseline run on datasets deleted, or done again)
  forget(prefix) {
    for (const m of [this.runs, this.shown]) for (const k of [...m.keys()]) if (k.startsWith(prefix)) m.delete(k);
  }

  // the model of a run, and the listing it is in (of every model, else of the baselines, else of the runs on datasets)
  _find(id) {
    for (const d of [this.data, this.base, this.jobs]) {
      const m = d?.models.find((x) => x.runs.some((r) => r.id === id));
      if (m) return { d, m };
    }
    return null;
  }

  // a run opened (from a table, or a link to it): its spike counts shown below the tables
  async show(id) {
    const { $, api, msgBox, esc } = this.h, seq = ++this._seq;
    const found = this._find(id);
    this.openId = found ? id : null;
    $('#model-view').hidden = false;
    if (!found) {
      $('#model-title').textContent = 'No such model run';
      $('#model-msgs').replaceChildren(msgBox('error', `<code>${esc(id)}</code> is not a run of the models listed (any more).`));
      $('#model-body').hidden = true;
      return;
    }
    const { d, m: model } = found, r = model.runs.find((x) => x.id === id), title = this._title(d, model, r);
    $('#model-title').textContent = title;
    $('#model-msgs').replaceChildren();
    $('#model-body').hidden = false;
    this._mark();
    if (!this.runs.has(id)) {
      $('#model-sub').textContent = 'Loading…';
      try { this.runs.set(id, await api(`/api/models/run?id=${encodeURIComponent(id)}`)); } catch (e) {
        $('#model-msgs').replaceChildren(msgBox('error', `Could not load it: ${esc(e.message)}`));
        $('#model-body').hidden = true;
        return;
      }
    }
    if (seq !== this._seq) return; // another opened since
    const run = this.runs.get(id);
    this.current = { id, model, run, title };
    if (run.imported?.drift?.length) {
      $('#model-msgs').replaceChildren(msgBox('warn', `Shown as imported into olfdb (${esc(this.h.fmtWhen(run.imported.at))}): its model's files on tensor
        differ since: ${esc(run.imported.drift.join('; '))}.`));
    }
    const prov = run.provenance;
    $('#model-prov-box').hidden = !prov;
    $('#model-prov').textContent = prov ? JSON.stringify(prov, null, 2) : '';
    const fit = run.fit_provenance;
    $('#model-fit-prov-box').hidden = !fit;
    $('#model-fit-prov-name').textContent = fit?.file ?? '';
    $('#model-fit-prov').textContent = fit ? (fit.provenance ? JSON.stringify(fit.provenance, null, 2) : fit.error) : '';
    const n = model.n_spikes;
    $('#model-n-spikes').textContent = n ? fmtValue(n.value) : 'n';
    $('#model-responding').disabled = !n;
    $('#model-responding').checked = !!n && this.filter.responding;
    $('#model-responding-from').textContent = n ? `(${n.from})` : '(n_spikes_for_response: not given for this model)';
    this._showRun();
    $('#model-view').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // the KCs kept: of each, whether its most spikes to any odor are over 0 (spiking) and at
  // least n_spikes_for_response (responding)
  _kept(run, values, filter, nSpikes) {
    const nK = run.kcs.labels.length, nO = run.odors.length, keep = [];
    for (let k = 0; k < nK; k++) {
      let max = -Infinity;
      for (let o = 0; o < nO; o++) { const v = values[k * nO + o]; if (v > max) max = v; }
      if (filter.spiking && !(max > 0)) continue;
      if (filter.responding && nSpikes != null && !(max >= nSpikes)) continue;
      keep.push(k);
    }
    return keep;
  }

  // the filter as named in Compare (and its key): 'responding', 'spiking' or 'all'
  _filterName() {
    const n = this.current?.model.n_spikes;
    return this.filter.responding && n ? 'responding' : this.filter.spiking ? 'spiking' : 'all';
  }

  _showRun() {
    const c = this.current;
    if (!c) return;
    const { $, decodeValues, MatrixViewer, fitViewer } = this.h, { run, model } = c;
    const filter = this._filterName(), key = `${c.id}|${filter}`, nSpikes = model.n_spikes?.value;
    if (!this.shown.has(key)) {
      const values = decodeValues(run.values_b64), nO = run.odors.length;
      const keep = this._kept(run, values, { spiking: filter !== 'all', responding: filter === 'responding' }, nSpikes);
      const kept = new Float32Array(keep.length * nO);
      keep.forEach((k, i) => kept.set(values.subarray(k * nO, (k + 1) * nO), i * nO));
      const groups = run.kcs.groups;
      this.shown.set(key, {
        payload: {
          meta: { kind: 'sequential', scale: 'log', colormap: 'oklab-red', value_label: 'spike count', row_label: run.kcs.levels[0] || 'KC', col_label: 'odor' },
          cluster: { rows: 'KCs', cols: 'odors' },
          rows: keep.map((k) => ({ label: run.kcs.labels[k], group: groups ? groups[k] : null })),
          cols: run.odors.map((o) => ({ label: o.odor, chem: o.chem ? { ...o.chem, odor_id: o.odor_id } : null })),
          values: kept, chemicals: run.odors.some((o) => o.chem) ? { matched: true, axis: 'cols' } : null, kcAxis: 'rows',
        },
        n: keep.length,
        keep, // (each row's KC in its file)
      });
    }
    const shown = this.shown.get(key), nK = run.kcs.labels.length, nO = run.odors.length;
    const matched = run.odors.filter((o) => o.odor_id != null).length;
    $('#model-sub').innerHTML = `${shown.n.toLocaleString()} of its ${nK.toLocaleString()} KCs shown (${filter === 'all' ? 'all of them'
      : filter === 'spiking' ? 'those spiking to some odor' : `those with at least ${fmtValue(nSpikes)} spikes to some odor`}) × ${nO} odors
      (${matched === nO ? 'all' : `${matched} of them`} matched to the Odors tab)${run.kcs.groups ? ', grouped by KC type' : ''}.
      From ${run.job ? `model run #${run.job} (above): ` : ''}<code>${this.h.esc(run.file)}</code>${run.imported ? `, as imported into olfdb ${this.h.esc(this.h.fmtWhen(run.imported.at))}` : ''}${run.transposed
        ? ', its KCs as columns there' : ''}.`;
    if (!this.viewer) {
      const el = $('#model-viewer');
      this.viewer = new MatrixViewer(el, { kind: 'model' });
      this.viewer.onShow = () => fitViewer(el, this.viewer);
    }
    const stem = `olfdb_model_${c.title}`.replace(/[^\w.-]+/g, '_');
    this.viewer.setGenericData(shown.payload, { title: c.title, fileStem: stem, kind: 'model' });
  }

  // the run shown, to save: its files, or its spike counts as shown (`MatrixViewer.shownKeys`:
  // of the KCs kept, their rows in its file)
  _download(format) {
    const c = this.current;
    if (!c) return;
    const json = { id: c.id, format }, shown = this.shown.get(`${c.id}|${this._filterName()}`);
    const k = format === 'wide' && shown && this.viewer?.source === shown.payload ? this.viewer.shownKeys() : null;
    // (of KCs summed over, all of those kept)
    if (k) Object.assign(json, { rows: (k.rows || shown.keep.map((_, i) => i)).map((i) => shown.keep[i]), cols: k.cols, transpose: k.transpose });
    this.h.download('/api/models/run/download', this.h.$('#model-download-status'), { json });
  }

  // the run shown compared with datasets: in the Compare tab, with the KCs as shown
  _compare() {
    const c = this.current;
    if (!c) return;
    this.h.onCompare({ id: c.id, title: c.title, filter: this._filterName(),
      nSpikes: c.model.n_spikes?.value ?? null, run: c.run });
  }
}
