// The Models tab (model_data.py): every model of the folders of model data added in the Admin
// tab, by its parameters, in a table sorted by any of them; and a model run opened below it,
// its KCs' spike counts to odors in a viewer (KCs that never spike left out, unless chosen),
// to compare with datasets in the Compare tab.

const KEY = 'olfdb.models'; // {hideSame, sort: {col, dir}}: kept in this browser

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

export class ModelsPage {
  // helpers: { $, esc, api, msgBox, fmtWhen, LS, decodeValues, fitViewer, MatrixViewer (class),
  // download(url, status, {json}), wireMenu(menu, onPick), onCompare(model: {id, title, filter, nSpikes, run}) }
  constructor(h) {
    this.h = h;
    this.data = null; // as from /api/models
    this.loadedAt = 0;
    this.runs = new Map(); // run id -> as from /api/models/run
    this.shown = new Map(); // `${run id}|${filter}` -> the matrix shown (kept, for its edits to stay)
    this.current = null; // the run shown: {id, model, run, label}
    this.openId = null; // the run opened (its row marked in the table), shown once loaded
    this._seq = 0;
    this.viewer = null;
    this.o = { hideSame: false, sort: { col: null, dir: 1 }, ...h.LS.get(KEY, {}) };
    this.filter = { spiking: true, responding: false };
    const { $ } = h;
    $('#models-hide-same').checked = !!this.o.hideSame;
    $('#models-hide-same').addEventListener('change', (e) => { this.o.hideSame = e.target.checked; this._save(); this.render(); });
    $('#models-table').addEventListener('click', (e) => {
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
    });
    $('#model-spiking').addEventListener('change', (e) => { this.filter.spiking = e.target.checked; this._showRun(); });
    $('#model-responding').addEventListener('change', (e) => { this.filter.responding = e.target.checked; this._showRun(); });
    $('#model-compare').addEventListener('click', () => this._compare());
    h.wireMenu($('#model-download'), ({ format }) => this._download(format));
  }

  _save() { this.h.LS.set(KEY, { hideSame: this.o.hideSame, sort: this.o.sort }); }

  // after folders of model data were added, changed or deleted (Admin tab): read again when next shown
  invalidate() { this.data = null; this.runs.clear(); this.shown.clear(); }

  // the tab opened with no run: none shown
  close() {
    this._seq++;
    this.openId = null;
    this.current = null;
    this.h.$('#model-view').hidden = true;
    this.render();
  }

  // each time the tab is opened (models may have been added to the folders since), but not
  // again within a few seconds (as when a run is opened)
  async load() {
    if (this.data && Date.now() - this.loadedAt < 10e3) return;
    const { $, api, msgBox, esc } = this.h;
    try { this.data = await api('/api/models'); } catch (e) {
      $('#models-status').replaceChildren(msgBox('error', `Could not load the models: ${esc(e.message)}`));
      return;
    }
    this.loadedAt = Date.now();
    this.render();
  }

  // a model's value of a parameter: its row's, if its params.csv has the column, else its experiment's
  _value(m, col) {
    const e = this.data.experiments[m.experiment];
    return (e.params.includes(col) ? m.params[col] : e.flat[col]) ?? null;
  }

  // what a run of a model is called: its experiment's folder, its tag (or variant) after what its
  // file's name has before that (e.g. its base: models of two bases may share a tag), and its condition
  _title(m, r) {
    const e = this.data.experiments[m.experiment], name = [r.prefix, m.params.tag ?? m.params.variant].filter((x) => x != null && x !== '');
    return `${e.folder || this._source(e.source)?.path || 'model'}: ${name.join('_')}, ${r.label || 'spike counts'}`;
  }

  _source(id) { return this.data.sources.find((s) => s.id === id); }

  // the columns shown: all, or (as chosen) those whose models that have them differ
  _columns() {
    const cols = this.data.columns;
    if (!this.o.hideSame) return cols;
    return cols.filter((c) => {
      let first;
      for (const m of this.data.models) {
        const v = this._value(m, c);
        if (v == null) continue;
        if (first === undefined) first = v;
        else if (v !== first) return true;
      }
      return false;
    });
  }

  render() {
    const { $, esc, msgBox } = this.h, d = this.data;
    if (!d) return;
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
    if (notes.length) status.push(Object.assign(document.createElement('p'), { className: 'muted small', innerHTML: notes.join(' ') }));
    $('#models-status').replaceChildren(...status);
    $('#models-table-step').hidden = !d.models.length;
    if (!d.models.length) return;
    const cols = this._columns(), s = this.o.sort;
    $('#models-shown').textContent = this.o.hideSame ? `${cols.length} of ${d.columns.length} parameters shown` : `${d.columns.length} parameters`;
    const order = d.models.map((m, i) => i);
    if (s.col && cols.includes(s.col)) {
      const v = d.models.map((m) => this._value(m, s.col));
      order.sort((i, j) => (v[i] == null || v[j] == null ? cmpValues(v[i], v[j]) : s.dir * cmpValues(v[i], v[j])) || i - j);
    }
    const head = cols.map((c) => {
      const on = s.col === c;
      return `<th aria-sort="${on ? (s.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" class="th-sort${on ? ' on' : ''}" data-sort="${esc(c)}"
        title="Sort by ${esc(c)}${on ? (s.dir > 0 ? ' (again: descending)' : ' (again: ascending)') : ''}">${esc(c)}${on ? (s.dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`;
    }).join('');
    const open = this.openId;
    const body = order.map((i) => {
      const m = d.models[i];
      const runs = m.runs.length ? m.runs.map((r) => `<button type="button" class="btn small${r.id === open ? ' primary' : ''}" data-run="${esc(r.id)}"
        title="${esc(`Open ${r.file}`)}">${esc(r.label || 'spike counts')}</button>`).join('')
        : '<span class="muted small" title="No spike count file of it was found">none</span>';
      const cells = cols.map((c) => {
        const v = this._value(m, c);
        if (v == null) return '<td></td>';
        const text = c === 'generated' && typeof v === 'string' ? this.h.fmtWhen(v) : fmtValue(v);
        return `<td${typeof v === 'number' ? ' class="num"' : ''} title="${esc(`${c}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)}">${esc(text)}</td>`;
      }).join('');
      return `<tr${m.runs.some((r) => r.id === open) ? ' class="hl"' : ''}><td class="model-runs">${runs}</td>${cells}</tr>`;
    }).join('');
    $('#models-table').innerHTML = `<table class="odor-table models-table"><thead><tr><th>Open</th>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  // a run opened (from the table, or a link to it): its spike counts shown below the table
  async show(id) {
    const { $, api, msgBox, esc } = this.h, seq = ++this._seq;
    const model = this.data?.models.find((m) => m.runs.some((r) => r.id === id));
    this.openId = model ? id : null;
    $('#model-view').hidden = false;
    if (!model) {
      $('#model-title').textContent = 'No such model run';
      $('#model-msgs').replaceChildren(msgBox('error', `<code>${esc(id)}</code> is not a run of the models listed (any more).`));
      $('#model-body').hidden = true;
      return;
    }
    const r = model.runs.find((x) => x.id === id);
    $('#model-title').textContent = this._title(model, r);
    $('#model-msgs').replaceChildren();
    $('#model-body').hidden = false;
    this.render(); // (its row marked)
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
    this.current = { id, model, run, title: this._title(model, r) };
    const prov = run.provenance;
    $('#model-prov-box').hidden = !prov;
    $('#model-prov').textContent = prov ? JSON.stringify(prov, null, 2) : '';
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
      From <code>${this.h.esc(run.file)}</code>${run.transposed ? ', its KCs as columns there' : ''}.`;
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
