// MetaCyc in the Reference data tab: a reference dataset of a MetaCyc bundle looked into
// (/api/reference/<id>/metacyc/...): a pathway drawn as a graph of its main compounds and
// reactions, with its classes, organisms and summary; a compound, with its pathways; the
// paths from two compounds' lowest common ancestors (pathway LCA distance); and a search.

const NODE_H = 34, LINE_H = 12, COL_GAP = 26, ROW_GAP = 58, WRAP_GAP = 22, PAD = 16, CHAR_W = 6.3, MAX_CHARS = 24, MAX_ROW_W = 1100;

// a label broken into lines of at most `max` characters (at spaces, hyphens and commas where it can)
function wrap(text, max = MAX_CHARS, lines = 3) {
  const words = String(text).split(/(?<=[\s,-])/);
  const out = [];
  let cur = '';
  for (const w of words) {
    if ((cur + w).trim().length > max && cur.trim()) { out.push(cur.trim()); cur = ''; }
    cur += w;
    while (cur.length > max) { out.push(cur.slice(0, max)); cur = cur.slice(max); }
  }
  if (cur.trim()) out.push(cur.trim());
  if (out.length > lines) { out.length = lines; out[lines - 1] = `${out[lines - 1].slice(0, max - 1)}…`; }
  return out;
}

// A pathway's main compounds laid out in layers, top to bottom, along its reactions: each
// reaction an edge from each main compound it consumes to each it produces (reversible ones
// and those of unknown direction turned to run away from where the pathway starts: its
// primary reactants, else what it consumes and does not make, else compounds no reaction
// produces; as MetaCyc writes them where that does not tell), cycles broken, each compound in
// the layer below the lowest one feeding it, and the order within layers by the mean
// position of their neighbours (a few sweeps down and up). A layer wider than MAX_ROW_W
// goes on over more rows.
// -> {nodes: Map(id -> {id, x, y, w, h, lines, rank}), edges: [{from, to, reaction, both}], width, height}
export function layoutPathway(p) {
  const ids = [];
  const seen = new Set();
  for (const r of p.reactions) for (const c of [...r.left, ...r.right]) if (!seen.has(c)) { seen.add(c); ids.push(c); }
  const raw = [];
  for (const r of p.reactions) for (const a of r.left) for (const b of r.right) if (a !== b) raw.push({ a, b, r, fixed: r.direction === 'L2R' });
  const made = new Set(p.products || []), into = new Set(raw.filter((e) => e.fixed).map((e) => e.b));
  let starts = (p.primary_reactants || []).filter((c) => seen.has(c));
  if (!starts.length) starts = (p.reactants || []).filter((c) => seen.has(c) && !made.has(c));
  if (!starts.length) starts = ids.filter((c) => !into.has(c) && raw.some((e) => e.a === c));
  if (!starts.length && ids.length) starts = [ids[0]];
  // how far each compound is from there (by any reaction, either way)
  const hop = new Map(starts.map((c) => [c, 0]));
  const nbrs = new Map(ids.map((c) => [c, []]));
  for (const e of raw) { nbrs.get(e.a).push(e.b); nbrs.get(e.b).push(e.a); }
  for (const queue = [...starts]; queue.length;) {
    const u = queue.shift();
    for (const v of nbrs.get(u)) if (!hop.has(v)) { hop.set(v, hop.get(u) + 1); queue.push(v); }
  }
  const edges = raw.map((e) => {
    const flip = !e.fixed && hop.has(e.a) && hop.has(e.b) && hop.get(e.b) < hop.get(e.a);
    return { from: flip ? e.b : e.a, to: flip ? e.a : e.b, reaction: e.r, both: e.r.direction === 'REV' };
  });
  // cycles broken (by a depth-first search from the starts), for ranking only
  const out = new Map(ids.map((c) => [c, []]));
  edges.forEach((e, k) => out.get(e.from).push(k));
  const state = new Map(), back = new Set();
  const visit = (u) => {
    state.set(u, 1);
    for (const k of out.get(u)) {
      const v = edges[k].to;
      if (state.get(v) === 1) back.add(k);
      else if (!state.has(v)) visit(v);
    }
    state.set(u, 2);
  };
  for (const c of [...starts, ...ids]) if (!state.has(c)) visit(c);
  const rank = new Map(ids.map((c) => [c, 0]));
  for (let changed = true, n = 0; changed && n < ids.length + 1; n++) {
    changed = false;
    edges.forEach((e, k) => {
      if (back.has(k)) return;
      if (rank.get(e.to) < rank.get(e.from) + 1) { rank.set(e.to, rank.get(e.from) + 1); changed = true; }
    });
  }
  const layers = [];
  for (const c of ids) (layers[rank.get(c)] ||= []).push(c);
  for (let i = 0; i < layers.length; i++) layers[i] ||= [];
  const pos = new Map();
  const place = () => layers.forEach((l) => l.forEach((c, i) => pos.set(c, i)));
  place();
  const upN = new Map(ids.map((c) => [c, []])), downN = new Map(ids.map((c) => [c, []]));
  edges.forEach((e) => { downN.get(e.from).push(e.to); upN.get(e.to).push(e.from); });
  const sweep = (layer, of) => {
    const bc = new Map(layer.map((c) => {
      const ns = of.get(c).filter((n) => pos.has(n));
      return [c, ns.length ? ns.reduce((s, n) => s + pos.get(n), 0) / ns.length : pos.get(c)];
    }));
    layer.sort((a, b) => bc.get(a) - bc.get(b));
  };
  for (let it = 0; it < 4; it++) {
    for (let i = 1; i < layers.length; i++) { sweep(layers[i], upN); place(); }
    for (let i = layers.length - 2; i >= 0; i--) { sweep(layers[i], downN); place(); }
  }
  // sizes, then each layer centred (in rows of at most MAX_ROW_W)
  const nodes = new Map();
  for (const c of ids) {
    const info = p.compounds[c] || { name: c };
    const label = info.odor ? `${info.name} (${info.odor.abbrev || info.odor.name})` : info.name;
    const lines = wrap(label);
    const w = Math.max(70, Math.min(MAX_CHARS, Math.max(...lines.map((l) => l.length))) * CHAR_W + 18);
    nodes.set(c, { id: c, lines, w, h: NODE_H + (lines.length - 1) * LINE_H, rank: rank.get(c) });
  }
  const rowsOf = (l) => {
    const out = [[]];
    let w = 0;
    for (const c of l) {
      const nw = nodes.get(c).w;
      if (out.at(-1).length && w + COL_GAP + nw > MAX_ROW_W) { out.push([]); w = 0; }
      w += (out.at(-1).length ? COL_GAP : 0) + nw;
      out.at(-1).push(c);
    }
    return out;
  };
  const rowW = (r) => r.reduce((s, c) => s + nodes.get(c).w, 0) + Math.max(0, r.length - 1) * COL_GAP;
  const wrapped = layers.map(rowsOf);
  const width = Math.max(240, ...wrapped.flat().map(rowW)) + 2 * PAD;
  let y = PAD + 14;
  wrapped.forEach((rows) => {
    rows.forEach((r, k) => {
      let x = (width - rowW(r)) / 2;
      const h = Math.max(NODE_H, ...r.map((c) => nodes.get(c).h));
      for (const c of r) { const n = nodes.get(c); n.x = x; n.y = y + (h - n.h) / 2; x += n.w + COL_GAP; }
      y += h + (k < rows.length - 1 ? WRAP_GAP : ROW_GAP);
    });
  });
  return { nodes, edges, width, height: y - ROW_GAP + PAD + 8, layers };
}

// The pathway drawn (SVG markup): its compounds (those of the odor table colored and named
// by it, those of `mark` outlined), its reactions as arrows labelled by their EC numbers
// (two-headed if reversible), each with its whole equation and side compounds on hover.
export function pathwaySvg(p, lay, { esc, mark = [] }) {
  const marked = new Set(mark);
  const parts = [];
  const mid = (n) => n.x + n.w / 2;
  // several reactions between the same two compounds drawn apart
  const pairs = new Map();
  for (const e of lay.edges) { const k = [e.from, e.to].sort().join('\u0001'); pairs.set(k, [...(pairs.get(k) || []), e]); }
  for (const e of lay.edges) {
    const a = lay.nodes.get(e.from), b = lay.nodes.get(e.to);
    const same = pairs.get([e.from, e.to].sort().join('\u0001')), k = same.indexOf(e), bend = (k - (same.length - 1) / 2) * 34;
    let d;
    let lx, ly;
    if (b.rank > a.rank) {
      const x1 = mid(a), y1 = a.y + a.h, x2 = mid(b), y2 = b.y;
      const cy = (y1 + y2) / 2;
      d = `M${x1},${y1} C${x1 + bend},${cy} ${x2 + bend},${cy} ${x2},${y2 - 2}`;
      // (its label three quarters of the way along, by the compound it makes)
      const t = 0.72, u = 1 - t, at = (p0, p1, p2, p3) => u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
      lx = at(x1, x1 + bend, x2 + bend, x2); ly = at(y1, cy, cy, y2 - 2);
    } else if (b.rank === a.rank) {
      const left = a.x < b.x, x1 = left ? a.x + a.w : a.x, x2 = left ? b.x : b.x + b.w, y0 = a.y + a.h / 2;
      d = `M${x1},${y0} C${(x1 + x2) / 2},${y0 - 30 - bend} ${(x1 + x2) / 2},${y0 - 30 - bend} ${x2},${b.y + b.h / 2}`;
      lx = (x1 + x2) / 2; ly = y0 - 26 - bend;
    } else { // back up (a cycle): around the right
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x + b.w, y2 = b.y + b.h / 2, out = Math.max(x1, x2) + 40 + Math.abs(bend);
      d = `M${x1},${y1} C${out},${y1} ${out},${y2} ${x2 + 2},${y2}`;
      lx = out - 10; ly = (y1 + y2) / 2;
    }
    const r = e.reaction;
    const side = [...r.side_left.map((c) => p.compounds[c]?.name || c), '→', ...r.side_right.map((c) => p.compounds[c]?.name || c)];
    const tip = `${r.equation}${r.ec.length ? `\nEC ${r.ec.join(', ')}` : ''}${r.side_left.length || r.side_right.length ? `\nalso: ${side.join(' ')}` : ''}\n${r.id}`;
    parts.push(`<g class="mc-edge${e.both ? ' rev' : ''}" data-reaction="${esc(r.id)}"><title>${esc(tip)}</title>
      <path d="${d}" marker-end="url(#mc-arrow)"${e.both ? ' marker-start="url(#mc-arrow-start)"' : ''}/>
      <path class="hit" d="${d}"/>
      ${r.ec.length ? `<text x="${lx + 6}" y="${ly + 3}">${esc(r.ec[0])}${r.ec.length > 1 ? '…' : ''}</text>` : ''}</g>`);
  }
  for (const n of lay.nodes.values()) {
    const c = p.compounds[n.id] || { name: n.id };
    const cls = ['mc-node', c.odor ? 'odor' : '', marked.has(n.id) ? 'marked' : '', c.is_class ? 'class' : '',
      (p.primary_reactants || []).includes(n.id) || (p.primary_products || []).includes(n.id) ? 'primary' : ''].filter(Boolean).join(' ');
    const links = p.links?.[n.id]?.filter((l) => l.id).length || 0;
    const tip = `${c.name}${c.formula ? ` (${c.formula})` : ''}${c.odor ? `\nodor table: ${c.odor.name}` : ''}${c.is_class ? '\na class of compounds' : ''}`
      + `${links ? `\nin ${links}${p.links[n.id].length > links ? '+' : ''} other pathways` : ''}\n${n.id}`;
    const ty = n.y + n.h / 2 - ((n.lines.length - 1) * LINE_H) / 2 + 4;
    parts.push(`<g class="${cls}" data-compound="${esc(n.id)}" tabindex="0"><title>${esc(tip)}</title>
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${c.is_class ? 12 : 5}"/>
      ${n.lines.map((l, i) => `<text x="${n.x + n.w / 2}" y="${ty + i * LINE_H}">${esc(l)}</text>`).join('')}
      ${links ? `<text class="badge" x="${n.x + n.w - 4}" y="${n.y + 10}">↗${links}</text>` : ''}</g>`);
  }
  return `<svg class="mc-graph" xmlns="http://www.w3.org/2000/svg" width="${lay.width}" height="${lay.height}" viewBox="0 0 ${lay.width} ${lay.height}">
    <defs><marker id="mc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z"/></marker>
    <marker id="mc-arrow-start" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M10,0 L0,5 L10,10 z"/></marker></defs>
    ${parts.join('')}</svg>`;
}

// The panel under a MetaCyc reference dataset's matrix: what was last looked at in it, with
// a way back, and a search of its pathways and compounds.
export class MetacycPanel {
  constructor(el, { api, esc, msgBox }) {
    Object.assign(this, { el, api, esc, msgBox });
    this.refId = null;
    this.history = [];
    this.seq = 0;
    el.innerHTML = `<div class="mc-head"><h3>MetaCyc</h3>
        <input type="search" class="mc-search" placeholder="Find a pathway or compound (name or MetaCyc ID)" spellcheck="false" autocomplete="off" aria-label="Find in MetaCyc">
        <button type="button" class="btn small mc-back" hidden>← Back</button></div>
      <div class="mc-results" hidden></div><div class="mc-body"></div>`;
    this.body = el.querySelector('.mc-body');
    this.results = el.querySelector('.mc-results');
    let timer = null;
    const input = el.querySelector('.mc-search');
    input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => this.search(input.value), 250); });
    input.addEventListener('keydown', (e) => { if (e.key === 'Escape') { input.value = ''; this.results.hidden = true; } });
    el.querySelector('.mc-back').addEventListener('click', () => this.back());
    // a click on anything naming a pathway or compound (here or in a graph) shows it
    el.addEventListener('click', (e) => {
      const t = e.target.closest('[data-show-pathway], [data-show-compound], .mc-node, .mc-edge');
      if (!t || !el.contains(t)) return;
      if (t.dataset.showPathway) { e.preventDefault(); this.show({ kind: 'pathway', id: t.dataset.showPathway, mark: (t.dataset.mark || '').split(' ').filter(Boolean) }); }
      else if (t.dataset.showCompound || t.classList.contains('mc-node')) { e.preventDefault(); this.show({ kind: 'compound', id: t.dataset.showCompound || t.dataset.compound }); }
      else if (t.classList.contains('mc-edge')) this._highlightReaction(t.dataset.reaction);
    });
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.classList?.contains('mc-node')) e.target.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
  }

  // a reference dataset (its id, and as loaded) whose bundle this looks into
  setRef(id, ref) {
    if (id === this.refId) return;
    this.refId = id;
    this.ref = ref;
    this.history = [];
    this.body.innerHTML = `<p class="muted small">${ref.pairwise ? 'Click a cell of the matrix for the paths from the two chemicals\' lowest common ancestors' : 'Click a cell of the matrix to draw its pathway'}, or right-click a label; or find any pathway or compound above.</p>`;
    this.el.querySelector('.mc-back').hidden = true;
    this.el.querySelector('.mc-search').value = '';
    this.results.hidden = true;
  }

  _url(what, rest = '') { return `/api/reference/${encodeURIComponent(this.refId)}/metacyc/${what}${rest}`; }

  async search(q) {
    if (q.trim().length < 2) { this.results.hidden = true; return; }
    const seq = ++this.seq;
    let res;
    try { res = await this.api(this._url('search', `?q=${encodeURIComponent(q)}`)); } catch (e) {
      this.results.hidden = false; this.results.replaceChildren(this.msgBox('error', this.esc(e.message))); return;
    }
    if (seq !== this.seq) return;
    const { esc } = this;
    const list = (items, kind) => (items.length ? items.map((it) => `<button type="button" class="mc-hit" data-show-${kind}="${esc(it.id)}">${esc(it.name)} <span class="muted">${esc(it.id)}</span></button>`).join('')
      : '<span class="muted small">none</span>');
    this.results.hidden = false;
    this.results.innerHTML = `<div><b>Pathways</b>${list(res.pathways, 'pathway')}</div><div><b>Compounds</b>${list(res.compounds, 'compound')}</div>`;
  }

  back() {
    this.history.pop();
    const prev = this.history.pop();
    if (prev) this.show(prev);
  }

  // what to show: {kind: 'pathway', id, mark}, {kind: 'compound', id} or {kind: 'lca', a, b, labels}
  async show(what) {
    this.results.hidden = true;
    const seq = ++this.seq;
    this.history.push(what);
    this.el.querySelector('.mc-back').hidden = this.history.length < 2;
    this.body.innerHTML = '<p class="muted">Loading…</p>';
    let data;
    try {
      data = await this.api(what.kind === 'lca' ? this._url('lca', `?a=${encodeURIComponent(what.a)}&b=${encodeURIComponent(what.b)}`)
        : this._url(what.kind, `/${encodeURIComponent(what.id)}`));
    } catch (e) {
      if (seq === this.seq) this.body.replaceChildren(this.msgBox('error', this.esc(e.message)));
      return;
    }
    if (seq !== this.seq) return;
    if (what.kind === 'pathway') this._pathway(data, what.mark || []);
    else if (what.kind === 'compound') this._compound(data);
    else this._lca(data, what);
    this.el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  _link(url, text) { return url ? `<a href="${this.esc(url)}" target="_blank" rel="noopener">${this.esc(text)}</a>` : this.esc(text); }

  _pathway(p, mark) {
    const { esc } = this;
    const lay = layoutPathway(p);
    const chips = (items, kind, extra = '') => items.map((it) => `<button type="button" class="chip mc-chip" data-show-${kind}="${esc(it.id)}"${extra}>${esc(it.name)}</button>`).join(' ');
    const sp = p.species;
    const agg = (s, k) => s[k].map((v) => `${esc(v.value)}${v.n > 1 ? ` <span class="muted">(${v.n})</span>` : ''}`).join(', ');
    const speciesRows = sp.map((s) => `<tr><td>${this._link(s.url, s.name)}${s.strain ? ` <span class="muted">${esc(s.strain)}</span>` : ''}</td>
      <td>${esc(s.synonyms.join(', '))}</td><td class="num">${s.genomes || ''}</td><td>${agg(s, 'environment')}</td><td>${agg(s, 'oxygen')}</td>
      <td>${agg(s, 'temperature')}</td><td>${agg(s, 'pathogenicity')}</td><td>${agg(s, 'host')}</td></tr>`).join('');
    const odors = Object.values(p.compounds).filter((c) => c.odor);
    const reactions = p.reactions.map((r, i) => `<tr data-reaction="${esc(r.id)}"><td class="num">${i + 1}</td><td>${esc(r.equation)}</td>
      <td>${esc(r.ec.join(', '))}</td><td>${{ L2R: '→', REV: '⇄', '?': '?' }[r.direction] || '?'}</td><td>${this._link(r.url, r.id)}</td></tr>`).join('');
    this.body.innerHTML = `
      <div class="mc-title"><h3>${esc(p.name)}</h3> <span class="muted">${this._link(p.url, p.id)}${p.version ? ` · MetaCyc ${esc(p.version)}` : ''}${p.engineered ? ' · engineered' : ''}</span></div>
      <p class="mc-meta">${p.types.map((t) => `<span class="chip" title="${esc(t.lineage.join(' › '))}">${esc(t.lineage.join(' › '))}</span>`).join(' ')}
        ${p.taxonomic_range.length ? `<span class="muted">· expected in</span> ${esc(p.taxonomic_range.join(', '))}` : ''}</p>
      ${p.super.length ? `<p class="mc-meta"><span class="muted">In superpathways:</span> ${chips(p.super, 'pathway')}</p>` : ''}
      ${p.subs.length ? `<p class="mc-meta"><span class="muted">Pathways all of whose reactions it has:</span> ${chips(p.subs, 'pathway')}</p>` : ''}
      ${odors.length ? `<p class="mc-meta"><span class="muted">Chemicals of the odor table in it:</span> ${odors.map((c) => `<button type="button" class="chip mc-chip odor" data-show-compound="${esc(c.id)}">${esc(c.odor.name)}</button>`).join(' ')}</p>` : ''}
      <div class="mc-graph-wrap">${lay.nodes.size ? pathwaySvg(p, lay, { esc, mark }) : '<p class="muted">No reactions with compounds to draw.</p>'}</div>
      <p class="muted small">${p.reactions.length} reaction${p.reactions.length === 1 ? '' : 's'}, top to bottom from where it starts; compounds in color are in the odor table,
        rounder ones are classes of compounds (e.g. "an alcohol"), ↗ how many other pathways a compound is in. Common cofactors (ATP, NAD(P)H, H₂O, ...) and compounds in more than 100 pathways are left out of the graph
        (they are in each reaction's equation, on hover). Click a compound for its details; ⇄ reversible.</p>
      <details class="mc-section" open><summary>Organisms (${sp.length})</summary>${sp.length ? `<div class="mc-scroll"><table class="odor-table mc-table">
        <thead><tr><th>Species</th><th>Common names</th><th class="num" title="BioCyc's genome databases of the species">Genomes</th><th>Environments</th><th>Oxygen</th><th>Temperature</th><th>Pathogenicity</th><th>Hosts</th></tr></thead>
        <tbody>${speciesRows}</tbody></table></div><p class="muted small">Where it was shown to occur, as MetaCyc says; their environments and the rest are of BioCyc's databases of their genomes (how many of them say so).</p>` : '<p class="muted">None listed.</p>'}</details>
      ${p.summary ? `<details class="mc-section"><summary>Summary</summary>${p.summary.split('\n\n').map((t) => `<p>${esc(t)}</p>`).join('')}</details>` : ''}
      <details class="mc-section"><summary>Reactions (${p.reactions.length})</summary><div class="mc-scroll"><table class="odor-table mc-table">
        <thead><tr><th class="num">#</th><th>Equation</th><th>EC</th><th></th><th>MetaCyc</th></tr></thead><tbody>${reactions}</tbody></table></div></details>`;
  }

  _highlightReaction(id) {
    this.body.querySelectorAll('.mc-edge.on, tr.on').forEach((x) => x.classList.remove('on'));
    this.body.querySelectorAll(`[data-reaction="${CSS.escape(id)}"]`).forEach((x) => x.classList.add('on'));
  }

  _compound(c) {
    const { esc } = this;
    const list = (items, mark) => (items.length ? items.map((it) => `<button type="button" class="chip mc-chip" data-show-pathway="${esc(it.id)}" data-mark="${esc(mark)}">${esc(it.name)}</button>`).join(' ')
      : '<span class="muted">none</span>');
    const rx = (items, n) => `${items.map((r) => `<li>${esc(r.equation)} <span class="muted">${esc(r.id)}</span></li>`).join('')}${n > items.length ? `<li class="muted">and ${n - items.length} more</li>` : ''}`;
    this.body.innerHTML = `
      <div class="mc-title"><h3>${esc(c.name)}</h3> <span class="muted">${this._link(c.url, c.id)}${c.version ? ` · MetaCyc ${esc(c.version)}` : ''}${c.is_class ? ' · a class of compounds' : ''}</span></div>
      <div class="mc-compound">${c.inchi ? `<img class="mc-structure" alt="" src="${esc(this._url('structure', `/${encodeURIComponent(c.id)}`))}">` : ''}
        <dl class="mc-dl">
          ${c.odor ? `<dt>Odor table</dt><dd><a href="#odors/${esc(c.odor.id)}">${esc(c.odor.name)}</a>${c.odor.abbrev ? ` (${esc(c.odor.abbrev)})` : ''}</dd>` : ''}
          ${c.formula ? `<dt>Formula</dt><dd>${esc(c.formula)}</dd>` : ''}
          ${c.cid ? `<dt>PubChem CID</dt><dd>${this._link(`https://pubchem.ncbi.nlm.nih.gov/compound/${c.cid}`, c.cid)}</dd>` : ''}
          ${c.kegg ? `<dt>KEGG</dt><dd>${this._link(`https://www.genome.jp/entry/${c.kegg}`, c.kegg)}</dd>` : ''}
          ${c.inchikey ? `<dt>InChIKey</dt><dd><code>${esc(c.inchikey)}</code></dd>` : ''}
          ${c.smiles ? `<dt>SMILES</dt><dd><code class="mc-wrap">${esc(c.smiles)}</code></dd>` : ''}
          ${c.classes.length ? `<dt>Classes</dt><dd>${c.classes.map((k) => `<button type="button" class="chip mc-chip" data-show-compound="${esc(k.id)}">${esc(k.name)}</button>`).join(' ')}</dd>` : ''}
        </dl></div>
      <p class="mc-meta"><span class="muted">In ${c.pathways.length} pathway${c.pathways.length === 1 ? '' : 's'}${c.is_class ? ' (through its reactions)' : ''}:</span> ${list(c.pathways, c.id)}</p>
      ${c.general_pathways.length ? `<p class="mc-meta"><span class="muted">And through classes it is of, in:</span> ${list(c.general_pathways, c.id)}</p>` : ''}
      <details class="mc-section"><summary>Reactions consuming it (${c.n_consumed}) and producing it (${c.n_produced})</summary>
        <h4>Consuming it</h4><ul class="mc-rx">${rx(c.consumed, c.n_consumed)}</ul><h4>Producing it</h4><ul class="mc-rx">${rx(c.produced, c.n_produced)}</ul></details>`;
  }

  _lca(res, what) {
    const { esc } = this;
    const [la, lb] = what.labels || [what.a, what.b];
    if (res.distance == null) {
      this.body.innerHTML = `<div class="mc-title"><h3>${esc(la)} and ${esc(lb)}</h3></div>
        <p>${res.missing?.length ? `Not in the network: ${res.missing.map(esc).join(', ')}.` : 'They have no common ancestor in the network: no distance.'}</p>`;
      return;
    }
    const node = (id) => {
      const n = res.nodes[id] || { id, name: id, kind: 'compound' };
      const attr = n.kind === 'pathway' ? `data-show-pathway="${esc(id)}"` : n.kind === 'reaction' ? '' : `data-show-compound="${esc(id)}"`;
      const pw = (n.pathways || []).slice(0, 4);
      return `<div class="mc-step ${n.kind}"><button type="button" class="mc-stepname" ${attr} title="${esc(n.equation || n.name)}${n.kind === 'reaction' ? '' : ` (${n.kind})`}\n${esc(id)}">${esc(n.kind === 'reaction' ? (n.equation || n.name) : n.name)}</button>
        <span class="muted small">${esc(n.kind)}${pw.length ? ` in ${pw.map((q) => `<button type="button" class="linkish" data-show-pathway="${esc(q.id)}" data-mark="${esc(what.a)} ${esc(what.b)}">${esc(q.name)}</button>`).join(', ')}${n.pathways.length > pw.length ? ', …' : ''}` : ''}</span></div>`;
    };
    const entries = (res.entries || []).map((e, i) => {
      const through = e.filter((x) => x.id !== (i ? what.b : what.a));
      return through.length ? `${esc(i ? lb : la)} is taken as ${through.map((x) => esc(x.name)).join(', ')} (what it is of, not being in the network). ` : '';
    }).join('');
    this.body.innerHTML = `
      <div class="mc-title"><h3>${esc(la)} and ${esc(lb)}: ${res.distance}</h3></div>
      <p class="muted small">The fewest steps from a common ancestor to each, summed: ${esc(res.described)}. ${entries}
        ${res.lcas.length > 1 ? `${res.lcas.length} common ancestors are as near.` : ''}</p>
      ${res.lcas.map((l) => `<div class="mc-lca"><div class="mc-lca-top">${node(l.node)}</div>
        <div class="mc-lca-paths">${l.paths.map((pth, i) => `<div class="mc-path"><div class="muted small">to ${esc(i ? lb : la)} (${pth.length - 1})</div>
          ${pth.slice(1).map((id) => `<div class="mc-arrow">↓</div>${node(id)}`).join('')}</div>`).join('')}</div></div>`).join('')}`;
  }
}
