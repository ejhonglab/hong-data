// LOTUS in the Reference data tab: a reference dataset of a LOTUS bundle: above its matrix,
// how the matrix was made (each setting, and where it can be changed); below it, a cell
// looked into (/api/reference/<id>/lotus/cell): the organisms of its column the chemical is
// reported in, each with the papers reporting it there.

// a view's settings (its `lotus`, as lotus.describe says them), as a list
export function decisionsHtml(lt, esc) {
  return `<dl class="mc-dl lt-dl">${lt.decisions.map((d) => `<dt>${esc(d.what)}</dt><dd>${esc(d.chosen)}
    <span class="muted small">· ${d.admin === true ? 'can be changed in the Admin tab (Reference data: <i>Change settings…</i>)' : `can be changed through ${esc(d.admin)}`}</span></dd>`).join('')}</dl>`;
}

const year = (date) => (date || '').slice(0, 4);

export class LotusPanel {
  // el: under the matrix (a cell looked into); madeEl: above it (how it was made)
  constructor(el, madeEl, { api, esc, msgBox }) {
    this.el = el;
    this.api = api;
    this.esc = esc;
    this.msgBox = msgBox;
    this.seq = 0;
    madeEl.innerHTML = '<details class="lt-made" open><summary>How this matrix was made</summary><div class="lt-decisions"></div></details>';
    el.innerHTML = '<div class="lt-body"></div>';
    this.made = madeEl.querySelector('.lt-decisions');
    this.body = el.querySelector('.lt-body');
  }

  // a reference dataset (its id, and as loaded)
  setRef(id, ref) {
    const lt = ref.meta.lotus;
    this.made.innerHTML = decisionsHtml(lt, this.esc);
    if (id === this.refId && JSON.stringify(lt) === this._shownFor) return;
    this.refId = id;
    this._shownFor = JSON.stringify(lt);
    this.body.innerHTML = `<p class="muted small">Click a cell of the matrix for the ${lt.level === 'organism' ? 'papers reporting the chemical in the organism'
      : `organisms of the ${this.esc(lt.level)} the chemical is reported in, each with the papers reporting it there`}.</p>`;
  }

  // a cell: its row and column (as the matrix has them, each with its `lotus`)
  async show(row, col) {
    if (!row?.lotus?.length || !col?.lotus) return;
    const { esc } = this, seq = ++this.seq, t = col.lotus;
    const q = new URLSearchParams({ structures: row.lotus.join(','), name: t.name || '', kingdom: t.kingdom || '' });
    if (t.organism != null) q.set('organism', t.organism);
    this.body.innerHTML = '<p class="muted">Loading…</p>';
    let res;
    try { res = await this.api(`/api/reference/${encodeURIComponent(this.refId)}/lotus/cell?${q}`); } catch (e) {
      if (seq === this.seq) this.body.replaceChildren(this.msgBox('error', esc(e.message)));
      return;
    }
    if (seq !== this.seq) return;
    const tx = res.taxon, several = res.structures.length > 1;
    const sname = new Map(res.structures.map((s) => [s.id, s.name || s.iupac_name || s.inchikey]));
    const link = (url, text, title = '') => (url ? `<a href="${esc(url)}" target="_blank" rel="noopener"${title ? ` title="${esc(title)}"` : ''}>${text}</a>` : text);
    const structures = res.structures.map((s) => `<span class="chip" title="${esc([s.formula, s.inchikey, s.cid ? `CID ${s.cid}` : ''].filter(Boolean).join(' · '))}">${esc(sname.get(s.id))}
      ${s.links.lotus ? link(s.links.lotus, esc(s.lts), 'LOTUS') : ''}${s.links.wikidata ? ` ${link(s.links.wikidata, 'Wikidata')}` : ''}</span>`).join(' ');
    const paper = (p) => `<li>${esc([year(p.date), p.author].filter(Boolean).join(' · '))}${p.date || p.author ? ': ' : ''}${link(p.url, esc(p.title || p.doi || 'untitled'), p.doi || '')}
      ${p.journal ? `<span class="muted">${esc(p.journal)}</span>` : ''}${p.validated ? ' <span class="chip" title="Checked by hand by LOTUS">validated</span>' : ''}
      ${several ? `<span class="muted small">(${p.structures.map((s) => esc(sname.get(s) || s)).join(', ')})</span>` : ''}</li>`;
    const rows = res.organisms.map((o) => `<tr><td><i>${esc(o.name)}</i>${o.wikidata ? ` ${link(`https://www.wikidata.org/wiki/${o.wikidata}`, 'Wikidata')}` : ''}
        ${o.lineage ? `<div class="muted small">${esc(o.lineage)}${o.source ? ` (${esc(o.source)})` : ''}</div>` : ''}</td>
      <td><ol class="lt-papers">${o.papers.map(paper).join('')}</ol></td></tr>`).join('');
    const of = tx.level === 'organism' ? '' : ` of the ${esc(tx.level)}`;
    this.body.innerHTML = `<div class="mc-title"><h3>${esc(row.label)} in ${tx.level === 'organism' ? `<i>${esc(tx.name)}</i>` : esc(tx.name)}</h3>
        <span class="muted">${tx.level === 'organism' ? '' : `${esc(tx.level)} · `}${esc(tx.lineage || '')}${tx.source ? ` (as ${esc(tx.source)} has it)` : ''}</span></div>
      <p class="mc-meta">${res.n_organisms.toLocaleString()} organism${res.n_organisms === 1 ? '' : 's'}${of}, ${res.n_papers.toLocaleString()} paper${res.n_papers === 1 ? '' : 's'}.
        As LOTUS's structure${several ? 's' : ''}: ${structures}</p>
      <div class="mc-scroll"><table class="odor-table mc-table lt-table"><thead><tr><th>Organism</th><th>Papers reporting it there</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${res.n_organisms > res.organisms.length ? `<p class="muted small">The ${res.organisms.length} with the most papers shown.</p>` : ''}`;
  }
}
