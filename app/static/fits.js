// Fits of al-diag-seg models in the AL segmentation tab (fits.py, fit_worker.py): asking for one
// (as the name set in Settings), those queued and running (who asked, from where, when, how far,
// and when it should end), and past ones above them (their scores on the held-out
// flies, and their model to download, or to delete).

const STAGE_NAMES = { load: 'Finding and loading the flies', train: 'Training Cellpose', atlas: 'Building the atlas',
  evaluate: 'Scoring the held-out flies' };
const STATUS_NAMES = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed', cancelled: 'Stopped',
  interrupted: 'Interrupted' };
const REF_NAMES = { grabe2016: 'Grabe et al. 2016' }; // fits.REFERENCES

// a fit's reference atlases, by name
const refNames = (o) => (o.refs || []).map((r) => REF_NAMES[r] || r);

// a duration in seconds as '1 h 5 min', '12 min', '40 s'
export function fmtDuration(s) {
  if (s == null || !Number.isFinite(s)) return '—';
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ''}`;
}

const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
const num2 = (v) => (v == null ? '—' : v.toFixed(2));
// the final ROIs, by their kind (al-diag-seg's)
const FINAL_NAMES = { pred: "Cellpose's own, named", refined_prior: 'refined with the atlas', refined_tuning: 'refined by tuning' };

// a score plane by plane and as 3D objects ('46% · 30%'), or as 3D objects alone (fits before
// al-diag-seg scored them plane by plane)
export function both(x, key, fmt = pct) {
  return x[`${key}_2d`] == null ? fmt(x[key]) : `${fmt(x[`${key}_2d`])} <span class="muted">· ${fmt(x[key])}</span>`;
}

// how a fit's or a prediction's ROIs were scored (al-diag-seg's): as glomeruli drawn on some of their
// planes only (often their clearest), so not counted wrong where none is drawn; or strictly (before)
export const partlyDrawn = (x) => x.scoring === 'partly drawn';
export const scoringNote = (x) => (partlyDrawn(x) ? 'not counted wrong where no glomerulus is drawn'
  : 'scored strictly: also counted wrong where no glomerulus is drawn');

export class FitsPanel {
  // helpers: { $, esc, api, msgBox, askDialog, fmtWhen, fmtSize, getName (the name in Settings, '' if none) }
  constructor(h) {
    this.h = h;
    this.data = null;
    this.timer = null;
    this.open = new Set(); // past fits whose details are shown
    const { $ } = h;
    $('#fit-start').addEventListener('click', () => this.start());
    const atlas = $('#fit-atlas'), grabe = $('#fit-grabe');
    const sync = () => { grabe.disabled = !atlas.checked; grabe.closest('label').classList.toggle('disabled', !atlas.checked); };
    atlas.addEventListener('change', sync);
    sync();
    $('#fits-list').addEventListener('click', (e) => this._action(e));
    $('#fits-past').addEventListener('click', (e) => this._action(e));
  }

  // while the tab is shown: every 5 s if a fit is queued or running, else every 30 s
  async load() {
    await this.refresh();
    if (this.timer) return;
    let n = 0;
    this.timer = setInterval(() => {
      if (this.h.$('#tab-al-segmentation').hidden) return;
      n += 1;
      const busy = this.data?.fits.some((f) => f.status === 'queued' || f.status === 'running');
      if (busy || n % 6 === 0) this.refresh();
    }, 5000);
  }

  async refresh() {
    try { this.data = await this.h.api('/api/fits'); } catch (e) {
      this.h.$('#fit-worker').replaceChildren(this.h.msgBox('error', `Could not ask the server about fits: ${this.h.esc(e.message)}`));
      return;
    }
    this.render();
  }

  render() {
    const { $, esc, msgBox, fmtWhen } = this.h;
    const name = this.h.getName();
    $('#fit-who').innerHTML = name
      ? `Fits say who asked for them: this one will be <b>${esc(name)}</b>'s (as set in <a href="#settings/name">Settings</a>).`
      : 'Enter your first name in <a href="#settings/name">Settings</a> first: fits say who asked for them.';
    $('#fit-start').disabled = !name;
    const w = this.data?.worker;
    $('#fit-worker').replaceChildren(!w ? '' : w.alive
      ? msgBox('ok', `The fit worker is running on ${esc(w.host || '?')}: ${w.gpu ? `GPU ${esc(w.gpu)}` : 'no GPU (training would be very slow)'}${
        w.alseg?.commit ? `, al-diag-seg <code>${esc(w.alseg.commit.slice(0, 7))}</code>` : ''}.`)
      : msgBox('warn', `No fit worker is running${w.seen ? ` (last heard from ${esc(fmtWhen(w.seen))})` : ''}: fits asked for wait
        until one starts. It runs in a container of its own, with the GPU (DEPLOY.md).`));
    const fits = this.data?.fits || [];
    const active = fits.filter((f) => f.status === 'queued' || f.status === 'running').reverse(); // oldest first
    $('#fits-list').innerHTML = active.length ? active.map((f) => this._activeHtml(f)).join('') : '';
    const past = fits.filter((f) => f.status !== 'queued' && f.status !== 'running');
    $('#fits-past-step').hidden = !past.length;
    $('#fits-past').innerHTML = past.length ? `<table class="odor-table fits-table"><thead><tr><th>Fit</th><th>Asked for by</th>
      <th>From</th><th>Started</th><th>Took</th><th>Status</th><th title="Of the held-out flies, the final ROIs: found (IoU ≥ 0.5) of their own ROIs, and of the ROIs predicted; mean Dice of those found (each plane by plane: a glomerulus's outline on each plane, as ROIs are drawn; then as 3D objects: its outlines on all its planes as one); and ROIs named as the ROI of theirs they overlap">Held-out flies:
      found, right, Dice <span class="muted">(per plane · 3D)</span>, named</th><th></th></tr></thead><tbody>${past.map((f) => this._pastHtml(f)).join('')}</tbody></table>` : '';
  }

  _who(f) {
    const { esc, fmtWhen } = this.h;
    return `<b>${esc(f.name)}</b> <span class="muted">from ${esc(f.ip || '?')}</span>, asked ${esc(fmtWhen(f.created_at))}`;
  }

  _activeHtml(f) {
    const { esc, fmtWhen } = this.h;
    const p = f.progress || {};
    const stage = f.status === 'queued' ? 'Waiting for the worker' : (STAGE_NAMES[p.stage] || 'Starting');
    const frac = p.total ? Math.min(1, p.done / p.total) : 0;
    const eta = f.eta ? `ends about ${esc(fmtWhen(f.eta.ends_at))} (in ${fmtDuration(f.eta.until)}${f.status === 'queued'
      ? `, after those before it: it takes about ${fmtDuration(f.eta.seconds)}` : ''}; ${f.eta.from
      ? `estimated from how long the last ${f.eta.from === 1 ? 'fit' : `${f.eta.from} fits`} took`
      : 'a rough guess: no fit has finished yet to estimate from'})` : '';
    const o = f.options || {};
    return `<div class="fit-active" data-id="${f.id}">
      <div class="fit-head"><span><b>Fit #${f.id}</b> · ${this._who(f)}${f.started_at ? `, started ${esc(fmtWhen(f.started_at))}` : ''}</span>
        <button type="button" class="btn small" data-act="cancel" ${f.cancel ? 'disabled' : ''}>${f.cancel ? 'Stopping…' : f.status === 'queued' ? 'Cancel' : 'Stop'}</button></div>
      <div class="fit-stage">${esc(stage)}${p.total ? ` <span class="muted">(${p.done} of ${p.total})</span>` : ''}</div>
      <progress max="1" value="${frac}"></progress>
      <div class="muted small">${eta}${eta ? ' · ' : ''}${o.held_out} held out, ${o.epochs} epochs${o.atlas
        ? `, atlas${refNames(o).map((r) => ` + ${esc(r)}`).join('')}` : ', no atlas'}${o.tentative ? '' : ', no tentative ROIs'}
        ${f.status === 'running' ? ` · <a href="/api/fits/${f.id}/log" target="_blank" rel="noopener">log</a>` : ''}</div></div>`;
  }

  _pastHtml(f) {
    const { esc, fmtWhen, fmtSize } = this.h;
    const took = f.started_at && f.finished_at ? (new Date(f.finished_at) - new Date(f.started_at)) / 1000 : null;
    const r = f.result;
    const scores = r ? `${both(r.mean, 'recall')}, ${both(r.mean, 'precision')}, ${both(r.mean, 'dice', num2)}, ${r.named}/${r.named_of}${partlyDrawn(r)
      ? '' : ` <span class="muted" title="${scoringNote(r)}">(strict)</span>`}` : '—';
    const status = `${STATUS_NAMES[f.status] || esc(f.status)}${f.error ? ` <span class="muted" title="${esc(f.error)}">(${esc(f.error.slice(0, 80))}${f.error.length > 80 ? '…' : ''})</span>` : ''}`;
    const open = this.open.has(f.id);
    return `<tr data-id="${f.id}"><td>#${f.id}</td><td>${esc(f.name)}</td><td>${esc(f.ip || '?')}</td>
      <td>${f.started_at ? esc(fmtWhen(f.started_at)) : '—'}</td><td>${fmtDuration(took)}</td><td>${status}</td>
      <td>${r ? `<button type="button" class="link" data-act="details">${scores}</button>` : scores}</td>
      <td class="fit-acts">${f.status === 'done' ? `<a class="btn small" href="/api/fits/${f.id}/download" download>Download${f.model_bytes ? ` (${fmtSize(f.model_bytes)})` : ''}</a>` : ''}
        <a class="btn small" href="/api/fits/${f.id}/log" target="_blank" rel="noopener">Log</a>
        <button type="button" class="btn small danger" data-act="delete">Delete</button></td></tr>
      ${r && open ? `<tr class="fit-details"><td colspan="8">${this._detailsHtml(f)}</td></tr>` : ''}`;
  }

  _detailsHtml(f) {
    const { esc } = this.h;
    const r = f.result, o = f.options || {};
    const flies = r.flies.map((x) => `<tr><td>${esc(x.fly)}</td><td>${x.n}</td><td>${both(x, 'recall')}</td><td>${both(x, 'precision')}</td>
      <td>${both(x, 'dice', num2)}</td><td>${num2(x.leak)} <span class="muted">(${num2(x.manual_leak)})</span></td><td>${x.named}/${x.named_of}</td></tr>`).join('');
    const missed = r.glomeruli.filter((g) => g.detected < g.flies);
    const refs = refNames(o).map(esc);
    return `<p class="small">${f.flies ? `Fit to ${f.flies.train.length} flies, ${o.held_out} held out, ${o.epochs} epochs${o.atlas
      ? `, with the atlas${refs.length ? ` (and ${refs.join(' and ')}'s in it)` : ''}` : ''}.` : ''}
      Scored are the final ROIs (${esc(FINAL_NAMES[r.final] || r.final)}), against each held-out fly's own${r.mean.recall_2d == null ? ', as 3D objects'
      : ": plane by plane (each glomerulus's outline on a plane, as ROIs are drawn), then as 3D objects (its outlines on all its planes as one)"}.
      ${partlyDrawn(r) ? `As glomeruli are often drawn on their clearest planes only, nothing counts against an ROI where nothing is drawn:
        on planes its glomerulus has no outline on, only its part in another glomerulus's outline does, and ROIs lying mostly
        outside every drawn glomerulus are left out of Right.` : `Scored strictly, as before that changed: an ROI's parts on planes its
        glomerulus has no outline on count against it, as does every ROI where no glomerulus is drawn, so these are lower than a newer
        fit's would be.`}</p>
      <table class="odor-table"><thead><tr><th>Held-out fly</th><th>ROIs</th>${['Found', 'Right', 'Dice'].map((h) => `<th>${h}${r.mean.recall_2d == null
        ? '' : ' <span class="muted">(per plane · 3D)</span>'}</th>`).join('')}
      <th title="Signal from neighbors in an ROI's border (its manual ROIs' in brackets)">Leak <span class="muted">(manual)</span></th><th>Named</th></tr></thead><tbody>${flies}</tbody></table>
      ${missed.length ? `<p class="small">Glomeruli not found (as 3D objects) in every held-out fly with one: ${missed.map((g) => `${esc(g.glomerulus)}
        <span class="muted">${g.detected}/${g.flies}</span>`).join(', ')}.</p>` : '<p class="small">Every glomerulus of the held-out flies was found.</p>'}`;
  }

  async _action(e) {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const id = Number(b.closest('[data-id]').dataset.id);
    const { $, api, msgBox, esc, askDialog } = this.h;
    if (b.dataset.act === 'details') {
      if (this.open.has(id)) this.open.delete(id); else this.open.add(id);
      this.render();
      return;
    }
    try {
      if (b.dataset.act === 'cancel') {
        await api(`/api/fits/${id}/cancel`, { method: 'POST' });
      } else if (b.dataset.act === 'delete') {
        const f = this.data.fits.find((x) => x.id === id);
        $('#delete-confirm-q').textContent = `Delete fit #${id}, ${f.name}'s?`;
        $('#delete-confirm-detail').textContent = 'Its model, its results and its log are deleted from the server, for everyone.';
        $('#delete-confirm-warn').replaceChildren();
        if (await askDialog($('#delete-confirm')) !== 'delete') return;
        await api(`/api/fits/${id}`, { method: 'DELETE' });
        this.open.delete(id);
      }
    } catch (err) {
      $('#fit-status').replaceChildren(msgBox('error', esc(err.message)));
    }
    await this.refresh();
  }

  async start() {
    const { $, api, msgBox, esc } = this.h;
    const name = this.h.getName();
    if (!name) { this.render(); return; }
    const atlas = $('#fit-atlas').checked;
    const options = { held_out: $('#fit-held-out').value, epochs: $('#fit-epochs').value, atlas,
      tentative: $('#fit-tentative').checked, refs: atlas && $('#fit-grabe').checked ? ['grabe2016'] : [] };
    $('#fit-start').disabled = true;
    try {
      const f = await api('/api/fits', { method: 'POST', json: { name, options } });
      $('#fit-status').replaceChildren(msgBox('ok', `Fit #${f.id} asked for: it is below while it waits and runs.`));
    } catch (err) {
      $('#fit-status').replaceChildren(msgBox('error', `Could not ask for a fit: ${esc(err.message)}`));
    }
    await this.refresh();
  }
}
