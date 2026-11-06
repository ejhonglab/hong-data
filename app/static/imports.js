// Bundles of reference data imported in the Admin tab (bundle_import.py): MetaCyc's
// SmartTables or LOTUS's files, as the folder its build script reads (or an archive of it).
// Choosing the folder lists the files of it the script reads (found as the server finds
// them) and only those are uploaded (LOTUS's gzipped first, as they are kept); the import is
// followed until its reference datasets are added; and those kept are listed.

const NOTES = 'notes.txt';
const rel = (f) => f.webkitRelativePath || f.name;
const isGz = (name) => /\.gz$/i.test(name);

// a glob's part (as fnmatch: * any characters) as a regular expression
const globRe = (g) => new RegExp(`^${g.split('*').map((s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`);

// as lotus_import.place: where a file (its path in the folder) goes if a glob of the build
// finds it: the last parts of its path the glob's match (after a leading **/: its name), or null
function placeOf(path, pattern) {
  const parts = path.split('/');
  let globs = pattern.split('/');
  if (globs[0] === '**') globs = globs.slice(1);
  if (parts.length < globs.length) return null;
  const tail = parts.slice(parts.length - globs.length);
  if (tail.some((t) => t === '' || t === '.' || t === '..') || !tail.every((t, i) => globRe(globs[i]).test(t))) return null;
  return tail.join('/');
}

// The files of a folder the build reads: role -> file, and notes kept with them, as the
// server picks them: by name (metacyc_import.pick: the first of a role's files there; of
// several named alike, the one at the path it is read from), or by the build's globs
// (lotus_import.pick: of the files the first of a role's globs finds, gzipped or not, the last
// by where it goes, of one there both ways the one gzipped; and notes.txt beside it or in the
// folder above).
function pickFiles(files, expected, how) {
  const picked = {}, notes = [];
  if (how === 'name') {
    const byName = new Map();
    for (const f of files) byName.set(f.name, [...(byName.get(f.name) || []), f]);
    for (const e of expected) {
      const path = e.files.find((x) => byName.has(x.split('/').pop()));
      if (path) { const alike = byName.get(path.split('/').pop()); picked[e.role] = alike.find((f) => `/${rel(f)}`.endsWith(`/${path}`)) || alike[0]; }
    }
    return { picked, notes };
  }
  const paths = files.map(rel), first = new Map();
  paths.forEach((p, i) => { if (!first.has(p)) first.set(p, i); });
  for (const e of expected) {
    const found = new Map();
    for (const pattern of e.files) {
      paths.forEach((p, i) => {
        const place = placeOf(p, pattern) || placeOf(p, `${pattern}.gz`);
        if (place && !found.has(place)) found.set(place, i);
      });
      if (found.size) break;
    }
    if (!found.size) continue;
    const key = (p) => [p.replace(/\.gz$/, ''), p.endsWith('.gz') ? 1 : 0];
    const place = [...found.keys()].reduce((a, b) => { const [ka, kb] = [key(a), key(b)]; return kb[0] > ka[0] || (kb[0] === ka[0] && kb[1] > ka[1]) ? b : a; });
    const i = found.get(place);
    picked[e.role] = files[i];
    const parts = place.split('/'), path = paths[i].split('/'), lead = path.slice(0, path.length - parts.length);
    for (const up of [1, 2]) {
      if (parts.length < up) break;
      const j = first.get([...lead, ...parts.slice(0, parts.length - up), NOTES].join('/'));
      if (j !== undefined && !notes.includes(files[j])) notes.push(files[j]);
    }
  }
  return { picked, notes };
}

// A file gzipped, as a Blob (onProgress(bytes read) as it goes). Rejects with an Unreadable
// if it could not be read whole. Its Blob is made of its chunks, in memory: one built from the
// stream (as Response.blob() builds it) Chrome keeps on disk past a few MB, in its profile,
// and where that disk is short of room it fails, saying only "Failed to fetch".
async function gzipFile(file, onProgress) {
  let n = 0;
  const count = new TransformStream({ transform(chunk, ctl) { n += chunk.byteLength; onProgress(n); ctl.enqueue(chunk); } });
  const reader = file.stream().pipeThrough(count).pipeThrough(new CompressionStream('gzip')).getReader(), chunks = [];
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
  } catch (e) {
    throw new Unreadable(file, n, e);
  }
  // (a file cut short while it is read is read only so far, without an error)
  if (n !== file.size) throw new Unreadable(file, n, null);
  return new Blob(chunks);
}

// a file that could not be read here (as chosen), after `n` bytes of it were; e: what the
// browser said (Chrome: a TypeError, "network error", or a NotReadableError), if anything
class Unreadable extends Error {
  constructor(file, n, e) {
    super(e?.message || 'read short');
    Object.assign(this, { file, n, cause: e });
  }
}

// whether a file can be read here as it was chosen (its first byte): else rejects with an Unreadable
async function readable(file) {
  try { await file.slice(0, 1).arrayBuffer(); } catch (e) { throw new Unreadable(file, 0, e); }
}

export class BundleImport {
  // p: the prefix of its elements' IDs (#<p>-import-folder, -archive, -folder-btn,
  // -archive-btn, -tensor-btn, -tensor, -path, -files, -msg, -btn, -list); kind: as in
  // /api/admin/<kind>; fields(form):
  // its options added to the form (MetaCyc's version); noLabel: an import without its label,
  // as said; done: said once one is imported; refTitle(id): a reference dataset's title;
  // onAdded(): its reference datasets were added
  constructor(p, kind, { api, postForm, esc, msgBox, listHtml, fmtBytes, fmtWhen, fields, noLabel, done = '', refTitle, onAdded }) {
    Object.assign(this, { p, kind, api, postForm, esc, msgBox, listHtml, fmtBytes, fmtWhen, fields, noLabel, done, refTitle, onAdded });
    this.state = null; // as from GET /api/admin/<kind>
    this.chosen = null; // the folder (its files the build reads) or archive chosen; with `tensor`, its path on tensor
    this.timer = 0;
    this.watched = null; // the import started or seen under way here, to say how it ended
    this.$('folder-btn').addEventListener('click', () => this.$('folder').click());
    this.$('archive-btn').addEventListener('click', () => this.$('archive').click());
    for (const how of ['folder', 'archive']) {
      this.$(how).addEventListener('change', (e) => {
        const files = [...e.target.files];
        e.target.value = '';
        if (files.length) this.choose(files, how);
      });
    }
    // a path on tensor, checked there as it is typed (as data folders are)
    this.pathSeq = 0;
    this.$('tensor-btn').addEventListener('click', () => {
      const open = this.$('tensor').hidden;
      this._showPath(open);
      if (open) {
        if (this.chosen) { this.chosen = null; this.showFiles(); } // (one way of choosing at a time)
        this.$('path').focus();
        if (this.$('path').value.trim()) this.checkPath();
      }
    });
    this.$('path').addEventListener('input', () => { clearTimeout(this.pathTimer); this.pathTimer = setTimeout(() => this.checkPath(), 350); });
    this.$('btn').addEventListener('click', () => this.run());
  }

  // the path on tensor's field shown (or hidden, what it chose forgotten)
  _showPath(open) {
    this.$('tensor').hidden = !open;
    this.$('tensor-btn').setAttribute('aria-expanded', String(open));
    if (!open && this.chosen?.tensor) { this.chosen = null; this.showFiles(); }
  }

  // The path on tensor typed, checked there (`Importer.tensor_source`): found, the folder
  // (the files of it the build reads, as the server picks them) or archive it is, chosen
  async checkPath() {
    const input = this.$('path'), msg = input.nextElementSibling, value = input.value.trim(), seq = ++this.pathSeq;
    const say = (kind, text) => { msg.className = `path-msg${kind ? ` ${kind}` : ''}`; msg.textContent = text; input.classList.toggle('invalid', kind === 'bad'); };
    if (this.chosen?.tensor) { this.chosen = null; this.showFiles(); }
    if (!value) { say('', ''); return; }
    say('', 'checking on tensor…');
    let c;
    try { c = await this.api(`/api/admin/${this.kind}/tensor/check`, { method: 'POST', json: { path: value } }); } catch (e) {
      if (seq === this.pathSeq) say('bad', `could not be checked: ${e.message}`);
      return;
    }
    if (seq !== this.pathSeq) return; // (typed on since)
    if (!c.ok) { say('bad', c.error); return; }
    say('ok', `✓ found on tensor: ${c.kind === 'archive' ? `an archive (${this.fmtBytes(c.size)})` : `a folder of ${c.n.toLocaleString()} files`}`);
    if (c.kind === 'archive') this.chosen = { tensor: c.path, archive: { name: c.path, size: c.size } };
    else {
      // (as a browser has the files of a folder chosen: named by their paths, after its name)
      const files = c.files.map((f) => ({ name: f.path.split('/').pop(), webkitRelativePath: f.path, size: f.size, lastModified: f.mtime * 1000 }));
      this.chosen = { tensor: c.path, folder: c.path, n: c.n, ...pickFiles(files, this.state?.expected || [], this.state?.how) };
    }
    this.$('msg').replaceChildren();
    this.showFiles();
  }

  $(what) { return document.getElementById(`${this.p}-import-${what}`); }

  // the files an upload is to have, the import under way (or the last), and those kept
  async load() {
    let s;
    try { s = await this.api(`/api/admin/${this.kind}`); } catch (e) {
      this.$('msg').replaceChildren(this.msgBox('error', `Could not load the imports: ${this.esc(e.message)}`));
      return;
    }
    this.state = s;
    this.showFiles();
    this.showJob(s.job);
    this.showImports();
  }

  // the import kept (of those listed) a reference dataset is of, if any
  importOf(refId) { return this.state?.imports.find((i) => i.refs.includes(refId)); }

  // A folder chosen: the files of it the build reads, by role, as the server picks them;
  // only these are uploaded. An archive: only the server looks into it.
  choose(files, how) {
    this._showPath(false);
    if (how === 'archive') this.chosen = { archive: files[0] };
    else this.chosen = { folder: rel(files[0]).split('/')[0], n: files.length, ...pickFiles(files, this.state?.expected || [], this.state?.how) };
    this.$('msg').replaceChildren();
    this.showFiles();
  }

  // whether a file of the folder chosen is gzipped here before it is uploaded (as the server
  // keeps it; it gzips those this browser cannot)
  _gzipHere(f) { return !this.chosen?.tensor && !!this.state?.gzip && typeof CompressionStream === 'function' && !isGz(f.name) && f.name !== NOTES; }

  // whether a file of a folder on tensor is gzipped by the server as it is kept
  _gzipThere(f) { return !!this.chosen?.tensor && !!this.state?.gzip && !isGz(f.name) && f.name !== NOTES; }

  // A file that could not be read here (an Unreadable), said: which, how far it was read,
  // and why that may be. The browser cannot tell why, but for the bytes read: none, it is not
  // as it was when chosen (changed, moved or gone since) or its drive would not give it; some,
  // its drive stopped giving it, or it changed while it was read.
  _unreadable(u) {
    const { esc, fmtBytes } = this, f = u.file, name = `<code>${esc(rel(f))}</code>`;
    const said = u.cause ? ` <span class="muted">(the browser said: ${esc(u.cause.name && u.cause.name !== 'TypeError' ? `${u.cause.name}: ` : '')}${esc(u.cause.message)})</span>` : '';
    const drive = 'the drive it is on (such as a network mount, e.g. over sshfs)';
    const why = u.n ? `${name} could be read here only to ${fmtBytes(u.n)} of its ${fmtBytes(f.size)}: ${drive} stopped giving it, or it changed while it was read`
      : `${name} could not be read here as it was when chosen: it changed, moved or was deleted since, or ${drive} would not give it`;
    return `${why}.${said} Choose the ${f.webkitRelativePath ? 'folder' : 'archive'} again; if it fails the same way, copy it to a disk of this computer and choose it there,
      or, if it is on tensor, give its path there (<i>or a path on tensor…</i>), for the server to read it.`;
  }

  // what was chosen, against what the build reads (before anything is uploaded)
  showFiles() {
    const { esc, fmtBytes } = this, box = this.$('files'), c = this.chosen, s = this.state;
    const expected = `<details class="imp-expected"><summary>The files it reads</summary>${this._table(null)}</details>`;
    if (!c) box.innerHTML = expected;
    else if (c.tensor && c.archive) {
      box.innerHTML = `<p><b>${esc(c.tensor)}</b> on tensor (${fmtBytes(c.archive.size)}) is read there by the server, nothing uploaded: it takes the files it reads
        out of it${s?.gzip ? ' (gzipping those that are not)' : ''}, and checks them before it builds anything.</p>${expected}`;
    } else if (c.tensor) {
      const files = [...Object.values(c.picked), ...c.notes], missing = this.missing();
      box.innerHTML = `<p>Of the ${c.n.toLocaleString()} files in <b>${esc(c.tensor)}</b> on tensor, the ${files.length} it reads are copied from there by the
        server (${fmtBytes(files.reduce((a, f) => a + f.size, 0))}${files.some((f) => this._gzipThere(f)) ? ', gzipped as they are kept' : ''}), nothing uploaded;
        the others are not.</p>${this._table(c.picked)}${c.notes.length
        ? `<p class="muted small">With them, the notes beside them: ${c.notes.map((f) => `<code>${esc(rel(f))}</code>`).join(', ')}.</p>` : ''}`;
      if (missing.length) {
        box.append(this.msgBox('error', `<b>Not ${esc(s.what)} as <code>${esc(s.script)}</code> reads them</b>: ${missing.length === 1 ? 'one is' : `${missing.length} are`}
          missing (above). Give the path of the folder of them as <code>${esc(s.like)}</code> has them${s.how === 'name' ? ', or of one with files of these names in it' : ', with its subfolders'}.`));
      }
    } else if (c.archive) {
      box.innerHTML = `<p><b>${esc(c.archive.name)}</b> (${fmtBytes(c.archive.size)}) is uploaded whole: the server takes the files it reads out of it${s?.gzip ? ' (gzipping those that are not)' : ''},
        and checks them before it builds anything.</p>${expected}`;
    } else {
      const files = [...Object.values(c.picked), ...c.notes], missing = this.missing();
      const size = files.reduce((a, f) => a + f.size, 0), gz = files.filter((f) => this._gzipHere(f));
      box.innerHTML = `<p>Of the ${c.n.toLocaleString()} files in <b>${esc(c.folder)}</b>, the ${files.length} it reads are uploaded (${fmtBytes(size)}${gz.length
        ? `, ${fmtBytes(gz.reduce((a, f) => a + f.size, 0))} of it gzipped first, as it is kept` : ''}); the others are not.</p>${this._table(c.picked)}${c.notes.length
        ? `<p class="muted small">With them, the notes beside them: ${c.notes.map((f) => `<code>${esc(rel(f))}</code>`).join(', ')}.</p>` : ''}`;
      if (missing.length) {
        box.append(this.msgBox('error', `<b>Not ${esc(s.what)} as <code>${esc(s.script)}</code> reads them</b>: ${missing.length === 1 ? 'one is' : `${missing.length} are`}
          missing (above). Choose the folder of them as <code>${esc(s.like)}</code> has them${s.how === 'name' ? ', or one with files of these names in it' : ', with its subfolders'}.`));
      }
    }
    this._syncButton();
  }

  // the roles required that the folder chosen has no file for
  missing() {
    const c = this.chosen;
    return c?.picked ? (this.state?.expected || []).filter((e) => e.required && !c.picked[e.role]) : [];
  }

  _syncButton() {
    this.$('btn').disabled = !this.chosen || !!this.state?.job.running || this.missing().length > 0;
  }

  // each file the build reads: what it is, and where it reads it from (or, of a folder
  // chosen, the file of it uploaded for it, if any)
  _table(picked) {
    const { esc, fmtBytes } = this, s = this.state;
    const rows = (s?.expected || []).map((e) => {
      const where = e.files.map((x) => `<code>${esc(x)}</code>`).join(' or ') + (s.how === 'pattern' ? ' <span class="muted">(gzipped or not)</span>' : '');
      let cell = where;
      if (picked?.[e.role]) {
        const f = picked[e.role];
        cell = `✓ <code>${esc(rel(f))}</code> <span class="muted">(${fmtBytes(f.size)}${this._gzipHere(f) ? ', gzipped first' : this._gzipThere(f) ? ', gzipped as it is kept' : ''})</span>`;
      } else if (picked) cell = e.required ? `<span class="bad">missing</span>: ${where}` : `<span class="muted">not there (${where})</span>`;
      return `<tr><td>${esc(e.about)}${e.required ? '' : ' <span class="muted">(if there)</span>'}</td><td>${cell}</td></tr>`;
    });
    return `<table class="odor-table imp-files"><thead><tr><th>It reads</th><th>${picked ? (this.chosen?.tensor ? 'Copied from tensor' : 'Uploaded') : 'From (in the folder)'}</th></tr></thead>
      <tbody>${rows.join('')}</tbody></table>`;
  }

  async run() {
    const { esc, msgBox, fmtBytes } = this, c = this.chosen, out = this.$('msg');
    if (!c) return;
    this.$('btn').disabled = true;
    if (c.tensor) { await this._runFromTensor(c); return; }
    const form = new FormData(), mtimes = {}; // (when each was last changed, kept as theirs)
    try {
      if (c.archive) { await readable(c.archive); form.append('file', c.archive, c.archive.name); }
      for (const f of c.archive ? [] : [...Object.values(c.picked), ...c.notes]) {
        // (one sent as it is is read only as it is uploaded, where a file that cannot be read looks like a server that cannot be reached)
        if (!this._gzipHere(f)) { await readable(f); form.append('file', f, rel(f)); mtimes[rel(f)] = f.lastModified; continue; }
        let shown = -1;
        const blob = await gzipFile(f, (n) => {
          const pct = Math.floor((100 * n) / (f.size || 1));
          if (pct !== shown) out.replaceChildren(msgBox('info', `<b>Gzipping ${esc(f.name)} before it is uploaded…</b> ${fmtBytes(n)} of ${fmtBytes(f.size)} (${shown = pct}%)`));
        });
        form.append('file', blob, `${rel(f)}.gz`);
        mtimes[`${rel(f)}.gz`] = f.lastModified;
      }
    } catch (e) {
      out.replaceChildren(msgBox('error', `<b>Not imported</b> (nothing was uploaded): ${e instanceof Unreadable ? this._unreadable(e)
        : `a file could not be read here (${esc(e.message)})`}`));
      this._syncButton();
      return;
    }
    form.append('mtimes', JSON.stringify(mtimes));
    this.fields?.(form);
    out.replaceChildren(msgBox('info', '<b>Uploading…</b>'));
    let res;
    try {
      res = await this.postForm(`/api/admin/${this.kind}`, form, (sent, total) => out.replaceChildren(msgBox('info',
        `<b>Uploading…</b> ${fmtBytes(sent)} of ${fmtBytes(total)} (${Math.floor((100 * sent) / total)}%)`)));
    } catch (e) { res = { ok: false, errors: [e.message] }; }
    if (!res.ok) {
      out.replaceChildren(msgBox('error', `<b>Not imported</b> (nothing was kept):${this.listHtml(res.errors)}`));
      this._syncButton();
      return;
    }
    this.chosen = null;
    this.watched = res.job.id;
    this.state.job = res.job;
    this.showFiles();
    this.showJob(res.job);
  }

  // An import of a path on tensor started: the server reads it there (in the background,
  // as `showJob` follows), nothing uploaded
  async _runFromTensor(c) {
    const { esc, msgBox } = this, out = this.$('msg'), opts = new FormData();
    this.fields?.(opts);
    out.replaceChildren(msgBox('info', `<b>Starting the import of <code>${esc(c.tensor)}</code>, on tensor…</b>`));
    let res;
    try { res = await this.api(`/api/admin/${this.kind}/tensor`, { method: 'POST', json: { ...Object.fromEntries(opts), path: c.tensor } }); } catch (e) {
      res = { ok: false, errors: [e.message] };
    }
    if (!res.ok) {
      out.replaceChildren(msgBox('error', `<b>Not imported</b> (nothing was kept):${this.listHtml(res.errors)}`));
      this._syncButton();
      return;
    }
    this.chosen = null;
    this.$('path').value = '';
    this.$('path').nextElementSibling.textContent = '';
    this._showPath(false);
    this.watched = res.job.id;
    this.state.job = res.job;
    this.showFiles();
    this.showJob(res.job);
  }

  // the import under way, checked again until it ends; then how it ended, if it was
  // started or seen under way here
  showJob(job) {
    const { esc, msgBox, listHtml } = this, out = this.$('msg'), name = this.state.name, v = job.label ? ` ${esc(job.label)}` : '';
    clearTimeout(this.timer);
    if (job.running) {
      this.watched = job.id;
      const stage = { uploading: 'Its files are being uploaded', copying: 'Its files are being copied from tensor', building: 'Building its bundle',
        adding: 'Adding it to the Reference data tab' }[job.stage];
      out.replaceChildren(msgBox('info', `<b>Importing ${esc(name)}${v}: ${stage || 'under way'}…</b> (started ${esc(this.fmtWhen(job.started_at))})${
        job.note ? `<p class="small">${esc(job.note)}</p>` : ''}${job.log.length ? listHtml(job.log.slice(-3)) : ''}`));
      this.timer = setTimeout(() => this.poll(), 1500);
      return;
    }
    if (!job.id || job.id !== this.watched) return;
    this.watched = null;
    const errors = (job.error || '').split('\n').filter(Boolean).map((x) => x.replace(/^- /, ''));
    if (!job.added.length) {
      out.replaceChildren(msgBox('error', `<b>Not imported</b> (nothing was kept):${listHtml(errors)}`));
      return;
    }
    const imp = this.state.imports.find((i) => i.id === job.id);
    const links = job.added.map((a) => `<a href="#reference/${esc(a.id)}">${esc(a.title)}</a>${a.state === 'ok' ? ''
      : ` (${a.state === 'pending' ? 'read once the server restarts' : `but it cannot be read: ${esc(a.error)}`})`}`);
    out.replaceChildren(msgBox(errors.length ? 'warn' : 'ok', `<b>Imported ${esc(name)}${v}</b>: ${links.join(' and ')}, in the Reference data tab.${
      imp ? ` The files ${imp.tensor_path ? 'copied from tensor' : 'uploaded'} and its bundle are kept in <code>${esc(imp.dir)}</code>.` : ''}${this.done}${
      job.note ? `<p class="small">${esc(job.note)}</p>` : ''}${errors.length ? listHtml(errors) : ''}`));
    this.onAdded?.();
  }

  async poll() {
    let s;
    try { s = await this.api(`/api/admin/${this.kind}`); } catch {
      this.timer = setTimeout(() => this.poll(), 5000); // (the server restarting, say)
      return;
    }
    this.state = s;
    this.showJob(s.job);
    if (!s.job.running) { this.showImports(); this._syncButton(); }
  }

  // the imports kept, with their reference datasets
  showImports() {
    const { esc } = this, box = this.$('list'), list = this.state?.imports || [];
    if (!list.length) { box.replaceChildren(); return; }
    box.innerHTML = `<h4>Imported here</h4><table class="odor-table"><thead><tr><th>${esc(this.state.name)}</th><th>Uploaded</th><th>Kept in</th><th>Its reference datasets</th></tr></thead>
      <tbody>${list.map((i) => `<tr><td>${i.note ? `<details class="imp-note"><summary>${esc(i.label || this.noLabel)}</summary><p class="small">${esc(i.note)}</p></details>`
        : esc(i.label || this.noLabel)}</td><td>${esc(this.fmtWhen(i.uploaded_at))}${i.tensor_path
          ? `<br><span class="muted small">from <code>${esc(i.tensor_path)}</code> on tensor</span>` : ''}</td>
        <td><code>${esc(i.dir)}</code> <span class="muted">(${this.fmtBytes(i.size)})</span></td>
        <td>${i.refs.length ? i.refs.map((id) => `<a href="#reference/${esc(id)}">${esc(this.refTitle(id))}</a>`).join('<br>') : '<span class="muted">none</span>'}</td></tr>`).join('')}</tbody></table>`;
  }
}
