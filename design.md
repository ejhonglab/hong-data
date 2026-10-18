# Design

How the app is put together, and why: the decisions that shape the code, where each part
lives, how data is stored, and the HTTP API between the frontend and the server. Using
the app is described in [README.md](README.md), hosting it in [DEPLOY.md](DEPLOY.md).

## Main decisions

- **One small server, the heavy work in the browser.** A single Starlette app, served by
  uvicorn with one worker (one process avoids SQLite write contention), keeps everything
  in SQLite plus files under `data/`, in one container. The server parses, validates,
  stores and serves matrices; displaying, clustering and figure export all happen in the
  browser. The frontend is plain JavaScript modules, with no build step and no
  dependencies (the figure export writes its PDFs itself for that reason).
- **Intranet only, no login.** The network is trusted and nothing secret is entered, so
  users are just names: a user's tensor folders are saved on the server under their
  name, upload form defaults in the browser (localStorage). It is plain HTTP unless a
  reverse proxy adds TLS (DEPLOY.md, section 6), hence the browser warnings in the
  README, and web workers cannot share memory (that needs HTTPS and cross-origin
  isolation), which shapes the clustering below. The question the Admin tab asks first
  (are you Tom, Jon or Betty, or did one of them ask you) is a reminder, not a lock; a
  reverse proxy can put a password on `/api/admin`.
- **Uploads are staged.** A dropped file is parsed and validated into
  `data/staging/<id>/` and shown before anything is stored. Odor YAMLs, the
  `Experiment.xml`, tensor lookups and README generation all act on the staged upload,
  and only the final commit writes to the database and `data/datasets/<id>/`. Stages
  never saved are deleted after `OLFDB_STAGE_TTL_HOURS`. Validation collects every
  problem it can find, so a file can be fixed in one go.
- **Original files are kept verbatim** (data file, odor YAMLs, `Experiment.xml`), and
  every form value is stored on its own (database and `metadata.json`), apart from the
  README, which can be edited freely. Anything parsed from them can be redone.
- **Each matrix is stored twice**: as a long `responses` table (one row per cell) for
  SQL queries, and as a float32 blob (`matrices`) for fast serving to the viewer.
- **Row levels that say nothing are dropped on upload** (`ingest.uninformative_levels`),
  and from the trials of datasets stored before that whenever they are read
  (`ingest.drop_uninformative`), so old and new datasets are shown and downloaded alike.
- **Odor strings follow `hong2p.olf`** (`app/odors.py` is a small port of it), so
  matrices written by hong2p-based analysis can be uploaded as they are.
- **Tensor is read by the server**, mounted read-only, with users' folders given
  relative to its top, so the same paths work from any computer and the browser never
  has to reach the NAS. Flies are looked for in hong2p's layout
  (`<folder>/<date>/<fly>/<recording>/Experiment.xml`, with dates and fly folders in
  several spellings). The `Experiment.xml` of the first recording is attached, and odor
  YAMLs are picked greedily until they describe every odor in the matrix: each time the
  one describing the most odors not yet covered, preferring, on ties, ones named in the
  recordings' ThorImage notes over ones found by a fly's date in their path.
- **The folders the server may use are limited in two layers.** The deployment decides
  what it can reach at all (the container's mounts, a read-only image, no capabilities;
  or the systemd unit's sandboxing), which the app cannot change. Within that, the Admin
  tab sets folders it may read and folders it may also write (`app/sandbox.py`, saved in
  `data/admin.json`), fixed when the server starts: the app checks the paths it reads on
  tensor against them (to explain refusals), and has the kernel hold it to them through
  Landlock, its code, Python and system folders readable too, and its data folder the
  only other one writable. A process cannot lift Landlock, so changes wait for a
  restart, which the Admin tab asks for by stopping the server (SIGTERM, as `docker
  stop` does) for Docker's restart policy or systemd to start it again: a server that
  could widen its own limits while running would not be limited. So that it writes
  nothing outside its data folder, temporary files go to `data/tmp/`, and SQLite keeps
  its temporary tables in memory.
- **The laser wavelength is never read from `Experiment.xml`**: ThorImage's
  `MultiPhotonLaser` value does not follow changes made to the wavelength at the laser,
  so the wavelength is only ever entered in the form.
- **The odor table is the app's own copy** of a Google sheet, synced one way. The sheet
  is read through its CSV export (shared by link: no credentials, nothing written back),
  its link set in the Admin tab and kept in the database (`settings`), so that it is
  used at once and backed up with everything else (`OLFDB_ODOR_SHEET_URL` is the
  fallback).
  A sync is two steps: a preview validates the sheet and diffs it against the database,
  then an apply adds the new odors plus only the changes and deletions a user confirmed,
  logged in `odor_syncs`. The rest of the app uses the database copy, so it keeps
  working without the sheet. A dataset's odors are matched to the table through its odor
  YAML when it has one, otherwise by abbreviation. PubChem structure images are fetched
  once per CID and cached in `data/odor_structures/`, InChIs in the `pubchem` table (a
  hundred CIDs per request).
- **The heatmap is a virtualised canvas**: only the visible cells are drawn, so 10k+
  rows scroll smoothly. The default colormap is built in Oklab so that lightness is
  linear in |value|, with white at 0 and blue and red poles of equal lightness: "twice
  as dark" means "twice as large" on either side, which the ColorBrewer and Moreland
  maps only approximate.
- **Clustering runs in web workers.** Distances use only the entries both vectors have
  (like pandas' `DataFrame.corr`), so missing values are fine. They are computed in
  blocks of rows spread over the workers, and average linkage (UPGMA, merges in scipy's
  convention) then runs in one worker directly on those blocks, never copied into one
  buffer; blocks are transferred between workers, as they cannot share memory. Memory
  limits it to 15,000 rows.
- **Figure export** embeds the cells as an image drawn without smoothing, so they stay
  crisp at any zoom, while text and lines stay vector in SVG and PDF.
- **Downloads** are built in `data/exports/` and deleted once sent. Across datasets,
  trials are matched on panel, odor components and repeat.
- **Reference matrices** are read from `reference_data/` at request time, so adding or
  editing one needs no restart.

## Code layout

```
app/main.py        HTTP API (Starlette) and static file serving
app/config.py      runtime configuration (environment variables)
app/ingest.py      CSV/parquet loading + validation → units, trials, float32 matrix
app/odors.py       odor string parsing/formatting (port of the relevant olf.py bits)
app/glomeruli.py   accepted glomerulus names (config/glomeruli.txt overrides)
app/yamlcheck.py   odor YAML parsing and the matrix ⊂ YAML check
app/thorimage.py   Experiment.xml summary
app/metadata.py    form model (pydantic) and README.md generation
app/db.py          SQLite schema and queries
app/reference.py   fixed reference matrices
app/export.py      downloads: zips of dataset files, concatenated wide/long matrices
app/odormeta.py    odor table: sheet parsing / validation, sync diff, links, dataset matching
app/tensor.py      users' folders on tensor: path checks, finding a dataset's flies, recordings, YAMLs
app/sandbox.py     folders the server may read and write (Admin tab): settings, path checks, Landlock
config/odor_sheet_template.csv   odor sheet columns, with the odors from the thesis tables
app/static/        frontend: index.html, app.js, heatmap.js, colormaps.js (+ mpl_luts.js),
                   app.css, cluster.js (+ cluster-worker.js), figure.js (SVG/PDF/PNG export)
manage.py          list / show / export / delete datasets from the server
deploy/            systemd units; Dockerfile + docker-compose.yml at the top level
tests/             pytest tests
```

## Storage

`data/olfdb.sqlite3` tables: `datasets` (metadata, README, summary), `dataset_panels`,
`units` (date, fly_num, roi), `trials` (panel, odor, components, repeat), `responses`
(one row per cell, for SQL queries), `matrices` (the same values as a float32 blob for
fast serving), `users` (each user's folders on tensor), `odors` and `odor_syncs`,
`pubchem` (each odor CID's InChI), and `settings` (site-wide settings from the Admin tab:
the odor sheet's link). Files: `data/datasets/<id>/` holds the original data
file, `README.md`, `metadata.json`, `odor_yaml/*.yaml` and the `Experiment.xml`. Also
under `data/`: `admin.json` (the folders set in the Admin tab), `staging/`, `exports/`,
`odor_structures/`, `odor_previews/` (parsed sheets awaiting confirmation) and `tmp/`
(temporary files, such as uploads being received).

## HTTP API

Used by the frontend; handy for scripts too.

```
GET    /api/health                          {"ok": true, "time", "started"} (when the server started)
GET    /api/options                         dropdown values (indicators, drivers, panels, ...)
POST   /api/metadata/check                  {"metadata": {...}} → problems with upload form defaults
GET    /api/user?name=                      a user's folders on tensor ("user": null if none saved)
POST   /api/user                            {"name", "raw_dirs", "mocorr_dirs", "odor_dirs"} → each
                                            folder checked on tensor; saved if all are there
POST   /api/stage                           multipart 'file' → validation + matrix payload
GET    /api/stage/{id}                      a staged upload and its matrix payload
POST   /api/stage/{id}/lookup               {"name"} → that user's folders searched for the flies,
                                            recordings and odor YAMLs of the staged dataset
POST   /api/stage/{id}/odor_yaml            multipart 'file' (repeatable), or {"tensor_paths": [...]}
                                            → subset check
DELETE /api/stage/{id}/odor_yaml/{name}
POST   /api/stage/{id}/experiment_xml       multipart 'file', or {"tensor_path"}
DELETE /api/stage/{id}/experiment_xml
POST   /api/stage/{id}/readme               {"metadata": {...}} → README.md text
POST   /api/stage/{id}/commit               {"metadata", "readme", "readme_edited"} → {"dataset_id"}
GET    /api/datasets                        list
GET    /api/datasets/{id}                   metadata + README (markdown and HTML)
GET    /api/datasets/{id}/matrix            units, trials, base64 float32 values
GET    /api/datasets/{id}/files/{path}      original files
GET    /api/datasets/{id}/odors             its odors (per panel) matched to the odor table
GET    /api/download?ids=1,2&format=files|wide|long
                                            a zip, or a CSV for the matrix of a single dataset
GET    /api/odors                           the odor table (with each odor's `inchi`)
GET    /api/odors/status                    the odor sheet, when it was last read, whether it
                                            changed since the last sync
POST   /api/odors/preview                   the sheet (or multipart 'file', a CSV) validated and
                                            diffed against the odor table
POST   /api/odors/apply                     {"fingerprint", "applied_by", "changes", "deletions"}
GET    /api/odors/structure/{cid}           PubChem structure image (cached)
GET    /api/odors/template.csv              config/odor_sheet_template.csv
GET    /api/reference, /api/reference/{id}
GET    /api/admin                           the folders the server may read and write (saved, in
                                            effect since it started, checked), the odor sheet's
                                            link; Landlock, restarts
POST   /api/admin/check                     {"read_dirs", "write_dirs"} → each folder checked
POST   /api/admin                           {"read_dirs", "write_dirs"} → saved, in effect from
                                            the next restart
POST   /api/admin/odor_sheet                {"url"} (empty to clear it) → saved, in use at once;
                                            the sheet read to check it
GET    /api/admin/host                      the server's name and IP addresses
POST   /api/admin/restart                   the server stops, for Docker or systemd to start it
                                            again (409 if neither would)
```

Matrix payloads carry `values_b64`: little-endian float32, row-major
`(n_units, n_trials)`; `NaN` marks missing cells.
