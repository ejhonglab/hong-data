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
  reverse proxy can put a password on `/api/admin`, which is also where datasets are
  deleted. A deleted dataset's folder is moved to `data/deleted/<id>/` in the same
  transaction that deletes its rows, rather than erased, as no one has to log in to do it.
- **Uploads are staged.** A dropped file is parsed and validated into
  `data/staging/<id>/` and shown before anything is stored. Odor YAMLs, the
  `Experiment.xml`, tensor lookups and README generation all act on the staged upload,
  and only the final commit writes to the database and `data/datasets/<id>/`. Stages
  never saved are deleted after `OLFDB_STAGE_TTL_HOURS`. Validation collects every
  problem it can find, so a file can be fixed in one go.
- **Original files are kept verbatim** (data file, odor YAMLs, `Experiment.xml`), and
  every form value is stored on its own (database and `metadata.json`), apart from the
  README, which can be edited freely, at upload and later from the dataset's page (its
  file and database copy both overwritten, once the user confirms, unless it changed
  since the edit began). Anything parsed from them can be redone.
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
  logged in `odor_syncs` (tied to no one, whether from the sheet or a CSV import from the
  Admin tab). The rest of the app uses the database copy, so it keeps
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
- **Rows and columns moved or removed in the viewer are edits of the data's axes**, not
  of the screen: keys (units, odors) removed and the order they were dragged into,
  applied to the order chosen in the toolbar, so they hold when the matrix is transposed
  or shown by trial. Clustering always starts from one fixed order of what is left, as
  average linkage keeps the input order on ties and in which branch is drawn first.
- **Movies are read on the server, a few planes at a time.** The browser asks for planes
  (`/api/movie/planes`) and gets plain little-endian numbers, which it colours itself:
  it never parses a TIFF or reads a whole movie, and Z max projections are computed
  where the data are. A TIFF is memory-mapped when its data are in one piece (as
  tifffile writes ImageJ hyperstacks), and read page by page otherwise; a ThorImage
  `.raw` is memory-mapped by its `Experiment.xml` (time, then Z with flyback frames,
  then channel, as hong2p reads it). Open movies are kept (the last few), so a TIFF's
  header is not read again for every plane. ImageJ ROIs are read with roifile, and each
  is drawn on its Z plane: the hyperstack Z of the ROI, or its stack position on a
  volume (as hong2p reads them, ROIs being drawn on per-plane average images).
- **Figure export** embeds the cells as an image drawn without smoothing, so they stay
  crisp at any zoom, while text and lines stay vector in SVG and PDF.
- **Downloads** are built in `data/exports/` and deleted once sent. Across datasets,
  trials are matched on panel, odor components and repeat.
- **Merged downloads scale whole groups, by one factor each** (`app/merge.py`): fitted
  through zero by least squares on mean responses (over repeats; per glomerulus when
  every ROI is one, so the same glomeruli are compared, else over ROIs) to the stimuli a
  group shares with the target. One factor per group keeps each group's relative
  responses as they were, and a fit through zero keeps zero (no response) where it is.
  Concentrations of an odor stand for each other less than a tolerance apart (so that, by
  default, a panel's 10-fold steps stay apart), a dataset's exact one first, only for the
  fit: the matrices keep their own odors. The average target is the mean over groups of
  each response more than one of them has. By power, groups are made greedily in order
  of power, a dataset joining the first group it is close enough to every member of, so
  that no two in a group are further apart than allowed.
- **Reference matrices** are built in (`reference_data/`, read when asked for) or added in
  the Admin tab: a file anywhere on the server that the app only reads, into a copy in
  `data/references/<id>/` (its labels and identifiers as JSON, its numbers as float32),
  made again whenever the file's size or time changes; a failed read is kept too, so a
  broken file is not read on every listing. A file outside the folders the server may
  read is added to them as itself (Landlock takes a rule on a single file), not its
  folder, and read after the next restart, like any other change to them; deleting the
  matrix takes it off again unless another one needs it. Its chemicals are matched to
  the odor table when it is served, not when it is read, so that the names shown follow
  the Odors tab: by CID, else InChI, else CAS, MetaCyc or LOTUS ID, name or
  abbreviation, never against a CID or InChI given (a name alone can name another
  isomer).

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
app/reference.py   reference matrices: built-in and added ones, reading them, matching their chemicals
app/export.py      downloads: zips of dataset files, concatenated wide/long matrices
app/merge.py       merged downloads of a coding unit: grouping datasets, scaling groups to each other
app/odormeta.py    odor table: sheet parsing / validation, sync diff, links, dataset matching
app/tensor.py      users' folders on tensor: path checks, finding a dataset's flies, recordings, YAMLs
app/sandbox.py     folders the server may read and write (Admin tab): settings, path checks, Landlock
app/movies.py      a dataset's movies on tensor: finding them, reading their planes, their ImageJ ROIs
config/odor_sheet_template.csv   odor sheet columns, with the odors from the thesis tables
app/static/        frontend: index.html, app.js, heatmap.js, colormaps.js (+ mpl_luts.js),
                   app.css, cluster.js (+ cluster-worker.js), figure.js (SVG/PDF/PNG export),
                   player.js (movie player)
manage.py          list / show / export / delete datasets from the server
deploy/            systemd units; Dockerfile + docker-compose.yml at the top level
tests/             pytest tests
```

## Storage

`data/olfdb.sqlite3` tables: `datasets` (metadata, README, summary), `dataset_panels`,
`units` (date, fly_num, roi), `trials` (panel, odor, components, repeat), `responses`
(one row per cell, for SQL queries), `matrices` (the same values as a float32 blob for
fast serving), `users` (each user's folders on tensor), `odors` and `odor_syncs`,
`pubchem` (each odor CID's InChI), `settings` (site-wide settings from the Admin tab:
the odor sheet's link), and `reference_datasets` (reference matrices added there). Files: `data/datasets/<id>/` holds the original data
file, `README.md`, `metadata.json`, `odor_yaml/*.yaml` and the `Experiment.xml`. Also
under `data/`: `admin.json` (the folders set in the Admin tab), `deleted/<id>/` (the
folders of datasets deleted in the Admin tab), `references/<id>/` (reference matrices
added there, as read), `staging/`, `exports/`,
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
POST   /api/tensor/check                    {"path"} → a dataset's data folder checked on tensor
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
GET    /api/datasets/{id}                   metadata + README (markdown and HTML, and its path)
PUT    /api/datasets/{id}/readme            {"readme", "base"} → the README overwritten (409 if it is
                                            no longer "base", the one the edit started from)
GET    /api/datasets/{id}/matrix            units, trials, base64 float32 values
GET    /api/datasets/{id}/files/{path}      original files
GET    /api/datasets/{id}/odors             its odors (per panel) matched to the odor table
GET    /api/datasets/{id}/data_paths        where its raw and motion-corrected data are on the
                                            server, and what is in them (or why they are not)
PUT    /api/datasets/{id}/data_paths        {"raw_data_path", "mocorr_data_path"} (either) → set, if
                                            found on tensor
GET    /api/datasets/{id}/movies            its movies in its raw and motion-corrected data folders
GET    /api/movie?path=                     a movie's shape (t, z, c, y, x), pixel type, frame rate
GET    /api/movie/planes?path=&t=&n=&z=&c=  n planes from time t (z: a plane, or 'max'), as numbers;
                                            their type in X-Dtype, their shape (n, y, x) in X-Shape
GET    /api/movie/rois?path=                the ImageJ ROIs of a RoiSet.zip (or .roi), with their planes
GET    /api/download?ids=1,2&format=files|wide|long
                                            a zip, or a CSV for the matrix of a single dataset
GET    /api/merge                           the coding units recorded from, with their datasets
POST   /api/merge/preview                   {"coding_unit", "mode": "none"|"groups", "by": [...],
                                            "target", "tolerance", "power_tolerance"} → the groups,
                                            their factors and each dataset's
GET    /api/merge/download?coding_unit=&format=wide|long&mode=&by=a,b&target=&tolerance=&power_tolerance=
                                            every dataset of the coding unit, scaled, as a zip
GET    /api/merge/scales?(the same, without format)
                                            each dataset's group and scale factor (scaling.csv)
GET    /api/odors                           the odor table (with each odor's `inchi`)
GET    /api/odors/status                    the odor sheet, when it was last read, whether it
                                            changed since the last sync
POST   /api/odors/preview                   the sheet (or multipart 'file', a CSV) validated and
                                            diffed against the odor table
POST   /api/odors/apply                     {"fingerprint", "changes", "deletions"}
GET    /api/odors/structure/{cid}           PubChem structure image (cached)
GET    /api/odors/template.csv              config/odor_sheet_template.csv
GET    /api/reference                       every reference matrix: whether it can be read (why
                                            not), how many of its chemicals are identified
GET    /api/reference/{id}                  one, for the viewer (its chemicals as in the odor table)
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
DELETE /api/admin/datasets/{id}             the dataset deleted, its folder moved to
                                            data/deleted/<id>/ → {"files": where it went}
POST   /api/admin/references/check          {"path", "chemical_axis", "kind"} → whether it can be
                                            added, read now (a preview) or after a restart (steps)
POST   /api/admin/references                {"path", "title", "description", "source", "kind",
                                            "chemical_axis", "value_label", "row_label", "col_label"}
                                            → added (its file to the folders it may read, if need be)
DELETE /api/admin/references/{id}           its copy deleted (never its file) → {"revoked": the path
                                            taken off the folders it may read, if any}
```

Matrix payloads carry `values_b64`: little-endian float32, row-major
`(n_units, n_trials)`; `NaN` marks missing cells.
