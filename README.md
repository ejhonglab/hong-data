# Odor response database

A small intranet web app for a fly olfaction lab: upload matrices of mean neural
responses (per fly and ROI) to panels of odors, validate them, describe the
experiment, and browse everything as scrollable heatmaps. Fixed reference
matrices (chemicals × fruits, chemicals × pathways, ...) get their own tab.

- **Upload**: drop a CSV/parquet → it is checked and shown as a heatmap *before*
  anything is saved → fill in the experiment details (required: your name, indicator,
  Gal4 driver, coding unit) → optional odor YAML (every odor in the matrix must appear
  with the same abbreviation and concentration), ThorImage `Experiment.xml`,
  power/age/wavelength/rig/paths/notes → a README.md is generated and can be edited →
  saved to SQLite plus files on disk. The details are filled in from your settings,
  from the file (panel names; glomerulus ROIs mean coding unit "glomeruli") and from
  your data on tensor: the server looks for each fly (`<folder>/<date>/<fly>/`, as in
  hong2p) in your raw and motion-corrected data folders, attaches the Experiment.xml of
  the first recording and the fewest odor YAMLs
  that together describe every odor (those named in the recordings' ThorImage notes
  first, then ones with a fly's date in their path), and fills in the raw and
  motion-corrected data paths. Each field is colored by where its value came from
  (settings, tensor, the files), until you change it; the README says which files came
  from tensor, and from where.
- **Settings**: your name, and your raw data, motion-corrected data and odor YAML
  folders on tensor (relative to its top; absolute paths are refused). Each folder is
  checked on the server as you leave its field, and once all are found there ("All
  Paths Accessible") they are saved on the server under your name, so that any
  computer you use, and the upload form, gets them. Below that, defaults for the
  upload form, checked as the form checks its fields and kept in the browser only.
- **Browse**: datasets grouped by panel or by the date of their last fly; README and
  original files one click away; the same viewer as the upload preview.
- **Download** (one dataset from its page, or several ticked in the list): a zip with
  the original files, or the response matrices concatenated into one wide CSV
  (trials × ROIs, with `dataset_id` as an extra column level; trials matched on panel,
  odor components and repeat) or long CSV (one row per value), always with
  `datasets.csv` (every dataset's details), `odors.csv` (their odors, with metadata and
  links) and the READMEs. API: `/api/download?ids=1,2&format=files|wide|long` (see
  `app/export.py`).
- **Odors**: the app's own table of odor metadata, one row per chemical (name and
  aliases, abbreviation, PubChem CID, CAS, solvent, supplier + catalog number, MetaCyc
  and LOTUS IDs, inventory codes, notes), with links and PubChem structure images
  (cached on the server). It is synced one way from a Google sheet (or an imported CSV)
  with the columns of `config/odor_sheet_template.csv`, which starts it off with the
  odors from the thesis tables: the app shows what a sync would do, adds new odors,
  and applies changes to existing ones and deletions only where ticked, warning about
  renames that would unlink datasets. A badge on the tab shows when the sheet changed
  since its last sync. Each dataset has a thesis-style odor table, its odors matched
  through its odor YAML when it has one, otherwise by abbreviation (setup in DEPLOY.md).
- **Viewer**: canvas heatmap virtualised to ~10k ROIs, odor labels pinned at the top
  and fly/ROI labels at the left while scrolling, average over repeats or individual
  trials, rows by fly→ROI, ROI→fly or ungrouped, odor order as in the file or by name, a
  blue–white–red colormap built in Oklab (lightness linear in |value|, white at 0,
  full data range by default), hover tooltips.
  - *Clustering*: ROIs and/or odors ordered by hierarchical clustering (average
    linkage, correlation or Euclidean distance on trial-averaged responses, missing
    values skipped pairwise), with dendrograms. ROIs are always clustered all
    together, whichever fly they come from, and shown ungrouped in the clustered
    order (up to 15,000 ROIs; 15,000 need ~450 MB in the browser). Odors are clustered
    within each panel, by their responses across all ROIs. Distances are computed in
    parallel web workers, the linkage in one of them.
  - *Export figure*: SVG, PDF or PNG of the whole matrix as shown (order, colors,
    cell size, dendrograms) with a colorbar; cells are an embedded image, labels and
    lines stay vector in SVG/PDF.
- **Reference matrices**: any CSV in `reference_data/` listed in its `manifest.json`.

Hosting instructions: [DEPLOY.md](DEPLOY.md).

## Opening it in a browser

The site is plain HTTP, reachable only from the intranet: `http://<server>:8080/` (type
and bookmark it with the `http://`). Browsers label it "Not secure", which is expected:
there is no login and nothing secret is entered. Usually only Chrome's download
warnings need dealing with:

- **Chrome** (Edge, Brave and other Chromium-based browsers are similar)
  - *Downloads*: dataset downloads (always a zip), PDF/SVG figure exports and single
    original files other than CSV (parquet, YAML, XML, README.md) are stopped with
    "Insecure download blocked" (CSV and PNG files are not). Open the download bubble
    (or `chrome://downloads`) and click **Download insecure file**. To stop this for
    the site for good: click "Not secure" at the left of the address bar →
    *Site settings* → *Insecure content* → *Allow*.
  - *Pages*: since Chrome 154 (October 2026), *Always use secure connections* is on by
    default, set to *Warns you for insecure public sites*. That setting leaves private
    IP addresses, single-label hostnames (`http://olfdb/`) and non-default ports
    (`:8080`) alone. If you chose *Warns you for insecure public & private sites*
    (Settings → Privacy and security → Security), or the site is proxied on port 80
    under a public domain name, a visit can first show "The connection to … is not
    secure": click **Continue to site** (remembered for a while).
- **Firefox**: nothing to change. HTTPS-First (on by default) skips non-default ports
  and local addresses, and otherwise falls back to HTTP by itself; downloads from an
  HTTP page are not blocked. Only with *HTTPS-Only Mode* on (Settings → Privacy &
  Security) can you get "Secure Site Not Available": click **Continue to HTTP Site**, or
  add the site under *Manage Exceptions…* there.
- **Safari**: nothing to change. It tries HTTPS first and quietly falls back to HTTP.
  If *Warn before connecting to a website over HTTP* is on (Safari → Settings →
  Security; on iPhone/iPad, Settings → Apps → Safari → *Not Secure Connection
  Warning*), confirm on the warning page that you want to continue.
- **macOS 15 (Sequoia) or later, any browser except Safari**: macOS asks once whether
  each app may connect to devices on the local network. If that was refused, the site
  does not load in that browser (Chrome: `ERR_ADDRESS_UNREACHABLE`; Firefox: fails to
  connect or times out), even though Safari, which is exempt, loads it fine. Turn the
  browser on under System Settings → Privacy & Security → Local Network and restart it
  (if it is already on, turn it off and on again). This only applies when the server is
  on the same network segment as the Mac (not reached through a router or VPN).
  Browsers other than Safari on iPhone/iPad have the same switch under Settings →
  Privacy & Security → Local Network.

Serving the site over HTTPS, with a certificate browsers already trust (DEPLOY.md,
section 6), would remove all of the above except the macOS item.

## Data file format

The wide layout written by pandas for a frame with MultiIndex rows and columns:

```
date,,,,,2023-11-19 00:00:00,2023-11-19 00:00:00,...
fly_num,,,,,1,1,...
roi,,,,,D,DA2,...
panel,is_pair,odor1,odor2,repeat,
glomeruli_diagnostics,False,3mtp @ -5,solvent,0,-0.0377,0.13,...
```

- **Columns** (three header rows): `date`, `fly_num`, `roi`; every column unique.
  `roi` is a glomerulus name from `config/glomeruli.txt` or a non-negative integer.
- **Rows**: at least `odor1` (or `odor`); optionally `panel`, `is_pair`, `odor2`, ...,
  `repeat`. Several `odor<n>` levels are mixture components presented together and
  are displayed joined with ` + ` (solvent components dropped), as in `hong2p.olf`.
- **Odor strings**: `<abbrev> @ <log10_conc>` with single spaces, concentration 0 or
  negative; `solvent` for an unused component.
- A long layout (`date, fly_num, roi, [panel,] odor1, ..., [repeat,] value`) is also
  accepted for CSV and parquet (parquet with integer `fly_num` column labels cannot be
  written in the wide layout by pandas, so use long there).

Validation reports every problem it finds at once (bad odor strings with the exact
reason, unknown ROI names, duplicate columns, non-numeric cells, ...).

## Running it locally for development

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8080      # data goes to ./data
python -m pytest tests/
```

## Layout

```
app/main.py        HTTP API (Starlette) and static file serving
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
config/odor_sheet_template.csv   odor sheet columns, with the odors from the thesis tables
app/static/        frontend: index.html, app.js, heatmap.js, colormaps.js, app.css,
                   cluster.js (+ cluster-worker.js), figure.js (SVG/PDF/PNG export)
manage.py          list / show / export / delete datasets from the server
deploy/            systemd units; Dockerfile + docker-compose.yml at the top level
```

## Storage

`data/olfdb.sqlite3` tables: `datasets` (metadata, README, summary), `dataset_panels`,
`units` (date, fly_num, roi), `trials` (panel, odor, components, repeat), `responses`
(one row per cell, for SQL queries), `matrices` (the same values as a float32 blob
for fast serving) and `users` (each user's folders on tensor). Files: `data/datasets/<id>/` holds the original data file,
`README.md`, `metadata.json`, `odor_yaml/*.yaml` and the `Experiment.xml`.

## API (used by the frontend; handy for scripts)

```
GET  /api/options                       dropdown values (indicators, drivers, panels, ...)
POST /api/metadata/check                {"metadata": {...}} → problems with upload form defaults
GET  /api/user?name=                    a user's folders on tensor ("user": null if none saved)
POST /api/user                          {"name", "raw_dirs", "mocorr_dirs", "odor_dirs"} → each
                                        folder checked on tensor; saved if all are there
POST /api/stage                         multipart 'file' → validation + matrix payload
POST /api/stage/{id}/lookup             {"name"} → that user's folders searched for the flies,
                                        recordings and odor YAMLs of the staged dataset
POST /api/stage/{id}/odor_yaml          multipart 'file' (repeatable), or {"tensor_paths": [...]}
                                        → subset check
POST /api/stage/{id}/experiment_xml     multipart 'file', or {"tensor_path"}
POST /api/stage/{id}/readme             {"metadata": {...}} → README.md text
POST /api/stage/{id}/commit             {"metadata", "readme", "readme_edited"} → {"dataset_id"}
GET  /api/datasets                      list
GET  /api/datasets/{id}                 metadata + README (markdown and HTML)
GET  /api/datasets/{id}/matrix          units, trials, base64 float32 values
GET  /api/datasets/{id}/files/{path}    original files
GET  /api/reference, /api/reference/{id}
```

Matrix payloads carry `values_b64`: little-endian float32, row-major
`(n_units, n_trials)`; `NaN` marks missing cells.
