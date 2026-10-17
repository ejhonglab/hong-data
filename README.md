# Odor response database

A small intranet web app for a fly olfaction lab: upload matrices of mean neural
responses (per fly and ROI) to panels of odors, validate them, describe the
experiment, and browse everything as scrollable heatmaps. Fixed reference
matrices (chemicals × fruits, chemicals × pathways, ...) get their own tab.

- **Upload**: drop a CSV/parquet → it is checked and shown as a heatmap *before*
  anything is saved → fill in the experiment details (your name is remembered on the
  machine; required: indicator, Gal4 driver, coding unit) → optional odor YAML (every
  odor in the matrix must appear with the same abbreviation and concentration),
  ThorImage `Experiment.xml`, power/age/wavelength/rig/paths/notes → a README.md is
  generated and can be edited → saved to SQLite plus files on disk.
- **Browse**: datasets grouped by panel or by the date of their last fly; README and
  original files one click away; the same viewer as the upload preview.
- **Download** (one dataset from its page, or several ticked in the list): a zip with
  the original files, or the response matrices concatenated into one wide CSV
  (trials × ROIs, with `dataset_id` as an extra column level; trials matched on panel,
  odor components and repeat) or long CSV (one row per value), always with
  `datasets.csv` (every dataset's details) and the READMEs. API:
  `/api/download?ids=1,2&format=files|wide|long` (see `app/export.py`).
- **Viewer**: canvas heatmap virtualised to ~10k ROIs, odor labels pinned at the top
  and fly/ROI labels at the left while scrolling, average over repeats or individual
  trials, rows by fly→ROI, ROI→fly or ungrouped, odor order as in the file or by name, a
  blue–white–red colormap built in Oklab (lightness linear in |value|, white at 0,
  full data range by default), hover tooltips.
  - *Clustering*: ROIs and/or odors ordered by hierarchical clustering (average
    linkage, correlation or Euclidean distance on trial-averaged responses, missing
    values skipped pairwise), with dendrograms. ROIs are clustered within each fly;
    with rows grouped by ROI the ROIs (averaged over flies) are clustered; ungrouped,
    all rows together (up to 15,000 per clustering; 15,000 rows need ~450 MB in the
    browser). Odors are clustered within each panel. Distances are computed in
    parallel web workers, the linkage in one of them.
  - *Export figure*: SVG, PDF or PNG of the whole matrix as shown (order, colors,
    cell size, dendrograms) with a colorbar; cells are an embedded image, labels and
    lines stay vector in SVG/PDF.
- **Reference matrices**: any CSV in `reference_data/` listed in its `manifest.json`.

Hosting instructions: [DEPLOY.md](DEPLOY.md).

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
python3 -m venv .venv && . .venv/bin/activate
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
app/static/        frontend: index.html, app.js, heatmap.js, colormaps.js, app.css,
                   cluster.js (+ cluster-worker.js), figure.js (SVG/PDF/PNG export)
manage.py          list / show / export / delete datasets from the server
deploy/            systemd units; Dockerfile + docker-compose.yml at the top level
```

## Storage

`data/olfdb.sqlite3` tables: `datasets` (metadata, README, summary), `dataset_panels`,
`units` (date, fly_num, roi), `trials` (panel, odor, components, repeat), `responses`
(one row per cell, for SQL queries) and `matrices` (the same values as a float32 blob
for fast serving). Files: `data/datasets/<id>/` holds the original data file,
`README.md`, `metadata.json`, `odor_yaml/*.yaml` and the `Experiment.xml`.

## API (used by the frontend; handy for scripts)

```
GET  /api/options                       dropdown values (indicators, drivers, panels, ...)
POST /api/stage                         multipart 'file' → validation + matrix payload
POST /api/stage/{id}/odor_yaml          multipart 'file' (repeatable) → subset check
POST /api/stage/{id}/experiment_xml     multipart 'file'
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
