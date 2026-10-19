# Odor response database

A small intranet web app for a fly olfaction lab, to collect matrices of mean neural
responses (per fly and ROI) to panels of odors. Each upload is checked and described
with its experiment details, then everything can be browsed, clustered, exported and
downloaded as scrollable heatmaps, next to fixed reference matrices (chemicals × fruits,
chemicals × pathways, ...) and the lab's table of odor metadata.

## Using it

Open `http://<server>:8080/` in a browser on the lab network (type the `http://`, and
bookmark it). There is no login. The browser calls the site "Not secure": that is
expected, and nothing secret is entered. If a download or the page is blocked, see
[Browser warnings](#browser-warnings) below.

1. **Settings** (once): your name, and your raw data, motion-corrected data and odor
   YAML folders on tensor, written from the top of tensor (e.g. `tom/raw_data`). Once
   all of them are found, they are saved under your name for any computer you use.
2. **Upload**: drop your CSV or parquet file of responses (see
   [Data file format](#data-file-format)). It is checked and shown as a heatmap before
   anything is saved; if problems are listed, fix them all and drop the file again. Fill
   in the experiment details (your name, indicator, Gal4 driver and coding unit are
   required). Most are filled in for you from your settings, the file and your data on
   tensor, each colored by where it came from. Generate the README, edit it if you like,
   and save.
3. **Browse**: datasets by panel, or by the date of their last fly. Click one for its
   heatmap, README, original files, odors, and where its raw and motion-corrected data
   are on tensor.
4. **Download**: from a dataset's page, or tick several in the list: the original files
   (a zip), or the response matrices as CSV.
5. **Merge**: every dataset of a coding unit (e.g. glomeruli) in one download, optionally
   with groups of datasets (by experimenter, rig or power) scaled to each other first.

### Browser warnings

- **Chrome** (Edge, Brave and other Chromium-based browsers are similar)
  - *Downloads*: zips, PDF/SVG figure exports and single original files other than CSV
    (parquet, YAML, XML, README.md) are stopped with "Insecure download blocked" (CSV
    and PNG files are not). Open the download bubble (or `chrome://downloads`) and click
    **Download insecure file**. To stop this for the site for good: click "Not secure"
    at the left of the address bar → *Site settings* → *Insecure content* → *Allow*.
  - *Pages*: since Chrome 154 (October 2026), *Always use secure connections* is on by
    default, set to *Warns you for insecure public sites*, which leaves private IP
    addresses, single-label hostnames (`http://olfdb/`) and non-default ports (`:8080`)
    alone. If you chose *Warns you for insecure public & private sites* (Settings →
    Privacy and security → Security), or the site is proxied on port 80 under a public
    domain name, a visit can first show "The connection to … is not secure": click
    **Continue to site** (remembered for a while).
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

## Hosting

[DEPLOY.md](DEPLOY.md): running it on an intranet server (Docker or a plain venv, with
systemd), setting up the odor sheet and the tensor mount, which folders the server may
read and write (set in the Admin tab), configuration, backups and maintenance.

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
- Row levels that say nothing are dropped on upload: `is_pair` if it is False on every
  row, and `odor<n>` (from `odor2` on) together with every odor level after it, if all
  their values are the same one of: empty, `solvent`, `pfo`, `water`, `pfo @ 0`,
  `water @ 0` (with the 0 written any way: `pfo @ 0.0` too). (Only there are `pfo` and
  `water` accepted without a concentration.) Datasets stored before this are shown and
  downloaded the same way.
- A long layout (`date, fly_num, roi, [panel,] odor1, ..., [repeat,] value`) is also
  accepted for CSV and parquet (parquet with integer `fly_num` column labels cannot be
  written in the wide layout by pandas, so use long there).

Validation reports every problem it finds at once (bad odor strings with the exact
reason, unknown ROI names, duplicate columns, non-numeric cells, ...).

## The tabs in detail

- **Upload**: an optional odor YAML must have every odor in the matrix, with the same
  abbreviation and concentration. Other optional details: ThorImage `Experiment.xml`,
  power, age, wavelength, rig, data paths, notes. The raw and motion-corrected data paths
  are folders from the top of tensor, checked on the server as you leave their field
  (as in Settings), and must be found there to save (unless tensor itself is not on the
  server, when they are taken as entered). Details are filled in from your
  settings, from the file (panel names; glomerulus ROIs mean coding unit "glomeruli")
  and from tensor: each fly (`<folder>/<date>/<fly>/`, as in hong2p) is looked for in
  your raw and motion-corrected data folders, which give the data paths, the
  `Experiment.xml` of the first recording, and the fewest odor YAMLs that together
  describe every odor (those named in the recordings' ThorImage notes first, then ones
  with a fly's date in their path). A field stays colored by where its value came from
  (settings, tensor, the files) until you change it, and the README says which files
  came from tensor, and from where. Every value is stored on its own, whatever is
  changed in the README.
- **Settings**: folders are relative to the top of tensor (absolute paths are refused),
  each checked on the server as you leave its field; once all are found ("All Paths
  Accessible") they are saved on the server under your name. Below that, defaults for
  the upload form, checked as the form checks its fields and kept in this browser only.
- **Browse**: each dataset has a thesis-style odor table (with each odor's CID and InChI),
  its odors matched to the Odors tab through its odor YAML when it has one, otherwise by
  abbreviation. "Raw data" and "Motion corrected / traces" show where those are on
  tensor (to copy, as browsers cannot open folders on your computer) and what is in
  them. They are red, saying why, when no path was entered, or the server cannot find
  it (from the top of tensor, or from a folder set in the Admin tab) or may not read it.
  The path can be entered or corrected in their menus, checked as at upload (the README
  is left as it is). Its README can be edited there too (Edit, in the README box): saving asks
  whether to overwrite the README on the server (its path is shown) or discard the
  edits, and refuses if someone else changed it meanwhile. Unsaved edits are kept while
  you look at other datasets.
- **Movies** ("Play movies…" in the Raw data or Motion corrected menu of a dataset whose
  folder is found): a player for the movies of the dataset's flies in its data folders.
  Motion-corrected ones are TIFFs (as al_analysis writes them: `mocorr.tif`, else
  `flipped.tif` or `raw.tif`, though any TIFF is listed), drawn with the ImageJ ROIs of
  the `RoiSet.zip` next to them, or in a folder above (each on the Z plane it was drawn
  on; ROIs of other planes dashed). Raw ones are ThorImage `.raw` files, laid out by the
  `Experiment.xml` next to them (flyback frames left out). Play / pause, time and Z
  sliders, max projection over Z, channel, speed, contrast, zoom (wheel) and move
  (drag), with the value of the pixel under the pointer; keys as in ImageJ where they
  exist (space or `\` to play, `<` / `>` and the arrows to step). Only the planes shown
  are read, on the server, so even large movies open at once.
- **Download**: the matrices of several datasets are concatenated into one wide CSV
  (trials × ROIs, with `dataset_id` as an extra column level; trials matched on panel,
  odor components and repeat) or long CSV (one row per value). Zips always have
  `datasets.csv` (every dataset's details), `odors.csv` (their odors, with metadata and
  links) and the READMEs. The matrix of a single dataset is just one CSV instead, laid
  out as uploaded (less the row levels dropped on upload) and named
  `<panels>_<uploader>_<coding unit>_<last fly date>.csv`: the panels in the data,
  sorted, without diagnostic ones (unless there are only those), and the uploader's
  name lowercased without periods, e.g. `megamat_tom_glomeruli_2023-11-19.csv` (the
  long one ends in `_long.csv`).
- **Merge**: the matrices of every dataset of the chosen coding unit (matched in any
  case), concatenated as above, in a zip that also has `scaling.csv` (each dataset's group
  and scale factor). Datasets can be left at their own scale, or grouped by any of
  experimenter, recording system and power at the fly, each group then multiplied by one
  factor to match a chosen group, or the average of the groups. By power, only datasets
  with power measured are grouped (the rest are left as they are): every two in a group
  have power ranges (power ± its tolerance, ± 0.5 mW where none was entered) that overlap,
  or are at most a chosen power tolerance apart (0 mW by default). A group's factor best
  matches (least squares, through zero) its mean responses to the stimuli it shares with
  the target to the target's, per glomerulus when all ROIs are glomeruli, else averaged
  over ROIs. A dataset's own concentration of an odor is used when it has it, else its
  nearest less than the tolerance away (1 log10 unit by default: `x @ -3.9` then stands
  for `x @ -3`, unless the dataset also has `x @ -3`, while `x @ -4` does not). The tab
  shows each group's factor, the stimuli and responses it was fitted on, and Pearson's r
  over them, then the factor each dataset is multiplied by, which can be downloaded on
  its own (as `scaling.csv`, by `dataset_id` as in the matrices). Odors keep their own
  concentrations in the matrices.
- **Odors**: the app's own table of odor metadata, one row per chemical, with links and
  PubChem structure images. It is synced one way from a Google sheet (its link set in
  the Admin tab) or a CSV imported in the Admin tab, with the columns of
  `config/odor_sheet_template.csv` (downloadable there), which starts it off with
  the odors from the thesis tables: the tab shows what a sync would do, adds new odors,
  and applies changes to existing ones and deletions only where ticked, warning about
  renames that would unlink datasets. A badge on the tab shows when the sheet changed
  since its last sync.
- **Viewer** (datasets and the upload preview): odor labels stay at the
  top and fly/ROI labels at the left while scrolling, with hover tooltips. Average over
  repeats or show individual trials; rows by fly→ROI, ROI→fly or ungrouped; odors as in
  the file or by name. The default colormap is blue–white–red, with lightness linear in
  |value| and white at 0, over the full data range. *Transpose* shows odors as rows
  and ROIs as columns (in reference matrices too, rows as columns).
  - *Moving and removing rows and columns* (also in reference matrices): drag a label
    to move it within its group (fly, panel, ...), or a group's name to move the whole
    group; with individual trials, an odor's trials move with its label (the repeat
    numbers are not handles). Labels of clustered ROIs or odors stay put. Ctrl-click a
    label to remove what it labels at once (an ROI, an odor with its trials, or a whole
    fly or panel by its name); right-click it for a menu that asks first, and can also
    remove a glomerulus from every fly. Choosing another order in the toolbar replaces the dragged one. The orange
    *Reset rows & columns* brings back everything removed, in the toolbar's order.
    Edits are kept for each dataset while you look at others.
  - *Clustering*: ROIs and/or odors ordered by hierarchical clustering (average
    linkage, correlation or Euclidean distance on trial-averaged responses, missing
    values skipped pairwise), with dendrograms. ROIs are always clustered all
    together, whichever fly they come from, and shown ungrouped in the clustered
    order (up to 15,000 ROIs; 15,000 need ~450 MB in the browser). Odors are clustered
    within each panel, by their responses across all ROIs. Only what is not removed is
    clustered, and the result is the same whatever order it was shown in.
  - *Export figure*: SVG, PDF or PNG of the whole matrix as shown (order, colors,
    cell size, dendrograms) with a colorbar; cells are an embedded image, labels and
    lines stay vector in SVG/PDF.
- **Reference matrices**: tables added in the Admin tab (below), and any in
  `reference_data/` listed in its `manifest.json` (format in
  [reference_data/README.md](reference_data/README.md)). Their chemicals are labelled by
  their names in the Odors tab where they can be matched (or abbreviations, or as in the
  file: *Chemicals* in the viewer), the rest as in the file. A yellow warning is by a
  matrix with chemicals not identified by a CID or InChI (in the file, or through the
  Odors tab), naming them; one that cannot be read is listed by its name alone, with why.
- **Admin** (to the right of Docs; it first asks whether you are Tom, Jon or Betty, or
  have asked one of them): settings for the whole site.
  - *Folders on the server* the app may read (by default tensor) and also write (none
    by default: for backups, later), besides its own data folder. Saved changes take
    effect when the server restarts, which a button there does when it runs under
    Docker or systemd. How they are enforced:
    [DEPLOY.md](DEPLOY.md#folders-it-may-read-and-write).
  - *Odor sheet*: the Google sheet's link, used as soon as it is saved, and importing a
    CSV instead (with the template to start from).
  - *Reference matrices*: add a CSV, TSV or parquet file on the server (checked as you
    type its path, when the server may read it: its size, and how many of its chemicals
    match the Odors tab), with a title, description, source, labels, colors (or from its
    values) and whether its rows or columns are chemicals (or found from identifier
    columns, e.g. `cid`, `pubchem_cid`, `inchi`, `cas`, `name`, or rows labelled so).
    The app never changes the file: it reads a copy into `data/references/`, again
    whenever the file changes. A file outside the folders the server may read is added
    to them (itself, to read only), and read once the server restarts, with what that
    takes in Docker or systemd. Chemicals are matched to the Odors table by CID, else
    InChI, else CAS, MetaCyc or LOTUS ID, name or abbreviation, never against a CID or
    InChI given. Deleting one deletes the copy (never the file), and takes its file off
    the folders the server may read if it was added there for it. When one cannot be
    read, it says why, with a red dot on the Admin tab.
  - *This server*: its name, and its IP addresses (in Docker, those the network's DNS
    gives its name).
  - *Delete datasets*: every dataset, newest upload first (with a filter), each with a
    Delete button that asks first. A deleted dataset is gone from the site at once; its
    files (as uploaded, with `metadata.json` holding every detail entered) are moved to
    `data/deleted/<id>/` on the server, from where it could be uploaded again.

## Scripting

Downloads: `/api/download?ids=1,2&format=files|wide|long` (as from the Browse tab).
Datasets: `/api/datasets`, and each one's matrix at `/api/datasets/{id}/matrix`; the
odor table, with each odor's InChI: `/api/odors`. Every endpoint is listed in
[design.md](design.md#http-api). On the server, `manage.py` lists, shows, exports and
deletes datasets ([DEPLOY.md](DEPLOY.md), section 7).

## Development

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8080      # data goes to ./data
```

Settings and upload lookups need tensor mounted (at `/mnt/tank0/tensor`, or wherever
`OLFDB_TENSOR_ROOT` points), and syncing the odor table from a sheet needs its link (in
the Admin tab); the rest of the configuration is in
[DEPLOY.md](DEPLOY.md#configuration-reference).

[design.md](design.md) describes how the app is put together and why: the main design
decisions, the code layout, storage and the HTTP API.

Tests:

```bash
python -m pytest tests/
```
