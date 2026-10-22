# Design

How the app is put together, and why: the decisions that shape the code, how the Compare
tab's statistics are tested, where each part lives, how data is stored, and the HTTP API
between the frontend and the server. Using the app is described in
[README.md](README.md), hosting it in [DEPLOY.md](DEPLOY.md).

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
  several spellings). The `Experiment.xml` of the most recent functional recording is
  attached. Odor YAMLs are found panel by panel, following how olf (the olfactometer's
  program) makes them: from a panel's config (`<panel>.yaml`) it generates one YAML per
  recording (`<time>_stimuli/<time>_stimuli_<i>.yaml`), whose path goes into that
  recording's ThorImage notes, and recordings are named `<panel>_<n>` for its YAML
  `_stimuli_<n-1>`, or `<panel>` alone if it fits in one (with `_redo` or `_001` for one
  replacing another). So each panel's odors are described by the YAMLs named in the notes
  of one fly's recordings of it (the last fly's that describe them all; one whose notes
  name none taken to be of its number, from its others' run), which are what was run
  then; else the fewest of those of any fly, the panel's
  config (as it is now, which may have changed since), YAMLs named in the notes of
  recordings of no panel, and YAMLs with a fly's date in their path, in that order. The
  panels' YAMLs need not be in one folder, nor of one date.
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
- **The files the server reads and writes in its own folder are recorded as it runs**
  (`app/filelog.py`), for the Admin tab's tree of them, rather than listed from what is
  there: a listing would miss the files deleted since (uploads being received, expired
  staging folders, downloads sent) and could not tell its files from others put there.
  A Python audit hook sees each file the server's Python code opens, moves or opens as
  a database (returning at once on other events); a file only a C library opens
  (pyarrow reading parquet) is not seen, but each such file in its folders is one it
  wrote first. They are kept in memory and saved to the database every 30 s and as it
  stops. In Docker, Compose gives the container the folder it was run from
  (`OLFDB_HOST_DIR`), checked against the data folder's mount in `/proc/self/mountinfo`,
  so that paths in the container are shown as on the server.
- **Backups are made by the server itself**, from the Admin tab, into a folder it may
  write (`app/backup.py`), so that they need nothing set up on the server but that folder.
  Each is whole on its own (a folder laid out like `data/`, restored by copying it back),
  while files unchanged since the one before are hard links to it (as `rsync
  --link-dest` does), so keeping many costs the space of what changed. The database is
  copied with SQLite's backup API, from one moment. Whether the last is current is told by
  its `backup.json`: the size and time of each file, and a fingerprint of the database's
  rows (but `file_log`, which changes as the server merely reads; the tables written only
  with a dataset by their row counts, as its row in `datasets` tells of any change).
- **The Compare tab computes similarities in the browser**, as the viewer does the rest:
  the server only averages each dataset (`app/compare.py`: trials, then flies per
  glomerulus) and gives the ECFP fingerprints of the odor table's chemicals, which need
  RDKit (Morgan fingerprints with ECFP's atom invariants, radius = diameter / 2, 2048
  bits, from the InChI PubChem gives for their CID), kept per diameter while the odors
  stay the same. Every space (a dataset, a reference dataset, chemical structure) is
  keyed by the odor table's ids, so they line up whatever each calls its odors; a
  dataset's odor counts only alone (not in a mixture) and once, at the concentration
  chosen. Similarities of similarities are correlated over one side of the matrices
  (without the diagonal), pairs missing in either left out; their p-values, how they are
  tested and why, are under [Statistical testing](#statistical-testing). Two spaces side
  by side are two viewers of the same odors in the same order, linked (`linked` in
  `MatrixViewer`) rather than one matrix of both, so each keeps its own colors and units:
  an edit of either's rows is copied to the other (rows keyed by their place in that
  shared order), as are cell sizes, transposing, chemical labels and scrolling, and their
  headers and toolbars are made as high as each other's so the rows stay level.
- **Odor names are lowercased only where shown or downloaded, never where stored.**
  Lowercase odor names are a choice of each viewer (Settings, kept in the browser,
  sent to the server as `lower=1` with downloads), so datasets, the odor table and the
  original files keep their names as written, and odors are matched as before. Each
  name is lowercased unless another name shown or downloaded with it (in one dataset,
  the odor table, or every dataset of a download) differs from it only in case: those
  stay as written (`odors.lowercase_map`, and `lowercaseMap` in the browser), so that
  lowercasing never makes two odors one, nor puts the trials of two odors on one row of
  a download. The Admin tab can refuse new names with capitals (in uploads and in the
  odor sheet) while allowing any already in the database, so that the names in use
  can stay as they are while new ones are kept lowercase.
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
  YAML when it has one, otherwise by abbreviation, and a dataset is saved only once each
  is one odor there, so that every dataset's odors are known; the upload form can sync
  the new odors from the sheet itself (the apply above, with no changes or deletions), as
  only a person can add one there. An upload waiting on that is kept: in the browser
  (`localStorage`), and on the server for `OLFDB_STAGE_TTL_HOURS` after it was last used
  (each request to it touches its folder). PubChem structure images are fetched
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
- **Reference datasets** are built in (`reference_data/`, read when asked for) or added in
  the Admin tab: a file anywhere on the server that the app only reads, into a copy in
  `data/references/<id>/` (its labels and identifiers as JSON, its numbers as float32),
  made again whenever the file's size or time changes; a failed read is kept too, so a
  broken file is not read on every listing. A file outside the folders the server may
  read is added to them as itself (Landlock takes a rule on a single file), not its
  folder, and read after the next restart, like any other change to them; deleting the
  dataset takes it off again unless another one needs it. Its chemicals are matched to
  the odor table when it is served, not when it is read, so that the names shown follow
  the Odors tab: by CID, else InChI, else CAS, MetaCyc or LOTUS ID, name or
  abbreviation, never against a CID or InChI given (a name alone can name another
  isomer).

## Statistical testing

The Compare tab gives each correlation between two spaces' odor–odor similarities (the r
of "Between similarities") a p-value against no correlation, adjusted for testing every
two spaces shown at once. All of it is computed in the browser: the tests in
`app/static/pvalue-worker.js` (a worker, so the page stays responsive), the pairs two
spaces share and the adjustment in `similarity.js` (`sharedSimilarities`, `holm`).

### What each test does

Two spaces A and B (datasets, reference datasets, chemical structure) share n odors. For
each pair of them, i < j, x is A's similarity of i and j and y is B's; pairs either lacks
are left out. r is the Pearson or Spearman correlation of the xs and ys, over the
n(n − 1)/2 pairs (or fewer). Its p-value is two-sided, from 5000 resamplings or
permutations by default (set in the options). Each test's random numbers are seeded by
its two spaces (an FNV-1a hash of their keys, for mulberry32), so it gives the same p
whatever else is shown. A resample whose r cannot be computed (under 3 pairs, or a side
constant) is left out of the counts, and the CSV's `resamplings` says how many were used.

- **Permutation of the odors (a Mantel test; the default).** B's odor labels are
  shuffled (its rows and columns together, by a Fisher–Yates shuffle), so that
  x = A(i, j) is paired with y = B(π(i), π(j)), and r is recomputed. p = (the number of
  shuffles with |r_π| ≥ |r|, + 1) / (the number of shuffles + 1): the share of shuffles
  that look at least as related as the data do, at least 1 / (permutations + 1). The + 1
  counts the data's own arrangement, which keeps the test valid when the shuffles are
  random rather than all of them ([Phipson & Smyth 2010](https://doi.org/10.2202/1544-6115.1585)).
  This is [Mantel's test](https://en.wikipedia.org/wiki/Mantel_test), and the
  stimulus-label randomization test of
  [Nili et al. 2014](https://doi.org/10.1371/journal.pcbi.1003553).
- **Bootstrap of the pairs of odors.** The (x, y) pairs are resampled with replacement,
  as many as there are, and r recomputed (r*). p = 2 × (the fewer of r* ≤ 0 and r* ≥ 0,
  + 1) / (resamplings + 1), at most 1: the smallest α at which the (1 − α) percentile
  bootstrap interval of r leaves out 0, at least 2 / (resamplings + 1). It treats the
  pairs as independent of each other.
- **Bootstrap of the odors.** n odors are drawn with replacement and the pairs of those
  drawn taken (an odor drawn twice is not paired with itself, which would be the
  matrices' diagonal, but its pairs with others come twice), with r* and p as above. It
  resamples what was sampled, the odors (as Nili et al. 2014 resample stimuli), so pairs
  sharing an odor stay together.

### How each holds up

Each test was run on pairs of unrelated spaces, so every p < α is a false positive: a
valid test gives p < 0.05 for 5% of them and p < 0.2 for 20%. The tables below are the
output of `tests/simulate_pvalues.mjs` (`gjs -m tests/simulate_pvalues.mjs`, about ten
minutes; seeded, so it gives them again). Each space had n odors (15 or 30), each a
vector of 30 values (as across 30 glomeruli), standard normal values (Box–Muller, from
a seeded sfc32 generator); its similarities were the Pearson correlations of those; r
was the Spearman correlation of two spaces' similarities; each test had 1000
resamplings or permutations; each row is 500 pairs of spaces, so a share is within about
1% of the truth at 5%, 1.8% at 20% (one standard error). The odors of a space were:

- *alike*: every value independent;
- *saturated*: g·u + e, u a pattern of the space's (30 values), g each odor's gain,
  2 × standard normal, and e noise: most similarities near ±1;
- *broad responders*: as saturated, but g = exp(0.7 × standard normal), positive: some
  odors respond much more strongly than others to the same glomeruli, as broadly
  activating odors (or higher concentrations) do.

Unrelated spaces with p < 0.05, for 15 / 30 odors (5% is right):

| Odors            | bootstrap of odor pairs | bootstrap of odors | permutation (Mantel) |
|------------------|-------------------------|--------------------|----------------------|
| alike            | 5.4% / 5.4%             | 0.4% / 0.2%        | 5.4% / 5.8%          |
| saturated        | 4.2% / 5.8%             | 0.4% / 0.4%        | 3.8% / 5.6%          |
| broad responders | **36.2% / 50.8%**       | 5.6% / 6.8%        | 4.8% / 4.8%          |

and with p < 0.2 (20% is right):

| Odors            | bootstrap of odor pairs | bootstrap of odors | permutation (Mantel) |
|------------------|-------------------------|--------------------|----------------------|
| alike            | 19.2% / 22.2%           | 5.6% / 5.0%        | 20.4% / 21.6%        |
| saturated        | 11.6% / 10.6%           | 2.4% / 1.4%        | 22.0% / 23.6%        |
| broad responders | 57.0% / 67.8%           | 22.2% / 23.8%      | 19.4% / 20.6%        |

- **Permutation** is right in every case to within sampling error: its furthest miss,
  23.6% at 0.2 (saturated, 30 odors), is 2 standard errors out, as one of 12 rates may
  well be by chance.
- **The bootstrap of odor pairs** is right for alike odors, where two pairs sharing an
  odor are about uncorrelated (for independent random vectors, a's similarity to b and
  its similarity to c are). For broad responders it is badly wrong: p < 0.05 for a third
  to half of unrelated spaces. There, each odor's gain raises or lowers all its
  similarities together (a row effect), in each space. Two unrelated spaces' row effects
  agree or disagree by chance about as much as a correlation of n values does, but that
  agreement is spread over all n(n − 1)/2 pairs. Resampling the pairs as if they were
  independent sees only the much smaller spread of a correlation of n(n − 1)/2 values,
  so its p-values come out too small, more so with more odors (the pairs grow as n²,
  the odors carrying the effect only as n). For saturated odors it is about right at
  0.05 but not at 0.2 (11%): its p-values are not uniform when nothing is related.
- **The bootstrap of the odors** keeps the row effects, so it is about right for broad
  responders (5.6–6.8%). Otherwise it is far too conservative (0.2–0.4% at 0.05):
  its resamples spread wider than r really varies, as an odor drawn twice brings all its
  pairs twice and about a third of the odors are left out of each draw. It would rarely
  find a relation that is there.

### Why permutation is the default

- It answers the question asked, whether two spaces' similarities are related at all,
  with an exact test. If they are unrelated, shuffling B's odor labels changes nothing
  about how r is distributed, so the data's r is just one of the equally likely
  shuffled ones, however the pairs depend on each other. It needs no model of that
  dependence, which is exactly what both bootstraps get wrong.
- The bootstrap of odor pairs fails in the case most like real data: odors differ
  widely in how strongly they drive ORNs and PNs (some activate many glomeruli strongly,
  and concentration scales responses). It is still offered, as it was asked for first,
  with a warning in the p-values' heading.
- The bootstrap of the odors has little power to test against 0, far too conservative
  unless odors differ a lot in how strongly they respond. It suits other questions, such
  as a confidence interval for r or whether one space matches B better than another
  does, which is how Nili et al. 2014 use it.
- It costs no more than the bootstrap of the odors (every pair, each draw).

Its limits:

- With few odors there are few distinct shuffles (n!: 24 for 4 odors), so p cannot go
  much below 1 / n! however many permutations are run. No test can show much from a
  handful of odors.
- A small p says the spaces are related, not how strongly (r says that). A large p from
  few odors is not evidence that they are unrelated.
- No relation is taken to mean all shared odors are interchangeable. Shuffling ignores
  any structure among them, such as one chemical at several concentrations (each counts
  once here, at the concentration chosen).
- The simulations cover only Spearman correlations, Pearson similarities within
  spaces, normal data of 30 dimensions and three kinds of odors. They do not include
  missing values or the ties of binary reference data.

### Correcting for testing every two spaces

k spaces shown give m = k(k − 1)/2 tests: each two spaces once, as the matrix is
symmetric. Their p-values are adjusted by Holm's method
([Holm 1979](https://www.jstor.org/stable/4615733);
[Wikipedia](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)). Sorted
ascending, p(1) ≤ … ≤ p(m), the i-th becomes min(1, (m − i + 1) × p(i)), raised to at
least the one before it so the order is kept. This keeps the family-wise error rate, the
chance of any false positive among the m tests, at α whatever the dependence between
them. These tests do depend on each other: A–B and A–C share A, and all share odors.
Holm's method rejects everything Bonferroni's correction does, and possibly more.

The alternatives were not chosen:

- Hochberg's and Hommel's methods, and Benjamini–Hochberg's false discovery rate, have
  more power but need the tests to depend on each other in a particular positive way
  (PRDS), which is not known here.
- Benjamini–Yekutieli allows any dependence, but controls only the false discovery
  rate, a looser guarantee, at a similar cost in power for few tests.

With a handful of spaces (a few to tens of tests), each relation claimed should hold on
its own, which is what a family-wise error rate gives. The correction cannot rescue a
test whose own p-values are too small, as the pair bootstrap's can be: it adjusts for
many tests, not for an invalid one.

### A first simulation, discarded

The first run of these simulations drew its normal values from a hand-written linear
congruential generator, seed = (seed × 1103515245 + 12345) mod 2³¹, in JavaScript
numbers. The product exceeds 2⁵³, so its low bits were lost and it fell into a cycle of
10,466 values (after 4,871 draws from seed 11). Each run needed over half a million
values (2 per normal one), so every space was cut from the same 10,466, overlapping and
shifted, and spaces meant to be unrelated often shared odors' values. For alike odors,
from 300 pairs of spaces each, it gave these rates of p < 0.05:

| Test                    | 15 odors | 30 odors |
|-------------------------|----------|----------|
| bootstrap of odor pairs | 4.3%     | 8.0%     |
| bootstrap of odors      | 0%       | 0%       |

Permutation, run later with the same generator (after other draws, but into the same
cycle), gave exactly the same rates as the bootstrap of odor pairs. An exact test should
not reach 8.0% on unrelated spaces, and that gave the flaw away. These figures were
reported, and briefly committed (in `0a65d85`: "somewhat liberal", 4–8%), before the
generator was replaced: with `Math.random`, whose figures were the same within sampling
error, then in `tests/simulate_pvalues.mjs` with a seeded sfc32, which gave the tables
above.

## Code layout

```
app/main.py        HTTP API (Starlette) and static file serving
app/config.py      runtime configuration (environment variables)
app/ingest.py      CSV/parquet loading + validation → units, trials, float32 matrix
app/odors.py       odor string parsing/formatting (port of the relevant olf.py bits)
app/glomeruli.py   accepted glomerulus names (config/glomeruli.txt overrides)
app/yamlcheck.py   odor YAML parsing, what looks wrong in one, and the matrix ⊂ YAML check
app/thorimage.py   Experiment.xml summary
app/metadata.py    form model (pydantic) and README.md generation
app/db.py          SQLite schema and queries
app/reference.py   reference datasets: built-in and added ones, reading them, matching their chemicals
app/export.py      downloads: zips of dataset files, concatenated wide/long matrices
app/merge.py       merged downloads of a coding unit: grouping datasets, scaling groups to each other
app/compare.py     datasets averaged for the Compare tab (odors × glomeruli or ROIs); ECFP fingerprints (RDKit)
app/odormeta.py    odor table: sheet parsing / validation, sync diff, links, dataset matching
app/tensor.py      users' folders on tensor: path checks, finding a dataset's flies, recordings, YAMLs
app/sandbox.py     folders the server may read and write (Admin tab): settings, path checks, Landlock
app/filelog.py     the files the server has read and written in its own folders (Admin tab), paths on the server
app/backup.py      backups of the database and files into a folder the server may write (Admin tab)
app/movies.py      a dataset's movies on tensor: finding them, reading their planes, their ImageJ ROIs
config/odor_sheet_template.csv   odor sheet columns, with the odors from the thesis tables
app/static/        frontend: index.html, app.js, heatmap.js, colormaps.js (+ mpl_luts.js),
                   app.css, cluster.js (+ cluster-worker.js), figure.js (SVG/PDF/PNG export),
                   player.js (movie player), similarity.js (odor similarities, and between them;
                   pvalue-worker.js their p-values)
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
the odor sheet's link), `reference_datasets` (reference datasets added there) and
`file_log` (the files the server has read or written in its own folders). Files:
`data/datasets/<id>/` holds the original data file, `README.md`, `metadata.json`,
`odor_yaml/*.yaml` and the `Experiment.xml`. Also
under `data/`: `admin.json` (the folders set in the Admin tab), `deleted/<id>/` (the
folders of datasets deleted in the Admin tab), `references/<id>/` (reference datasets
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
POST   /api/stage                           multipart 'file' → validation + matrix payload (422 with
                                            ``rois`` when ROI names are not accepted: each, and the
                                            accepted glomerulus names)
GET    /api/stage/{id}                      a staged upload and its matrix payload, with its odor
                                            YAMLs' checks (to continue it)
GET    /api/stage/{id}/odors                its odors matched to the odor table, as for a dataset
POST   /api/stage/{id}/lookup               {"name", "raw_data_path"} → the flies searched for in the
                                            raw data path (else that user's folders), and the
                                            recordings and odor YAMLs of the staged dataset
POST   /api/stage/{id}/odor_yaml            multipart 'file' (repeatable), or {"tensor_paths": [...]}
                                            → what was added (or replaced), each file's own check
                                            (against the odor table), and the subset check
DELETE /api/stage/{id}/odor_yaml/{name}     → the same, without that file
POST   /api/stage/{id}/experiment_xml       multipart 'file', or {"tensor_path"}
DELETE /api/stage/{id}/experiment_xml
POST   /api/stage/{id}/readme               {"metadata": {...}} → README.md text
POST   /api/stage/{id}/commit               {"metadata", "readme", "readme_edited"} → {"dataset_id"}
                                            (422 unless each odor is one in the odor table)
GET    /api/datasets                        list
GET    /api/datasets/{id}                   metadata + README (markdown and HTML, and its path)
PUT    /api/datasets/{id}/readme            {"readme", "base"} → the README overwritten (409 if it is
                                            no longer "base", the one the edit started from)
GET    /api/datasets/{id}/matrix            units, trials, base64 float32 values
GET    /api/datasets/{id}/files/{path}      original files
GET    /api/compare/datasets/{id}           its odors × glomeruli (or ROIs), every trial (and fly) averaged,
                                            each odor's odor table id if it is one chemical alone
GET    /api/compare/ecfp?diameter=6         ECFP fingerprints of the odor table's chemicals (bits set)
GET    /api/datasets/{id}/odors             its odors (per panel) matched to the odor table; rows
                                            for the odor sheet for those not in it (and as cells
                                            to paste, in the order of the sheet's columns)
GET    /api/datasets/{id}/odors.csv         those odors as a file: odors.csv's columns, or a LaTeX
GET    /api/datasets/{id}/odors.tex         longtable (named as its other downloads, with _odors);
                                            ?lower=1 for odor names in lowercase, as downloads take
GET    /api/datasets/{id}/data_paths        where its raw and motion-corrected data are on the
                                            server, and what is in them (or why they are not), with
                                            the recordings in one and not the other, and all of
                                            both side by side, in the order they were recorded
PUT    /api/datasets/{id}/data_paths        {"raw_data_path", "mocorr_data_path"} (either) → set, if
                                            found on tensor
GET    /api/datasets/{id}/movies            its movies in its raw and motion-corrected data folders
GET    /api/movie?path=                     a movie's shape (t, z, c, y, x), pixel type, frame rate
GET    /api/movie/planes?path=&t=&n=&z=&c=  n planes from time t (z: a plane, or 'max'), as numbers;
                                            their type in X-Dtype, their shape (n, y, x) in X-Shape
GET    /api/movie/rois?path=                the ImageJ ROIs of a RoiSet.zip (or .roi), with their planes
GET    /api/download?ids=1,2&format=files|wide|long
                                            a zip, or a CSV for the matrix of a single dataset;
                                            &lower=1 for odor names in lowercase (but in files)
GET    /api/merge                           the coding units recorded from, with their datasets
POST   /api/merge/preview                   {"coding_unit", "mode": "none"|"groups", "by": [...],
                                            "target", "tolerance", "power_tolerance"} → the groups,
                                            their factors and each dataset's
GET    /api/merge/download?coding_unit=&format=wide|long&mode=&by=a,b&target=&tolerance=&power_tolerance=
                                            every dataset of the coding unit, scaled, as a zip
                                            (&lower=1 too)
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
GET    /api/reference                       every reference dataset: whether it can be read (why
                                            not), how many of its chemicals are identified
GET    /api/reference/{id}                  one, for the viewer (its chemicals as in the odor table)
GET    /api/admin                           the folders the server may read and write (saved, in
                                            effect since it started, checked), the odor sheet's
                                            link; Landlock, restarts
POST   /api/admin/check                     {"read_dirs", "write_dirs"} → each folder checked
GET    /api/admin/backups                   the backups in the folders it may write, the one being
                                            made, the last one and whether it is current
POST   /api/admin/backups                   {"folder"} → a backup started there (409 if one is being
                                            made, or it may not write there); as from GET
GET    /api/admin/files                     the files the server has read or written in its own
                                            folders (not the repository's), as paths on the server
POST   /api/admin                           {"read_dirs", "write_dirs"} → saved, in effect from
                                            the next restart
POST   /api/admin/odor_sheet                {"url"} (empty to clear it) → saved, in use at once;
                                            the sheet read to check it
POST   /api/admin/odor_case                 {"lowercase_only": bool} → whether new odor names may
                                            only be lowercase, at once
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
