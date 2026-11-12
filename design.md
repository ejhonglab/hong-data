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
  dependencies (the figure export writes its PDFs itself for that reason), but MathJax,
  for labels' math (below), vendored as its one self-contained file.
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
  `data/staging/<id>/` and shown before anything is stored. Several dropped together are
  one dataset (`ingest.combine`): each validated on its own, then their ROIs side by side
  over the union of their trials (NaN where a file lacks one; odor levels and `is_pair`
  a file dropped as saying nothing taken as solvent and False), each file kept; a ROI in
  two of them, or other column or row levels, are refused, as a fly of one person's in two
  datasets would be. Odor YAMLs, the
  `Experiment.xml`, tensor lookups and README generation all act on the staged upload,
  and only the final commit writes to the database and `data/datasets/<id>/`. Stages
  never saved are deleted after `OLFDB_STAGE_TTL_HOURS`. Validation collects every
  problem it can find, so a file can be fixed in one go.
- **Original files are kept verbatim** (data file, odor YAMLs, `Experiment.xml`), and
  every form value is stored on its own (database and `metadata.json`), apart from the
  README, which can be edited freely, at upload and later from the dataset's page (its
  file and database copy both overwritten, once the user confirms, unless it changed
  since the edit began). The form values can be edited later too (all but the files),
  checked as an upload's and saved likewise; what follows from them (title, panels, and
  a README not edited by hand) is made again, and `metadata.json` keeps each edit, with
  each field before and after. Anything parsed from them can be redone.
- **Each matrix is stored twice**: as a long `responses` table (one row per cell) for
  SQL queries, and as a float32 blob (`matrices`) for fast serving to the viewer.
- **Row levels that say nothing are dropped on upload** (`ingest.uninformative_levels`),
  and from the trials of datasets stored before that whenever they are read
  (`ingest.drop_uninformative`), so old and new datasets are shown and downloaded alike.
- **Odor strings follow `hong2p.olf`**, whose conventions, abbreviations and mixture
  formatting `app/odors.py` uses, so matrices written by hong2p-based analysis can be
  uploaded as they are. hong2p is installed without its `all` extra (none of its heavy
  dependencies). Uploaded odor strings are parsed more strictly than hong2p parses them,
  to tell users what is wrong with one.
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
- **A full restart is asked of systemd on the server through a file, not done by the
  site.** Rebuilding the images and making the containers again (`restart-olfdb`) needs
  root, or Docker's socket, which is as much; given to the container, either would undo
  every limit above. Instead the web server puts `data/restart-requested` in its data
  folder, and a systemd path unit on the server (`deploy/olfdb-restart.path`) runs
  `restart-olfdb` as root when it appears: the site can ask for that one thing, which
  takes nothing from the request, and no other. The script installs the units itself
  where they changed, so the site's own restart keeps them current too. It builds before
  it stops anything, so a build that fails leaves the site up, to say so from
  `data/restart.log`; the server takes back a request not taken within 5 s, so one
  never lies there to restart the site later. A restart stops the jobs under way (the
  worker's fits and predictions, and the web server's backups, imports and re-lookups),
  so it is refused, listing them with when they should end, until asked again to stop
  them; `restart-olfdb` asks too, getting them from the web server (`/api/admin/jobs`), or
  from the worker's container (`predictions.worker_jobs`) when the web server is down, as
  the worker may still be running a fit.
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
  its `backup.json`: the size and time of each file, and a fingerprint of each of the
  database's tables, of their rows (but `file_log`, which changes as the server merely
  reads; the tables written only with a dataset by their row counts, as its row in
  `datasets` tells of any change; and the fit worker's heartbeat, a setting it rewrites
  every 10 s, which from 2026-11-05 kept every backup from being current), so the Admin tab
  says which tables changed since. Of a backup from before they were kept there, they are
  read from its copy of the database, once.
  AL segmentation fits are backed up (a trained model takes a GPU for over an hour), but
  not their `work/`, nor predictions (minutes each, from a fit's model).
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
  tested and why, are under [Statistical testing](#statistical-testing). Spaces can be
  compared on glomeruli instead of odors (one or the other for the whole tab, never
  both): then each space is keyed by glomerulus names (a dataset of glomeruli's, by their
  responses to its odors; a reference dataset's whose labels on one axis are mostly
  glomerulus names, `reference.glomeruli`, by its values along the other), and all the
  rest (similarities, their correlations and p-values, those side by side) is the
  same, of glomeruli; no odor–odor similarity is shown then. Or on a panel's odors, for
  datasets with a panel in common (whose odors need not be in the odor table, nor be
  alone): then only datasets are spaces, each keyed by the panel's odors as presented
  (an odor of the odor table alone by its id and concentration, anything else by its
  string from the data, but for case; `_datasetPanelSpace`), so that mixtures and every
  concentration count, which keying by the odor table's ids cannot. That other axis, when it is
  not chemicals, is taken to be KCs (`kcAxisOf`, in the browser: no reference dataset
  yet is of glomeruli by anything else), which the viewer can sum over (only to show:
  similarities are always of the KCs). Two spaces side
  by side are two viewers of the same odors in the same order, linked (`linked` in
  `MatrixViewer`) rather than one matrix of both, so each keeps its own colors and units:
  an edit of either's rows is copied to the other (rows keyed by their place in that
  shared order), as are cell sizes, transposing, chemical labels and scrolling, and their
  headers and toolbars are made as high as each other's so the rows stay level.
- **Data without flies is stored with none, not made one up.** A file of ROIs alone (no
  `date` or `fly_num`) has units with both null (`units` remade once, in a database from
  before, to allow it: `db._units_nullable`), so it counts no flies, is grouped in the
  viewer as "fly not given", has nothing looked for on tensor, and downloads as it came
  (one header line). Such files were accepted for a while; flies and trials are required
  again (`ingest.FLIES_NEEDED`, `TRIALS_NEEDED`), but those stored then are still shown,
  so storage, the viewer and downloads keep handling them, and files of ROIs alone are
  still read far enough to say what they lack.
- **A permalink names a dataset by what it is, not its number** (`app/permalinks.py`,
  table `permalinks`). Datasets are replaced by deleting and uploading again, which gives
  them new numbers, so a permalink keeps what its dataset is (`dataset_keys`): its
  identity (`IDENTITY`: uploader, indicator, driver, coding unit, manipulations, response
  calculation and the panels that name it, `export.main_panels`, as in its downloads'
  names; keys kept before the response calculation was one have none), its flies
  (date, fly_num) and its trials (panel, odor, repeat). Panels are part of the identity
  because one person's datasets of a coding unit are told apart by them (Tom's megamat,
  validation2 and kiwi glomeruli datasets). While a permalink points to a dataset, an
  upload of its identity is refused, whatever its flies and trials (one dataset of
  each); once its dataset is deleted (the permalink then kept as it was, `orphaned_at`),
  an upload of that identity with some panels or flies in common (`compare`: related)
  takes it over: at once if all is the same, else once Proceed is pressed under what
  differs (`differences`), which saving checks again (`permalink_proceed`). Another
  calculation of the same recordings is another identity: it is not refused while the
  first's permalink points to it, nor takes that over once orphaned (an upload must press
  Proceed for any related orphan, which would hand one calculation's name to another). Its name
  follows its dataset's default (`default_slug`: its download's stem, ASCII, lowercase,
  '-' for the rest) until renamed by hand (`custom`); an edit changing the default asks
  before renaming it (409 `permalink_change` until `permalink_proceed` names the new one).
  `/d/<name>` leads to `#browse/<name>`, which Browse looks up (an orphaned one says what
  it was of); a name needs a letter, so `#browse/<number>` stays a dataset's number.
- **Duplicates are the same data, or the same flies of one person, coding unit, manipulations and
  response calculation.** A
  file is refused if it is any of a stored dataset's files byte for byte (`data_files`), or its data
  is (the same flies, ROIs, trials and values: `main._same_data`), whoever uploaded it,
  as soon as it is read. Its flies, (date, fly_num), are only compared with datasets of
  the same uploader and coding unit (names and units matched in any case; every way of
  writing KCs one), once those are entered (at the README step, and again on saving):
  all and only the same flies are refused, some of them warned of, naming the dataset
  that may need deleting, as a fuller upload may replace an earlier one. Another
  person's fly numbers of a day are their own, one fly may be recorded in two places
  (coding units), and a dataset of the same flies otherwise treated (the upload form's
  manipulations, `datasets.manipulations`: e.g. after a drug, matched in any case and
  order) is another; so is one of the same recordings calculated otherwise
  (`datasets.response_calc`, matched but for spacing: e.g. Yang's APL16 KCs as her
  whole-recording z, in units of each KC's baseline noise, and unnormalized), as many ways
  as there are, so long as each says how (`metadata.other_calculation`): one without a
  response calculation beside another of its flies is refused, the error saying which of
  the two needs one (`main._same_flies_error`), as edits are (`main._same_flies_problems`).
  The same data is refused whatever its manipulations or calculation. A
  manipulation is of a whole dataset, not of its flies: a fly is still its (date,
  fly_num) in it. A fly here is its ROIs' other column levels too
  (`ingest.fly_key`): each side of a fly's brain uploaded apart is not the other's
  duplicate.
- **The same recordings calculated otherwise are datasets of their own, told apart
  wherever datasets are named or pooled** (`Database.calculations`: the same flies,
  uploader, coding unit and manipulations, each with its own response calculation). The
  calculation is in their titles (`metadata.dataset_title`, as written, math too:
  `APL16: MB247dbLexA > GCaMP6s (KCs) [TNT], baseline-noise units`), the names of their
  downloads and permalinks (`export.dataset_stem`, in ASCII: `$\Delta F/F$` as `Delta-FF`)
  and a permalink's identity; Browse shows a dataset's calculation right of its title,
  as a menu of the others when there are (`calculations`, with `GET
  /api/datasets/{id}`), and a merged download takes one calculation of each
  (`merge.one_calculation`). Each set has a default (`Database.default_calculations`):
  the one last chosen with *Set Default Response Calc.*, in Browse or Compare (the setting
  `default_calcs`, the ids chosen, so that a set is matched anew each time, as datasets
  are edited or deleted), else the first uploaded. Sets calculated the same ways (their
  calculations alike, as `metadata.same_response_calc` matches them: `sameCalcSets` in
  app.js) are usually one experimenter's recordings each analysed alike, so choosing one
  set's default offers to choose the same calculation as the others', in one request
  (`Database.set_default_calculations`, all or none), listing the others by what each
  defaults to now. They are matched by name only when asked, not tied together: each
  set's default stays its own. The lists of Browse and Compare list
  only it (`listedDatasets` in app.js), the others named at the end of its item in
  Browse and a menu under it in Compare (Browse's controls, `calcControls`, and
  `saveDefaultCalc`, which tells both tabs), so that one recording is one item; listed by
  upload, it is as of the first uploaded, whichever is the default, as that is when the
  recordings came. Compare needs nothing more: each dataset is a space of its own there,
  labelled by its title. *Compare All Response Calcs.* shows them side by side in place of
  everything else (`o.calcs`, `_renderCalcs`), unit by unit, of glomeruli too (a ROI of a
  fly being the unit each calculation is of, not a glomerulus averaged over flies:
  `?units=1`), over the ROIs and odors all of them have, so that their rows and columns
  are linked (`MatrixViewer.linked`: each viewer's others, any number of them); it is its
  own view, not the side by side of spaces (`o.pair`, which are compared on odors of the
  odor table), as those would drop odors not in it and average glomeruli over flies. Being given every time
  it is entered, rather than only once another calculation is uploaded, a calculation
  names a dataset the same way whatever else is stored, so no title, file or permalink
  name changes when another calculation of its recordings comes or goes.
- **Panels are renamed when read, not stored so.** The Admin tab's renames (the setting
  `panel_renames`) are applied wherever a stored dataset's panels are read (`Database`:
  its trials, panels, title, metadata and summary), so every display and download has
  them, and removing one undoes it. Uploads are stored as their files say: their odor
  YAMLs and recordings on tensor are found by the panels in the file, and the upload
  form shows those; an upload is compared with stored data (for duplicates) with the
  renames applied to both. Two panels of one dataset may not become one name (their
  trials of an odor and repeat would then be the same).
- **Other column levels are part of a ROI's identity.** Besides date, fly_num and roi
  (and bad), a file's column levels (e.g. `hemibrain`) are kept with each unit
  (`units.extra_json`, as `extra` in the arrays served), and a unit is its fly, those
  and its name (`ingest.unit_key`): the left and right cells 0 of a fly are two units,
  as their recordings are two parts of the fly (for KC ids, numbered each on its own
  when their ids repeat across them; for Sum over KCs, summed apart). A fly's count of
  ROIs is still over all its parts. In Compare, a glomerulus column is the mean over the
  flies and sides that have it, as a reference dataset's glomerulus is one; ROI columns
  are grouped by fly and labelled as Browse's rows are (`flyKey`, `roiLabel`: 'L 0' under
  'YYYY-MM-DD fly 2'), in Browse's first order (`ROI_ORDER`): the server gives each ROI's
  fields, not its label, so that the two cannot drift apart. Downloads have a level for each (empty for datasets
  without it), so that a download reads back as it was uploaded.
- **Dates are read value by value.** As YYYYMMDD (text or a number) first, then any way
  pandas reads one: a number pandas would take for nanoseconds since 1970 is not a date,
  and one file may mix ways of writing them.
- **ROIs marked bad are kept, and left out where used.** KC datasets must have every KC
  of each fly (ids from 0 or 1 to the highest, `ingest.kc_id_problems`), so that a
  dataset says which KCs were segmented and not used rather than leaving gaps: those
  are marked in a `bad` column level, stored with each unit (`units.bad`, NULL where a
  file did not say). Viewers of KCs hide them by default (as rows or columns removed,
  but not among the user's own removals); Compare's computations and Merge's scale
  factors leave them out; downloads keep them, with the level. Pandas cannot read back
  a parquet file with a column level of True and False (it writes them as text), so
  `ingest._read_parquet` reads such a level as its text.
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
- **Light or dark from color tokens.** Every color of the page is a variable on `:root`
  in `app.css`, given again under `:root[data-theme="dark"]`. `data-theme` is set by a
  script in `index.html`'s head, before the page is drawn (so that it never flashes the
  other theme), from Settings (kept in the browser), else the system's or browser's
  `prefers-color-scheme`. `app.js` keeps it up to date as either changes, and says so
  (`olfdb-theme` on `window`) so that canvases, whose colors are read from the tokens
  when drawn (heatmaps, colorbars), are drawn again. Exported figures keep their own
  light colors (`figure.js`), and chemical structures (black on white) are inverted.
- **One set of experiment detail fields, the upload form's.** The Browse tab's Edit panel and
  the Settings tab's upload form defaults are made at page load of copies of the upload
  form's fields in `index.html` (`fieldsFrom` in `app.js`: each `[data-field]` goes to the
  forms its `data-in` names), and every form is read, filled in and checked by the same
  functions (`fieldValue`, `setFieldValue`, `wireFields`, `metadataProblems`), so a field
  added to or changed in the upload form is the same in all three: its dropdown and
  Other…, hints, and checks. The defaults leave out what is of one dataset only (its
  uploader, manipulations, data paths and files), and require nothing; the server checks
  all of them as an upload's (`Metadata`).
- **The AL segmentation download is tensor as it is, with a manifest.** Its files are where
  al_analysis leaves them (paths from the top of tensor), not renamed into a layout of the
  training code's: al-diag-seg reads this one (its `alseg/olfdb.py`), through
  `roi_training.json`, which says what each file is and names every recording's odors as
  in the odor table, so flies of different people line up by odor. Only what is not on
  tensor is made here: a fly's average image where al_analysis did not write its
  `mocorr_concat_avg.tif` (the mean of `mocorr_concat.tif`, ~3 s a fly, kept in
  `data/roi_training/` by the movie's path, size and time). The zip is stored, not
  compressed (TIFFs hardly compress), and written as it is sent, never to disk: it is a
  few GB. Files are read eight at a time ahead of being sent, as each file on tensor
  takes a while to start reading (an anatomical stack's ~1000 files: 45 s one after
  another, 3 s eight at a time).
- **The AL segmentation tab counts glomeruli as al-diag-seg trains on them** (`roi_training.roi_glomeruli`:
  the same rules as its `alseg/olfdb.py`, through `movies.roi_kind`), from each fly's
  `RoiSet.zip` (read eight at a time, as the download's files are), so what it shows is what
  a model is trained on. A fly's panels are those it has a response in, in the datasets it
  is exported through, not those its datasets have. Which glomeruli are infrequent, and
  everything said of them, is worked out in the browser from each fly's glomeruli, so a new
  cutoff needs no request.
- **Fits run in a worker container of their own** (`fit_worker.py`, docker-compose.yml's
  `worker`), not the web server: it alone is given the GPU, and has PyTorch, Cellpose and
  al-diag-seg (an image of about 9 GB, from the Dockerfile's `worker` target; the site's
  stays 1.2 GB). A fit takes hours and the CPU too (ANTs), while the web server is one
  process whose restart (Admin tab) would end one. The two share only the database and the
  data folder: a fit is a row of `fits` (asked for, and stopped, by the site; claimed and
  run, one at a time, by the worker) and a folder, `data/fits/<id>/` (~1.2 GB, nearly all
  the trained Cellpose-SAM's ~300 million weights, as float32;
  Cellpose's own copy, written into `work/` as it trains, is removed once the fit is over,
  and by the worker as it starts, as nothing reads it then); the worker says it is
  there, and with which GPU, every 10 s (a setting the tab reads). It is held to the
  folders the server may use as the web server is (Landlock), but may also write to the
  GPU's device files, and to its own threads' entries in /proc, which CUDA does as it starts
  (without, it says the GPU is not there). al-diag-seg reads the flies where they are on tensor, through a
  manifest like the download's with every path absolute, so nothing is copied.
- **al-diag-seg is installed in the worker's image apart from what it needs**: its
  dependencies (PyTorch with CUDA, Cellpose, ANTsPy: several GB) pinned in one layer,
  Cellpose's weights in the next, al-diag-seg itself (`--no-deps`, then `pip check`) last,
  so that a new al-diag-seg, which this young code needs often, rebuilds only its own layer
  rather than downloading PyTorch again while the site is down (a deploy stops the stack
  before it builds). Cellpose-SAM's training batches are as large as fit in the GPU's
  memory (al-diag-seg halves them from 8 until one does: 4 on the 24 GiB RTX 3090), and the
  worker's tests include one epoch of real training where there is a GPU, as the stand-in
  segmenter of its other tests skips what Cellpose writes and needs.
- **A fit's ROIs are Cellpose's own, named** (al-diag-seg's `Options.final`, since 97aba30):
  on fit 2's held-out flies, its odor-tuning and atlas refinements shrank Cellpose's ROIs and
  broke them into pieces (plane by plane, 46% of the manual ROIs found at IoU ≥ 0.5 before,
  29% after), as voxel-wise tuning is too noisy to judge borders by. Cellpose trains on every
  outline of one glomerulus, named or not (numbered and `?` ROIs were half of those drawn,
  and were taught as background); naming, the atlas and scores use the named ones.
- **How a fit's ROIs are made is one of its options, al-diag-seg's defaults by default**
  (`fits.ROI_DEFAULTS`, which a worker test checks against al-diag-seg's `Options`): Cellpose's
  cell probability threshold −2 and flow threshold 0.8, masks not grown, and the planes' masks
  joined into 3D objects one to one (al-diag-seg's `join_planes`) where they overlap by ≥ 0.2 of
  the smaller one and their mean odor tunings correlate ≥ 0.7. Cellpose's own stitching joins
  each mask to its best IoU in the plane before, so also neighbouring glomeruli that overlap
  between planes. These won together on 12 of fit 6's flies (its 4 held-out and 8 of its
  training flies), scored as partly drawn (below): picked on 11 and scored on the 12th, the same
  every time. Picked on the training flies, on the held-out ones they found 60% of the manual
  ROIs plane by plane and made 85% right, and 50% and 71% as 3D objects; the settings before
  them (−1 and 0.4, grown back by the 1 px training ROIs are eroded by, joined at 0.3 and 0.5:
  the best when scored strictly) 51%, 94%, 46% and 66%. So more glomeruli are found, but more
  ROIs are wrong where that can be checked, and more lie where nothing is drawn, where it can't.
  The fit form has them under "How its ROIs are made", and they are saved with the model.
  Models saved before they were options take al-diag-seg's defaults, so predictions with fits 2
  and 6 use them, though those fits' scores on the tab are as their fits made them; a
  prediction's provenance.json has the options its model ran with (`alseg_options`).
- **A fit trains Cellpose for 15 epochs by default** (`fits.DEFAULTS`; an epoch is each
  training plane and a flipped or turned copy of it once: 1584 for fit 6's 82 flies, ~5
  minutes on the 3090). In fits 2 and 6 the held-out flies' loss had stopped improving by
  epoch 10, where Cellpose's learning rate ends its warm-up (fit 6: 0.268 at epoch 5, 0.255 at 10, 0.258 at 20; fit 2: 0.229 at 10, 0.231 at
  20, 0.227 at 30, then up to 0.31 by 200, while its training loss kept falling). A fit trains
  on every fly with ROIs, so later ones take more steps per epoch, not fewer. That is the
  loss, not the scores: no fit has been run at 15 epochs to compare with 30.
- **Scores are shown plane by plane and as 3D objects** (al-diag-seg's `*_2d` columns,
  `metrics.match_report_2d`): ROIs are drawn and used per plane, and planes 10 µm apart make
  Cellpose's per-plane masks, stitched into objects, reach other planes than the manual ones,
  which the 3D match counts against them (46% found per plane, 30% as objects, for fit 2).
- **Scores count against ROIs only what the manual ROIs can say is wrong** (al-diag-seg's
  `partly_drawn` scoring): glomeruli are often drawn on their clearest planes only, so an ROI's
  parts on planes where its glomerulus has no outline count against it only inside another
  glomerulus' outline, and "right" is of the ROIs matched or lying mostly in drawn glomeruli
  (plane by plane too). "Found" is as before. A fit's and a prediction's result say how they
  were scored (`scoring`; earlier ones have none, and were scored strictly), and the tab marks
  the strict ones, whose numbers are lower than they would be now.
- **A fit's held-out flies are taken from each dataset in turn**, spread through its flies,
  so that each uploader's are scored; their names are al-diag-seg's (top folder on tensor,
  date, number). A fit records who asked for it (the first name set in Settings: the box
  sends people there, its name field selected) and from which IP address; anyone may
  delete any fit, as anyone may start one.
- **A fit adds the Grabe et al. 2016 atlas to its flies' atlas by default** (al-diag-seg's
  `refs`, `fits.REFERENCES`; fit 2, from before it was an option, had none), only with that
  atlas, which it is aligned to (by the glomeruli both have). It is downloaded into the worker's image as al_atlas exported it, at a pinned
  commit and checked against checksums (the Dockerfile), rather than kept in this
  repository, a second copy, or read from a home folder the worker cannot see. Its
  glomeruli are saved in the model's atlas, so the model runs without it. al_atlas'
  hemibrain export is not offered: some of its glomeruli are fragments of them.
- **A fit's time left is estimated stage by stage** (loading, training, atlas, scoring),
  from how long each took per unit of work in the last ten fits that finished (per fly
  loaded, per epoch and training fly, per fly or reference atlas in the atlas, per held-out fly), and for the
  stage it is in, from how far it has got; a fit waiting starts when those before it end.
  Before any fit has finished, the estimate is a guess, and says so.
- **Predictions of a fly's ROIs run in the fit worker, beside fits** (`predictions.py`,
  `fit_worker.run_prediction`): only its image has al-diag-seg and PyTorch. A thread of
  their own takes them one at a time, on the GPU when no fit holds it (a lock: a fit
  claimed waits for a prediction on it to end, minutes at most), else on the CPU, so a
  prediction never waits hours for a fit, nor takes GPU memory a fit's training needs (it
  would halve its batches, or start again). A child process would separate them better,
  but it inherits the worker's Landlock rules, which let CUDA write only the worker's own
  /proc entries, so it would not find the GPU; the two threads' printing goes each to its
  own log instead (`_Streams`: sys.stdout and sys.stderr dispatched by thread).
- **The worker gives the GPU's memory back after each fit** (and each prediction on it:
  `_free_gpu`, PyTorch's `empty_cache`), which PyTorch otherwise keeps for reuse: idle after
  fit 2 it held 19 GB of the 24 GB RTX 3090 (what training at batches of 4 peaked at), which
  other programs on the server could not use. CUDA's context (a few hundred MB) stays.
  Idle after fit 2 it also held 6.8 GB of RAM, 5.5 GB of it its heap (loading the 86 flies
  alone leaves nothing: 4.9 GB, back to 0.7 GB once let go), so it also asks glibc to give
  back what was freed (`_free_ram`, `malloc_trim`), saying in its own log how much it held
  before and after. If that does not bring it down, the memory is still referenced: a process
  per fit would give back all of it, but needs the Landlock rule for CUDA's /proc entries
  widened (see above), so it was not done yet.
- **The worker is given the GPU through CDI, and restarts if it loses it anyway.** With
  `--gpus` (the toolkit's hook adding device rules Docker does not know of), systemd dropped
  them from the running worker when it reloaded, under Docker's systemd cgroup driver: the
  GPU's device files could no longer be opened, Cellpose fell back to the CPU, and fit 5
  failed after an hour (its first step of training). CDI lists the device files in the
  container's spec, so they stay (a container update, the same mechanism, kept them). As
  well, before a fit or a prediction on the GPU, and as a fit's training starts, the worker
  opens the GPU's device files (`gpu_lost`); if it cannot, it puts the fit or prediction back
  in the queue and exits (code 1), for Docker to start it again: a container started anew has
  the GPU (checked with `docker update` on a throwaway container). Not after any CUDA error:
  one a fit causes itself would make it restart, and run the fit again, endlessly.
- **A prediction is one per fit and fly**, the fly known by its motion-corrected data
  folder (its real path, however it was entered), so asking again replaces it (the tab
  asks first). Its folder keeps what al-diag-seg's `alseg-predict` would write, the fly's
  average image (for the viewer: al_analysis' `mocorr_concat_avg.tif` is saved as frames,
  not planes), and `provenance.json`; a prediction outlives its fit's deletion (its
  provenance has the model's details), but cannot be made again then. The movie player
  shows a prediction's files as it shows tensor's: `_server_file` also allows the
  predictions folder (only it, of the data folder). Odors are matched to the model's as
  for a fit's flies, through the odor table, so a fly's odors named otherwise than the
  training flies' still count.
- **Model data is read where it is, never imported** (`model_data.py`): the Admin tab keeps
  only a folder's path on tensor, its lists of parameters and its patterns of spike count
  files that are not models (`model_sources`), and the
  Models tab scans it each time it is opened (its `params.csv`, `provenance.json` and spike
  count files, by name, size and time: a few tenths of a second for 2,000 files; parsed
  again only when they changed), so experiments archived into it since are listed without
  adding it again, and nothing of it is backed up or kept stale. A run is read from its
  parquet file when opened. Its id is its file's path on tensor, so links to it, and
  Compare's list of runs (in the browser), outlive the folder's being deleted and added
  again.
- **A model is a `params.csv` row; its experiment's `provenance.json` adds parameters** to
  each of its rows (flattened: a nested key by its path, `base->n_spikes_for_response`; a
  list as its JSON), the row's own column winning where both have one, even when empty
  (the archive leaves a cell empty where its experiment's value does not apply, e.g.
  `claw_thr` of a model with no per-claw threshold). A parameter is "the same for every
  model" when the models that have it agree: with blanks counted as values, every one of
  the KC-remapping archive's parameters would differ, as most are of one experiment.
- **A nested `provenance.json` key under a model's tag, variant or base is of that model
  alone** (`model_data.own_params`), named with `*` for the name (`runs->*->seconds`), and
  not shown for the others: E37's `runs`, keyed by tag, alone made 828 of the archive's
  1,679 columns, each set for one of its 15 models and blank for the other 907 (808 are
  left). Only keys equal to a name of a model of the experiment count, in a dict most of
  whose keys are such names (`model_data.keyed_by_model`: `variants` in E11, E14, E15, E17
  and E38; `bases` in E1, E2, E3, E16 and E43; `runs` in E37; E6's `meta`, as the archive's
  writers said), and not at the top level (E3's `bouton`, its base's settings): elsewhere
  such a dict is of the whole experiment (E54's `runs`, one record of the run; E9's
  `scales`, keyed kc, bouton and both, though bouton is the base of some of its models).
  That is shown once, for an experiment clicked, rather than as columns on every model.
- **A `provenance_<tag>.json` beside a `params.csv` is its model of that tag's own**
  (`model_data.FIT_PROVENANCE`; exact tags): the joint fits' (bounds, loss weights and
  targets, the fit they started from, commits), 116 in the archive, 105 of one model each.
  It is shown when one of that model's runs is opened and put in its files' download, not
  made columns; those of no tag (E51's `provenance_comparison.json`) are listed with their
  experiment.
- **The Models tab shows `params.csv`'s columns by default** (96 of the 808), and the
  `provenance.json` keys of those names (an experiment's value for all its models): the
  rest are what made the models (scripts, commits, hosts, notes, plans) and each model's
  settings, which a box shows. Its table draws only the rows and columns in view (and some
  either side, again once the view nears their edge), its columns' widths fixed from their
  longest values (`table-layout: fixed`): drawn whole, 908 models by 1,679 parameters were
  1.5 million cells, which took Chrome 40 s and crashed a browser; drawn so, ~3,000, in
  about a second (mostly the server reading the folders). The browser's find sees only
  those drawn, so the tab has its own, over every value shown.
- **Baseline models are chosen in the Admin tab and shown first** (`model_baselines`,
  `model_data.baseline_listing`), in a table of their own with the `params.csv` parameters
  they have, from a request that reads only their experiments' folders (~20 ms, against
  ~0.5–1.5 s for every model's), so they are shown while the rest load. A baseline is kept
  by its experiment's folder on tensor, its tag (or variant), base (E14's models of two
  bases share tags) and row (E35 has two rows of one tag), not by the folder's id or the
  row alone, so it survives the folder being deleted and added again, and rows added
  before it.
- **A baseline's files are imported when it is chosen, and it is shown as imported**
  (`model_data.import_baseline`, into `data/model_baselines/<id>/`, laid out as its
  experiment's folder: its runs, `params.csv`, `provenance.json` and its fit's
  `provenance_<tag>.json`), unlike other model data, which is only read on tensor: so
  backups have the models others are measured against, and a baseline stays what it was
  when chosen, also if its files on tensor are changed, rewritten or lost. Its runs'
  ids are of the copy (`olfdb:baselines/<id>/<file>`), so opening, comparing and
  downloading it read the copy; the table of every model still reads tensor. Each time
  it is listed or a run of it opened, its files on tensor are compared with the copy
  (`model_data.drift`): its own row of `params.csv` (not the file, whose other rows
  change as rows are added), the keys of `provenance.json`, and its other files by
  SHA-256 (hashed again only when their size or time changes, so this costs a stat per
  file), and runs or a fit's provenance added on tensor since. A difference is said, not
  taken: *Import again…* in the Admin tab takes tensor's files in place of the copy (the
  backups made before keep the old one). Files rather than BLOBs in the database, as for
  datasets: the code reading runs takes them as they are, by path.
- **A spike count file is matched to a model by name**, as the archive names them
  (`[<base>_sp<rate>_]<tag>_<condition>_spike_counts.parquet`, with no separator between
  the parts but `_`, which tags have too): the model whose tag (or variant) is in it as
  whole `_`-separated parts, the longest such; of models with one tag (E14's, of two bases),
  the one whose base begins the name. What follows is its condition. A file of a folder
  with a `provenance.json` of its own and no `params.csv` (E5's `hal_probe/`, models named
  otherwise) is of no model of its parent's, rather than matched to one by a shared tag.
  A link (inside tensor) is read as the file it leads to, under its own name.
- **Spike count files known not to be models are named by patterns set per folder**, in
  the Admin tab, not by rules about one archive's names in the code: the KC-remapping
  archive keeps intermediate fits (`…_base_acute_off`, the untuned base's APL-off run, which
  only weights APL by each KC's breadth) and copies (`partB/`, made by its trial-noise
  pipeline) beside its models. Those of no model that a pattern matches are only counted,
  so the Models tab's list of files of no model points at what is missing from a
  `params.csv` (as E49_low_conc_kc's was, 2026-11-09). The patterns never take a run from a
  model: `*_base_acute_off_spike_counts.parquet` also matches E13's `bouton_base` model's
  `acute_off` run (and 64 others), which stays its run.
- **A model run is compared as a dataset is, on odors only**: in the Compare tab its odors
  (matched to the odor table as a dataset's are: as `<abbreviation> @ <log10 conc>`) by its
  KCs (those the Models tab kept: spiking to any odor, by default), so its odor–odor
  similarities, their correlations with others' and side by side are those of datasets.
  An odor a run names with no concentration (`2h`, not `2h @ -3`) is still matched to the
  odor table, but to no dataset's odor (theirs have one).
  It is not compared on glomeruli (it has none) nor on a panel's odors (it has no panel).
  Its spike counts are shown on a log scale of 1 + each count (they are counts, most of
  them 0), here and in the Models tab, white to red: the red half of the responses'
  blue–white–red (`oklab-red`, its lightness linear too), so a model's counts read in the
  same hue as the responses they are compared with, counts having no negative side.
- **Matrices other than neural data are clustered only where they say what their rows and
  columns are** (`cluster` of what `setGenericData` is given: so far a model run's, KCs
  and odors), by `clusterMatrix`, as `clusterNeural` clusters a dataset's: its rows all
  together (then ungrouped), its columns within each group, over what is not removed. Not
  reference data, whose axes are often not alike enough to be (a chemical's pathways), nor
  Compare's linked matrices, whose rows must stay level with each other's. Whether to is
  kept by kind of viewer (`clustered` in the browser), apart from datasets', so that
  clustering a model's 1,700 KCs does not start a 25,000-ROI clustering in Browse.
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
  (each request to it touches its folder). Odor structures are drawn on the server by
  RDKit, as SVG (`odormeta.structure_svg`, kept in memory), from the InChI of each CID:
  fetched from PubChem once (a hundred CIDs per request, the first time a structure is
  wanted without one) and kept in the `pubchem` table. PubChem is asked again about a CID
  it had no InChI for only after an hour. Vapor pressures and molar masses come from
  PubChem too, through chemutils (`chemutils.inchi2vapor_pressure`, `cid2molar_mass`),
  looked up in a thread after each sync and when the server starts (once the InChIs
  missing are fetched), one odor with an InChI at a time, each saved in the `pubchem`
  table as it comes (none found is saved as looked up, and not looked up again; one that
  could not be is tried again after an hour), and then those added meanwhile. What else
  the Headspace calculator needs is looked up and kept the same way, when it is first
  needed (see [Headspace calculator](#headspace-calculator)).
  chemutils runs in a process of its own (`chemprops.py`), as importing it changes the
  logging of the process that imports it (a handler printing stack traces with
  warnings, and a log file in the home folder, or in older versions on the root logger,
  beside its code), with its cache and its log of warnings (`CHEMUTILS_LOG`, read on
  import) in the data folder, its warnings not printed, and a ChemSpider API key set, so
  that it reads and writes nothing in the home folder, which Landlock denies the server. chemutils keeps what it looked up in that cache; it
  does not keep failures, so the server remembers those itself (in memory) to not ask
  again at once, and tries again, after a few seconds, a lookup a server was too busy
  for (PubChem answers 429 to more than 5 requests a second, as the first lookup of
  every odor's molar mass made).
  As both the database and chemutils' cache keep what was looked up, a newer chemutils
  (pinned to a commit in `requirements.txt`) changes nothing by itself. So each value
  keeps the commit that looked it up (`pubchem.<property>_chemutils`), and
  `chemprops.CHANGES` lists the commits that changed what a lookup gives (95a66c6:
  vapor pressures only from PubChem's values near 25 °C; 9196a0c: densities relative to
  water read; fda8e3b: PubChem's estimated vapor pressures used only without measured
  ones, and each said to be measured or not, and Henry's law constants of rows given as
  inequalities left out), which the Admin tab compares them with, to look those up again with
  chemutils' `ignore_cache` (which also replaces what its cache keeps). Values from
  before commits were kept count as older than every change. A chemutils commit that
  changes a lookup needs adding there, and each one pinned to `chemprops.COMMITS`
  (DEPLOY.md, "Updating chemutils").
- **The heatmap is a virtualised canvas**: only the visible cells are drawn, so 10k+
  rows scroll smoothly. The default colormap is built in Oklab so that lightness is
  linear in |value|, with white at 0 and blue and red poles of equal lightness: "twice
  as dark" means "twice as large" on either side, which the ColorBrewer and Moreland
  maps only approximate. Distances (a reference dataset's `pairwise: "distance"`, or its
  values in Compare) start in plasma reversed, not viridis like other values of one sign:
  the nearest, those that matter, brightest, in a map that can't be taken for viridis
  (and whose ends are neither black nor white, against either theme's background); the
  sequential colormap chosen is remembered apart for them (`_genericColormap`).
- **Log color scale** (`makeNorm('log')`): for sequential colors, of the positive values
  (0 and below the lowest color), chosen automatically when none are negative and the
  positive ones span 3 or more orders of magnitude (`LOG_DECADES`), as VCF's
  quantities do (1e-6 to 2e6 mg/kg); its colorbar is labelled at its ends and every power
  of ten between them, or every 2nd, 3rd... evenly, where their labels would overlap
  (`spacedTicks`, as for every colorbar, on screen and in figures, each tick marked). With
  only its ends and one or two between, a log scale read as linear: LOTUS's 2, nearly
  halfway from 0 to the 10 labelled, looked too bright for a 2. A reference dataset can ask for it whatever its values
  span (its meta's `scale`: `lotus.scale`, of counts of organisms or papers, where most
  cells are a few and some hundreds). Counts (none negative nor fractional, some 0) are
  instead on a log scale of 1 + each (`makeNorm('log1p')`), as 0 would otherwise share
  the lowest color with 1: a chemical not reported in a family, with one reported in a
  single organism of it. The scale chosen is kept for the matrix it was chosen for (by
  its file stem), not for all: log chosen for LOTUS made MetaCyc's 0, 0.5 and 1 its 0.5
  to 1, 0 the color of 0.5.
- **The version in the footer** (`app/version.py`): the commit the code is of, whether
  the code built (`app`, `config`, `manage.py`, `requirements.txt`, `Dockerfile`) differs
  from it, and when it was built. The image is built from the working tree, which is
  often ahead of its last commit, so "with changes not committed" (and the build time)
  says when the commit alone does not name the code. Only the Dockerfile's first stage
  sees `.git` (with git), writing `VERSION.json` for the image; run from a checkout, git
  is asked as the server starts. A page shows the version it loaded, and on being looked
  at again (at most every 30 s) asks `/api/health` again: a tab left open across a
  deploy runs the old scripts, which once passed for a fix that did not work.
- **Colors as the colormap has them**, to be read off the colorbar: its bar is drawn in
  whole pixels of the screen (`drawColorbar`), as columns a CSS pixel wide left seams at
  125% or 150% display scaling, through which the page darkened every color (a quarter,
  on the dark theme); and the hovered cell is left out of the crosshair shading its row
  and column, which lightened it twice.
- **Sorting a matrix** (`sortedByValues`): of reference data, each row (or column) by the
  mean or largest of its values over the columns (or rows) shown, within its group; an
  axis shared with a linked viewer (side by side in Compare) keeps that viewer's order.
- **Clustering runs in web workers.** Distances use only the entries both vectors have
  (like pandas' `DataFrame.corr`), so missing values are fine. They are computed in
  blocks of rows spread over the workers, and average linkage (UPGMA, merges in scipy's
  convention) then runs in one worker directly on those blocks, never copied into one
  buffer; blocks are transferred between workers, as they cannot share memory. Memory
  limits it to 30,000 rows (a KC dataset's): their distances, as float32, take 1.8 GB,
  all in the linkage's worker; about 45 s of distances (spread over the workers) and
  25 s of linkage, measured in SpiderMonkey. So a stored dataset's clusterings are kept
  on the server once a browser has made them (`clusterings`, deleted with the dataset,
  the 24 most recently used of each, left out of backups' fingerprint as a cache): each of
  ROIs, or of a panel's odors, under what it is of (`CLUSTER_VERSION`, metric, the ROIs and
  odors left out, axis, panel and number of items), looked for before clustering (so
  clustering odors too does not cluster the ROIs again). The server checks that one is a
  clustering of that many items (a permutation, merges of clusters there are), not that
  it is right: anyone on the network can keep one, as anyone can upload.
- **Rows and columns moved or removed in the viewer are edits of the data's axes**, not
  of the screen: keys (units, odors) removed and the order they were dragged into,
  applied to the order chosen in the toolbar, so they hold when the matrix is transposed
  or shown by trial. Clustering always starts from one fixed order of what is left, as
  average linkage keeps the input order on ties and in which branch is drawn first.
- **Movies are read on the server, a few planes at a time.** The browser asks for planes
  (`/api/movie/planes`) and gets plain little-endian numbers, which it colours itself:
  it never parses a TIFF or reads a whole movie, and Z max projections are computed
  where the data are. A TIFF is memory-mapped when its data are in one piece (as
  tifffile writes ImageJ hyperstacks), in the file's byte order (hong2p writes them
  big-endian; tifffile gives the type in the machine's), and read page by page
  otherwise; a ThorImage `.raw` is memory-mapped by its `Experiment.xml` (time, then Z
  with flyback frames, then channel, as hong2p reads it). Open movies are kept (the last
  few), so a TIFF's header is not read again for every plane. ImageJ ROIs are read with
  roifile, and each is drawn on its Z plane: the hyperstack Z of the ROI, or its stack
  position on a volume (as hong2p reads them, ROIs being drawn on per-plane average
  images). Contrast is set in the browser from the plane shown, as ImageJ's B&C window
  sets it (its Auto, Reset and slider arithmetic, over a 256-bin histogram).
- **Motion-corrected movies are found as al_analysis lays out a fly's folder**, not by
  listing every TIFF in it, which also has its dF/F volumes, averages and the suite2p
  run (twice, through its link): the fly's recordings registered together
  (`mocorr_concat.tif`, linked to the suite2p run's folder, whose `<recording>.tif` are
  the recordings in it), and one movie per recording folder not in it (`mocorr.tif`,
  else `flipped.tif` or `raw.tif`). A movie of several recordings is in each of them
  when the raw and motion-corrected data are compared. Only a fly's folder with neither
  (or a folder with no fly's folder in it) has every TIFF listed. Nothing caps how many
  are found: comparisons are of every recording. A movie's trials (odors by frame) are
  al_analysis' `trial_frames_and_odors.json` next to it; a raw movie has its recording's
  from the motion-corrected data, the frames being the same.
- **The movie player's ROIs are told apart on the server** (`movies.roi_kind`), by the
  rules an upload names glomeruli by (`config/glomeruli.txt`, several joined by `/`, a
  `_t<n>` suffix removed: Sam's mark of a name not certain, kept apart as tentative),
  and by the dataset's own ROIs' names, so the player's *glomeruli only* hides just what
  an upload would not take as a glomerulus. Zooming out stops where the movie fits the
  window, and dragging where its edges meet the window's: nothing but the movie is
  ever shown around it, but across (or down) the fitted movie, where it is narrower.
- **A recording redone supersedes the one it is named for**: a fly's `<recording>_redo`
  folder (in its raw or motion-corrected data) means `<recording>` was done again, and
  `<recording>`'s movies are left out of both (`movies.drop_redone`), rather than shown as
  a recording never motion corrected (hong2p's `paired_thor_dirs` skips it the same way,
  so al_analysis does not register it). Its folder may still be in the motion-corrected
  data (6 of the 20 on tensor, in 2026-11, as `raw.tif` and analysis outputs, none in a
  dataset). Checked then against the raw data, by their pixels: each of those holds that
  recording's own data, every `_redo`'s motion-corrected movies its redo's, and no
  `mocorr_concat.tif` has a recording redone in it, so names were never swapped or the
  suffix dropped.
- **Figure export** embeds the cells as an image drawn without smoothing, so they stay
  crisp at any zoom, while text and lines stay vector in SVG and PDF.
- **Labels with LaTeX math** (a dataset's response calculation, its colorbar's label:
  `static/latex.js`) are written as matplotlib's are (`$\Delta F/F$`), as the lab's
  analysis code writes them. Their math is typeset by MathJax 3.2.2
  (`static/vendor/mathjax/tex-svg-full.js`, Apache 2.0: every TeX extension and the SVG
  font in one file, so nothing more is fetched; 2 MB, loaded only once a label has
  math), as SVG, which is read into paths, rectangles, clips (MathJax's nested `<svg>`,
  which crop stretched glyphs) and the odd character its fonts lack: the page's SVG of a
  label is made again from those (a label is entered by anyone, and MathJax's own output
  may carry links, classes and styles), and so is a figure's, in SVG, PDF (quadratic
  curves as cubic ones, the clips by `W n`) or PNG (`Path2D`). Its `require`, `autoload`,
  `html` and `action` extensions are left out, and errors are thrown rather than drawn,
  to be shown under the field, which cannot be saved with one. The server keeps the label
  as written: only the browser can typeset it.
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
- **A merged download takes one calculation of the same recordings** (`merge.one_calculation`):
  of datasets that are the same recordings calculated otherwise (`Database.calculations`),
  the one of the calculation chosen (`plan`'s `calc`, the tab's menu of those of the
  coding unit), else their default (`Database.default_calculations`, as Browse lists
  them: the first uploaded unless another was chosen there, so that by default a merge
  is what it was before another calculation was uploaded). The others are left out, and said to be (in the tab
  and the download's README), as no recording should be in a merge twice, nor a group's
  factor be fitted to the same responses in two units. Groups can be made by response
  calculation too (`GROUP_BY`'s `calc`, matched as `same_response_calc`), so that
  datasets of other recordings in other units are not scaled by one factor either.
- **Reference datasets** are built in (`reference_data/`, read when asked for; those
  marked as examples only while there is no other, built in or added) or added in
  the Admin tab: a file anywhere on the server that the app only reads, into a copy in
  `data/references/<id>/` (its labels and identifiers as JSON, its numbers as float32,
  or of a matrix most of whose cells are 0, its other cells: `sparse.Sparse`, as
  `values.npz`), made again whenever the file's size or time changes; a failed read is kept too, so a
  broken file is not read on every listing. A file outside the folders the server may
  read is added to them as itself (Landlock takes a rule on a single file), not its
  folder, and read after the next restart, like any other change to them; deleting the
  dataset takes it off again unless another one needs it. Its chemicals are matched to
  the odor table when it is served, not when it is read, so that the names shown follow
  the Odors tab: by CID, else InChI, else InChI but for stereochemistry
  (`odormeta.stereo_compatible`: the same without stereo layers, and of the stereocenters
  and double bonds both specify, none different, compared by atom number, with `/m` and
  relative or racemic `/s` taken into account; of two such odors, neither), else CAS,
  MetaCyc or LOTUS ID, name or abbreviation, never otherwise against a CID or InChI given
  (a name alone can name another isomer). The stereo step is for chemicals named by a
  source that may not tell stereoisomers apart, or an Odors table entry that does not
  (VCF's dl-menthone and isomenthone, of a bottle sold as menthone of mixed isomers); it
  is warned of (`reference.stereo_warning`, with how much of the data it brings in,
  from each item's count of cells not 0 or missing kept in its import, ``nonzero``,
  record version 3), and the Compare tab makes the items of one odor one by the
  dataset's ``merge`` field (mean by default; VCF's amounts summed). Summing counts twice
  where a source gives a total of every stereoisomer beside its parts (VCF's linalool
  beside (R)- and (S)-linalool in a few papers). An axis whose labels are odors as datasets write them (`reference.odor_axis`:
  `<abbreviation> @ <log10 conc>`, at least 3 and over half of its labels, e.g. ORN
  responses by glomeruli) is its chemicals unless the file gives identifiers or the Admin
  says otherwise (before the guess from labels, which would be no chemicals at all for
  such labels), matched as a dataset's odors are (`OdorIndex.match`: the abbreviation,
  then the name), as they are the same odors, written by the same people; each keeps its
  concentration to label it by. They are then checked as an upload's are
  (`main._odor_matches`, each label a trial of no panel: `odor_check` beside the
  dataset, and beside the Admin tab's check of a file), so the odor sheet's missing rows
  and its sync are offered by the same code in the browser (`odorSheetHelp`). A MetaCyc
  view's are matched as `metacyc.match_odors` matched them when it was made (its
  record's `matched`), each identified by its frame ID: matched again by
  label or CID, a second compound of one odor (its other stereoisomer, or the acid of
  the anion MetaCyc keeps) would be it too.
- **Downloads of a reference dataset or a model run** (`matrix_download.py`) are written by
  the server, as a dataset's are (a matrix may be MetaCyc's 43 million cells), but of what
  the browser shows: it sends the rows and columns drawn, as their indices in the matrix as
  read, in the order drawn (`MatrixViewer.shownKeys`, read off the model drawn, so whatever
  ordered it, dragging, sorting or clustering, needs no copy on the server), and whether it
  is transposed. Each item of a table of one value per line or a DataArray keeps its value
  of each field (`fields`, imports of record version 2), so that each is a level of its own
  (VCF's product and reference): its label alone joins them into one text.
- **MetaCyc** (`metacyc_build.py`, `metacyc.py`, `static/metacyc.js`) is a reference
  dataset of a kind of its own: a bundle (one SQLite file, read only and immutable, so no
  locks on NFS) built by a script from MetaCyc's SmartTables, not a matrix. Two views,
  each its own reference dataset (added once for each), made when it is imported (in
  about a second) and imported again whenever the odor table changes, as which compounds
  are its chemicals does (its fingerprint is kept with the import, as the file's size and
  time are):
  - *Compounds by pathways*: every compound in a pathway, the odor table's chemicals
    first (of MetaCyc 27.1, classes as fallback: 12,121 by 3,514, 43 million cells, of
    which 87 thousand, 0.2%, are not 0). It is kept and sent as its cells other than 0
    (`sparse.Sparse`; `values.npz`, 1 MB, not 170 MB; `values_sparse` in the payload),
    and made whole only in the browser (170 MB, in 60 ms; the viewer shows it as given,
    not copied, while nothing is removed or moved). The Reference data tab can show only
    the odor table's chemicals, and the pathways they are in (`refSubset`), as the view
    was made before; the Compare tab compares its odors only along those pathways, as
    pathways none of them is in would add agreement of 0s to every correlation.
  - *Pathway LCA distance*: pairwise (`pairwise: "distance"`: its columns its rows,
    served `square`) between the odor table's chemicals alone (as its notes say): of every
    compound in the network (12,566 of 27.1) it would be 158 million cells (630 MB), nearly
    none missing (each reaches about 23 thousand of its 34 thousand nodes), from a
    breadth-first search of the whole network for each compound and, for each pair, a sum
    over every node. Made again, as the pathways view, whenever the odor table changes
    (`odors_fingerprint`). Compared by
    its own values negated (the nearer, the more alike), as a distance between two odors
    is not a vector to correlate.
  - *Looked into*: what is drawn and listed below the matrix is asked of the bundle when
    needed (`/api/reference/{id}/metacyc/...`).
  - *Its tables*: `BioCyc/raw` holds SmartTables of MetaCyc 27.1 (January–February 2024,
    as the thesis used, though it says 27.0), each exported twice, by frame IDs and by names, in the same row
    order: IDs come from the first and the names of what each cell lists from the second,
    by position. Reaction directions come from their equations' arrows (`->` and `<-`
    both run from the reactants to the products as listed by ID; `<-->` reversible).
    Not in them: the pathway class tree beyond each pathway's own classes (only part of
    it, folded, in `pathway_ontology.txt`), which compounds of a reaction are its main
    ones in a pathway, and in which direction a reversible one runs there. These are in
    MetaCyc's flat files (parsed, of 28.0, in `BioCyc/processed/flatfiles-28.0`), which a
    later bundle could be built from instead.
  - *The pathway LCA distance* (`metacyc.Network`): the smallest sum, over common
    ancestors other than the two, of the fewest edges down to each (breadth-first up from
    each chemical over its parents, then the minimum over nodes of the sums: n × nodes
    distances, n² sums). The thesis's Methods describe a network of compounds and pathways
    (and leaving out compounds in over 240 pathways), but the matrices of February 2024
    in `BioCyc/processed/networks/reaction_network/` (no code with them) are of compounds
    and reactions: through reactions, without the same 15 currency metabolites (by frame
    ID), and a compound with nothing above it taken as the class it is of (as 1-penten-3-one
    as "a long-chain ketone" there), this reproduces all 380 off-diagonal values of
    `lca_with_ontology` for its 20 odors. That is the default; through pathways (each
    pathway a parent of every compound of its reactions, a compound in a pathway only
    through a class one edge further) is an option, as is the cutoff.
  - *Odors to compounds*: MetaCyc keeps compounds as at pH 7.3 (acetate, not acetic
    acid) and often one stereoisomer, so after the MetaCyc ID the odor table gives (as
    the thesis matched them by hand), InChI, InChIKey and CID, the first block of the
    InChIKey (the skeleton) matches, choosing the compound in the most pathways, then a
    neutral one.
  - *A pathway drawn* (`layoutPathway`): its main compounds (all but common cofactors and
    compounds in more than 100 pathways, which are shown on hover) in layers, top to
    bottom, each reaction an arrow from each main reactant to each main product.
    Reversible reactions run away from where the pathway starts (its primary reactants,
    else what it consumes and does not make), as MetaCyc writes some backwards (MEP ⇄ DXP);
    cycles are broken by a depth-first search from there, layers are the longest path to
    each compound, ordered by the mean position of their neighbours (sweeps down and up),
    and wrap when wider than 1,100 px. No graph library: none is used elsewhere, and
    pathways are small (a superpathway's tens of reactions).
- **LOTUS** (`lotus_build.py`, `lotus.py`, `static/lotus.js`) is a bundle too, read as
  MetaCyc's is (`reference.bundle_kind`: by the kind its `about` says, as both are
  `.sqlite`), for the same reason: 220,823 structures by 36,800 organisms, of which only
  the odor table's chemicals can be compared with datasets. Its one view is made over
  them, then (as MetaCyc's compounds by pathways is over all its compounds, a box
  showing only the odor table's) its other chemicals of at most 250 Da by default
  (`others`: 200, 250, 300 or none): all 165k, merged but for R/S, are 1.2 GB by
  families as the grid the viewer makes of a matrix (5.7 by genera, 21.5 by species),
  though 99.9% of their cells are 0; odorants are small, and those of at most 250 Da are
  22k (104 MB by the 1,178 families they are in), as many cells as MetaCyc's. The view is
  kept sparse (`Sparse`), as MetaCyc's is. It is made by the settings in its `fields`;
  unlike MetaCyc's views, these are changed in place
  (`PATCH /api/admin/references/{id}`, `reference.update`, which any reference dataset
  added in the Admin tab could use), the import made again as for any change to `fields`.
  The settings and what the bundle was made from are said above the matrix
  (`lotus.describe`: each with whether the Admin tab changes it), as they were chosen
  for, not read off, the data.
  - *Which release*: `LOTUS/raw` holds two. The 2021 one (`NPOC2021`, the MongoDB dump of
    lotus.naturalproducts.net, from which `LOTUS/processed` was made) has 668k
    structure–organism pairs, LOTUS IDs, and every organism's lineage in each of 9
    taxonomy sources, but no CIDs; the frozen export of Wikidata of 2023-01-06
    (`230106_frozen`, 533k pairs with their papers) with the metadata of 2022-09-16 (Zenodo
    7534077: every structure's CID, names, and lineages in 13 sources, order included)
    has all 64 odors of the thesis template with a CID, where the 2021 one has 58. It is
    not a superset (247k pairs are only in 2021's), but is the one LOTUS curates; the
    bundle is made of it, with LOTUS IDs joined from the 2021 SDF by InChIKey (90% of
    structures), as the odor table links odors to lotus.naturalproducts.net by them.
  - *Stereochemistry*: LOTUS gives each record a structure as its source drew it, often
    an enantiomer the paper did not determine, or none: (−)-α-pinene is in 116 organisms
    as its exact structure, 1,152 as any α-pinene. By default an odor is every structure
    of its InChI without tetrahedral stereo (`/t`, `/m`, `/s` layers): enantiomers,
    diastereomers and unspecified stereocenters merged (so menthol with neomenthol), but
    not cis/trans isomers (`/b`), which are other odors (geraniol, nerol). Exact and
    skeleton (InChIKey's first block) are options, kept as keys of each structure.
  - *Taxonomy*: LOTUS gives an organism's lineage in each source that has it, and they
    differ (kingdoms by other names: Viridiplantae, Chloroplastida; families of two names;
    homonyms across kingdoms, 35 genera in GBIF alone, e.g. *Crambe*, a plant and a moth).
    One source is used first (GBIF's backbone, of 32,798 of the 36,800 organisms), then
    the others in turn where those before give the organism nothing at the level; kingdoms
    are kept as GBIF names them (`kingdom_of`), as are the eight families conserved under
    two names (`FAMILY_NAMES`: EUNIS has Compositae), and taxa are told apart by name and
    kingdom. Columns are ordered by lineage, kingdoms with the most organisms first.
  - *A cell* (`lotus.cell`, `/api/reference/{id}/lotus/cell`): the organisms of its taxon
    that its row's structures are reported in, each with its papers (titles as plain
    text: Crossref's markup, in 17k of them, dropped with the spaces it ran over kept).
- **Bundles imported in the Admin tab** (`bundle_import.py`, with `metacyc_import.py`
  and `lotus_import.py` for each kind; `static/imports.js`): MetaCyc's and LOTUS's are
  made on the server from what their scripts read, the raw folder (or an archive of it).
  Its files are found as the script finds them and checked as it checks them
  (`metacyc_build.problems`, `lotus_build.problems`: each required file there, with the
  columns it needs; then, building, MetaCyc's frame IDs where they should be and its
  compounds' pathways among its pathways), each problem saying what was expected. The
  browser lists them from the folder chosen and uploads only those, so a folder of
  anything else is refused before it is sent. Or the folder (or archive) is a path on
  tensor (`Importer.tensor_source`, `tensor.locate`: checked as data folders are, with
  their messages): the server lists the folder's files, picks those the script reads as
  it picks an upload's, and the tab shows them as it shows a folder chosen; importing,
  it copies them (gzipping as kept) or takes them out of the archive, in the background
  (a stage of the job, "copying": LOTUS's 1.1 GB SDF takes a while to gzip), then checks
  them as an upload's, and keeps the path in the import's record. They are kept, laid
  out as the script reads them (it can be run on them again), with the bundle, in
  `data/<kind>/<when>/` (backed up), and only what was built and added from: an upload
  refused or failing keeps nothing. The build is the script itself
  (`python -m app.<kind>_build`), in a process of its own, one import of each kind at
  a time, polled by the tab: LOTUS's takes 1.7 GB for a minute, which the server's
  process would keep after, and holds Python's GIL for most of it. That process is held
  to the server's folders as the server is (Landlock is inherited), so its temporary
  files (SQLite's, sorting to index) go in `data/tmp/`. The datasets are added as the
  Admin tab adds any (`reference.add`). The folder goes to `data/deleted/` with the last
  dataset of its bundle, as a deleted dataset's files do, so nothing uploaded is left
  unused and unlisted.
  - *MetaCyc's* files are found by name (the paths it reads them from are fixed): 89 MB
    of `BioCyc/raw`'s 283. Its version is told from them (`metacyc_build.identify`), as
    SmartTables do not say it and one typed by hand is easily wrong (the thesis says
    27.0 of tables that are 27.1's): MetaCyc's release notes say how many pathways (not
    counting super-pathways) and compounds each release has, and exports match them (27.0's
    of July 2023 exactly; 28.0's of May 2024 within 1 and 4), so the release is the one
    they match, else the last they have more than and fewer than the next (as metacyc.org
    had it between releases: January 2024's has 5 pathways more than 27.1's notes say).
    When the files were last changed (kept from the browser's, or the archive's) is said
    beside it, as a check, not a basis: a copy changes it. A version given that is another
    release's is refused; one that isn't a release's (or of tables whose release cannot be
    told) is taken as given. `RELEASES` needs each new release's counts added.
  - *LOTUS's* are found by their paths (sent as the files' names), with the script's
    globs (`lotus_build.FILES`, each gzipped or not), and kept at the part of the path a
    glob matched: the release's dates are in the folders' names (`metadata_2022-09-16/`),
    and the notes beside the files (`notes.txt`, where they came from) are kept with them,
    as the script reads them. They are kept gzipped, as the script reads them too: of a
    file there both as is and gzipped, the gzipped one is uploaded, and the browser gzips
    the others first (`CompressionStream`: the 1.1 GB SDF to 97 MB in 7 s), as the server
    does those it was sent as they are or takes out of archives. Of `LOTUS/raw`'s 7.3 GB,
    330 MB is sent. Each gzipped copy is a Blob made of its chunks, held in memory until
    the upload: one built from the stream (`Response.blob()`) Chrome keeps in files of its
    profile past a few MB (165 MB of a 170 MB one, tried), and on a computer short of disk
    room that fails part-way, saying only "Failed to fetch" (as an import from a laptop
    did, at 120 MB of the 1.1 GB SDF). A file chosen is a snapshot: once it changes (even
    its time alone) or its drive will not give it, Chrome refuses to read it, and one cut
    short while it is read is read only so far, with no error. So each file is checked
    before anything is sent (the first byte of those sent as they are, which would
    otherwise fail the upload as a server that cannot be reached; all of those gzipped
    first), and one that fails is named, with how much of it was read (`Unreadable`, in
    `static/imports.js`).

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

## Headspace calculator

The Headspace calculator tab gives how much of each odor of a mixture is in the air above
it: odors of the odor table diluted together in one vial (each a log10 v/v concentration,
as in the data), at equilibrium at 25 °C, in paraffin oil or water. The physics is in
`headspace.py`; the properties it needs of each odor are looked up through chemutils
(`chemprops.py`) the first time a calculation needs them, and kept in the `pubchem` table
with the odor's CID, so that a calculation needing nothing new does not start chemutils.
(Constants in oil are computed by chemutils, not looked up: with `thermo` and `ugropy`,
offline, a second or so each, the first time.)

- **Paraffin oil, by Henry's law** (the default): p = c / H_cp, with c from the odor's
  volume, density and molar mass, and H_cp = c_oil / (γ∞·p_sat), as chemutils estimates it
  for n-hexadecane (`inchi2k_henry_pfo`, with K_oil/air = c_solv·R·T / (γ∞·p_sat), and
  n-hexadecane's own c_solv, 3401.8 mol/m³), but of the oil (`headspace._law`): c_oil is
  its moles in a volume, from its density and molar mass, and γ∞ the activity coefficient
  of the odor when dilute by the same modified UNIFAC (from its functional groups,
  `unifac_groups`, kept with chemutils' constant), in the n-alkane of about the oil's
  molar mass (`headspace._gamma_inf`, with `thermo`, a millisecond or so; C32 at 450
  g/mol), and p_sat the one chemutils used, from `thermo`'s fits to measurements where it
  has them (else PubChem's, else estimates). A test checks that of n-hexadecane's own
  molar mass and density (226.45 g/mol, 0.7703 g/mL) it is chemutils' constant.
  γ∞ cannot stay n-hexadecane's when the molar mass changes: a larger solvent lowers both
  c_oil and γ∞, which mostly cancel, as K_oil/air is by volume. From C16 to C32, by
  UNIFAC, c_oil halves, but H_cp falls only to 0.64-0.79× of what it was (at one density).
  Halving H_cp would overestimate the headspace by 1.3-1.6×. That is Raoult's law below
  times γ∞: e.g. ethyl acetate's is 2.0 (at -3, 138 Pa, 1360 ppm) and ethanol's 20. No measured solubility in oil or alkanes is used
  (chemutils can take one, but none is at hand), so each constant is an estimate, its
  vapor pressure measured or not (both said in the table). Like Henry's law in water, it holds for
  dilute solutions: alcohols and acids associate as they get more concentrated, which
  lowers their headspace again (chemutils' note on them says so), and p is at most p_sat.
  With no constant estimated (no UNIFAC groups or parameters for it), Raoult's law.
- **Paraffin oil, by Raoult's law only** (if chosen): an ideal solution, p = x·p_sat. The
  moles of each odor are from its volume, density and molar mass; those of the oil from
  its density and molar mass. E.g. ethyl acetate at -3 (0.900 g/mL, 12.4 kPa): x = 0.0054,
  67 Pa, 660 ppm. An ideal solution takes molecules of about one size, which a viscous
  oil's and an odor's are not: x rises in proportion to the oil's molar mass, without the
  fall in γ∞ below 1 that comes with it (limonene's 1.05 in n-hexadecane, 0.78 at C32), so
  it overestimates more the larger the molar mass, and is above Henry's law for odors
  with γ∞ below 1.
- **Paraffin oil's density and molar mass**: given with the mixture (`oil_density`,
  `oil_molar_mass`; `headspace.SOLVENTS` where blank), for the mixture, the serial
  dilutions and the oil to match VCF's foods. By default 0.85 g/mL, within what is sold as
  viscous paraffin oil (e.g. Sigma's 18512: 0.827-0.890 g/mL and 110-230 mPa·s at 20 °C, as
  Ph. Eur. has liquid paraffin), and 450 g/mol: such an oil is a mixture, and its suppliers
  do not give an average. JECFA's classes of mineral oil have averages of 300-400 g/mol at
  3-7 mm²/s at 100 °C, 400-480 at 7-8.5, 480-500 at 8.5-11 and over 500 above; 110-230
  mPa·s at 20 °C is roughly 7-11 mm²/s at 100 °C (by typical white oils' viscosity
  indices: not measured), so 400-500 g/mol, and 450 its middle. Between those ends, the
  headspace changes by about ±5% by Henry's law and ±11% by Raoult's law.
- **Water**: Henry's law, p = c / H_cp, with c from the odor's volume, density and molar
  mass, and H_cp (mol/m³/Pa) from henrys-law.org (Sander's compilation), else NIST, both
  at 25 °C. It holds for dilute solutions only, so p is at most the odor's vapor
  pressure. With no constant found, Raoult's law for an ideal solution in water, which is
  too low for most odors (esters' activity coefficients in water are from tens to thousands).
- **Vapor pressure**: PubChem's (the mean of its measured values at or near 25 °C, else
  of those with no temperature given, else of the estimates it lists), else estimated
  from the structure by SIMPOL.1 (`chemutils.inchi2est_vapor_pressure`, often within 3×,
  but several times too low for furans, tertiary alcohols and some phenols); in oil by
  Henry's law, the one its constant was estimated with. Each is labelled measured or an estimate as chemutils
  says (`pubchem.<property>_measured`): SIMPOL.1's always are estimates; PubChem's are
  measured unless it lists only estimates (those looked up with chemutils before
  fda8e3b are not labelled until looked up again in the Admin tab).
- **Density**: PubChem's (most of its values relative to water, taken as g/mL), or
  ChemSpider's, given a key; else 1 g/mL, for most odors between 0.8 and 1.1 g/mL, so
  off by at most ~20%.
- **Units**: from the partial pressure at 1 atm and 25 °C: ppm and ppb of the air by
  volume (p / 101325 Pa), log10 of that v/v, mol/L and molecules/cm³ by the ideal gas
  law, and µg/L (mg/m³) with the molar mass. These are computed in the browser, so that
  changing them asks the server nothing.
- **How each was calculated**: `calculate` also gives the steps to each partial pressure
  (each odor's v/v, density as used, moles, and its pressure by Henry's law before it is
  at most p_sat; the solvent's moles, and all of them), which the browser shows by each
  headspace, with the values used, to the value shown: what was computed, not computed
  again (the browser itself computes only the units, c = n·10⁶, c_oil, and the sums it
  shows). What each symbol is, and its value, is said where it is first used.
- **Kept in the browser**: each calculation since the page was opened, by its mixture,
  solvent, model and oil, so that going back to one shows it at once (and asks nothing).
- **Panels**: a mixture can start from a panel of the datasets: the odors presented in it
  (over every dataset with it, matched to the odor table as in Browse) that are in the
  odor table with a CID, but solvents (`headspace.SOLVENT_NAMES`), each at the highest
  concentration presented, with what was left out said. E.g. kiwi: EtOH -2, IAol -3.6,
  IaA -3.7, ea -4.2 and eb -3.5 (not pfo, nor the mixture kmix), as in its YAML.

What is left out: UNIFAC is within about 2× of measured oil/air partitioning for simple
functional groups, less sure for acids (which dimerize); how it changes with the oil's
size has not been checked against measurements here, and it takes the oil as an n-alkane,
with no rings or branches (squalane, C30 and branched, gives about what a straight C30
does); Raoult's law only leaves out that polar odors are less soluble in oil than an ideal
solution, so it is too low for them, and the size of the oil's molecules, so it is too
high for odors with little else to tell them from oil. Neither models air flowing over the vial (which depletes the headspace) or
diluting it before it reaches the fly.

### Serial dilutions

Below the mixture, how to make each of its odors at its concentration from the neat
chemical (`dilution.py`, in the server, so that it is tested; `POST
/api/headspace/dilutions` as typed), in the mixture's solvent, each in vials of its own:
into each vial the same amount of solvent (2 mL by default; paraffin oil weighed, as it
is viscous, by the density given with the mixture; water measured by volume, with no
density), then the neat chemical (into the first) or some of the vial before. Volumes are taken as additive, so v µL into V µL of
solvent dilutes by v / (v + V), not v / V: 2 µL of the neat chemical into 2 mL is -3.0004.

- **The neat chemical's volume** changes from odor to odor, within its limits (2 to 200
  µL by default, so from -3.0 to -1.04 in 2 mL): an odor within them needs no dilution;
  one above them cannot be made in one vial's solvent (what it would need is said).
- **The dilutions' volume** is the same for every dilution of every odor where it can be,
  so that their pipette can be left on one setting (as diluting viscous paraffin oil is
  most of the work). Of round volumes (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 × 10^k µL),
  and those that just make an odor's concentration with the most or least of its neat
  chemical (for neat chemicals' limits too narrow for round volumes), it is chosen: so
  that as many odors as can be are made within the neat chemical's limits; within the
  dilutions' limits (by default from 100 µL, or a vial's solvent if less, to a vial's
  solvent), unless none within them does as well; the same for as many odors as can be
  (each volume so in turn, of those left: a volume of their own only where no one volume
  makes them all); with the fewest dilutions; with the least volume pipetted (of the neat
  chemicals and the dilutions) as large as can be, as pipettes are least precise at
  their least, then the next least, and so on; then round (so only between volumes as
  good by all else, e.g. a dilutions' least typed as 23.7 µL is kept over 25). With d =
  log10((v + V) / v) the decades a dilution takes, an odor at c needs the fewest n with
  c + n·d at least the neat chemical's least; any d no more than the neat chemical's
  range (1.96 decades by default, any v of at least 22 µL in 2 mL) makes every odor, but
  a larger d (a smaller v) can take fewer dilutions: e.g. -4.2 and -6 take one and three
  of 100 µL, one and two of 20 µL, or one each of 2 µL. The least by default, 100 µL, is
  so that viscous paraffin oil is not pipetted in volumes too small to pipette well; as
  smaller dilutions need more of each neat chemical (for as many dilutions), the volume
  chosen is mostly that least.
- **Volumes** are rounded to 3 significant digits, as pipettes are set, and the
  concentrations of the vials shown are from them as rounded.

What is left out: the volume taken out of each vial before the last (it does not change
the next's concentration), and mixing the odors in one vial (each is made on its own).

### Foods' volatiles (VCF)

Below the mixture, a search of VCF's foods (`vcf.py`): of a reference dataset of one value
per line whose file also has `k_henry` and `vapor_pressure` columns (`vcf.sources`), as
the VCF CSVs made with chemutils do. The file is read again as its lines (kept in memory
while it is unchanged), its chemicals and products being the reference dataset's rows
and columns (by the fields chosen for it), and its values taken as mg/kg.

- **Search**: what is typed, in a product's name, category or group (but for case).
- **One quantity per chemical**: each reference's lines on a product averaged (as in the
  reference dataset's matrix: e.g. two cultivars), then reduced across the product's
  references, then across products (median by default; mean, geometric mean, max or
  min), each over those that quantify it: a paper that does not list a chemical may not
  have looked for it, so absence is not taken as 0.
- **Headspace over the samples** (`headspace.sample`, `vcf.add_headspace`): each taken as
  water (0.997 g/mL), each chemical on its own (dilute): c = ppm · ρ_water / M,
  p = c / H_cp, at most its vapor pressure; else Raoult's law with its vapor pressure,
  too low for most volatiles; else none. M is from the InChI's formula
  (`vcf.molar_mass`), so nothing is looked up for it. Sugars, salts and fat in a real
  sample change these (fat holds hydrophobic volatiles back); nothing models that.
- **Constants and vapor pressures** (`vcf.USED`): looked up again through chemutils
  (`inchi2k_henry`: henrys-law.org, else NIST, whose mol/kg/bar are taken per kg of
  water; `inchi2vapor_pressure`: PubChem's at or near 25 °C), as the file's are from an
  older chemutils (its vapor pressures from values at any temperature: isoamyl alcohol's
  1.7 kPa, now 0.32), and henrys-law.org has several values for most chemicals. What
  chemutils finds is used, else the file's, else estimates (`inchi2est_k_henry`:
  solubility in water over vapor pressure, VP/AS, only reasonable for chemicals not very
  soluble; `inchi2est_vapor_pressure`: SIMPOL.1), asked for only once neither the file
  nor the lookup has one. A difference of more than `vcf.MISMATCH` (10%; about two in
  three of banana's) is shown with both values and a warning. They are kept in the
  `vcf_properties` table, by CID and property, apart from the odor table's `pubchem`
  (so that the Admin tab's counts and looking up again are of the odor table's alone),
  and looked up in a thread of the server's (`main._vcf_look_up`, 10 CIDs at a time) by
  priority (`main.VCF_PRIORITY`): of chemicals the file has neither of, then no
  constant of, then what paraffin oil needs, then checking the file's. Each search asks
  for what its chemicals lack (those that could not be looked up in the last hour
  aside), and the browser asks again every 3 s while any is being looked up.
- **Paraffin oil for the same headspace** (`headspace.invert`): `calculate` backwards, of
  every chemical with a headspace, together. By Henry's law each v/v is from its own
  headspace, f = p·H·M/(10⁶·ρ); by Raoult's law from x = p/p_sat, with the moles of the
  whole solution N = (n_oil·(1 − F_H) + n_H) / (1 − Σ x·(1 − n_oil/n_pure)) (F_H and n_H
  the volume and moles of those by Henry's law, n_pure the moles in a mL of each, pure),
  so that `calculate` of the result gives the headspace back exactly (tested). A
  headspace of at least the vapor pressure cannot be reached. What it needs of each
  (density, vapor pressure, constant in oil) is looked up through chemutils as above,
  kept in `vcf_properties`. At most 500 chemicals with a headspace per search, so that
  a search of `a` does not look up thousands for oil.
- **Sorting**: the tables are in the order chosen (quantity, headspace or name; the
  matching lines by their chemicals', then by value), but *Make them the mixture* takes
  those with the most headspace first.
- **Make them the mixture**: those whose v/v was computed that are odors of the odor table
  (matched as reference datasets' chemicals are, by CID, else InChI, ...), at most 50,
  most headspace first, become the mixture above in paraffin oil, by the same model, to
  check or change (their headspace is then calculated as the mixture's).

## Code layout

```
app/main.py        HTTP API (Starlette) and static file serving
app/config.py      runtime configuration (environment variables)
app/ingest.py      CSV/parquet loading + validation → units, trials, float32 matrix
app/odors.py       odor string parsing/formatting (hong2p.olf's conventions; stricter parsing of uploads)
app/glomeruli.py   accepted glomerulus names (config/glomeruli.txt overrides)
app/yamlcheck.py   odor YAML parsing, what looks wrong in one, and the matrix ⊂ YAML check
app/thorimage.py   Experiment.xml summary, and a .raw file's layout (with hong2p.thor)
app/fits.py        fits of al-diag-seg models: records, options, held-out flies, their manifest, time estimates
app/fit_worker.py  the fit worker (python -m app.fit_worker): runs fits one at a time, with al-diag-seg,
                   and predictions beside them (GPU or CPU)
app/predictions.py predictions of flies' glomerulus ROIs by fits' models: records, the fly's folders, the viewer's
                   images, downloads
app/model_data.py  model data for the Models tab: folders of model outputs on tensor, scanned into a table of
                   models (params.csv rows, provenance.json keys), their spike count files matched, a run read
app/metadata.py    form model (pydantic) and README.md generation
app/db.py          SQLite schema and queries
app/reference.py   reference datasets: built-in and added ones, reading them, matching their chemicals
app/metacyc.py     MetaCyc bundles as reference data: odors matched to compounds, compounds × pathways, the
                   pathway LCA distance, pathways, compounds and searches for the Reference data tab
app/metacyc_build.py  MetaCyc's SmartTables checked and read into a bundle (python -m app.metacyc_build), and
                   its two reference datasets added through the Admin tab's API
app/bundle_import.py  bundles imported in the Admin tab: the raw folder uploaded, kept, built (by the build
                   script, in a process of its own) and its reference datasets added
app/metacyc_import.py MetaCyc's files of an upload, found by name, checked as metacyc_build.py checks them, its
                   version told from them
app/lotus.py       LOTUS bundles as reference data: its view's settings (and how they are said), odors matched to
                   structures, chemicals × taxa, a cell's organisms and papers
app/lotus_build.py LOTUS's frozen records and metadata checked and read into a bundle (python -m
                   app.lotus_build), and its reference dataset added through the Admin tab's API
app/lotus_import.py LOTUS's files of an upload, found by the paths lotus_build.py finds them by, checked as it
                   checks them
app/export.py      downloads: zips of dataset files, concatenated wide/long matrices
app/merge.py       merged downloads of a coding unit: one calculation of the same recordings, grouping datasets,
                   scaling groups to each other
app/compare.py     datasets averaged for the Compare tab (odors × glomeruli or ROIs); ECFP fingerprints (RDKit)
app/odormeta.py    odor table: sheet parsing / validation, sync diff, links, dataset matching
app/chemprops.py   odors' vapor pressures, molar masses, densities and Henry's law constants, looked up through
                   chemutils in a process of its own
app/headspace.py   headspace concentrations of odors diluted together (Headspace calculator), over samples
                   taken as water, and the concentrations giving a headspace
app/dilution.py    serial dilutions of odors from their neat chemicals (Headspace calculator)
app/vcf.py         VCF's foods, from a reference dataset of its quantities: search, one quantity per chemical
app/tensor.py      users' folders on tensor: path checks, finding a dataset's flies, recordings, YAMLs
app/sandbox.py     folders the server may read and write (Admin tab): settings, path checks, Landlock
app/filelog.py     the files the server has read and written in its own folders (Admin tab), paths on the server
app/backup.py      backups of the database and files into a folder the server may write (Admin tab)
app/movies.py      a dataset's movies on tensor: finding them, reading their planes, their ImageJ ROIs
config/odor_sheet_template.csv   odor sheet columns, with the odors from the thesis tables
app/static/        frontend: index.html, app.js, heatmap.js, colormaps.js (+ mpl_luts.js),
                   app.css, cluster.js (+ cluster-worker.js), figure.js (SVG/PDF/PNG export),
                   player.js (movie player), similarity.js (odor similarities, and between them;
                   pvalue-worker.js their p-values), metacyc.js (MetaCyc below its matrices: pathways
                   drawn, compounds, LCA paths, search), lotus.js (LOTUS with its matrices: how
                   made, a cell's organisms and papers), imports.js (MetaCyc and LOTUS imported
                   in the Admin tab), fits.js and predictions.js (the AL segmentation tab's fits and
                   predictions), models.js (the Models
                   tab: its table of models and a run's viewer), latex.js (labels with
                   LaTeX math, by MathJax: vendor/mathjax/)
manage.py          list / show / export / delete datasets from the server
deploy/            systemd units; Dockerfile + docker-compose.yml at the top level
tests/             pytest tests
```

## Storage

`data/olfdb.sqlite3` tables: `datasets` (metadata, README, summary), `dataset_panels`,
`units` (date, fly_num, roi, bad, other column levels), `trials` (panel, odor, components, repeat), `responses`
(one row per cell, for SQL queries), `matrices` (the same values as a float32 blob for
fast serving), `users` (each user's folders on tensor), `odors` and `odor_syncs`,
`pubchem` (each odor CID's InChI, and its properties looked up through chemutils: vapor pressure,
estimated vapor pressure, molar mass, density and Henry's law constants in water and paraffin oil, each
with where it is from, when it was looked up, the chemutils commit that did, whether measured, and for
constants in oil what they were estimated with), `vcf_properties` (the same of VCF's chemicals, and
estimated Henry's law constants, by CID and property, apart from the odor table's), `settings` (site-wide settings from the Admin tab:
the odor sheet's link, panels shown renamed), `reference_datasets` (reference datasets added there),
`model_sources` (folders of model data added there, by their path on tensor, with their lists of parameters and of spike count files that are not models),
`model_baselines` (models chosen there to show first, by their experiment's folder on tensor, name, base and row; when their files were imported, and their sizes, times and SHA-256), `fits`
(fits of al-diag-seg models) and
`file_log` (the files the server has read or written in its own folders). Files:
`data/datasets/<id>/` holds the original data file (or files, if several were combined:
`datasets.data_files`, with each one's sha256; `data_file` and `data_sha256` are the first's),
`README.md`, `metadata.json`,
`odor_yaml/*.yaml` and the `Experiment.xml`. Also
under `data/`: `admin.json` (the folders set in the Admin tab), `deleted/<id>/` (the
folders of datasets deleted in the Admin tab), `references/<id>/` (reference datasets
added there, as read), `staging/`, `exports/`,
`odor_previews/` (parsed sheets awaiting confirmation), `tmp/`
(temporary files, such as uploads being received) and `chemutils_cache.sqlite`
(chemutils' cache of what it looked up; not backed up). An `odor_structures/` folder, of
PubChem's images, is left from before structures were drawn here, and can be deleted.

## HTTP API

Used by the frontend; handy for scripts too.

```
GET    /api/health                          {"ok": true, "time", "started", "version"} (when the server started; app/version.py)
GET    /api/options                         dropdown values (indicators, drivers, panels, ...)
POST   /api/metadata/check                  {"metadata": {...}} → problems with upload form defaults
GET    /api/user?name=                      a user's folders on tensor ("user": null if none saved)
POST   /api/user                            {"name", "raw_dirs", "mocorr_dirs", "odor_dirs"} → each
                                            folder checked on tensor; saved if all are there
POST   /api/tensor/check                    {"path"} → a dataset's data folder checked on tensor
GET    /api/tensor/list?path=[&archives=1]  what is in a folder on tensor ('' its top), to complete a path
                                            typed: its folders (and archives; or with &files=reference,
                                            the files a reference dataset can be read from), not hidden
                                            ones nor a NAS's own (`tensor.list_dir`: listing one is
                                            0–2 ms there)
POST   /api/stage                           multipart 'file' (repeatable: their data one dataset's) →
                                            validation + matrix payload (422 with ``rois`` when ROI
                                            names are not accepted: each, and the accepted glomerulus
                                            names; of several, ``files``: each not accepted, with its
                                            problems)
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
POST   /api/stage/{id}/readme               {"metadata": {...}} → README.md text, and the orphaned
                                            permalink it would take over: {slug, exact, differences}
POST   /api/stage/{id}/commit               {"metadata", "readme", "readme_edited", "permalink_proceed"}
                                            → {"dataset_id", "permalink": the one it took over}
                                            (422 unless each odor is one in the odor table)
GET    /api/datasets                        list, and calculations (the sets of the same recordings
                                            calculated otherwise, each [{ids, default}])
GET    /api/datasets/{id}                   metadata + README (markdown and HTML, and its path), its
                                            permalink and default_permalink (the name one would have),
                                            calculations (the other datasets of its recordings
                                            calculated otherwise: [{id, title, response_calc}]) and
                                            default_calc (the id of their default, null if none)
PUT    /api/datasets/{id}/default_calc      optional {"others": [id]} → it made the default calculation
                                            of its recordings, and each of "others" of its own:
                                            {ok, ids, default, others: [{ids, default}]} (422, setting
                                            none, if one has no other calculation, or two are of the
                                            same recordings)
PUT    /api/datasets/{id}/readme            {"readme", "base"} → the README overwritten (409 if it is
                                            no longer "base", the one the edit started from)
PUT    /api/datasets/{id}/metadata          {"metadata", "base"} → the values entered at upload replaced,
                                            checked as an upload's: {changed, readme: "generated" |
                                            "kept", permalink} (409 if they are no longer "base"; 409
                                            {permalink_change: {from, to}} until "permalink_proceed" is
                                            "to", if they would rename its permalink)
PUT    /api/datasets/{id}/permalink         {"slug"} → its permalink made or renamed (422 if not a
                                            name; 409 if another's)
GET    /api/permalinks/{name}               {slug, dataset_id (null: orphaned), custom, summary, …}
GET    /d/{name}                            → #browse/{name}
GET    /api/datasets/{id}/matrix            units, trials, base64 float32 values
GET    /api/datasets/{id}/files/{path}      original files
GET    /api/compare/datasets/{id}           its odors × glomeruli (or ROIs), every trial (and fly) averaged,
                                            each odor's odor table id if it is one chemical alone;
                                            ?units=1: by ROI, even of glomeruli
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
GET    /api/movie/rois?path=[&dataset=]     the ImageJ ROIs of a RoiSet.zip (or .roi), with their planes and
                                            kinds (glomerulus, tentative, other; also by the dataset's ROIs)
GET    /api/movie/trials?path=              the trials of an al_analysis trial_frames_and_odors.json: frames, odor
GET    /api/download?ids=1,2&format=files|wide|long
                                            a zip, or a CSV for the matrix of a single dataset;
                                            &lower=1 for odor names in lowercase (but in files)
GET    /api/merge                           the coding units recorded from, with their datasets (and how
                                            many are other calculations of another's recordings)
POST   /api/merge/preview                   {"coding_unit", "mode": "none"|"groups", "by": [...],
                                            "target", "tolerance", "power_tolerance", "calc"} → the
                                            datasets merged and those left out (other calculations of
                                            their recordings), the groups, their factors and each dataset's
GET    /api/merge/download?coding_unit=&format=wide|long&mode=&by=a,b&target=&tolerance=&power_tolerance=&calc=
                                            every dataset of the coding unit (one calculation of the same
                                            recordings), scaled, as a zip (&lower=1 too)
GET    /api/merge/scales?(the same, without format)
                                            each dataset's group and scale factor (scaling.csv)
GET    /api/roi_training[?datasets=1,2]     the flies of glomeruli datasets with a diagnostic panel to
                                            export for ROI training (each its files' total size, its
                                            glomeruli and its panels but diagnostic ones), and those
                                            left out, with why (roi_training.py)
GET    /api/fits                            every fit (those running and waiting with when they should end), and
                                            the fit worker's status (fits.py)
POST   /api/fits                            ask for a fit: {name (a first name), options}; who and from where are kept
POST   /api/fits/{id}/cancel                stop a fit (one waiting now, one running at its next step)
DELETE /api/fits/{id}                       delete a fit and its files (409 while it runs)
GET    /api/fits/{id}/log                   its log
GET    /api/fits/{id}/download              its model and results, as a zip sent as it is written
GET    /api/models                          every model of the folders of model data added (model_data.listing):
                                            the folders (whether they can be read), the table's columns,
                                            experiments (params.csv's columns, provenance.json as
                                            parameters), models (params.csv rows, their own provenance.json
                                            parameters, runs) and files of none; gzipped (~1.2 MB → ~100 kB)
GET    /api/models/baselines                the models chosen as baselines, as /api/models has them (with
                                            whether each is found, and why not), as imported (with how
                                            their files on tensor differ: "drift"), else read on tensor
GET    /api/models/run?id=<path on tensor>  a run's spike counts: KCs (labels, types) × odors (matched to
                                            the odor table), float32 base64, and its provenance.json; of a
                                            baseline's copy (id olfdb:baselines/<id>/<file>), with
                                            "imported": {"at", "drift"}
POST   /api/models/run/download             {"id", "format": "files" | "wide", "rows", "cols", "transpose"}
                                            → its files (zip) or its spike counts as shown (CSV), see
                                            matrix_download.py
GET    /api/roi_training/download[?stacks=0&datasets=1,2]
                                            a zip of their files where they are on tensor, sent as it is
                                            written, with roi_training.json, and their anatomical stacks
                                            (but with stacks=0)
GET    /api/odors                           the odor table (with each odor's `inchi`,
                                            `vapor_pressure_kpa`, `molar_mass_g_mol`, ...)
GET    /api/odors/status                    the odor sheet, when it was last read, whether it
                                            changed since the last sync
POST   /api/odors/preview                   the sheet (or multipart 'file', a CSV) validated and
                                            diffed against the odor table
POST   /api/odors/apply                     {"fingerprint", "changes", "deletions"}
GET    /api/odors/structure/{cid}           its structure, drawn by RDKit (SVG)
GET    /api/odors/template.csv              config/odor_sheet_template.csv
GET    /api/odors.csv, /api/odors.tex       the odor table as a file (?q= as the Odors tab's
                                            filter; ?lower=1 for abbreviations in lowercase)
POST   /api/headspace                       {"solvent": "paraffin oil" or "water", "model": "henry"
                                            or "raoult" (oil only), "oil_density", "oil_molar_mass"
                                            (oil only; optional), "components": [{"odor_id",
                                            "log10_conc"}, ...]}: the partial pressure (Pa) above
                                            the mixture of each odor, with the properties used,
                                            where they are from and whether measured, and the
                                            steps to it
GET    /api/headspace/panels                the panels of the datasets as mixtures
POST   /api/headspace/dilutions             {"targets": [log10 v/v, ...], "solvent_ml", "neat_ul":
                                            [least, most], "dilution_ul": [least or null, most or
                                            null]}: for each, the µL of its neat chemical, of the
                                            dilutions, and the log10 v/v of each vial
GET    /api/headspace/vcf                   the VCF reference datasets to search, with the names of
                                            their products, categories and groups
POST   /api/headspace/vcf                   {"source", "query", "across_references", "across_products",
                                            "model": null, "henry" or "raoult", "oil_density",
                                            "oil_molar_mass" (optional)}: the matching lines,
                                            one quantity per chemical, the headspace over the samples
                                            (constants and vapor pressures as chemutils finds them,
                                            else the file's, else estimates; where they differ), and
                                            (with a model) the v/v in paraffin oil for it, with how
                                            many are still being looked up
GET    /api/headspace.csv, .tex             a calculation's table as a file: ?solvent=, model=,
                                            oil_density=, oil_molar_mass=,
                                            odors=<odor id>:<log10 conc>,... (calculated again, from
                                            what was looked up for it), units= (for LaTeX), lower=1
GET    /api/reference                       every reference dataset: whether it can be read (why
                                            not), how many of its chemicals are identified
GET    /api/reference/{id}                  one, for the viewer (its chemicals as in the odor table;
                                            of odors as datasets write them, "odor_check")
POST   /api/reference/{id}/download         {"format": "files" | "wide", "rows", "cols", "transpose"} → its
                                            files (zip) or its matrix as shown (CSV), see matrix_download.py
GET    /api/reference/{id}/metacyc/pathway/{frame}    of a MetaCyc bundle's: a pathway, to draw (its
                                            reactions' main and side compounds) and describe
                                            (classes, superpathways, species and their genomes,
                                            summary)
GET    /api/reference/{id}/metacyc/compound/{frame}   a compound or class: identifiers, classes,
                                            pathways (and through classes), reactions
GET    /api/reference/{id}/metacyc/structure/{frame}  its structure from its InChI, as SVG
GET    /api/reference/{id}/metacyc/lca      ?a=, b= (frame IDs) → their pathway LCA distance in its
                                            view's network, and the paths from their LCAs
GET    /api/reference/{id}/metacyc/search   ?q= → pathways and compounds by name or frame ID
GET    /api/reference/{id}/lotus/cell       ?structures= (its row's lotus), name=, kingdom= (its
                                            column's; or organism=) → of a LOTUS bundle's: the
                                            organisms of the taxon the structures are reported in,
                                            each with its papers
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
GET    /api/admin/panels                    the panels shown renamed, and every panel of the datasets
POST   /api/admin/panels                    {"renames": [{"from", "to"}, ...]}: saved, if allowed
GET    /api/admin/chemutils                 the chemutils installed, what was looked up through
                                            it (how much with an older one than changed how), and
                                            the last re-lookup
POST   /api/admin/chemutils/relookup        {"props": [...]}: look up again (in the background,
                                            not from chemutils' cache) every value of these
POST   /api/admin/restart                   {} : the server stops, for Docker or systemd to start it
                                            again (409 if neither would); {"full": true}: everything
                                            rebuilt and restarted by restart-olfdb, asked of systemd
                                            (409 if nothing takes the request); either 409 with
                                            "jobs" (each "what", "started_at", "ends_at", "then") and
                                            "queued" while jobs it would stop are under way, unless
                                            "stop_jobs": true
GET    /api/admin/restart                   the last full restart's output (data/restart.log): its
                                            last "lines" (30), and "failed": not restarted
GET    /api/admin/jobs                      what a full restart would stop ("jobs", "queued", as
                                            POST /api/admin/restart gives them), for restart-olfdb
DELETE /api/admin/datasets/{id}             the dataset deleted, its folder moved to
                                            data/deleted/<id>/ → {"files": where it went, "orphaned":
                                            its permalink, if any}
GET    /api/admin/permalinks                every permalink, the orphaned last
DELETE /api/admin/permalinks/{name}         an orphaned permalink deleted (409 if it points to one)
POST   /api/admin/references/check          {"path", "chemical_axis", "kind"} → whether it can be
                                            added, read now (a preview) or after a restart (steps);
                                            of odors as datasets write them, "odor_check" (as an
                                            upload's)
POST   /api/admin/references                {"path", "title", "description", "source", "kind",
                                            "chemical_axis", "value_label", "row_label", "col_label"}
                                            → added (its file to the folders it may read, if need be)
PATCH  /api/admin/references/{id}           any of those adding it takes but "path" (e.g. "fields":
                                            a LOTUS bundle's settings) → changed, its file read
                                            again by them
DELETE /api/admin/references/{id}           its copy deleted (never its file) → {"revoked": the path
                                            taken off the folders it may read, if any; "moved": where
                                            MetaCyc or LOTUS imported for it was moved, if it was
                                            the last of it}
POST   /api/admin/models                    {"path" (on tensor, from its top), "hidden", "first", "last"
                                            (comma-separated parameters), "skip" (comma-separated
                                            patterns of spike count files that are not models, by
                                            their paths in it)} → the folder added, with what it has (400 if it
                                            has no params.csv; 409 if added already)
PATCH  /api/admin/models/{id}               {"hidden", "first", "last", "skip"}: its lists changed
DELETE /api/admin/models/{id}               forgotten (nothing on tensor touched)
POST   /api/admin/models/baselines          {"experiment" (as /api/models keys it: "<folder id>/<its
                                            folder>"), "row"} → that model chosen as a baseline (404 if
                                            no such model; 409 if it is one already), its files imported
POST   /api/admin/models/baselines/{id}/import  its files imported again from tensor, in place of its
                                            copy (404 if its model is not there any more)
DELETE /api/admin/models/baselines/{id}     a model no longer a baseline (its copy moved to data/deleted/)
GET    /api/admin/metacyc                   MetaCyc imported here: what an upload is to have (each
                                            role's files and what it is, and how they are found:
                                            "how"), the import under way or the last ({"running",
                                            "stage", "label", "log", "error", "added"}), and those kept
                                            (folder, size, their reference datasets)
POST   /api/admin/metacyc                   multipart 'file' (repeatable: the files of the raw folder,
                                            or archives of it), 'mtimes' (JSON: each file's
                                            lastModified, by name), 'version' (else told from them)
                                            → 422 {"errors": what is missing or not as expected, or a
                                            version not theirs}, 409 while one is under way, else
                                            {"job"} (its "label" and "note": how the version was
                                            told): built and added in the background
GET    /api/admin/lotus                     LOTUS imported here, as /api/admin/metacyc
POST   /api/admin/lotus                     multipart 'file' (repeatable: the files of LOTUS/raw,
                                            gzipped or not, named by their paths in it; or archives
                                            of it) → as /api/admin/metacyc
POST   /api/admin/{metacyc|lotus}/tensor/check  {"path"} (from the top of tensor) → {ok, error, path
                                            (normalized), kind: "folder" | "archive", size, n, files:
                                            those of the folder the build reads}
POST   /api/admin/{metacyc|lotus}/tensor    {"path", "version"} → 422 {"errors"} if the path is not
                                            one, 409 while one is under way, else {"job", "source"}:
                                            copied, checked, built and added in the background
```

Matrix payloads carry `values_b64`: little-endian float32, row-major
`(n_units, n_trials)`; `NaN` marks missing cells. A reference dataset's of a matrix most
of whose cells are 0 (MetaCyc's compounds by pathways) carries `values_sparse` instead:
`index_b64` (little-endian uint32: the flat, row-major indices of its other cells) and
`values_b64` (theirs, as above).
