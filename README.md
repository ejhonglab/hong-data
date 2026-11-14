# Odor response database

A small intranet web app for a fly olfaction lab, to collect matrices of mean neural
responses (per fly and ROI) to panels of odors. Each upload is checked and described
with its experiment details, then everything can be browsed, clustered, exported and
downloaded as scrollable heatmaps, next to fixed reference data (chemicals × fruits,
chemicals × pathways, ...) and the lab's table of odor metadata.

## Using it

Open `http://hal:8080/` in a browser on the lab network (type the `http://`, and
bookmark it). If `hal` is not found, use `http://10.0.0.5:8080/` instead (for now, until
every computer's DNS has `hal`). There is no login. The browser calls the site "Not secure": that is
expected, and nothing secret is entered. If a download or the page is blocked, see
[Browser warnings](#browser-warnings) below.

1. **Settings** (once): your first name (alone, with no spaces), and your raw data,
   motion-corrected data and odor YAML folders on tensor, written from the top of tensor
   (e.g. `tom/raw_data`). Every field of a path on tensor lists what is in the folder
   typed so far as you type: Up/Down and Enter (or a click) take one, Tab completes. Once all of them are found, they are saved under your first name
   for any computer you use, and an upload under that name finds them.
2. **Upload**: drop your CSV or parquet file of responses (see
   [Data file format](#data-file-format)), or several with the same experiment details
   (e.g. one per fly), whose ROIs become one dataset. It is checked and shown as a heatmap
   before anything is saved; if problems are listed, fix them all and drop the file again. Fill
   in the experiment details (your first name, indicator, driver and coding unit are
   required). Most are filled in for you from your settings, the file and your data on
   tensor, each colored by where it came from. If some of its odors are not in the odor
   table yet, the form lists them, with rows to paste into the odor sheet and a button
   to sync from it; you can leave and come back to the Upload tab, which keeps your
   upload for two days. Generate the README, edit it if you like, and save.
3. **Browse**: datasets by panel (diagnostics panels last, in gray), by the date of
   their last fly, by uploader or by upload date, of every coding unit or just one. Click one for its heatmap, README, original files, odors, and where its
   raw and motion-corrected data are on tensor.
4. **Download**: from a dataset's page, or tick several in the list: the original files
   (a zip), or the response matrices as CSV.
5. **Compare**: tick datasets to see them side by side, averaged, with how alike their
   odors are, and how that compares with chemical structure or reference data.
6. **Merge**: every dataset of a coding unit (e.g. glomeruli) in one download (one
   calculation of the same recordings), optionally with groups of datasets (by
   experimenter, rig, response calculation or power) scaled to each other first.
7. **Models**: models of KCs (their outputs on tensor, added in the Admin tab), each by its
   parameters, in a table sorted by any of them (the baseline models first, in a table of
   their own); open one to see its KCs' spike counts to each odor, or compare them with
   datasets in the Compare tab. *Run a baseline model on ORN data* runs one on your own
   datasets of glomeruli (one run at a time, with how far along it is), to see and save
   its KCs' spike counts, its plots and what is needed to run it again.
8. **Headspace calculator**: search for odors of the odor table to put in a mixture (or
   start from a panel's odors), give each a concentration (log10 v/v, as in your data),
   and calculate how much of each is in the air above it (in ppm, µg/L and other units).
   Below it, how to make each odor from its neat chemical by serial dilutions; then
   search VCF's foods (e.g. `kiwi`) for their volatiles, the headspace over
   them, and the dilutions in paraffin oil that would give the same headspace.
9. **AL segmentation**: a fly's glomerulus ROIs drawn and named by a model
   ([al-diag-seg](https://github.com/tom-f-oconnell/al-diag-seg)) fit on the server, viewed
   beside the fly's own, edited and downloaded; the signals of a fly's ROIs (traces, dF/F and
   responses) extracted from its movies as al_analysis does; one download of what there is on tensor to train
   such a model, of every fly of a glomeruli dataset with a diagnostic panel, and the flies
   left out, with why; how many glomeruli each fly has, and which glomeruli few of the flies
   have. Models fit to them on the server's GPU, scored on flies held out, and downloaded to
   run on your own flies.

The footer says which version of the site the page runs (its commit, and whether it had
changes not yet committed). After the site is updated, a page left open says so once you
come back to it, with a link to reload it.

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

To restart or update the site, any sudoer can run `restart-olfdb` on hal, or use Admin →
*Update and restart everything* on the site. To deploy new code, run `cd /opt/olfdb && git
pull` first. Either way, it lists any jobs it would stop (e.g. a fit running) and asks
first ([DEPLOY.md](DEPLOY.md#7-day-to-day-maintenance)).

## Data file format

The wide layout written by pandas for a frame with MultiIndex rows and columns (an
example of the antennal lobe format, ORN responses by glomerulus, also linked from the
Upload tab:
[`examples/megamat_mean_pebbled_ij_certain-roi_stats.csv`](examples/megamat_mean_pebbled_ij_certain-roi_stats.csv)):

```
date,,,,,2023-11-19 00:00:00,2023-11-19 00:00:00,...
fly_num,,,,,1,1,...
roi,,,,,D,DA2,...
panel,is_pair,odor1,odor2,repeat,
glomeruli_diagnostics,False,3mtp @ -5,solvent,0,-0.0377,0.13,...
```

- **Columns** (three header rows): `date`, `fly_num`, `roi`; every column unique. Every
  ROI's fly (`date` and `fly_num`) is required: a file without them, or without one of
  them, is refused, saying which (as is a CSV with a single header line of ROIs alone).
  Dates may be written e.g. `2024-07-10` (any way pandas reads one) or `20240710`, as
  text or a number, each value on its own.
- **Other column levels** (any number, in any order, e.g. `hemibrain`: `L` or `R`) are
  kept with each ROI: ROIs of the same fly and name that differ in them are different
  ROIs (e.g. cells of each side of one fly's brain, each numbered from 0), told apart
  everywhere: in the viewer (labelled `L 3` under their fly; *Sum over KCs* sums each
  side apart), the Compare tab and downloads (a level of its own, empty for the ROIs of
  datasets without it). A fly's sides uploaded as separate datasets are not taken for
  duplicates of each other. Every ROI needs a value of each; none may be named
  `dataset_id`, `value` or as a row level.
  `roi` is a glomerulus name from `config/glomeruli.txt` or a non-negative integer. A
  file with other ROI names is refused with all of them listed (and the accepted name
  each is but for upper and lower case), along with the accepted names. The level may
  be called `cells` instead (in a long file too): its values are then KC ids, so must be
  non-negative integers, and the upload form takes the coding unit to be KCs.
- **KCs**: every KC of each fly, their ids running from 0 (or 1) to the highest with
  none missing (of each side, if other column levels tell sides apart that are numbered
  each on their own); a file with gaps is refused, saying which ids each fly lacks (KCs
  by a `cells` level when the file is read, else numbered ROIs once the coding unit is
  KCs).
  Those not to be used are marked in one more column level, `bad` (True or False): a
  header row after the ROIs' in a wide CSV (`date, fly_num, cells, bad`), a column in a
  long file. Any ROIs may be marked so. Downloads keep the level (False for the ROIs of
  datasets without it, among others).
- **Rows**: `panel`, at least `odor1` (or `odor`) and `repeat` (or `trial`: each trial's
  number, an integer); optionally `is_pair`, `odor2`, .... A file without trials (one
  row per odor, as when averaged over trials) is refused, as is one without a panel on
  every row. A panel names the set of odors presented in an experiment (e.g. `megamat`);
  which odors it has may vary somewhat from experiment to experiment. Several `odor<n>` levels are mixture components presented together and
  are displayed joined with ` + ` (solvent components dropped), as in `hong2p.olf`. A
  mixture written in one level that way (`ea @ -4.2 + eb @ -3.5`) is split into its
  components, as if each were in a level of its own (with a warning).
- **Odor strings**: `<abbrev> @ <log10_conc>` with single spaces, concentration 0 or
  negative; `solvent` for an unused component. Odors written `<abbrev>(<log10_conc>)`
  (`1-5ol(-3)`) are read as `1-5ol @ -3`, and a solvent alone (`PFO`, `water`) as at 0
  (`pfo @ 0`), with a warning listing them. Odors of diagnostic panels (`diagnostic`,
  `diagnostics` or `glomeruli_diagnostics`, in any case) may be written with a `diag_`
  prefix (`diag_ea @ -4`), removed on upload with a warning saying how many; a file with
  such an odor in any other panel is refused. Concentrations are shown to at most 3
  significant digits, or 2 after a leading `0.` (`ea @ -2.69897` as `ea @ -2.7`,
  `va @ -0.30103` as `va @ -0.3`); downloads keep them as uploaded.
- Row levels that say nothing are dropped on upload: `is_pair` if it is False on every
  row, and `odor<n>` (from `odor2` on) together with every odor level after it, if all
  their values are the same one of: empty, `solvent`, `pfo`, `water`, `pfo @ 0`,
  `water @ 0` (with the 0 written any way: `pfo @ 0.0` too). (Only there are `pfo` and
  `water` accepted without a concentration.) Datasets stored before this are shown and
  downloaded the same way.
- A long layout (`date, fly_num, roi, [panel,] odor1, ..., repeat, value`) is also
  accepted for CSV and parquet (parquet with integer `fly_num` column labels cannot be
  written in the wide layout by pandas, so use long there). Any other columns are other
  levels of the ROIs (as above), but `Unnamed: <n>` ones (an index `to_csv` wrote), which
  are ignored.
- Datasets stored while files without flies or trials were accepted (for a while) are
  still shown: such a dataset says "flies not given", has no fly folders, raw data or
  movies to find, and downloads without `date` or `fly_num` (in the wide layout, with
  one header line).
- A `_t0`, `_t1`, ... suffix is removed from ROI names (`DA1_t0` becomes `DA1`), and an
  ROI of glomeruli that could not be told apart may be named by them joined with `/`
  (`VA1d/VA1v`), kept as one ROI (each said in a warning).

Validation reports every problem it finds at once (bad odor strings with the exact
reason, unknown ROI names, duplicate columns, non-numeric cells, ...).

## The tabs in detail

- **Upload**: several files dropped (or chosen) together are one dataset: each is
  checked on its own (problems listed per file), then their ROIs are put together, in
  order of file name, each with its values for the trials of any of them (trials being
  the same by panel, odor and repeat; empty for those its file lacks, with a warning).
  They must have the same column levels and other row levels, and no ROI may be in two of
  them; every file is kept as uploaded. A file is refused if it was uploaded already, by
  anyone: the same file (whatever its name, alone or among others), or the same data in
  another (the same flies, ROIs, trials and values). Once your first name and the coding unit are entered (when the README is made, and
  again when saving), it is refused too if its flies (each `date` and `fly_num`) are
  exactly those of a dataset of yours of that coding unit and the same manipulations,
  unless it is the same recordings calculated otherwise, each saying how (its *Response
  calculation*, below: e.g. z-scored, and in units of each cell's baseline noise, as two
  datasets; as many as there are), the error saying which needs one if either has none;
  one sharing only some flies with one is accepted with a warning saying which of the two
  may need to be deleted if they are the same recordings. Each names that dataset and who
  uploaded it: to replace one, delete it first (Admin tab). Other uploaders' datasets,
  other coding units', or other manipulations', may have the same flies (one person's
  fly 1 of a day is not another's, one fly may be recorded in two places, and before and
  after a drug).
  *Manipulations* (optional): anything done to the flies that sets the dataset apart,
  each a name and a value (`drug` = `CNO`) or a name alone (`starved`), shown in brackets
  after the dataset's name (`[drug=CNO; starved]`: in Browse and Compare, its title and
  README, and the names of its downloads). Names and values seen before are suggested;
  they are matched in any case and order.
  *Response calculation* (optional): how the responses were calculated, shown as the
  label of their colorbar (in Browse, Compare and exported figures), written as a
  matplotlib label is: text, with LaTeX math between `$` signs (`$\Delta F/F$`,
  `$\frac{\Delta F}{F_0}$ (z-scored)`; `\$` for a dollar sign), or all math if it has a
  TeX command and no `$` (`\Delta F/F`). It is drawn under the field as you type, with
  anything MathJax cannot read, which must be fixed to save. Ones used before are
  suggested. It is also written after the dataset's name, as entered (`APL16: MB247dbLexA >
  GCaMP6s (KCs) [TNT], baseline-noise units`: in Browse and Compare, its title and README),
  and in the names of its downloads and permalink (in ASCII, `$\Delta F/F$` as `Delta-FF`),
  so that the same recordings calculated otherwise are told apart.
  An optional odor YAML must have every odor in the matrix, with the same
  abbreviation and concentration. Each YAML added is also checked on its own, as a
  warning only: its odors against the odor table (missing, or named, abbreviated or with
  a CID unlike there), odors without a concentration, and an abbreviation used for two
  odors. A YAML of the same name replaces the one attached, if it can be read. Next to
  the button, it says which YAMLs are attached, or why none was found on tensor. Other
  optional details: ThorImage `Experiment.xml`, power, age, wavelength, rig, data paths,
  notes. The raw and motion-corrected data paths are folders from the top of tensor,
  checked on the server as you leave their field (as in Settings), and must be found
  there to save (unless tensor itself is not on the server, when they are taken as
  entered). Details are filled in from your settings, from the file (panel names;
  glomerulus ROIs mean coding unit "glomeruli") and from tensor. The data paths are your
  raw and motion-corrected data folders in Settings (of several, the one with these
  flies, else the first), or, without any, the folder with the flies found on tensor:
  each fly (`<folder>/<date>/<fly>/`, as in hong2p, or the folder given if it is a fly's
  own) is looked for in the raw data path as entered (else in your raw data folders) and
  your motion-corrected data folders. Only the folders of the dataset's own flies (by
  date and fly number) are used, which give the `Experiment.xml` of the most recent
  functional recording (of the last fly; not those in folders named `anat` or
  `anatomical`, or starting with either then `_`, taken to be anatomical; unless
  you added or chose one), and the odor YAMLs in your odor YAML folders describing
  every odor, panel by panel: for each panel, the YAMLs olf generated for its
  recordings, as named in their ThorImage notes, of the last fly whose describe all its
  odors. Recordings are named `<panel>_<n>` (n from 1, as olf numbers its YAMLs
  `<time>_stimuli_<n-1>.yaml`), or just `<panel>` if it is the panel's only one; one
  ending `_redo` or `_001` replaces the one it is named after; one whose notes name no
  YAML takes the one of its number from its others' run (said so under the panel, as
  are notes naming another number). Else the panel's config (`<panel>.yaml`, anywhere
  in those folders), then YAMLs with a fly's date in their path. What was found for each
  panel is listed (with why), and attached if every odor is described. A field stays
  colored by where its value came from
  (settings, tensor, the files) until you change it, and the README says which files
  came from tensor, and from where. Every value is stored on its own, whatever is
  changed in the README.
  A dataset is saved only once each of its odors is one in the odor table (matched as in
  Browse: through its odor YAMLs, else by abbreviation). Those that are not are listed
  under the odor YAML, with the rows to copy into the odor sheet (as in Browse) and a
  button syncing the odor table from it there: the odors new in the sheet are added, and
  any other change waits for the Odors tab. An upload under way is kept (in the browser,
  and on the server for `OLFDB_STAGE_TTL_HOURS`, 48, after it was last worked on), so the
  Upload tab continues it where it was left when opened again.
- **Settings**: folders are relative to the top of tensor (absolute paths are refused),
  each checked on the server as you leave its field; once all are found ("All Paths
  Accessible") they are saved on the server under your name. A name must be a first name
  alone, here and in the upload form (one with a space in it is refused, though a dataset
  uploaded before keeps its uploader's full name when edited), as an upload finds the
  folders saved under the name it is given. Below that, defaults for
  the upload form: its fields (dropdowns and all) but those of one dataset only (manipulations,
  data paths and files), none required, checked as the form checks them and kept in this
  browser only;
  and whether odor names (abbreviations) are shown in lowercase (in this browser only):
  e.g. `IaA @ -3` as `iaa @ -3`, in Browse, Compare, Reference data and the Odors tab,
  and in every download but of the original files. Names that differ from another
  shown (or downloaded) with them only in case stay as they are, so that no two odors
  become one. The upload form shows a file's odors as they are in it. The page is light
  or dark as your system or browser is set, unless chosen there (in this browser only;
  exported figures are always light).
- **Browse** (and the Compare tab's list): datasets grouped by panel (the panels the
  upload form gave, else the file's; those named with "diag" last, in light gray), by
  the month of their last fly, by uploader (each one's newest upload first) or by the
  month they were uploaded (the last two saying when each was). Outside the grouping by
  panel, each dataset's item lists all its panels, in the same order (e.g.
  `megamat,validation2,glomeruli_diagnostics`). The same recordings calculated more than
  one way (below) are one item: their default calculation, its last line ending with that
  calculation and then the others (`other response calcs.: …`), listed by upload as of
  the first of them uploaded. Each dataset has a thesis-style odor table (with each odor's CID, InChI, vapor pressure and molar mass),
  its odors matched to the Odors tab through its odor YAML when it has one, otherwise by
  abbreviation. Those not in the odor table, or that more than one odor there could be,
  are listed above it and highlighted in it; a button copies rows for the odor sheet for
  those not in it (one per chemical: name, CID, CAS and solvent from the odor YAML, the
  abbreviation as in the data), tab-separated in the order of the sheet's own columns,
  to paste into its first empty row (the sheet and the Odors tab linked there) before
  syncing it. "Download table" saves it as a CSV (every odor with the odor table's
  details and links, as in downloads of several datasets) or as LaTeX (a booktabs
  longtable by panel, linked, without InChI or structures, to `\input`: it needs the
  booktabs, longtable, array and hyperref packages, and fits a landscape page with 2 cm
  margins), named as the dataset's other downloads with `_odors`. The Odors tab and the
  Headspace calculator's results have the same "Download table" menu. "Raw data" and "Motion corrected / traces" show where those are on
  tensor (to copy, as browsers cannot open folders on your computer) and what is in
  them. They are red, saying why, when no path was entered, or the server cannot find
  it (from the top of tensor, or from a folder set in the Admin tab) or may not read it;
  orange when one has recordings the other does not (by fly and recording folder name,
  not counting anatomical ones: e.g. recordings not motion corrected yet). A recording
  redone, `<recording>_redo` (in either), is in place of `<recording>`, which is left out
  of both, in their menus, the movie player and the comparison. Only the
  dataset's own flies' folders (`<date>/<fly>`) in them count: those of other flies are
  left out (and named); movies in no fly's folder (a recording's folder given as the
  path, say) are of the fly the folders in their path name, else by folder. Recordings in
  only one are listed in its menu, and apart in the movie player. "Show data mismatches"
  (orange, to their left, then) lists every recording of both side by side, by fly in
  the order they were recorded (as their Experiment.xml says), the ones in only one
  highlighted; with every recording in both, a green "All data in both" is there instead.
  The path can be entered or corrected in their menus, checked as at upload (the README
  is left as it is). Its README can be edited there too (Edit, in the README box): saving asks
  whether to overwrite the README on the server (its path is shown) or discard the
  edits, and refuses if someone else changed it meanwhile. Unsaved edits are kept while
  you look at other datasets. *Edit* (next to *README*) changes any detail entered at
  upload but its files, in the upload form's own fields: checked as the upload form checks them (data paths only if
  changed, an upload's refusals too, e.g. another of your datasets of that coding unit
  with the same flies and manipulations, and the same response calculation or none), saved once you answer *Yes* to whether you are
  sure (*No* is the default). Its title and panels follow, and its README too unless it
  was edited by hand; `metadata.json` keeps each edit (when, and each field before and
  after). Right of its title (under it, in a narrow window) is how its responses were
  calculated (*Response calculation*). For the same recordings calculated more than one
  way (datasets of the same flies, experimenter, coding unit and manipulations, each with
  its own response calculation), it is a menu of their calculations, to show another, with
  *Set Default Response Calc.* to its right: the default (at first, the first uploaded) is
  the one listed here and in Compare, and merged unless another is chosen in Merge. Where
  other recordings are calculated the same ways (e.g. each of Yang's in whole-recording z,
  baseline-noise units and unnormalized dFc), it asks whether to make that calculation
  their default too, saying what each defaults to now. Compare's list has the same menu
  and button under each such entry.
- **Permalinks**: under a dataset's title, *Make a permalink…* names a link to it that
  stays whatever its number, `http://hal:8080/d/<name>` (lowercase letters, digits
  and `-`; by default as its downloads are named, e.g. `megamat-tom-glomeruli-2023-06-22`),
  with *Copy* and *Rename…*. An edit of its details that would change that default name
  (its uploader, coding unit, manipulations, response calculation, or panel names if its
  file has none) asks
  first, saying what the permalink would be renamed to, unless it was renamed by hand.
  A permalink keeps what its dataset is: its uploader, indicator, driver, coding unit,
  manipulations, response calculation and panels (but diagnostic ones), and its flies and
  trials. While it points to a dataset, an upload of the same uploader, indicator, driver,
  coding unit, manipulations, response calculation and panels is refused, whatever its
  flies, odors and trials: one dataset is kept of each (to replace one, delete it first in
  the Admin tab). Deleting a dataset
  orphans its permalink (the Admin tab warns first): it then leads to what it was of,
  until an upload like that dataset takes it over, at once if it is the same, else once
  *Proceed* is pressed under a summary of what differs (its flies, odors or trials).
- **Movies** ("Play movies…" in the Raw data or Motion corrected menu of a dataset whose
  folder is found): a player for the movies of the dataset's flies in that menu's data
  folder (raw, or motion corrected).
  Motion-corrected ones are TIFFs, as al_analysis writes them: one per fly of all its
  recordings registered together (`mocorr_concat.tif`, listed as the fly,
  `<date>/<fly>`, with the recordings in it), else one per recording (`mocorr.tif`,
  else `flipped.tif` or `raw.tif`); in a fly's folder with neither, every TIFF in it.
  They are drawn with the ImageJ ROIs of the `RoiSet.zip` next to them, or in a folder
  above (for a fly's movie, the one its recordings share), each on the Z plane it was
  drawn on. *ROIs of other planes*: those of the planes above (red) and below (blue)
  dashed, the default; none; or all of them, dashed and paler. *glomeruli only* (ticked
  at first) lists and draws only ROIs named as glomeruli: as an upload accepts them
  (`config/glomeruli.txt`, or several joined by `/`) or as in the dataset's data, so not
  numbered ones or those ending in `+` or `?`; how many it hides is next to it. ROIs
  named as glomeruli but for a `_t0`, `_t1`… suffix (tentative) have their own box, when
  there are any. Raw ones are ThorImage `.raw` files (not its
  `RoiMask.raw`), laid out by the `Experiment.xml` next to them (flyback frames left
  out). The odor of each trial is written over a movie (grey before the odor comes on,
  white with a red dot after), from al_analysis' `trial_frames_and_odors.json` next to
  it (for a raw movie, its recording's in the motion-corrected data), unless *odor
  names* is unticked. Play / pause, time and Z sliders (the wheel over a slider steps
  it, ten with Shift; a movie of one time point, such as an average over time, has no
  play button, time slider or speed), max projection over Z, channel, speed, zoom (wheel; out no
  further than the movie fits the window) and move (drag; never past its edges), with
  the value of the pixel under the pointer.
  Brightness and contrast are set as in ImageJ's B&C window: the plane's histogram with
  its black and white levels, Minimum, Maximum, Brightness and Contrast sliders, *Auto*
  (as ImageJ's: few pixels saturated, more contrast at each press) and *Reset* (the
  plane's full range). Keys as in ImageJ where they exist (space or `\` to play, `<` /
  `>` and the arrows to step, `A` auto contrast), `R` to reset the contrast and `O` for
  odor names. Only the planes shown are read, on the server, so even large movies open
  at once.
- **Download**: the matrices of several datasets are concatenated into one wide CSV
  (trials × ROIs, with `dataset_id` as an extra column level; trials matched on panel,
  odor components and repeat) or long CSV (one row per value). Zips always have
  `datasets.csv` (every dataset's details), `odors.csv` (their odors, with metadata and
  links) and the READMEs. The matrix of a single dataset is just one CSV instead, laid
  out as uploaded (less the row levels dropped on upload) and named
  `<panels>_<uploader>_<coding unit>_<last fly date>.csv`: the panels in the data,
  sorted, without diagnostic ones (as above; unless there are only those), and the
  uploader's name lowercased without periods, e.g. `megamat_tom_glomeruli_2023-11-19.csv`
  (the long one ends in `_long.csv`).
- **Compare**: datasets ticked on the left (listed as in Browse) side by side, each its
  odors (rows, by panel) by its glomeruli or ROIs, every trial of an odor averaged, and
  for glomeruli every fly too (a column per glomerulus, the mean of the flies with it;
  ROIs grouped by fly and named as in Browse, in its order);
  in the viewer of reference data (colors, cell size, transpose, moving and removing
  rows and columns, figure export), without trials, ROI ordering or clustering. Under an
  entry of the same recordings calculated more than one way are Browse's menu of their
  calculations (choosing one lists it instead, and if one of them is ticked, ticks and
  shows it instead) and *Set Default Response Calc.*, and *Compare All Response Calcs.*,
  which shows every one of them side by side in place of everything else (that entry
  ticked alone, until it is unticked or anything else is ticked): each its ROIs by its
  odors as Browse shows them (of glomeruli too, fly by fly), under its number and
  calculation, the recordings' name over them all; over the ROIs and odors all of them
  have, with rows and columns moved or removed, cell sizes, transposing and scrolling the
  same in all. On the right:
  - *Compare on* odors (the default: those of the odor table, alone, at the
    concentration chosen), a panel's odors, or glomeruli. On a panel's odors, datasets
    are compared with each other only, on the panel chosen of those of the datasets
    shown (by its name in the file, else the one in the upload form, but for case; by
    default the one most of them have): each over the panel's odors as presented,
    concentrations and mixtures as they are, an odor of the odor table alone matched by
    it and its concentration, anything else by how the data names it (but for case).
    Everything below (similarities, their correlations and p-values, those side by
    side) is then of those odors. On glomeruli, everything below is of
    glomeruli instead of odors: only datasets of glomeruli are compared (as glomeruli
    by odors, every fly averaged), to reference datasets of glomeruli (found by their
    labels: see [reference_data/README.md](reference_data/README.md)), by
    glomerulus–glomerulus similarity (of their responses across a dataset's odors, or
    their values across a reference dataset's other axis), and side by side on the
    glomeruli all of them have; no odor–odor similarity is shown, nor chemical structure.
  - *Show* each space as its data (by default) or its odor–odor similarity: one menu
    for the datasets, the reference data and those side by side (chemical structure
    only ever as similarity). A similarity matrix's columns are always its rows, in
    their order (a dataset's grouped by panel): sorting, moving or removing one moves or
    removes both.
  - *Odor–odor similarity* within datasets (Pearson or Spearman correlation, or cosine
    similarity, of two odors' responses across the glomeruli or ROIs both have), and
    within reference data (Pearson correlation or cosine similarity, of their values
    along its other axis; of one with chemicals not in the Odors table too, only along
    what any of those in it is not 0 or missing in, e.g. the pathways they are in).
  - *Compare to*: chemical structure, as the Jaccard similarity (1 − Jaccard distance)
    of the odors' ECFP fingerprints (RDKit's Morgan fingerprints from their InChI, ECFP6
    by default, of any even diameter up to 12), and every reference dataset with
    chemicals. Odors are matched across them by the Odors table (one chemical alone, not
    mixtures; an odor at several concentrations as its highest, lowest or the mean of
    them; a reference dataset's chemicals that are one odor, e.g. its stereoisomers, made
    one as it was added to say: averaged, summed or the highest); each says how many of
    the odors shown it has (and, in yellow, how many of them it has only as the same but
    for stereochemistry), and its data or similarities over those are shown.
  - *Side by side*: two or more of the spaces shown (chosen there, *Add another* for
    more; or by clicking a cell of the correlations between similarities, for its two),
    over the odors all of them have, in one order (the first dataset's, if one is), next
    to each other with each row level with the others': moving or removing a row, cell
    sizes, transposing, chemical labels and scrolling stay the same in all (and of
    columns too, where they have the same: similarities, or the same ROIs), and hovering
    an odor in one marks it in the others.
  - *Between similarities*: a matrix of the Spearman (or Pearson) correlation of each two
    of the datasets and spaces compared to: of their similarities for each pair of the
    odors both have (one side of each matrix, without its diagonal), each cell saying r
    and of how many odors. Datasets are named there by what was recorded: their coding
    unit (kc or kcs as KCs, orn or orns as ORNs, pn or pns as PNs, in any case), or for
    glomeruli ORNs (a pebbled driver), PNs (GH146) or glomeruli; by their full names (as
    in its CSV) when two would share one. To its right, the p-value of each correlation
    against 0: two-sided, by permuting one space's odors (a Mantel test, the default and
    the only one of the three that is exact; 5000 permutations by default) or by
    bootstrap (with replacement, of the pairs of odors or of the odors), Holm–Bonferroni
    adjusted for every two spaces shown, each cell saying the adjusted and raw p (the
    options panel says why, with links, under *How the p-values are found, and why*). Like every matrix of the tab, they can be
    exported as a figure (SVG, PDF, PNG, with the numbers in the cells when they are large
    enough) or a table (CSV: for these, each two spaces once with r, odors, pairs and both
    p-values).
- **Merge**: the matrices of every dataset of the chosen coding unit (matched in any
  case), concatenated as above, in a zip that also has `scaling.csv` (each dataset's group
  and scale factor). Of the same recordings calculated more than one way (datasets of the
  same flies, experimenter and manipulations, each with its own response calculation),
  one is merged: that of the calculation chosen (*Of the same recordings calculated more
  than one way, merge*, shown when there are some), else their default (as set in Browse,
  else the first uploaded); the others
  are listed as left out, in the tab and the zip's README. Datasets can be left at their own scale, or grouped by any of
  experimenter, recording system, response calculation and power at the fly, each group then multiplied by one
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
  structures (drawn as vector graphics by RDKit, from the InChI PubChem gives for each
  CID), vapor pressures (in kPa: the mean of the experimental values on the odor's
  PubChem page at or near 25 °C, else of those with no temperature given, looked up
  through [chemutils](https://github.com/ejhonglab/chemutils) once an odor with a CID is
  added; blank where PubChem has none) and molar masses (g/mol, PubChem's, likewise).
  It is synced one way from a Google sheet (its link set in
  the Admin tab) or a CSV imported in the Admin tab, with the columns of
  `config/odor_sheet_template.csv` (downloadable there), which starts it off with
  the odors from the thesis tables: the tab shows what a sync would do, adds new odors,
  and applies changes to existing ones and deletions only where ticked, warning about
  renames that would unlink datasets. A badge on the tab shows when the sheet changed
  since its last sync. An odor's *aliases* (in one cell, separated by `;`) are its other
  abbreviations and names alike: data, odor YAMLs and reference data naming it by any of
  them are matched to it, e.g. `IaA` with aliases `IAA; isopentyl acetate`. The sheet can
  also have a second tab named `aliases` that lists more of them, one per row, under the
  headers `abbrev` (the odor: its abbreviation, else its name) and `alias`; other columns
  there (e.g. where an alias is from) are ignored. They are added to those of the odors'
  own rows, and nothing is synced while an alias there is of an odor not in the sheet,
  is given twice (or is already that odor's name, abbreviation or alias), or is the
  name, abbreviation or alias of another odor: the preview says which rows. "Download
  table" saves the odors shown (all, or those the filter matches), in the tab's order, as
  a CSV (every field of each odor, its links and the datasets using it) or LaTeX (a
  longtable of names, IDs, supplier and catalog number, vapor pressure and molar mass,
  linked as on the site).
- **Viewer** (datasets and the upload preview): odor labels stay at the
  top and fly/ROI labels at the left while scrolling, with hover tooltips (in every
  matrix shown: a click on a cell pins its tooltip, to read or screenshot it, until its ×,
  a click anywhere else, or Escape closes it). Average over
  repeats or show individual trials; rows by fly→ROI, ROI→fly or ungrouped; odors as in
  the file or by name. The default colormap is blue–white–red, with lightness linear in
  |value| and white at 0, over the full data range (of other values, viridis; of
  distances, such as MetaCyc's LCA distance, plasma reversed: the nearer, the brighter; of
  models' spike counts, never negative, its red half, white to red, on a log scale).
  *Transpose* shows odors as rows and ROIs as columns (in reference data too, rows as
  columns). Every matrix shown (here, in reference data and in Compare) has *Full screen*
  (Esc, or the button again, to leave; the browser's window filled where full screen is
  not available) and *Hide controls*, folding away everything above the matrix but one
  line: what is shown (its size and range), its colorbar, and the buttons (remembered for
  the matrices shown next). The colorbar is labelled with what its values are: of a
  dataset, its *Response calculation*; of reference data, its *Values are* (Admin tab);
  of similarities, their metric.
  - *Moving and removing rows and columns* (also in reference data): drag a label
    to move it within its group (fly, panel, ...), or a group's name to move the whole
    group; with individual trials, an odor's trials move with its label (the repeat
    numbers are not handles). Labels of clustered ROIs or odors stay put. Ctrl-click a
    label to remove what it labels at once (an ROI, an odor with its trials, or a whole
    fly or panel by its name); right-click it for a menu that asks first, and can also
    remove a glomerulus from every fly. Choosing another order in the toolbar replaces the dragged one. The orange
    *Reset rows & columns* brings back everything removed, in the toolbar's order.
    Edits are kept for each dataset while you look at others.
  - *Sum over KCs* (KC data): each fly's KCs (those not removed or hidden) summed into
    one row per odor or trial, missing values left out; ROIs are then neither clustered
    nor reordered.
  - *Hide bad ROIs* (KC data only): KCs marked bad are hidden from the matrix, its
    clustering and exports, by default whenever a dataset has any; untick to show them.
    In Compare they are also left out of everything computed (similarities, their
    correlations), and in Merge of the means datasets are scaled by.
  - *Clustering*: ROIs and/or odors ordered by hierarchical clustering (average
    linkage, correlation or Euclidean distance on trial-averaged responses, missing
    values skipped pairwise), with dendrograms. ROIs are always clustered all
    together, whichever fly they come from, and shown ungrouped in the clustered
    order (up to 30,000 ROIs, e.g. a KC dataset's: 30,000 need ~1.8 GB in the browser,
    and about a minute). Odors are clustered
    within each panel, by their responses across all ROIs. Only what is not removed is
    clustered, and the result is the same whatever order it was shown in. A stored
    dataset's clusterings are kept on the server (its 24 most recently used, by what was
    clustered: the metric, and the ROIs and odors removed or hidden), so the next to
    cluster it the same way, in any browser and after restarts, gets them at once. A
    model run's spike counts (Models tab) are clustered the same way by their KCs and/or
    odors: the KCs all together (shown ungrouped, each with its type after its label),
    not while they are summed; whether to is kept apart from datasets'.
  - *Export figure*: SVG, PDF or PNG of the whole matrix as shown (order, colors,
    cell size, dendrograms) with a colorbar; cells are an embedded image, labels and
    lines stay vector in SVG/PDF.
- **Reference data**: tables added in the Admin tab (below), and any in
  `reference_data/` listed in its `manifest.json` (format in
  [reference_data/README.md](reference_data/README.md)); its two synthetic examples
  only until there is any other. The list ends with *ECFP*, not a table: how to compare
  data with chemical structure (ECFP6 fingerprints) in the Compare tab, with a link there
  with it ticked; under the list, a link to the Admin tab to add reference datasets.
  *Download* (by its title) saves one as its matrix, a wide CSV of its rows and columns as
  shown (those not removed, in the order shown, transposed if so; of KCs summed over, each
  of them), each row's details in its first columns and each column's in its header lines,
  a level each: each field chosen for them (e.g. of VCF, a product and a reference each a
  level), else its labels (headed as its first column is), groups, identifiers and what is
  said of each; of chemicals, the Odors table's name, abbreviation, CID and concentration
  too. Read it with pandas as `pd.read_csv(f, header=[0, …], index_col=[0, …])`, one for
  each level. Or as its original files: a zip of the file as added (of a MetaCyc or LOTUS
  import, the files uploaded), with a README of how it is read.
  Their chemicals are labelled by
  their names in the Odors tab where they can be matched (or abbreviations, or as in the
  file: *Chemicals* in the viewer), the rest as in the file; those written as datasets
  write odors keep their concentration (`2-heptanone @ -3`). Odors so written that are
  not in the Odors table are listed above its matrix as an upload's are, with their rows
  for the odor sheet and a button to sync it. A yellow warning is by a
  table with chemicals not identified by a CID or InChI (in the file, or through the
  Odors tab), naming the first of them, with all of them to show and copy (also when
  adding one in the Admin tab); these are left out of its matrix unless a box by the
  warning is ticked. Another is by one with chemicals matched to the Odors tab only as
  the same but for stereochemistry (e.g. VCF's dl-menthone as menthone, whose
  stereocenters the Odors tab leaves unspecified): how many of the odors it has (each
  once) are, which, and how many of its values (not 0 or missing) are theirs. A third,
  by the matrix, lists the odors that may be miscounted, unresolved, as the Headspace
  calculator's VCF search does (below). Of a table with chemicals not in the Odors table too, a box under
  its notes shows only those that are (and of its other axis, only what any of them is
  not 0 or missing in, e.g. the pathways they are in). One that cannot be read is listed
  by its name alone, with why.
  A table of glomeruli (by their labels) says it can be compared with datasets on their
  glomeruli. Unless its other axis is odors (as datasets write them, e.g. ORN responses),
  it has no chemicals, and so no warning, and its other axis is taken to be KCs:
  *Sum over KCs* in the viewer (here and in the Compare tab) shows each glomerulus's
  values summed over them (those not removed; missing values left out), as one row or
  column.
  *MetaCyc* (a bundle of its tables, imported in the Admin tab or made by a script, see
  [Scripting](#scripting)) is shown as either of two datasets, its compounds matched to
  the odor table's chemicals (made again whenever the odor table changes): *compounds by
  the pathways they are in* (every compound of MetaCyc in a pathway, the odor table's
  first; 1 in the pathway itself, 0.5 only through a class it is of, such as "an
  alcohol", if it is in none itself), or the *pathway LCA distance* between the odor
  table's chemicals (as in chapter 2 of O'Connell's
  thesis: the fewest steps from a common ancestor down to both, in MetaCyc's network of
  compounds and reactions; odor × odor, compared in the Compare tab by its own values).
  Below its matrix: click a cell for its pathway drawn (its main compounds top to bottom,
  reactions as arrows with their EC numbers and equations on hover, the odor table's
  chemicals in color, the one clicked outlined), its classes, superpathways, organisms
  (with the environments, oxygen and hosts of BioCyc's genomes of each) and summary; or,
  of the LCA distance, the paths from the two odors' lowest common ancestors. Click a
  compound for its structure, identifiers, classes and pathways, or find any pathway or
  compound by name or MetaCyc ID.
  *LOTUS* (a bundle of its records of natural products: which papers report which
  structure in which organism, imported in the Admin tab or made by a script, see
  [Scripting](#scripting)) is shown over the odor table's chemicals that are in it (made
  again whenever the odor table changes), then, by default, its other chemicals of at
  most 250 Da, odorant-sized (22k of its 2023 release; or 200 or 300 Da, or none), by the
  taxa of the organisms they are reported in: by default, *chemicals by the families of
  organisms they are reported in*, each cell how many of the family's organisms (as
  LOTUS names them) the chemical is reported in. As for MetaCyc, a box above the matrix
  shows only the odor table's chemicals. Above its matrix, *How this matrix was
  made* says each choice behind it, and which can be changed in the Admin tab: the
  columns (organisms as LOTUS names them, or their species, genera, families, orders,
  classes, phyla or kingdoms), the cells (organisms, papers, or presence), the
  stereochemistry (by default, LOTUS's structures that are the odor's but for R/S:
  enantiomers, diastereomers and unspecified stereocenters merged, cis/trans isomers kept
  apart; or exactly the odor's; or any of its skeleton), the taxonomy used first (GBIF's
  backbone by default, then the Catalogue of Life, the Open Tree of Life, NCBI and the rest
  in turn, where those before give an organism none) and the other chemicals (up to what
  mass). Below it, click a cell for the
  organisms it is of, each with its lineage and the papers reporting the chemical there
  (year, first author, title linked to its DOI, journal; those checked by hand by LOTUS
  marked), and the structures of LOTUS its row is (with links to LOTUS and Wikidata).
  *Sort rows* and *columns* order each by its values over those of the other axis shown
  (by mean or largest value, highest first, those with none last, within its group;
  remove columns to sort by some of them, e.g. one product of VCF's). Sequential colors
  are on a log scale when the values are none negative and span 3 or more orders of
  magnitude (*Scale*: auto, saying which it chose, or linear or log, 0 then the lowest color; kept for each matrix), as VCF's do, and
  LOTUS's counts of organisms or papers always are; counts with some 0, such as LOTUS's,
  are on a log scale of 1 + each, so 0 is the lowest color and 1 apart from it.
- **Models**: every model of the folders of model data added in the Admin tab (read where
  they are on tensor each time the tab is opened, so models added to them since are listed:
  there is nothing to import again after changing those files), a row each: an
  experiment's `params.csv` row, with the keys of the `provenance.json` beside it as
  parameters too (nested ones by their path, as `base->n_spikes_for_response`; where both
  have one, the row's; those under one of its models' tag, variant or base, in a dict
  keyed mostly by them, of that model alone, as `runs->*->seconds`), first its experiment
  (click it for its `provenance.json`, once) and when they were generated (its
  `generated`). Shown are the parameters that are `params.csv` columns (and the
  `provenance.json` keys of those names, an experiment's for all its models); tick *Also
  provenance.json's other keys* for the rest (what made them, notes, each model's
  settings: hundreds). The baseline models (chosen in the Admin tab) come first, in a table
  of their own with the `params.csv` parameters they have, loaded before the rest: as
  imported into the site when chosen, with a warning saying how their files on tensor
  differ from that, if they do.
  *Run a baseline model on ORN data*, under them, runs one on datasets of glomeruli: choose
  the baseline, the datasets (several are taken as one: the mean of theirs) and which of
  their panels' odors the model is run on (all but diagnostic panels, unless chosen). Each
  glomerulus's mean response to each odor (over the flies; odors of one component, and
  glomeruli measured for every one of them) is scaled into a change in spike rate by one
  factor: the one that best matches (least squares, through zero) the responses, to every
  panel's odors, that they have in common with [Hallem & Carlson's
  (2006)](#reference/hallem-carlson-2006), a built-in reference dataset of all they
  tabulate (110 odors at 10⁻², 10 of them down to 10⁻⁸, and fruits), per odor and
  glomerulus, their dilution taken as one log₁₀ unit above the concentration here (their
  10⁻² as 10⁻³: as pyolfsysm takes them; else the nearest less than 1 away). With fewer
  than 5 odors (50 odor–glomerulus pairs) in common, the datasets are first scaled to the
  combination of all datasets of their coding unit, made as the Merge tab merges them
  (each of the others a group, scaled to the average of what more than one has), and
  that combination to Hallem & Carlson's; what was done is said before it is run, and
  kept with it. The model is run by pyolfsysm's `fit_and_plot_mb_model`, with the
  baseline's settings (its `params.csv` row's `base_kwargs_json`, and its other columns
  that are settings of it), either *as the baseline was tuned* (its own threshold and
  APL weights: the same model, on this input; the default) or *tuned again on these
  odors* (to the response rate it was tuned to). A baseline with anything
  `fit_and_plot_mb_model` does not take (another experiment's own mechanism: a gain on
  some glomeruli, a breadth exponent) cannot be run, and is listed saying which. One
  model is run at a time (others wait their turn, in the order asked for), with a bar of
  how far along it is (by how long the last runs took) and what it last said; it can be
  stopped. A run done is listed with its response rate and factor: *Show* opens its KCs'
  spike counts to each odor below, as any model's (to filter, cluster, compare with
  datasets, or save as shown: *Download › Spike counts, wide CSV*); *Spike counts (CSV)*
  saves them as the model wrote them (KCs as rows, an odor a column); *Everything (zip)*
  saves all `fit_and_plot_mb_model` wrote (`output/model/`: its plots, spike counts,
  responses, weights, parameters and `model_provenance.json`), `inputs/` (the mean
  responses, those scaled, every response compared with Hallem & Carlson's and their
  table, the baseline's parameters, the arguments of the call), `run.py` (the script
  that ran it: with olfsysm installed at the commit given, `python run.py` runs it
  again from those files), `provenance.json`, a README and its log.
  Click a
  parameter to sort by it (again: the other way), type in *Find models* to list only those
  with it in a value shown (or a run's name), and tick *Hide parameters the same for every
  model that has them* to see what differs. Only the rows and columns in view are drawn, so
  that hundreds of models by hundreds of parameters scroll as quickly as a few (and the
  browser's own find sees only those). Each row's buttons open its runs, one per
  condition (`on`, `acute_off`, …): its `*_spike_counts.parquet` files, there or in a folder
  under it, found by its `tag` or `variant` in their names (models of two bases with one
  tag told apart by the base their names begin with). A run is shown below the table, its
  KCs (grouped by type) by its odors (labelled as in the Odors tab), on a log scale (of 1 +
  each count), KCs that spike to no odor left out unless unticked, and with *Only KCs with
  at least n spikes to some odor* those below the model's `n_spikes_for_response` too.
  Above it, its experiment's `provenance.json` (click to open), and the model's own
  `provenance_<tag>.json` beside it, if it has one (a fit's). *Compare* opens the Compare
  tab with it at the top of the list on the left (its KCs as shown), to tick datasets to
  compare with: there it is shown as a dataset is (odors by KCs, or its odor–odor
  similarity), correlated with the others' similarities, and side by side, on odors only.
  *Download* saves its spike counts as shown (as a reference dataset's matrix is, above:
  each KC's levels as in its file, and each odor's, with the Odors table's name,
  abbreviation, CID and concentration), or its original files: its spike count file, its
  experiment's `params.csv` and `provenance.json`, and a README of its model's parameters.
- **Headspace calculator**: odors are added by any part of a name, abbreviation, alias,
  CID or CAS number, or a mixture starts from a panel of the datasets: its odors in the
  odor table with a CID, but solvents, each at the highest concentration presented (what
  was left out is said). The solvent (paraffin oil or water) follows what the odor table
  says of the odors, unless one is chosen. The headspace is at equilibrium in a closed
  vial at 25 °C: over paraffin oil by Henry's law with constants estimated as chemutils
  does (`inchi2k_henry_pfo`, from activity coefficients by modified UNIFAC), or by Raoult's
  law only (an ideal solution) if chosen, either way for the oil's density and molar mass,
  which can be given (by default 0.85 g/mL and 450 g/mol, those of a viscous oil, and
  also used for the serial dilutions and VCF's foods below); over water by Henry's law. What it
  needs of each odor (molar mass, density, vapor pressure, Henry's law constant) is
  looked up through chemutils the first time (a few seconds an odor) and kept with the
  odor table; where something is missing, what was assumed instead is noted by the odor
  (a vapor pressure estimated from the structure, a density of 1 g/mL). Vapor pressures
  and Henry's law constants are labelled *measured* or *estimate* (where chemutils says:
  every constant in oil is an estimate, with the vapor pressure it was estimated with
  shown, labelled too). Point at a headspace (or click it, to keep it open) for how it
  was calculated: each step, from its concentration and every value used, to the value
  shown, saying what each symbol is, and its value, where it is first used. What was calculated since the page was opened is shown again at
  once, for another solvent, model or oil too. "Download table" saves the results as a CSV
  (a row per odor, then one of them all together: what was used of each and of the oil, whether
  measured, where from, and its headspace in every unit) or LaTeX (as shown, in the
  units chosen, linked, measured values and estimates marked). The mixture, solvent,
  model, oil and units are kept in this browser. How it is computed, and what it leaves out:
  [design.md](design.md#headspace-calculator).
  - *Serial dilutions*: for each odor of the mixture with a concentration, the µL of its
    neat chemical into the first of its vials, how many dilutions of how many µL of each
    vial into the next, and each vial's concentration, in the mixture's solvent: each vial
    gets the same amount of it (2 mL by default; paraffin oil weighed, at the density
    given with the mixture; water by volume); then how many vials (aliquots of solvent) they take in all,
    and how much solvent.
    The neat chemical's volume is kept from 2 to 200 µL (as typed), and the dilutions are
    of one volume for every odor where they can be, with the fewest dilutions, then as
    large volumes as can be. The dilutions' least (100 µL by default, so as not to pipette
    very little paraffin oil; a smaller one can take fewer dilutions) and most (optional:
    a vial's solvent) are kept to only where every odor can still be made within the
    neat chemical's limits. What is typed is kept in this browser. How they are chosen:
    [design.md](design.md#serial-dilutions).
  - *Foods' volatiles (VCF)*: needs a reference dataset of VCF's CSV (Admin tab: one value
    per line, e.g. rows `name` with `inchi`, columns `product_name`, values `mid_ppm`)
    whose file also has `k_henry` and `vapor_pressure` columns (with their `_units` and
    `_sources`). Typed text is matched against every product's name, category and group
    (suggested as you type). Shown: the products matched, every matching line, each
    chemical's quantity in the samples (mg/kg: each reference's lines on a product
    averaged, then the median of a product's references, then of the products; or the
    mean, geometric mean, maximum or minimum; its chemicals that are one odor of the
    Odors tab one, as the Compare tab has them: in each sample merged as the dataset
    says, e.g. summed, named by the odor with theirs under it), and the headspace over the samples taken
    as water, by Henry's law, else by Raoult's law with its vapor pressure, saying which.
    Both are looked up again through chemutils in the background (those the file lacks
    first, then those it has, to check them; kept apart from the odor table's, and the
    tables fill in as they come): what it finds is used, else the file's (*file*), else
    estimates (*estimate*: a Henry's law constant from solubility over vapor pressure, a
    vapor pressure from the structure). Where what it finds differs from the file's by
    more than 10%, a warning lists those chemicals, and the file's value is shown by each
    (⚠). Warnings above the tables (and ⚠ by each, said in its CSV too) list what may
    be miscounted, unresolved: an odor given in a sample both as itself and as a
    stereoisomer of it (maybe its total, then summed twice; or VCF's isomenthone, whose
    InChI gives no stereochemistry, beside dl-menthone), a chemical of no structure given
    that may be an odor by its name (`pentanol (unkn.str.)`, 1-pentanol: not counted as
    it), one that could be two odors (neither); and then the odors here only as the same
    but for stereochemistry. *Sort the tables by* quantity (the default), headspace or name. With up to 8
    products, each product's values are columns too. If
    chosen, also the v/v of each in paraffin oil, all together, that gives the same
    headspace (by Henry's law with constants estimated as chemutils does, or Raoult's law only);
    what that needs of each chemical is looked up through chemutils in the background
    the first time (a few seconds a chemical, at most 500 chemicals a search), and the
    table fills in as it comes. *Make them the mixture* puts those that are odors of the
    odor table (at most 50) in the mixture above, at those v/v. "Download tables" saves
    a CSV per chemical (each product's values, the headspace in every unit, the oil) or
    of the matching lines (with where each constant and vapor pressure is from, and the
    file's).
- **AL segmentation**: *Predict glomerulus ROIs*, at the top, draws a fly's glomerulus ROIs
  with a fit's model (by default the last fit done; each says how many flies it was fit
  to): a fly of the site's datasets (saying which the model was trained on or held out),
  or any on tensor, by its motion-corrected and raw data folders. The fit worker makes them,
  on the GPU, or on the CPU while a fit is using the GPU (slower, but not hours later), and
  keeps them on the server (`data/predictions/<id>/`): its ROIs as a `RoiSet.zip` named by
  glomerulus, each kind al-diag-seg makes, the fly's average image, and `provenance.json`
  (the model and its fit, the fly's files read with their sizes and times, the software,
  who asked). Asked for again, of the same model and fly, it asks whether to view those or
  make them again, replacing them. *View the ROIs* opens them in the movie player, on the
  fly's average image (as the model saw it) and its motion-corrected movies: only the ROIs
  of the plane shown, those predicted and the fly's own (if it has them) in two colours,
  each shown or not. Where the fly has ROIs of its own, the predicted ones are scored
  against them. Every prediction is listed below, to view, download or delete. One waiting
  or running says when it should end, and the box how long one asked for now should take:
  the median of the last ten done like it (on the GPU or the CPU; with an anatomical stack
  to register or not, ROIs of its own to score against or not), a guess, said to be one,
  until one has finished. How long each took, and each of its stages, is kept with it.
  The predicted ROIs can be edited in that viewer, as a copy of them (a prediction has
  one): *Simplify* makes it with fewer points, each outline moving no more than the pixels
  in the box beside it from the predicted one (1 by default, which leaves about a tenth of
  the points; Ramer–Douglas–Peucker, so the points kept are the predicted outline's own),
  always from the ROIs as predicted, so simplifying again at another number starts over;
  *Edit as predicted* copies them with every point. The copy is drawn in a colour of its
  own, with its own box to show it, ticked, and the predicted ROIs' box is unticked. Click
  an edited ROI (or its name in the list) to choose it; its points can then be dragged,
  and, as in ImageJ, Shift-clicking a point adds one beside it (the point becomes two) and
  Alt-clicking one deletes it (Ctrl-click does too, as some desktops keep Alt-click for
  moving windows). *Undo* and *Redo* (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y) step through the
  changes made since the viewer was opened, simplifying and *Discard* (which deletes the
  copy, so that the predicted ROIs are shown again) among them. Each change is saved at
  once, as `<fly>_pred-edited_RoiSet.zip` beside the prediction's other files (with
  `<fly>_pred-edited.json`: when, by whom, how simplified), so it is in its download; if
  someone else changed them meanwhile, it says so instead of overwriting theirs. The ROIs
  as predicted, and the fly's own, are only ever shown: nothing changes them. Predicting a
  fly's ROIs again deletes its edited ones with the rest (it says so when asking).
  Clicking a predicted ROI's name, in the viewer's list or where it is drawn on the image, opens a table of what that name
  cost, which stays until closed (its ×, Escape, or a click outside it): a column for each
  term of the cost (odor tuning, position, shape, atlas overlap, layout; hover over one
  for what its value is), their weights, then a row for each of the 5 glomeruli that cost
  least for the ROI, with each term's value (0: a perfect fit) and the cost, their
  weighted mean; the one the ROI was named is marked, and a glomerulus that names another
  ROI says so. An ROI is named only where that cost is no more than the most a name may
  cost (0.6 by default, set when the model is fit), else numbered. The values are of the
  ROI as predicted, a 3D object, so a copy's name shows them only while its outlines are
  all as predicted; once simplified or edited, it says why it cannot (computing them
  again needs the fly's responses and the atlas registered to it, as predicting does).
  Predictions made before these were kept (`<fly>_identity_terms*.json`, in the download)
  say to predict again.
  *Extract signals*, below it, computes the signals of a fly's ROIs from its
  motion-corrected movies as al_analysis does, with the same numbers for its response
  window (checked against what al_analysis left for 9 recordings of 4 flies: identical to
  the last digit): for each
  recording of the fly (a folder with `mocorr.tif`, else `flipped.tif` or `raw.tif`, and
  `trial_frames_and_odors.json`; one done again, `<name>_redo`, in place of the first), each
  ROI's trace (the mean of the movie over its pixels on a plane), its dF/F in each trial
  ((F − F0) / F0, F0 its mean over the trial's time points before the odor), and its
  response to each trial's odor: the dF/F of largest magnitude, with its sign, in the
  response window, the first volumes from the odor's onset. The window is as many volumes
  as the recording's odor pulse can reach (from the pulse's length in the stimulus YAML its
  notes name and a volume's in its `Experiment.xml`: 3 for a 3 s pulse at 0.67 volumes/s,
  2 at 0.44), and never fewer than the 2 al_analysis always takes; the box says how many
  that is in each of the fly's recordings, and how it was worked out when hovered over (2,
  saying why, where the pulse's or a volume's length is not found). Enter a number of
  volumes to use that in every recording instead: 2 for al_analysis' own numbers. Choose the fly (one of the site's, one whose ROIs
  were predicted, or any on tensor by its folder) and its ROIs: those a model predicted,
  as edited in the viewer or as predicted, or the fly's own (the `RoiSet.zip` of each
  recording, as al_analysis reads them; ROIs named `AL` or ending in `+` are left out, as
  there). An ROI drawn on several planes (one name, several outlines) gives one signal,
  in one of four ways: its best plane in each recording, the one whose largest response
  over that recording's trials is greatest (al_analysis' way, the default); its best plane
  over all the fly's recordings, the same in each; the mean of all its voxels; or its
  planes' traces, each relative to its mean, averaged with weights 1 / the variance of its
  dF/F before the odors (the less noisy planes counting more). The fit worker runs them,
  one at a time, beside fits and predictions (a few seconds per GB of movies; those waiting
  and running say when they should end, from how long the last ones took), and keeps each
  on the server (`data/extractions/<id>/`) to download as a zip: `responses.csv` (trials ×
  ROIs, every recording's trials, with al_analysis' panel and odor names),
  `responses_glomeruli.csv` (the same of the ROIs named as glomeruli alone, which the
  Upload tab takes as it is), and per recording `traces.csv`, `plane_traces.csv` (every
  plane of every ROI), `dff.csv`, `responses.csv` (as al_analysis' own table of it) and
  `planes.csv` (each plane's pixels, largest response, noise, weight, and whether it is
  al_analysis' best), with `provenance.json` (the files read, how it was computed, the
  software, who asked), a `README.txt` and its log. The numbers are al_analysis', but not
  all that is around them: a few odor abbreviations differ (it also uses a cache of its own
  computer's), a `_redo` recording replaces the first whole (al_analysis blanks only the
  odors given again), every recording with a movie and its trials is taken, and nothing
  al_analysis does after its per-recording table is done (ROIs merged across recordings,
  uncertain ROIs dropped, concentrations that are not a panel's final ones blanked, means).
  These, and what is questionable in the calculation itself with by how much it matters,
  are in [design.md](design.md#signals-extracted-as-al_analysis-does).
  *Download training data*: of every glomeruli dataset with a diagnostic panel, each fly whose
  folder in the dataset's motion-corrected data (as al_analysis lays it out) has
  diagnostic recordings (folders with `trialmean_dff.tif`, the mean dF/F of each odor, and
  `trial_frames_and_odors.json`, more than half of whose odors are of the dataset's
  diagnostic panels, through the odor table), a `RoiSet.zip`, and its average image
  (`mocorr_concat_avg.tif`, else a `mocorr_concat.tif` to average), and whose raw data has
  the `Experiment.xml` of a diagnostic recording. The download is a zip, sent as it is made
  (several GB: the browser saves it as it comes), of those files where they are on tensor
  (from its top), with the average image computed on the server where al_analysis has not
  written it (kept there, so only once), and unless unticked the flies' anatomical stacks:
  the `anat` folders of their raw data, higher resolution stacks without odors, which some
  flies have (as recorded, so not mirrored as al_analysis mirrors right ALs' functional
  data), through which al-diag-seg registers flies to each other. Its `roi_training.json` lists each fly's
  files, each recording's odors as named in the odor table (alike in every fly), the
  glomerulus names uploads accept (to tell glomeruli from other ROIs, as the movie player
  does), and the flies left out. A fly in several datasets is in it once.
  The page also counts each fly's glomeruli, as al-diag-seg trains on them (ROIs named as
  a glomerulus, or several joined by `/`, and tentative ones, `DA1_t0`, as that glomerulus;
  not numbered ROIs, nor names ending in `+` or `?`): their range and mean at the top, each
  fly's in the *Flies* table, and a histogram of them (one per uploader, side by side, if
  ticked). Under it, the glomeruli infrequently identified (by default in at most half the
  flies): those labelled by one uploader only, then the others, each with how many flies
  (of each uploader and dataset) have it, the datasets without it, and the panels (but
  diagnostics) of the flies with it; then those only in flies with a panel not every fly
  has. The box below it sets the cutoff (a number or percent of the flies, `≤` or
  `<`), and whether all glomeruli are counted at the top, in the table and in the
  histogram, or only those identified more often, or only the infrequent ones.
  *Fit a model* fits [al-diag-seg](https://github.com/tom-f-oconnell/al-diag-seg)'s
  model to those flies, read where they are on tensor, on the server's GPU (Cellpose
  fine-tuned on their ROIs: every outline of one glomerulus, numbered and uncertain ones
  too, but not `+` ROIs or outlines of the whole AL; an atlas of where each glomerulus is,
  and each one's odor tuning), holding out a few flies (from each dataset in turn) to score
  its predicted ROIs (Cellpose's own, named) against theirs; with how many to hold out, Cellpose's epochs (15 by default), whether to build the atlas
  (and add the [Grabe et al. 2016](https://doi.org/10.1016/j.celrep.2016.08.063) atlas to
  it, as by default), whether to train on tentative ROIs, and (under *How its ROIs are made and named*) Cellpose's mask thresholds, how much
  to grow its ROIs, and how much each plane's must overlap and be tuned alike to be joined into 3D objects (the
  defaults scored best on fit 6's flies), and the most a name may cost (0.6: an ROI no glomerulus costs
  less for is numbered, not named). Anyone may start one, once their first name is set in
  Settings (the box links to it). One runs at a time; those running and waiting say who
  asked for them, from which IP address, when, how far they have got, and when they should
  end (estimated from how long earlier fits took, stage by stage, which is kept with each);
  under the options, how long a fit asked for now with them should take. *Past fits*, above it,
  has every fit's scores, plane by plane (each glomerulus's outline on a plane, as ROIs
  are drawn) and as 3D objects (each held-out fly's, and the glomeruli not found, when
  clicked; nothing counts against an ROI where no glomerulus is drawn, as glomeruli are often drawn
  on their clearest planes only), its log, and its download: its model, with how to run it on your own flies
  (`alseg-predict`, which writes their ROIs as RoiSets named by glomerulus), and its
  held-out flies' predicted ROIs. Anyone may delete one (it asks first). The bottom of
  *Fit a model* says what such a download has, and how to install al-diag-seg and run a
  model on a fly (the versions the fit worker has filled in).
- **Admin** (to the right of Docs; it first asks whether you are Tom, Jon or Betty, or
  have asked one of them): settings for the whole site.
  - *Folders on the server* the app may read (by default tensor) and also write (none
    by default: for backups), besides its own data folder. Saved changes take
    effect when the server restarts, which a button in the message saving them, or the
    *Restart* section near the bottom, does when it runs under Docker or systemd (there
    too, *Update and restart everything*: the site rebuilt from the code on the server and
    restarted, fit worker and all, as `restart-olfdb` does there, showing its output as it
    goes; either first lists the jobs it would stop, with when they should end, and asks). How
    they are enforced:
    [DEPLOY.md](DEPLOY.md#folders-it-may-read-and-write). Below them, a tree of each
    file the server has read or written in the folder it runs from (e.g. `/opt/olfdb`,
    as a path on the server even in Docker), with sizes and modification times, those
    deleted since struck through (or hidden, with the box above it): its database, uploads
    and the rest of `data/`, but not the repository's own files, nor those in the folders
    above or of reference data.
  - *Backups*: a button copying everything the site keeps into one of the folders it may
    write: the SQLite database (a consistent copy), and every dataset's files as uploaded,
    odor YAMLs, `Experiment.xml`, README and `metadata.json`, with reference data as read
    in, the baseline models' files as imported, AL segmentation fits (each ~1.2 GB: its
    model, held-out flies' results and log), and datasets deleted there (not raw or
    motion-corrected data, nor uploads under way, nor other model data, which is read on
    tensor, nor AL segmentation predictions, which a fit's model makes again, but for their ROIs as edited by hand in
    the viewer, which are kept; nor signals extracted, made again likewise). Each backup is a folder `olfdb-backup-<date>_<time>`
    laid out like `data/`; files unchanged since the backup before it there are hard
    links to it, so each takes the space of what changed only. Listed with when each was
    made, its size and what is new in it; above them, when the last was made and whether
    it is current (else what changed since: the database, or files new, changed or
    removed, and under *What changed*, which of the database's tables, with their rows
    then and now, and which files). The fit worker's heartbeat, which it writes to the
    database every 10 s, is not a change.
  - *Reference data*: add a CSV, TSV or parquet file on the server, or an xarray
    DataArray saved as netCDF (`.nc`), by its path from the top of tensor (completed as you
    type it, as other paths on tensor are, its folders and the files it can read listed) or
    its absolute path (completed only under tensor's), checked as you type it (when the
    server may read it: its size, and how many of its chemicals match the Odors tab), with a title,
    description, source, labels, colors (or from its values) and whether its rows or
    columns are chemicals (or found from identifier columns, e.g. `cid`, `pubchem_cid`,
    `inchi`, `cas`, `name`, or rows labelled so; of a DataArray, coordinates named so).
    Once read, its fields are listed, to choose which make the matrix (the rest are
    ignored): a table laid out as its matrix, or one value per line (a long table:
    suggested when the labels of its rows repeat, each on several lines; VCF's CSV, as
    the vcf repo makes it, is recognized, and its own fields suggested: rows `name` with
    `inchi` and `cid`, columns `product_name` with `reference`, `group` and `category`,
    values `mid_ppm` in ppm, its chemicals of one odor summed), whose rows and columns are each
    combination of the fields chosen for them (labelled by the first, the others told
    apart and shown on hover), and whose values are a field of numbers (the mean of
    several lines of a cell) or presence (1 where a line lists a row and column); of a
    DataArray, which coordinates label and describe its rows and columns; and how the
    Compare tab makes one of its chemicals that are one odor (averaged, summed, e.g.
    amounts of an odor's stereoisomers in a sample, or the highest). Each choice
    shows what it makes (its size, chemicals identified) before it is added. A MetaCyc
    bundle (`.sqlite`) is added once for each view: compounds by pathways, or the
    pathway LCA distance (through reactions, as computed for the thesis, or through
    pathways, as its Methods describe, optionally without compounds in more than so many
    pathways), and whether chemicals in no pathway are taken as the classes they are of.
    A LOTUS bundle is added with its settings (columns, cells, stereochemistry, taxonomy
    first: each value said, the defaults marked, what each makes shown before it is
    added), and these are changed later by its *Change settings…* in the list (the form
    filled in with them, then *Save changes*): its matrix is made again by them. A file's
    fields are changed so too (e.g. its value field, or how its chemicals of one odor are
    merged): a field that identifies chemicals (`cid`, ...), is of its rows or columns, or
    has no numbers is not offered or not taken; one most of whose lines are not numbers is
    warned of (VCF's `quantity`, whose ranges `mid_ppm` has numbers for), as is a change
    of values under a colorbar label typed for the old ones; of VCF's file, the check says
    whether (and with what values) the Headspace calculator searches it.
    The app never changes the file: it reads a copy into `data/references/`, again
    whenever the file changes. A file outside the folders the server may read is added
    to them (itself, to read only), and read once the server restarts, with what that
    takes in Docker or systemd. Chemicals are matched to the Odors table by CID, else
    InChI, else InChI but for stereochemistry (stereocenters or cis/trans one of the two
    leaves unspecified, the rest alike; warned of, above), else CAS, MetaCyc or LOTUS ID,
    name or abbreviation, never otherwise against a CID or InChI given. An axis whose labels are mostly odors as datasets write them
    (`<abbreviation> @ <log10 conc>`, e.g. `2h @ -3`; at least 3) is taken to be its
    chemicals (unless *Chemicals are* says otherwise), matched by abbreviation as a
    dataset's odors are; those not in the Odors table are listed as an upload's are, with
    their rows to copy into the odor sheet and a button to sync it, here and by the dataset
    in the Reference data tab. Deleting one deletes the copy (never the file), and takes its file off
    the folders the server may read if it was added there for it; *Delete all…*, under
    the list, deletes every one added here so, after one question. When one cannot be
    read, it says why, with a red dot on the Admin tab.
    *Import MetaCyc*, below it, does what the script does ([Scripting](#scripting)) from
    the folder of MetaCyc's SmartTables (as `BioCyc/raw`) or a `.zip`/`.tar.gz` of it:
    choosing the folder lists the files the script reads (what each is, and which are
    missing), and only those are uploaded. *or a path on tensor…* takes either from
    tensor instead (its path from the top of tensor, checked as you type as data folders
    are, e.g. `tom/BioCyc/raw`): the server reads it there, so nothing is uploaded, and
    copies the files the script reads (or takes them out of the archive) before
    building. The server checks them first (each there, with the columns it needs) and
    refuses them saying what is missing or not as expected; else it keeps them in
    `data/metacyc/<when>/raw/` (with the times they were last changed), builds the
    bundle there (about 20 s, its progress shown) and adds its two reference datasets.
    Its version, unless given, is told from the files (SmartTables do not say it): by
    how many pathways (not counting super-pathways) and compounds they have, against how
    many each release's notes say it has (e.g. `BioCyc/raw`'s, of January 2024, are
    27.1's: more than 27.1 had, fewer than 28.0). How it was told is shown, and kept; a
    version given that is another release's is refused. Imports are listed with their
    folders and datasets; the folder is moved to `data/deleted/metacyc/` with the last
    of its datasets deleted. Backups include them.
    *Import LOTUS* does the same for LOTUS, from the folder `LOTUS/raw` (or a
    `.zip`/`.tar.gz` of it, or either on tensor): its files are found as the script
    finds them, by their paths there (the release's dates are in its folders' names),
    and kept gzipped, laid out as they were, in `data/lotus/<when>/raw/`. Where the
    folder has a file both as is and gzipped, the gzipped one is uploaded; the browser
    gzips the others first, so 330 MB of the folder's 7.3 GB is sent. A file the browser
    cannot read as it was chosen (changed since, or on a network mount that will not
    give it, as one over sshfs may not) is named, with how much of it was read: copy the
    folder to a disk of the computer the browser is on and choose it there (or, if it is
    on tensor, give its path there). The bundle takes about a minute to build; its
    reference dataset (chemicals by families) is then added, its settings changed after by
    *Change settings…*.
  - *Model data*: folders of model outputs on tensor for the Models tab (*Choose a folder
    on tensor*: its path from the top of tensor, completed as it is typed), read there,
    never copied. Each `params.csv` in it, at any depth, is an experiment (a subfolder with
    a `provenance.json` and no `params.csv` is of other models: its spike counts are
    listed as of no model). Three comma-separated lists of parameters: those not shown
    (`n_kcs,n_glomeruli` by default), those first, in order (`base`), and those last
    (`tag,variant,rate_on,rate_off,source_table`); the rest come between, those in more of
    the files first. And patterns of spike count files known not to be models, by their
    path in the folder (`*` any text, across folders too), e.g. `*/partB/*` (copies), so
    that those of no model are only counted, not listed as missing from a `params.csv`;
    a file of a model is its run whatever the patterns. *Change…* changes these lists after; *Delete…* forgets a folder
    (nothing on tensor is touched). *Baseline models*: models of them to show first in the
    Models tab (*Find a model* narrows the list to choose from by its experiment, tag or
    base), each kept by its experiment's folder, tag (or variant) and base, so it is found
    again after rows are added before it or its folder is added again. Its files (its spike
    counts, its experiment's `params.csv` and `provenance.json`, its own
    `provenance_<tag>.json`) are imported into the site when it is chosen, so backups have
    them, and it is shown as imported. Each time it is shown or opened, its files on tensor
    are compared with that (its row of `params.csv`, the keys of `provenance.json`, the
    other files by SHA-256, runs gone or added), and any difference is said, there and in
    the Models tab. *Import again…* takes them as they are on tensor now; *Remove* takes it
    off (it is still in the table of every model), its copy moved to `data/deleted/`.
  - *Delete datasets*: every dataset, newest upload first (with a filter), each with a
    Delete button that asks first. A deleted dataset is gone from the site at once; its
    files (as uploaded, with `metadata.json` holding every detail entered) are moved to
    `data/deleted/<id>/` on the server, from where it could be uploaded again. Below them,
    the orphaned permalinks (of datasets deleted), each with what it was of, and a Delete
    button.
  - *Odor sheet*: the Google sheet's link, used as soon as it is saved, and importing a
    CSV instead (with the template to start from). Below them, whether new odor names
    may only be lowercase: then uploads whose odors (as named in the data) have capital
    letters are refused, as are odor sheet rows whose name, abbreviation or aliases do,
    unless that name is already in the database (in the odor table, or a dataset). The
    sheet also warns of abbreviations of different odors that differ only in case, and
    the upload form of odors of a file that do.
  - *Panel names*: panels to show under another name (e.g. `glomeruli_diagnostics` as
    `diagnostics`), for every dataset, uploaded already or later, wherever panels are
    shown or downloaded (lists, viewers, Compare, Merge, the Headspace calculator,
    matrices, odor tables, titles), at once. Datasets keep them as uploaded (so do their
    original files and READMEs), so a rename can be undone; the upload form shows a
    file's panels as they are in it (its odor YAMLs and recordings are found by them).
    Refused: a panel renamed to one itself renamed, and two panels of one dataset renamed
    to one name. Below, every panel of the datasets, how many have it and how it is shown.
  - *Odor properties from chemutils*: the chemutils installed, and for each property
    looked up through it (vapor pressure, molar mass, and what the Headspace calculator
    needs), how many odors have it, and how many were looked up with a chemutils older
    than the last change to how it is looked up (from the server's first version with
    this list: those then in the database count as older). Those are ticked, to look
    them up again (in the background, about a second each, with a progress bar; not
    from chemutils' cache, each kept as it comes) after a newer chemutils is installed
    ([DEPLOY.md](DEPLOY.md#updating-chemutils)).
  - *This server*: its name, and its IP addresses (in Docker, those the network's DNS
    gives its name).

## Scripting

MetaCyc: its SmartTables (as exported from BioCyc, e.g. `BioCyc/raw`, MetaCyc 27.1 by
default) are read into one bundle, whose two reference datasets are then added through
the Admin tab's API (or add the bundle in the Admin tab by hand; or upload the folder in
its *Import MetaCyc*, which does all this on the server):

```bash
python -m app.metacyc_build /mnt/d0/Scraped-Databases-Copy/BioCyc/raw /local/olfdb/metacyc-27.1.sqlite --register http://localhost:8000
```

It takes about 20 s. The server must be able to read the bundle: one outside the folders
it may read is added to them, and read after a restart (the script prints what that
takes). `--file <role>=<path>` reads another export for a table (`--help` lists them). Files
missing or without the columns it needs are said, with what was expected, before
anything is read. Which MetaCyc they are of is told from them (as *Import MetaCyc* tells
it), unless `--version` is given (refused if another release's).
Bundles hold MetaCyc's data: keep them out of this repository, and check MetaCyc's
license terms before sharing one.

LOTUS: its frozen export of its Wikidata records (`230106_frozen/230106_frozen.csv`), the
metadata of their structures, organisms and papers (Zenodo record 7534077,
`metadata_2022-09-16/`) and its 2021 release's SDF (for LOTUS IDs), as laid out in
`LOTUS/raw`, are read into one bundle, whose reference dataset (chemicals by families) is
then added through the Admin tab's API (or add the bundle in the Admin tab by hand; or
upload the folder in its *Import LOTUS*, which does all this on the server):

```bash
python -m app.lotus_build /mnt/d0/Scraped-Databases-Copy/LOTUS/raw /local/olfdb/lotus-2023-01-06.sqlite --register http://localhost:8000
```

It takes about a minute, and the bundle is about 280 MB. Each file may be gzipped or not
(of one there both ways, the one as is is read). As for MetaCyc, the server must be able
to read it, `--file <role>=<path>` reads another file for a role (`--help`), files missing
or without the columns it needs are said before anything is read, and bundles are kept
out of this repository. (`LOTUS/raw/NPOC2021`, the MongoDB dump of the
2021 release that `LOTUS/processed` was made from, is not read.)

Downloads: `/api/download?ids=1,2&format=files|wide|long` (as from the Browse tab).
Datasets: `/api/datasets`, and each one's matrix at `/api/datasets/{id}/matrix`; the
odor table, with each odor's InChI, vapor pressure and molar mass: `/api/odors`. Every endpoint is listed in
[design.md](design.md#http-api). On the server, `manage.py` lists, shows, exports and
deletes datasets ([DEPLOY.md](DEPLOY.md), section 7).

## Development

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements-dev.txt               # needs git (chemutils and hong2p are installed from GitHub)
uvicorn app.main:app --reload --port 8080      # data goes to ./data
```

pip does not replace chemutils (or hong2p) when its pin in `requirements.txt` changes
(the version number stays the same): `pip install --force-reinstall --no-deps "$(grep
'^chemutils' requirements.txt)"` (or `'^hong2p'`), then `pip install -r requirements.txt`
again for any new dependencies of it. Updating the pin takes a few more steps:
[DEPLOY.md](DEPLOY.md#updating-chemutils).

Settings and upload lookups need tensor mounted (at `/local/tensor`, or wherever
`OLFDB_TENSOR_ROOT` points), and syncing the odor table from a sheet needs its link (in
the Admin tab); the rest of the configuration is in
[DEPLOY.md](DEPLOY.md#configuration-reference).

[design.md](design.md) describes how the app is put together and why: the main design
decisions, the Compare tab's statistical testing (and the simulations behind it), the
code layout, storage and the HTTP API.

Tests:

```bash
python -m pytest tests/
```

The simulations behind the Compare tab's p-values
([design.md](design.md#statistical-testing)) are a script of their own, run with gjs
(about ten minutes; `--quick` for a short check that it runs):

```bash
gjs -m tests/simulate_pvalues.mjs
```
