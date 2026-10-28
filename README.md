# Odor response database

A small intranet web app for a fly olfaction lab, to collect matrices of mean neural
responses (per fly and ROI) to panels of odors. Each upload is checked and described
with its experiment details, then everything can be browsed, clustered, exported and
downloaded as scrollable heatmaps, next to fixed reference data (chemicals × fruits,
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
   [Data file format](#data-file-format)), or several with the same experiment details
   (e.g. one per fly), whose ROIs become one dataset. It is checked and shown as a heatmap
   before anything is saved; if problems are listed, fix them all and drop the file again. Fill
   in the experiment details (your name, indicator, driver and coding unit are
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
5. **Merge**: every dataset of a coding unit (e.g. glomeruli) in one download, optionally
   with groups of datasets (by experimenter, rig or power) scaled to each other first.
6. **Compare**: tick datasets to see them side by side, averaged, with how alike their
   odors are, and how that compares with chemical structure or reference data.
7. **Headspace calculator**: search for odors of the odor table to put in a mixture (or
   start from a panel's odors), give each a concentration (log10 v/v, as in your data),
   and calculate how much of each is in the air above it (in ppm, µg/L and other units).
   Below it, how to make each odor from its neat chemical by serial dilutions; then
   search VCF's foods (e.g. `kiwi`) for their volatiles, the headspace over
   them, and the dilutions in paraffin oil that would give the same headspace.

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
  (`pfo @ 0`), with a warning listing them. Concentrations are shown to at most 3
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
  another (the same flies, ROIs, trials and values). Once your name and the coding unit are entered (when the README is made, and
  again when saving), it is refused too if its flies (each `date` and `fly_num`) are
  exactly those of a dataset of yours of that coding unit and the same manipulations;
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
  Accessible") they are saved on the server under your name. Below that, defaults for
  the upload form, checked as the form checks its fields and kept in this browser only;
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
  month they were uploaded (the last two saying when each was). Each dataset has a
  thesis-style odor table (with each odor's CID, InChI, vapor pressure and molar mass),
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
  not counting anatomical ones: e.g. recordings not motion corrected yet). Only the
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
  you look at other datasets.
- **Movies** ("Play movies…" in the Raw data or Motion corrected menu of a dataset whose
  folder is found): a player for the movies of the dataset's flies in that menu's data
  folder (raw, or motion corrected).
  Motion-corrected ones are TIFFs (as al_analysis writes them: `mocorr.tif`, else
  `flipped.tif` or `raw.tif`, though any TIFF is listed), drawn with the ImageJ ROIs of
  the `RoiSet.zip` next to them, or in a folder above (each on the Z plane it was drawn
  on; ROIs of other planes dashed). Raw ones are ThorImage `.raw` files (not its
  `RoiMask.raw`), laid out by the `Experiment.xml` next to them (flyback frames left out). Play / pause, time and Z
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
- **Compare**: datasets ticked on the left (listed as in Browse) side by side, each its
  odors (rows, by panel) by its glomeruli or ROIs, every trial of an odor averaged, and
  for glomeruli every fly too (a column per glomerulus, the mean of the flies with it);
  in the viewer of reference data (colors, cell size, transpose, moving and removing
  rows and columns, figure export), without trials, ROI ordering or clustering. On the
  right:
  - *Compare on* odors (the default: those of the odor table, alone, at the
    concentration chosen), a panel's odors, or glomeruli. On a panel's odors, datasets
    are compared with each other only, on the panel chosen of those of the datasets
    shown (by its name in the file, else the one in the upload form, but for case; by
    default the one most of them have): each over the panel's odors as presented,
    concentrations and mixtures as they are, an odor of the odor table alone matched by
    it and its concentration, anything else by how the data names it (but for case).
    Everything below (similarities, their correlations and p-values, the two side by
    side) is then of those odors. On glomeruli, everything below is of
    glomeruli instead of odors: only datasets of glomeruli are compared (as glomeruli
    by odors, every fly averaged), to reference datasets of glomeruli (found by their
    labels: see [reference_data/README.md](reference_data/README.md)), by
    glomerulus–glomerulus similarity (of their responses across a dataset's odors, or
    their values across a reference dataset's other axis), and side by side on the
    glomeruli both have; no odor–odor similarity is shown, nor chemical structure.
  - *Show* each space as its data (by default) or its odor–odor similarity: one menu
    for the datasets, the reference data and the two side by side (chemical structure
    only ever as similarity).
  - *Odor–odor similarity* within datasets (Pearson or Spearman correlation, or cosine
    similarity, of two odors' responses across the glomeruli or ROIs both have), and
    within reference data (Pearson correlation or cosine similarity, of their values
    along its other axis).
  - *Compare to*: chemical structure, as the Jaccard similarity (1 − Jaccard distance)
    of the odors' ECFP fingerprints (RDKit's Morgan fingerprints from their InChI, ECFP6
    by default, of any even diameter up to 12), and every reference dataset with
    chemicals. Odors are matched across them by the Odors table (one chemical alone, not
    mixtures; an odor at several concentrations as its highest, lowest or the mean of
    them); each says how many of the odors shown it has, and its data or similarities
    over those are shown.
  - *Side by side*: two of the spaces shown (chosen there, or by clicking a cell of the
    correlations between similarities), over the odors both have, in one order (the
    dataset's, if one is), next to each other with each row level with its other: moving
    or removing a row, cell sizes, transposing, chemical labels and scrolling stay the
    same in both, and hovering an odor in one marks it in the other.
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
    options panel says why, with links). Like every matrix of the tab, they can be
    exported as a figure (SVG, PDF, PNG, with the numbers in the cells when they are large
    enough) or a table (CSV: for these, each two spaces once with r, odors, pairs and both
    p-values).
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
  them are matched to it, e.g. `IaA` with aliases `IAA; isopentyl acetate`. "Download
  table" saves the odors shown (all, or those the filter matches), in the tab's order, as
  a CSV (every field of each odor, its links and the datasets using it) or LaTeX (a
  longtable of names, IDs, supplier and catalog number, vapor pressure and molar mass,
  linked as on the site).
- **Viewer** (datasets and the upload preview): odor labels stay at the
  top and fly/ROI labels at the left while scrolling, with hover tooltips. Average over
  repeats or show individual trials; rows by fly→ROI, ROI→fly or ungrouped; odors as in
  the file or by name. The default colormap is blue–white–red, with lightness linear in
  |value| and white at 0, over the full data range. *Transpose* shows odors as rows
  and ROIs as columns (in reference data too, rows as columns). Every matrix shown (here,
  in reference data and in Compare) has *Full screen* (Esc, or the button again, to
  leave; the browser's window filled where full screen is not available) and *Hide
  controls*, folding away everything above the matrix but one line: what is shown (its
  size and range), its colorbar, and the buttons (remembered for the matrices shown next).
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
    cluster it the same way, in any browser and after restarts, gets them at once.
  - *Export figure*: SVG, PDF or PNG of the whole matrix as shown (order, colors,
    cell size, dendrograms) with a colorbar; cells are an embedded image, labels and
    lines stay vector in SVG/PDF.
- **Reference data**: tables added in the Admin tab (below), and any in
  `reference_data/` listed in its `manifest.json` (format in
  [reference_data/README.md](reference_data/README.md)). Their chemicals are labelled by
  their names in the Odors tab where they can be matched (or abbreviations, or as in the
  file: *Chemicals* in the viewer), the rest as in the file. A yellow warning is by a
  table with chemicals not identified by a CID or InChI (in the file, or through the
  Odors tab), naming the first of them, with all of them to show and copy (also when
  adding one in the Admin tab); these are left out of its matrix unless a box by the
  warning is ticked. One that cannot be read is listed by its name alone, with why.
  A table of glomeruli (by their labels) has no chemicals, and so no warning: it says it
  can be compared with datasets on their glomeruli. Its other axis is taken to be KCs:
  *Sum over KCs* in the viewer (here and in the Compare tab) shows each glomerulus's
  values summed over them (those not removed; missing values left out), as one row or
  column.
  *Sort rows* and *columns* order each by its values over those of the other axis shown
  (by mean or largest value, highest first, those with none last, within its group;
  remove columns to sort by some of them, e.g. one product of VCF's). Sequential colors
  are on a log scale when the values are none negative and span 3 or more orders of
  magnitude (*Scale*: auto, or linear or log, 0 then the lowest color), as VCF's do.
- **Headspace calculator**: odors are added by any part of a name, abbreviation, alias,
  CID or CAS number, or a mixture starts from a panel of the datasets: its odors in the
  odor table with a CID, but solvents, each at the highest concentration presented (what
  was left out is said). The solvent (paraffin oil or water) follows what the odor table
  says of the odors, unless one is chosen. The headspace is at equilibrium in a closed
  vial at 25 °C: over paraffin oil by Henry's law with constants chemutils estimates
  (`inchi2k_henry_pfo`, from activity coefficients in hexadecane by modified UNIFAC), or
  by Raoult's law only (an ideal solution) if chosen; over water by Henry's law. What it
  needs of each odor (molar mass, density, vapor pressure, Henry's law constant) is
  looked up through chemutils the first time (a few seconds an odor) and kept with the
  odor table; where something is missing, what was assumed instead is noted by the odor
  (a vapor pressure estimated from the structure, a density of 1 g/mL). Vapor pressures
  and Henry's law constants are labelled *measured* or *estimate* (where chemutils says:
  every constant in oil is an estimate, with the vapor pressure it was estimated with
  shown, labelled too). What was calculated since the page was opened is shown again at
  once, for another solvent or model too. "Download table" saves the results as a CSV
  (a row per odor, then one of them all together: what was used of each, whether
  measured, where from, and its headspace in every unit) or LaTeX (as shown, in the
  units chosen, linked, measured values and estimates marked). The mixture, solvent,
  model and units are kept in this browser. How it is computed, and what it leaves out:
  [design.md](design.md#headspace-calculator).
  - *Serial dilutions*: for each odor of the mixture with a concentration, the µL of its
    neat chemical into the first of its vials, how many dilutions of how many µL of each
    vial into the next, and each vial's concentration, in the mixture's solvent: each vial
    gets the same amount of it (2 mL by default; paraffin oil weighed, at 0.85 g/mL as
    typed; water by volume); then how many vials (aliquots of solvent) they take in all,
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
    mean, geometric mean, maximum or minimum), and the headspace over the samples taken
    as water, by Henry's law, else by Raoult's law with its vapor pressure, saying which.
    Both are looked up again through chemutils in the background (those the file lacks
    first, then those it has, to check them; kept apart from the odor table's, and the
    tables fill in as they come): what it finds is used, else the file's (*file*), else
    estimates (*estimate*: a Henry's law constant from solubility over vapor pressure, a
    vapor pressure from the structure). Where what it finds differs from the file's by
    more than 10%, a warning lists those chemicals, and the file's value is shown by each
    (⚠). *Sort the tables by* quantity (the default), headspace or name. With up to 8
    products, each product's values are columns too. If
    chosen, also the v/v of each in paraffin oil, all together, that gives the same
    headspace (by Henry's law with chemutils' estimated constants, or Raoult's law only);
    what that needs of each chemical is looked up through chemutils in the background
    the first time (a few seconds a chemical, at most 500 chemicals a search), and the
    table fills in as it comes. *Make them the mixture* puts those that are odors of the
    odor table (at most 50) in the mixture above, at those v/v. "Download tables" saves
    a CSV per chemical (each product's values, the headspace in every unit, the oil) or
    of the matching lines (with where each constant and vapor pressure is from, and the
    file's).
- **Admin** (to the right of Docs; it first asks whether you are Tom, Jon or Betty, or
  have asked one of them): settings for the whole site.
  - *Folders on the server* the app may read (by default tensor) and also write (none
    by default: for backups), besides its own data folder. Saved changes take
    effect when the server restarts, which a button in the message saving them, or the
    *Restart* section near the bottom, does when it runs under Docker or systemd. How
    they are enforced:
    [DEPLOY.md](DEPLOY.md#folders-it-may-read-and-write). Below them, a tree of each
    file the server has read or written in the folder it runs from (e.g. `/opt/olfdb`,
    as a path on the server even in Docker), with sizes and modification times, those
    deleted since struck through: its database, uploads and the rest of `data/`, but not
    the repository's own files, nor those in the folders above or of reference data.
  - *Backups*: a button copying everything the site keeps into one of the folders it may
    write: the SQLite database (a consistent copy), and every dataset's files as uploaded,
    odor YAMLs, `Experiment.xml`, README and `metadata.json`, with reference data as read
    in, and datasets deleted there (not raw or motion-corrected
    data, nor uploads under way). Each backup is a folder `olfdb-backup-<date>_<time>`
    laid out like `data/`; files unchanged since the backup before it there are hard
    links to it, so each takes the space of what changed only. Listed with when each was
    made, its size and what is new in it; above them, when the last was made and whether
    it is current (else what changed since: the database, or files new, changed or
    removed).
  - *Reference data*: add a CSV, TSV or parquet file on the server, or an xarray
    DataArray saved as netCDF (`.nc`) (checked as you type its path, when the server may
    read it: its size, and how many of its chemicals match the Odors tab), with a title,
    description, source, labels, colors (or from its values) and whether its rows or
    columns are chemicals (or found from identifier columns, e.g. `cid`, `pubchem_cid`,
    `inchi`, `cas`, `name`, or rows labelled so; of a DataArray, coordinates named so).
    Once read, its fields are listed, to choose which make the matrix (the rest are
    ignored): a table laid out as its matrix, or one value per line (a long table, e.g.
    VCF's: suggested when it looks like one), whose rows and columns are each
    combination of the fields chosen for them (labelled by the first, the others told
    apart and shown on hover), and whose values are a field of numbers (the mean of
    several lines of a cell) or presence (1 where a line lists a row and column); of a
    DataArray, which coordinates label and describe its rows and columns. Each choice
    shows what it makes (its size, chemicals identified) before it is added.
    The app never changes the file: it reads a copy into `data/references/`, again
    whenever the file changes. A file outside the folders the server may read is added
    to them (itself, to read only), and read once the server restarts, with what that
    takes in Docker or systemd. Chemicals are matched to the Odors table by CID, else
    InChI, else CAS, MetaCyc or LOTUS ID, name or abbreviation, never against a CID or
    InChI given. Deleting one deletes the copy (never the file), and takes its file off
    the folders the server may read if it was added there for it. When one cannot be
    read, it says why, with a red dot on the Admin tab.
  - *Delete datasets*: every dataset, newest upload first (with a filter), each with a
    Delete button that asks first. A deleted dataset is gone from the site at once; its
    files (as uploaded, with `metadata.json` holding every detail entered) are moved to
    `data/deleted/<id>/` on the server, from where it could be uploaded again.
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

Downloads: `/api/download?ids=1,2&format=files|wide|long` (as from the Browse tab).
Datasets: `/api/datasets`, and each one's matrix at `/api/datasets/{id}/matrix`; the
odor table, with each odor's InChI, vapor pressure and molar mass: `/api/odors`. Every endpoint is listed in
[design.md](design.md#http-api). On the server, `manage.py` lists, shows, exports and
deletes datasets ([DEPLOY.md](DEPLOY.md), section 7).

## Development

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements-dev.txt               # needs git (chemutils is installed from GitHub)
uvicorn app.main:app --reload --port 8080      # data goes to ./data
```

pip does not replace chemutils when its pin in `requirements.txt` changes (the version
number stays the same): `pip install --force-reinstall --no-deps "$(grep '^chemutils'
requirements.txt)"`, then `pip install -r requirements.txt` again for any new
dependencies of it. Updating the pin takes a few more steps:
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
