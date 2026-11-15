# Plan: keeping neighbouring glomeruli out of predicted ROIs

For an agent to implement. Six approaches, all to be tryable in the site (AL segmentation tab),
in the order of interest: **3** trim by tuning, **4** unmix traces, **6** purity score, **5** split
merged masks, **1** geometric margin, **2** cores from Cellpose. **Build 6 first**: it is small, and
every other approach is judged by it. Then 3, 4, 5, 1, 2.

## What is known (fit 7, its 4 held-out flies, 2026-11-14)

Flies: `Sam_2024-06-11_2`, `Tom_2023-05-08_1`, `Tom_2023-11-21_3`, `Tom_2024-09-12_2`; their
predicted ROIs are `data/fits/7/results/<fly>_pred.tif` (read only). Numbers are of in-plane masks,
on the diagnostic recordings' trial-mean dF/F (other panels alike). Made by
`notes/roi_contamination_edges.py` and `notes/roi_contamination_variants.py` (usage in each):

| | predicted | manual |
|---|---|---|
| median gap to the nearest ROI on its plane | 0.5 um (1 px) | 1.9 um |
| masks touching another | 48% | 0% |
| neighbour's share, edge facing it | 0.30 | 0.15 |
| neighbour's share, edge away from it | 0.03 | 0.04 |
| neighbour's share, whole ROI's mean | 0.04 | 0.03 |
| ROI-odor pairs over 0.3 dF/F, 183 matched masks | 18.0% | 18.3% |

So edges facing a neighbour are about twice as contaminated as manual ones, but whole-ROI means
are not measurably broader. 11 of 455 masks hold >= 40% of two manual ROIs (merges). Quick tries:
erode 1 px: nothing touching, 85% of the area, `leak` 0.07 -> 0.04; erode 2 px: gap 2.3 um, 72%,
0.03; removing only pixels within 2-3 px of another ROI: `leak` 0.06 (it cannot see neighbours that
got no ROI); trimming by the 20 odors' tuning: touching 48 -> 30%, responses unchanged; unmixing
with an assumed blur: fraction responding 17-23% by the blur assumed (unstable, and done on dF/F
maps, which is wrong: see 4).

*Neighbour's share* (the yardstick): per in-plane mask with a core (>= 8 px left after eroding
3 px) and a neighbour with a core within 4 px; `v` = the mean response vector of some of its pixels
(the edge ring 2 px deep, the part of it within 6 px of the neighbour, the part over 10 px from it,
or the whole mask); non-negative least squares `v = a core + b neighbour's core` over odors; share
`b / (a + b)`; median over masks.

## Where things go

- **Steps that change masks (3, 5, 1, 2): al-diag-seg** (`~/src/al-diag-seg`; olfdb pins a commit of
  it in `requirements-alseg.txt`). New module `alseg/clean.py`: each step a function
  `(labels (Z, Y, X), fly, **params) -> labels`, in-plane only (planes are 10 um apart), distances in
  um by the fly's `spacing`, an ROI's label kept (so its name carries over). Steps only remove
  pixels: the old refinement (`alseg/refine.py`, off since 97aba30) also grew and moved them, and
  73-81% of its masks touch.
- `Options.clean` (ordered step names) and their parameters, saved with a model and settable at
  prediction like `cli.MASK_OPTIONS`. `Model.predict` already returns several kinds of labels, each
  written and scored by `evaluate`: add one kind per step applied alone (`clean_<step>`) and `clean`
  (the chosen steps in order); `Options.final` may be `clean`. Cleaned kinds take `pred`'s names (no
  renaming, so that only masks differ), except where 5 makes new ROIs.
- **Trying them in the site without a refit:** a prediction now runs with the model's saved options
  (`fit_worker.run_prediction`). Add per-prediction options (stored with the prediction, laid over
  `model.options`), a "How its ROIs are cleaned" group in the prediction form (and the fit form:
  `fits.ROI_DEFAULTS`, `FLOAT_LIMITS`, which `tests/test_fits.py` checks against al-diag-seg's
  defaults), each kind selectable in the ROI viewer and as the ROIs of *Extract signals*, and each
  kind's scores in the tab. Check first what the viewer and `extract.ROI_KINDS` already allow.
- **4 changes traces, not masks: olfdb's `app/extract.py`** (*Extract signals*), an option, off by
  default: with it off, output must stay identical to al_analysis' (`tests/test_extract.py`, and the
  bit-for-bit check in the memory note *Checking against al_analysis*).
- Each step: tests (al-diag-seg's synthetic flies, `make_synthetic.py` / `smoke_test.py`; olfdb's
  with the `stand-in` segmenter), the pin bumped, and its numbers on the 4 flies added to a results
  table (al-diag-seg's README, olfdb's `design.md`). Tune parameters leaving one fly out, as before.
- House rules (CLAUDE.md, memory): live `data/` read only; docker test runs with memory and pid
  limits and stopped after; other sessions edit this tree; deploy only by `restart-olfdb`. A fit
  may be using the GPU: run trials on the CPU.

## 6. Purity score per fit and prediction (build first)

- al-diag-seg, beside `refine.crosstalk_report` (whose `leak` the tab already shows, with the manual
  ROIs' own): `edge_share` (neighbour's share at the facing edge, as defined above, in um),
  `whole_share`, `gap_um` (median) and `touching` (fraction), for each kind of ROIs and for the
  manual ones; and `breadth`: on masks matched to a manual one (IoU >= 0.5), the fraction of
  ROI-odor pairs over a threshold, predicted and manual.
- Into `summary.csv`; olfdb: `fit_worker.SCORES` / `summary`, the tab's score tables (predicted
  beside manual, as `leak` is), blank for older fits.
- Done when: the table above is reproduced by a prediction of fit 7's model on those flies.

## 3. Trim by tuning

- Step `trim`: drop a mask's pixels that carry a neighbour more than the mask's own core. Per pixel,
  the same regression as the yardstick (`pixel = a own core + b neighbour's core`, each neighbour
  within ~2 um in turn); drop where `b / (a + b)` > `max_share` (start 0.3). Also drop pixels that
  follow their own core worse than nearly all core pixels do (below the core pixels' 5th percentile
  of correlation): this catches neighbours that got no ROI. Smooth the evidence within the mask
  (sigma ~0.5 um, normalized by the mask) first; keep the largest connected piece; a mask that would
  lose over half its area is left whole and flagged (a candidate for 5).
- **Evidence, in order of cost** (20 trial-mean values per pixel was too noisy, twice over):
  1. every recording's per-trial maps, not only the diagnostic trial means: `trial_dff.tif` in each
     recording's folder (check its axes), ~3x the trials and all panels;
  2. the movies (`mocorr.tif` per recording): one chunked pass accumulating, per candidate pixel,
     its sums of products with its own and its neighbours' core traces (cores are fixed beforehand,
     so one pass does it). Correlate dF/F within trials, or F less a running baseline: raw F makes
     every pixel follow every other through bleaching and movement. Costs about an extraction
     (~20 s per GB of movie); warn of it in the form.
  Try 1; go to 2 only if 1 does not reach the target.
- al-diag-seg reads these itself (`alseg/fly.py`, keeping to hong-data's layout); the fly passed to
  a step holds the evidence, so 5 can reuse it.
- Target: `edge_share` at or below the manual ROIs' (0.15), median area >= 85% of `pred`'s, no
  matched mask lost, `breadth` unchanged or lower.

## 4. Unmix the traces

- *Extract signals* option `unmix`. Per plane, model **fluorescence, not dF/F** (light adds; dF/F
  does not): `F(pixel, t) = sum_k a_k(pixel) f_k(t) + background`. Since the solution is linear, an
  ROI's unmixed trace is a weighted sum of pixels (`pinv` of the footprints): `plane_traces` goes
  from mask means to weight images in the same chunked pass, and the rest (windows, `MODES`) stays.
- Footprints `a_k`, two variants to compare: (i) the mask blurred by a Gaussian whose width is
  **fit per fly** (least residual over a grid, on dF maps), never assumed; (ii) seeded NMF: `a_k >=
  0`, started at the mask, confined to the mask grown ~2 um, alternating non-negative least squares
  on dF (trial maps, or the movie binned in time). A background term per plane, so shared
  out-of-focus light is not pushed into ROIs. No term for the planes above and below unless its
  weight is fit too and the data wants it (assuming 0.2 moved everything).
- **dF/F trap:** if unmixing lowers an ROI's baseline, its dF/F is inflated. Unmix dF, and divide by
  the plain mask mean's F0 (state this in the extraction's README).
- Guards: ridge or refuse where two footprints are nearly collinear (report the condition number);
  write plain and unmixed traces both, and per ROI how much its responses changed.
- Check: on synthetic movies with a known blur, the true traces come back; on the 4 flies, where a
  core is quiet and its neighbour responds the ROI's response falls to the core's, where the
  neighbour is quiet responses are unchanged (slope 1 against plain), and it makes no new negative
  responses (count pairs below -threshold, before and after).
- Risk to report on, not hide: neighbouring glomeruli often truly respond together, and a footprint
  too wide subtracts that.

## 5. Split or flag merged masks

- Positives to recover: the 11 masks holding >= 40% of two manual ROIs; negatives: the 266 holding
  one (none should split).
- Within each in-plane mask, 2 clusters of the (smoothed) pixel evidence of 3. Split when both
  parts are connected, each at least a minimum area, tuned differently from each other (correlation
  of their means below ~0.5) and alike within. Do it on Cellpose's per-plane masks **before**
  `segment.join_planes`, so the parts are joined across planes and named as any mask is.
- Masks that fail the test but are incoherent (`crosstalk_report`'s `coherence`), and those 3 left
  whole, are flagged: a column in the identities CSV, shown in the ROI viewer for redrawing.

## 1. Geometric margin

- Step `margin`: erode every in-plane mask all round by `erode_um` (start 0.5 um, 1 px), then remove
  pixels within `gap_um` of another ROI; an ROI that would fall under a minimum size keeps its
  largest inscribed part. Eroding all round is what lowers `leak`; the gap alone does not.
- After naming and joining, so detection and names are untouched. Its IoU with manual ROIs drops by
  construction: show `pred`'s detection scores beside the cleaned kind's purity.

## 2. Cores from Cellpose

- 2a, no refit: have `segment.predict` also return Cellpose's cell probability map (on the fly's
  grid); step `core`: a mask's pixels with probability >= `core_cellprob` (try 0 and +1; masks are
  found at -2), largest piece, minimum size as in 1.
- 2b, a refit: train on outlines eroded 2 px (`Options.erode` exists; add it to the fit form),
  masks not grown. About 2 h; compare by 6 against fit 7.

## Done when

Each approach is an option in the site with defaults that leave today's output unchanged, and one
table compares all of them on the 4 held-out flies by 6's scores, area kept, ROIs lost, and the
per-plane found / right scores.
