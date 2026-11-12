# AL segmentation: which registration the fits use

As of 2026-11-11, with al-diag-seg pinned at `d2e8d0e` (`requirements-alseg.txt`).

## What is used

ANTsPy (`antspyx==0.6.3`, `requirements-worker.txt`) is the library that does every
registration. Affine and SyN are two of its transform types (`type_of_transform`), not
alternatives to it.

The fit form offers no registration setting (`fits.DEFAULTS`, `fits.ROI_DEFAULTS`), and
`fit_worker` passes none to `alseg.model.Options`. So every fit takes al-diag-seg's defaults.
The saved models of fits 2, 6 and 7 (`data/fits/<id>/model/model.json`) all have them:

| Step | Option | Used |
|---|---|---|
| Training flies onto the atlas template, and the atlas onto each new fly (`alseg/atlas.py`: `build_atlas`, `atlas_to_fly`) | `reg` | **SyN** (affine, then deformable) |
| A fly's anatomical stack onto its own functional planes, and the left-right mirror check (`hires_to_functional`, `orient_hires`) | `hires_reg` | **Rigid** |
| A reference atlas (Grabe 2016, on by default) onto the flies' atlas (`alseg/reference.py`: `align_reference`) | `ref_reg` | **SyN**, after a linear fit to glomerulus centroids (affine or similarity, whichever predicts left-out glomeruli better) |

Of fit 7's 86 flies, 14 have an anatomical stack. The other 72 are registered between flies
by their average image: about 8–12 planes, roughly 10 µm apart.

## What has been compared

Mostly nothing, and nothing on the database's flies:

- **Affine vs. SyN between flies (`reg`):** never compared. There is no comparison in
  al-diag-seg's README, design.md or history, nor in the tuning of the fits since. SyN is the
  default on the reasoning that glomerulus layouts differ between flies, untested. It may
  matter most for flies without a stack: al-diag-seg's design.md says registration between
  few-plane volumes is poorly constrained in z, so a deformable fit could overfit there.
  That is a concern, not a finding.
- **Rigid vs. Affine within a fly (`hires_reg`):** on synthetic data only
  (`make_synthetic.py`). Label Dice was 0.94–0.95 for rigid, 0.77–0.94 for affine, 0.42–0.64
  with no registration (al-diag-seg design.md).
- **Affine vs. SyN for a reference atlas (`ref_reg`):** once, aligning Grabe to the hemibrain
  atlas (two atlases, not the database's flies). Shared glomeruli overlapped with median Dice
  0.46 after affine and 0.63 after SyN (al-diag-seg README). Its effect on naming in held-out
  flies is unmeasured.

## How they could be compared

The final ROIs are Cellpose's own (`Options.final = "pred"`), so the found and right scores
do not depend on registration. Registration only changes which glomerulus each ROI is named
(fit 7: 83 of 119 named right), and the atlas outputs (`*_atlas_maxprob.tif`, the refined ROI
sets).

A fair test reuses fit 7's trained segmenter (`alseg-fit --skip-train --model ...`) and only
rebuilds the atlas with `--reg Affine` (and/or `--ref-reg Affine`), then compares naming on
the same 4 held-out flies. In fit 7 the atlas stage took ~10 min (97 registrations: 82 flies,
14 stacks, 1 reference) and scoring the held-out flies ~10 min, so about 20 min per setting,
with no retraining. ANTs registration differs slightly between runs unless `--deterministic`
(about 2.4× slower), which matters for close calls.
