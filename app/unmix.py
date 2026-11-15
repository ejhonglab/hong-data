"""Neighbouring ROIs' signals unmixed: *Extract signals*' option ``unmix`` (`extract.run`).

An ROI's mask mean holds some of its neighbours' light (blur, and masks that reach into them:
demixing_plan.md, section 4). Per plane, the movie's fluorescence over the plane's ROIs (its
`region`: their masks, and the pixels within `REGION_UM` of them) is modelled as each ROI's
footprint times its signal: F(p, t) = sum_k a_k(p) f_k(t), and with `BACKGROUND` a background
uniform over the region, + b(t). In fluorescence, not dF/F: light adds, ratios do not. Least
squares makes each f_k a weighted sum of the region's pixels (the footprints' pseudo-inverse),
so an ROI's unmixed trace is read in the same pass over the movie as its mask mean
(`extract.plane_traces`, given each ROI-plane's weights).

Footprints, three ways (`VARIANTS`), all fit to the trials' dF maps (`trial_maps`: the mean F
of a trial's response window less that of its baseline, per pixel):

- ``blur``: each mask blurred by a Gaussian whose width is fit per set of ROIs (the least
  residual of the dF maps, over `SIGMAS_UM`), never assumed; over the masks alone;
- ``nmf``: seeded non-negative matrix factorization, each footprint started at its mask and
  kept within `NMF_REACH_UM` of it (`nmf`), over as far;
- ``nmf_rims``: the same, with a component more per ROI-plane for the region's pixels outside
  every mask that are nearest its own (its rim: `rims`), started at those pixels and kept
  within `RIM_IN_PX` of them. The rims' signals are not kept: they stand for the light of
  glomeruli without an ROI, and neighbours' spill, which is then not read as an ROI's own.
  Its factorization has rarely settled by `NMF_ITERS` (design.md: where it stops matters
  little to the responses, more to which ROI-planes are refused).

Chosen on fit 7's held-out flies (design.md): the blur's region is the masks alone, since with
pixels round them its width grew with how far they reached (to the widest tried), explaining
the light of glomeruli without an ROI by its ROIs' footprints, which then took too much from
their neighbours; no background is modelled, as it took real responses, its level being set by
the region's pixels outside the ROIs, which are mostly such glomeruli, not out-of-focus light;
and rims reach a pixel into the masks, since reaching as far in as the ROIs' footprints reach
out they took 5 to 10% of the ROIs' own responses, and kept outside the masks, less of their
neighbours' light (both tried with footprints fit to every frame of the movies).

An ROI's own part of its mask mean is u = mean over its mask of a_k, times f_k. Its unmixed
trace (`unmixed`) is its mask mean less the change of the rest (its neighbours' light, and the
background's) from each trial's baseline: so its F0 is its mask mean's, and its dF/F its own
dF over that F0. Unmixing takes light from F0 as well as from dF, so the dF/F of the unmixed F
alone would be inflated. Left as their mask means, and said to be: an ROI-plane whose
footprint is nearly a combination of the others' (its variance inflation over `VIF_MAX`), and
one holding others (`layout`'s outlines), which is left out of the model.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

VARIANTS = {
    "blur": "each mask blurred by a Gaussian, its width fit to the trials' dF maps over the masks (per set of ROIs)",
    "nmf": "seeded non-negative factorization of the trials' dF maps, each footprint from its mask, kept near it",
    "nmf_rims": "seeded non-negative factorization of the trials' dF maps, with a component more per ROI for the pixels just outside its mask (its rim)",
}
DEFAULT = "blur"
SIGMAS_UM = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)  # the blur's widths (s.d.) tried
NMF_REACH_UM = 2.0  # how far from its mask an NMF footprint may reach
RIM_IN_PX = 1.0  # how far beyond its own pixels (so into the masks) a rim's footprint may reach
RIM_MIN_PX = 8  # the pixels a rim has at least: a mask with fewer of the region's nearest it has none
REGION_UM = {"blur": 0.0, "nmf": NMF_REACH_UM, "nmf_rims": NMF_REACH_UM}  # how far from the masks each variant's model is fit (its region; see above)
BACKGROUND = False  # whether a background uniform over the region is modelled (see above)
PIXEL_UM = 0.5  # a pixel's width, where the recording's Experiment.xml does not say
VIF_MAX = 10.0  # an ROI-plane's variance inflation over which it is left as its mask mean
HELD = 0.5  # the share of a smaller ROI-plane's mask within another's that makes that other an outline (left out)
NMF_ITERS = 100
NMF_TOL = 1e-4  # an NMF's relative change of its footprints, under which it has converged


def pixel_um(recording: Optional[Path]) -> Optional[float]:
    """A pixel's width in um, by a raw recording's Experiment.xml (its LSM's pixelSizeUM); None
    if it does not say, or is not there."""
    from . import thorimage

    try:
        lsm = thorimage.parse_experiment_xml((recording / "Experiment.xml").read_text(encoding="utf-8", errors="replace"))["info"]["lsm"]
        v = float(lsm["pixelSizeUM"])
    except (OSError, TypeError, KeyError, ValueError, thorimage.ExperimentXmlError):
        return None
    return v if v > 0 else None


def layout(rois: list[dict], yx: tuple[int, int], reach_px: float) -> list[dict]:
    """The planes of a set of ROI-planes (`extract.read_rois`) that have any: each its ``z``,
    ``box`` (y0, y1, x0, x1: its masks, and `reach_px` round them), ``members`` (its ROI-planes
    modelled, by their index), ``masks`` ((members, h, w), bool, in the box), ``region`` ((h,
    w): the pixels within `reach_px` of a mask, which the model is fit to), ``idx`` (the
    region's pixels, flat in the box) and ``outlines``: {index: the indices of the ROI-planes it
    holds} of those left out of the model, holding at least `HELD` of a smaller one's mask (an
    outline of several glomeruli, or of the whole lobe, as Sam's ``AL_plane3``: not a source of
    light of its own, and a footprint over all those it holds)."""
    from scipy import ndimage as ndi

    ny, nx = yx
    pad = int(math.ceil(reach_px)) + 1
    out = []
    for z in sorted({r["z"] for r in rois}):
        on = [i for i, r in enumerate(rois) if r["z"] == z]
        y0 = max(0, min(rois[i]["y0"] for i in on) - pad)
        x0 = max(0, min(rois[i]["x0"] for i in on) - pad)
        y1 = min(ny, max(rois[i]["y0"] + rois[i]["mask"].shape[0] for i in on) + pad)
        x1 = min(nx, max(rois[i]["x0"] + rois[i]["mask"].shape[1] for i in on) + pad)
        masks = np.zeros((len(on), y1 - y0, x1 - x0), bool)
        for j, i in enumerate(on):
            r = rois[i]
            masks[j, r["y0"] - y0:r["y0"] - y0 + r["mask"].shape[0], r["x0"] - x0:r["x0"] - x0 + r["mask"].shape[1]] = r["mask"]
        flat = masks.reshape(len(on), -1).astype(np.float64)
        n = flat.sum(axis=1)
        both = flat @ flat.T  # [j, k]: pixels in masks j and k
        holds = {on[j]: [on[k] for k in range(len(on)) if k != j and n[k] < n[j] and both[j, k] >= HELD * n[k]] for j in range(len(on))}
        outlines = {i: held for i, held in holds.items() if held}
        keep = [j for j, i in enumerate(on) if i not in outlines]
        masks = masks[keep]
        if not len(masks):
            continue
        union = masks.any(axis=0)
        region = ndi.distance_transform_edt(~union) <= reach_px if reach_px > 0 else union
        out.append({"z": z, "box": (y0, y1, x0, x1), "members": [on[j] for j in keep], "masks": masks, "region": region,
                    "idx": np.flatnonzero(region), "outlines": outlines})
    return out


def _flat(plane: dict, images: np.ndarray) -> np.ndarray:
    """Images ((n, h, w), in a plane's box) as (region pixels, n)."""
    return images.reshape(len(images), -1)[:, plane["idx"]].T


def trial_maps(movie, trials: list[dict], n_response: int, planes: list[dict],
               progress: Optional[Callable[[int, int], None]] = None) -> list[np.ndarray]:
    """Each trial's dF map on each plane of `planes` (`layout`): the mean of the movie over the
    trial's response window (its first `n_response` time points from the odor's onset, as many
    as it has) less its mean over its time points before the onset, at each pixel of the
    plane's region. Only those time points are read; `progress` is told (trials read, of how
    many).

    Returns:
        per plane, (region pixels, trials), float32; NaN for a trial with no time point before
        its onset.
    """
    out = [np.full((len(p["idx"]), len(trials)), np.nan, dtype=np.float32) for p in planes]
    for k, t in enumerate(trials):
        n_base, n_win = t["onset"] - t["start"], min(t["end"] + 1, t["onset"] + n_response) - t["onset"]
        if n_base > 0 and n_win > 0:
            d = movie.volumes(t["onset"], n_win).mean(axis=0, dtype=np.float64) - movie.volumes(t["start"], n_base).mean(axis=0, dtype=np.float64)
            for j, p in enumerate(planes):
                y0, y1, x0, x1 = p["box"]
                out[j][:, k] = d[p["z"], y0:y1, x0:x1].ravel()[p["idx"]]
        if progress:
            progress(k + 1, len(trials))
    return out


def blur(plane: dict, sigma_px: float) -> np.ndarray:
    """A plane's footprints as its masks blurred by a Gaussian of s.d. `sigma_px` (none for 0):
    (region pixels, members)."""
    from scipy import ndimage as ndi

    m = plane["masks"].astype(np.float64)
    if sigma_px > 0:
        m = np.stack([ndi.gaussian_filter(x, sigma_px, mode="constant") for x in m])
    return _flat(plane, m)


def design(A: np.ndarray, background: bool = BACKGROUND) -> np.ndarray:
    """The footprints (pixels, ROIs), with the background's (uniform over the region) last."""
    return np.column_stack([A, np.ones(len(A))]) if background else A


def _known(D: np.ndarray) -> np.ndarray:
    return D[:, ~np.isnan(D).any(axis=0)].astype(np.float64)


def residual(A: np.ndarray, D: np.ndarray) -> float:
    """The sum of squares of dF maps (`D`: pixels, maps) not explained by footprints `A` (least
    squares; footprints that are combinations of others add nothing)."""
    X = _known(D)
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    u = u[:, s > s.max() * 1e-10]
    return float((X ** 2).sum() - ((u.T @ X) ** 2).sum())


def fit_blur(planes: list[dict], maps: list[np.ndarray], sigmas_px, background: bool = BACKGROUND) -> dict:
    """The blur whose footprints (`blur`) best explain the dF maps (`maps`: per plane of
    `planes`, as `trial_maps`, of every recording joined): the least residual (over all planes)
    of `sigmas_px` (``residuals``: each one's), then between the two either side of it
    (bounded Brent search): ``sigma_px``, with ``at_edge`` if it is the widest tried (so the
    best may be wider still)."""
    from scipy.optimize import minimize_scalar

    def res(s: float) -> float:
        return sum(residual(design(blur(p, s), background), D) for p, D in zip(planes, maps))

    grid = [res(s) for s in sigmas_px]
    i = int(np.argmin(grid))
    lo, hi = sigmas_px[max(0, i - 1)], sigmas_px[min(len(sigmas_px) - 1, i + 1)]
    best, least = float(sigmas_px[i]), grid[i]
    if hi > lo:
        r = minimize_scalar(res, bounds=(lo, hi), method="bounded", options={"xatol": 0.02})
        if r.fun < least:
            best, least = float(r.x), float(r.fun)
    return {"sigma_px": best, "residual": least, "residuals": grid, "at_edge": best >= sigmas_px[-1]}


def rims(plane: dict) -> np.ndarray:
    """A plane's rims ((rims, h, w), bool, in its box): of each of its masks, the region's
    pixels outside every mask that are nearest that one, where there are at least `RIM_MIN_PX`
    of them. Each is a component more of the plane's `nmf` (its `extra`), so that the light
    round the ROIs (of glomeruli without one, or their neighbours' spill) need not be taken
    for an ROI's."""
    from scipy import ndimage as ndi

    near = np.stack([ndi.distance_transform_edt(~m) for m in plane["masks"]]).argmin(axis=0)
    halo = plane["region"] & ~plane["masks"].any(axis=0)
    out = [halo & (near == j) for j in range(len(plane["masks"]))]
    return np.array([r for r in out if r.sum() >= RIM_MIN_PX], bool).reshape(-1, *halo.shape)


def nmf(plane: dict, D: np.ndarray, reach_px: float, background: bool = BACKGROUND, iters: int = NMF_ITERS,
        tol: float = NMF_TOL, extra: Optional[np.ndarray] = None, extra_px: float = RIM_IN_PX) -> tuple[np.ndarray, dict]:
    """A plane's footprints by seeded non-negative matrix factorization of its dF maps (`D`:
    region pixels, maps): D = A C (+ the background's), A >= 0, each column of A started at its
    mask and 0 beyond `reach_px` of it; C (the maps' coefficients) of any sign. Alternating
    least squares: C by least squares, A by HALS (each footprint in turn, exactly, given the
    others), each footprint then scaled to a mean of 1 over its mask (one that would have none
    there is its mask again). With `extra` ((n, h, w), bool, in the plane's box: its `rims`),
    a component more for each, after the members': as theirs, with those pixels for its mask,
    and 0 beyond `extra_px` of them.

    Returns:
        (A: (region pixels, members and then `extra`), {iterations, change: the last relative
        change of A, reset: how many footprints were set back to their mask})
    """
    from scipy import ndimage as ndi

    masks = plane["masks"] if extra is None else np.concatenate([plane["masks"], extra])
    reach = [reach_px] * len(plane["masks"]) + [extra_px] * (len(masks) - len(plane["masks"]))
    M = _flat(plane, masks.astype(np.float64))
    S = _flat(plane, np.stack([ndi.distance_transform_edt(~m) <= r for m, r in zip(masks, reach)])) > 0
    X = _known(D)
    A, K = M.copy(), M.shape[1]
    n = M.sum(axis=0)
    reset, change, it = 0, float("nan"), 0
    for it in range(1, iters + 1):
        C, *_ = np.linalg.lstsq(design(A, background), X, rcond=None)
        E = X - C[K][np.newaxis] if background else X
        Ck = C[:K]
        G, H = Ck @ Ck.T, E @ Ck.T
        old = A.copy()
        for k in range(K):
            if G[k, k] > 0:
                A[:, k] = np.where(S[:, k], np.maximum(0.0, (H[:, k] - A @ G[:, k]) / G[k, k] + A[:, k]), 0.0)
        own = (A * M).sum(axis=0) / n
        dead = ~(own > 0)
        reset += int(dead.sum())
        A[:, dead] = M[:, dead]
        A[:, ~dead] /= own[~dead]
        change = float(np.linalg.norm(A - old) / np.linalg.norm(old))
        if change < tol:
            break
    return A, {"iterations": it, "change": change, "reset": reset}


def solve(plane: dict, A: np.ndarray, background: bool = BACKGROUND) -> dict:
    """What reads each of a plane's ROI-planes' own part of its mask mean out of the region's
    pixels, given their footprints `A` (region pixels, members; then any more of the model's,
    as `nmf`'s `extra`: fit with them, their own signals not kept): least squares (the
    pseudo-inverse), each ROI's row times the mean of its footprint over its mask.

    Returns:
        ``weights`` (members, region pixels); per member: ``vif`` (its variance inflation: 1 /
        (1 - R²) of its footprint on the others', the background's and any more among them),
        ``gain`` (the variance of its unmixed trace over its mask mean's, were the pixels'
        noise alike and independent), ``share`` ((members, members): [k, j] the mean of
        footprint j over mask k, over footprint k's) and ``refused`` (vif over `VIF_MAX`: left
        as the mask mean); ``cond``: the condition number of all the footprints, each of norm 1.
    """
    M = _flat(plane, plane["masks"].astype(np.float64))
    Af = design(A, background)
    K = M.shape[1]
    W = np.linalg.pinv(Af)[:K]
    n = M.sum(axis=0)
    mean_in = (M.T @ A[:, :K]) / n[:, np.newaxis]  # [k, j]: footprint j's mean over mask k
    own = np.diag(mean_in).copy()
    N = Af / np.linalg.norm(Af, axis=0)
    lam, V = np.linalg.eigh(N.T @ N)
    lam = np.maximum(lam, lam.max() * 1e-15)
    vif = ((V ** 2) / lam).sum(axis=1)[:K]
    weights = W * own[:, np.newaxis]
    gain = (weights ** 2).sum(axis=1) * n
    return {"weights": weights, "vif": vif, "gain": gain, "share": mean_in / own[:, np.newaxis], "refused": vif > VIF_MAX,
            "cond": float(np.sqrt(lam.max() / lam.min()))}


def fit(n_rois: int, planes: list[dict], maps: list[list[np.ndarray]], variant: str, px_um: float, background: bool = BACKGROUND) -> dict:
    """The footprints of a set of `n_rois` ROI-planes (`planes`: their `layout`), by `variant`
    (of `VARIANTS`), fit to the dF maps of their recordings (`maps`: each recording's
    `trial_maps`), and what reads each one's own part of its mask mean out of the movie.

    Returns:
        ``weights`` (per plane, for `extract.plane_traces`); per ROI-plane: ``refused`` (by
        `solve`, or an outline of others: `layout`), ``holds`` (an outline's: the ROI-planes it
        holds), ``vif``, ``gain`` (as `solve`), ``cond`` (its plane's) and ``neighbours`` (the
        other ROI-planes whose footprints are most in its mask: (index, `solve`'s share),
        largest first, those of 0.01 or more); ``how`` the footprints were fit (the blur's
        ``sigma_um``, with the ``residuals`` of `SIGMAS_UM`; an NMF's iterations, per plane,
        and with rims how many it has, ``rims``: per plane, and in all).
    """
    joined = [np.concatenate([m[j] for m in maps], axis=1) for j in range(len(planes))]
    if variant == "blur":
        f = fit_blur(planes, joined, [s / px_um for s in SIGMAS_UM], background)
        A = [blur(p, f["sigma_px"]) for p in planes]
        how = {"sigma_um": f["sigma_px"] * px_um, "sigma_px": f["sigma_px"], "at_edge": f["at_edge"],
               "residuals": {f"{s:g}": r / f["residuals"][0] for s, r in zip(SIGMAS_UM, f["residuals"])}}
    elif variant in ("nmf", "nmf_rims"):
        extra = [rims(p) if variant == "nmf_rims" else None for p in planes]
        fits = [nmf(p, D, NMF_REACH_UM / px_um, background, extra=x) for p, D, x in zip(planes, joined, extra)]
        A = [a for a, _ in fits]
        how = {"planes": [{"z": p["z"], **i, **({} if x is None else {"rims": len(x)})} for p, (_, i), x in zip(planes, fits, extra)]}
        if variant == "nmf_rims":
            how["rims"] = sum(len(x) for x in extra)
    else:
        raise ValueError(f"no such footprints: {variant}")
    out = {"weights": [], "refused": np.zeros(n_rois, bool), "vif": np.full(n_rois, np.nan), "gain": np.full(n_rois, np.nan),
           "cond": np.full(n_rois, np.nan), "neighbours": [[] for _ in range(n_rois)], "holds": [[] for _ in range(n_rois)], "how": how}
    for p, a in zip(planes, A):
        s = solve(p, a, background)
        out["weights"].append({k: p[k] for k in ("z", "box", "idx", "members")} | {"weights": s["weights"]})
        for j, i in enumerate(p["members"]):
            out["refused"][i], out["vif"][i], out["gain"][i], out["cond"][i] = s["refused"][j], s["vif"][j], s["gain"][j], s["cond"]
            sh = s["share"][j]
            out["neighbours"][i] = [(p["members"][k], float(sh[k])) for k in np.argsort(-sh) if k != j and sh[k] >= 0.01]
        for i, held in p["outlines"].items():
            out["refused"][i], out["holds"][i] = True, held
    return out


def unmixed(plain: np.ndarray, own: np.ndarray, trials: list[dict]) -> np.ndarray:
    """ROI-planes' unmixed traces ((time points, ROI-planes)), from their mask means (`plain`)
    and their own parts of them (`own`): the mask mean less the change of the rest (plain -
    own) from its mean over each trial's time points before the odor, so that a trial's F0 is
    the mask mean's, and F - F0 the ROI's own. Time points from a trial's first to the next
    one's are of that trial (those before the first, of the first); a trial with no time point
    before the odor is of its first."""
    rest = plain - own
    base = np.zeros_like(plain)
    order = sorted(trials, key=lambda t: t["start"])
    for i, t in enumerate(order):
        b = rest[t["start"]:t["onset"]]
        upto = order[i + 1]["start"] if i + 1 < len(order) else len(plain)
        base[0 if i == 0 else t["start"]:upto] = b.mean(axis=0) if len(b) else rest[t["start"]]
    return plain - (rest - base)
