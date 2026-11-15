"""What alseg.nmf was tried out as (run.py runs it, score.py counts what it finds): a plane's dF movie factorized (non-negative, or with
time courses of either sign) with a footprint per predicted mask and free ones started where those leave the most unexplained;
the free footprints' places as candidates (`additions`), and each mask's own pixels factorized in two (`splits`: which did not
tell masks over two glomeruli from the rest)."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def blur(x, sigma):
    """(N, Y, X) tensor blurred by a Gaussian of s.d. sigma px (zero padding)."""
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(x[:, None], k.view(1, 1, -1, 1), padding=(r, 0))
    return F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))[:, 0]


def noise_sd(D):
    """Per-pixel noise s.d. of a (T, Y, X) movie: from the median absolute difference of successive frames."""
    d = (D[1:] - D[:-1]).abs()
    return 1.4826 * d.median(0).values / math.sqrt(2)


def disk(center, radius, shape, device):
    yy, xx = torch.meshgrid(torch.arange(shape[0], device=device), torch.arange(shape[1], device=device), indexing="ij")
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius ** 2


def update_C(X, A, nonneg, C=None, iters=3):
    """C (K, T) for X (P, T) = A (P, K) C."""
    G = A.T @ A
    B = A.T @ X
    K = A.shape[1]
    if not nonneg:
        return torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * torch.eye(K, device=A.device), B)
    if C is None:
        C = torch.clamp(torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * torch.eye(K, device=A.device), B), min=0)
    for _ in range(iters):
        for k in range(K):
            if G[k, k] > 0:
                C[k] = torch.clamp(C[k] + (B[k] - G[k] @ C) / G[k, k], min=0)
    return C


def update_A(X, A, C, S):
    G, H = C @ C.T, X @ C.T
    for k in range(A.shape[1]):
        if G[k, k] > 0:
            A[:, k] = torch.where(S[:, k], torch.clamp(A[:, k] + (H[:, k] - A @ G[:, k]) / G[k, k], min=0), torch.zeros((), device=A.device))
    return A


def factorize(D, masks, n_free=30, sigma_px=1.0, seed_sigma_px=3.0, win_px=20, reach_px=4, iters=30, nonneg=True,
              region=None, seed_gap_px=3.0, log=None):
    """D: (T, Y, X) tensor of unit-noise dF (a plane). masks: (K0, Y, X) bool array, the predicted masks; free components are
    started (greedily, where what the seeds leave is largest) only in `region` and `seed_gap_px` from every mask, and kept off
    the masks. Returns A (K, Y, X), C (K, T), the free components' centers and the peaks they were started at, and X (P, T)."""
    T, Y, X_ = D.shape
    dev = D.device
    X = blur(D, sigma_px).reshape(T, -1).T.contiguous()  # (P, T)
    P = X.shape[0]
    K0 = len(masks)
    union = masks.any(0) if K0 else np.zeros((Y, X_), bool)
    free_ok = torch.as_tensor(~union, device=dev).reshape(-1)
    seed_ok = torch.as_tensor(ndi.distance_transform_edt(~union) >= seed_gap_px if K0 else np.ones((Y, X_), bool), device=dev).reshape(-1)
    if region is not None:
        seed_ok &= torch.as_tensor(region, device=dev).reshape(-1)
    cols, sup = [], []
    for m in masks:
        cols.append(torch.as_tensor(m, device=dev).reshape(-1).float())
        sup.append(torch.as_tensor(ndi.distance_transform_edt(~m) <= reach_px, device=dev).reshape(-1))
    A = torch.stack(cols, 1) if cols else torch.zeros((P, 0), device=dev)
    S = torch.stack(sup, 1) if sup else torch.zeros((P, 0), dtype=torch.bool, device=dev)
    C = update_C(X, A, nonneg) if K0 else torch.zeros((0, T), device=dev)
    for _ in range(3 if K0 else 0):  # the seeds settle first
        A = update_A(X, A, C, S)
        C = update_C(X, A, nonneg, C)
    R = X - A @ C if K0 else X.clone()
    centers, peaks = [], []
    zero = torch.zeros((), device=dev)
    for i in range(n_free):
        Rb = blur(R.T.reshape(T, Y, X_), seed_sigma_px).reshape(T, -1).T
        E = (torch.clamp(Rb, min=0) if nonneg else Rb).pow(2).sum(1)
        E = torch.where(seed_ok, E, zero)
        p = int(E.argmax())
        if float(E[p]) <= 0:
            break
        cy, cx = divmod(p, X_)
        s = disk((cy, cx), win_px, (Y, X_), dev).reshape(-1) & free_ok
        c = torch.clamp(Rb[p], min=0) if nonneg else Rb[p].clone()
        a = torch.zeros(P, device=dev)
        for _ in range(5):
            a = torch.where(s, torch.clamp(R @ c / (c @ c + 1e-12), min=0), zero)
            if float(a @ a) == 0:
                break
            c = (a @ R) / (a @ a)
            if nonneg:
                c = torch.clamp(c, min=0)
        if float(a @ a) == 0 or float(c @ c) == 0:
            seed_ok[p] = False
            continue
        R = R - a[:, None] * c[None]
        A = torch.cat([A, a[:, None]], 1)
        C = torch.cat([C, c[None]], 0)
        S = torch.cat([S, s[:, None]], 1)
        centers.append((cy, cx))
        peaks.append(float(E[p]) / T)
    for it in range(iters):
        C = update_C(X, A, nonneg, C)
        A = update_A(X, A, C, S)
    C = update_C(X, A, nonneg, C)
    if log:
        log(f"    K0 {K0} free {A.shape[1] - K0} residual {float((X - A @ C).pow(2).sum() / X.pow(2).sum()):.3f} peaks {peaks[0]:.2f}..{peaks[-1]:.2f}")
    return A.T.reshape(-1, Y, X_).cpu().numpy(), C.cpu().numpy(), centers, peaks, X


def largest_cc(m, through=None):
    cc, n = ndi.label(m)
    if n == 0:
        return m
    if through is not None and cc[through] > 0:
        return cc == cc[through]
    return cc == 1 + np.argmax(ndi.sum(m, cc, range(1, n + 1)))


def disk_np(r):
    k = int(math.ceil(r))
    y, x = np.ogrid[-k:k + 1, -k:k + 1]
    return x * x + y * y <= r * r


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def additions(A, C, K0, pred_plane, peaks, theta=0.35, min_px=40, open_px=2.0, near_px=6.0):
    """The free components' places: each its pixels where it gives more light than any other component and at least `theta`
    of its most, opened by a disk (so that slivers between ROIs go), the piece with its peak (else the largest)."""
    rms = np.sqrt((C ** 2).mean(1))
    contrib = A * rms[:, None, None]
    owner = contrib.argmax(0)
    out = []
    for k in range(K0, len(A)):
        a = A[k]
        if a.max() <= 0:
            continue
        m = (a >= theta * a.max()) & (owner == k) & (pred_plane == 0)
        raw = int(m.sum())
        m = ndi.binary_opening(m, disk_np(open_px)) if open_px > 0 else m
        if m.sum() < min_px:
            continue
        m = ndi.binary_fill_holes(largest_cc(m, np.unravel_index(a.argmax(), a.shape))) & (pred_plane == 0)
        if m.sum() < min_px:
            continue
        # the components next to it (their masks, or places, within near_px), and how alike their time courses are
        near = ndi.distance_transform_edt(~m) <= near_px
        nb = [j for j in range(len(A)) if j != k and A[j].sum() > 0 and (A[j] * near).sum() > 0.2 * A[j].sum()]
        nbc = max([corr(C[k], C[j]) for j in nb if rms[j] > 0] or [0.0])
        inscribed = float(ndi.distance_transform_edt(m).max())
        out.append({"k": k, "kind": "add", "of": None, "eval_mask": m, "px": int(m.sum()), "raw_px": raw, "rms": float(rms[k]),
                    "peak": peaks[k - K0], "order": k - K0, "nb_corr": nbc, "inscribed": inscribed, "c": C[k]})
    return out


def nmf2(Xm, iters=30):
    """Rank 1 and rank 2 NMF of (n, T): returns sse1, sse2, A2 (n, 2), C2 (2, T)."""
    dev = Xm.device
    a = torch.clamp(Xm.mean(1), min=0)
    c = torch.clamp(a @ Xm / (a @ a + 1e-12), min=0)
    for _ in range(10):
        a = torch.clamp(Xm @ c / (c @ c + 1e-12), min=0)
        c = torch.clamp(a @ Xm / (a @ a + 1e-12), min=0)
    R = Xm - a[:, None] * c[None]
    sse1 = float(R.pow(2).sum())
    U, S_, V = torch.pca_lowrank(R, q=2, center=False)
    u = U[:, 0]
    if float((u > 0).float().mean()) in (0.0, 1.0):
        return sse1, sse1, None, None
    A = torch.stack([a * (u > 0), a * (u <= 0)], 1)
    C = update_C(Xm, A, True)
    S = torch.ones_like(A, dtype=torch.bool)
    for _ in range(iters):
        A = update_A(Xm, A, C, S)
        C = update_C(Xm, A, True, C)
    return sse1, float((Xm - A @ C).pow(2).sum()), A.cpu().numpy(), C.cpu().numpy()


def splits(X, pred_plane, min_px=40, open_px=2.0, min_frac=0.2, loose=False):
    """Per predicted mask: its two places by a rank 2 NMF of its pixels, where they are two (each one piece, large enough)."""
    Y, X_ = pred_plane.shape
    out = []
    for lab in np.unique(pred_plane):
        if lab == 0:
            continue
        m = pred_plane == lab
        n = int(m.sum())
        if n < 2 * min_px:
            continue
        idx = torch.as_tensor(np.flatnonzero(m.reshape(-1)), device=X.device)
        sse1, sse2, A, C = nmf2(X[idx])
        if A is None:
            continue
        rms = np.sqrt((C ** 2).mean(1))
        own = (A * rms).argmax(1)
        parts = []
        for j in (0, 1):
            p = np.zeros_like(m)
            p[m] = own == j
            parts.append(p)
        opened = [largest_cc(ndi.binary_opening(p, disk_np(open_px))) for p in parts]
        if min(o.sum() for o in opened) < max(min_px, min_frac * n):
            if not loose or min(p.sum() for p in parts) < min_px:
                continue
            opened = [largest_cc(p) for p in parts]
        connected = min(largest_cc(p).sum() / max(p.sum(), 1) for p in parts)
        # the rest of the mask to the nearer part
        d0, d1 = ndi.distance_transform_edt(~opened[0]), ndi.distance_transform_edt(~opened[1])
        p0 = m & (d0 <= d1)
        p1 = m & ~p0
        small = p1 if p1.sum() <= p0.sum() else p0
        out.append({"k": int(lab), "kind": "split", "of": int(lab), "eval_mask": small, "other_mask": m & ~small, "px": int(small.sum()),
                    "rms": float(rms.min()), "corr2": corr(C[0], C[1]), "gain": (sse1 - sse2) / max(sse1, 1e-12), "connected": float(connected),
                    "c": C[0] if (small is p0) else C[1], "c_other": C[1] if (small is p0) else C[0], "frac": float(small.sum() / n)})
    return out
