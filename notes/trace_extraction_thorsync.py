"""A ThorSync recording's acquisition and odor valve signals: how many blocks of imaging, how many
odor pulses, how long each, and when each pulse is within its block. What showed, for
trace_extraction_notes.md (beside this), that imaging runs on through a recording (one block for
all its trials), and that the odor pulse is 3.0 s (pulses of a millisecond among them are noise
on the valve's line). Needs h5py, which the site's image does not have (pip install h5py):

    python notes/trace_extraction_thorsync.py <a ThorSync folder's Episode001.h5>...

e.g. /local/tensor/Tom/2p_data/raw_data/2023-05-08/1/SyncData001/Episode001.h5 (1.9 GB: a minute)."""
import sys

import h5py
import numpy as np

for path in sys.argv[1:]:
    print(f"\n===== {path}")
    with h5py.File(path, "r") as f:
        names = {g: list(f[g].keys()) for g in f.keys()}
        print({g: v for g, v in names.items()})
        gctr = f["Global/GCtr"][:, 0].astype(np.float64) / 20e6  # (ThorSync's 20 MHz clock)

        def edges(name):
            group = next(g for g, v in names.items() if name in v)
            x = f[f"{group}/{name}"][:, 0]
            hi = x > (x.max() / 2 if x.dtype.kind == "f" else 0)
            d = np.diff(hi.astype(np.int8))
            return gctr[np.nonzero(d == 1)[0] + 1], gctr[np.nonzero(d == -1)[0] + 1]

        scope_on, scope_off = edges("scopePin")
        odor_on, odor_off = edges("olfDispPin")
        frame_on, _ = edges("Frame Out")
    n = min(len(scope_on), len(scope_off))
    print(f"imaging (scope_pin): {len(scope_on)} blocks, each {np.round(np.unique(np.round(scope_off[:n] - scope_on[:n], 1)), 1)} s; "
          f"between them {np.round(np.unique(np.round(scope_on[1:n] - scope_off[:n - 1], 1)), 1)} s")
    m = min(len(odor_on), len(odor_off))
    dur = odor_off[:m] - odor_on[:m]
    print(f"odor valve (olf_disp_pin): {len(odor_on)} pulses, each {dur.min():.3f} to {dur.max():.3f} s; onset to onset "
          f"{np.round(np.unique(np.round(np.diff(odor_on), 1)), 1)} s")
    if len(scope_on):
        first = np.array([odor_on[(odor_on >= a) & (odor_on <= b)][0] - a for a, b in zip(scope_on[:n], scope_off[:n])
                          if ((odor_on >= a) & (odor_on <= b)).any()])
        per = [int(((odor_on >= a) & (odor_on <= b)).sum()) for a, b in zip(scope_on[:n], scope_off[:n])]
        print(f"odor pulses per block: {sorted(set(per))}; a block's start to its first pulse: {first.min():.2f} to {first.max():.2f} s")
    if len(frame_on) > 1:
        print(f"frames (frame_out): {len(frame_on)}, every {np.median(np.diff(frame_on)) * 1000:.1f} ms")
