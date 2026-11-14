"""Each trial's odor frames, from a recording's ThorSync data: the volumes its odor pulse (the valve's
signal, `olf_disp_pin`) reaches, against the first odor frame hong2p assigns and the one in
al_analysis' trials. What showed, for trace_extraction_notes.md (beside this) and the response
window of Extract signals (`extract.window_volumes`): that the valve's pulses are found the same
at hong2p's 4.5 V as at the middle of the line's two levels; that hong2p's assignment of frames is
al_analysis' `trial_frames_and_odors.json`, frame for frame; that the valve opens zero to one volume
before the middle of the first odor volume; how many volumes from that one each trial's pulse
reaches, against floor(pulse / volume + 1/2) + 1; and how much of a response is already in the
volume before the first odor volume (little: 0 to 4% of its peak).

Reads all of a recording's ThorSync lines it needs (1 to 2 GB: 10 s to a few minutes each). A
recording is paired with the ThorSync folder started nearest to it in time, whole hours aside (one
clock may be UTC); where that fails, name it: <part of the recording's name>:<SyncDataNNN>. Needs
h5py, which the site's image does not have, and hong2p, which it has; from the repository:

    docker run --rm --memory 8g --memory-swap 8g --pids-limit 512 --user root -v "$PWD":/srv/olfdb:ro \\
        -v /local/tensor:/local/tensor:ro -w /srv/olfdb -e PYTHONPATH=/srv/olfdb -e OLFDB_DATA_DIR=/tmp/x --entrypoint sh olfdb:latest \\
        -c "pip install -q h5py && python notes/trace_extraction_odor_frames.py <fly's raw data folder> [part of a recording's name]..."

e.g. /local/tensor/Tom/2p_data/raw_data/2023-05-08/1 megamat1"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from hong2p import thor

from app import extract, movies

warnings.simplefilter("ignore")
fly = Path(sys.argv[1])
wanted = sys.argv[2:]

def paired(fly: Path) -> list:
    """Each recording of the fly with the ThorSync folder started nearest in time to it (within two minutes)."""
    images = [d for d in sorted(fly.iterdir()) if (d / "Experiment.xml").is_file()]
    syncs = {d: thor.get_thorsync_time(d) for d in sorted(fly.iterdir()) if (d / "Episode001.h5").is_file()}
    out = []
    for d in images:
        named = next((w.split(":")[1] for w in wanted if ":" in w and w.split(":")[0] in d.name), None)
        if named:  # (given, as <part of the recording's name>:<its ThorSync folder>)
            out.append((d, fly / named))
            continue
        at = thor.get_thorimage_time(thor.get_thorimage_xmlroot(d))
        # (the two clocks may be whole hours apart: one is the computer's time, one UTC)
        off = lambda s: abs(((syncs[s] - at).total_seconds() + 1800) % 3600 - 1800)  # noqa: E731
        best = min(syncs, key=off, default=None)
        if best is not None and off(best) < 120:
            out.append((d, best))
    return out


pairs = paired(fly)
wanted = [w.split(":")[0] for w in wanted]
print(f"\n===== {fly}: {len(pairs)} recordings paired with their ThorSync data")
for image, sync in pairs:
    image, sync = Path(image), Path(sync)
    if wanted and not any(w in image.name for w in wanted):
        continue
    t0 = time.time()
    try:
        names = thor.load_thorsync_hdf5(sync, return_dataset_names_only=True)
        print(f"    (its ThorSync lines: {names})")
        # (only what frames and odors are assigned by: these files have tens of millions of samples a line)
        df = thor.load_thorsync_hdf5(sync, datasets=["gctr", "frame_out", "olf_disp_pin", "scope_pin"])
    except Exception as e:  # noqa: BLE001
        print(f"--- {image.name} ({sync.name}): could not read its ThorSync data: {type(e).__name__}: {e}")
        continue
    print(f"--- {image.name} ({sync.name}; read in {time.time() - t0:.0f} s): ThorImage {thor.get_thorimage_version(thor.get_thorimage_xmlroot(image))}; "
          f"lines {[c for c in df.columns if c != 'time_s']}")
    # the valve's line: which lines have pulses at all, and of what levels
    for col in df.columns:
        if "olf" in col or "valve" in col or "odor" in col:
            v = df[col].to_numpy()
            print(f"    {col}: {v.dtype}, from {v.min():.2f} to {v.max():.2f}; levels (1st, 50th, 99.9th percentile) "
                  f"{np.percentile(v[::50], [1, 50, 99.9]).round(2).tolist()}")
    try:
        frame_times = thor.get_frame_times(df, image)
        bounds = thor.assign_frames_to_odor_presentations(df, image)
        on, off = thor.get_col_onset_offset_times(df, "olf_disp_pin")
    except Exception as e:  # noqa: BLE001
        print(f"    hong2p could not assign its frames: {type(e).__name__}: {str(e)[:300]}")
        continue
    blocks = thor.assign_frame_times_to_blocks(frame_times)
    dt = float(np.median(np.diff(frame_times)))
    # a pulse as hong2p finds it (over 4.5 V), and as a cleaner rule would: over the middle of the line's two levels, lasting 50 ms or more
    v = df["olf_disp_pin"].to_numpy()
    lo, hi = np.percentile(v[::50], [1, 99.9])
    high = v > (lo + hi) / 2
    edges = np.flatnonzero(np.diff(high.astype(np.int8)))
    t = df["time_s"].to_numpy()
    if high[0]:
        edges = edges[1:]
    every = [(t[a + 1], t[b + 1]) for a, b in zip(edges[::2], edges[1::2])]
    imaged = lambda a: frame_times[0] - dt / 2 <= a <= frame_times[-1] + dt / 2  # noqa: E731
    pulses = [(a, b) for a, b in every if b - a >= 0.05 and imaged(a)]
    outside = [(round(a - frame_times[0], 1), round(b - a, 3)) for a, b in every if b - a >= 0.05 and not imaged(a)]
    short = [round(b - a, 4) for a, b in every if b - a < 0.05]
    kept = np.array([imaged(a) for a in on])
    print(f"    {len(frame_times)} volumes of {dt:.3f} s in {len(blocks)} block(s) of imaging; {len(bounds)} trials; pulses over 4.5 V: {len(on)} "
          f"({kept.sum()} while imaging), of {np.round(np.unique(np.round((off - on)[kept], 2)), 2).tolist()} s; by the middle of its levels "
          f"({(lo + hi) / 2:.2f} V), 50 ms or more, while imaging: {len(pulses)}; shorter ones {short}; others not while imaging "
          f"(s from the first volume, s long): {outside}")
    mine_on, mine_off = np.array([a for a, _ in pulses]), np.array([b for _, b in pulses])
    on, off = on[kept], off[kept]
    if len(on) != len(bounds) or len(pulses) != len(bounds):
        print("    (as many pulses as trials are needed to go on)")
        continue
    print(f"    the two ways to find a pulse agree to within {np.abs(mine_on - on).max() * 1000:.2f} ms (its start) and "
          f"{np.abs(mine_off - off).max() * 1000:.2f} ms (its end)")
    k = np.array([b[1] for b in bounds])
    delta = frame_times[k] - on  # from the valve's opening to the middle of the first odor volume
    # the volumes with part of the pulse in them: volume j is from frame_times[j] - dt / 2 to frame_times[j] + dt / 2
    first = np.searchsorted(frame_times + dt / 2, on)  # the volume the pulse begins in
    last = np.searchsorted(frame_times - dt / 2, off) - 1  # the one it ends in
    n_from_k = last - k + 1
    print(f"    valve open to the middle of the first odor volume: {delta.min() / dt:.2f} to {delta.max() / dt:.2f} volumes (should be 0 to 1); "
          f"the pulse begins in the volume before the first odor volume in {np.mean(first < k) * 100:.0f}% of trials, in it in {np.mean(first == k) * 100:.0f}%; "
          f"volumes from the first odor volume to the pulse's end: {dict(zip(*np.unique(n_from_k, return_counts=True)))}; "
          f"at most by the pulse and the volume's lengths: {int(np.floor(np.median(off - on) / dt + 0.5)) + 1}")
    # against al_analysis' own
    trials = Path(str(image).replace("/raw_data/", "/analysis_intermediates/")) / "trial_frames_and_odors.json"
    if trials.is_file():
        theirs = [(x["start_frame"], x["first_odor_frame"], x["end_frame"]) for x in json.loads(trials.read_text())]
        mine = [tuple(int(x) for x in b) for b in bounds]
        same = theirs == mine
        print(f"    al_analysis' {trials.name}: {'the same frames' if same else 'DIFFERENT frames'}"
              + ("" if same else f" (theirs {theirs[:3]}, hong2p's now {mine[:3]}; trials that differ: {sum(a != b for a, b in zip(theirs, mine))} of {len(mine)})"))
    else:
        print(f"    (no {trials})")
        continue
    # whether a response has begun in the volume before the first odor volume, by when in it the valve opened
    rec = trials.parent
    movie_path = next((rec / n for n in movies.PREFERRED if (rec / n).is_file()), None)
    roiset = extract.own_roiset(rec, rec.parent)
    if movie_path is None or roiset is None:
        print("    (no movie or ROIs of its own to look at its responses with)")
        continue
    movie = movies.open_movie(movie_path)
    s = movie.shape
    rois = extract.read_rois(roiset, (s["z"], s["y"], s["x"]), lambda x: None)
    tr_list = extract.read_trials(trials)
    r = {"traces": extract.plane_traces(movie, rois), "trials": tr_list}
    extract.combine("best_plane", rois, [r])
    tr = r["roi_traces"]
    early, frac = [], []
    for i, x in enumerate(tr_list):
        pre = tr[x["start"]:x["onset"]]
        if len(pre) < 2:
            continue
        base = pre[:-1].mean(axis=0)
        peak = tr[x["onset"]:x["onset"] + 3].max(axis=0)
        strong = (peak - base) / base > 0.5
        if strong.sum() >= 3:
            early.append(float(np.median(((pre[-1] - base) / (peak - base))[strong])))
            frac.append(float(delta[i] / dt))
    early, frac = np.array(early), np.array(frac)
    if len(early) >= 6:
        late, soon = early[frac < 0.5], early[frac >= 0.5]  # the valve opened in the first odor volume itself; in the one before it
        print(f"    the volume before the first odor volume, as a share of the response's peak (ROIs responding over 0.5; {len(early)} trials): "
              f"{np.median(soon):+.2f} where the valve opened in it (n = {len(soon)}), {np.median(late):+.2f} where it opened after it (n = {len(late)}); "
              f"against how early in it the valve opened, r = {np.corrcoef(frac, early)[0, 1]:+.2f}")
