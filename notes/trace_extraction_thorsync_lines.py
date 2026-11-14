"""What a fly's raw data folder has: each ThorImage recording's software, timing and notes, and each
ThorSync recording's lines (their names and number types). What showed, for
trace_extraction_notes.md (beside this), which line is the odor valve's on each setup and ThorImage
version (AI/olfDispPin, analog, on all looked at) and how the acquisition's lines are named.
Needs h5py, which the site's image does not have (pip install h5py):

    python notes/trace_extraction_thorsync_lines.py <fly's raw data folder>..."""
import re
import sys
from pathlib import Path

import h5py

for arg in sys.argv[1:]:
    fly = Path(arg)
    print(f"\n===== {fly}")
    for d in sorted(p for p in fly.iterdir() if p.is_dir()):
        xml = d / "Experiment.xml"
        h5 = d / "Episode001.h5"
        if xml.is_file():
            t = xml.read_text(errors="replace")
            get = lambda pat: (re.search(pat, t) or [None, None])[1]  # noqa: E731
            print(f"  image {d.name}: ThorImage {get(r'<Software version=\"([^\"]*)\"')}, {get(r'<Date date=\"([^\"]*)\"')}, "
                  f"frameRate {get(r'frameRate=\"([^\"]*)\"')}, steps {get(r'<ZStage[^>]*steps=\"([^\"]*)\"')}, "
                  f"flyback {get(r'flybackFrames=\"([^\"]*)\"')}, zFast {get(r'zFastEnable=\"([^\"]*)\"')}, average {get(r'averageMode=\"([^\"]*)\"')}/"
                  f"{get(r'averageNum=\"([^\"]*)\"')}, frames {get(r'<Streaming[^>]*frames=\"([^\"]*)\"')}, stimulusTriggering "
                  f"{get(r'stimulusTriggering=\"([^\"]*)\"')}, notes {(get(r'ExperimentNotes text=\"([^\"]*)\"') or '')[-60:]!r}")
        elif h5.is_file():
            with h5py.File(h5, "r") as f:
                lines = {f"{g}/{k}": f"{f[g][k].dtype}{f[g][k].shape}" for g in f.keys() for k in f[g].keys()}
            print(f"  sync  {d.name}: {lines}")
