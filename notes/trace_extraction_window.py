"""The response window Extract signals would take in each recording of raw data folders of flies
(or of every fly of a date's folder): what `extract.window` works out, and from what (the odor
pulse's length, by the stimulus YAML the recording's notes name, and a volume's, by its
Experiment.xml), or why it cannot. Run where the site's packages are (its image), from the
repository, with OLFDB_DATA_DIR set to a scratch folder:

    PYTHONPATH=. python notes/trace_extraction_window.py <a fly's raw data folder, or a date's>..."""
import sys
from pathlib import Path

from app import extract

for arg in sys.argv[1:]:
    top = Path(arg)
    flies = [top] if any((p / "Experiment.xml").is_file() for p in top.iterdir() if p.is_dir()) else sorted(p for p in top.iterdir() if p.is_dir())
    for fly in flies:
        print(f"\n===== {fly}")
        for d in sorted(p for p in fly.iterdir() if (p / "Experiment.xml").is_file()):
            w = extract.window(d, extract.WINDOW)
            t = f"pulse {w['pulse_s']} s, volume {w['volume_s']:.4f} s" if not w["missing"] else (
                f"volume {w['volume_s']:.4f} s; " if w.get("volume_s") else "") + f"MISSING: {w['missing']}"
            print(f"  {d.name}: {w['volumes']} volumes (the pulse can reach {w['reach']}); {t}")
