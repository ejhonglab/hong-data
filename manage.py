#!/usr/bin/env python3
"""Server-side maintenance commands (run inside the container or the venv).

  python manage.py list                 list datasets
  python manage.py show <id>            print a dataset's metadata and README
  python manage.py delete <id>          delete a dataset (database rows + files)
  python manage.py export <id> <dir>    copy a dataset's files + a long CSV of responses to <dir>
  python manage.py cleanup-staging      remove expired staging uploads

With Docker:  docker compose exec olfdb python manage.py list
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from app import config
from app.db import Database


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    config.ensure_dirs()
    db = Database(config.DB_PATH)
    cmd, args = argv[0], argv[1:]
    if cmd == "list":
        for d in db.list_datasets():
            print(f"#{d['id']:<4} {d['last_fly_date']}  {d['title']}  ({d['n_flies']} flies, {d['n_units']} ROIs, {d['n_odors']} odors; {d['uploader_name']})")
        return 0
    if cmd == "show":
        d = db.get_dataset(int(args[0]))
        if d is None:
            print("not found"); return 1
        print(json.dumps(d["metadata"], indent=1)); print(); print(d["readme_md"])
        return 0
    if cmd == "delete":
        ds_id = int(args[0])
        if db.get_dataset(ds_id) is None:
            print("not found"); return 1
        answer = input(f"Delete dataset #{ds_id} and its files? [y/N] ")
        if answer.lower() != "y":
            return 1
        db.delete_dataset(ds_id)
        shutil.rmtree(config.DATASETS_DIR / str(ds_id), ignore_errors=True)
        with db.connect() as con:
            con.execute("VACUUM")
        print("deleted")
        return 0
    if cmd == "export":
        ds_id, out = int(args[0]), Path(args[1])
        d = db.get_dataset(ds_id)
        if d is None:
            print("not found"); return 1
        out.mkdir(parents=True, exist_ok=True)
        src = config.DATASETS_DIR / str(ds_id)
        for p in src.rglob("*"):
            if p.is_file():
                dest = out / p.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        import csv
        with db.connect() as con, open(out / "responses_long.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "fly_num", "roi", "panel", "odor", "repeat", "value"])
            for r in con.execute(
                """SELECT u.date, u.fly_num, u.roi, t.panel, t.odor, t.repeat, r.value
                   FROM responses r JOIN units u ON u.id = r.unit_id JOIN trials t ON t.id = r.trial_id
                   WHERE r.dataset_id = ? ORDER BY u.idx, t.idx""", (ds_id,)):
                w.writerow(r)
        print(f"exported to {out}")
        return 0
    if cmd == "cleanup-staging":
        from app.main import cleanup_stages
        cleanup_stages(); print("ok"); return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
