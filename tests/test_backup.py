"""Tests of backups made from the Admin tab. Run with: python -m pytest tests/"""
from __future__ import annotations

import os
import sqlite3
import time

import pytest

from app import backup, config, sandbox
from app.db import Database, content_fingerprint
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A data folder with a database and files of each kind, and a folder the server may write."""
    data, into = tmp_path / "data", tmp_path / "backups"
    for rel in ("datasets/1/data.csv", "datasets/1/README.md", "datasets/1/odor_yaml/a.yaml", "references/r1/table.csv",
                "deleted/2/data.csv", "odor_structures/176.png", "staging/s1/up.csv", "exports/x.zip", "tmp/t",
                "odor_previews/p.json"):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text(rel)
    into.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "ADMIN_FILE", data / "admin.json")
    sandbox.save({"read_dirs": [], "write_dirs": [str(into)]})
    monkeypatch.setattr(sandbox, "STATUS", {"settings": sandbox.load(), "started_at": None, "kernel": None})
    monkeypatch.setattr(sandbox, "ACTIVE", None)
    sandbox.activate(sandbox.STATUS["settings"])
    monkeypatch.setattr(backup, "JOB", dict(backup.JOB))
    db = Database(data / "olfdb.sqlite3")
    db.save_user("Ann", {"raw_dirs": ["ann/raw"]}, when="2026-10-19T10:00:00")
    return db, data, into


def test_backup(site):
    db, data, into = site
    s = backup.status(db)
    assert s["folders"] == [{"path": str(into), "ok": True, "error": None, "backups": []}] and s["last"] is None
    assert s["current"] is None and s["pending"] == []

    first = backup.make(db, into)
    b = into / first["name"]
    kept = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    assert kept == ["admin.json", "backup.json", "datasets/1/README.md", "datasets/1/data.csv", "datasets/1/odor_yaml/a.yaml",
                    "deleted/2/data.csv", "odor_structures/176.png", "olfdb.sqlite3", "references/r1/table.csv"]
    assert (b / "datasets/1/README.md").read_text() == "datasets/1/README.md"
    with sqlite3.connect(b / "olfdb.sqlite3") as con:  # the database as it was, whole
        assert con.execute("SELECT name FROM users").fetchall() == [("Ann",)]
        assert content_fingerprint(con) == db.fingerprint() == first["database"]["fingerprint"]
    assert first["n_files"] == 8 and first["linked"] == 0 and first["linked_to"] is None
    s = backup.status(db)
    assert s["last"]["name"] == first["name"] and s["last"]["folder"] == str(into) and "files" not in s["last"]
    assert s["current"] == {"ok": True, "database": False, "added": 0, "changed": 0, "removed": 0}

    # not current once anything kept changes; the record of files read and written is not that
    db.record_files({"/x": 1}, when=None)
    assert backup.status(db)["current"]["ok"]
    (data / "datasets/3").mkdir()
    (data / "datasets/3/data.csv").write_text("new")
    (data / "datasets/1/README.md").write_text("edited")
    os.utime(data / "datasets/1/README.md", ns=(0, 10**18))
    (data / "deleted/2/data.csv").unlink()
    db.save_user("Bob", {"raw_dirs": ["bob/raw"]}, when="2026-10-19T11:00:00")
    assert backup.status(db)["current"] == {"ok": False, "database": True, "added": 1, "changed": 1, "removed": 1}

    # the next backup links to the one before what did not change
    time.sleep(1.1)  # named by the second
    second = backup.make(db, into)
    c = into / second["name"]
    assert os.stat(c / "datasets/1/data.csv").st_ino == os.stat(b / "datasets/1/data.csv").st_ino
    assert os.stat(c / "datasets/1/README.md").st_ino != os.stat(b / "datasets/1/README.md").st_ino
    assert (c / "datasets/1/README.md").read_text() == "edited" and not (c / "deleted/2/data.csv").exists()
    assert second["linked"] == 5 and second["linked_to"] == first["name"]  # all but README.md and the new file
    assert second["copied_bytes"] == second["database"]["bytes"] + len("edited") + len("new")
    s = backup.status(db)
    assert [x["name"] for x in s["folders"][0]["backups"]] == [second["name"], first["name"]] and s["current"]["ok"]

    # one left unfinished is listed so, and removed by the next
    (into / f"{backup.PREFIX}2020-01-01_000000{backup.PARTIAL}").mkdir()
    assert backup.status(db)["folders"][0]["backups"][-1] == {"name": f"{backup.PREFIX}2020-01-01_000000.partial", "complete": False}
    time.sleep(1.1)
    backup.make(db, into)
    assert not (into / f"{backup.PREFIX}2020-01-01_000000{backup.PARTIAL}").exists()


def test_backup_api(site, main, monkeypatch):  # noqa: F811
    db, data, into = site
    monkeypatch.setattr(main, "db", db)
    status, r = call(main.admin_backup_start, body={"folder": "/elsewhere"})
    assert status == 409 and "not one of the folders" in r["error"]
    status, r = call(main.admin_backup_start, body={"folder": str(into)})
    assert status == 200
    for _ in range(100):
        if not backup.JOB["running"]:
            break
        time.sleep(0.05)
    r = call(main.admin_backups)[1]
    assert r["job"]["error"] is None and r["last"]["name"] == r["job"]["name"] and r["current"]["ok"]
    assert r["job"]["done"] == r["job"]["total"] == 7

    # a folder saved, but not in effect until the server restarts
    sandbox.save({"read_dirs": [], "write_dirs": [str(into), str(data / "later")]})
    r = call(main.admin_backups)[1]
    assert r["pending"] == [str(data / "later")] and [f["path"] for f in r["folders"]] == [str(into)]
    os.chmod(into, 0o555)
    try:
        if os.geteuid() != 0:  # root may write anywhere
            assert call(main.admin_backups)[1]["folders"][0]["error"] == "the server may not write it"
            assert "may not write" in call(main.admin_backup_start, body={"folder": str(into)})[1]["error"]
    finally:
        os.chmod(into, 0o755)
