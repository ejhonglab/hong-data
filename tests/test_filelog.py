"""Tests of the record of files the server reads and writes in its own folders (Admin tab). Run with: python -m pytest tests/"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest

from app import config, filelog, sandbox
from app.db import Database
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A folder the server runs from (with a git repository holding one file) and its data folder in it."""
    home = tmp_path / "olfdb"
    (home / "data").mkdir(parents=True)
    (home / "conf.txt").write_text("committed")
    (home / "notes.txt").write_text("not committed")
    (home / "data" / "old.txt").write_text("there before")
    subprocess.run(["git", "init", "-q", str(home)], check=True)
    subprocess.run(["git", "-C", str(home), "add", "conf.txt"], check=True)
    monkeypatch.setattr(config, "ROOT", home)
    monkeypatch.setattr(config, "DATA_DIR", home / "data")
    monkeypatch.setattr(config, "ADMIN_FILE", home / "data" / "admin.json")
    monkeypatch.setattr(sandbox, "STATUS", {"settings": None, "started_at": None, "kernel": None})
    # taken back after the test, so the hook (which stays) notes nothing
    for name, value in [("_scope", ()), ("_skip", ()), ("_known", {}), ("_pending", {})]:
        monkeypatch.setattr(filelog, name, value)
    return home


def by_path(listing):
    return {f["path"]: f for f in listing["files"]}


def test_file_log(server, tmp_path):
    data = server / "data"
    db = Database(data / "olfdb.sqlite3")
    filelog.start(db)
    assert set(by_path(filelog.listing(db))) == {"data/old.txt", "data/olfdb.sqlite3"}  # there before it began

    (data / "new.txt").write_text("x")
    (data / "old.txt").read_text()
    (server / "conf.txt").read_text()  # the repository's
    (server / "notes.txt").read_text()
    (tmp_path / "elsewhere.txt").write_text("x")  # not in its folders
    with pytest.raises(OSError):
        open(data / "nope")
    with tempfile.NamedTemporaryFile(dir=data, prefix="t_", delete=False) as t:
        pass
    os.remove(t.name)
    (data / "a").write_text("x")
    os.replace(data / "a", data / "b")
    (data / "d1").mkdir()
    (data / "d1" / "f").write_bytes(b"12")
    shutil.move(str(data / "d1"), data / "d2")
    os.close(os.open(data / "fd.txt", os.O_WRONLY | os.O_CREAT))
    (server / "shared").mkdir()
    (server / "shared" / "x.txt").write_text("x")
    (server / "shared" / "x.txt").read_text()
    sqlite3.connect(data / "other.db").close()

    files = by_path(filelog.listing(db, exclude=[str(server / "shared"), str(server)]))
    assert files["data/old.txt"] == {"path": "data/old.txt", "read": True, "wrote": True, "size": 12, "first_at": None}
    assert files["data/new.txt"]["wrote"] and not files["data/new.txt"]["read"] and files["data/new.txt"]["first_at"]
    assert files[f"data/{os.path.basename(t.name)}"]["size"] is None  # deleted since
    assert files["data/a"]["size"] is None and files["data/b"]["size"] == 1
    assert files["data/d2/f"] == {**files["data/d2/f"], "wrote": True, "size": 2}
    assert files["data/fd.txt"]["wrote"] and files["data/other.db"]["read"] and files["notes.txt"]["read"]
    assert "conf.txt" not in files and "data/nope" not in files and "shared/x.txt" not in files
    assert not any("elsewhere" in p for p in files)

    # saved: seen again as the server starts again
    filelog._known.clear()
    filelog.start(db)
    assert by_path(filelog.listing(db)).keys() == files.keys() | {"shared/x.txt"}

    # without git to tell, all in its folder but the data folder is taken to be the repository's
    os.rename(server / ".git", server / "git")
    assert not [p for p in by_path(filelog.listing(db)) if not p.startswith("data/")]


def test_record_files(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    db.record_files({"/a": 1, "/b": 2}, when="2026-10-19T10:00:00")
    db.record_files({"/a": 2, "/b": 1, "/c": 3}, when="2026-10-19T11:00:00")
    assert db.file_log() == [{"path": "/a", "read": 1, "wrote": 1, "first_at": "2026-10-19T10:00:00"},
                             {"path": "/b", "read": 1, "wrote": 1, "first_at": "2026-10-19T10:00:00"},
                             {"path": "/c", "read": 1, "wrote": 1, "first_at": "2026-10-19T11:00:00"}]


def test_server_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", Path("/srv/olfdb"))
    monkeypatch.setattr(config, "DATA_DIR", Path("/data"))
    assert filelog.server_paths()[0]("/data/x") == "/data/x"  # not in Docker
    monkeypatch.setattr(filelog, "_in_docker", lambda: True)
    monkeypatch.setattr(filelog, "MOUNTINFO", tmp_path / "mountinfo")
    monkeypatch.delenv("OLFDB_HOST_DIR", raising=False)
    assert "OLFDB_HOST_DIR" in filelog.server_paths()[1]

    def mounted(root, host):
        (tmp_path / "mountinfo").write_text(f"1 0 0:1 / / rw - overlay overlay rw\n"
                                            f"2 1 8:2 {root} /data rw,relatime - ext4 /dev/sda2 rw\n")
        monkeypatch.setenv("OLFDB_HOST_DIR", host)
        return filelog.server_paths()

    to_server, note = mounted("/opt/olfdb/data", "/opt/olfdb")
    assert note is None and to_server("/data/x/y") == "/opt/olfdb/data/x/y" and to_server("/srv/olfdb/app") == "/opt/olfdb/app"
    assert to_server("/datax") == "/datax"
    # the data folder in a file system mounted at /opt, or in a btrfs subvolume
    assert mounted("/olfdb/data", "/opt/olfdb")[0]("/data") == "/opt/olfdb/data"
    assert mounted("/@/opt/olfdb/data", "/opt/olfdb/")[0]("/data") == "/opt/olfdb/data"
    assert mounted("/opt/my\\040olfdb/data", "/opt/my olfdb")[1] is None
    # docker compose run from a folder in it (finding docker-compose.yml above), or from elsewhere with -f
    assert "does not hold" in mounted("/opt/olfdb/data", "/opt/olfdb/data")[1]
    assert "does not hold" in mounted("/opt/olfdb/data", "/home/tom")[1]
    (tmp_path / "mountinfo").write_text("1 0 0:1 / / rw - overlay overlay rw\n")
    assert "not a folder of the server" in filelog.server_paths()[1]


def test_admin_files(main, server, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "REFERENCE_DIR", server / "reference_data")
    monkeypatch.setattr(main, "db", Database(server / "data" / "olfdb.sqlite3"))
    filelog.start(main.db)
    (server / "reference_data").mkdir()
    (server / "reference_data" / "mine.csv").write_text("x")
    (server / "reference_data" / "mine.csv").read_text()
    (server / "data" / "backups").mkdir()
    (server / "data" / "backups" / "b").write_text("x")
    (server / "data" / "kept").write_text("x")
    sandbox.save({"read_dirs": ["/local/tensor"], "write_dirs": [str(server / "data" / "backups")]})
    status, r = call(main.admin_files)
    assert status == 200 and r["root"] == str(server) and r["note"] is None and r["since"]
    files = by_path(r)
    assert files.keys() == {"data/old.txt", "data/olfdb.sqlite3", "data/olfdb.sqlite3-wal", "data/olfdb.sqlite3-shm",
                            "data/admin.json", "data/admin.tmp", "data/kept"}
    assert files["data/admin.tmp"]["size"] is None  # written, then renamed admin.json
    state = call(main.admin_get)[1]
    assert state["data_dir"] == str(server / "data") and not state["in_container"]
