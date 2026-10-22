"""Tests of the folders the server may read and write (Admin tab). Run with: python -m pytest tests/"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from app import config, odormeta, sandbox, tensor
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)


@pytest.fixture(autouse=True)
def settings_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_FILE", tmp_path / "admin.json")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sandbox, "ACTIVE", None)
    monkeypatch.setattr(sandbox, "STATUS", {"settings": None, "started_at": None, "kernel": None})


def test_normalize():
    assert sandbox.normalize(" /mnt//tensor/./tom/ ") == "/mnt/tensor/tom"
    for bad, msg in [("mnt/tensor", "absolute"), ("~/x", "absolute"), ("/mnt/../etc", "'..'"), ("/", "whole"), ("//", "whole")]:
        with pytest.raises(sandbox.SandboxError, match=msg):
            sandbox.normalize(bad)


def test_check_folder(tmp_path):
    (tmp_path / "ro").mkdir()
    (tmp_path / "f").write_text("")
    assert sandbox.check_folder(f"{tmp_path}/ro/") == {"input": f"{tmp_path}/ro/", "path": f"{tmp_path}/ro", "ok": True,
                                                       "error": None, "warning": None, "fix": None}
    assert "not on the server" in sandbox.check_folder(f"{tmp_path}/nothing")["warning"]
    assert sandbox.check_folder(f"{tmp_path}/f")["ok"]  # a file alone may be read (as a reference dataset is)
    assert sandbox.check_folder(f"{tmp_path}/f", write=True)["error"] == "this is a file, not a folder"
    assert not sandbox.check_folder("relative")["ok"]
    if os.geteuid() != 0:  # root may write anywhere
        os.chmod(tmp_path / "ro", 0o555)
        try:
            assert sandbox.check_folder(str(tmp_path / "ro"))["warning"] is None
            assert "cannot write" in sandbox.check_folder(str(tmp_path / "ro"), write=True)["warning"]
        finally:
            os.chmod(tmp_path / "ro", 0o755)


def test_check_folder_mounts(tmp_path, monkeypatch):
    # what it takes for the server to see a folder it cannot, in Docker or the systemd unit
    monkeypatch.setenv("OLFDB_HOST_DIR", "/opt/olfdb")
    c = sandbox.check_folder("/local/tensor-archive/Betty/olfdb_backups", write=True, supervisor="Docker")
    assert c["ok"] and c["warning"] == "not inside the container: only folders mounted into it are"
    assert ("<code>- /local/tensor-archive/Betty/olfdb_backups:/local/tensor-archive/Betty/olfdb_backups</code> under "
            "<code>volumes:</code> in <code>/opt/olfdb/docker-compose.yml</code>, then run <code>docker compose up -d</code> "
            "in <code>/opt/olfdb</code>") in c["fix"]
    assert ":/local/x:ro</code>" in sandbox.check_folder("/local/x", supervisor="Docker")["fix"]  # to read only
    assert "BindPaths=-/local/x" in sandbox.check_folder("/local/x", write=True, supervisor="systemd")["fix"]
    assert "BindReadOnlyPaths=-/local/x" in sandbox.check_folder("/local/x", supervisor="systemd")["fix"]
    assert sandbox.check_folder("/srv/x", supervisor="systemd")["fix"] is None
    assert sandbox.check_folder("/srv/x")["fix"] is None and "not on the server" in sandbox.check_folder("/srv/x")["warning"]
    if os.geteuid() != 0:  # a folder there, but not writable: in a read-only mount, perhaps
        (tmp_path / "ro").mkdir()
        os.chmod(tmp_path / "ro", 0o555)
        try:
            fix = sandbox.check_folder(str(tmp_path / "ro"), write=True, supervisor="Docker")["fix"]
            assert fix.startswith("If it is in a folder mounted read-only (as tensor is): on the server, add") and "uid 10001" in fix
        finally:
            os.chmod(tmp_path / "ro", 0o755)


def test_settings_saved(tmp_path):
    assert sandbox.load() == {"read_dirs": [str(config.TENSOR_ROOT)], "write_dirs": [], "updated_at": None}
    checks, dirs = sandbox.check_folders({"read_dirs": ["/a/", "/a", " ", "b"], "write_dirs": ["/c"]})
    assert dirs == {"read_dirs": ["/a"], "write_dirs": ["/c"]} and not checks["read_dirs"][2]["ok"]
    sandbox.save(dirs)
    saved = sandbox.load()
    assert saved["read_dirs"] == ["/a"] and saved["write_dirs"] == ["/c"] and saved["updated_at"]
    config.ADMIN_FILE.write_text("{not json")
    assert sandbox.load()["read_dirs"] == [str(config.TENSOR_ROOT)]


def test_can_read_and_write(tmp_path):
    for d in ("tensor/tom", "tensor/jon", "backups", "elsewhere"):
        (tmp_path / d).mkdir(parents=True)
    assert sandbox.can_read(tmp_path / "elsewhere") and sandbox.can_write("/")  # not started: nothing refused
    sandbox.activate({"read_dirs": [str(tmp_path / "tensor/tom")], "write_dirs": [str(tmp_path / "backups")]})
    assert sandbox.can_read(tmp_path / "tensor/tom") and sandbox.can_read(tmp_path / "tensor/tom/raw/x.xml")
    assert sandbox.can_read(tmp_path / "backups/x") and sandbox.can_read(config.DATA_DIR / "olfdb.sqlite3")
    assert not sandbox.can_read(tmp_path / "tensor/jon") and not sandbox.can_read(tmp_path / "tensor/tomato")
    assert not sandbox.can_read(tmp_path / "tensor") and not sandbox.can_read(tmp_path / "elsewhere")
    os.symlink(tmp_path / "elsewhere", tmp_path / "tensor/tom/link")
    assert not sandbox.can_read(tmp_path / "tensor/tom/link/x")
    assert sandbox.can_write(tmp_path / "backups/x") and sandbox.can_write(config.DATA_DIR / "x")
    assert not sandbox.can_write(tmp_path / "tensor/tom/x")
    assert sandbox.readable_within(tmp_path / "tensor") == [tmp_path / "tensor/tom"]
    assert sandbox.readable_within(tmp_path / "tensor/tom") == [tmp_path / "tensor/tom"]


def test_tensor_only_where_allowed(root, tmp_path):
    flies = [{"date": "2023-11-19", "fly_num": 1}]
    dirs = {"raw_dirs": ["tom/raw"], "odor_dirs": ["tom/stim"]}
    sandbox.activate({"read_dirs": [str(root / "tom/raw")], "write_dirs": []})
    assert tensor.check_dir("tom/raw", root)["ok"]
    assert tensor.check_dir("tom/stim", root)["error"] == tensor.NOT_ALLOWED
    with pytest.raises(tensor.PathError, match="Admin tab"):
        tensor.file_path("tom/stim/abc_stim.yaml", root, (".yaml",))
    r = tensor.lookup(root, dirs, flies, [("ea", -4.0)])
    assert r["raw_data_path"] == "tom/raw/2023-11-19/1" and r["odor_yamls"]["candidates"] == []
    # tensor is there as long as the folders the server may read in it are not empty
    assert tensor.root_status(root)["ok"]
    sandbox.activate({"read_dirs": [str(root / "nobody")], "write_dirs": []})
    assert "unmounted" in tensor.root_status(root)["error"]
    sandbox.activate({"read_dirs": [str(tmp_path / "elsewhere")], "write_dirs": []})
    assert "Admin tab" in tensor.root_status(root)["error"]


def test_admin_api(main, root, tmp_path):  # noqa: F811
    state = call(main.admin_get)[1]
    assert state["saved"]["read_dirs"] == [str(root)] and state["active"] is None and not state["pending"]
    assert state["checks"]["read_dirs"][0]["ok"] and not state["restart"]["possible"]

    status, r = call(main.admin_check, body={"read_dirs": [str(root / "tom"), "tom"], "write_dirs": [str(tmp_path / "x")]})
    assert status == 200 and not r["ok"] and [c["ok"] for c in r["checks"]["read_dirs"]] == [True, False]
    assert "not on the server" in r["checks"]["write_dirs"][0]["warning"]

    # in effect: the settings the server started with
    sandbox.STATUS["settings"] = sandbox.load()
    body = {"read_dirs": [str(root / "tom") + "/"], "write_dirs": [str(tmp_path)]}
    status, r = call(main.admin_save, body={**body, "read_dirs": ["tom"]})
    assert status == 422 and "absolute" in r["errors"][0] and sandbox.load()["updated_at"] is None
    status, r = call(main.admin_save, body=body)
    assert status == 200 and r["ok"] and r["pending"] and r["saved"]["read_dirs"] == [str(root / "tom")]
    assert r["active"]["read_dirs"] == [str(root)] and r["saved"]["updated_at"]
    assert call(main.admin_save, body={"read_dirs": "/x"})[0] == 400

    status, r = call(main.admin_restart, body={})
    assert status == 409 and "by hand" in r["error"]


SHEET = "https://docs.google.com/spreadsheets/d/abc123/edit#gid=7"


def test_admin_odor_sheet(main, monkeypatch):  # noqa: F811
    read = []

    def fetch(url, timeout=20):
        read.append(url)
        if "broken" in url:
            raise odormeta.SheetError("could not read the Google sheet (403)")
        return config.ODOR_TEMPLATE.read_text()

    monkeypatch.setattr(odormeta, "fetch_sheet", fetch)
    monkeypatch.setattr(main, "sheet_watch", odormeta.SheetWatch(600))
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    assert call(main.admin_get)[1]["odor_sheet"] == {"url": None, "saved": False, "updated_at": None, "env": False}
    assert not call(main.odors_status)[1]["configured"]
    assert call(main.odors_preview, body={})[0] == 400

    status, r = call(main.admin_odor_sheet, body={"url": "https://example.com/sheet"})
    assert status == 422 and "not a Google sheet" in r["errors"][0] and main.db.get_setting("odor_sheet_url") is None
    status, r = call(main.admin_odor_sheet, body={"url": f" {SHEET} "})
    assert status == 200 and r["odor_sheet"]["url"] == SHEET and r["odor_sheet"]["saved"]
    assert r["read"]["error"] is None and r["read"]["n_rows"] > 50 and read == [SHEET]
    s = call(main.odors_status)[1]
    assert s["configured"] and s["url"] == SHEET and s["changed"] and len(read) == 1  # read once, when saved
    assert call(main.odors_preview, body={})[1]["source"] == "Google sheet"

    # saved even if it cannot be read
    status, r = call(main.admin_odor_sheet, body={"url": SHEET.replace("abc123", "broken")})
    assert status == 200 and "403" in r["read"]["error"] and "403" in call(main.odors_status)[1]["error"]

    # cleared: OLFDB_ODOR_SHEET_URL is used again, if set
    monkeypatch.setattr(config, "ODOR_SHEET_URL", SHEET)
    status, r = call(main.admin_odor_sheet, body={"url": ""})
    assert r["odor_sheet"] == {"url": SHEET, "saved": False, "updated_at": None, "env": True} and r["read"]["error"] is None


def test_syncs_not_tied_to_anyone(main, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "ODOR_PREVIEWS_DIR", tmp_path / "previews")
    config.ODOR_PREVIEWS_DIR.mkdir()
    monkeypatch.setattr(main, "sheet_watch", odormeta.SheetWatch(600))
    text = config.ODOR_TEMPLATE.read_text()
    # neither a CSV import nor a sync from the sheet needs a name, or keeps one given
    fp = main._preview(text, "file odors.csv")["fingerprint"]
    status, r = call(main.odors_apply, body={"fingerprint": fp, "applied_by": "Tom"})
    assert status == 200 and r["added"] > 50
    status, r = call(main.odors_apply, body={"fingerprint": main._preview(text, main.SHEET_SOURCE, SHEET)["fingerprint"]})
    assert status == 200
    syncs = main.db.odor_syncs()
    assert [s["source"] for s in syncs] == ["Google sheet", "file odors.csv"] and not any("applied_by" in s for s in syncs)
    with main.db.connect() as con:
        assert [r[0] for r in con.execute("SELECT applied_by FROM odor_syncs")] == ["", ""]


def test_lowercase_only_sync(main, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(config, "ODOR_PREVIEWS_DIR", tmp_path / "previews")
    config.ODOR_PREVIEWS_DIR.mkdir()
    text = config.ODOR_TEMPLATE.read_text()  # with names and abbreviations in capitals (IaA, HCl...)
    fp = main._preview(text, "file odors.csv")["fingerprint"]
    call(main.admin_odor_case, body={"lowercase_only": True})
    # new names with capitals are refused: at apply, if set since the preview; and at preview
    status, r = call(main.odors_apply, body={"fingerprint": fp})
    assert status == 422 and any("(hydrochloric acid): 'HCl'" in e for e in r["errors"])
    p = main._preview(text, "file odors.csv")
    assert not p["ok"] and any(e.endswith("(isoamyl acetate): 'IaA', 'IAA' have capital letters, and new odor names, abbreviations "
                                       "and aliases must be lowercase (set in the Admin tab)") for e in p["errors"])
    call(main.admin_odor_case, body={"lowercase_only": False})
    fp = main._preview(text, "file odors.csv")["fingerprint"]
    assert call(main.odors_apply, body={"fingerprint": fp})[0] == 200
    # then they are in the database, and allowed; only new ones are refused
    call(main.admin_odor_case, body={"lowercase_only": True})
    assert main._preview(text, "file odors.csv")["ok"]
    p = main._preview(text + "New Odor,new,,,,,,,,,,,\nnewer odor,nw,,,,,,,,,,,\n", "file odors.csv")
    assert [e.split(":")[0] for e in p["errors"]] == [f"row {len(text.splitlines()) + 1} (New Odor)"]


def test_admin_host(main, monkeypatch):  # noqa: F811
    r = call(main.admin_host)[1]
    assert r["name"] == socket.gethostname() and r["source"] == "server"
    assert not any(a.startswith(("127.", "::1", "fe80")) for a in r["addresses"])
    # in a Docker container: the server's name, as mounted in from its /etc/hostname
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv" or real_exists(p))
    monkeypatch.setattr(config, "HOST_HOSTNAME_FILE", Path(main.config.DATA_DIR) / "no-such-file")
    r = call(main.admin_host)[1]
    assert r["source"] == "container" and r["addresses"] == []
    hostname = config.ADMIN_FILE.with_name("hostname")
    hostname.write_text("localhost\n")
    monkeypatch.setattr(config, "HOST_HOSTNAME_FILE", hostname)
    assert call(main.admin_host)[1] == {"name": "localhost", "source": "mounted", "addresses": []}  # loopback left out


LANDLOCK_SCRIPT = """
import json, os, sys
from app import config, sandbox
config.ensure_dirs()
json.dump({"read_dirs": [sys.argv[1] + "/allowed", sys.argv[1] + "/single/f"], "write_dirs": [sys.argv[1] + "/out"]},
          open(config.ADMIN_FILE, "w"))
status = sandbox.start()

def can(f):
    try:
        f()
        return True
    except PermissionError:
        return False

t = sys.argv[1]
print(json.dumps({
    "kernel": status["kernel"],
    "read allowed": can(lambda: open(t + "/allowed/f").read()),
    "read elsewhere": can(lambda: open(t + "/denied/f").read()),
    "list elsewhere": can(lambda: os.listdir(t + "/denied")),
    "stat elsewhere": can(lambda: os.stat(t + "/denied/f")),
    "read a file allowed alone": can(lambda: open(t + "/single/f").read()),
    "read the file beside it": can(lambda: open(t + "/single/g").read()),
    "write out": can(lambda: open(t + "/out/f", "w").close()),
    "write allowed": can(lambda: open(t + "/allowed/g", "w").close()),
    "write data": can(lambda: open(os.path.join(os.environ["OLFDB_DATA_DIR"], "f"), "w").close()),
    "temporary file": can(lambda: __import__("tempfile").TemporaryFile().close()),
}))
"""


@pytest.mark.skipif(not sandbox.landlock_abi(), reason="the kernel has no Landlock")
def test_landlock(tmp_path):
    for d in ("allowed", "denied", "out", "single"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f").write_text("x")
    (tmp_path / "single" / "g").write_text("x")
    env = {**os.environ, "OLFDB_DATA_DIR": str(tmp_path / "data"), "OLFDB_LANDLOCK": "1"}
    out = subprocess.run([sys.executable, "-c", LANDLOCK_SCRIPT, str(tmp_path)], cwd=Path(__file__).parent.parent, env=env,
                         capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    assert r.pop("kernel")["enforced"]
    assert r == {"read allowed": True, "read elsewhere": False, "list elsewhere": False, "stat elsewhere": True,
                 "read a file allowed alone": True, "read the file beside it": False, "write out": True,
                 "write allowed": False, "write data": True, "temporary file": True}
