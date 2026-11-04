"""The version of the code the server runs (app/version.py), as the page's footer shows it."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from app import config, version
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args], check=True,
                   capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_from_git(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "x.py").write_text("x = 1\n")
    (repo / "notes.md").write_text("notes\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "First: x")
    git(repo, "remote", "add", "origin", "git@github.com:lab/hong-data.git")
    monkeypatch.setattr(config, "ROOT", repo)
    v = version.from_git()
    assert v["subject"] == "First: x" and len(v["commit"]) == 40 and v["commit"].startswith(v["short"]) and not v["dirty"]
    assert v["url"] == f"https://github.com/lab/hong-data/commit/{v['commit']}" and v["date"][:2] == "20"
    # a change outside the code built does not make it differ; one in it (or a file not committed there) does
    (repo / "notes.md").write_text("more notes\n")
    assert not version.from_git()["dirty"]
    (repo / "app" / "y.py").write_text("y = 1\n")
    assert version.from_git()["dirty"]
    (repo / "app" / "y.py").unlink()
    (repo / "app" / "x.py").write_text("x = 2\n")
    assert version.from_git()["dirty"]
    git(repo, "remote", "set-url", "origin", "https://example.org/lab/hong-data.git")
    assert version.from_git()["url"] is None
    # (not a repository: none)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert version.from_git() is None


def test_read_built(tmp_path, monkeypatch, main):  # noqa: F811
    """In the image, as its build found it; and as /api/health gives it."""
    built = {"commit": "a" * 40, "short": "aaaaaaa", "date": "2026-11-04T01:35:12-08:00", "subject": "x", "dirty": True,
             "url": None, "built": "2026-11-04T09:36:20+00:00"}
    (tmp_path / "VERSION.json").write_text(json.dumps(built))
    monkeypatch.setattr(version, "FILE", tmp_path / "VERSION.json")
    assert version.read() == built
    monkeypatch.setattr(version, "VERSION", version.read())
    assert call(main.health)[1]["version"] == built
