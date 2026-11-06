"""The version of the code the server runs, for the page's footer (`/api/health`).

In the Docker image, as its build found it (`VERSION.json`, written by ``python -m
app.version <path>`` in the Dockerfile's first stage, from the repository's ``.git``);
otherwise (a checkout, e.g. ``uvicorn --reload``), as git says now; else None.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config

FILE = config.ROOT / "VERSION.json"
# what the image is built of (the Dockerfile's COPYs), where a change not committed makes it differ from its commit
CODE = ("app", "config", "manage.py", "requirements.txt", "requirements-worker.txt", "requirements-alseg.txt", "Dockerfile")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(config.ROOT), *args], capture_output=True, text=True, timeout=10,
                          check=True).stdout.strip()


def from_git() -> Optional[dict]:
    """The commit checked out, from git: {commit (full), short, date (ISO), subject, dirty (the
    code differs from it: `CODE` changed, or files there not committed), url (on GitHub, if
    its origin is there)}; None if git cannot say."""
    try:
        commit, short, date, subject = _git("log", "-1", "--format=%H%x00%h%x00%cI%x00%s").split("\0", 3)
        dirty = bool(_git("status", "--porcelain", "--untracked-files=normal", "--", *CODE))
        origin = _git("config", "--get", "remote.origin.url")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    m = re.match(r"(?:git@github\.com:|https://github\.com/)(.+?)(?:\.git)?/?$", origin)
    return {"commit": commit, "short": short, "date": date, "subject": subject, "dirty": dirty,
            "url": f"https://github.com/{m.group(1)}/commit/{commit}" if m else None}


def read() -> Optional[dict]:
    """The version, as the image's build found it (with ``built``: when), else from git."""
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return from_git()


VERSION = read()  # (read once, as the server starts: before its sandbox, which may not let it read here)


if __name__ == "__main__":  # (the Dockerfile's: the version of the code built, kept with it)
    v = from_git()
    if v is not None:
        v["built"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    Path(sys.argv[1]).write_text(json.dumps(v))
