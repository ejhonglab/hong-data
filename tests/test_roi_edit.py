"""Runs the tests of what the movie player edits a prediction's ROIs with (app/static/roi_edit.js:
tests/roi_edit.test.mjs), where there is a node to run them."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_roi_edit_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("no node")
    done = subprocess.run([node, str(Path(__file__).with_name("roi_edit.test.mjs"))], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
