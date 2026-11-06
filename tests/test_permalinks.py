"""Tests of permalinks (app/permalinks.py): names of datasets that stay theirs whatever their
numbers, orphaned with their dataset deleted until an upload like it takes them over. Run
with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from starlette.requests import Request

from app import config, odormeta
from app.permalinks import compare, slug_problem, slugify, unique
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_tensor import call, main, root, upload  # noqa: F401 (fixtures)

MD = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "GH146", "coding_unit": "glomeruli"}
SLUG = "p1-ann-glomeruli-2024-01-02"


def test_slugs():
    assert slugify("p1_ann_glomeruli_drug-CNO_starved_2024-01-02") == "p1-ann-glomeruli-drug-cno-starved-2024-01-02"
    assert slugify("  Ünïcode  x__y+z ") == "unicode-x-y-z"
    long = slugify("word-" * 30)
    assert len(long) <= 80 and long.endswith("word") and slug_problem(long) is None
    assert slug_problem("megamat-tom-2") is None
    for bad, why in [("", "give it"), ("Megamat", "only lowercase"), ("a_b", "only"), ("-a", "only"), ("a--b", "only"),
                     ("2024-01-02", "needs a letter"), ("x" * 81, "at most 80")]:
        assert why in slug_problem(bad), bad
    assert unique("a", {"a", "a-2"}) == "a-3" and unique("a", set()) == "a"


def test_compare():
    old = {**MD, "manipulations": [], "panels": ["p1"], "flies": [["2024-01-01", 1], ["2024-01-02", 1]],
           "trials": [["p1", "va @ -3", 0], ["p1", "va @ -3", 1]]}
    assert compare(old, {**old, "uploader_name": " ann ", "coding_unit": "Glomeruli"})["exact"]
    c = compare(old, {**old, "flies": [["2024-01-01", 1], ["2024-01-03", 1]], "panels": ["p1", "p2"]})
    assert c["differs"] == ["panels"] and c["related"] and not c["same"] and c["flies"]["added"] == [("2024-01-03", 1)]
    assert not compare(old, {**old, "driver": "pebbled"})["related"]
    # KCs however written are one coding unit; manipulations in any order and case
    kc = {**old, "coding_unit": "KCs", "manipulations": [{"key": "drug", "value": "CNO"}, {"key": "starved", "value": None}]}
    assert compare(kc, {**kc, "coding_unit": "kenyon cells", "manipulations": [{"key": "Starved"}, {"key": "DRUG", "value": "cno"}]})["exact"]


@pytest.fixture
def env(main, tmp_path, monkeypatch):  # noqa: F811
    """Uploads (staged, their README asked for, saved) and the endpoints of permalinks."""
    for k, sub in (("STAGING_DIR", "staging"), ("DATASETS_DIR", "datasets"), ("DELETED_DIR", "deleted")):
        monkeypatch.setattr(config, k, tmp_path / sub)
    monkeypatch.setattr(config, "ODOR_SHEET_URL", "")
    rows, _, _ = odormeta.parse_sheet("name,abbrev\nvaleric acid,va\nethyl acetate,ea\n")
    main.db.apply_odor_sync(added=odormeta.diff(rows, [])["added"], changed=[], deleted=[], source="test",
                            fingerprint="0" * 64, details={}, when="2026-10-19T10:00:00")
    n = iter(range(1000))

    class Env:
        def stage(self, cols=COLS, rows=ROWS, values=None):
            v = values if values is not None else np.arange(len(rows) * len(cols), dtype=float).reshape(len(rows), len(cols)) + next(n)
            status, r = upload(main, write_csv(make_wide(rows, cols, v), tmp_path, f"x{next(n)}.csv"))
            assert status == 200, r
            return {"stage_id": r["stage"]["stage_id"]}

        def readme(self, st, **md):
            return call(main.stage_readme, body={"metadata": {**MD, **md}}, path_params=st)[1]

        def commit(self, st, proceed=None, **md):
            return call(main.stage_commit, body={"metadata": {**MD, **md}, "readme": "#", "permalink_proceed": proceed},
                        path_params=st)[1]

        def name(self, ds_id, slug):
            return call(main.dataset_permalink_save, body={"slug": slug}, path_params={"ds_id": ds_id})

        def dataset(self, ds_id):
            return call(main.dataset_get, path_params={"ds_id": ds_id})[1]

        def link(self, slug):
            return call(main.permalink_get, path_params={"slug": slug})

        def delete(self, ds_id):
            return call(main.admin_dataset_delete, path_params={"ds_id": ds_id})[1]

        def edit(self, ds_id, proceed=None, **md):
            base = main.db.get_dataset(ds_id)["metadata"]
            return call(main.dataset_metadata_save, body={"metadata": {**base, **md}, "base": base, "permalink_proceed": proceed},
                        path_params={"ds_id": ds_id})
    return Env()


def test_naming(main, env):  # noqa: F811
    first = env.commit(env.stage())
    ds = first["dataset_id"]
    d = env.dataset(ds)
    assert d["permalink"] is None and d["default_permalink"] == SLUG and first["permalink"] is None
    status, r = env.name(ds, "P1 ann")
    assert status == 422 and "only lowercase" in r["errors"][0]
    status, r = env.name(ds, SLUG)
    assert status == 200 and r["permalink"]["slug"] == SLUG and not r["permalink"]["custom"]
    assert r["permalink"]["summary"] == "#1, p1: GH146 > GCaMP6f (glomeruli): uploaded by Ann, 3 flies (2024-01-01 to 2024-01-02), 4 ROIs, 3 odors, 6 trials"
    status, r = env.link(SLUG)
    assert status == 200 and r["dataset_id"] == ds and r["orphaned_at"] is None
    assert env.link("nope")[0] == 404
    # followed: to Browse, which looks it up
    scope = {"type": "http", "method": "GET", "path": f"/d/{SLUG}", "query_string": b"", "headers": [], "path_params": {"slug": SLUG}}
    response = asyncio.run(main.permalink_follow(Request(scope)))
    assert response.status_code == 302 and response.headers["location"] == f"/#browse/{SLUG}"
    # the list says which datasets have one; another's name is taken, as is an orphaned one's
    assert [x["permalink"] for x in call(main.datasets_list)[1]["datasets"]] == [SLUG]
    other = env.commit(env.stage(), manipulations=[{"key": "drug", "value": "CNO"}])["dataset_id"]
    assert env.dataset(other)["default_permalink"] == "p1-ann-glomeruli-drug-cno-2024-01-02"
    status, r = env.name(other, SLUG)
    assert status == 409 and r["errors"] == [f"'{SLUG}' is dataset #{ds}'s already"]
    # renamed: custom, kept whatever its details
    status, r = env.name(ds, "first-data")
    assert status == 200 and r["permalink"]["custom"] and env.link(SLUG)[0] == 404
    status, r = env.edit(ds, uploader_name="Bea")
    assert status == 200 and r["permalink"] == "first-data"


def test_default_followed_on_edit(main, env):  # noqa: F811
    ds = env.commit(env.stage())["dataset_id"]
    env.name(ds, SLUG)
    # details that do not change its default: saved
    status, r = env.edit(ds, notes="more")
    assert status == 200 and r["permalink"] == SLUG
    # one that does: Proceed first, then renamed
    status, r = env.edit(ds, uploader_name="Bea")
    assert status == 409 and r["permalink_change"] == {"from": SLUG, "to": "p1-bea-glomeruli-2024-01-02"} and not r["ok"]
    assert main.db.get_dataset(ds)["uploader_name"] == "Ann"
    status, r = env.edit(ds, uploader_name="Bea", proceed="p1-bea-glomeruli-2024-01-02")
    assert status == 200 and r["permalink"] == "p1-bea-glomeruli-2024-01-02"
    p = env.dataset(ds)["permalink"]
    assert p["slug"] == "p1-bea-glomeruli-2024-01-02" and not p["custom"] and env.link(SLUG)[0] == 404
    assert main.db.get_permalink(p["slug"])["keys"]["uploader_name"] == "Bea"  # (what it keeps follows)


def test_orphaned_and_taken_over(main, env):  # noqa: F811
    st = env.stage()
    ds = env.commit(st)["dataset_id"]
    env.name(ds, SLUG)
    # while it points to a dataset, one of the same identity is refused, whatever its flies
    others = [("2025-02-01", 1, "DM2"), ("2025-02-01", 1, "DL5")]
    r = env.readme(env.stage(others))
    assert not r["ok"] and r["errors"] == [
        f"the permalink '{SLUG}' is of dataset #{ds} (p1: GH146 > GCaMP6f (glomeruli)), of the same uploader, indicator, driver, "
        "coding unit, manipulations and panels (p1) as this, and one dataset is kept of each, whatever their flies, odors and "
        f"trials. If this replaces it, delete #{ds} first (Admin tab): its permalink then points to this, once saved"]
    assert env.readme(env.stage(others), driver="pebbled")["ok"]  # (another identity)
    # deleted: orphaned, keeping what it was
    r = env.delete(ds)
    assert r["ok"] and r["orphaned"] == SLUG
    p = main.db.get_permalink(SLUG)
    assert p["dataset_id"] is None and p["orphaned_at"] and len(p["keys"]["flies"]) == 3 and p["keys"]["dataset_id"] == ds
    listed = call(main.admin_permalinks)[1]["permalinks"]
    assert [(x["slug"], x["dataset_id"]) for x in listed] == [(SLUG, None)] and listed[0]["summary"].startswith(f"#{ds}, ")
    assert env.link(SLUG)[1]["dataset_id"] is None
    # an upload of another identity does not take it over; one of the same, with other flies, once Proceed is pressed
    assert env.readme(env.stage(), uploader_name="Bea")["permalink"] is None
    r = env.readme(env.stage(others))["permalink"]
    assert r["slug"] == SLUG and not r["exact"] and r["differences"][0] == (
        "flies: 1 not there before (2025-02-01 fly 1); 3 of those before not here (2024-01-01 fly 1, 2024-01-02 fly 1, 2024-01-02 fly 2)")
    # the same again: taken over as it is saved
    r = env.readme(env.stage())
    assert r["ok"] and r["permalink"]["slug"] == SLUG and r["permalink"]["exact"] and r["permalink"]["differences"] == []
    again = env.commit(env.stage())
    assert again["ok"] and again["permalink"] == SLUG
    p = main.db.get_permalink(SLUG)
    assert p["dataset_id"] == again["dataset_id"] and p["orphaned_at"] is None and not p["custom"]
    # deleted, then uploaded with a fly fewer and an odor more: Proceed first
    env.delete(again["dataset_id"])
    rows = ROWS + [("p1", False, "va @ -2", "solvent", 0)]
    fewer = env.stage(COLS[:3], rows)
    r = env.readme(fewer)
    assert r["ok"] and not r["permalink"]["exact"] and r["permalink"]["differences"] == [
        "flies: 1 of those before not here (2024-01-02 fly 2)", "odors: 1 not there before (va @ -2 (p1))"]
    r = env.commit(fewer)
    assert not r["ok"] and f"the orphaned permalink '{SLUG}' was of, but not the same: Proceed" in r["errors"][-1]
    assert r["permalink"]["slug"] == SLUG
    r = env.commit(fewer, proceed=SLUG)
    assert r["ok"] and r["permalink"] == SLUG and main.db.get_permalink(SLUG)["dataset_id"] == r["dataset_id"]
    assert len(main.db.get_permalink(SLUG)["keys"]["flies"]) == 2


def test_delete_orphaned(main, env):  # noqa: F811
    ds = env.commit(env.stage())["dataset_id"]
    env.name(ds, "kept")
    delete = lambda slug: call(main.admin_permalink_delete, path_params={"slug": slug})
    assert delete("kept")[0] == 409 and delete("nope")[0] == 404
    env.delete(ds)
    assert delete("kept") == (200, {"ok": True}) and main.db.get_permalink("kept") is None
    assert call(main.admin_permalinks)[1]["permalinks"] == []
