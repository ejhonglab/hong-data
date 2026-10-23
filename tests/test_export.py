"""Download tests (concatenating stored datasets). Run with: python -m pytest tests/"""
from __future__ import annotations

import asyncio
import io
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from starlette.requests import Request

from app import config, odormeta
from app.db import Database
from app.export import ExportError, dataset_odors_download, dataset_stem, odors_frame, odors_tex, trial_keys, write_download
from app.ingest import ParsedData, load_file
from tests.test_ingest import COLS, ROWS, make_wide, write_csv
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)


@pytest.fixture
def store(tmp_path):
    return Database(tmp_path / "db.sqlite3"), tmp_path / "datasets"


def add(store, parsed: ParsedData, driver="GH146-Gal4") -> int:
    db, datasets_dir = store
    metadata = {"uploader_name": "me", "indicator": "GCaMP6f", "driver": driver, "coding_unit": "glomeruli",
                "panel_names": ["p1"], "notes": "two\nlines"}
    ds_id = db.insert_dataset(
        created_at="2026-01-02T03:04:05", title=f"p1: {driver}", metadata=metadata, readme_md="# readme\n",
        readme_edited=False, data_file="data.csv", data_format=parsed.source_format, data_sha256="0" * 64,
        odor_yaml_files=[], experiment_xml_file=None, experiment_xml_info=None, summary=parsed.summary(),
        units=parsed.units, trials=parsed.trials, values=parsed.values,
    )
    d = datasets_dir / str(ds_id)
    (d / "odor_yaml").mkdir(parents=True)
    (d / "data.csv").write_text("original")
    (d / "odor_yaml" / "odors.yaml").write_text("odors: []")
    (d / "README.md").write_text("# readme\n")
    (d / "metadata.json").write_text("{}")
    return ds_id


def parse(tmp_path, rows, cols, row_names=("panel", "is_pair", "odor1", "odor2", "repeat"), name="x.csv", nan_at=None):
    df = make_wide(rows, cols, row_names=row_names)
    if nan_at:
        df.iloc[nan_at] = np.nan
    return load_file(write_csv(df, tmp_path, name))


def read_member(zip_path: Path, name: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf, zf.open(name) as f:
        return pd.read_csv(f, **kw)


@pytest.fixture
def two(store, tmp_path):
    a = parse(tmp_path, ROWS, COLS, name="a.csv", nan_at=(1, 2))
    # one odor level (padded with 'solvent'), a trial shared with `a`, a new panel, and a
    # (date, fly_num, roi) that `a` also has
    b_rows = [("p1", False, "va @ -3", 0), ("p2", False, "ms @ -3", 0)]
    b = parse(tmp_path, b_rows, [("2024-01-01", 1, "DM2"), ("2024-02-01", 1, "DL5")],
              row_names=("panel", "is_pair", "odor1", "repeat"), name="b.csv")
    return (add(store, a), a), (add(store, b, driver="pebbled"), b)


def write_zip(*args) -> str:
    name, media_type = write_download(*args)
    assert media_type == "application/zip"
    return name


def test_wide(store, tmp_path, two):
    (ia, a), (ib, b) = two
    out = tmp_path / "out.zip"
    name = write_zip(store[0], [ib, ia, ia], "wide", out, store[1])
    assert name == f"olfdb_datasets-{ia}-{ib}_wide.zip"
    stem = name[:-4]
    df = read_member(out, f"{stem}/responses_wide.csv", header=[0, 1, 2, 3], index_col=[0, 1, 2, 3])
    assert list(df.columns.names) == ["dataset_id", "date", "fly_num", "roi"]
    # is_pair is False throughout, so it was dropped on upload
    assert list(df.index.names) == ["panel", "odor1", "odor2", "repeat"]
    # a's 6 trials, then b's 'ms @ -3' (b's 'va @ -3' trial is a's first row)
    assert df.shape == (7, 6)
    assert list(df.index.get_level_values("odor2"))[-1] == "solvent"
    vals = df.to_numpy()
    np.testing.assert_allclose(vals[:6, :4], a.values.T, rtol=1e-6)
    assert np.isnan(vals[1, 2]) and np.isnan(vals[6, :4]).all()
    np.testing.assert_allclose(vals[[0, 6], 4:], b.values.T, rtol=1e-6)
    assert np.isnan(vals[1:6, 4:]).all()
    assert df.columns[4] == (str(ib), "2024-01-01", "1", "DM2")


def test_long(store, tmp_path, two):
    (ia, a), (ib, b) = two
    out = tmp_path / "out.zip"
    stem = write_zip(store[0], [ia, ib], "long", out, store[1])[:-4]
    df = read_member(out, f"{stem}/responses_long.csv")
    assert list(df.columns) == ["dataset_id", "date", "fly_num", "roi", "panel", "odor1", "odor2", "repeat", "value"]
    assert len(df) == np.isfinite(a.values).sum() + np.isfinite(b.values).sum()
    r = df[(df.dataset_id == ib) & (df.roi == "DL5") & (df.odor1 == "ms @ -3")]
    assert r.odor2.tolist() == ["solvent"] and r.value.tolist() == pytest.approx([b.values[1, 1]], rel=1e-6)


def test_files_and_tables(store, tmp_path, two):
    (ia, _), (ib, _) = two
    out = tmp_path / "out.zip"
    stem = write_zip(store[0], [ia, ib], "files", out, store[1])[:-4]
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        readme = zf.read(f"{stem}/README.md").decode()
    for i in (ia, ib):
        assert {f"{stem}/dataset-{i}/{p}" for p in ("data.csv", "odor_yaml/odors.yaml", "README.md", "metadata.json")} <= names
    assert f"#{ib} p1: pebbled" in readme
    meta = read_member(out, f"{stem}/datasets.csv")
    assert meta.dataset_id.tolist() == [ia, ib] and meta.driver.tolist() == ["GH146-Gal4", "pebbled"]
    assert meta.notes.tolist() == ["two\nlines"] * 2 and meta.n_units.tolist() == [4, 2]


def test_odors_csv(store, tmp_path, two):
    (ia, _), (ib, _) = two
    db = store[0]
    rows, _, _ = odormeta.parse_sheet("name,abbrev,cid\nvaleric acid,va,7991\nmethyl salicylate,ms,4133\n")
    db.apply_odor_sync(added=rows, changed=[], deleted=[], source="test", fingerprint="0",
                       details={}, when="2026-10-17T00:00:00")
    inchi = "InChI=1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)"
    db.save_inchis({7991: inchi}, when="2026-10-17T00:00:00")  # none fetched yet for ms
    out = tmp_path / "out.zip"
    stem = write_zip(db, [ia, ib], "wide", out, store[1])[:-4]
    odors = read_member(out, f"{stem}/odors.csv")
    assert list(odors.columns[odors.columns.get_loc("cid"):][:3]) == ["cid", "inchi", "cas"]
    by_odor = {(r.dataset_id, r.odor): r for r in odors.itertuples()}
    assert by_odor[(ia, "va @ -3")].inchi == inchi and by_odor[(ib, "va @ -3")].inchi == inchi
    assert by_odor[(ib, "ms @ -3")].cid == 4133 and pd.isna(by_odor[(ib, "ms @ -3")].inchi)
    assert pd.isna(by_odor[(ia, "ea @ -2")].inchi)  # not in the odor table


def test_dataset_odors_download(main, store, tmp_path, two):  # noqa: F811
    (ia, _), _ = two
    db, datasets_dir = store
    rows, errors, _ = odormeta.parse_sheet("name,abbrev,cid,supplier,catalog_no,metacyc\n"
                                           "valeric acid,va,7991,Sigma-Aldrich,240370,CPD_1%#\n")
    assert not errors
    db.apply_odor_sync(added=rows, changed=[], deleted=[], source="test", fingerprint="0", details={}, when="2026-10-17T00:00:00")

    text, name, media_type = dataset_odors_download(db, ia, datasets_dir, "csv")
    assert (name, media_type) == ("p1_me_glomeruli_2024-01-02_odors.csv", "text/csv")
    df = pd.read_csv(io.StringIO(text))
    # the columns of every export's odors.csv, but which dataset
    assert list(df.columns) == [c for c in odors_frame(db, [db.get_dataset(ia)], datasets_dir).columns if c != "dataset_id"]
    assert list(df["odor"]) == ["va @ -3", "ea @ -2"] and df["cid"].iloc[0] == 7991 and pd.isna(df["name"].iloc[1])
    assert df.columns[:5].tolist() == ["panel", "odor", "abbrev", "log10_conc", "name"]

    text, name, media_type = dataset_odors_download(db, ia, datasets_dir, "tex")
    assert name == "p1_me_glomeruli_2024-01-02_odors.tex" and media_type == "text/x-tex"
    lines = text.splitlines()
    # only the columns an odor has a value in: CID, supplier, catalog number, MetaCyc
    assert r"\begin{longtable}{lllrlll}" in lines
    assert r"Name & Abbrev. & $\log_{10}$ conc. & CID & Supplier & Catalog no. & MetaCyc ID \\" in lines
    assert r"\multicolumn{7}{l}{\textit{p1}} \\" in lines
    # linked as on the site (Sigma-Aldrich's catalog numbers to its search), &, # and % escaped in links and text
    assert (r"valeric acid & va & $-3$ & \href{https://pubchem.ncbi.nlm.nih.gov/compound/7991}{7991} & Sigma-Aldrich & "
            r"\href{https://www.sigmaaldrich.com/US/en/search/240370?focus=products\&term=240370\&type=product_number}{240370} & "
            r"\href{https://biocyc.org/compound?orgid=META\&id=CPD_1\%25\%23}{CPD\_1\%\#} \\") in lines
    assert r"\textit{not in the odor table} & ea & $-2$ &  &  &  &  \\" in lines
    assert text.count(r"\midrule") == 3  # under each header, and above the continued note: one panel

    with pytest.raises(ExportError, match="unknown format"):
        dataset_odors_download(db, ia, datasets_dir, "xlsx")
    with pytest.raises(ExportError, match="not found"):
        dataset_odors_download(db, 99, datasets_dir, "csv")


def test_lowercase(store, tmp_path):
    db, datasets_dir = store
    rows = [("p1", o, r) for o in ("IaA @ -3", "AA @ -2", "aa @ -2") for r in range(2)]
    ia = add(store, parse(tmp_path, rows, COLS, row_names=("panel", "odor1", "repeat"), name="a.csv"))
    ib = add(store, parse(tmp_path, [("p1", "iaa @ -3", 0)], COLS[:1], row_names=("panel", "odor1", "repeat"), name="b.csv"))
    out = tmp_path / "out"
    odors_of = lambda df: list(dict.fromkeys(df.index.get_level_values("odor1") if "odor1" not in df else df["odor1"]))
    # one dataset: its names lowercased, but those differing from another only in case
    write_download(db, [ia], "wide", out, datasets_dir, lower=True)
    assert odors_of(pd.read_csv(out, header=[0, 1, 2], index_col=[0, 1, 2])) == ["iaa @ -3", "AA @ -2", "aa @ -2"]
    write_download(db, [ia], "long", out, datasets_dir, lower=True)
    assert odors_of(pd.read_csv(out)) == ["iaa @ -3", "AA @ -2", "aa @ -2"]
    write_download(db, [ia], "wide", out, datasets_dir)
    assert odors_of(pd.read_csv(out, header=[0, 1, 2], index_col=[0, 1, 2]))[0] == "IaA @ -3"
    text = dataset_odors_download(db, ia, datasets_dir, "csv", lower=True)[0]
    assert list(pd.read_csv(io.StringIO(text))["abbrev"]) == ["iaa", "AA", "aa"]
    assert r"\textit{not in the odor table} & iaa & $-3$" in dataset_odors_download(db, ia, datasets_dir, "tex", lower=True)[0]
    assert r"\textit{not in the odor table} & IaA & $-3$" in dataset_odors_download(db, ia, datasets_dir, "tex")[0]
    # several: over all of them, so IaA and iaa (of two datasets) stay apart, as they were
    zip_path = tmp_path / "out.zip"
    name, _ = write_download(db, [ia, ib], "long", zip_path, datasets_dir, lower=True)
    stem = name[:-4]
    assert set(read_member(zip_path, f"{stem}/responses_long.csv")["odor1"]) == {"IaA @ -3", "iaa @ -3", "AA @ -2", "aa @ -2"}
    with zipfile.ZipFile(zip_path) as zf:
        assert "Odor names (abbreviations) are in lowercase" in zf.read(f"{stem}/README.md").decode()
    ib2 = add(store, parse(tmp_path, [("p1", "HCl @ -1", 0)], COLS[:1], row_names=("panel", "odor1", "repeat"), name="c.csv"))
    name, _ = write_download(db, [ia, ib2], "wide", zip_path, datasets_dir, lower=True)
    assert set(read_member(zip_path, f"{name[:-4]}/odors.csv")["abbrev"]) == {"iaa", "AA", "aa", "hcl"}
    # the original files as they are
    name, _ = write_download(db, [ia, ib2], "files", zip_path, datasets_dir, lower=True)
    assert set(read_member(zip_path, f"{name[:-4]}/odors.csv")["abbrev"]) == {"IaA", "AA", "aa", "HCl"}



def test_dataset_without_flies(store, tmp_path, monkeypatch):
    from app import db as dbmod
    from tests.test_ingest import ROIS_ALONE

    (tmp_path / "rois.csv").write_text(ROIS_ALONE)
    parsed = load_file(tmp_path / "rois.csv")
    db, datasets_dir = store
    ds_id = add(store, parsed)
    m = db.get_matrix_arrays(ds_id)
    assert m["units"][0] == {"date": None, "fly_num": None, "roi": "D", "roi_kind": "glomerulus", "bad": None}
    d = db.get_dataset(ds_id)
    assert d["n_flies"] == 0 and d["last_fly_date"] is None
    # downloaded as uploaded (a single header line; long without date or fly_num), so it uploads again
    out = tmp_path / "out"
    name, _ = write_download(db, [ds_id], "wide", out, datasets_dir)
    assert out.read_text().splitlines()[0] == "odor1,D,DA1,VA1d/VA1v,V,DM2"
    again = load_file(out.rename(tmp_path / "again.csv"))
    assert again.units == parsed.units and [t["odor"] for t in again.trials] == [t["odor"] for t in parsed.trials]
    np.testing.assert_allclose(again.values, parsed.values, rtol=1e-6)
    write_download(db, [ds_id], "long", out, datasets_dir)
    assert out.read_text().splitlines()[0] == "roi,odor1,value"
    np.testing.assert_allclose(load_file(out.rename(tmp_path / "again_long.csv")).values, parsed.values, rtol=1e-6)

    # a database from before: its units remade so that they may have no fly, nothing else changed
    old = tmp_path / "old.sqlite3"
    with monkeypatch.context() as mp:
        mp.setattr(dbmod, "SCHEMA", dbmod.SCHEMA.replace("    date TEXT,\n    fly_num INTEGER,", "    date TEXT NOT NULL,\n    fly_num INTEGER NOT NULL,"))
        olddb = Database(old)
    with_flies = add((olddb, tmp_path / "old_datasets"), load_file(write_csv(make_wide(ROWS, COLS), tmp_path, "a.csv")))
    notnull = lambda: {r[1]: r[3] for r in sqlite3.connect(old).execute("PRAGMA table_info(units)")}
    n_responses = lambda: sqlite3.connect(old).execute("SELECT count(*) FROM responses").fetchone()[0]
    assert notnull()["date"] == 1
    before, responses = olddb.get_matrix_arrays(with_flies), n_responses()
    newdb = Database(old)
    assert notnull()["date"] == 0 and notnull()["fly_num"] == 0
    after = newdb.get_matrix_arrays(with_flies)
    assert after["units"] == before["units"] and np.array_equal(after["values"], before["values"], equal_nan=True)
    assert n_responses() == responses > 0 and sqlite3.connect(old).execute("PRAGMA foreign_key_check").fetchall() == []
    rois_alone = add((newdb, tmp_path / "old_datasets2"), parsed)
    assert newdb.get_matrix_arrays(rois_alone)["units"][0]["date"] is None
    # deleting a dataset still deletes its units and their responses
    newdb.delete_dataset(with_flies)
    with sqlite3.connect(old) as con:
        assert con.execute("SELECT count(*) FROM units WHERE dataset_id = ?", (with_flies,)).fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM responses WHERE dataset_id = ?", (with_flies,)).fetchone()[0] == 0


def test_bad_rois(store, tmp_path):
    db, datasets_dir = store
    # KCs (a 'cells' level) with a 'bad' level: downloaded as uploaded, and uploaded again the same
    cols = [("2024-01-01", 1, 0, False), ("2024-01-01", 1, 1, True), ("2024-01-02", 1, 1, False)]
    parsed = parse(tmp_path, ROWS, [c[:3] for c in cols], name="kc.csv")
    for u, c in zip(parsed.units, cols):
        u["bad"] = c[3]
    parsed.col_levels = ["date", "fly_num", "cells", "bad"]
    ia = add(store, parsed)
    assert [u["bad"] for u in db.get_matrix_arrays(ia)["units"]] == [False, True, False]
    out = tmp_path / "kc_down.csv"
    write_download(db, [ia], "wide", out, datasets_dir)
    again = load_file(out)
    assert again.col_levels == ["date", "fly_num", "cells", "bad"] and [u["bad"] for u in again.units] == [False, True, False]
    write_download(db, [ia], "long", out, datasets_dir)
    assert list(pd.read_csv(out)["bad"].drop_duplicates()) == [False, True]
    # with another that does not say: its ROIs False
    ib = add(store, parse(tmp_path, ROWS, COLS[:1], name="g.csv"))
    zip_path = tmp_path / "both.zip"
    name, _ = write_download(db, [ia, ib], "wide", zip_path, datasets_dir)
    with zipfile.ZipFile(zip_path) as zf, zf.open(f"{name[:-4]}/responses_wide.csv") as f:
        wide = pd.read_csv(f, header=list(range(5)), index_col=list(range(4)))  # (panel, odor1, odor2, repeat)
        readme = zf.read(f"{name[:-4]}/README.md").decode()
    assert list(wide.columns.names) == ["dataset_id", "date", "fly_num", "roi", "bad"]
    assert [b for _, _, _, _, b in wide.columns] == ["False", "True", "False", "False"]
    assert "Header rows: dataset_id, date, fly_num, roi, bad" in readme and "`bad`: whether each ROI was marked" in readme
    name, _ = write_download(db, [ia, ib], "long", zip_path, datasets_dir)
    assert read_member(zip_path, f"{name[:-4]}/responses_long.csv").groupby("dataset_id")["bad"].any().to_dict() == {ia: True, ib: False}


def test_odors_tex():
    row = dict.fromkeys(["cid", "solvent", "glomerulus", "supplier", "catalog_no", "metacyc", "lotus", "cas", "vapor_pressure_kpa",
                         "pubchem_url", "catalog_url", "metacyc_url", "lotus_url", "cas_url", "vapor_pressure_url"])
    vapor_url = "https://pubchem.ncbi.nlm.nih.gov/compound/1#section=Vapor-Pressure"
    rows = [{**row, "panel": "diag_1", "name": "(-)-α-pinene & β", "abbrev": "a_p", "log10_conc": "-1.5", "solvent": "pfo",
             "vapor_pressure_kpa": 11.274},
            {**row, "panel": "diag_1", "name": "100% {x}", "abbrev": "x", "log10_conc": None, "vapor_pressure_kpa": 3417.8},
            {**row, "panel": "megamat", "name": "b", "abbrev": "b", "log10_conc": "-3", "cas": "1-2-3", "cas_url": "https://c/?a=1&b=2",
             "vapor_pressure_kpa": 5.25e-6, "vapor_pressure_url": vapor_url}]
    lines = odors_tex(rows, "Odors of #3: 50% done", "tab:x").splitlines()
    assert r"\caption{Odors of \#3: 50\% done}\label{tab:x} \\" in lines
    assert r"(-)-$\alpha$-pinene \& $\beta$ & a\_p & $-1.5$ & pfo &  & 11.3 \\" in lines
    assert r"100\% \{x\} & x &  &  &  & 3418 \\" in lines
    assert (r"b & b & $-3$ &  & \href{https://c/?a=1\&b=2}{1-2-3} & "
            r"\href{https://pubchem.ncbi.nlm.nih.gov/compound/1\#section=Vapor-Pressure}{$5.25 \times 10^{-6}$} \\") in lines
    # a rule between the panels, each titled
    i = lines.index(r"\multicolumn{6}{l}{\textit{megamat}} \\")
    assert lines[i - 1] == r"\midrule" and lines[lines.index(r"\multicolumn{6}{l}{\textit{diag\_1}} \\") - 1] == r"\endlastfoot"


def test_dataset_odors_api(main, store, tmp_path, two, monkeypatch):  # noqa: F811
    (ia, _), _ = two
    monkeypatch.setattr(main, "db", store[0])
    monkeypatch.setattr(config, "DATASETS_DIR", store[1])

    def get(fmt, ds_id=ia):
        scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b"",
                 "path_params": {"ds_id": ds_id, "fmt": fmt}}
        return asyncio.run(main.dataset_odors_file(Request(scope)))

    r = get("tex")
    assert r.status_code == 200 and r.headers["content-disposition"] == 'attachment; filename="p1_me_glomeruli_2024-01-02_odors.tex"'
    assert r.body.decode().count(r"\begin{longtable}") == 1
    assert get("csv").headers["content-type"].startswith("text/csv")
    assert get("xlsx").status_code == 404 and get("csv", 99).status_code == 404


def test_unknown_and_empty(store, tmp_path, two):
    with pytest.raises(ExportError, match="not found: 99"):
        write_zip(store[0], [two[0][0], 99], "wide", tmp_path / "x.zip", store[1])
    with pytest.raises(ExportError):
        write_zip(store[0], [], "wide", tmp_path / "x.zip", store[1])
    with pytest.raises(ExportError):
        write_zip(store[0], [two[0][0]], "xlsx", tmp_path / "x.zip", store[1])


def test_single_dataset(store, tmp_path, two):
    (ia, a), _ = two
    out = tmp_path / "out"
    assert write_download(store[0], [ia], "wide", out, store[1]) == ("p1_me_glomeruli_2024-01-02.csv", "text/csv")
    # laid out as uploaded (no dataset_id), and uploads again to the same data
    again = load_file(out.rename(tmp_path / "again.csv"))
    assert again.row_levels == ["panel", "odor1", "odor2", "repeat"] and again.units == a.units
    assert [t["odor"] for t in again.trials] == [t["odor"] for t in a.trials]
    np.testing.assert_allclose(again.values, a.values, rtol=1e-6)
    assert write_download(store[0], [ia], "long", out, store[1]) == ("p1_me_glomeruli_2024-01-02_long.csv", "text/csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["date", "fly_num", "roi", "panel", "odor1", "odor2", "repeat", "value"]
    assert len(df) == np.isfinite(a.values).sum()
    # the original files stay a zip
    assert write_zip(store[0], [ia], "files", out, store[1]) == f"olfdb_dataset-{ia}_files.zip"


def read_wide(path: Path, n_row_levels: int) -> pd.DataFrame:
    """A wide CSV as pandas reads it, with its date and fly_num column levels as dates and integers."""
    df = pd.read_csv(path, header=[0, 1, 2], index_col=list(range(n_row_levels)))
    cols = df.columns.to_frame(index=False)
    df.columns = pd.MultiIndex.from_arrays(
        [pd.to_datetime(cols["date"]), cols["fly_num"].astype(int), cols["roi"]], names=["date", "fly_num", "roi"]
    )
    return df


# dates as the lab's files have them (pandas writing Timestamps in a column level)
REAL_COLS = [("2023-11-19 00:00:00", 1, "DM2"), ("2023-11-19 00:00:00", 1, "DL5"), ("2023-11-19 00:00:00", 2, "DM2"),
             ("2023-11-21 00:00:00", 1, "VA2"), ("2023-11-21 00:00:00", 1, "DM2")]


@pytest.mark.parametrize("rows,row_names", [
    # mixtures flagged by is_pair, and odor2 not all one solvent: nothing to drop
    ([("megamat", pair, o1, o2, r) for pair, o1, o2 in [(False, "va @ -3", "solvent"), (False, "ea @ -2", "solvent"),
                                                        (True, "ea @ -2", "va @ -3")] for r in range(3)],
     ("panel", "is_pair", "odor1", "odor2", "repeat")),
    ([("2h @ -6", r) for r in range(2)] + [("ms @ -3", r) for r in range(2)], ("odor1", "repeat")),
    ([("diag", "1-6ol @ -3"), ("megamat", "1-6ol @ -3"), ("megamat", "pa @ -3")], ("panel", "odor")),
])
def test_wide_download_matches_upload(store, tmp_path, rows, row_names):
    values = np.random.default_rng(1).normal(size=(len(rows), len(REAL_COLS)))
    values[1, 2] = values[0, 4] = np.nan
    uploaded = write_csv(make_wide(rows, REAL_COLS, values, row_names=row_names), tmp_path, "uploaded.csv")
    parsed = load_file(uploaded)
    assert parsed.row_levels == list(row_names)  # no levels dropped
    out = tmp_path / "downloaded.csv"
    write_download(store[0], [add(store, parsed)], "wide", out, store[1])
    # values are stored as float32
    pd.testing.assert_frame_equal(read_wide(out, len(row_names)), read_wide(uploaded, len(row_names)),
                                  check_exact=False, rtol=1e-6, atol=0)


def test_dataset_stem():
    d = {"id": 1, "uploader_name": "É.J. Hong", "coding_unit": "KCs", "last_fly_date": "2023-11-19",
         "metadata": {"panel_names": ["from form"]}}
    trials = lambda *panels: [{"panel": p} for p in panels]
    assert dataset_stem(d, trials("megamat", "glomeruli_diagnostics", "Kiwi", "megamat")) == "Kiwi_megamat_ej-hong_KCs_2023-11-19"
    assert dataset_stem(d, trials("glomeruli_diagnostics")) == "glomeruli_diagnostics_ej-hong_KCs_2023-11-19"
    assert dataset_stem(d, trials(None)) == "from-form_ej-hong_KCs_2023-11-19"


def test_stored_before_levels_were_dropped(store, tmp_path):
    """Trials stored with levels that say nothing are read (and downloaded) without them."""
    rows = [(False, "va @ -3", "pfo @ 0", 0), (False, "ea @ -2", "pfo @ 0", 0)]
    with_levels = load_file(write_csv(make_wide(rows, COLS, row_names=("is_pair", "odor1", "odor2", "repeat")), tmp_path))
    # as it was parsed before: odor2 and is_pair dropped now
    assert with_levels.row_levels == ["odor1", "repeat"]
    with_levels.trials = [{**t, "odor": f"{t['odor']} + pfo @ 0", "components": [*t["components"], "pfo @ 0"],
                           "extra": {"is_pair": False}} for t in with_levels.trials]
    with_levels.row_levels = ["is_pair", "odor1", "odor2", "repeat"]
    ds_id = add(store, with_levels)
    m = store[0].get_matrix_arrays(ds_id)
    assert [(t["odor"], t["components"], t["extra"]) for t in m["trials"]] == [("va @ -3", ["va @ -3"], {}), ("ea @ -2", ["ea @ -2"], {})]
    assert store[0].get_trials(ds_id)[0] == {"panel": None, "odor": "va @ -3", "components": ["va @ -3"]}
    out = tmp_path / "out"
    write_download(store[0], [ds_id], "wide", out, store[1])
    assert out.read_text().splitlines()[3].split(",")[:3] == ["odor1", "repeat", ""]


def test_extra_levels(tmp_path):
    with_pair = parse(tmp_path, [("p1", False, "va @ -3", 0), ("p1", True, "va @ -3", 0)], COLS,
                      row_names=("panel", "is_pair", "odor1", "repeat"), name="a.csv")
    no_pair = parse(tmp_path, [("va @ -3", 0), ("ea @ -2", 0)], COLS, row_names=("odor1", "repeat"), name="b.csv")
    other_pair = parse(tmp_path, [("p1", True, "ea @ -2", 0), ("p1", False, "ms @ -3", 0)], COLS,
                       row_names=("panel", "is_pair", "odor1", "repeat"), name="c.csv")
    levels_of = lambda *ps: trial_keys([{"trials": p.trials} for p in ps])
    # is_pair is only kept if every dataset has it ...
    assert levels_of(no_pair, other_pair)[0] == ["panel", "odor1", "repeat"]
    assert levels_of(with_pair, other_pair)[0] == ["panel", "is_pair", "odor1", "repeat"]
    # ... or some dataset needs it to keep its trials apart
    levels, keys = levels_of(with_pair, no_pair)
    assert levels == ["panel", "is_pair", "odor1", "repeat"]
    assert keys[1] == [(None, None, "va @ -3", 0), (None, None, "ea @ -2", 0)]


def test_edit_readme(main, store, tmp_path, two, monkeypatch):  # noqa: F811
    db, datasets_dir = store  # the same database as main's
    monkeypatch.setattr(config, "DATASETS_DIR", datasets_dir)
    (ia, _), (ib, _) = two
    (datasets_dir / str(ia) / "metadata.json").write_text('{"readme_edited": false}')
    save = lambda body, i=ia: call(main.dataset_readme_save, body=body, path_params={"ds_id": i})
    status, r = save({"readme": "# new\n\nedited", "base": "# readme\n"})
    assert status == 200 and r["readme_path"] == str(datasets_dir / str(ia) / "README.md") and "<h1>new</h1>" in r["readme_html"]
    assert (datasets_dir / str(ia) / "README.md").read_text() == "# new\n\nedited"
    d = db.get_dataset(ia)
    assert d["readme_md"] == "# new\n\nedited" and d["readme_edited"] == 1
    assert '"readme_edited": true' in (datasets_dir / str(ia) / "metadata.json").read_text()
    assert db.get_dataset(ib)["readme_md"] == "# readme\n"
    # refused if it changed since the edit began, if empty, or for no dataset
    assert save({"readme": "# again", "base": "# readme\n"})[0] == 409 and db.get_dataset(ia)["readme_md"] == "# new\n\nedited"
    assert save({"readme": "  ", "base": "# new\n\nedited"})[0] == 422 and save({"readme": "# x"})[0] == 400
    assert save({"readme": "# x", "base": ""}, i=ib + 1)[0] == 404
    # downloads have the new README, from the files and from the database
    for fmt in ("files", "long"):
        out = tmp_path / f"{fmt}.zip"
        stem = write_zip(db, [ia, ib], fmt, out, datasets_dir)[:-4]
        with zipfile.ZipFile(out) as zf:
            assert zf.read(f"{stem}/dataset-{ia}/README.md").decode() == "# new\n\nedited"
    status, r = call(main.dataset_get, path_params={"ds_id": ia})
    assert r["readme_path"] == str(datasets_dir / str(ia) / "README.md")


def test_delete_dataset(main, store, tmp_path, two, monkeypatch):  # noqa: F811
    db, datasets_dir = store  # the same database as main's
    deleted = tmp_path / "deleted"
    monkeypatch.setattr(config, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(config, "DELETED_DIR", deleted)
    (ia, _), (ib, _) = two
    delete = lambda i: call(main.admin_dataset_delete, path_params={"ds_id": i})

    def rows(i):  # in each table
        with db.connect() as con:
            return {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE dataset_id = ?", (i,)).fetchone()[0]
                    for t in ("dataset_panels", "units", "trials", "responses", "matrices")}

    assert all(rows(ia).values())

    # nothing is deleted if its folder cannot be moved
    def fail(*args):
        raise OSError("read-only")

    with monkeypatch.context() as m:  # (undoing only this)
        m.setattr(main.shutil, "move", fail)
        status, r = delete(ia)
    assert status == 500 and "nothing was deleted (read-only)" in r["error"]
    assert db.get_dataset(ia) is not None and all(rows(ia).values()) and (datasets_dir / str(ia) / "data.csv").exists()
    # its rows go, its folder is moved
    assert delete(ia) == (200, {"ok": True, "files": str(deleted / str(ia))})
    assert db.get_dataset(ia) is None and not any(rows(ia).values()) and all(rows(ib).values())
    assert [d["id"] for d in db.list_datasets()] == [ib]
    assert not (datasets_dir / str(ia)).exists() and (deleted / str(ia) / "data.csv").read_text() == "original"
    assert delete(ia)[0] == 404 and call(main.dataset_get, path_params={"ds_id": ia})[0] == 404
    # beside a folder of the same name put there by hand
    (deleted / str(ib)).mkdir()
    status, r = delete(ib)
    assert status == 200 and r["files"].startswith(str(deleted / f"{ib}_")) and (Path(r["files"]) / "README.md").exists()
    assert not list((deleted / str(ib)).iterdir()) and db.list_datasets() == []
