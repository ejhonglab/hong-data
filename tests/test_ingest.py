"""Validation tests. Run with: python -m pytest tests/"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ingest import ValidationError, load_file
from app.odors import OdorFormatError, format_mix_from_strs, parse_odor, validate_odor_levels
from app.yamlcheck import check_subset, lint, parse_odor_yaml

HERE = Path(__file__).parent
SAMPLE = HERE / "data" / "validation2_sample.csv"


def make_wide(rows, cols, values=None, row_names=("panel", "is_pair", "odor1", "odor2", "repeat"),
              col_names=("date", "fly_num", "roi")) -> pd.DataFrame:
    ridx = pd.MultiIndex.from_tuples(rows, names=list(row_names))
    cidx = pd.MultiIndex.from_tuples(cols, names=list(col_names))
    if values is None:
        values = np.random.default_rng(0).normal(size=(len(rows), len(cols)))
    return pd.DataFrame(values, index=ridx, columns=cidx)


def write_csv(df: pd.DataFrame, tmp_path: Path, name="x.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p)
    return p


ROWS = [
    ("p1", False, "va @ -3", "solvent", 0), ("p1", False, "va @ -3", "solvent", 1),
    ("p1", False, "ea @ -2", "solvent", 0), ("p1", False, "ea @ -2", "solvent", 1),
    ("p1", False, "ea @ -2", "va @ -3", 0), ("p1", False, "ea @ -2", "va @ -3", 1),
]
COLS = [("2024-01-01", 1, "DM2"), ("2024-01-01", 1, "DL5"), ("2024-01-02", 1, "DM2"), ("2024-01-02", 2, "VA2")]


# ------------------------------------------------------------------ odor strings
@pytest.mark.parametrize("s,name,conc", [("va @ -3", "va", -3), ("ethyl acetate @ -2", "ethyl acetate", -2),
                                         ("euc @ -1.75", "euc", -1.75), ("2,3-b @ 0", "2,3-b", 0), ("+pul @ -2", "+pul", -2)])
def test_parse_odor_ok(s, name, conc):
    o = parse_odor(s)
    assert o.name == name and o.log10_conc == conc


@pytest.mark.parametrize("s", ["va@-3", "va @-3", "va @ 3", "va @ -3 ", " va @ -3", "va", "va @ x", "@ -3", "va @ -3 @ -2"])
def test_parse_odor_bad(s):
    with pytest.raises(OdorFormatError):
        parse_odor(s)


def test_solvent_and_mix():
    assert parse_odor("solvent").is_solvent
    assert format_mix_from_strs(["va @ -3", "solvent"]) == "va @ -3"
    assert format_mix_from_strs(["solvent", "ea @ -2", "va @ -3"]) == "ea @ -2 + va @ -3"
    assert format_mix_from_strs(["solvent", "solvent"]) == "solvent"


def test_odor_levels():
    assert validate_odor_levels(["panel", "odor1", "odor2", "repeat"]) == ["odor1", "odor2"]
    assert validate_odor_levels(["odor", "repeat"]) == ["odor"]
    with pytest.raises(ValueError):
        validate_odor_levels(["panel", "repeat"])
    with pytest.raises(ValueError):
        validate_odor_levels(["odor", "odor1"])
    with pytest.raises(ValueError):
        validate_odor_levels(["odor2", "odor3"])


# ------------------------------------------------------------------ wide CSV
def test_sample_file():
    p = load_file(SAMPLE)
    s = p.summary()
    assert s["n_units"] == 224 and s["n_trials"] == 141 and s["n_flies"] == 6
    assert s["panel_names"] == ["glomeruli_diagnostics", "validation2"]
    assert s["repeats_min"] == s["repeats_max"] == 3
    assert s["roi_kind"] == "glomeruli"
    assert p.trials[0]["odor"] == "3mtp @ -5" and p.trials[0]["components"] == ["3mtp @ -5"]
    assert p.values.shape == (224, 141)
    assert p.row_levels == ["panel", "odor1", "repeat"]
    assert p.warnings == ["row levels that say nothing were dropped: is_pair (all False); odor2 (all 'solvent')"]


@pytest.mark.parametrize("odor2,odor3,dropped", [
    (["solvent"] * 6, ["solvent"] * 6, ["odor2", "odor3"]),
    (["pfo @ 0"] * 6, ["pfo @ 0"] * 6, ["odor2", "odor3"]),
    # a 0 concentration however it is written
    (["pfo @ 0.0", "pfo @ 0", "pfo @ -0", "pfo @ 0.00", "pfo @ 0e0", "pfo @ 0."], ["pfo @ 0"] * 6, ["odor2", "odor3"]),
    (["water @ 0"] * 6, ["water @ 0.0"] * 6, ["odor2", "odor3"]),
    (["pfo @ 0"] * 6, ["water @ 0"] * 6, ["odor3"]),
    ([np.nan] * 6, [np.nan] * 6, ["odor2", "odor3"]),
    # only the trailing levels that hold one and the same value
    (["solvent"] * 6, ["pfo"] * 6, ["odor3"]),
    (["va @ -3"] * 6, ["water"] * 6, ["odor3"]),
    (["solvent"] * 5 + ["va @ -3"], ["solvent"] * 6, ["odor3"]),
    (["solvent"] * 6, ["solvent"] * 5 + ["va @ -3"], []),
    (["pfo @ -1"] * 6, ["solvent"] * 6, ["odor3"]),
])
def test_uninformative_levels(tmp_path, odor2, odor3, dropped):
    odor1 = ["va @ -3", "va @ -3", "ea @ -2", "ea @ -2", "ms @ -3", "ms @ -3"]
    rows = [("p1", False, o1, o2, o3, i % 2) for i, (o1, o2, o3) in enumerate(zip(odor1, odor2, odor3))]
    p = load_file(write_csv(make_wide(rows, COLS, row_names=("panel", "is_pair", "odor1", "odor2", "odor3", "repeat")), tmp_path))
    kept = [lvl for lvl in ("odor1", "odor2", "odor3") if lvl not in dropped]
    # is_pair is False on every row
    assert p.row_levels == ["panel", *kept, "repeat"]
    assert all(len(t["components"]) == len(kept) and "is_pair" not in t["extra"] for t in p.trials)
    assert p.trials[0]["components"] == ["va @ -3", odor2[0], odor3[0]][:len(kept)]


def test_informative_levels_kept(tmp_path):
    # is_pair True somewhere; odor1 is never dropped (here it is all one odor)
    rows = [("p1", i == 0, "va @ -3", "solvent", i) for i in range(3)]
    p = load_file(write_csv(make_wide(rows, COLS), tmp_path))
    assert p.row_levels == ["panel", "is_pair", "odor1", "repeat"]
    assert [t["extra"] for t in p.trials] == [{"is_pair": True}, {"is_pair": False}, {"is_pair": False}]


def test_minimal_wide_ok(tmp_path):
    p = load_file(write_csv(make_wide(ROWS, COLS), tmp_path))
    assert [u["roi"] for u in p.units] == ["DM2", "DL5", "DM2", "VA2"]
    assert [t["odor"] for t in p.trials][4:] == ["ea @ -2 + va @ -3"] * 2
    assert p.summary()["n_odors"] == 3


def test_int_rois_and_single_odor_level(tmp_path):
    rows = [("p1", "va @ -3", 0), ("p1", "va @ -3", 1), ("p1", "ea @ -2", 0)]
    cols = [("2024-01-01", 1, 0), ("2024-01-01", 1, 1), ("2024-01-01", 1, 25)]
    df = make_wide(rows, cols, row_names=("panel", "odor", "repeat"))
    p = load_file(write_csv(df, tmp_path))
    assert p.summary()["roi_kind"] == "integers"
    assert [u["roi"] for u in p.units] == ["0", "1", "25"]


def test_trials_required(tmp_path):
    # each trial numbered: without, refused (as averaged over trials, say)
    e = errors_for(make_wide([("p1", "va @ -3"), ("p1", "ea @ -2")], COLS, row_names=("panel", "odor1")), tmp_path)
    assert e.startswith("no trials: each trial's number is needed, in a row level (in a long file, a column) named repeat "
                        "(or trial) of integers (0, 1, 2... or 1, 2, 3...): one row per trial, not averaged over trials")
    # 'trial' for 'repeat', in either layout; integers
    df = make_wide([("p1", "va @ -3", 1), ("p1", "va @ -3", 2)], COLS, row_names=("panel", "odor1", "trial"))
    p = load_file(write_csv(df, tmp_path))
    assert [t["repeat"] for t in p.trials] == [1, 2] and "repeat" in p.row_levels and p.summary()["has_repeat_level"]
    long = df.stack(list(range(3)), future_stack=True).rename("value").reset_index()
    long.to_csv(tmp_path / "long.csv", index=False)
    assert [t["repeat"] for t in load_file(tmp_path / "long.csv").trials] == [1, 2]
    e = errors_for(make_wide([("p1", "va @ -3", 1.5), ("p1", "va @ -3", "two")], COLS, row_names=("panel", "odor1", "trial")), tmp_path)
    assert "'trial' values must be non-negative integers (each trial's number): '1.5', 'two'" in e


def errors_for(df, tmp_path) -> str:
    with pytest.raises(ValidationError) as ei:
        load_file(write_csv(df, tmp_path))
    return "\n".join(ei.value.errors)


def test_bad_roi(tmp_path):
    cols = COLS[:-1] + [("2024-01-02", 2, "DM2?")]
    e = errors_for(make_wide(ROWS, cols), tmp_path)
    assert "roi" in e and "DM2?" in e


def test_bad_roi_names_listed(tmp_path):
    # every name not accepted, with the accepted one it is but for case, and the names accepted
    cols = COLS[:1] + [("2024-01-01", 1, n) for n in ("dm2", "VA1D", "DM2?", "x1", "x2", "x3", "x4", "x5", "x6")]
    with pytest.raises(ValidationError) as ei:
        load_file(write_csv(make_wide(ROWS, cols), tmp_path))
    rois = ei.value.details["rois"]
    assert [(b["roi"], b["same_but_case"]) for b in rois["not_accepted"]] == [
        ("dm2", "DM2"), ("DM2?", None), ("VA1D", "VA1d"), ("x1", None), ("x2", None), ("x3", None), ("x4", None), ("x5", None),
        ("x6", None)]
    assert rois["accepted"][:3] == ["D", "DA1", "DA2"] and "VM7v" in rois["accepted"]


def test_example_files():
    for p in (Path(__file__).parent.parent / "examples").glob("megamat_mean_pebbled_*.csv"):
        assert load_file(p).summary()["roi_kind"] == "glomeruli"


def test_negative_int_roi(tmp_path):
    cols = COLS[:-1] + [("2024-01-02", 2, -1)]
    e = errors_for(make_wide(ROWS, cols), tmp_path)
    assert "'-1'" in e


def test_duplicate_column(tmp_path):
    cols = COLS[:-1] + [("2024-01-01", 1, "DM2")]
    e = errors_for(make_wide(ROWS, cols), tmp_path)
    assert "unique" in e


def test_bad_odor_string(tmp_path):
    rows = [("p1", False, "va@-3", "solvent", 0), ("p1", False, "va @ 2", "solvent", 0)]
    e = errors_for(make_wide(rows, COLS), tmp_path)
    assert "va@-3" in e and "0 or negative" in e


def test_missing_odor_level(tmp_path):
    df = make_wide([("p1", 0), ("p1", 1)], COLS, row_names=("panel", "repeat"))
    e = errors_for(df, tmp_path)
    assert "odor1" in e


def test_wrong_column_levels(tmp_path):
    df = make_wide(ROWS, [("2024-01-01", 1, "DM2", "x")], col_names=("date", "fly_num", "roi", "extra"))
    e = errors_for(df, tmp_path)
    assert "date" in e and "fly_num" in e


def test_bad_date_and_fly(tmp_path):
    cols = [("not-a-date", 1, "DM2"), ("2024-01-01", 0, "DM2")]
    e = errors_for(make_wide(ROWS, cols), tmp_path)
    assert "date" in e and "fly_num" in e


def test_duplicate_rows(tmp_path):
    rows = ROWS + [ROWS[0]]
    e = errors_for(make_wide(rows, COLS), tmp_path)
    assert "identical index" in e


def test_non_numeric_values(tmp_path):
    df = make_wide(ROWS, COLS).astype(object)
    df.iloc[0, 0] = "oops"
    e = errors_for(df, tmp_path)
    assert "non-numeric" in e and "oops" in e


def test_long_format_csv(tmp_path):
    wide = make_wide(ROWS, COLS)
    long = wide.stack(list(range(3)), future_stack=True).rename("value").reset_index()
    p = tmp_path / "long.csv"
    long.to_csv(p, index=False)
    parsed = load_file(p)
    s = parsed.summary()
    assert s["n_units"] == 4 and s["n_trials"] == 6 and s["source_format"] == "long-csv"
    # values round-trip (long rows are in a different order than the wide matrix)
    ref = load_file(write_csv(wide, tmp_path))
    ref_lookup = {(u["date"], u["fly_num"], u["roi"]): i for i, u in enumerate(ref.units)}
    for i, u in enumerate(parsed.units):
        j = ref_lookup[(u["date"], u["fly_num"], u["roi"])]
        t_ref = {(t["odor"], t["repeat"]): k for k, t in enumerate(ref.trials)}
        for k, t in enumerate(parsed.trials):
            assert parsed.values[i, k] == pytest.approx(ref.values[j, t_ref[(t["odor"], t["repeat"])]], rel=1e-6)


def test_tsv_and_garbage(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValidationError) as ei:
        load_file(p)
    assert "expected format" in ei.value.errors[0]


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")
def test_parquet_roundtrip(tmp_path):
    wide = make_wide(ROWS, [(d, str(f), r) for d, f, r in COLS])  # parquet needs string column labels
    p = tmp_path / "x.parquet"
    wide.to_parquet(p)
    parsed = load_file(p)
    assert parsed.summary()["n_units"] == 4 and parsed.source_format == "wide-parquet"



# ------------------------------------------------------------------ without flies or trials
# as tables of mean responses are often written: a single header line, of the row levels
# then the ROIs (some with a suffix), and no repeats
ROIS_ALONE = ("odor1,D,DA1_t0,VA1d/VA1v_t0,V_t1,DM2\n"
              "IaA @ -5,0.1,0.2,0.3,0.4,0.5\n"
              "IaA @ -4,0.2,0.3,0.4,0.5,0.6\n"
              "ban @ -0,0.3,0.4,0.5,0.6,0.7\n")


def test_rois_alone(tmp_path):
    # without flies or trials: refused, saying both (and what else there is to say)
    p = tmp_path / "x.csv"
    p.write_text(ROIS_ALONE)
    with pytest.raises(ValidationError) as ei:
        load_file(p)
    assert [e.split(":")[0] for e in ei.value.errors] == ["no flies", "no trials", "no panel"]
    assert ei.value.errors[0].startswith("no flies: the columns are ROIs alone, without date or fly_num, but each ROI's fly is "
                                         "needed, by its date and its number that day: in a wide file, header rows date, fly_num")
    w = "\n".join(ei.value.warnings)
    assert "'DA1_t0 → DA1'" in w and "'V_t1 → V'" in w and "'VA1d/VA1v'" in w


def test_roi_suffixes(tmp_path):
    # suffixes removed, glomeruli that could not be told apart kept as one ROI, each said
    cols = [("2024-01-01", 1, r) for r in ("D", "DA1_t0", "VA1d/VA1v_t0", "V_t1")]
    parsed = load_file(write_csv(make_wide(ROWS, cols), tmp_path))
    assert [u["roi"] for u in parsed.units] == ["D", "DA1", "VA1d/VA1v", "V"]
    assert all(u["roi_kind"] == "glomerulus" for u in parsed.units)
    w = "\n".join(parsed.warnings)
    assert "'DA1_t0 → DA1'" in w and "'V_t1 → V'" in w and "'VA1d/VA1v'" in w


def test_panel_required(tmp_path):
    # every trial's panel: a level, with a value on every row
    e = errors_for(make_wide([("va @ -3", 0), ("ea @ -2", 0)], COLS, row_names=("odor1", "repeat")), tmp_path)
    assert e.startswith("no panel: each trial's panel is needed (the name of the set of odors presented in that experiment, "
                        "e.g. megamat; which odors it has may vary somewhat from experiment to experiment)")
    e = errors_for(make_wide([("p1", "va @ -3", 0), ("", "ea @ -2", 0), (None, "ea @ -2", 1)], COLS,
                             row_names=("panel", "odor1", "repeat")), tmp_path)
    assert e.startswith("2 of its 3 rows have no panel, but each trial's panel is needed")


def test_flies_required(tmp_path):
    def errors(path):
        with pytest.raises(ValidationError) as ei:
            load_file(path)
        return ei.value.errors

    # ROIs alone, with trials, in each layout: refused for their flies only
    rows = [r for r in ROWS if r[3] == "solvent"]
    wide = make_wide(rows, [("DM2",), ("DL5",), ("VA2",)], col_names=("roi",))
    wide.columns = wide.columns.get_level_values(0).rename("roi")
    assert [e.split(":")[0] for e in errors(write_csv(wide, tmp_path))] == ["no flies"]
    long = wide.stack(future_stack=True).rename("value").reset_index()
    long.to_csv(tmp_path / "long.csv", index=False)
    assert [e.split(":")[0] for e in errors(tmp_path / "long.csv")] == ["no flies"]
    if importlib.util.find_spec("pyarrow"):
        wide.to_parquet(tmp_path / "x.parquet")
        assert [e.split(":")[0] for e in errors(tmp_path / "x.parquet")] == ["no flies"]
    # one of date and fly_num: named
    long = make_wide(ROWS, COLS).stack(list(range(3)), future_stack=True).rename("value").reset_index()
    long.drop(columns="date").to_csv(tmp_path / "nodate.csv", index=False)
    assert errors(tmp_path / "nodate.csv")[0].startswith("no date column: each ROI's fly is needed")
    two = make_wide(ROWS, [(d, r) for d, _, r in COLS], col_names=("date", "roi"))
    e = errors(write_csv(two, tmp_path, "two.csv"))
    assert e[0].startswith("no fly_num column level: each ROI's fly is needed") and e[0].endswith("(found ['date', 'roi'])")
    if importlib.util.find_spec("pyarrow"):
        two.to_parquet(tmp_path / "two.parquet")
        assert errors(tmp_path / "two.parquet")[0].startswith("no fly_num column level")


def test_rois_alone_errors(tmp_path):
    def errors(text, name="x.csv"):
        (tmp_path / name).write_text(text)
        with pytest.raises(ValidationError) as ei:
            load_file(tmp_path / name)
        return "\n".join(ei.value.errors)

    # the same ROI twice, once their suffixes are removed: nothing tells them apart
    e = errors("odor1,DA1_t0,DA1_t1,DM2\nva @ -3,1,2,3\n")
    assert "ROIs must be unique" in e and "'DA1'" in e and "after removing their suffixes" in e
    # only glomeruli may be joined
    assert "'VA1d/XX'" in errors("odor1,VA1d/XX,DM2\nva @ -3,1,2\n")
    # a long file with fly_num but not date
    e = errors("fly_num,roi,odor1,repeat,value\n1,DM2,va @ -3,0,0.5\n", "long.csv")
    assert e.startswith("no date column")



def test_cells_are_kc_ids(tmp_path):
    # 'cells' for 'roi': KC ids, integers
    cols = [("2024-01-01", 1, 0), ("2024-01-01", 1, 1), ("2024-01-02", 1, 1)]
    p = load_file(write_csv(make_wide(ROWS, cols, col_names=("date", "fly_num", "cells")), tmp_path))
    assert p.col_levels == ["date", "fly_num", "cells"] and [u["roi"] for u in p.units] == ["0", "1", "1"]
    assert {u["roi_kind"] for u in p.units} == {"int"} and p.summary()["roi_kind"] == "integers"
    assert {u["bad"] for u in p.units} == {None}  # (no level saying)
    # long, too
    long = make_wide(ROWS, cols, col_names=("date", "fly_num", "cells")).stack(list(range(3)), future_stack=True).rename("value").reset_index()
    long.to_csv(tmp_path / "long.csv", index=False)
    q = load_file(tmp_path / "long.csv")
    assert q.col_levels == ["date", "fly_num", "cells"] and sorted(u["roi"] for u in q.units) == ["0", "1", "1"]
    # glomeruli are not KC ids
    e = errors_for(make_wide(ROWS, [("2024-01-01", 1, "DM2"), ("2024-01-01", 1, 3)], col_names=("date", "fly_num", "cells")), tmp_path)
    assert "'cells' values are KC ids, so must be non-negative integers; not: 'DM2'" in e


def test_kc_ids_all_there(tmp_path):
    # every KC of each fly, from 0 or 1 to the highest: else refused, with what is missing
    cols = [("2024-01-01", 1, i) for i in (0, 1, 2, 5, 9)] + [("2024-01-02", 1, i) for i in (2, 3)] + [("2024-01-03", 1, 1)]
    e = errors_for(make_wide(ROWS, cols, col_names=("date", "fly_num", "cells")), tmp_path)
    assert ("KC data must have every KC of each fly, their ids running from 0 (or 1) to the highest, none missing: "
            "2024-01-01 fly 1 has no 3–4, 6–8; 2024-01-02 fly 1 has no 1. Upload all of each fly's KCs, with one more column "
            "level 'bad' (True or False) marking those not to be used") in e
    # ROIs named otherwise than 'cells' are not taken to be KCs here (the coding unit says so, at upload)
    load_file(write_csv(make_wide(ROWS, cols), tmp_path))


def test_bad_level(tmp_path):
    # a 'bad' level, of any ROIs: True or False (as written), wherever it is
    cols = [("2024-01-01", 1, 0, False), ("2024-01-01", 1, 1, True), ("2024-01-01", 1, 2, "false"), ("2024-01-02", 1, 1, 1)]
    df = make_wide(ROWS, cols, col_names=("date", "fly_num", "cells", "bad"))
    p = load_file(write_csv(df, tmp_path))
    assert p.col_levels == ["date", "fly_num", "cells", "bad"] and [u["bad"] for u in p.units] == [False, True, False, True]
    assert p.summary()["n_bad"] == 2
    q = load_file(write_csv(df.reorder_levels([3, 0, 1, 2], axis=1), tmp_path, "first.csv"))
    assert [u["bad"] for u in q.units] == [False, True, False, True] and any("reordered" in w for w in q.warnings)
    long = df.stack(list(range(4)), future_stack=True).rename("value").reset_index()
    long.to_csv(tmp_path / "long.csv", index=False)
    r = load_file(tmp_path / "long.csv")
    assert r.col_levels == ["date", "fly_num", "cells", "bad"] and not r.warnings[1:] and sum(u["bad"] for u in r.units) == 2
    make_wide(ROWS, [c[:3] + (bool(i % 2),) for i, c in enumerate(cols)], col_names=("date", "fly_num", "cells", "bad")).to_parquet(
        tmp_path / "x.parquet")
    assert [u["bad"] for u in load_file(tmp_path / "x.parquet").units] == [False, True, False, True]
    # glomeruli too, and without flies
    g = load_file(write_csv(make_wide(ROWS, [("2024-01-01", 1, "DM2", True), ("2024-01-01", 1, "DL5", False)],
                                      col_names=("date", "fly_num", "roi", "bad")), tmp_path, "g.csv"))
    assert [u["bad"] for u in g.units] == [True, False]
    nf = make_wide(ROWS, [(0, False), (1, True)], col_names=("cells", "bad"))
    nf.to_parquet(tmp_path / "nf.parquet")
    with pytest.raises(ValidationError) as ei:
        load_file(tmp_path / "nf.parquet")
    assert [e.split(":")[0] for e in ei.value.errors] == ["no flies"]
    # True or False only, and one such level
    bad = make_wide(ROWS, [("2024-01-01", 1, 0, "maybe"), ("2024-01-01", 1, 1, "")], col_names=("date", "fly_num", "cells", "bad"))
    assert "'bad' values must be True or False: '', 'maybe'" in errors_for(bad, tmp_path)


def test_mixture_in_one_level(tmp_path):
    # a mixture written in one odor level: its components, as if in levels of their own
    rows = [("p1", "ea @ -4.2 + eb @ -3.5", "solvent", 0), ("p1", "ea @ -4.2", "solvent", 0), ("p1", "ea @ -4.2", "IaA @ -3", 0)]
    p = load_file(write_csv(make_wide(rows, COLS, row_names=("panel", "odor1", "odor2", "repeat")), tmp_path))
    assert [t["components"] for t in p.trials] == [["ea @ -4.2", "eb @ -3.5", "solvent"], ["ea @ -4.2", "solvent", "solvent"],
                                                    ["ea @ -4.2", "solvent", "IaA @ -3"]]
    assert [t["odor"] for t in p.trials] == ["ea @ -4.2 + eb @ -3.5", "ea @ -4.2", "ea @ -4.2 + IaA @ -3"]
    assert any("1 rows name a mixture in one odor level" in w and "'ea @ -4.2 + eb @ -3.5'" in w for w in p.warnings)
    # each component is checked
    e = errors_for(make_wide([("p1", "ea @ -4.2 + eb", 0), ("p1", "ea @ -3", 0)], COLS, row_names=("panel", "odor1", "repeat")), tmp_path)
    assert "'eb'" in e
    # the same mixture written both ways is the same trial twice
    e = errors_for(make_wide([("p1", "ea @ -4 + eb @ -3", "solvent", 0), ("p1", "ea @ -4", "eb @ -3", 0)], COLS,
                             row_names=("panel", "odor1", "odor2", "repeat")), tmp_path)
    assert "duplicate trials" in e


# ------------------------------------------------------------------ YAML check
def test_yaml_check():
    text = """
odors:
 - name: valeric acid
   log10_conc: -3
   abbrev: va
 - name: ethyl acetate
   log10_conc: -2
 - name: methyl salicylate
   abbrev: ms
   log10_conc: -3
"""
    odors, meta = parse_odor_yaml(text)
    assert len(odors) == 3
    res = check_subset([("va", -3.0), ("ea", -2.0), ("ms", -4.0), ("zzz", -1.0)], odors)
    assert not res["ok"]
    assert [m["odor"] for m in res["matched"]] == ["va @ -3", "ea @ -2"]  # 'ea' via the hardcoded abbreviation map
    missing = {m["odor"]: m["hint"] for m in res["missing"]}
    assert "ms @ -4" in missing and "ms @ -3" in missing["ms @ -4"]
    assert "zzz @ -1" in missing


def test_yaml_lint():
    odors, _ = parse_odor_yaml("odors:\n - {name: valeric acid, abbrev: va, log10_conc: -3}\n - {name: valeric acid, abbrev: va, log10_conc: -4}\n"
                               " - {name: vanillin, abbrev: va, log10_conc: -2}\n - {name: hexanal, abbrev: 6al}\n"
                               " - {name: hexanal, abbrev: hex, log10_conc: -3}\n")
    assert lint(odors) == ["no log10_conc for hexanal", "abbreviation va is used for valeric acid and vanillin",
                           "hexanal is abbreviated both 6al and hex"]
    assert lint(parse_odor_yaml("odors:\n - {name: valeric acid, abbrev: va, log10_conc: -3}\n")[0]) == []


def test_yaml_pins2odors():
    text = "pins2odors:\n  3: {name: hexanal, log10_conc: -3, abbrev: 6al}\n  4: {name: solvent, log10_conc: null}\n"
    odors, _ = parse_odor_yaml(text)
    assert [o.name for o in odors] == ["hexanal", "solvent"]
    assert check_subset([("6al", -3.0)], odors)["ok"]
