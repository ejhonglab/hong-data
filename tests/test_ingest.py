"""Validation tests. Run with: python -m pytest tests/"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ingest import ValidationError, load_file
from app.odors import OdorFormatError, format_mix_from_strs, parse_odor, validate_odor_levels
from app.yamlcheck import check_subset, parse_odor_yaml

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
    assert p.trials[0]["odor"] == "3mtp @ -5" and p.trials[0]["components"] == ["3mtp @ -5", "solvent"]
    assert p.values.shape == (224, 141)
    assert not p.warnings


def test_minimal_wide_ok(tmp_path):
    p = load_file(write_csv(make_wide(ROWS, COLS), tmp_path))
    assert [u["roi"] for u in p.units] == ["DM2", "DL5", "DM2", "VA2"]
    assert [t["odor"] for t in p.trials][4:] == ["ea @ -2 + va @ -3"] * 2
    assert p.summary()["n_odors"] == 3


def test_int_rois_and_single_odor_level(tmp_path):
    rows = [("va @ -3", 0), ("va @ -3", 1), ("ea @ -2", 0)]
    cols = [("2024-01-01", 1, 0), ("2024-01-01", 1, 1), ("2024-01-01", 1, 25)]
    df = make_wide(rows, cols, row_names=("odor", "repeat"))
    p = load_file(write_csv(df, tmp_path))
    assert p.summary()["roi_kind"] == "integers"
    assert [u["roi"] for u in p.units] == ["0", "1", "25"]


def test_no_repeat_level_warns(tmp_path):
    df = make_wide([("va @ -3",), ("ea @ -2",)], COLS, row_names=("odor1",))
    p = load_file(write_csv(df, tmp_path))
    assert any("repeat" in w for w in p.warnings)
    assert all(t["repeat"] == 0 for t in p.trials)


def errors_for(df, tmp_path) -> str:
    with pytest.raises(ValidationError) as ei:
        load_file(write_csv(df, tmp_path))
    return "\n".join(ei.value.errors)


def test_bad_roi(tmp_path):
    cols = COLS[:-1] + [("2024-01-02", 2, "DM2?")]
    e = errors_for(make_wide(ROWS, cols), tmp_path)
    assert "roi" in e and "DM2?" in e


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


def test_yaml_pins2odors():
    text = "pins2odors:\n  3: {name: hexanal, log10_conc: -3, abbrev: 6al}\n  4: {name: solvent, log10_conc: null}\n"
    odors, _ = parse_odor_yaml(text)
    assert [o.name for o in odors] == ["hexanal", "solvent"]
    assert check_subset([("6al", -3.0)], odors)["ok"]
