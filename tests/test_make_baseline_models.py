"""Tests of scripts/make_baseline_models.py (the reference models olfdb shows as baselines, run by
pyolfsysm): the folder it writes, laid out as the KC-remapping archive (params.csv, the spike count
files), read by the Models tab and run by the model worker as made (`model_runs.plan`), and the
keys it registers them by; with a stand-in for pyolfsysm (the models themselves are olfsysm's to
test). Run with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import model_data, model_runs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_baseline_models.py"
BASE = {"one_row_per_claw": True, "prat_claws": True, "prat_boutons": True, "use_connectome_APL_weights": True,
        "pn_claw_to_apl": True, "n_spikes_for_response": 2, "target_sparsity": 0.05}
VARIANTS = {
    "bouton_base": dict(BASE),
    "uniform_glomeruli_claws_uniform_apl": dict(BASE, prat_claws=False, pn2kc_connections="uniform-glomeruli-boutons", n_claws=7,
                                                seed=0, uniform_apl_weights=True),
    "shuffled_connectome_pairs_kept_seed0": dict(BASE, shuffle_connectome_seed=0, shuffle_apl_pairs=True),
    "shuffled_connectome_pairs_broken_seed0": dict(BASE, shuffle_connectome_seed=0, shuffle_apl_pairs=False),
    "bouton_base_input_glomeruli": dict(BASE, input_glomeruli_only=True),
}
# the arguments fit_mb_model takes (those of them here), as the model worker says
KWARGS = ["orn_deltas", "target_sparsity", "fixed_thr", "wAPLKC", "wKCAPL", "wAPLPN", "wPNAPL", "one_row_per_claw", "prat_claws",
          "prat_boutons", "use_connectome_APL_weights", "pn_claw_to_apl", "n_spikes_for_response", "pn2kc_connections", "n_claws",
          "seed", "uniform_apl_weights", "shuffle_connectome_seed", "shuffle_apl_pairs", "input_glomeruli_only"]
ODORS = ["2h @ -3", "eb @ -3", "paa @ -4"]


@pytest.fixture
def script(monkeypatch):
    """The script as a module, with a stand-in for pyolfsysm.baselines: each model a few KCs, its
    spike counts from its name, its threshold and APL weights from its settings."""
    def baseline_kws(name, **overrides):
        return dict(VARIANTS[name], **overrides)

    def run_baseline(name, orn, **overrides):
        kws = baseline_kws(name, **overrides)
        ix = pd.MultiIndex.from_arrays([[1, 2, 3, 4], ["ab", "g", "g", "ab"]], names=["kc_id", "kc_type"])
        spk = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3) % (len(name) % 4 + 2), index=ix, columns=pd.Index(ODORS, name="odor"))
        off = spk * 2
        return {"kws": kws, "spike_counts_on": spk, "responses_on": (spk >= 2).astype(float), "spike_counts_off": off,
                "responses_off": (off >= 2).astype(float), "wPNKC": pd.DataFrame(np.zeros((8, 5))),
                "params": {"fixed_thr": 250.0 + len(name), "wAPLKC_scale": 0.13, "wKCAPL_scale": 7.7e-05, "wAPLPN_scale": 0.13,
                           "wPNAPL_scale": 3.4e-4}}

    def save_baseline_outputs(r, out, tag):
        for cond, k in (("on", "on"), ("acute_off", "off")):
            r[f"responses_{k}"].to_parquet(Path(out) / f"{tag}_{cond}_responses.parquet")
            r[f"spike_counts_{k}"].to_parquet(Path(out) / f"{tag}_{cond}_spike_counts.parquet")

    calls = []
    fake = types.SimpleNamespace(BOUTON_BASE_KWS=BASE, BASELINE_VARIANTS=VARIANTS, baseline_kws=baseline_kws,
                                 run_baseline=lambda *a, **k: calls.append(a[0]) or run_baseline(*a, **k),
                                 save_baseline_outputs=save_baseline_outputs,
                                 apl16_orn_deltas=lambda: pd.DataFrame(np.ones((2, 3)), index=pd.Index(["DL5", "DM2"], name="glomerulus"),
                                                                       columns=pd.MultiIndex.from_product([["megamat"], ODORS], names=["panel", "odor"])))
    pkg = types.ModuleType("pyolfsysm")
    pkg.baselines = fake
    monkeypatch.setitem(sys.modules, "pyolfsysm", pkg)
    monkeypatch.setitem(sys.modules, "pyolfsysm.baselines", fake)
    spec = importlib.util.spec_from_file_location("make_baseline_models", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # (the provenance needs pyolfsysm's and hong2p's records of what ran: stood in for)
    monkeypatch.setattr(mod, "provenance", lambda variants, orn_file, ran: {"generated": "2026-11-19T00:00:00+00:00", "variants": variants,
                                                                            "inputs": {"orn_deltas.csv": mod.sha256(orn_file)}, "runs": ran})
    mb = types.ModuleType("pyolfsysm.mb_model")
    mb.model_provenance = lambda **inputs: {"inputs": sorted(inputs)}
    monkeypatch.setitem(sys.modules, "pyolfsysm.mb_model", mb)
    mod.calls = calls
    return mod


def test_params_row(script):
    """A model's row: its settings in base_kwargs_json (all but the response rate tuned to) and as columns, its
    seed only where it is a setting of it, its tuned threshold and APL weights as absolute values."""
    kws = VARIANTS["uniform_glomeruli_claws_uniform_apl"]
    params = {"fixed_thr": np.float64(120.5), "wAPLKC_scale": 0.2, "wKCAPL_scale": 1e-4, "wAPLPN_scale": 0.3, "wPNAPL_scale": 5e-4}
    r = script.params_row("uniform_glomeruli_claws_uniform_apl", kws, params, n_kcs=1732, n_inputs=389, rate_on=0.05, rate_off=0.1)
    assert json.loads(r["base_kwargs_json"]) == {k: v for k, v in kws.items() if k != "target_sparsity"}
    assert r["target_rate"] == 0.05 and "target_sparsity" not in r
    assert {k: r[k] for k in ("tag", "variant", "base", "seed", "fixed_thr", "fixed_thr_mean", "wAPLKC", "wAPLKC_scale", "wPNAPL_scale")} == {
        "tag": "uniform_glomeruli_claws_uniform_apl", "variant": "uniform_glomeruli_claws_uniform_apl", "base": "bouton", "seed": 0,
        "fixed_thr": 120.5, "fixed_thr_mean": 120.5, "wAPLKC": 0.2, "wAPLKC_scale": 0.2, "wPNAPL_scale": 5e-4}
    assert all(r[k] == v for k, v in kws.items() if k != "target_sparsity")
    s = script.params_row("shuffled_connectome_pairs_broken_seed0", VARIANTS["shuffled_connectome_pairs_broken_seed0"], params, n_kcs=1,
                          n_inputs=1, rate_on=0, rate_off=0)
    assert s["seed"] is None and s["shuffle_connectome_seed"] == 0 and s["shuffle_apl_pairs"] is False
    # a threshold per KC: only its mean, which is no setting to run it with
    v = script.params_row("bouton_base", BASE, {**params, "fixed_thr": np.array([1.0, 3.0])}, n_kcs=2, n_inputs=1, rate_on=0, rate_off=0)
    assert v["fixed_thr"] is None and v["fixed_thr_mean"] == 2.0


def test_folder(script, tmp_path):
    """The folder written: an experiment the Models tab reads (a model per row, its on and acute_off runs), each run by
    the model worker as made, at its tuned threshold and APL weights or tuned again; a model done before is kept."""
    out = tmp_path / "olfdb_baselines"
    rows = script.run(out, list(VARIANTS))
    assert script.calls == list(VARIANTS) and [r["tag"] for r in rows] == list(VARIANTS)
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(["params.csv", "provenance.json", "model_provenance.json", "orn_deltas.csv",
                            *(f"{t}_{c}_{w}.parquet" for t in VARIANTS for c in ("on", "acute_off") for w in ("spike_counts", "responses"))])
    df = pd.read_csv(out / "params.csv")
    assert list(df.columns[:4]) == ["source_table", "tag", "variant", "base"] and len(df) == len(VARIANTS)
    for c in ("seed", "target_rate", "fixed_thr", "wAPLKC_scale", "wKCAPL_scale", "wAPLPN_scale", "wPNAPL_scale", "rate_on", "rate_off",
              "base_kwargs_json", "pn2kc_connections", "n_claws", "uniform_apl_weights", "shuffle_connectome_seed", "shuffle_apl_pairs", "input_glomeruli_only"):
        assert c in df.columns, c
    s = model_data.scan(out)
    assert model_data.summary(s) == {"experiments": 1, "models": len(VARIANTS), "runs": 2 * len(VARIANTS), "unmatched": 0, "skipped": 0}
    for m in s["models"]:
        tag = m["params"]["tag"]
        assert sorted(r["label"] for r in m["runs"]) == ["acute_off", "on"]
        p = model_runs.plan(m["params"], KWARGS)
        assert p["runnable"] is True and p["why"] == [] and p["tunings"] == ["fixed", "tuned"], (tag, p)
        assert p["model"] == {k: v for k, v in VARIANTS[tag].items() if k != "target_sparsity"}, tag
        assert p["target"] == 0.05 and p["fixed"]["fixed_thr"] == 250.0 + len(tag) and p["fixed"]["wAPLPN_scale"] == 0.13
    # without the new settings (an older pyolfsysm), the models that need them cannot be run, and say why
    old = [k for k in KWARGS if k not in ("pn2kc_connections", "n_claws", "seed", "uniform_apl_weights", "shuffle_connectome_seed",
                                           "shuffle_apl_pairs", "input_glomeruli_only")]
    plans = {m["params"]["tag"]: model_runs.plan(m["params"], old) for m in s["models"]}
    assert plans["bouton_base"]["runnable"] is True
    assert plans["shuffled_connectome_pairs_kept_seed0"]["runnable"] is False and plans["bouton_base_input_glomeruli"]["runnable"] is False
    assert "shuffle_connectome_seed" in plans["shuffled_connectome_pairs_kept_seed0"]["why"][0]
    # run again: those done kept; one forced, run again
    script.calls.clear()
    assert [r["tag"] for r in script.run(out, ["bouton_base"])] == list(VARIANTS) and script.calls == []
    script.run(out, ["shuffled_connectome_pairs_kept_seed0"], force=True)
    assert script.calls == ["shuffled_connectome_pairs_kept_seed0"] and len(pd.read_csv(out / "params.csv")) == len(VARIANTS)
    assert json.loads((out / "provenance.json").read_text())["inputs"]["orn_deltas.csv"] == script.sha256(out / "orn_deltas.csv")


def test_experiment_key(script, tmp_path):
    """The folder's key as the site lists it: of the folder of model data that has it (the deepest)."""
    listing = {"sources": [{"id": 1, "path": "Tom/claude_kc_remapping/data/model_outputs"}, {"id": 3, "path": "Tom"}]}
    out = tmp_path / "Tom/claude_kc_remapping/data/model_outputs/olfdb_baselines"
    out.mkdir(parents=True)
    assert script.experiment_key(listing, out, tmp_path) == "1/olfdb_baselines"
    with pytest.raises(SystemExit):
        script.experiment_key({"sources": [{"id": 2, "path": "Remy"}]}, out, tmp_path)


def test_register(script, tmp_path, monkeypatch):
    """Registering: by default every model of the folder but the bouton base (a baseline already, from the archive),
    each by its row as the site lists it."""
    out = tmp_path / "Tom/model_outputs/olfdb_baselines"
    script.run(out, list(VARIANTS))
    posted = []

    def request(url, body=None):
        if body is None:
            return {"sources": [{"id": 1, "path": "Tom/model_outputs"}],
                    "models": [{"experiment": "1/olfdb_baselines", "row": i, "params": {"tag": t}} for i, t in enumerate(VARIANTS)]}
        posted.append(body)
        return {"id": len(posted), "experiment": body["experiment"], "row": body["row"]}

    monkeypatch.setattr(script, "_request", request)
    script.register("http://site/", out, tensor=tmp_path)
    assert posted == [{"experiment": "1/olfdb_baselines", "row": i} for i in range(1, len(VARIANTS))]
    posted.clear()
    script.register("http://site", out, ["bouton_base", "nope"], tensor=tmp_path)
    assert posted == [{"experiment": "1/olfdb_baselines", "row": 0}]
