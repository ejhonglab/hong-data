"""Tests of baseline models run on datasets of glomeruli (model_runs.py, orn_input.py,
model_worker.py), with a stand-in for pyolfsysm (the model itself is olfsysm's to test). Run
with: OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/."""
from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import config, model_data, model_runs, model_worker, odormeta, orn_input
from app.ingest import load_file
from tests.test_ingest import make_wide, write_csv
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

GLOMS = ["DA3", "DA4l", "DA4m", "DC1", "DL1", "DL3", "DL4", "DL5", "DM2"]
H_ODORS = ["ea", "eb", "pa", "2h", "ms", "1-6ol"]  # (as Hallem's table names them: the lab's abbreviations)
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_data"  # (the repository's: the built-in reference data)
H = pd.read_csv(REFERENCE_DIR / orn_input.HALLEM_FILE, index_col=0)
BOUTON = {"one_row_per_claw": True, "prat_claws": True, "n_spikes_for_response": 2}
ROW = {"source_table": "s.csv", "tag": "bouton_base", "variant": "bouton_base", "base": "bouton", "target_rate": 0.05,
       "fixed_thr_mean": 254.4, "wAPLKC": 0.13, "wAPLKC_scale": 0.13, "wKCAPL_scale": 7.7e-05, "n_kcs": 1732, "rate_on": 0.0516,
       "base_kwargs_json": json.dumps(BOUTON), **BOUTON}
KWARGS = ["target_sparsity", "fixed_thr", "wAPLKC", "wKCAPL", "one_row_per_claw", "prat_claws", "n_spikes_for_response", "claw_thr", "seed"]

# a stand-in for pyolfsysm, in the Python the worker runs models with: the same calls, files of the same names
FAKE_MB_MODEL = '''
import json
import pandas as pd

optimizable_model_params = {"fixed_thr": "fixed_thr", "wAPLKC_scale": "wAPLKC", "wKCAPL_scale": "wKCAPL"}


def read_orn_deltas(path):
    return pd.read_csv(path, header=[0, 1], index_col=0)


def fixed_params_to_fit_mb_model_kws(params):
    return {optimizable_model_params[k]: v for k, v in params.items()}


def olfsysm_clean_problems():
    return []


def model_provenance(**inputs):
    return {"pyolfsysm": {"git_hash": "abc123", "dirty": False}}


def fit_mb_model(orn_deltas, *, target_sparsity=None, fixed_thr=None, wAPLKC=None, wKCAPL=None, one_row_per_claw=False,
                 prat_claws=False, n_spikes_for_response=None, claw_thr=None, seed=1, _wPNKC=None):
    pass


def fit_and_plot_mb_model(plot_dir, *, try_cache=True, plot_dirname=None, orn_deltas=None, **model_kws):
    if "boom" in orn_deltas.columns.get_level_values("odor")[0]:
        raise RuntimeError("the model blew up")
    out = plot_dir / plot_dirname
    out.mkdir(parents=True)
    odors = orn_deltas.columns.get_level_values("odor")
    index = pd.MultiIndex.from_arrays([[10, 11, 12], ["ab", "g", "ab"]], names=["kc_id", "kc_type"])
    counts = pd.DataFrame([[0.0] * len(odors), [1.0] * len(odors), [float(len(orn_deltas))] * len(odors)], index=index,
                          columns=pd.Index(odors, name="odor"))
    counts.to_parquet(out / "spike_counts.parquet")
    counts.to_csv(out / "spike_counts.csv")
    (counts >= 2).to_parquet(out / "responses.parquet")
    orn_deltas.to_csv(out / "orn_deltas.csv")
    params = {**model_kws, "fixed_thr": model_kws.get("fixed_thr", 111.0), "wAPLKC_scale": model_kws.get("wAPLKC", 0.2), "output_dir": plot_dirname}
    (out / "params.json").write_text(json.dumps(params))
    (out / "corr.pdf").write_bytes(b"%PDF")
    print("writing", out / "spike_counts.csv")
    return params
'''


@pytest.fixture
def env(main, root, tmp_path, monkeypatch):  # noqa: F811
    """main, with its data folder the test's own, and the stand-in pyolfsysm where the worker's model Python finds it."""
    data = tmp_path / "data"
    for k, v in {"DATA_DIR": data, "MODEL_RUNS_DIR": data / "model_runs", "MODEL_BASELINES_DIR": data / "model_baselines",
                 "REFERENCE_DIR": REFERENCE_DIR}.items():
        monkeypatch.setattr(config, k, v)
    fake = tmp_path / "fake"
    (fake / "pyolfsysm").mkdir(parents=True)
    (fake / "pyolfsysm" / "__init__.py").write_text("")
    (fake / "pyolfsysm" / "mb_model.py").write_text(FAKE_MB_MODEL)
    (fake / "olfsysm.py").write_text("def build_info():\n    return {'git_hash': 'abc123', 'cpp_dirty': 0, 'git_diff': ''}\n")
    monkeypatch.setenv("PYTHONPATH", str(fake))
    monkeypatch.setenv("OLFDB_MODEL_PYTHON", sys.executable)
    monkeypatch.setattr(model_worker, "POLL", 0.05)
    return main


def add(db, tmp_path, name, odors, factor=0.01, gloms=GLOMS, panel="p1", extra=(), **md):
    """A dataset of glomeruli (two flies): to each of `odors` (at 10^-3) each glomerulus's response Hallem's to
    it (at 10^-2) times `factor`, and to those of `extra` ((panel, odor string): not Hallem's) 100 × `factor` × (1 + its index)."""
    rows = [(panel, False, f"{o} @ -3", "solvent", rep) for o in odors for rep in (0, 1)]
    rows += [(p, False, o, "solvent", 0) for p, o in extra] + [(panel, False, f"{odors[0]} @ -3", f"{odors[1]} @ -3", 0)]
    cols = [(date, 1, g) for date in ("2024-01-01", "2024-01-02") for g in gloms]
    values = np.array([[factor * H.loc[g, f"{r[2].split(' @ ')[0]} @ -2"] if r[2].split(" @ ")[0] in H_ODORS and r[3] == "solvent" and g in H.index
                        else factor * 100 * (1.0 + gloms.index(g)) for _, _, g in cols] for r in rows])
    parsed = load_file(write_csv(make_wide(rows, cols, values=values), tmp_path, f"{name}.csv"))
    metadata = {"uploader_name": "Ann", "indicator": "GCaMP6f", "driver": "pebbled", "coding_unit": "glomeruli", **md}
    return db.insert_dataset(
        created_at="2026-01-02T03:04:05", title=name, metadata=metadata, readme_md="#", readme_edited=False,
        data_file=f"{name}.csv", data_format=parsed.source_format, data_sha256="0" * 64, odor_yaml_files=[],
        experiment_xml_file=None, experiment_xml_info=None, summary=parsed.summary(), units=parsed.units,
        trials=parsed.trials, values=parsed.values)


def index(db):
    return odormeta.OdorIndex(db.list_odors())


def test_hallem_reference(env):
    # all they tabulate: 110 odors at 10^-2, 10 of them lower, and the fruits; DM3 once (Or47a's)
    assert H.shape == (23, 176) and list(H.index).count("DM3") == 1
    assert {"ea @ -2", "ea @ -4", "ea @ -8", "apple @ 0", "apple @ -6", "ammonium hydroxide @ -2"} <= set(H.columns)
    hal = orn_input.hallem(index(env.db))
    assert hal[(("name", "ea"), -4.0)]["DM2"] == H.loc["DM2", "ea @ -4"] and hal[(("name", "ea"), -4.0)][None] == "ea @ -4"
    # shown as a reference dataset, its columns odors as datasets write them
    listed = next(r for r in call(env.reference_list)[1]["datasets"] if r["id"] == "hallem-carlson-2006")
    assert listed["state"] == "ok" and listed["shape"] == [23, 176] and listed["glomeruli"]["axis"] == "rows"


def test_scale_by_hallem(env, tmp_path):
    a = add(env.db, tmp_path, "a", H_ODORS, extra=[("glomeruli_diagnostics", "x1 @ -2")])
    assert orn_input.panels(env.db, [a]) == [{"name": "p1", "n_odors": 6, "default": True},
                                             {"name": "glomeruli_diagnostics", "n_odors": 1, "default": False}]
    s = orn_input.scale(env.db, [a], index(env.db), ["p1"])
    # their 10^-2 our 10^-3: every response in common, the factor that undoes ours
    assert s["method"] == "hallem" and s["factor"] == pytest.approx(100) and s["hallem"]["r"] == pytest.approx(1)
    assert (s["hallem"]["n_odors"], s["hallem"]["n_points"], s["hallem"]["enough"]) == (6, 54, True)
    assert s["odors"] == [f"{o} @ -3" for o in H_ODORS] and s["glomeruli"] == sorted(GLOMS) and s["mixtures"] == ["ea @ -3 + eb @ -3"]
    assert np.allclose(s["orn_deltas"].to_numpy(), H.loc[sorted(GLOMS), [f"{o} @ -2" for o in H_ODORS]].to_numpy(), atol=1e-3)
    frame = orn_input.model_frame(s)
    assert frame.columns.names == ["panel", "odor"] and frame.index.name == "glomerulus" and set(frame.columns.get_level_values(0)) == {"p1"}
    # every panel's odors, if not said (and the factor from them all, whichever are the model's)
    assert orn_input.scale(env.db, [a], index(env.db))["odors"][-1] == "x1 @ -2"
    assert "the 6 odors (54 odor-glomerulus pairs, r = 1.00)" in orn_input.said(s)
    # a glomerulus not measured for every odor is left out; too few of them refused
    with pytest.raises(orn_input.InputError, match="only 4 of its glomeruli are measured"):
        orn_input.scale(env.db, [add(env.db, tmp_path, "few", H_ODORS, gloms=GLOMS[:4])], index(env.db))
    pns = add(env.db, tmp_path, "kcs", H_ODORS, gloms=[str(i) for i in range(9)])
    with pytest.raises(orn_input.InputError, match="not a dataset of glomeruli"):
        orn_input.scale(env.db, [pns], index(env.db))


def test_scale_by_combination(env, tmp_path):
    # too few odors in common with Hallem's (2), and no other dataset to scale it to
    extra = [("p1", "x1 @ -2"), ("p1", "x2 @ -2")]
    b = add(env.db, tmp_path, "b", H_ODORS[:2], factor=0.005, extra=extra)
    with pytest.raises(orn_input.InputError, match="it has 2 odors .* no other dataset of its coding unit"):
        orn_input.scale(env.db, [b], index(env.db))
    # with one that has enough, sharing x1 and x2 with it (and its 2): scaled to their merge, and that to Hallem's
    a = add(env.db, tmp_path, "a", H_ODORS, factor=0.01, extra=extra)
    s = orn_input.scale(env.db, [b], index(env.db))
    c = s["combination"]
    assert s["method"] == "combination" and not s["hallem"]["enough"] and [d["id"] for d in c["datasets"]] == [a]
    assert c["hallem"]["n_odors"] == 6 and c["hallem"]["enough"]
    # (b's Hallem responses are half a's: its factor twice a's, though by its odors alone it could not be told)
    assert s["factor"] == pytest.approx(2 * orn_input.scale(env.db, [a], index(env.db))["factor"], rel=0.05)
    assert "too little in common" in orn_input.said(s) and "it and 1 other" in orn_input.said(s)
    summary = orn_input.summary(s)
    assert json.dumps(summary) and "points" not in summary["hallem"] and summary["combination"]["hallem"]["n_odors"] == 6


def test_plan():
    p = model_runs.plan(ROW, KWARGS)
    assert p["runnable"] and p["why"] == [] and p["model"] == BOUTON and p["target"] == 0.05 and p["tunings"] == ["fixed", "tuned"]
    assert p["fixed"] == {"fixed_thr": 254.4, "wAPLKC_scale": 0.13, "wKCAPL_scale": 7.7e-05}
    # its other columns that are settings of fit_mb_model are passed; those that are off, not
    p = model_runs.plan({**ROW, "claw_thr": 70.0, "gain": 1.0, "k": 0.0, "kc_kc_inh": 0.0, "seed": float("nan")}, KWARGS)
    assert p["runnable"] and p["model"] == {**BOUTON, "claw_thr": 70.0}
    # another experiment's own mechanism: not run
    p = model_runs.plan({**ROW, "gain": 1.5, "k": 2.0, "gain_glomeruli": "DM2,DL5"}, KWARGS)
    assert p["runnable"] is False and "gain = 1.5, k = 2.0, gain_glomeruli = DM2,DL5" in p["why"][0]
    p = model_runs.plan({**ROW, "base_kwargs_json": json.dumps({**BOUTON, "prat_boutons": True})}, KWARGS)
    assert p["runnable"] is False and p["why"] == ["prat_boutons is not a setting fit_and_plot_mb_model takes"]
    assert model_runs.plan({k: v for k, v in ROW.items() if k != "base_kwargs_json"}, KWARGS)["runnable"] is False
    # no worker yet: not known; without its threshold, only tuned again
    assert model_runs.plan(ROW, None)["runnable"] is None
    assert model_runs.plan({k: v for k, v in ROW.items() if k != "fixed_thr_mean"}, KWARGS)["tunings"] == ["tuned"]


def baseline(env, root, row=ROW, name="E13"):  # noqa: F811
    """A baseline chosen in the Admin tab, of an experiment of one model on tensor."""
    exp = root / "tom/model_outputs" / name
    exp.mkdir(parents=True)
    pd.DataFrame([row]).to_csv(exp / "params.csv", index=False)
    (exp / "provenance.json").write_text(json.dumps({"generated": "2026-11-05T03:46:05+00:00"}))
    pd.DataFrame([[1.0]], index=pd.Index([1], name="kc_id"), columns=pd.Index(["ea @ -3"], name="odor")).to_parquet(
        exp / f"{row['tag']}_on_spike_counts.parquet")
    sources = model_data.sources(env.db)
    sid = sources[0]["id"] if sources else call(env.admin_models_add, body={"path": "tom/model_outputs"})[1]["id"]
    status, b = call(env.admin_models_baseline_add, body={"experiment": f"{sid}/{name}", "row": 0})
    assert status == 200, b
    return b["id"]


def body_of(response) -> bytes:
    async def read():
        return b"".join([chunk async for chunk in response.body_iterator])
    return asyncio.run(read())


def test_runs(env, root, tmp_path):  # noqa: F811
    db = env.db
    b1, b2 = baseline(env, root), baseline(env, root, {**ROW, "tag": "ref", "variant": "ref", "gain": 1.5}, "E44")
    a = add(db, tmp_path, "megamat", H_ODORS, extra=[("glomeruli_diagnostics", "x1 @ -2")])
    # before any worker: nothing is known to be runnable, and nothing is run
    out = call(env.model_runs_list)[1]
    assert out["worker"] is None and [b["plan"]["runnable"] for b in out["baselines"]] == [None, None]
    assert [d["id"] for d in out["datasets"]] == [a] and out["runs"] == [] and out["view"]["models"] == []
    ask = {"name": "Ann", "baseline": b1, "datasets": [a]}
    assert call(env.model_runs_create, body=ask)[0] == 422
    # the worker says what fit_mb_model takes (the stand-in's): one baseline can be run, the other not, said why
    caps = model_worker.capabilities()
    assert {"fixed_thr", "prat_claws", "claw_thr"} <= set(caps["kwargs"]) and "_wPNKC" not in caps["kwargs"] and "orn_deltas" not in caps["kwargs"]
    model_worker._say(caps)
    out = call(env.model_runs_list)[1]
    assert out["worker"]["alive"] and "kwargs" not in out["worker"] and out["worker"]["olfsysm"]["checkout"]["git_hash"] == "abc123"
    plans = {b["id"]: b["plan"] for b in out["baselines"]}
    assert plans[b1]["runnable"] is True and plans[b2]["runnable"] is False and "gain = 1.5" in plans[b2]["why"][0]
    # how its datasets would be scaled, and their panels (not the diagnostic one, unless chosen)
    status, pre = call(env.model_runs_preview, body={"datasets": [a]})
    assert status == 200 and pre["chosen"] == ["p1"] and pre["scaling"]["method"] == "hallem" and len(pre["scaling"]["odors"]) == 6
    assert len(call(env.model_runs_preview, body={"datasets": [a], "panels": ["p1", "glomeruli_diagnostics"]})[1]["scaling"]["odors"]) == 7
    # asked for: by someone, of a baseline that can be run, on datasets of glomeruli
    for bad, why in (({**ask, "name": ""}, "say who you are"), ({**ask, "baseline": b2}, "cannot be run"), ({**ask, "datasets": []}, "choose a dataset"),
                     ({**ask, "tuning": "other"}, "cannot be run other"), ({**ask, "panels": ["nope"]}, "at least one of its panels")):
        status, r = call(env.model_runs_create, body=bad)
        assert status == 422 and why in r["error"], bad
    status, r1 = call(env.model_runs_create, body=ask)
    assert status == 201 and r1["status"] == "queued" and r1["options"]["tuning"] == "fixed" and r1["options"]["panels"] == ["p1"]
    assert r1["options"]["fixed"]["fixed_thr"] == 254.4 and r1["options"]["target"] is None and r1["options"]["model"] == BOUTON
    r2 = call(env.model_runs_create, body={**ask, "tuning": "tuned"})[1]
    r3 = call(env.model_runs_create, body=ask)[1]
    assert r2["options"]["target"] == 0.05 and r2["options"]["fixed"] is None
    assert [(r["id"], r["position"]) for r in call(env.model_runs_list)[1]["runs"]] == [(r3["id"], 3), (r2["id"], 2), (r1["id"], 1)]
    # one at a time: the oldest, and no other while it runs
    job = model_runs.claim(db)
    assert job["id"] == r1["id"] and model_runs.claim(db) is None
    running = next(r for r in call(env.model_runs_list)[1]["runs"] if r["id"] == r1["id"])
    assert running["status"] == "running" and 0 <= running["progress"]["fraction"] <= 0.98 and running["progress"]["ends_at"]
    assert model_runs.worker_jobs(db)["queued"] == 2 and "Model run #1 (bouton_base), asked for by Ann" in model_runs.worker_jobs(db)["jobs"][0]["what"]
    assert call(env.model_runs_delete, path_params={"id": r1["id"]})[0] == 409
    model_worker.run(db, job, caps)
    done = model_runs.get(db, r1["id"])
    assert done["status"] == "done" and done["error"] is None, done["error"]
    res = done["result"]
    assert (res["n_kcs"], res["n_odors"], res["n_glomeruli"], res["plots"]) == (3, 6, 9, 1) and res["rate"] == pytest.approx(1 / 3)
    assert res["scaling"]["factor"] == pytest.approx(100) and res["fixed_thr"] == 254.4
    d = model_runs.run_dir(r1["id"])
    assert {p.name for p in d.iterdir()} == {"inputs", "output", "view", "run.py", "provenance.json", "README.txt", "log.txt"}
    assert {p.name for p in (d / "inputs").iterdir()} == {"responses.csv", "orn_deltas.csv", "scaling.json", "scaling_points.csv", "baseline.json",
                                                         "model_kwargs.json", orn_input.HALLEM_FILE}
    kw = json.loads((d / "inputs/model_kwargs.json").read_text())
    assert kw == {"model": BOUTON, "fixed": {"fixed_thr": 254.4, "wAPLKC_scale": 0.13, "wKCAPL_scale": 7.7e-05}, "target_sparsity": None}
    assert json.loads((d / "output/model/params.json").read_text())["wAPLKC"] == 0.13  # (as pyolfsysm names it)
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["datasets"][0]["title"] == "megamat" and prov["worker"]["olfsysm"]["checkout"]["git_hash"] == "abc123" and prov["response_rate"] == res["rate"]
    assert len(pd.read_csv(d / "inputs/scaling_points.csv")) == 54 and "writing" in (d / "log.txt").read_text()
    assert "run.py" in (d / "README.txt").read_text() and "megamat" in (d / "README.txt").read_text()
    # listed, and opened as a model's run: its KCs' spike counts to each odor
    out = call(env.model_runs_list)[1]
    listed = next(r for r in out["runs"] if r["id"] == r1["id"])
    assert listed["run"] == f"olfdb:runs/{r1['id']}/run{r1['id']}_on_spike_counts.parquet" and listed["title"] == f"bouton_base on #{a} (fixed)"
    m = out["view"]["models"][0]
    assert m["runs"][0]["id"] == listed["run"] and m["params"]["tuning"] == "fixed" and m["n_spikes"]["value"] == 2
    assert out["view"]["experiments"][m["experiment"]]["folder"] == f"run #{r1['id']} (bouton_base on #{a} (fixed))"
    status, shown = call(env.models_run, params={"id": listed["run"]})
    assert status == 200 and shown["job"] == r1["id"] and shown["kcs"]["labels"] == ["10", "11", "12"] and shown["kcs"]["groups"] == ["ab", "g", "ab"]
    assert [o["odor"] for o in shown["odors"]] == [f"{o} @ -3" for o in H_ODORS] and shown["provenance"]["asked_by"] == "Ann"
    # saved: its spike counts as the model wrote them, and everything (but the copy the tab reads)
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "path_params": {"id": r1["id"]}}
    from starlette.requests import Request
    csv_ = asyncio.run(env.model_runs_download(Request({**scope, "query_string": b"what=csv"})))
    assert Path(csv_.path) == d / "output/model/spike_counts.csv" and "spike_counts.csv" in csv_.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(body_of(asyncio.run(env.model_runs_download(Request({**scope, "query_string": b""}))))))
    top = f"olfdb_model_run_{r1['id']}_bouton_base"
    assert {f"{top}/run.py", f"{top}/provenance.json", f"{top}/output/model/corr.pdf", f"{top}/output/model/spike_counts.csv",
            f"{top}/inputs/orn_deltas.csv"} <= set(z.namelist()) and not any("/view/" in n for n in z.namelist())
    # the next, tuned again: the target passed, no weights; one queued stopped, a done one deleted
    job = model_runs.claim(db)
    assert job["id"] == r2["id"]
    model_worker.run(db, job, caps)
    assert json.loads((model_runs.run_dir(r2["id"]) / "inputs/model_kwargs.json").read_text())["target_sparsity"] == 0.05
    assert model_runs.get(db, r2["id"])["result"]["fixed_thr"] == 111.0  # (as the model tuned it)
    assert call(env.model_runs_cancel, body={}, path_params={"id": r3["id"]})[1]["status"] == "cancelled" and model_runs.claim(db) is None
    assert call(env.model_runs_delete, path_params={"id": r1["id"]}) == (200, {"deleted": True}) and not d.exists()
    assert call(env.models_run, params={"id": listed["run"]})[0] == 404
    assert call(env.model_runs_download, path_params={"id": r1["id"]})[0] == 404


def test_run_failed_and_interrupted(env, root, tmp_path):  # noqa: F811
    db = env.db
    b1 = baseline(env, root)
    caps = model_worker.capabilities()
    model_worker._say(caps)
    # a model that fails: said, with the end of its log; its folder kept
    boom = add(db, tmp_path, "boom", ["ea", "eb", "pa", "2h", "ms", "1-6ol"], extra=[("p0", "boom @ -2")], panel="p1")
    ask = {"name": "Ann", "baseline": b1}
    r = call(env.model_runs_create, body={**ask, "datasets": [boom], "panels": ["p0"]})[1]
    model_worker.run(db, model_runs.claim(db), caps)
    failed = model_runs.get(db, r["id"])
    assert failed["status"] == "failed" and "RuntimeError: the model blew up" in failed["error"] and (model_runs.run_dir(r["id"]) / "log.txt").is_file()
    # a dataset deleted (or no longer scalable) since it was asked for: failed, saying so
    r = call(env.model_runs_create, body={**ask, "datasets": [boom]})[1]
    job = model_runs.claim(db)
    job["options"]["datasets"] = [999]
    model_worker.run(db, job, caps)
    assert model_runs.get(db, r["id"])["status"] == "failed" and "no dataset #999" in model_runs.get(db, r["id"])["error"]
    # one left running by a worker that stopped
    r = call(env.model_runs_create, body={**ask, "datasets": [boom]})[1]
    model_runs.claim(db)
    assert model_runs.interrupt_running(db) == [r["id"]] and model_runs.get(db, r["id"])["status"] == "interrupted"
    # a worker that has not said so for a while is taken to be gone
    w = json.loads((model_runs.root() / model_runs.WORKER_FILE).read_text())
    (model_runs.root() / model_runs.WORKER_FILE).write_text(json.dumps({**w, "at": w["at"] - 1000}))
    assert call(env.model_runs_list)[1]["worker"]["alive"] is False
