"""A model fit with more odors than a fly has, against one fit to the same flies with the odors that fly has: the same ROIs and names?

Of the flies of a fit's manifest (as the site makes it: `fits.manifest`) that have the odors asked for (REQUIRE), with their
anatomical stacks left unread, two models are made without training (Cellpose's weights are another fit's, the same for both; no
atlas, which no odor enters): A with all the odors those flies share, its images from the odors every fly of the manifest has
(`Options.image_odors`); B from those flies with only the odors every fly of the manifest has. Each then predicts held-out flies
given only B's odors, and the two predictions are compared voxel by voxel and name by name; and A with all its odors, to say what
the others add. Flies of the manifest that lack some of the odors asked for (read from their folders, as a prediction reads them)
are predicted by A and by a model made the same way with the odors each has.

Usage (in the fit worker's image, on its GPU, with the al-diag-seg to check on PYTHONPATH; tensor at /local/tensor):
same_as_fewer.py <manifest.json> <Cellpose weights: a fit's model/segmenter>"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from alseg import model as am
from alseg import olfdb as ao
from alseg.fly import load_fly

REQUIRE = ["acetophenone @ -4", "valeric acid @ -3"]
man = json.loads(Path(sys.argv[1]).read_text())
weights = sys.argv[2]
odors_of = lambda f: {o for r in f["recordings"] for o in r["odors"]}  # noqa: E731
shared = [o for o in dict.fromkeys(o for r in man["flies"][0]["recordings"] for o in r["odors"]) if all(o in odors_of(f) for f in man["flies"])]
chosen = [f for f in man["flies"] if all(o in odors_of(f) for o in REQUIRE)]
others = [f for f in man["flies"] if f not in chosen]
print(f"{len(man['flies'])} flies share {len(shared)} odors; {len(chosen)} have {' and '.join(REQUIRE)} too", flush=True)
tmp = Path(tempfile.mkdtemp())
(tmp / "manifest.json").write_text(json.dumps({**man, "flies": [{**f, "stacks": []} for f in chosen]}))
data = ao.load_export(tmp / "manifest.json")
flies = data["flies"]
held = [flies[int((i + 0.5) * len(flies) / 4)]["name"] for i in range(4)]
core = [data["odors"].index(o) for o in shared]
print(f"their {len(data['odors'])} odors; beside the shared ones: {[o for o in data['odors'] if o not in shared]}; held out: {held}", flush=True)
few = [{**f, "responses": f["responses"][core], "odors": list(shared)} for f in flies]
kw = dict(segmenter="cellpose", atlas=False)
A = am.fit(flies, held, am.Options(image_odors=list(shared), **kw), weights=weights, odors=data["odors"], aliases=data["aliases"],
           glomeruli=data["glomeruli"])
B = am.fit(few, held, am.Options(**kw), weights=weights, odors=list(shared), aliases=data["aliases"], glomeruli=data["glomeruli"])
print(f"A: {len(A.odors)} odors, images from {len(A.image_odors())}; B: {len(B.odors)} odors; the same components: "
      f"{np.array_equal(A.pca.components_, B.pca.components_) and np.array_equal(A.pca.mean_, B.pca.mean_)}", flush=True)


def same(pa, pb):
    labels = all(np.array_equal(pa["labels"][k], pb["labels"][k]) for k in pb["labels"])
    a, b = pa["identities"]["pred"], pb["identities"]["pred"]
    cost = max([abs(a[l][1] - b[l][1]) for l in b if l in a] or [0.0])
    return labels, {l: v[0] for l, v in a.items()} == {l: v[0] for l, v in b.items()}, cost


def scores(model, f):
    res = model.evaluate([f])
    s = next(r for r in res["summary"] if r["rois"] == "pred")
    return f"{s['n']} ROIs, found {100 * s['recall_2d']:.0f}%, right {100 * s['precision_2d']:.0f}%, named {res['identity'][0]['correct']}/{res['identity'][0]['total']}"


print("\nheld-out flies given only the shared odors: A against B")
for f, g in zip(flies, few):
    if f["name"] not in held:
        continue
    pa, pb = A.predict(g), B.predict(g)
    labels, names, cost = same(pa, pb)
    full = A.predict(f)
    moved = sum(full["identities"]["pred"].get(l, (None,))[0] != v[0] for l, v in pb["identities"]["pred"].items())
    print(f"  {f['name']}: ROIs the same {labels}, names the same {names} (costs within {cost:.1e}); with A's other odors too: ROIs the same "
          f"{np.array_equal(full['labels']['pred'], pb['labels']['pred'])}, {moved} names differ", flush=True)
    print(f"      B: {scores(B, g)}\n      A, shared odors: {scores(A, g)}\n      A, all its odors: {scores(A, f)}", flush=True)
print("\nflies that lack some of A's odors, as a prediction reads them: A against a model fit, to the same flies, with the odors each has")
made = {tuple(shared): B}
for e in others[::max(1, len(others) // 6)][:6]:
    d = Path(e["recordings"][0]["responses"]).parent.parent
    fa = load_fly(d, A, raw_dir=e["raw_dir"], stack=False)
    key = tuple(fa["odors"])
    if key not in made:  # (its images from the shared odors too, as the site's fit of those flies with those odors has them)
        idx = [data["odors"].index(o) for o in key]
        made[key] = am.fit([{**f, "responses": f["responses"][idx], "odors": list(key)} for f in flies], held,
                           am.Options(image_odors=list(shared), **kw), weights=weights, odors=list(key), aliases=data["aliases"],
                           glomeruli=data["glomeruli"])
    labels, names, cost = same(A.predict(fa), made[key].predict(load_fly(d, made[key], raw_dir=e["raw_dir"], stack=False)))
    print(f"  {e['dir']}: has {len(key)} of A's {len(A.odors)} odors (lacks {', '.join(fa['missing'])}); ROIs the same {labels}, names the same "
          f"{names} (costs within {cost:.1e})", flush=True)
