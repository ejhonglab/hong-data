"""Tests of MetaCyc bundles (metacyc_build.py) and their views as reference datasets
(metacyc.py), on a small MetaCyc made up here. Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import asyncio
import json
import sqlite3

import numpy as np
import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request

from app import metacyc, metacyc_build, odormeta, sandbox
from tests.test_reference import INCHI_LIN, env  # noqa: F401 (fixtures)
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

INCHI_GER = "InChI=1S/C10H18O/c1-9(2)5-4-6-10(3)7-8-11/h5,7,11H,4,6,8H2,1-3H3/b10-7+"
INCHI_RLIN = INCHI_LIN + "/t10-/m0/s1"  # (R)-linalool: linalool's skeleton
INCHI_ACETIC = "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)"
INCHI_ACETATE = INCHI_ACETIC + "/p-1"  # as MetaCyc keeps it


def tsv(path, header, rows, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("\n".join("\t".join(r) for r in [header, *rows]).encode(encoding) + b"\n")


CPD_COLS = ["Compound", "Common-Name", "SMILES", "InChI", "InChI-Key", "PubChem", "Chemical Formula", "Pathways of compound",
            "General Pathways of compound", "Ontology - direct parents of entity", "Kegg", "Reactions that consume compound",
            "Reactions that produce compound", "Reactions of compounds"]
PWY_COLS = ["Pathways", "Object ID", "Common-Name", "Primary-Products", "Primary-Reactants", "Products of pathway",
            "Reactants of pathway", "Reactions of pathway", "Ontology - pathway type", "Species"]
RXN_COLS = ["Reaction", "EC-Number", "In-Pathway", "KEGG reaction", "Substrates", "Pathways of a reaction", "Products of reaction",
            "Reactants of reaction", "Substrates (reactants and products) of a reaction", "Object ID", "Common-Name"]


def make_raw(d):
    """A MetaCyc made up, as SmartTables in folder `d` (as BioCyc/raw): geranyl diphosphate
    makes geraniol and myrcene (PWY-A); geraniol is broken down to geranial and a
    monoterpenol (PWY-B); a superpathway has both; acetate is made in PWY-C. Linalool is in no
    reaction, but is a monoterpenol (and so in PWY-B through that class)."""
    link = "<a href='http://pubchem.ncbi.nlm.nih.gov/summary/summary.cgi?cid=~A'>{}</a>"
    cpds = [  # id, name, inchi, cid, pathways, general pathways, parents (id, name)
        ("GERANIOL", "geraniol", INCHI_GER, "637566", "PWY-A // PWY-B // PWY-S", "PWY-A // PWY-B // PWY-S", [("Monoterpenols", "a monoterpenol")]),
        ("MYRCENE", "&beta;-myrcene", "", "31253", "PWY-A // PWY-S", "PWY-A // PWY-S", [("Monoterpenes", "a monoterpene")]),
        ("GPP", "geranyl diphosphate", "", "", "PWY-A // PWY-S", "PWY-A // PWY-S", []),
        ("GERANIAL", "geranial", "", "", "PWY-B // PWY-S", "PWY-B // PWY-S", []),
        ("LINALOOL", "(<i>R</i>)-linalool", INCHI_RLIN, "443158", "", "PWY-B", [("Monoterpenols", "a monoterpenol")]),
        ("ACET", "acetate", INCHI_ACETATE, "175", "PWY-C", "PWY-C", []),
        ("WATER", "H<sub>2</sub>O", "", "962", "PWY-C", "PWY-C", []),
    ]
    tsv(d / "compounds/all_metacyc_compounds_with_gen_pathways__frame_ids__2024-02-05.tsv", CPD_COLS,
        [[i, n, "", inchi, "", link.format(cid) if cid else "", "", p, g, " // ".join(k for k, _ in par), "", "", "", ""]
         for i, n, inchi, cid, p, g, par in cpds])
    tsv(d / "compounds/all_metacyc_compounds_with_gen_pathways__common_name__2024-02-05.tsv", CPD_COLS,
        [[n, n, "", inchi, "", cid, "", "", "", " // ".join(m for _, m in par), "", "", "", ""] for _, n, inchi, cid, _, _, par in cpds])
    pwys = [  # id, name, products, reactants, reactions, types (id, name), species (id, name)
        ("PWY-A", "monoterpene biosynthesis", ["GERANIOL", "MYRCENE"], ["GPP"], ["R1", "R2"],
         [("SECONDARY-METABOLITE-BIOSYNTHESIS", "Secondary Metabolite Biosynthesis")], [("TAX-1", "Abies grandis"), ("TAX-2", "Mentha")]),
        ("PWY-B", "geraniol degradation", ["GERANIAL", "Monoterpenols"], ["GERANIOL"], ["R3", "R4"],
         [("Alcohol-Degradation", "Alcohol Degradation")], [("TAX-2", "Mentha")]),
        ("PWY-S", "superpathway of monoterpenes", ["MYRCENE", "Monoterpenols"], ["GPP"], ["R1", "R2", "R3", "R4"],
         [("Super-Pathways", "Superpathways"), ("SECONDARY-METABOLITE-BIOSYNTHESIS", "Secondary Metabolite Biosynthesis")], []),
        ("PWY-C", "acetate formation", ["ACET"], ["ACETYL-X"], ["R5"], [("Other-biosynthesis", "Other Biosynthesis")], [("TAX-3", "Escherichia coli")]),
    ]
    names = {c[0]: c[1] for c in cpds} | {"Monoterpenols": "a monoterpenol", "ACETYL-X": "an acetyl donor", "PPI": "diphosphate",
                                         "NAD": "NAD<sup>+</sup>", "NADH": "NADH"}
    js = lambda ids, of=lambda i: i: " // ".join(of(i) for i in ids)  # noqa: E731
    tsv(d / "pathways/all_metacyc_pathways__frame_ids__2024-01-24.txt", PWY_COLS,
        [[i, i, n, "", js(r[:1]), js(p), js(r), js(rx), js(t, lambda x: x[0]), js(s, lambda x: x[0])] for i, n, p, r, rx, t, s in pwys])
    tsv(d / "pathways/all_metacyc_pathways__common_name__2024-01-24.txt", PWY_COLS,
        [[n, i, n, "", js(r[:1], names.get), js(p, names.get), js(r, names.get), "", js(t, lambda x: x[1]), js(s, lambda x: x[1])]
         for i, n, p, r, rx, t, s in pwys])
    rxns = [  # id, equation, EC, reactants, products (as listed by frame ID: reactants first, whatever the arrow)
        ("R1", "geranyl diphosphate  ->  geraniol + diphosphate", "EC-3.1.7.11", ["GPP"], ["GERANIOL", "PPI"]),
        ("R2", "geranyl diphosphate  ->  &beta;-myrcene + diphosphate", "EC-4.2.3.15", ["GPP"], ["MYRCENE", "PPI"]),
        ("R3", "geraniol + NAD<sup>+</sup>  <-->  geranial + NADH", "EC-1.1.1.347", ["GERANIOL", "NAD"], ["GERANIAL", "NADH"]),
        ("R4", "geranial  ->  a monoterpenol", "", ["GERANIAL"], ["Monoterpenols"]),
        ("R5", "acetate  <-  an acetyl donor + H<sub>2</sub>O", "", ["ACETYL-X", "WATER"], ["ACET"]),
    ]
    inp = {r: [p[0] for p in pwys if r in p[4]] for r, *_ in rxns}
    tsv(d / "reactions/all_metacyc_reactions__frame_ids.tsv", RXN_COLS,
        [[i, ec, js(inp[i]), "", "", "", js(p), js(r), "", i, ""] for i, _, ec, r, p in rxns])
    tsv(d / "reactions/all_metacyc_reactions__common_name.tsv", RXN_COLS,
        [[eq, ec, "", "", "", "", js(p, names.get), js(r, names.get), "", i, ""] for i, eq, ec, r, p in rxns])
    tsv(d / "organisms/species-from-metacyc_species.txt", ["Organisms", "Object ID", "Common-Name", "Strain-Name", "PGDB-Unique-ID",
                                                           "PGDB-Name", "Synonyms"],
        [["Abies grandis", "TAX-1", "Abies grandis", "", "", "", "grand fir // lowland fir"], ["Mentha", "TAX-2", "Mentha", "", "", "", ""],
         ["Escherichia coli", "TAX-3", "Escherichia coli", "", "", "", ""]])
    tsv(d / "organisms/all_organism_databases.txt", ["All BioCyc Organism Databases", "Organism name", "Environment", "Depth-or-Altitude",
                                                     "Health-or-Disease-State", "Host", "Pathogenicity", "Relationship-to-Oxygen",
                                                     "Temperature-Range", "Trophic-Level", "Strain-Name"],
        [["ECOLI", "Escherichia coli K-12 substr. MG1655", "feces // gut", "", "", "TAX-9606", "", "FACULTATIVE", "MESOPHILE", "", "K-12"],
         ["GCF_1", "Escherichia coli O157", "feces", "", "", "", "pathogenic", "FACULTATIVE", "", "", ""],
         ["GCF_2", "Bacillus subtilis 168", "soil", "", "", "", "", "AEROBE", "", "", ""]])
    # a summary over several lines, with frames, citations, and a byte of Windows-1252 (an en dash)
    (d / "pathways/pathway_summary.txt").write_bytes(
        b"Pathways\tObject ID\tCommon-Name\tSummary\n"
        b"monoterpene biosynthesis\tPWY-A\tmonoterpene biosynthesis\tAbout this Pathway\n\n|FRAME: GERANIOL| is made\n"
        b"from |FRAME: GPP \"GPP\"| |CITS: [123]|, 10\x9620 times.\n"
        b"acetate formation\tPWY-C\tacetate formation\tAcetate.\n")
    tsv(d / "pathways/all_metacyc_pathways(2).txt",
        ["Pathways", "Common-Name", "Object ID", "Reaction-List", "Species", "Ontology", "Products", "Reactants", "Primary-Products",
         "Primary-Reactants", "Taxonomic-Range", "Engineered?", "Summary"],
        [["monoterpene biosynthesis", "monoterpene biosynthesis", "PWY-A", "", "", "", "", "", "", "", "Tracheophyta", "", "line one\nline two"]])
    (d / "ontology").mkdir(exist_ok=True)
    (d / "ontology/pathway_ontology.txt").write_text(
        "    Pathways\n        Biosynthesis (3)\n            Secondary Metabolite Biosynthesis (2)\n                monoterpene biosynthesis\n"
        "        Degradation/Utilization/Assimilation (1)\n            Alcohol Degradation (1)\n                geraniol degradation\n")
    return d


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("metacyc")
    out = d / "metacyc-test.sqlite"
    metacyc_build.build(make_raw(d / "raw"), out, version="99.0", log=lambda *a: None)
    return out


ODORS = [  # as Database.list_odors gives them
    {"id": 1, "name": "geraniol", "abbrev": "ger", "aliases": [], "cid": 637566, "inchi": INCHI_GER, "metacyc": None},
    {"id": 2, "name": "beta-myrcene", "abbrev": "B-myr", "aliases": [], "cid": None, "inchi": None, "metacyc": "MYRCENE"},
    {"id": 3, "name": "linalool", "abbrev": "lin", "aliases": [], "cid": 6549, "inchi": INCHI_LIN, "metacyc": None},
    {"id": 4, "name": "acetic acid", "abbrev": "aa", "aliases": [], "cid": 176, "inchi": INCHI_ACETIC, "metacyc": None},
    {"id": 5, "name": "ethyl acetate", "abbrev": "ea", "aliases": [], "cid": 8857, "inchi": None, "metacyc": None},
]


def test_build(bundle_path):
    b = metacyc.Bundle(bundle_path)
    assert b.about["kind"] == "metacyc" and b.about["version"] == "99.0" and b.about["counts"]["compounds"] == 7
    assert b.compounds["MYRCENE"]["name"] == "β-myrcene" and b.compounds["LINALOOL"]["name"] == "(R)-linalool"
    assert b.compounds["GERANIOL"]["cid"] == 637566 and b.compounds["GERANIOL"]["parents"] == ["Monoterpenols"]
    assert b.names["WATER"] == "H₂O" and b.names["NAD"] == "NAD⁺" and b.names["Monoterpenols"] == "a monoterpenol"
    # an equation's arrow: reactants (as listed by ID) to products, either way it is written; or reversible
    assert {k: r["direction"] for k, r in b.reactions.items()} == {"R1": "L2R", "R2": "L2R", "R3": "REV", "R4": "L2R", "R5": "L2R"}
    assert b.reactions["R1"]["ec"] == ["3.1.7.11"] and b.reactions["R5"]["reactants"] == ["ACETYL-X", "WATER"]
    pa, ps = b.pathways["PWY-A"], b.pathways["PWY-S"]
    assert sorted(ps["subs"]) == ["PWY-A", "PWY-B"] and pa["super"] == ["PWY-S"] and pa["species"] == ["TAX-1", "TAX-2"]
    assert pa["summary"] == "About this Pathway\n\ngeraniol is made from GPP, 10–20 times."  # (cited, frames named, wrapped)
    assert pa["taxonomic_range"] == ["Tracheophyta"] and b.pathways["PWY-B"]["summary"] is None
    assert b.pathway_classes["Alcohol-Degradation"] == {"id": "Alcohol-Degradation", "name": "Alcohol Degradation",
                                                        "parent": "Degradation/Utilization/Assimilation"}
    assert b.species["TAX-1"]["synonyms"] == ["grand fir", "lowland fir"]
    # genome databases by the longest species name they start with
    assert sorted(g["id"] for g in b.genomes["TAX-3"]) == ["ECOLI", "GCF_1"] and "TAX-9" not in b.genomes
    assert metacyc_build.plain("CO<SUB>2</SUB> + <i>cis</i>-x &beta; <sub>n-1</sub>") == "CO₂ + cis-x β ₙ₋₁"


def test_build_errors(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        metacyc_build.build(tmp_path, tmp_path / "x.sqlite", log=lambda *a: None)
    with pytest.raises(metacyc.MetacycError):
        metacyc.read_about(tmp_path / "x.sqlite")  # (none)
    sqlite3.connect(tmp_path / "other.db").execute("CREATE TABLE about (key TEXT, value TEXT)").connection.commit()
    with pytest.raises(metacyc.MetacycError, match="not a MetaCyc bundle"):
        metacyc.read_about(tmp_path / "other.db")


def test_match_odors(bundle_path):
    b = metacyc.bundle(bundle_path)
    got = [(m["odor"]["name"], m["compound"], m["how"]) for m in metacyc.match_odors(b, ODORS)]
    assert got == [("geraniol", "GERANIOL", "inchi"), ("beta-myrcene", "MYRCENE", "metacyc"), ("linalool", "LINALOOL", "skeleton"),
                   ("acetic acid", "ACET", "skeleton"), ("ethyl acetate", None, None)]
    assert metacyc.match_odors(b, [{**ODORS[0], "inchi": None}])[0]["how"] == "cid"
    assert metacyc.match_odors(b, [{"id": 9, "name": "Acetate", "cid": None, "inchi": None}])[0]["how"] == "name"
    assert metacyc.odors_fingerprint(ODORS) != metacyc.odors_fingerprint(ODORS[:-1])


def test_pathway_matrix(bundle_path):
    rec, v = metacyc.parse(bundle_path, {"layout": "metacyc", "view": "pathways"}, ODORS)
    assert [r["label"] for r in rec["rows"]] == ["geraniol", "beta-myrcene", "linalool", "acetic acid"]
    assert [c["metacyc"] for c in rec["cols"]] == ["PWY-B", "PWY-C", "PWY-A", "PWY-S"]  # (by class, then name)
    assert v.tolist() == [[1, 0, 1, 1], [0, 0, 1, 1], [0.5, 0, 0, 0], [0, 1, 0, 0]]  # linalool through a monoterpenol
    assert rec["kind"] == "sequential" and rec["chemical_axis"] == "rows" and rec["rows"][2]["metacyc"] == "LINALOOL"
    assert "1 of the odor table's 5 chemicals are not in MetaCyc" in rec["notes"][0] and "ethyl acetate" in rec["notes"][0]
    assert rec["metacyc"]["view"] == "pathways" and rec["metacyc"]["version"] == "99.0"
    rec, v = metacyc.parse(bundle_path, {"view": "pathways", "classes": "never"}, ODORS)
    assert rec["kind"] == "binary" and v[2].tolist() == [0, 0, 0, 0] and any("in no pathway" in n for n in rec["notes"])


def test_lca(bundle_path):
    # through reactions: geraniol and myrcene both come from geranyl diphosphate (2 + 2);
    # linalool is taken as a monoterpenol, made from geranial, from geraniol (4 edges), and
    # their LCA is the reaction making geraniol (other than geraniol itself)
    rec, v = metacyc.parse(bundle_path, {"view": "lca"}, ODORS)
    assert [r["label"] for r in rec["rows"]] == ["geraniol", "beta-myrcene", "linalool", "acetic acid"]
    assert rec["pairwise"] == "distance" and rec["cols"] == rec["rows"] and rec["kind"] == "sequential"
    nan = np.nan
    np.testing.assert_array_equal(v, [[0, 4, 6, nan], [4, 0, 8, nan], [6, 8, 0, nan], [nan, nan, nan, 0]])
    assert any("linalool as a monoterpenol" in n for n in rec["notes"]) and any("3 pairs have no common ancestor" in n for n in rec["notes"])
    # linalool left out without classes
    rec, _ = metacyc.parse(bundle_path, {"view": "lca", "classes": "never"}, ODORS)
    assert "linalool" not in [r["label"] for r in rec["rows"]]
    # through pathways: geraniol and myrcene are both in the pathway that makes them; linalool
    # in geraniol's degradation through its class (an edge more)
    rec, v = metacyc.parse(bundle_path, {"view": "lca", "via": "pathways"}, ODORS)
    assert rec["metacyc"]["max_pathways"] == 240
    np.testing.assert_array_equal(v[:3, :3], [[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    # hubs left out: geraniol is in three pathways, so is taken as the class it is of (a
    # monoterpenol, made in the superpathway that makes myrcene, and in geraniol's degradation)
    rec, v = metacyc.parse(bundle_path, {"view": "lca", "via": "pathways", "max_pathways": 2}, ODORS)
    assert [r["label"] for r in rec["rows"]] == ["geraniol", "beta-myrcene", "linalool", "acetic acid"]
    assert v[0, 1] == 2 and v[0, 2] == 2 and any("geraniol as a monoterpenol" in n for n in rec["notes"])
    rec, v = metacyc.parse(bundle_path, {"view": "lca", "via": "pathways", "max_pathways": 1}, ODORS)
    assert [r["label"] for r in rec["rows"]] == ["acetic acid"]  # (the rest all in more than one)


def test_explain(bundle_path):
    b = metacyc.bundle(bundle_path)
    res = metacyc.explain(b, {"view": "lca"}, "GERANIOL", "MYRCENE")
    assert res["distance"] == 4 and [lca["node"] for lca in res["lcas"]] == ["GPP"]
    assert res["lcas"][0]["paths"] == [["GPP", "R1", "GERANIOL"], ["GPP", "R2", "MYRCENE"]]
    assert res["nodes"]["R1"]["kind"] == "reaction" and res["nodes"]["R1"]["pathways"][0] == {"id": "PWY-A", "name": "monoterpene biosynthesis"}
    res = metacyc.explain(b, {"view": "lca"}, "LINALOOL", "GERANIOL")
    assert res["distance"] == 6 and res["entries"][0] == [{"id": "Monoterpenols", "name": "a monoterpenol", "edges": 0}]
    assert res["lcas"][0]["paths"] == [["R1", "GERANIOL", "R3", "GERANIAL", "R4", "Monoterpenols"], ["R1", "GERANIOL"]]
    assert res["lcas"][0]["ends"] == [{"entry": "Monoterpenols", "edges": 0}, {"entry": "GERANIOL", "edges": 0}]
    # through pathways: geraniol's degradation (and the superpathway) has geraniol and makes the
    # monoterpenol that linalool is taken as
    res = metacyc.explain(b, {"view": "lca", "via": "pathways"}, "LINALOOL", "GERANIOL")
    assert res["distance"] == 2 and sorted(lca["node"] for lca in res["lcas"]) == ["PWY-B", "PWY-S"]
    assert next(lca for lca in res["lcas"] if lca["node"] == "PWY-B")["paths"] == [["PWY-B", "Monoterpenols"], ["PWY-B", "GERANIOL"]]
    assert metacyc.explain(b, {"view": "lca"}, "ACET", "GERANIOL")["distance"] is None
    assert metacyc.explain(b, {"view": "lca", "classes": "never"}, "LINALOOL", "GERANIOL")["missing"] == ["LINALOOL"]


def test_pathway_and_compound(bundle_path):
    b = metacyc.bundle(bundle_path)
    p = metacyc.pathway(b, "PWY-A", {"GERANIOL": {"id": 1, "name": "geraniol", "abbrev": "ger"}})
    assert [(r["id"], r["left"], r["right"], r["side_right"]) for r in p["reactions"]] == [
        ("R1", ["GPP"], ["GERANIOL"], ["PPI"]), ("R2", ["GPP"], ["MYRCENE"], ["PPI"])]  # (diphosphate a cofactor)
    assert p["compounds"]["GERANIOL"]["odor"]["abbrev"] == "ger" and p["super"] == [{"id": "PWY-S", "name": "superpathway of monoterpenes"}]
    assert p["types"][0]["lineage"] == ["Biosynthesis", "Secondary Metabolite Biosynthesis"]
    assert [s["name"] for s in p["species"]] == ["Abies grandis", "Mentha"] and p["species"][0]["synonyms"] == ["grand fir", "lowland fir"]
    assert p["links"]["GERANIOL"] == [{"id": "PWY-B", "name": "geraniol degradation"}, {"id": "PWY-S", "name": "superpathway of monoterpenes"}]
    ecoli = metacyc.pathway(b, "PWY-C", {})["species"][0]
    assert ecoli["genomes"] == 2 and ecoli["environment"] == [{"value": "feces", "n": 2}, {"value": "gut", "n": 1}]
    assert ecoli["oxygen"] == [{"value": "FACULTATIVE", "n": 2}] and ecoli["host"] == [{"value": "TAX-9606", "n": 1}]
    assert metacyc.pathway(b, "PWY-NONE", {}) is None
    c = metacyc.compound(b, "LINALOOL", {})
    assert c["pathways"] == [] and c["general_pathways"] == [{"id": "PWY-B", "name": "geraniol degradation"}]
    assert c["classes"] == [{"id": "Monoterpenols", "name": "a monoterpenol"}] and c["inchi"] == INCHI_RLIN
    k = metacyc.compound(b, "Monoterpenols", {})
    assert k["is_class"] and [q["id"] for q in k["pathways"]] == ["PWY-B", "PWY-S"] and k["n_produced"] == 1
    assert metacyc.compound(b, "NOTHING", {}) is None
    s = metacyc.search(b, "geraniol")
    assert [x["id"] for x in s["pathways"]] == ["PWY-B"] and [x["id"] for x in s["compounds"]] == ["GERANIOL"]
    assert metacyc.search(b, "x") == {"pathways": [], "compounds": []}


def path_call(endpoint, path_params, params=None):
    """An endpoint's response (not only JSON) to a GET."""
    from urllib.parse import urlencode

    scope = {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": urlencode(params or {}).encode(),
             "path_params": path_params}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return asyncio.run(endpoint(Request(scope, receive)))


def test_reference_dataset(env, bundle_path, tmp_path):  # noqa: F811
    """A bundle added in the Admin tab as two reference datasets, made again when the odor table changes."""
    sandbox.activate({"read_dirs": [str(bundle_path.parent)], "write_dirs": []})
    c = call(env.admin_reference_check, body={"path": str(bundle_path)})[1]
    assert c["ok"] and c["fields"]["kind"] == "metacyc" and c["fields"]["about"]["version"] == "99.0"
    assert c["fields"]["used"] == {"layout": "metacyc", "view": "pathways", "classes": "fallback"}
    assert c["preview"]["metacyc"]["view"] == "pathways" and c["preview"]["chemicals"]["matched"] == 1  # (of the env's odor table: linalool)
    lca = {"layout": "metacyc", "view": "lca"}
    c = call(env.admin_reference_check, body={"path": str(bundle_path), "fields": lca})[1]
    assert c["fields"]["used"] == {"layout": "metacyc", "view": "lca", "classes": "fallback", "via": "reactions", "max_pathways": None}
    assert c["preview"]["pairwise"] == "distance"
    for title, fields in (("MetaCyc pathways", {"layout": "metacyc", "view": "pathways"}), ("MetaCyc LCA", lca)):
        status, r = call(env.admin_reference_add, body={"path": str(bundle_path), "title": title, "fields": fields})
        assert status == 200 and r["state"] == "ok", r
    status, r = call(env.reference_get, path_params={"ref_id": "metacyc-lca"})
    # the env's odor table: ethyl acetate, 2-heptanone, hexanal and linalool (as a monoterpenol)
    assert status == 200 and r["square"] and r["pairwise"] == "distance" and r["meta"]["metacyc"]["via"] == "reactions"
    assert [x["label"] for x in r["rows"]] == ["linalool"] and r["cols"][0]["chem"]["name"] == "linalool"
    assert r["rows"][0]["metacyc"] == "LINALOOL" and r["chemicals"]["matched"] == 1
    # the odor table changed: made again
    rows, _, _ = odormeta.parse_sheet("name,abbrev,cid,metacyc\ngeraniol,ger,637566,\nmyrcene,myr,,MYRCENE\n")
    env.db.apply_odor_sync(added=rows, changed=[], deleted=[], source="test", fingerprint="1", details={}, when="2026-10-19T00:00:00")
    r = call(env.reference_get, path_params={"ref_id": "metacyc-lca"})[1]
    assert [x["label"] for x in r["rows"]] == ["geraniol", "linalool", "myrcene"] and r["shape"] == [3, 3]
    listed = {d["id"]: d for d in call(env.reference_list)[1]["datasets"]}
    assert listed["metacyc-lca"]["pairwise"] == "distance" and listed["metacyc-pathways"]["shape"] == [3, 3]
    # looked into
    pp = {"ref_id": "metacyc-pathways"}
    status, p = call(env.metacyc_pathway, path_params={**pp, "frame": "PWY-A"})
    assert status == 200 and p["compounds"]["GERANIOL"]["odor"]["name"] == "geraniol" and p["compounds"]["MYRCENE"]["odor"]["abbrev"] == "myr"
    assert call(env.metacyc_pathway, path_params={**pp, "frame": "PWY-X"})[0] == 404
    assert call(env.metacyc_compound, path_params={**pp, "frame": "GERANIOL"})[1]["odor"]["abbrev"] == "ger"
    res = call(env.metacyc_lca, path_params={"ref_id": "metacyc-lca"}, params={"a": "GERANIOL", "b": "MYRCENE"})[1]
    assert res["distance"] == 4 and call(env.metacyc_lca, path_params=pp)[0] == 400
    assert call(env.metacyc_search, path_params=pp, params={"q": "monoterpene"})[1]["pathways"][0]["id"] == "PWY-A"
    svg = path_call(env.metacyc_structure, {**pp, "frame": "GERANIOL"})
    assert svg.status_code == 200 and b"<svg" in svg.body
    assert path_call(env.metacyc_structure, {**pp, "frame": "GPP"}).status_code == 404  # (no InChI)
    with pytest.raises(HTTPException) as e:
        call(env.metacyc_pathway, path_params={"ref_id": "fruits", "frame": "PWY-A"})
    assert e.value.status_code == 404


def test_register(monkeypatch, tmp_path):
    """The script adds both views through the Admin tab's API."""
    sent = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def urlopen(req, timeout):
        body = json.loads(req.data)
        sent.append((req.full_url, body))
        return Resp({"ok": True, "id": body["fields"]["view"], "state": "ok", "error": None, "restart": False, "steps": []})

    monkeypatch.setattr(metacyc_build.urllib.request, "urlopen", urlopen)
    logged = []
    assert metacyc_build.register("http://olfdb:8000/", tmp_path / "m.sqlite", "27.0", log=logged.append) == 0
    assert [u for u, _ in sent] == ["http://olfdb:8000/api/admin/references"] * 2
    assert [b["fields"] for _, b in sent] == [{"layout": "metacyc", "view": "pathways"}, {"layout": "metacyc", "view": "lca"}]
    assert sent[1][1]["title"] == "MetaCyc 27.0: pathway LCA distance between odors" and len(logged) == 2
