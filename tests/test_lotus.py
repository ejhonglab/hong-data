"""Tests of LOTUS bundles (lotus_build.py) and their views as reference datasets (lotus.py),
on a small LOTUS made up here. Run with:
OLFDB_DATA_DIR=<a scratch folder> python -m pytest tests/"""
from __future__ import annotations

import gzip
import io
import json
import sqlite3
import tarfile
import zipfile

import numpy as np
import pytest
from rdkit import Chem
from starlette.exceptions import HTTPException

from app import bundle_import, config, lotus, lotus_build, lotus_import, odormeta, reference, sandbox
from tests.test_metacyc import finished, post_form
from tests.test_reference import env  # noqa: F401 (fixtures)
from tests.test_tensor import call, main, root  # noqa: F401 (fixtures)

INCHI_LIN = "InChI=1S/C10H18O/c1-5-10(4,11)8-6-7-9(2)3/h5,7,11H,1,6,8H2,2-4H3"
INCHI_RLIN = INCHI_LIN + "/t10-/m0/s1"
INCHI_SLIN = INCHI_LIN + "/t10-/m1/s1"
INCHI_GER = "InChI=1S/C10H18O/c1-9(2)5-4-6-10(3)7-8-11/h5,7,11H,4,6,8H2,1-3H3/b10-7+"
INCHI_NER = "InChI=1S/C10H18O/c1-9(2)5-4-6-10(3)7-8-11/h5,7,11H,4,6,8H2,1-3H3/b10-7-"
INCHI_ACETIC = "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)"
INCHI_ETAC = "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3"
KEY = {i: Chem.InchiToInchiKey(i) for i in (INCHI_LIN, INCHI_RLIN, INCHI_SLIN, INCHI_GER, INCHI_NER, INCHI_ACETIC)}

STRUCTURES = [  # inchi, cid, traditional name
    (INCHI_LIN, "6549", "linalool"), (INCHI_RLIN, "443158", "(R)-linalool"), (INCHI_SLIN, "67179", "(S)-linalool"),
    (INCHI_GER, "637566", "geraniol"), (INCHI_NER, "643820", "nerol"), (INCHI_ACETIC, "176", "acetic acid")]
PLANT = {"GBIF Backbone Taxonomy": ("Plantae", "Tracheophyta", "Magnoliopsida"), "NCBI": ("Viridiplantae", "Streptophyta", "Magnoliopsida")}
ORGANISMS = {  # name: [(source, rank, kingdom, phylum, class, order, family, genus, species)]
    "Lavandula angustifolia": [("GBIF Backbone Taxonomy", "species", *PLANT["GBIF Backbone Taxonomy"], "Lamiales", "Lamiaceae", "Lavandula", "Lavandula angustifolia"),
                               ("NCBI", "species", *PLANT["NCBI"], "Lamiales", "Lamiaceae", "Lavandula", "Lavandula angustifolia")],
    "Ocimum basilicum": [("GBIF Backbone Taxonomy", "species", *PLANT["GBIF Backbone Taxonomy"], "Lamiales", "Lamiaceae", "Ocimum", "Ocimum basilicum")],
    "Coriandrum sativum": [("NCBI", "species", *PLANT["NCBI"], "Apiales", "Apiaceae", "Coriandrum", "Coriandrum sativum")],
    "Rosa damascena": [("GBIF Backbone Taxonomy", "species", *PLANT["GBIF Backbone Taxonomy"], "Rosales", "Rosaceae", "Rosa", "Rosa damascena")],
    "Aster amellus": [("EUNIS", "species", "Plantae", "Tracheophyta", "Magnoliopsida", "Asterales", "Compositae", "Aster", "Aster amellus")],
    "Lavandula": [("GBIF Backbone Taxonomy", "genus", *PLANT["GBIF Backbone Taxonomy"], "Lamiales", "Lamiaceae", "Lavandula", "")],
    "Crambe maritima": [("GBIF Backbone Taxonomy", "species", *PLANT["GBIF Backbone Taxonomy"], "Brassicales", "Brassicaceae", "Crambe", "Crambe maritima")],
    "Crambe crambellus": [("GBIF Backbone Taxonomy", "species", "Animalia", "Arthropoda", "Insecta", "Lepidoptera", "Crambidae", "Crambe", "Crambe crambellus")],
    "Unplaced thing": [],
}
RECORDS = [  # structure, organism, doi
    (INCHI_LIN, "Lavandula angustifolia", "10.1/a"), (INCHI_LIN, "Ocimum basilicum", "10.1/b"),
    (INCHI_RLIN, "Coriandrum sativum", "10.1/c"), (INCHI_RLIN, "Lavandula angustifolia", "10.1/d"),
    (INCHI_SLIN, "Ocimum basilicum", "10.1/b"), (INCHI_LIN, "Lavandula", "10.1/i"),
    (INCHI_GER, "Rosa damascena", "10.1/e"), (INCHI_NER, "Rosa damascena", "10.1/f"), (INCHI_NER, "Crambe maritima", "10.1/g"),
    (INCHI_ACETIC, "Crambe crambellus", "10.1/h"), (INCHI_ACETIC, "Unplaced thing", "10.1/h"), (INCHI_ACETIC, "Aster amellus", "10.1/h"),
    (INCHI_LIN, "Lavandula angustifolia", "10.1/a"),  # (twice)
]
ODORS = [
    {"id": 1, "name": "linalool", "cid": 6549, "inchi": INCHI_LIN, "lotus": None},
    {"id": 2, "name": "geraniol", "cid": 637566, "inchi": INCHI_GER, "lotus": None},
    {"id": 3, "name": "acetic acid", "cid": 176, "inchi": None, "lotus": None},  # (its InChI not looked up yet)
    {"id": 4, "name": "(R)-linalool", "cid": None, "inchi": None, "lotus": "LTS0000002"},
    {"id": 5, "name": "ethyl acetate", "cid": 8857, "inchi": INCHI_ETAC, "lotus": None},
]
ORG_COLS = ["organismCleaned", "organismCleaned_id", "organismCleaned_rank", "organismCleaned_dbTaxo",
            "organismCleaned_dbTaxoTaxonRanks", "organismCleaned_dbTaxoTaxonomy", *lotus_build.RANK_COLUMNS.values()]


def tsv(path, header, rows, sep="\t"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sep.join(r) for r in [header, *rows]) + "\n")


def make_raw(d, double_gzip=False):
    """A LOTUS made up, laid out as in Scraped-Databases-Copy/LOTUS/raw."""
    header = ["structure_inchikey", "organism_name", "reference_doi", "manual_validation", "organism_wikidata",
              "structure_wikidata", "reference_wikidata"]
    lines = [[KEY[s], o, doi, "Y" if doi == "10.1/c" else "NA", f"http://www.wikidata.org/entity/Q{100 + i}",
              f"http://www.wikidata.org/entity/Q{200 + i}", f"http://www.wikidata.org/entity/Q{300 + i}"]
             for i, (s, o, doi) in enumerate(RECORDS)]
    csv = "\n".join(",".join(f'"{x}"' if " " in x else x for x in r) for r in [header, *lines]) + "\n"
    if double_gzip:
        p = d / "230106_frozen/230106_frozen.csv.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(gzip.compress(gzip.compress(csv.encode())))
    else:
        (d / "230106_frozen").mkdir(parents=True)
        (d / "230106_frozen/230106_frozen.csv").write_text(csv)
    m = d / "metadata_2022-09-16/7534077"
    scols = ["structureCleanedSmiles", "structureCleanedInchi", "structureCleanedInchikey", "structureCleaned_molecularFormula",
             "structureCleaned_exactMass", "structureCleaned_xlogp", "structureCleaned_stereocenters_unspecified",
             "structureCleaned_stereocenters_total", "structureCleaned_cid", "structureCleaned_nameIupac", "structureCleaned_nameTraditional"]
    srows = [["", i, KEY[i], "C10H18O", "154.1358", "2.7", "0", "1", cid, f"iupac {n}", n] for i, cid, n in STRUCTURES]
    srows.append(["", "InChI=1S/X", "NOTINRECORDS-UHFFFAOYSA-N", "", "", "", "", "", "1", "", "unused"])
    tsv(m / "structure_metadata.tsv", scols, srows)
    orows = []
    for name, rows in ORGANISMS.items():
        for src, rank, *ranks in rows:
            orows.append([name, f"id-{name[:3]}-{src[:4]}", rank, src, "kingdom|phylum|class|order|family|genus|species",
                          "|".join(ranks), *ranks, ""])
    orows.append(["Lavandula angustifolia", "dup", "species", "GBIF Backbone Taxonomy", "", "", "Fungi", "", "", "", "Wrongaceae", "", "", ""])
    tsv(m / "organism_metadata.tsv", ORG_COLS, orows)
    rcols = ["organismValue", "referenceCleanedDoi", "referenceCleanedPmcid", "referenceCleanedPmid", "referenceCleanedTitle",
             "referenceCleaned_journal", "referenceCleaned_date", "referenceCleaned_author"]
    rrows = [["x", f"10.1/{c}", "", "", f"Paper {c.upper()}", "J. Odors", f"20{10 + k:02d}-01", "Smith"] for k, c in enumerate("abcdefghi")]
    rrows[5][4] = "Nerol from<i>Rosa</i>petals &amp; CO<sub>2</sub>"  # (f: markup, as Crossref gives some)
    rrows.insert(1, ["y", "10.1/a", "", "", "", "", "", ""])  # (a paper on several lines, its fields on the first)
    tsv(m / "reference_metadata.tsv", rcols, rrows)
    sdf = d / "LOTUS_DB_LATEST/LOTUS_2021_03_simple.sdf"
    sdf.parent.mkdir(parents=True)
    sdf.write_text("".join(f"\n  CDK\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n> <lotus_id>\n{lts}\n\n> <inchikey>\n{KEY[i]}\n\n$$$$\n"
                           for lts, i in (("LTS0000001", INCHI_LIN), ("LTS0000002", INCHI_RLIN))))
    return d


@pytest.fixture
def bundle(tmp_path):
    raw = make_raw(tmp_path / "raw")
    out = tmp_path / "lotus.sqlite"
    about = lotus_build.build(raw, out, log=lambda *_: None)
    return out, about


def view(path, **fields):
    record, values = lotus.parse(path, fields, ODORS)
    rows = {r["label"]: i for i, r in enumerate(record["rows"])}
    cols = {c["label"]: j for j, c in enumerate(record["cols"])}
    return record, values, rows, cols


def test_build(bundle):
    out, about = bundle
    assert about["kind"] == "lotus" and about["release"] == {"pairs": "2023-01-06", "metadata": "2022-09-16", "lts": "2021-03"}
    c = about["counts"]
    assert (c["structures"], c["organisms"], c["papers"], c["records"], c["with_cid"], c["with_lts"]) == (6, 9, 9, 12, 6, 2)
    assert c["validated"] == 1 and c["with_lineage"] == 8
    assert about["taxonomy_sources"] == {"GBIF Backbone Taxonomy": 6, "NCBI": 2, "EUNIS": 1}
    con = sqlite3.connect(out)
    lav = con.execute("SELECT t.family, t.kingdom FROM taxonomy t JOIN organisms o ON o.id = t.organism "
                      "WHERE o.name = 'Lavandula angustifolia' AND t.source = 'NCBI'").fetchone()
    assert lav == ("Lamiaceae", "Plantae")  # (Viridiplantae, as GBIF names it)
    gbif = con.execute("SELECT t.family FROM taxonomy t JOIN organisms o ON o.id = t.organism "
                       "WHERE o.name = 'Lavandula angustifolia' AND t.source = 'GBIF Backbone Taxonomy'").fetchone()
    assert gbif == ("Lamiaceae",)  # the first line of it, not the later one
    assert con.execute("SELECT title, date FROM refs WHERE doi = '10.1/a'").fetchone() == ("Paper A", "2010-01")
    assert con.execute("SELECT title FROM refs WHERE doi = '10.1/f'").fetchone() == ("Nerol from Rosa petals & CO₂",)
    assert con.execute("SELECT family FROM taxonomy WHERE source = 'EUNIS'").fetchone() == ("Asteraceae",)
    r = con.execute("SELECT lts, cid, name, wikidata, key_no_rs, skeleton FROM structures WHERE inchikey = ?", (KEY[INCHI_RLIN],)).fetchone()
    assert r[:3] == ("LTS0000002", 443158, "(R)-linalool") and r[3].startswith("Q2")
    assert r[4] == Chem.InchiToInchiKey(INCHI_LIN) and r[5] == KEY[INCHI_LIN][:14]
    assert con.execute("SELECT COUNT(*) FROM structures WHERE inchikey = 'NOTINRECORDS-UHFFFAOYSA-N'").fetchone() == (0,)


def test_build_double_gzip(tmp_path):
    raw = make_raw(tmp_path / "raw", double_gzip=True)
    about = lotus_build.build(raw, tmp_path / "b.sqlite", log=lambda *_: None)
    assert about["counts"]["records"] == 12 and about["release"]["pairs"] == "2023-01-06"


def test_build_errors(tmp_path):
    (tmp_path / "raw").mkdir()
    assert lotus_build.resolve(tmp_path / "raw") == dict.fromkeys(lotus_build.FILES)
    with pytest.raises(lotus_build.BuildError, match="missing: LOTUS's frozen export of its Wikidata records") as e:
        lotus_build.build(tmp_path / "raw", tmp_path / "x.sqlite", log=lambda *_: None)
    assert str(e.value).count("missing:") == len(lotus_build.REQUIRED)  # (LOTUS IDs, not required: not said)
    assert lotus_build.main([str(tmp_path / "raw"), str(tmp_path / "x.sqlite")]) == 2
    raw = make_raw(tmp_path / "raw2")
    files = lotus_build.resolve(raw)
    assert lotus_build.problems(files, raw) == []
    # a table without the columns needed, one not tab-separated, an SDF without LOTUS IDs, a gzip cut short
    m = raw / "metadata_2022-09-16/7534077"
    tsv(m / "organism_metadata.tsv", ["organismCleaned", "x"], [["a", "b"]])
    (m / "reference_metadata.tsv").write_text("a,b\n1,2\n")
    (raw / "LOTUS_DB_LATEST/LOTUS_2021_03_simple.sdf").write_text("\n  CDK\n\nM  END\n> <name>\nx\n\n$$$$\n")
    (m / "structure_metadata.tsv").rename(m / "structure_metadata.tsv.gz")
    (m / "structure_metadata.tsv.gz").write_bytes(gzip.compress(b"structureCleanedInchikey\tx\n" * 1000)[:40])
    found = lotus_build.problems(lotus_build.resolve(raw), raw)
    assert len(found) == 4, found
    assert found[0].startswith("metadata_2022-09-16/7534077/structure_metadata.tsv.gz cannot be read (")
    assert found[1].startswith("metadata_2022-09-16/7534077/organism_metadata.tsv is not LOTUS's metadata of each organism")
    assert "lacks the columns organismCleaned_id, " in found[1] and found[1].endswith("(its columns: organismCleaned, x)")
    assert found[2].endswith("(its columns: none: it is not a table)")
    assert found[3].endswith("its first records have no lotus_id or inchikey field")
    with pytest.raises(lotus_build.BuildError, match="not as expected"):
        lotus_build.build(raw, tmp_path / "x.sqlite", log=lambda *_: None)
    assert not (tmp_path / "x.sqlite").exists()


def test_resolve_newest(tmp_path):
    """Of the files a glob finds, gzipped or not, the newest; of one there both ways, the one
    as is (the import, the one gzipped)."""
    raw = make_raw(tmp_path / "raw")
    old = raw / "metadata_2021-01-01/1/structure_metadata.tsv"
    old.parent.mkdir(parents=True)
    old.write_text("x")
    new = raw / "metadata_2022-09-16/7534077/structure_metadata.tsv"
    assert lotus_build.resolve(raw)["structures"] == new
    new.with_name(new.name + ".gz").write_bytes(gzip.compress(new.read_bytes()))
    assert lotus_build.resolve(raw)["structures"] == new
    new.unlink()
    assert lotus_build.resolve(raw)["structures"] == new.with_name(new.name + ".gz")


def test_keys():
    assert lotus.no_rs_inchi(INCHI_RLIN) == INCHI_LIN
    assert lotus.no_rs_inchi(INCHI_GER) == INCHI_GER  # (cis/trans kept)
    assert lotus.no_rs_inchi("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3/i1D/t2-/m0/s1") == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3/i1D"
    assert lotus.kingdom_of("NCBI", "Metazoa") == "Animalia"
    assert lotus.kingdom_of("World Flora Online", "wfo-0001379367") == "Plantae"
    assert lotus.kingdom_of("NCBI", None, "|Bacteria|Pseudomonadota|") == "Bacteria"
    assert lotus.kingdom_of("Open Tree of Life", None, "") is None


def test_options_and_describe():
    o = lotus.options(None)
    assert o == {"level": "family", "value": "organisms", "stereo": "enantiomers", "taxonomy": "GBIF Backbone Taxonomy"}
    assert lotus.options({"level": "organism"})["value"] == "papers"
    assert lotus.options({"level": "organism", "value": "organisms"})["value"] == "presence"
    assert lotus.options({"level": "nope", "stereo": "nope", "value": "nope"}) == o
    d = lotus.describe(o, {"release": {"pairs": "2023-01-06", "metadata": "2022-09-16", "lts": "2021-03"}})
    what = {x["what"]: x for x in d["decisions"]}
    assert "2023-01-06" in what["Records"]["chosen"] and "2021-03" in what["Records"]["chosen"]
    assert what["Records"]["admin"] is not True and what["Stereochemistry"]["admin"] is True
    assert "geraniol and nerol" in what["Stereochemistry"]["chosen"] and "kept apart" in what["Stereochemistry"]["chosen"]
    assert what["Taxonomy"]["chosen"].startswith("GBIF Backbone Taxonomy's lineage")
    assert "families" in d["summary"]
    assert lotus.priority("NCBI")[:3] == ["NCBI", "GBIF Backbone Taxonomy", "Catalogue of Life"]
    assert lotus.view_title(o, {"pairs": "2023-01-06"}) == "LOTUS 2023-01-06: chemicals by the families of organisms they are reported in"


def test_stereo(bundle):
    out, _ = bundle
    b = lotus.bundle(out)
    n = lambda stereo: {m["odor"]["name"]: (len(m["structures"]), m["how"]) for m in lotus.match_odors(b, ODORS, stereo)}  # noqa: E731
    assert n("exact") == {"linalool": (1, "inchi"), "geraniol": (1, "inchi"), "acetic acid": (1, "cid"),
                          "(R)-linalool": (1, "lotus"), "ethyl acetate": (0, None)}
    assert n("enantiomers") == {"linalool": (3, "inchi"), "geraniol": (1, "inchi"), "acetic acid": (1, "cid"),
                                "(R)-linalool": (3, "lotus"), "ethyl acetate": (0, None)}
    assert n("skeleton")["geraniol"] == (2, "inchi")  # with nerol


def test_families(bundle):
    out, _ = bundle
    record, values, rows, cols = view(out)
    assert list(rows) == ["linalool", "geraniol", "acetic acid", "(R)-linalool"]
    # by lineage (kingdoms with the most organisms first, then phylum, ...: Apiaceae by NCBI's, the rest by GBIF's);
    # only those any row is in
    assert list(cols) == ["Apiaceae", "Asteraceae", "Lamiaceae", "Rosaceae", "Crambidae"]
    lin, ger, ace = rows["linalool"], rows["geraniol"], rows["acetic acid"]
    # linalool: as any stereoisomer, in Lavandula angustifolia, Ocimum basilicum and Lavandula (a genus) of the Lamiaceae
    assert values[lin, cols["Lamiaceae"]] == 3 and values[lin, cols["Apiaceae"]] == 1  # (Coriandrum, by NCBI's lineage)
    assert values[ger, cols["Rosaceae"]] == 1 and "Brassicaceae" not in cols  # (nerol's, not geraniol's)
    assert values[ace, cols["Crambidae"]] == 1
    assert np.array_equal(values[rows["(R)-linalool"]], values[lin])
    assert record["kind"] == "sequential" and record["chemical_axis"] == "rows"
    assert record["ids"]["rows"][lin] == {"name": "linalool", "cid": 6549, "inchi": INCHI_LIN}
    assert record["rows"][lin]["detail"].startswith("LOTUS: linalool (LTS0000001), and 2 other structures of it, matched by InChI")
    assert sorted(record["rows"][lin]["lotus"]) == sorted(record["rows"][rows["(R)-linalool"]]["lotus"])
    assert record["cols"][cols["Lamiaceae"]]["detail"] == "Lamiales › Magnoliopsida › Tracheophyta › Plantae · 3 organisms of LOTUS"
    assert record["cols"][cols["Crambidae"]]["lotus"] == {"level": "family", "name": "Crambidae", "kingdom": "Animalia"}
    notes = " | ".join(record["notes"])
    assert "1 of the odor table's 5 chemicals are not in LOTUS" in notes and "ethyl acetate" in notes
    assert "1 organism of these chemicals left out" in notes  # (Unplaced thing)
    lo = record["lotus"]
    assert lo["stereo"] == "enantiomers" and lo["release"]["pairs"] == "2023-01-06" and len(lo["decisions"]) == 6
    assert lo["value_label"] == "organisms of the family it is reported in"


def test_values_and_levels(bundle):
    out, _ = bundle
    _, values, rows, cols = view(out, value="papers")
    assert values[rows["linalool"], cols["Lamiaceae"]] == 4  # a, b, d, i
    record, values, rows, cols = view(out, value="presence")
    assert record["kind"] == "binary" and set(np.unique(values)) <= {0, 1}
    assert lotus.scale(record["lotus"]) is None and lotus.scale(lotus.options({"value": "papers"})) == "log"
    record, values, rows, cols = view(out, level="genus", stereo="skeleton")
    assert {"Crambe (Plantae)", "Crambe (Animalia)"} <= set(cols)  # (homonyms told apart)
    assert values[rows["geraniol"], cols["Crambe (Plantae)"]] == 1 and values[rows["acetic acid"], cols["Crambe (Animalia)"]] == 1
    record, values, rows, cols = view(out, level="species")
    assert "Lavandula" not in cols and values[rows["linalool"], cols["Lavandula angustifolia"]] == 1
    assert any("left out" in n for n in record["notes"])
    record, values, rows, cols = view(out, level="organism")
    assert "Unplaced thing" in cols and "Lavandula" in cols  # (each organism, whatever its lineage)
    assert values[rows["linalool"], cols["Lavandula angustifolia"]] == 2  # papers a and d
    record, values, rows, cols = view(out, level="kingdom", stereo="exact")
    assert list(cols) == ["Plantae", "Animalia"] and values[rows["linalool"], cols["Plantae"]] == 3


def test_taxonomy_priority(bundle):
    out, _ = bundle
    gbif = view(out)[0]
    ncbi = view(out, taxonomy="NCBI")[0]
    lam = lambda r: next(c for c in r["cols"] if c["label"] == "Lamiaceae")  # noqa: E731
    assert "Tracheophyta" in lam(gbif)["detail"]
    # NCBI's first: the Lamiaceae as NCBI has them (it has Lavandula angustifolia's), its kingdom as GBIF names it
    assert "Streptophyta" in lam(ncbi)["detail"] and lam(ncbi)["lotus"]["kingdom"] == "Plantae"
    assert next(x for x in ncbi["lotus"]["decisions"] if x["what"] == "Taxonomy")["chosen"].startswith("NCBI's lineage")


def test_cell(bundle):
    out, _ = bundle
    record, _, rows, cols = view(out)
    b = lotus.bundle(out)
    c = lotus.cell(b, record["lotus"], record["rows"][rows["linalool"]]["lotus"], record["cols"][cols["Lamiaceae"]]["lotus"])
    assert c["taxon"]["name"] == "Lamiaceae" and c["n_organisms"] == 3 and c["n_papers"] == 4
    orgs = {o["name"]: o for o in c["organisms"]}
    assert list(orgs)[0] == "Lavandula angustifolia"  # (the most papers first)
    lav = orgs["Lavandula angustifolia"]
    assert [p["doi"] for p in lav["papers"]] == ["10.1/d", "10.1/a"]  # (newest first)
    assert lav["papers"][1]["title"] == "Paper A" and lav["papers"][1]["url"] == "https://doi.org/10.1/a"
    assert lav["source"] == "GBIF Backbone Taxonomy" and lav["lineage"].startswith("Plantae › Tracheophyta")
    oci = orgs["Ocimum basilicum"]["papers"]
    assert len(oci) == 1 and len(oci[0]["structures"]) == 2  # (linalool and (S)-linalool, in one paper)
    assert {s["name"] for s in c["structures"]} == {"linalool", "(R)-linalool", "(S)-linalool"}
    lts = next(s for s in c["structures"] if s["lts"] == "LTS0000001")
    assert lts["links"]["lotus"].endswith("/LTS0000001")
    cor = lotus.cell(b, record["lotus"], record["rows"][rows["linalool"]]["lotus"], record["cols"][cols["Apiaceae"]]["lotus"])
    assert cor["organisms"][0]["papers"][0]["validated"] is True
    with pytest.raises(lotus.LotusError):
        lotus.cell(b, record["lotus"], [1], {"name": "Nonaceae", "kingdom": "Plantae"})


def test_fingerprint():
    a = lotus.odors_fingerprint(ODORS)
    assert a == lotus.odors_fingerprint(list(reversed(ODORS)))
    assert a != lotus.odors_fingerprint([{**ODORS[0], "lotus": "LTS0000009"}, *ODORS[1:]])


def test_not_a_bundle(tmp_path):
    p = tmp_path / "x.sqlite"
    sqlite3.connect(p).executescript("CREATE TABLE about (key TEXT, value TEXT); INSERT INTO about VALUES ('kind', '\"metacyc\"');")
    assert not lotus.is_bundle(p)
    with pytest.raises(lotus.LotusError):
        lotus.read_about(p)


def test_bundle_kind(bundle, tmp_path):
    out, _ = bundle
    assert reference.bundle_kind(out) == "lotus"
    other = tmp_path / "other.sqlite"
    sqlite3.connect(other).executescript("CREATE TABLE about (key TEXT, value TEXT); INSERT INTO about VALUES ('kind', '\"metacyc\"');")
    assert reference.bundle_kind(other) == "metacyc"
    assert reference.bundle_kind(tmp_path / "none.sqlite") is None and reference.bundle_kind(tmp_path / "x.csv") is None
    (tmp_path / "junk.db").write_bytes(b"not sqlite")
    assert reference.bundle_kind(tmp_path / "junk.db") is None
    with pytest.raises(reference.ReferenceError, match="not a bundle it can read"):
        reference.parse(tmp_path / "junk.db")


def test_reference_dataset(env, tmp_path):  # noqa: F811
    """A bundle added in the Admin tab, looked into, its settings changed there, and made
    again when the odor table changes."""
    out = tmp_path / "bundles" / "lotus.sqlite"
    lotus_build.build(make_raw(tmp_path / "raw"), out, log=lambda *_: None)
    sandbox.activate({"read_dirs": [str(out.parent)], "write_dirs": []})
    c = call(env.admin_reference_check, body={"path": str(out)})[1]
    assert c["ok"] and c["fields"]["kind"] == "lotus" and c["fields"]["about"]["release"]["pairs"] == "2023-01-06"
    assert c["fields"]["used"] == {"layout": "lotus", **lotus.DEFAULT_OPTIONS}
    ch = c["fields"]["choices"]
    assert [x["value"] for x in ch["taxonomy"]] == ["GBIF Backbone Taxonomy", "NCBI", "EUNIS"]
    assert ch["taxonomy"][0]["label"].endswith("(6 organisms)")
    assert [x["value"] for x in ch["stereo"] if x["default"]] == ["enantiomers"] and len(ch["level"]) == 8
    lt = c["preview"]["lotus"]
    assert lt["title"] == "LOTUS 2023-01-06: chemicals by the families of organisms they are reported in"
    assert c["preview"]["chemicals"]["matched"] == 1  # (of the env's odor table: linalool)
    status, r = call(env.admin_reference_add, body={"path": str(out), "title": lt["title"], "fields": {"layout": "lotus"},
                                                     "value_label": lt["value_label"], "col_label": "family"})
    assert status == 200 and r["state"] == "ok", r
    pp = {"ref_id": r["id"]}
    r = call(env.reference_get, path_params=pp)[1]
    assert r["meta"]["lotus"]["stereo"] == "enantiomers" and [d["what"] for d in r["meta"]["lotus"]["decisions"]][2] == "Stereochemistry"
    assert r["meta"]["scale"] == "log"  # (counts of organisms: shown on a log color scale)
    assert [x["label"] for x in r["rows"]] == ["linalool"] and r["rows"][0]["chem"]["name"] == "linalool"
    structures = ",".join(map(str, r["rows"][0]["lotus"]))
    assert len(r["rows"][0]["lotus"]) == 3 and [x["label"] for x in r["cols"]] == ["Apiaceae", "Lamiaceae"]
    # a cell looked into
    status, cell = call(env.lotus_cell, path_params=pp, params={"structures": structures, "name": "Lamiaceae", "kingdom": "Plantae"})
    assert status == 200 and cell["n_organisms"] == 3 and cell["organisms"][0]["papers"][0]["title"] == "Paper D"
    assert call(env.lotus_cell, path_params=pp, params={"structures": "x"})[0] == 400
    assert call(env.lotus_cell, path_params=pp, params={})[0] == 400
    for ref_id, params in ((r["meta"]["id"], {"structures": structures, "name": "Nonaceae"}), ("fruits", {"structures": "1"})):
        with pytest.raises(HTTPException) as e:
            call(env.lotus_cell, path_params={"ref_id": ref_id}, params=params)
        assert e.value.status_code == 404
    # its settings changed in the Admin tab: made again by them, the rest kept
    status, res = call(env.admin_reference_update, path_params=pp, body={"fields": {"layout": "lotus", "stereo": "exact", "level": "genus"}})
    assert status == 200 and res["state"] == "ok", res
    r = call(env.reference_get, path_params=pp)[1]
    assert (r["meta"]["lotus"]["stereo"], r["meta"]["lotus"]["level"], r["meta"]["lotus"]["value"]) == ("exact", "genus", "organisms")
    assert len(r["rows"][0]["lotus"]) == 1 and r["meta"]["title"] == lt["title"] and r["meta"]["col_label"] == "family"
    status, res = call(env.admin_reference_update, path_params=pp, body={"col_label": "genus", "fields": {"layout": "lotus", "taxonomy": "NCBI"}})
    assert status == 200
    listed = {d["id"]: d for d in call(env.reference_list)[1]["datasets"]}[pp["ref_id"]]
    assert listed["lotus"]["taxonomy"] == "NCBI" and listed["lotus"]["level"] == "family" and listed["col_label"] == "genus"
    assert listed["value_label"] == lt["value_label"] and listed["kind_setting"] == "detect"
    assert call(env.admin_reference_update, path_params=pp, body={"title": " "})[0] == 422
    assert call(env.admin_reference_update, path_params={"ref_id": "nope"}, body={})[0] == 404
    assert call(env.admin_reference_update, path_params={"ref_id": "fruits"}, body={})[0] == 409
    # the odor table changed: made again (by CID, and by the LOTUS ID it gives)
    rows, _, _ = odormeta.parse_sheet("name,abbrev,cid,lotus\ngeraniol,ger,637566,\nR-linalool,rlin,,LTS0000002\n")
    env.db.apply_odor_sync(added=rows, changed=[], deleted=[], source="test", fingerprint="1", details={}, when="2026-11-03T00:00:00")
    r = call(env.reference_get, path_params=pp)[1]
    assert [x["label"] for x in r["rows"]] == ["geraniol", "linalool", "R-linalool"]
    assert "matched by the LOTUS ID in the odor table" in r["rows"][2]["detail"]


def test_register(monkeypatch, tmp_path):
    """The script adds the default view through the Admin tab's API."""
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
        sent.append((req.full_url, json.loads(req.data)))
        return Resp({"ok": True, "id": "lotus", "state": "ok", "error": None, "restart": False, "steps": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    logged = []
    about = {"release": {"pairs": "2023-01-06"}}
    assert lotus_build.register("http://olfdb:8000/", tmp_path / "l.sqlite", about, log=logged.append) == 0
    (url, body), = sent
    assert url == "http://olfdb:8000/api/admin/references" and body["fields"] == {"layout": "lotus", **lotus.DEFAULT_OPTIONS}
    assert body["title"] == "LOTUS 2023-01-06: chemicals by the families of organisms they are reported in"
    assert body["value_label"] == "organisms of the family it is reported in" and body["col_label"] == "family" and len(logged) == 1


# ------------------------------------------------------------- imported in the Admin tab


@pytest.fixture
def imports(env, tmp_path, monkeypatch):  # noqa: F811
    """main, with LOTUS's imports (and things deleted) in tmp_path, and none under way."""
    monkeypatch.setattr(config, "LOTUS_DIR", tmp_path / "data" / "lotus")
    monkeypatch.setattr(config, "DELETED_DIR", tmp_path / "data" / "deleted")
    monkeypatch.setattr(lotus_import.importer, "_job", bundle_import._idle())
    return env


def uploaded(raw, gzipped=lambda p: True) -> list[tuple[str, bytes]]:
    """A folder's files as the Admin tab uploads them (named by their paths in it), those not
    gzipped gzipped first (that `gzipped` says to, as a browser that can)."""
    out = []
    for p in sorted(raw.rglob("*")):
        if p.is_file():
            rel = f"{raw.name}/{p.relative_to(raw).as_posix()}"
            gz = p.name != "notes.txt" and not p.name.endswith(".gz") and gzipped(p)
            out.append((rel + ".gz", gzip.compress(p.read_bytes())) if gz else (rel, p.read_bytes()))
    return out


def test_pick():
    names = ["raw/230106_frozen/230106_frozen.csv", "raw/230106_frozen/230106_frozen.csv.gz",
             "raw/metadata_2021-01-01/1/structure_metadata.tsv", "raw/metadata_2022-09-16/7534077/structure_metadata.tsv.gz",
             "raw/metadata_2022-09-16/7534077/organism_metadata.tsv", "raw/metadata_2022-09-16/7534077.zip",
             "raw/metadata_2022-09-16/notes.txt", "elsewhere/reference_metadata.tsv.gz", "raw/LOTUS_DB_LATEST/LOTUS_2021_03_simple.sdf",
             "raw/LOTUS_DB_LATEST/LOTUS_2021_03.tsv", "raw/NPOC2021/NPOC2021/lotusUniqueNaturalProduct.bson", "raw/notes.txt",
             "raw\\LOTUS_DB_LATEST\\LOTUS_2021_03_simple.sdf"]
    picked = {names[i]: p for i, p in lotus_import.importer.pick(names).items()}
    assert picked == {
        "raw/230106_frozen/230106_frozen.csv.gz": "230106_frozen/230106_frozen.csv.gz",  # (gzipped, of the two)
        "raw/notes.txt": "notes.txt",  # (in the folder above it)
        "raw/metadata_2022-09-16/7534077/structure_metadata.tsv.gz": "metadata_2022-09-16/7534077/structure_metadata.tsv.gz",
        "raw/metadata_2022-09-16/notes.txt": "metadata_2022-09-16/notes.txt",
        "raw/metadata_2022-09-16/7534077/organism_metadata.tsv": "metadata_2022-09-16/7534077/organism_metadata.tsv",
        "elsewhere/reference_metadata.tsv.gz": "reference_metadata.tsv.gz",  # (found by its name alone)
        "raw/LOTUS_DB_LATEST/LOTUS_2021_03_simple.sdf": "LOTUS_DB_LATEST/LOTUS_2021_03_simple.sdf",  # (the first of two alike)
    }
    assert lotus_import.place("metadata_1/../structure_metadata.tsv", "metadata_*/*/structure_metadata.tsv") is None
    assert lotus_import.place("x/metadata_1/a/structure_metadata.tsv", "metadata_*/*/structure_metadata.tsv") == \
        "metadata_1/a/structure_metadata.tsv"


def test_import(imports, tmp_path):
    """The folder uploaded as the Admin tab uploads it (only the files the build reads, gzipped
    first; one not, as a browser that cannot sends it: gzipped by the server) is kept, built
    into a bundle, and its reference dataset added; the folder is moved to deleted/ with it."""
    main = imports
    raw = make_raw(tmp_path / "raw")
    (raw / "metadata_2022-09-16/notes.txt").write_text("Downloaded from Zenodo.")
    files = uploaded(raw, gzipped=lambda p: p.name != "structure_metadata.tsv") + [("raw/unread.txt", b"x")]
    status, r = post_form(main.admin_lotus_import, files, version="")
    assert status == 200 and r["ok"] and r["job"]["running"] and r["job"]["label"] == "2023-01-06", r
    m = "metadata_2022-09-16/7534077"
    assert r["files"] == {"pairs": "230106_frozen/230106_frozen.csv.gz", "structures": f"{m}/structure_metadata.tsv.gz",
                          "organisms": f"{m}/organism_metadata.tsv.gz", "references": f"{m}/reference_metadata.tsv.gz",
                          "lts": "LOTUS_DB_LATEST/LOTUS_2021_03_simple.sdf.gz"}
    job = finished(lotus_import.importer)
    assert job["error"] is None and [a["title"] for a in job["added"]] == [
        "LOTUS 2023-01-06: chemicals by the families of organisms they are reported in"], job
    assert job["added"][0]["state"] == "ok" and any(line.startswith("records: 12 ") for line in job["log"])
    d = config.LOTUS_DIR / job["id"]
    kept = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert kept == ["import.json", "lotus-2023-01-06.sqlite", *sorted(f"raw/{v}" for v in r["files"].values()),
                    "raw/metadata_2022-09-16/notes.txt"]
    assert gzip.decompress((d / "raw" / r["files"]["structures"]).read_bytes()) == (raw / m / "structure_metadata.tsv").read_bytes()
    about = lotus.read_about(d / "lotus-2023-01-06.sqlite")
    assert about["release"] == {"pairs": "2023-01-06", "metadata": "2022-09-16", "lts": "2021-03"}
    assert about["counts"]["with_lts"] == 2 and list(about["notes"].values()) == ["Downloaded from Zenodo."]
    ref_id = job["added"][0]["id"]
    r = call(main.reference_get, path_params={"ref_id": ref_id})[1]
    assert r["meta"]["lotus"]["level"] == "family" and [x["label"] for x in r["rows"]] == ["linalool"]
    s = call(main.admin_lotus)[1]
    assert s["how"] == "pattern" and s["gzip"] and s["script"] == "app/lotus_build.py" and not s["job"]["running"]
    assert [e["role"] for e in s["expected"]] == list(lotus_build.FILES)
    (i,) = s["imports"]
    assert i["refs"] == [ref_id] and i["label"] == "2023-01-06" and i["release"]["metadata"] == "2022-09-16" and i["kind"] == "lotus"
    # its reference dataset deleted: the folder with it
    moved = call(main.admin_reference_delete, path_params={"ref_id": ref_id})[1]["moved"]
    assert moved == str(config.DELETED_DIR / "lotus" / d.name) and not d.exists()
    assert (config.DELETED_DIR / "lotus" / d.name / "lotus-2023-01-06.sqlite").is_file()
    assert call(main.admin_lotus)[1]["imports"] == []


def test_import_archives(imports, tmp_path):
    """The folder as a zip (with more than the build reads, and a file both as is and gzipped)
    or a tar.gz (none gzipped: gzipped as they are taken out): only what it reads kept."""
    main = imports
    raw = make_raw(tmp_path / "LOTUS" / "raw")
    pairs = raw / "230106_frozen/230106_frozen.csv"
    pairs.with_name(pairs.name + ".gz").write_bytes(gzip.compress(pairs.read_bytes()))
    (raw / "NPOC2021").mkdir()
    (raw / "NPOC2021/lotusUniqueNaturalProduct.bson").write_bytes(b"\0" * 100)
    zipped, tarred = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as z:
        for p in sorted((tmp_path / "LOTUS").rglob("*")):
            z.write(p, p.relative_to(tmp_path).as_posix())
    pairs.with_name(pairs.name + ".gz").unlink()
    with tarfile.open(fileobj=tarred, mode="w:gz") as t:
        t.add(raw, "raw")
    for name, data in (("LOTUS.zip", zipped.getvalue()), ("raw.tar.gz", tarred.getvalue())):
        status, r = post_form(main.admin_lotus_import, [(name, data)])
        assert status == 200 and r["ok"] and all(v.endswith(".gz") for v in r["files"].values()), r
        job = finished(lotus_import.importer)
        assert job["error"] is None and len(job["added"]) == 1, job
        d = config.LOTUS_DIR / job["id"]
        kept = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
        assert kept == sorted(["import.json", "lotus-2023-01-06.sqlite", *(f"raw/{v}" for v in r["files"].values())])
    assert len(call(main.admin_lotus)[1]["imports"]) == 2


def test_import_refused(imports, tmp_path, monkeypatch):
    """Uploads that are not LOTUS's files as the build reads them: refused, saying what was
    expected, nothing kept; and a build that fails: nothing kept."""
    main = imports
    status, r = post_form(main.admin_lotus_import, [("a.csv", b"x,y\n1,2\n"), ("b.txt", b"")])
    assert status == 422 and not r["ok"] and r["errors"][0].startswith("none of the 2 files uploaded is one of LOTUS's files")
    assert sum(e.startswith("missing: ") for e in r["errors"]) == len(lotus_build.REQUIRED)
    # a file of the right name, but not the table (as kept: gzipped)
    raw = make_raw(tmp_path / "raw")
    good = uploaded(raw)
    tsv(raw / "metadata_2022-09-16/7534077/organism_metadata.tsv", ["organismCleaned"], [["x"]])
    status, r = post_form(main.admin_lotus_import, uploaded(raw))
    assert status == 422 and len(r["errors"]) == 1, r
    assert r["errors"][0].startswith("metadata_2022-09-16/7534077/organism_metadata.tsv.gz is not LOTUS's metadata of each organism")
    status, r = post_form(main.admin_lotus_import, [("raw.tar.gz", b"not a tar")])
    assert status == 422 and r["errors"] == ["raw.tar.gz is not a zip or tar archive"]
    assert call(main.admin_lotus)[1]["imports"] == [] and not any(config.LOTUS_DIR.iterdir())
    # the build failing: nothing kept
    monkeypatch.setattr(lotus_import.importer, "module", "app.no_such_build")
    status, r = post_form(main.admin_lotus_import, good)
    assert status == 200
    job = finished(lotus_import.importer)
    assert "No module named app.no_such_build" in job["error"] and job["added"] == [] and not any(config.LOTUS_DIR.iterdir())
    assert main.db.list_references() == []
