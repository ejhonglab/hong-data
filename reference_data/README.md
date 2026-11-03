# Reference data

Tables shown in the "Reference data" tab, built in. Others can be added in the
Admin tab, from any file on the server, in the same layout. These are read from this
directory when asked for, so adding or editing a file needs no restart.

`manifest.json` lists them:

```json
{"datasets": [
  {"id": "fruit-volatiles",                      // used in the URL (#reference/<id>)
   "title": "Volatile concentrations across fruits",
   "description": "one or two sentences shown under the title",
   "file": "fruit_volatiles.csv",                // CSV (or parquet) in this directory
   "kind": "sequential",                         // "sequential" (magnitudes), "binary" (0/1 presence) or "diverging" (signed)
   "value_label": "log10 ng/g",                  // shown next to the value range
   "row_label": "chemical", "col_label": "fruit",
   "source": "citation or URL"}
]}
```

CSV layout: the first column holds the row labels (its header is free), every other
column is one matrix column, and cells are numbers (blank = missing). An optional
column named `group` gives each row a group that the viewer uses for row
grouping and separators:

```
chemical,group,banana,strawberry,mango
ethyl acetate,esters,2.1,1.4,
isoamyl acetate,esters,2.8,0.3,1.1
hexanal,aldehydes,,1.9,0.8
```

The manifest may also give `"chemical_axis"`: `"rows"`, `"cols"`, `"none"` (not about
chemicals) or `"detect"` (the default). Chemicals are identified by columns named `cid`
(or `pubchem_cid`, `pubchem cid`), `inchi`, `cas`, `metacyc`, `lotus`, `name` or `abbrev`
(then the rows are chemicals), or rows labelled so (then the columns are); without
these, by their labels, as names or abbreviations in the Odors tab. They are labelled by
their names there, and a matrix with chemicals not identified by a CID or InChI is
shown with a warning, and without them unless they are chosen to be shown too.

```
chemical,cid,group,banana,strawberry
ethyl acetate,8857,esters,2.1,1.4
hexanal,6184,aldehydes,,1.9
```

A matrix of glomeruli by KCs (with no odors), e.g. the synapses each KC gets from the
PNs of each glomerulus, is found by its labels: when at least 3 of the labels on one
axis, and at least half of them, are glomerulus names as accepted for datasets' ROIs
(`DM2`, `VA1d`, ...; but for case). Its other axis is taken to be KCs, whatever it is
labelled by: the viewer can sum over them (*Sum over KCs*), and the Compare tab can
compare it with datasets of glomeruli, on their glomeruli.

```
bodyid,DM2,VA2,DL5
5813055184,3,0,1
5813054622,0,2,4
```

Rows and columns without a single number (notes, units) are left out, and cells that
are not numbers are shown as missing.

A table can instead have one value per line (a "list", or long table), as VCF's does:

```
inchi,name,quantity,product_name,reference,...
1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3,ethyl acetate,0.5,APPLE,"J. Agric. ..."
,linalool,q,BEER,"Flavour Fragr. J. ..."
```

Its manifest entry (or the Admin tab, when adding it) then says which of its fields make
the matrix, the rest being ignored:

```json
"fields": {"layout": "list",
           "rows": {"label": "name", "keep": ["inchi", "cas"]},
           "cols": {"label": "product_name", "keep": ["reference"]},
           "value": "quantity",          // or null: 1 where a line lists the row and column, else 0
           "empty": "missing"}           // or "zero": a cell no line lists
```

Each combination of the rows' fields on a line is a row (and so for columns), labelled
by `label`'s value, those labelled alike told apart by the others (and all shown on
hover). Several lines of one cell give the mean of their numbers; values that are not
numbers (`q`, `trace`, ...) are missing. Identifier fields (`cid`, `inchi`, with or
without its `InChI=`, `name`, ...) of the rows (else of the columns) identify chemicals.
A matrix of more than 20 million cells is refused.

A file can also be an xarray DataArray of two dimensions saved as netCDF (`.nc`, `.nc4`,
by `DataArray.to_netcdf`), read with `xarray.load_dataarray`. Its first dimension is the
rows, and its missing values are taken as 0 (absent). Each item is labelled by its
dimension's coordinate named `name`, `label` or `title`, or ending in "name" (e.g.
`product_name`), else the dimension's own coordinate. Items labelled alike are told apart
by the values of its other coordinates that differ among them, coarsest first (those with
the fewest values: e.g. a product's group, then its category, then the reference), and
every item's other coordinates are shown on hover (but identifiers and links). Its
`"fields"` can choose which coordinates label and describe its rows and columns
(`{"rows": {"label": ..., "keep": [...]}, "cols": ...}`), ignoring the rest.
Coordinates named as the identifier columns above (`cid`, `inchi`, `name`, ...) identify
the chemicals.

```
<xarray.DataArray (compound: 2768, source: 2686)>
Coordinates:
    name              (compound) <U102 '(+)-2-carene' ... 'δ-terpineol'
    cid               (compound) int64 78249 439263 ... 520383 81722
    inchi             (compound) <U244 'InChI=1S/C10H16/c1-7-4-5-8-9(6-7)...
    product_name      (source) <U50 'COGNAC (French brandy)' ... 'ROSEMARY'
    product_category  (source) <U50 'GRAPE BRANDY' ... 'ROSEMARY (Rosma...
    product_group     (source) <U30 'Alcoholic Beverages > 20% alc.' ...
    reference         (source) <U250 '1er Symp. Sci. Int. de Cognac, ...
```

A file can also be a LOTUS bundle (`.sqlite`), made by `python -m app.lotus_build <LOTUS
folder> <bundle>` (README, "Scripting"): LOTUS's structure–organism–paper records, with
the structures' identifiers, names and CIDs, each organism's lineage in every taxonomy
source that has it, and the papers' titles. It is shown over the chemicals of the odor
table that are in it, made again whenever the odor table changes, as its `"fields"` say
(each can be changed in the Admin tab, and is said above its matrix):

```json
"fields": {"layout": "lotus",
           "level": "family",       // columns: "organism" (as LOTUS names them), "species", "genus", "family",
                                    // "order", "class", "phylum" or "kingdom"
           "value": "organisms",    // a cell: how many of the column's organisms it is reported in; "papers"
                                    // reporting it in any of them; "presence" (1 or 0)
           "stereo": "enantiomers", // which structures an odor is: "enantiomers" (its InChI but for tetrahedral
                                    // stereo), "exact", or "skeleton" (its InChIKey's first block)
           "taxonomy": "GBIF Backbone Taxonomy"}  // the source of lineages used first (then the others in turn)
```

An odor is matched by the LOTUS ID the odor table gives it (then by the stereochemistry
setting from that structure), else its InChI, else its CID.

A file can also be a MetaCyc bundle (`.sqlite`), made from MetaCyc's SmartTables by
`python -m app.metacyc_build <raw folder> <bundle>` (README, "Scripting"), or by the
Admin tab's *Import MetaCyc* from the folder uploaded: its compounds,
pathways, reactions, species and BioCyc's genomes of them, in one SQLite file. It is
shown over the chemicals of the odor table that are in it (matched by the MetaCyc ID the
odor table gives, else InChI, InChIKey, CID, the first block of the InChIKey, as MetaCyc
keeps acids as anions, or name), made again whenever the odor table changes, as the
view its `"fields"` choose:

```json
"fields": {"layout": "metacyc",
           "view": "pathways",       // chemicals × the pathways they are in: 1 itself, 0.5 through a class only
           "classes": "fallback"}    // "never", "fallback" (a chemical in none itself), "always"

"fields": {"layout": "metacyc",
           "view": "lca",            // chemicals × chemicals: their pathway LCA distance
           "via": "reactions",       // or "pathways"
           "classes": "fallback",    // a chemical not in the network (or with nothing above it): as the classes it is of
           "max_pathways": null}     // leave out compounds in more than this many pathways (240 by default via pathways)
```

The pathway LCA distance of two chemicals is the smallest sum, over the nodes other than
the two from which both can be reached, of the fewest edges down to each, in a directed
network of compounds and either reactions (a reaction's reactants its parents, its
products its children) or pathways (a pathway's reactants its parents, its products and
every compound of its reactions its children), without 15 currency metabolites (ATP,
NADH, H₂O, ...). Through reactions, with classes as fallback, it gives the matrix of
O'Connell's thesis (ch. 2; MetaCyc 27.0, February 2024) exactly. 0 on the diagonal;
missing where there is no common ancestor.

The two `example_*.csv` files are synthetic placeholders, marked `"example": true` in
the manifest: they are shown only while there is no other reference dataset, built in
or added in the Admin tab.
