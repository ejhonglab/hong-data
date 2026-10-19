# Reference matrices

Matrices shown in the "Reference matrices" tab, built in. Others can be added in the
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
shown with a warning.

```
chemical,cid,group,banana,strawberry
ethyl acetate,8857,esters,2.1,1.4
hexanal,6184,aldehydes,,1.9
```

Rows and columns without a single number (notes, units) are left out, and cells that
are not numbers are shown as missing.

The two `example_*.csv` files are synthetic placeholders; delete them (and their
manifest entries) once real data is in place.
