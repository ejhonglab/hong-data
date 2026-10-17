# Reference matrices

Fixed matrices shown in the "Reference matrices" tab. They are read from this
directory at request time, so adding or editing a file needs no restart.

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

The two `example_*.csv` files are synthetic placeholders; delete them (and their
manifest entries) once real data is in place.
