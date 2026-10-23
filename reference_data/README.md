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
shown with a warning.

```
chemical,cid,group,banana,strawberry
ethyl acetate,8857,esters,2.1,1.4
hexanal,6184,aldehydes,,1.9
```

A matrix of glomeruli (by anything else, with no odors), e.g. counts of PNs or of
synapses per glomerulus, is found by its labels: when at least 3 of the labels on one
axis, and at least half of them, are glomerulus names as accepted for datasets' ROIs
(`DM2`, `VA1d`, ...; but for case). The Compare tab can then compare it with datasets
of glomeruli, on their glomeruli.

```
glomerulus,PN count,KC inputs
DM2,3,120
VA2,2,85
```

Rows and columns without a single number (notes, units) are left out, and cells that
are not numbers are shown as missing.

A file can also be an xarray DataArray of two dimensions saved as netCDF (`.nc`, `.nc4`,
by `DataArray.to_netcdf`), read with `xarray.load_dataarray`. Its first dimension is the
rows, and its missing values are taken as 0 (absent). Each item is labelled by its
dimension's coordinate named `name`, `label` or `title`, or ending in "name" (e.g.
`product_name`), else the dimension's own coordinate. Items labelled alike are told apart
by the values of its other coordinates that differ among them, coarsest first (those with
the fewest values: e.g. a product's group, then its category, then the reference), and
every item's other coordinates are shown on hover (but identifiers and links).
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

The two `example_*.csv` files are synthetic placeholders; delete them (and their
manifest entries) once real data is in place.
