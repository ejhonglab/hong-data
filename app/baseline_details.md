### The reference models

Eight reference models of KC responses, all made by pyolfsysm from one definition each
(`pyolfsysm.baselines`) and written to `data/model_outputs/olfdb_baselines/` on tensor: the
KC-remapping experiments' bouton base, three controls that replace the connectome's identities,
and a twin of each of the four restricted to the glomeruli with ORN input.

### The bouton base

- **Wiring.** The hemibrain mushroom body as a claw and bouton model: each KC has its claws as the
  connectome has them, and each claw is fed by the one PN bouton it contacts.
- **APL.** APL's weights are the connectome's synapse counts: onto and from each claw (APL→KC,
  KC→APL) and onto and from each bouton (APL→bouton, bouton→APL). Each set is scaled to a mean
  of 1 before tuning. APL is driven by the claws' input (whether or not the KC spikes) and by the
  boutons, and it inhibits both.
- **Tuning.** The somatic threshold and the APL scales are tuned so that 5 % of KC–odor pairs
  respond with APL on. A response is 2 or more spikes.
- **APL off (acute).** The tuned threshold is kept. APL→KC is set to 0, and with it KC→APL, which is
  derived from it. APL→bouton is set to 0, and bouton→APL is kept at its tuned scale.
- **Input.** ORN responses to the 16 APL16 odors (megamat and validation2, 36 glomeruli). paa,
  -aPine and euc are at the nearest concentration measured (paa −4 for −3, -aPine −1.5 and euc −1.75
  for −2).
- **Check.** Made again here, it is bit-identical to the archived run of the bouton-base experiment
  (E13): the same spike counts, threshold and APL scales.

### Uniform claws, uniform APL

- Each KC gets 7 claws. Each claw draws a glomerulus with equal chance from the 54 hemibrain
  glomeruli that have boutons, then a random bouton of that glomerulus (seed 0).
- pyolfsysm's older `uniform-boutons` draws each claw's glomerulus in proportion to its number of
  boutons instead (the same as drawing a bouton at random). Here no glomerulus is favoured.
- Every claw's APL→KC and KC→APL weight and every bouton's APL→bouton and bouton→APL weight is 1,
  which the base's mean-1 scaling leaves at 1.
- Everything else is as in the bouton base.

### The shuffled connectome

- One seed (`shuffle_connectome_seed`, 0 here) draws five permutations, always in this order: the
  claw rows of the PN→KC wiring, then APL→claw, claw→APL, APL→bouton and bouton→APL.
- Claw rows are permuted across all claws: each claw takes another claw's bouton. Each KC keeps its
  number of claws, each bouton its total output, and the total weight is unchanged.
- **Pairs kept.** One permutation per kind of site for both directions: a claw's (or bouton's) APL→X
  and X→APL weights move together.
- **Pairs broken.** The four APL permutations are independent.
- The kept and broken versions of one seed share the PN→KC permutation, so their tuned thresholds
  are the same.

### The input-glomeruli twins

`input_glomeruli_only=True`; the names end in `_input_glomeruli`.

- **Input glomeruli** are those in the ORN input the model is given: the 36 of the APL16 input
  here, and on another dataset, that dataset's.
- **Uniform model.** Claws draw only from the input glomeruli, so every claw gets odor input. Every
  KC keeps its 7 claws.
- **Connectome models.** The boutons of glomeruli without input are dropped, then the claws left with
  no input, then the KCs left with no claws. All of this happens before the shuffle and before the
  APL weights are scaled to mean 1.
- **Why it matters.** In the full models, glomeruli without input have no odor input but keep a
  spontaneous rate (the mean Hallem rate). So their claws give KCs spontaneous drive and take part in
  APL's input and output. In the uniform model about a third of the claws are on them.

What the restriction keeps of the bouton base, with the APL16 input:

| | full | input glomeruli only |
|---|---:|---:|
| glomeruli | 54 | 36 |
| boutons | 389 | 307 |
| claws | 9867 | 8188 |
| KCs | 1732 | 1700 |

The bouton base's tuned parameters with the restriction, against without it:

| | change |
|---|---:|
| threshold | +3.5 % |
| APL→KC scale | −4.5 % |
| KC→APL scale | −2.7 % |
| APL→bouton scale | −4.5 % |
| bouton→APL scale | +21.0 % |
| response rate, APL on | −1.0 % (0.0516 to 0.0511) |
| response rate, APL off | +0.3 % (0.0978 to 0.0981) |

### The eight models as tuned

Absolute values. The response rate is the fraction of KC–odor pairs responding; APL off is the acute
run.

| model | KCs | threshold | APL→KC | KC→APL | APL→bouton | bouton→APL | APL on | APL off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bouton_base | 1732 | 254.4 | 0.1343 | 7.75e-05 | 0.1343 | 3.45e-04 | 0.0516 | 0.0978 |
| uniform_glomeruli_claws_uniform_apl | 1732 | 206.7 | 0.1193 | 6.89e-05 | 0.1193 | 3.07e-04 | 0.0548 | 0.0978 |
| shuffled_connectome_pairs_kept_seed0 | 1732 | 245.7 | 0.1425 | 8.23e-05 | 0.1425 | 3.66e-04 | 0.0496 | 0.0980 |
| shuffled_connectome_pairs_broken_seed0 | 1732 | 245.7 | 0.1412 | 8.16e-05 | 0.1412 | 3.63e-04 | 0.0491 | 0.0980 |
| bouton_base_input_glomeruli | 1700 | 263.2 | 0.1282 | 7.54e-05 | 0.1282 | 4.18e-04 | 0.0511 | 0.0981 |
| uniform_glomeruli_claws_uniform_apl_input_glomeruli | 1732 | 298.5 | 0.0909 | 5.25e-05 | 0.0909 | 2.96e-04 | 0.0532 | 0.0977 |
| shuffled_connectome_pairs_kept_seed0_input_glomeruli | 1700 | 252.5 | 0.1193 | 7.02e-05 | 0.1193 | 3.89e-04 | 0.0539 | 0.0979 |
| shuffled_connectome_pairs_broken_seed0_input_glomeruli | 1700 | 252.5 | 0.1193 | 7.02e-05 | 0.1193 | 3.89e-04 | 0.0532 | 0.0979 |

### Running them on other datasets

- **olfdb's model worker** runs each model again from its `params.csv` row (its `base_kwargs_json`),
  at its tuned threshold and APL scales or tuned again. The seed draws the same random wiring or
  shuffle again.
- **The twins** restrict to the glomeruli of the dataset they are run on, so on other data they keep
  other glomeruli than here.
- **Rebuilding from saved pieces** (`pyolfsysm.rebuild`, or a saved wiring or APL weights passed to
  `fit_mb_model`) refuses these options. The saved pieces already contain the shuffle, the uniform
  draw or the restriction.
- **The seed columns.** `seed` is the random wiring's seed and is filled only for the uniform models.
  A shuffle's seed is its `shuffle_connectome_seed` column. olfdb passes a nonzero `seed` column on
  to the model.

### Where the code is

- pyolfsysm, `pyolfsysm.baselines`: `BOUTON_BASE_KWS`, `BASELINE_VARIANTS` (the eight),
  `run_baseline` (APL on, tuned, then acute APL off) and `save_baseline_outputs`.
- pyolfsysm's `fit_mb_model` settings: `pn2kc_connections='uniform-glomeruli-boutons'` (with
  `n_claws`, `seed`), `uniform_apl_weights`, `shuffle_connectome_seed`, `shuffle_apl_pairs` and
  `input_glomeruli_only`.
- olfdb (hong-data): `scripts/make_baseline_models.py` makes the folder and registers the models as
  baselines. The Summary texts are in `app/baseline_summaries.json` and this text is
  `app/baseline_details.md`.
- The outputs are in `Tom/claude_kc_remapping/data/model_outputs/olfdb_baselines/` on tensor:
  `params.csv`, `provenance.json`, and each model's spike counts and responses with APL on and off.
