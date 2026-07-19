# scDRS-FM toy example

A small, self-contained example that runs scDRS-FM end-to-end and demonstrates the
recommended way to analyze the results: scoring every cell for disease relevance, then
using the conditional (metacell) analysis to pull out **independent cell populations**.

## What's here

| File | Description |
|------|-------------|
| `toy_example.ipynb` | **Start here.** Loads the toy data, points at the precomputed scDRS-FM output, then walks through the marginal (per-cell) analysis, the conditional (per-metacell) analysis, and the **marginal × conditional** step that defines the independent populations. Ends with the two-panel UMAP. Ships executed, with outputs. |
| `make_toy_data.py` | Builds the toy inputs from the full TMS_FACS object (deterministic, `seed=0`). |
| `data/toy_tms_facs.h5ad` | Toy expression: ~2,000 cells from immune-rich TMS_FACS tissues (Marrow, Spleen, Thymus, Lung, Limb_Muscle), mouse, raw counts. gzip-compressed. |
| `data/toy_tms_facs.cov` | Matching covariate table (`const`, `n_genes`, `sex_male`, `age`). |
| `data/gs/PASS_Rheumatoid_Arthritis` | Rheumatoid arthritis gene set (MAGMA z-score weights, human symbols). |
| `data/out/` | Precomputed scDRS-FM output (marginal + conditional score files). |
| `fig_scores_and_independent_populations.png` / `.svg` | The figure the notebook produces. |

The trait is **rheumatoid arthritis** — an autoimmune disease whose signal concentrates in
T / NK compartments — and on this hematopoietic subset it resolves into two distinct
independent populations (a thymic developmental T-cell population and a peripheral
NK / CD8⁺ / mature-T population), which makes the conditional step easy to see.

## The figure

The notebook reproduces the plotting convention used throughout the project analysis
(`plotting/`): a **two-panel UMAP**.

* **Left** — every cell coloured by its scDRS-FM **conditional disease score**
  (`RdBu_r`, centred at 0); the **marginal × conditional causal cells** (significant in
  *both* the per-cell and per-metacell analyses) are drawn larger with a black outline.
* **Right** — the same UMAP with each **independent population** in its own colour, and a
  legend giving the cell-type composition of each population.

## Data size / compression

The `.h5ad` is kept lightweight **without changing anything scDRS-FM sees**:

* all ~2,000 cells are retained;
* genes that are zero in every toy cell are dropped (they contribute nothing to scoring);
* the count matrix is stored as float32 CSR with gzip level-9 compression.

This is **score-neutral**: scoring the compressed object gives bit-for-bit identical
marginal and conditional output (verified: max abs difference 0.0 across every column).

## Run the notebook

From the repository root, with the environment from the top-level `README.md` installed:

```bash
cd examples
jupyter lab toy_example.ipynb      # or: jupyter notebook
```

Or execute it headless:

```bash
cd examples
jupyter nbconvert --to notebook --execute --inplace toy_example.ipynb
```

The notebook computes a UMAP embedding for visualisation (~1 min). The scores are already
provided in `data/out/`, so nothing heavy runs by default.

## Regenerate the scDRS-FM output (optional)

To recompute the scores, run scDRS-FM from the repository root:

```bash
python run_scdrs_fm.py \
    examples/data/toy_tms_facs.h5ad \
    examples/data/toy_tms_facs.cov \
    examples/data/out \
    examples/data/gs \
    PASS_Rheumatoid_Arthritis \
    --h5ad_species mouse \
    --imputation magic \
    --flag_raw_count \
    --flag_filter
```

That finishes in under 30 seconds on a laptop.

## Regenerate the toy data (optional)

The toy inputs are already included. To rebuild them from the full TMS_FACS object,
point the script at your source files:

```bash
python make_toy_data.py \
    --src_h5ad /path/to/TMS_FACS.h5ad \
    --src_cov  /path/to/TMS_FACS.cov \
    --src_gs   /path/to/PASS_Rheumatoid_Arthritis \
    --out_dir  ./data
```

The subsample is deterministic (`seed=0`, tissue-stratified) and applies the same
score-neutral compression, so it reproduces exactly.
