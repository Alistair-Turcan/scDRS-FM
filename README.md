# scDRS-FM

**scDRS-FM** extends [scDRS](https://github.com/martinjzhang/scDRS) (single-cell
disease-relevance scoring) with **imputation** (MAGIC / ALRA / kNN) and a
**metacell-based conditional analysis** that re-scores disease relevance at the level
of small groups of transcriptomically similar cells. This yields two complementary
readouts per trait:

- **Marginal scores** — a disease-relevance score for **every cell** (this is the
  standard scDRS-style output).
- **Conditional (tagging) scores** — a disease-relevance score for each **metacell**,
  with the "independent signal" of each metacell after conditioning out structure
  shared across metacells. A signal that survives here is robust, not driven by a few
  outlier cells.

This README documents installation, the command-line interface, the input/output file
formats, performance characteristics, and a runnable toy example. It is based on
running the package end-to-end on a 10,000-cell Tabula Muris Senis (FACS) subset across
75 GWAS traits.

---

## Contents
1. [Installation](#1-installation)
2. [Quickstart / toy example](#2-quickstart--toy-example)
3. [Command-line usage](#3-command-line-usage)
4. [Input formats](#4-input-formats)
5. [Output formats](#5-output-formats)
6. [Analyzing results the proper way](#6-analyzing-results-the-proper-way)
7. [Performance notes](#7-performance-notes)
8. [Validation: `--imputation none` reproduces upstream scDRS](#8-validation---imputation-none-reproduces-upstream-scdrs)
9. [Repository layout](#9-repository-layout)
10. [Citation & license](#10-citation--license)

---

## 1. Installation

scDRS-FM is used as a **script directory** (run the `run_*.py` scripts from the repo
root); it is not currently packaged for `pip install`.

```bash
# clone / obtain this directory, then from its root:
python -m venv scdrsfm_env          # or: conda create -n scdrsfm python=3.11
source scdrsfm_env/bin/activate
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`): `numpy`, `pandas`, `scipy`, `scanpy`, `scdrs`,
`statsmodels`, `magic-impute`, `scprep`.

**Tested with** (the configuration used to produce the numbers in this README):
`python 3.11`, `scdrs 1.0.2`, `scanpy 1.10.4`. MAGIC imputation requires
`magic-impute` (+ `scprep`).

Verify the install:

```bash
python -c "import scdrs, scanpy, magic; import scdrs_fm; print('scDRS-FM ready')"
# run this from the repo root so scdrs_fm is importable
```

---

## 2. Quickstart / toy example

A complete, runnable example lives in [`examples/`](examples/):

- **`examples/toy_example.ipynb`** — the recommended starting point. On a small
  immune-rich toy dataset it walks through the *proper* way to analyze the results:
  per-cell **marginal** analysis, per-metacell **conditional** analysis, and the
  **marginal × conditional** step that pulls out **independent cell populations**. It
  ends with the project's two-panel **UMAP** (conditional disease score, and independent
  populations colored with their cell-type composition). The demo trait is
  **rheumatoid arthritis**, which resolves into two distinct populations on this subset.
- **`examples/make_toy_data.py`** — regenerates the toy inputs (deterministic).
- **`examples/README.md`** — how to regenerate the toy and run the notebook.

Minimal command-line run on the toy data (from the repo root):

```bash
python run_scdrs_fm.py \
    examples/data/toy_tms_facs.h5ad \
    examples/data/toy_tms_facs.cov \
    examples/data/out \
    examples/data/gs \
    PASS_Rheumatoid_Arthritis \
    --h5ad_species mouse --imputation magic --flag_raw_count --flag_filter
```

This runs in ~30 s and writes `PASS_Rheumatoid_Arthritis.marginal_score.gz` (per cell)
and `PASS_Rheumatoid_Arthritis.conditional.tagging_score.gz` (per metacell) into
`examples/data/out/`. The toy `.h5ad` is gzip-compressed and stores raw counts;
scoring it is bit-for-bit identical to the uncompressed object.

---

## 3. Command-line usage

The main entry point is **`run_scdrs_fm.py`**:

```
python run_scdrs_fm.py  h5ad_file  cov_file  out  gs_dir  trait [trait ...]  [options]
```

**Positional arguments**

| Argument     | Description |
|--------------|-------------|
| `h5ad_file`  | Input `.h5ad` expression object (see [Input formats](#4-input-formats)). |
| `cov_file`   | Covariate table (TSV). Use `-` or `none` to disable covariate correction. |
| `out`        | Output directory (created if missing). |
| `gs_dir`     | Directory containing gene-set (`.gs`-format) files. |
| `trait ...`  | One or more gene-set **filenames within `gs_dir`** to score (e.g. `PASS_Multiple_sclerosis`). Pass several to score many traits in one run — the one-time preprocessing (imputation, metacell assignment) is shared across all of them. |

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `--h5ad_species {human,mouse}` | `human` | Species of the expression matrix. Gene sets are human; if `mouse`, genes are mapped to mouse homologs internally. |
| `--imputation {magic,none,alra,knn}` | `magic` | Imputation applied after metacell assignment. `none` skips imputation and reduces to standard scDRS scoring (see [§8](#8-validation---imputation-none-reproduces-upstream-scdrs)). |
| `--flag_raw_count` | off | If set, `normalize_per_cell` + `log1p` are applied before scoring. **Set this when your h5ad holds raw counts.** |
| `--flag_filter` | off | If set, basic cell/gene filtering is applied before scoring. |
| `--include_ctrl_score` | off | If set, the 1000 `ctrl_norm_score_*` columns are included in the output tables (large/wide files). |
| `--ablation` | off | If set, emit extra ablation outputs (additional independent-signal methods + regression variants). |

**Worked example** (mouse expression, raw counts, MAGIC imputation, several traits):

```bash
python run_scdrs_fm.py \
    data/TMS_FACS.h5ad \
    data/TMS_FACS.cov \
    results/tms_facs \
    data/gs_split \
    PASS_Multiple_sclerosis UKB_460K.blood_LYMPHOCYTE_COUNT PASS_Rheumatoid_Arthritis \
    --h5ad_species mouse --imputation magic --flag_raw_count --flag_filter
```

---

## 4. Input formats

**Expression (`h5ad_file`)** — an AnnData `.h5ad`.
- `.X` holds the expression matrix; genes are indexed by symbol in `.var_names`.
- If `.X` is raw counts, pass `--flag_raw_count` (applies `normalize_per_cell` + `log1p`).
- `--h5ad_species mouse` triggers internal human→mouse homolog mapping of gene-set genes.
- `.obs` may carry any annotations you want for downstream plotting
  (`cell_ontology_class`, `tissue`, ...); they are not required for scoring.

**Covariates (`cov_file`)** — a tab-separated table.
- First column = cell barcode (must match `h5ad` `.obs_names`); remaining columns are
  covariates regressed out during scoring.
- Example columns used here: `const` (intercept, all 1s), `n_genes`, `sex_male`, `age`.
- Use `-` or `none` as the argument to skip covariate correction entirely.

**Gene sets (`gs_dir` + `trait`)** — one file per trait, in scDRS `.gs` format.
- Tab-separated with a header line `TRAIT<TAB>GENESET`, then one row per trait:
  `trait_name<TAB>GENE1:weight1,GENE2:weight2,...`.
- Genes are **human** symbols; weights are typically MAGMA z-scores.
- The `trait` arguments are the **filenames** in `gs_dir` (no extension), e.g.
  `PASS_Multiple_sclerosis`.

---

## 5. Output formats

For each trait, scDRS-FM writes gzip-compressed, tab-separated tables into `out/`.

### `<trait>.marginal_score.gz` — **per cell**
One row per cell (index = cell barcode).

| Column | Meaning |
|--------|---------|
| `raw_score` | Raw disease-relevance score for the cell. |
| `norm_score` | Normalized score (against control gene sets). **Use this** for most analyses; higher = more disease-relevant. |
| `mc_pval` | Monte-Carlo p-value (across control gene sets). |
| `pval` | Empirical p-value. |
| `nlog10_pval` | −log10(`pval`). |
| `zscore` | z-score corresponding to `pval`. |
| `metacell` | Metacell ID this cell was assigned to. **`-1` = outlier**, excluded from metacell aggregation (so it has no conditional score). |

### `<trait>.conditional.tagging_score.gz` — **per metacell**
One row per metacell (index = metacell ID).

| Column | Meaning |
|--------|---------|
| `cell_ids` | Comma-separated list of member cell barcodes. |
| `metacell_size` | Number of cells in the metacell. |
| `independent_signal` | The metacell's disease signal after conditioning out signal shared across metacells (sequential per-step). |
| `independent_signal_multi` | Multi-component-per-step variant of the independent signal (present in some runs). |
| `raw_score`, `norm_score`, `mc_pval`, `pval`, `nlog10_pval`, `zscore` | As above, but computed at the **metacell** level. |
| `ctrl_norm_score_0 … ctrl_norm_score_999` | Only present with `--include_ctrl_score`: the 1000 control-set normalized scores. |

### `<trait>.conditional_score_full.gz` (only with `--include_ctrl_score`)
The full control-score matrix (the `ctrl_norm_score_*` columns), for users who want the
raw null distribution per metacell.

> **Tip:** the compact `*.conditional.tagging_score.gz` (without control columns) is all
> you need for standard analysis. Omit `--include_ctrl_score` unless you specifically
> need the per-metacell null.

---

## 6. Analyzing results the proper way

The **marginal** and **conditional** tables answer different questions and are best read
**together** (see `examples/toy_example.ipynb` for a complete, executed walkthrough):

1. **Marginal (per cell).** Apply multiple-testing correction across cells
   (e.g. Benjamini–Hochberg on `pval`) and ask *which cell types* carry the signal by
   grouping `norm_score` by a cell-type annotation. Most sensitive, but a few
   high-scoring cells can dominate.

2. **Conditional (per metacell).** Metacells aggregate similar cells and are re-scored;
   a signal that stays significant here is robust. Correct across metacells separately.

3. **Marginal × conditional.** Each cell's `metacell` column links it to a metacell's
   conditional score. Comparing the two (per-cell marginal vs its metacell's conditional
   score, and per-metacell mean-marginal vs conditional) shows whether the enrichment is
   a coherent property of cell populations. High agreement (in the toy example,
   r ≈ 0.9) plus significant metacells concentrated in the expected cell types is a
   confident result. **Remember to drop `metacell == -1` outlier cells** before this
   join.

4. **Visualize on a UMAP.** Color the embedding by cell type, marginal `norm_score`,
   significance, and conditional score. When the score panels light up the same
   populations as the expected cell types, the disease-relevance call is trustworthy.

---

## 7. Performance notes

Measured on this project: **10,000-cell** TMS_FACS subset (mouse), **75 GWAS traits** in
a single run, `--imputation magic`. These are indicative figures on one machine, **not**
formal benchmarks — absolute numbers depend on hardware, thread counts, and data.

| Metric | scDRS-FM (MAGIC) | upstream scDRS |
|--------|------------------|----------------|
| End-to-end wall time (75 traits) | **≈ 28.8 min** (1726 s) | ≈ 105.9 min (6354 s) |
| Peak resident memory | **≈ 5.35 GB** | ≈ 1.42 GB |

- scDRS-FM was **~3.7× faster** end-to-end here, at the cost of **~3.8× more peak
  memory** (the extra memory is from MAGIC imputation and the metacell machinery).
- **Cost structure:** a large **one-time** cost (~140 s) covers load/preprocess,
  metacell assignment, and imputation and is **amortized across all traits** in the run;
  each additional trait then costs only ~20 s. **Pass all your traits to a single
  `run_scdrs_fm.py` call** rather than launching one process per trait.
- On the ~2,000-cell toy, a single-trait run completes in ~30 s total.

---

## 8. Validation: `--imputation none` reproduces upstream scDRS

With `--imputation none`, scDRS-FM's marginal scoring **reduces exactly to standard
scDRS**. Internally it calls the same upstream control-set selection
(`scdrs.method._select_ctrl_geneset` with `n_ctrl=1000`, `mean_var` matching,
`n_bin=20`, `seed=0`), so the per-cell normalized scores match.

In this project, across the full **10,000 cells × 75 traits**, scDRS-FM
(`--imputation none`) marginal `norm_score` was **byte-for-byte identical** to upstream
`scdrs.score_cell` (max |difference| = 0; Pearson r = 1.000000, the only deviations
being floating-point rounding in the correlation itself). This makes `--imputation none`
a useful sanity check that the pipeline is wired correctly before turning on imputation.

---

## 9. Repository layout

```
scDRS-FM-main/
├── README.md                     # this file
├── LICENSE
├── requirements.txt
├── run_scdrs_fm.py               # main CLI: marginal + conditional scoring
├── run_scdrs_fm.ipynb            # notebook version of the run
├── run_decompose_gradients.py    # gradient-decomposition analysis
├── decompose_gradients.ipynb
├── plotting/
│   └── plot_phenotypes.ipynb     # phenotype plotting helpers
├── scdrs_fm/                     # package modules
│   ├── __init__.py
│   ├── data_processing.py        # loading, filtering, metacells, imputation
│   ├── gene_sets.py              # gene-set parsing + homolog mapping
│   ├── marginal_analysis.py      # per-cell scoring
│   ├── conditional_analysis.py   # per-metacell conditional / independent signal
│   └── decompose_gradients.py
└── examples/                     # runnable toy example (see examples/README.md)
    ├── toy_example.ipynb
    ├── make_toy_data.py
    ├── README.md
    └── data/                     # toy inputs + outputs
```

---

## 10. Citation & license

- **scDRS-FM** builds on **scDRS** (Zhang et al., *Nature Genetics* 2022;
  <https://github.com/martinjzhang/scDRS>). Please cite scDRS when using this tool.
- Imputation uses **MAGIC** (van Dijk et al., *Cell* 2018).
- See [`LICENSE`](LICENSE) for license terms.
