# scDRS-FM

scDRS-FM is a research pipeline for computing single-cell disease-relevance scores and follow-up metacell/feature-module analyses from AnnData (`.h5ad`) expression matrices and scDRS-style gene sets. It builds on the original [scDRS](https://github.com/martinjzhang/scDRS) workflow by adding metacell aggregation, optional imputation, conditional independent-signal scoring, and phenotype-gradient decomposition utilities.

> **Status:** research software. Validate results for your data set and cite the upstream methods you rely on.

## What this repository contains

- `run_scdrs_fm.py` — end-to-end scoring pipeline for one `.h5ad` file and one or more trait gene-set files.
- `run_decompose_gradients.py` — second-stage phenotype-gradient decomposition over scDRS-FM outputs.
- `scdrs_fm/` — reusable Python modules for data processing, gene-set handling, marginal scoring, conditional scoring, and decomposition.
- `*.ipynb` notebooks — exploratory examples for running and plotting analyses.

## Features

- AnnData/Scanpy-based input handling.
- scDRS-compatible gene-set parsing and control-gene selection.
- Optional preprocessing for raw count matrices.
- Leiden-based metacell construction and metacell-level conditional scoring.
- Optional MAGIC, kNN, ALRA, or no imputation.
- Marginal cell-level scDRS-like scores.
- Conditional metacell-level independent-signal scores.
- Optional retention of control normalized scores for downstream empirical testing.
- Phenotype-gradient decomposition across marginal and conditional outputs.

## Repository layout

```text
.
├── scdrs_fm/
│   ├── conditional_analysis.py
│   ├── data_processing.py
│   ├── decompose_gradients.py
│   ├── gene_sets.py
│   └── marginal_analysis.py
├── docs/
│   ├── file_formats.md
│   ├── installation.md
│   ├── troubleshooting.md
│   └── usage.md
├── examples/
│   ├── minimal_pipeline.sh
│   └── decompose_gradients.sh
├── run_scdrs_fm.py
├── run_decompose_gradients.py
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
git clone https://github.com/<your-org>/scDRS-FM.git
cd scDRS-FM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the main pipeline:

```bash
scdrs-fm \
  data/expr.h5ad \
  data/cov.tsv \
  results/scdrs_fm \
  data/gene_sets \
  trait_a.gs trait_b.gs \
  --h5ad_species human \
  --flag_raw_count \
  --flag_filter \
  --imputation magic
```

The same entry point can be run directly from the script if you do not install the package:

```bash
python run_scdrs_fm.py data/expr.h5ad data/cov.tsv results/scdrs_fm data/gene_sets trait_a.gs
```

Use `-`, `none`, or `null` as the covariate file argument to disable covariate adjustment.

## Documentation

- [Installation guide](docs/installation.md)
- [Usage guide](docs/usage.md)
- [Input and output file formats](docs/file_formats.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](CONTRIBUTING.md)
- [Citation guidance](CITATION.md)

## Main command reference

```bash
scdrs-fm --help
```

Positional arguments:

1. `h5ad_file` — input AnnData file.
2. `cov_file` — tab-separated covariate table, or `-`/`none` to disable.
3. `out` — output directory.
4. `gs_dir` — directory containing gene-set files.
5. `traits` — one or more gene-set filenames inside `gs_dir`.

Common options:

- `--h5ad_species {human,mouse}` — species of expression matrix genes.
- `--flag_raw_count` — normalize and log-transform raw counts before scoring.
- `--flag_filter` — apply basic cell/gene filtering before scoring.
- `--imputation {magic,none,alra,knn}` — imputation method after metacell assignment.
- `--include_ctrl_score` — include normalized control scores in output tables.
- `--ablation` — emit additional conditional-analysis ablation outputs.

## Expected outputs

For each trait gene set, the pipeline writes compressed tab-separated score tables in the output directory. Core outputs include:

- `<trait>.marginal_score.gz` — cell-level marginal scores.
- `<trait>.marginal_score_full.gz` — cell-level marginal scores with control columns when `--include_ctrl_score` is enabled.
- `<trait>.conditional_score.gz` — metacell-level conditional scores.
- Additional conditional/ablation tables when `--ablation` is enabled.

See [file formats](docs/file_formats.md) for column definitions.

## Relationship to scDRS

The original scDRS project provides the disease-relevance scoring method, command-line tools, and documentation for standard single-cell disease association analyses. scDRS-FM uses scDRS components for preprocessing/control gene set selection and extends the workflow with metacell conditional analysis and phenotype-gradient decomposition. If you use scDRS-FM, you should also review and cite the original scDRS work where appropriate.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
