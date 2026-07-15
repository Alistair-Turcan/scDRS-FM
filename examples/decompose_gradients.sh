#!/usr/bin/env bash
set -euo pipefail

scdrs-fm-decompose \
  --traits example_trait.gs \
  --trait_dir results/scdrs_fm \
  --pheno_dir results/phenotypes \
  --out_dir results/decomposition \
  --adata data/expr.h5ad \
  --phenotypes phenotype_a phenotype_b \
  --n_controls 1000
