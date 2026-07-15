#!/usr/bin/env bash
set -euo pipefail

scdrs-fm \
  data/expr.h5ad \
  none \
  results/scdrs_fm \
  data/gene_sets \
  example_trait.gs \
  --h5ad_species human \
  --imputation none
