# File formats

## AnnData input (`.h5ad`)

The expression matrix is read from `adata.X`.

Expected conventions:

- Rows (`adata.obs_names`) are cell identifiers.
- Columns (`adata.var_names`) are gene symbols matching the gene-set files.
- If `--flag_raw_count` is supplied, `adata.X` is treated as raw counts and normalized/log-transformed.
- If `--flag_filter` is supplied, basic cell/gene filtering is applied before scoring.

## Covariate file

A tab-separated table with cell identifiers as the index and covariates as columns:

```text
cell_id	batch	n_genes
cell_001	batch_a	3210
cell_002	batch_a	2980
cell_003	batch_b	4102
```

Pass `none`, `null`, `-`, or `na` to disable covariate correction.

## Gene-set file

scDRS-style two-line gene-set format:

```text
TRAIT	GENESET
trait_name	GENE1,GENE2,GENE3:0.75,GENE4:1.25
```

Weights are optional. Unweighted genes receive weight `1.0`.

## Marginal score output

`<trait>.marginal_score.gz` is a gzip-compressed TSV indexed by cell ID. Core columns:

- `raw_score` — unnormalized disease/gene-set score.
- `norm_score` — normalized score against matched control gene sets.
- `mc_pval` — Monte Carlo p-value from control scores for the same cell.
- `pval` — pooled empirical p-value.
- `nlog10_pval` — `-log10(pval)`.
- `zscore` — normal-quantile transformed pooled p-value.
- `metacell` — metacell label, when metacells were assigned.

When `--include_ctrl_score` is enabled, `<trait>.marginal_score_full.gz` contains `ctrl_norm_score_0`, `ctrl_norm_score_1`, ... columns.

## Conditional score output

`<trait>.conditional_score.gz` is a metacell-level TSV. It includes normalized disease/control score columns and metadata used by the conditional independent-signal procedure. The exact ablation columns depend on whether `--ablation` was enabled.

## Decomposition output

The decomposition command writes TSV summaries for combinations of marginal and conditional trait/phenotype scores. Outputs include regression coefficients, empirical p-values/FDR values, and variance-explained summaries for phenotype gradients.
