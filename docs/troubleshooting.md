# Troubleshooting

## `Disease gene set is empty after filtering`

The genes in the gene-set file do not match `adata.var_names` after optional species mapping. Check gene symbols, capitalization, and species.

## Very high memory use

The marginal scoring implementation converts `adata.X` to a dense NumPy array. Try a smaller data subset first, run on a larger-memory node, or preprocess to a smaller feature set where scientifically appropriate.

## Missing covariate rows

Ensure the covariate TSV index matches `adata.obs_names`. Cell IDs should be strings and should not be duplicated.

## Leiden/metacell errors

Install compatible `leidenalg`/`igraph` packages, or verify that Scanpy can build PCA/neighbors on your input object.

## MAGIC import or runtime errors

Install `magic-impute` and `scprep`, or run with `--imputation none` or `--imputation knn` while debugging file formats.

## Output files are too large

Avoid `--include_ctrl_score` unless downstream analyses require the control normalized score columns.
