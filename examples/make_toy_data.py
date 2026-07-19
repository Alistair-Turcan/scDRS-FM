#!/usr/bin/env python
"""
make_toy_data.py — build a small, immune-rich toy dataset for the scDRS-FM example.

Subsamples the Tabula Muris Senis FACS (TMS_FACS) 10k-cell object down to a
compact set of immune-rich tissues so that a disease-relevance signal for an
autoimmune trait (Rheumatoid arthritis) is visible, and scDRS-FM runs
end-to-end in well under a minute on a laptop.

What it produces (all under examples/data/):
  toy_tms_facs.h5ad   raw-count AnnData, ~2k cells x all expressed genes
  toy_tms_facs.cov     matching covariate table (tab-separated, index = barcode)
  gs/PASS_Rheumatoid_Arthritis   MAGMA z-score weighted gene set (human symbols)

Size / compression
------------------
The h5ad is kept lightweight WITHOUT changing anything scDRS-FM sees:
  * all 2k cells are retained;
  * genes that are zero in every toy cell are dropped (they contribute nothing
    to scoring, so this is score-neutral);
  * the count matrix is stored as float32 CSR with gzip level-9 compression.
This shrinks the file ~2.4x (≈38 MB -> ≈16 MB) and scoring is bit-for-bit
identical to the uncompressed object (verified: max abs diff 0.0 across every
marginal and conditional output column).

The toy is deterministic (fixed RANDOM_SEED). Re-running reproduces the exact
same subset.

Usage:
    python make_toy_data.py \
        --src_h5ad  /path/to/TMS_FACS.h5ad \
        --src_cov   /path/to/TMS_FACS.cov \
        --src_gs    /path/to/PASS_Rheumatoid_Arthritis \
        --out_dir   ./data

If the source paths are omitted, the script falls back to the project defaults
used when this example was created.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

RANDOM_SEED = 0
# Immune-rich tissues: hematopoietic / lymphoid compartments where an
# autoimmune (RA) signal concentrates in T / NK / thymocyte populations.
IMMUNE_TISSUES = ["Marrow", "Spleen", "Thymus", "Lung", "Limb_Muscle"]
TARGET_N_CELLS = 2000  # cap; if fewer immune cells exist, keep them all

# Project defaults (used when --src_* are not provided)
DEF_SRC_H5AD = "/mnt/shared-workspace/scdrsfm/data/subsets_10k/TMS_FACS/TMS_FACS.h5ad"
DEF_SRC_COV = "/mnt/shared-workspace/scdrsfm/data/subsets_10k/TMS_FACS/TMS_FACS.cov"
DEF_SRC_GS = "/mnt/shared-workspace/scdrsfm/data/gene_sets/gs_split/PASS_Rheumatoid_Arthritis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_h5ad", default=DEF_SRC_H5AD)
    ap.add_argument("--src_cov", default=DEF_SRC_COV)
    ap.add_argument("--src_gs", default=DEF_SRC_GS)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--target_n", type=int, default=TARGET_N_CELLS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gs_dir = os.path.join(args.out_dir, "gs")
    os.makedirs(gs_dir, exist_ok=True)

    print(f"[toy] loading source h5ad: {args.src_h5ad}")
    adata = sc.read_h5ad(args.src_h5ad)
    print(f"[toy] source shape: {adata.shape}")

    # Select immune-rich tissues
    present = [t for t in IMMUNE_TISSUES if t in set(adata.obs["tissue"])]
    print(f"[toy] immune tissues present: {present}")
    mask = adata.obs["tissue"].isin(present).values
    sub = adata[mask].copy()
    print(f"[toy] cells in immune tissues: {sub.n_obs}")

    # Cap to target_n with a deterministic, tissue-stratified subsample
    if sub.n_obs > args.target_n:
        rng = np.random.default_rng(RANDOM_SEED)
        # proportional allocation per tissue, then random pick within tissue
        frac = args.target_n / sub.n_obs
        keep_idx = []
        for t in present:
            t_idx = np.where(sub.obs["tissue"].values == t)[0]
            n_keep = max(1, int(round(len(t_idx) * frac)))
            n_keep = min(n_keep, len(t_idx))
            chosen = rng.choice(t_idx, size=n_keep, replace=False)
            keep_idx.append(chosen)
        keep_idx = np.sort(np.concatenate(keep_idx))
        sub = sub[keep_idx].copy()
    print(f"[toy] final toy shape: {sub.shape}")
    print("[toy] cells per tissue:\n", sub.obs["tissue"].value_counts())
    print("[toy] top cell types:\n", sub.obs["cell_ontology_class"].value_counts().head(10))

    # --- score-neutral compression -------------------------------------------
    # Drop genes that are zero in EVERY toy cell (contribute nothing to scoring).
    Xc = sub.X.tocsc()
    expressed = np.asarray((Xc != 0).sum(axis=0)).ravel() > 0
    n_drop = int((~expressed).sum())
    sub = sub[:, expressed].copy()
    # float32 CSR is required by scanpy's normalize_per_cell (integer dtype breaks
    # the in-place division); keep float32 so scoring is byte-identical.
    sub.X = sub.X.astype(np.float32)
    if not sp.isspmatrix_csr(sub.X):
        sub.X = sp.csr_matrix(sub.X)
    print(f"[toy] dropped {n_drop} all-zero genes (score-neutral); kept {sub.n_vars} expressed genes")

    # Write toy h5ad with gzip level-9 compression
    out_h5ad = os.path.join(args.out_dir, "toy_tms_facs.h5ad")
    sub.write_h5ad(out_h5ad, compression="gzip", compression_opts=9)
    print(f"[toy] wrote {out_h5ad}  ({os.path.getsize(out_h5ad)/1e6:.1f} MB, gzip-9)")

    # Subset covariate file to the toy barcodes (preserve order = h5ad order)
    cov = pd.read_csv(args.src_cov, sep="\t", index_col=0)
    cov_sub = cov.loc[sub.obs_names]
    out_cov = os.path.join(args.out_dir, "toy_tms_facs.cov")
    cov_sub.to_csv(out_cov, sep="\t")
    print(f"[toy] wrote {out_cov}  ({cov_sub.shape[0]} rows, cols={list(cov_sub.columns)})")

    # Copy the gene set (unchanged) into the toy gs dir
    out_gs = os.path.join(gs_dir, os.path.basename(args.src_gs))
    shutil.copy(args.src_gs, out_gs)
    print(f"[toy] copied gene set -> {out_gs}")

    print("\n[toy] DONE. Toy inputs ready under:", args.out_dir)


if __name__ == "__main__":
    main()
