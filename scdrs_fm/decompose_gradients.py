#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

def _load_outcome_score_table(path: Path, n_controls: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="	", index_col=0)
    keep_cols = ["norm_score"] + [f"ctrl_norm_score_{i}" for i in range(n_controls)]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected outcome columns in {path}: {missing[:5]}")
    out = df[keep_cols].copy()
    return out


def _load_predictor_norm_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="	", index_col=0)
    if "norm_score" not in df.columns:
        raise ValueError(f"Missing expected predictor column 'norm_score' in {path}")
    out = df[["norm_score"]].copy()
    return out


def _expand_conditional_to_cells(path: Path, n_controls: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="	", index_col=0)
    if n_controls is None:
        keep_cols = ["norm_score"]
    else:
        keep_cols = ["norm_score"] + [f"ctrl_norm_score_{i}" for i in range(n_controls)]
    missing = [c for c in keep_cols + ["cell_ids"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing[:5]}")

    rows = []
    vals = df[keep_cols].values
    for i, ids in enumerate(df["cell_ids"].astype(str).values):
        cells = [x.strip() for x in ids.split(",") if x.strip()]
        if not cells:
            continue
        block = np.repeat(vals[[i], :], repeats=len(cells), axis=0)
        blk_df = pd.DataFrame(block, index=cells, columns=keep_cols)
        rows.append(blk_df)

    if not rows:
        print("[decompose] WARNING: no cell_ids expanded")
        return pd.DataFrame(columns=keep_cols)

    out = pd.concat(rows, axis=0)
    out = out[~out.index.duplicated(keep="first")]
    return out


def _single_predictor_ols_coeffs(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = x.shape[0]
    X = np.column_stack([np.ones(n, dtype=np.float64), x])
    XtX = X.T @ X
    XtY = X.T @ Y
    try:
        B = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(XtX) @ XtY
    return B[1, :]


def _multi_predictor_ols_coeffs(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xd = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])
    XtX = Xd.T @ Xd
    XtY = Xd.T @ Y
    try:
        B = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(XtX) @ XtY
    return B[1:, :]


def _stepwise_multivariate_mc(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    univ_fdr: np.ndarray,
    univ_fdr_thresh: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = X.shape[1]
    selected: list[int] = []
    candidate = [int(i) for i in np.flatnonzero(univ_fdr < univ_fdr_thresh)]

    full_coef_main = np.zeros(p, dtype=np.float64)
    full_p = np.ones(p, dtype=np.float64)
    full_var_explained = np.zeros(p, dtype=np.float64)

    def rss_for_cols(cols: list[int]) -> float:
        if len(cols) == 0:
            yhat = np.full(Y.shape[0], Y[:, 0].mean(), dtype=np.float64)
        else:
            Xd = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X[:, cols]])
            coef, *_ = np.linalg.lstsq(Xd, Y[:, 0], rcond=None)
            yhat = Xd @ coef
        r = Y[:, 0] - yhat
        return float(r @ r)

    while len(candidate) > 0:
        base_rss = rss_for_cols(selected)
        cand_meta = []
        for j in candidate:
            cols = selected + [j]
            coef_mat = _multi_predictor_ols_coeffs(X[:, cols], Y)
            cand_coef = coef_mat[-1, :]
            cand_p = _empirical_mc_p(float(cand_coef[0]), cand_coef[1:])
            rss_with = rss_for_cols(cols)
            var_explained = 0.0 if base_rss <= 1e-30 else max(0.0, (base_rss - rss_with) / base_rss)
            cand_meta.append((j, cand_coef, cand_p, var_explained))

        cand_pvals = np.array([m[2] for m in cand_meta], dtype=np.float64)
        cand_fdr = multipletests_safe(cand_pvals)

        passing = []
        for k, meta in enumerate(cand_meta):
            if cand_fdr[k] < univ_fdr_thresh:
                passing.append((meta[0], meta[1], meta[2], cand_fdr[k], meta[3]))

        if len(passing) == 0:
            break

        passing.sort(key=lambda x: x[4], reverse=True)
        best_j, best_coef, best_p, _best_fdr, best_var = passing[0]

        selected.append(int(best_j))
        candidate = [j for j in candidate if j != best_j]

        full_coef_main[int(best_j)] = float(best_coef[0])
        full_p[int(best_j)] = float(best_p)
        full_var_explained[int(best_j)] = float(best_var)

    return full_coef_main, full_p, full_var_explained


def _empirical_mc_p(main_coef: float, ctrl_coefs: np.ndarray) -> float:
    if main_coef <= 0:
        return 1.0
    return float((1.0 + np.sum(main_coef <= ctrl_coefs)) / (1.0 + ctrl_coefs.size))


def _compute_partial_r2_from_univariate_set(y: np.ndarray, X: np.ndarray, univ_p: np.ndarray, p_thresh: float = 0.05) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    univ_p = np.asarray(univ_p, dtype=np.float64)

    p = X.shape[1]
    out = np.zeros(p, dtype=np.float64)
    selected = [int(i) for i in np.flatnonzero(univ_p < p_thresh)]
    if len(selected) == 0:
        return out

    def rss_for_cols(cols: list[int]) -> float:
        if len(cols) == 0:
            yhat = np.full_like(y, y.mean())
        else:
            Xd = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X[:, cols]])
            coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            yhat = Xd @ coef
        r = y - yhat
        return float(r @ r)

    rss_full = rss_for_cols(selected)
    for j in selected:
        reduced = [c for c in selected if c != j]
        rss_reduced = rss_for_cols(reduced)
        if rss_reduced <= 1e-30:
            out[j] = 0.0
        else:
            out[j] = max(0.0, (rss_reduced - rss_full) / rss_reduced)
    return out


def multipletests_safe(pvals: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests

    pvals = np.asarray(pvals, dtype=np.float64)
    if pvals.size == 0:
        return pvals
    _rej, qvals, _a, _b = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return qvals


def _analyze_combo(
    trait: str,
    combo_name: str,
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    phenotype_names: List[str],
    *,
    cond_ridge: float,
    univ_fdr_thresh: float,
) -> pd.DataFrame:

    idx = X_df.index.intersection(Y_df.index)
    if idx.size == 0:
        raise ValueError(f"No overlapping samples for trait={trait}, combo={combo_name}")

    X = X_df.loc[idx, phenotype_names].values.astype(np.float64)
    Y = Y_df.loc[idx].values.astype(np.float64)

    n_ctrl = Y.shape[1] - 1
    p = len(phenotype_names)

    # Marginal associations: univariate OLS per phenotype.
    univ_coef = np.zeros(p, dtype=np.float64)
    univ_p = np.ones(p, dtype=np.float64)
    for j in range(p):
        coef_all = _single_predictor_ols_coeffs(X[:, j], Y)
        univ_coef[j] = float(coef_all[0])
        univ_p[j] = _empirical_mc_p(univ_coef[j], coef_all[1:])

    univ_fdr = multipletests_safe(univ_p)

    # Conditional associations: iterative stepwise with candidate pool defined by univariate FDR threshold.
    full_coef_main, full_p, full_var_explained = _stepwise_multivariate_mc(
        X,
        Y,
        univ_fdr=univ_fdr,
        univ_fdr_thresh=univ_fdr_thresh,
    )

    # Partial R2: use all predictors with univariate p < 0.05, remove one at a time.
    partial_r2 = _compute_partial_r2_from_univariate_set(y=Y[:, 0], X=X, univ_p=univ_p, p_thresh=0.05)

    out = pd.DataFrame(
        {
            "trait": trait,
            "combo": combo_name,
            "phenotype": phenotype_names,
            "univariate_coef": univ_coef,
            "univariate_mc_pval": univ_p,
            "multivariate_coef": full_coef_main,
            "multivariate_mc_pval": full_p,
            "multivariate_var_explained": full_var_explained,
            "partial_r2": partial_r2,
            "n_samples": int(idx.size),
            "n_ctrl": int(n_ctrl),
        }
    )
    out["univariate_fdr"] = pd.Series(univ_fdr, index=out.index)

    return out


def run_decomposition(
    adata_file: str,
    trait_dir: str,
    pheno_dir: str,
    out_dir: str,
    traits: List[str],
    phenotypes: List[str],
    additional_phenotypes: List[str],
    *,
    cell_type_col: str = "",
    cell_type: str = "",
    n_controls: int = 1000,
    cond_ridge: float = 1e-10,
    include_negative_phenotypes: bool = False,
    univ_fdr_thresh: float = 0.05,
) -> None:
    t_all = time.perf_counter()
    print("[decompose] Starting decomposition")

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    print(f"[decompose] Loading adata: {adata_file}")
    adata = sc.read_h5ad(adata_file)
    sc.pp.filter_cells(adata, min_genes=250)
    sc.pp.filter_genes(adata, min_cells=50)

    if cell_type_col and cell_type:
        print(f"[decompose] Subsetting {cell_type_col} == {cell_type}")
        adata = adata[adata.obs[cell_type_col] == cell_type].copy()

    cell_labels = pd.Index(adata.obs_names)
    all_pheno = phenotypes + additional_phenotypes
    print(f"[decompose] n_cells={cell_labels.size:,} n_phenotypes={len(all_pheno)} n_traits={len(traits)}")

    # Phenotype predictors
    pheno_marginal = {}
    pheno_conditional = {}
    phenotype_names_model = []
    pheno_iter = tqdm(all_pheno, desc="Unpacking phenotypes", unit="pheno")
    for p in pheno_iter:
        df_marg = _load_predictor_norm_table(Path(pheno_dir) / f"{p}.marginal_score.gz")
        df_cond = _expand_conditional_to_cells(Path(pheno_dir) / f"{p}.conditional.tagging_score.gz", n_controls=None)[["norm_score"]]

        pos_name = f"{p} (+)"
        pheno_marginal[pos_name] = df_marg.rename(columns={"norm_score": pos_name})
        pheno_conditional[pos_name] = df_cond.rename(columns={"norm_score": pos_name})
        phenotype_names_model.append(pos_name)

        if include_negative_phenotypes:
            neg_name = f"{p} (-)"
            pheno_marginal[neg_name] = (-df_marg).rename(columns={"norm_score": neg_name})
            pheno_conditional[neg_name] = (-df_cond).rename(columns={"norm_score": neg_name})
            phenotype_names_model.append(neg_name)

    X_marginal = pd.concat([pheno_marginal[pn] for pn in phenotype_names_model], axis=1)
    X_marginal = X_marginal.loc[X_marginal.index.intersection(cell_labels)]

    X_conditional = pd.concat([pheno_conditional[pn] for pn in phenotype_names_model], axis=1)
    X_conditional = X_conditional.loc[X_conditional.index.intersection(cell_labels)]


    trait_iter = tqdm(traits, desc="Processing traits", unit="trait")
    for trait in trait_iter:
        t_trait = time.perf_counter()

        y_marg = _load_outcome_score_table(Path(trait_dir) / f"{trait}.marginal_score.gz", n_controls)
        y_marg = y_marg.loc[y_marg.index.intersection(cell_labels)]

        y_cond = _expand_conditional_to_cells(Path(trait_dir) / f"{trait}.conditional.tagging_score.gz", n_controls=n_controls)
        y_cond = y_cond.loc[y_cond.index.intersection(cell_labels)]

        res_mm = _analyze_combo(
            trait,
            "marg_marg",
            X_marginal,
            y_marg,
            phenotype_names_model,
            cond_ridge=cond_ridge,
            univ_fdr_thresh=univ_fdr_thresh,
        )
        # res_mc = _analyze_combo(trait, "marg_cond", X_marginal, y_cond, phenotype_names_model, cond_ridge=cond_ridge, univ_fdr_thresh=univ_fdr_thresh)
        # res_cc = _analyze_combo(trait, "cond_cond", X_conditional, y_cond, phenotype_names_model, cond_ridge=cond_ridge, univ_fdr_thresh=univ_fdr_thresh)

        out_mm = outp / f"{trait}.marg-marg.decomposition.tsv.gz"
        # out_mc = outp / f"{trait}.marg-cond.decomposition.tsv.gz"
        # out_cc = outp / f"{trait}.cond-cond.decomposition.tsv.gz"

        res_mm.to_csv(out_mm, sep="\t", index=False, compression="gzip")
        # res_mc.to_csv(out_mc, sep="\t", index=False, compression="gzip")
        # res_cc.to_csv(out_cc, sep="\t", index=False, compression="gzip")

        trait_iter.set_postfix({"trait": trait, "sec": f"{time.perf_counter() - t_trait:,.2f}"})

    print(f"\n[decompose] ALL DONE in {time.perf_counter() - t_all:,.3f} s")
