from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def normalize_scores_against_controls(
    v_raw_score: np.ndarray,
    mat_ctrl_raw_score: np.ndarray,
    v_var_ratio_c2t: np.ndarray,
    save_intermediate: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    v_raw_score = np.asarray(v_raw_score, dtype=np.float64)
    mat_ctrl_raw_score = np.asarray(mat_ctrl_raw_score, dtype=np.float64)
    v_var_ratio_c2t = np.asarray(v_var_ratio_c2t, dtype=np.float64)

    if save_intermediate is not None:
        np.savetxt(save_intermediate + ".raw_score.tsv.gz", v_raw_score, fmt="%.9e", delimiter="\t")
        np.savetxt(save_intermediate + ".ctrl_raw_score.tsv.gz", mat_ctrl_raw_score, fmt="%.9e", delimiter="\t")

    ind_zero_score = v_raw_score == 0
    ind_zero_ctrl_score = mat_ctrl_raw_score == 0

    v_raw_score = v_raw_score - v_raw_score.mean()
    mat_ctrl_raw_score = mat_ctrl_raw_score - mat_ctrl_raw_score.mean(axis=0)
    mat_ctrl_raw_score = mat_ctrl_raw_score / np.sqrt(v_var_ratio_c2t)

    v_mean = mat_ctrl_raw_score.mean(axis=1)
    v_std = mat_ctrl_raw_score.std(axis=1)
    v_norm_score = (v_raw_score - v_mean) / v_std
    mat_ctrl_norm_score = ((mat_ctrl_raw_score.T - v_mean) / v_std).T

    v_norm_score = v_norm_score - v_norm_score.mean()
    mat_ctrl_norm_score = mat_ctrl_norm_score - mat_ctrl_norm_score.mean(axis=0)

    norm_score_min = min(v_norm_score.min(), mat_ctrl_norm_score.min())
    v_norm_score[ind_zero_score] = norm_score_min - 1e-3
    mat_ctrl_norm_score[ind_zero_ctrl_score] = norm_score_min

    return v_norm_score, mat_ctrl_norm_score


def empirical_p_values(v_t: np.ndarray, v_t_null: np.ndarray) -> np.ndarray:
    v_t = np.array(v_t)
    v_t_null = np.array(v_t_null)
    v_t_null = np.sort(v_t_null)
    v_pos = np.searchsorted(v_t_null, v_t, side="left")
    v_p = (v_t_null.shape[0] - v_pos + 1) / (v_t_null.shape[0] + 1)
    return v_p


def build_scdrs_score_table(
    index: pd.Index,
    v_raw_score: np.ndarray,
    mat_ctrl_raw_score: np.ndarray,
    v_var_ratio_c2t: np.ndarray,
    *,
    include_ctrl_norm_scores: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    v_raw_score = np.asarray(v_raw_score, dtype=np.float64)
    mat_ctrl_raw_score = np.asarray(mat_ctrl_raw_score, dtype=np.float64)
    n_ctrl = mat_ctrl_raw_score.shape[1]

    v_norm_score, mat_ctrl_norm_score = normalize_scores_against_controls(
        v_raw_score, mat_ctrl_raw_score, v_var_ratio_c2t, save_intermediate=None
    )

    mc_p = (1.0 + (mat_ctrl_norm_score.T >= v_norm_score).sum(axis=0)) / (1.0 + n_ctrl)

    pooled_p = empirical_p_values(v_norm_score, mat_ctrl_norm_score.flatten())
    nlog10_pooled_p = -np.log10(pooled_p)
    pooled_z = -stats.norm.ppf(pooled_p)
    pooled_z = np.clip(pooled_z, -10, 10)

    dic_res = {
        "raw_score": v_raw_score,
        "norm_score": v_norm_score,
        "mc_pval": mc_p,
        "pval": pooled_p,
        "nlog10_pval": nlog10_pooled_p,
        "zscore": pooled_z,
    }

    if include_ctrl_norm_scores:
        for i in range(n_ctrl):
            dic_res[f"ctrl_norm_score_{i}"] = mat_ctrl_norm_score[:, i]

    df_res = pd.DataFrame(index=index, data=dic_res, dtype=np.float32)
    return df_res, v_norm_score, mat_ctrl_norm_score


def fit_marginal_association_effects(
    Mw: np.ndarray,
    ones_w: np.ndarray,
    Yw: np.ndarray,
):
    s11 = float(ones_w @ ones_w)
    m1 = (Mw.T @ ones_w).astype(np.float64, copy=False)
    y1 = (ones_w @ Yw).astype(np.float64, copy=False)
    Mw64 = Mw.astype(np.float64, copy=False)
    diagMM = np.sum(Mw64 * Mw64, axis=0)
    MY = (Mw.T @ Yw).astype(np.float64, copy=False)

    denom = diagMM - (m1 * m1) / s11
    numer = MY - (m1[:, None] / s11) * y1[None, :]

    beta = np.full((Mw.shape[1], Yw.shape[1]), np.nan, dtype=np.float64)
    ok = denom > 1e-20
    beta[ok, :] = numer[ok, :] / denom[ok, None]
    return beta


def run_marginal_analysis(
    adata,
    Z: np.ndarray,
    Z_ctrl: np.ndarray,
    weights: np.ndarray,
    v_var_ratio_c2t: np.ndarray,
    *,
    include_ctrl_score: bool,
) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()

    X_cell = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_cell = np.asarray(X_cell, dtype=np.float64, order="C")
    n_cells, n_genes = X_cell.shape
    print(f"[marginal] X for marginal scoring: n_cells={n_cells:,} n_genes={n_genes:,}")

    sqrt_w = np.sqrt(weights.astype(np.float64, copy=False))
    ones_w = sqrt_w

    Y_all = np.column_stack([Z, Z_ctrl.T])
    Yw = Y_all * sqrt_w[:, None]

    Mw_cells = (X_cell.T * sqrt_w[:, None]).astype(np.float64, copy=False)

    beta = fit_marginal_association_effects(Mw=Mw_cells, ones_w=ones_w, Yw=Yw)

    beta_orig = beta[:, 0]
    beta_ctrl = beta[:, 1:].T

    W = float(weights.sum())
    xw_sum = X_cell @ weights
    x2w_sum = (X_cell * X_cell) @ weights
    xbar = xw_sum / W
    Sxx = x2w_sum - W * (xbar * xbar)
    Sxx = np.maximum(Sxx, 0.0)

    T0 = float(weights @ Z)
    T_ctrl = Z_ctrl @ weights

    v_raw = xbar + beta_orig * (Sxx / T0)
    mat_ctrl_raw = np.empty((n_cells, Z_ctrl.shape[0]), dtype=np.float64)
    for c in range(Z_ctrl.shape[0]):
        Tc = float(T_ctrl[c])
        if not np.isfinite(Tc) or Tc == 0.0:
            mat_ctrl_raw[:, c] = np.nan
        else:
            mat_ctrl_raw[:, c] = xbar + beta_ctrl[c, :] * (Sxx / Tc)

    df_res, _, _ = build_scdrs_score_table(
        index=adata.obs_names,
        v_raw_score=v_raw,
        mat_ctrl_raw_score=mat_ctrl_raw,
        v_var_ratio_c2t=v_var_ratio_c2t,
        include_ctrl_norm_scores=include_ctrl_score,
    )

    return df_res, time.perf_counter() - t0


def save_marginal_results(df_marginal: pd.DataFrame, *, out_folder: Path, score_basename: str) -> float:
    t0 = time.perf_counter()
    marg_out_file = out_folder / f"{score_basename}.marginal_score.gz"
    df_marginal.to_csv(marg_out_file, compression="gzip", sep="\t")
    print(f"[main] Saved marginal scores -> {marg_out_file}")
    return time.perf_counter() - t0


def save_marginal_results_split_ctrl(
    df_marginal: pd.DataFrame,
    *,
    out_folder: Path,
    score_basename: str,
) -> float:
    t0 = time.perf_counter()
    ctrl_cols = [c for c in df_marginal.columns if str(c).startswith("ctrl_norm_score_")]
    keep_cols = [c for c in df_marginal.columns if c not in ctrl_cols]

    marg_out_file = out_folder / f"{score_basename}.marginal_score.gz"
    df_marginal.loc[:, keep_cols].to_csv(marg_out_file, compression="gzip", sep="\t")
    print(f"[main] Saved marginal scores (without ctrl columns) -> {marg_out_file}")

    marg_full_out_file = out_folder / f"{score_basename}.marginal_score_full.gz"
    df_marginal.to_csv(marg_full_out_file, compression="gzip", sep="\t")
    print(f"[main] Saved marginal full scores (with ctrl columns) -> {marg_full_out_file}")
    return time.perf_counter() - t0
