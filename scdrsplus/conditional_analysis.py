from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, eigsh
from statsmodels.stats.multitest import multipletests

from .marginal_analysis import build_scdrs_score_table


def build_projected_gram_matrices(
    Mw: np.ndarray,
    ones_w: np.ndarray,
    Yw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    Mw64 = np.asarray(Mw, dtype=np.float64)
    ones64 = np.asarray(ones_w, dtype=np.float64)
    Yw64 = np.asarray(Yw, dtype=np.float64)

    s11 = float(ones64 @ ones64)
    if not np.isfinite(s11) or s11 <= 0:
        raise RuntimeError("Invalid s11 in intercept projection.")

    m1 = (Mw64.T @ ones64).astype(np.float64, copy=False)
    y1 = (ones64 @ Yw64).astype(np.float64, copy=False)

    MM = (Mw64.T @ Mw64).astype(np.float64, copy=False)
    MY = (Mw64.T @ Yw64).astype(np.float64, copy=False)

    S = MM - (m1[:, None] * m1[None, :]) / s11
    XY = MY - (m1[:, None] * y1[None, :]) / s11
    S = 0.5 * (S + S.T)

    y0 = Yw64[:, 0]
    y0Ty0 = float(y0 @ y0)
    y0cTy0c = y0Ty0 - (float(y1[0]) * float(y1[0])) / s11
    y0cTy0c = float(max(y0cTy0c, 0.0))
    return S, XY, s11, y0cTy0c


def fit_conditional_effects_with_evidence_shrinkage(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    y0cTy0c: float,
    n_eff: int,
    ridge: float,
    progress: bool = False,
) -> Tuple[np.ndarray, float, float, float]:
    S = np.asarray(S, dtype=np.float64)
    XY = np.asarray(XY, dtype=np.float64)
    n, _ = S.shape
    n_resp = XY.shape[1]
    denom_floor = 1e-20

    diagS = np.diag(S).copy()
    diagS = np.maximum(diagS, 0.0)
    good = diagS > denom_floor

    beta = np.full((n, n_resp), np.nan, dtype=np.float64)
    if good.sum() == 0:
        return beta, 0.0, np.nan, 0.0

    xy0 = XY[:, 0]
    d, Q = np.linalg.eigh(S)
    d = np.maximum(d, 0.0)
    qt_xy0 = Q.T @ xy0

    n_eff = max(int(n_eff), 1)
    y0cTy0c = float(max(float(y0cTy0c), denom_floor))

    mu = float(np.mean(d)) if d.size else 1.0
    if not np.isfinite(mu) or mu <= 0.0:
        mu = 1.0
    alpha = 1.0 / mu
    beta_noise = float(n_eff / y0cTy0c)

    gamma = np.nan
    for it in range(500):
        denom = alpha + beta_noise * d
        inv = 1.0 / np.maximum(denom, 1e-300)

        m_eig = beta_noise * inv * qt_xy0
        m2 = float(np.sum(m_eig * m_eig))
        gamma = float(np.sum((beta_noise * d) * inv))

        mTq = float(np.sum(m_eig * qt_xy0))
        mTSm = float(np.sum(d * (m_eig * m_eig)))
        rss = y0cTy0c - 2.0 * mTq + mTSm
        rss = float(max(rss, denom_floor))

        alpha_new = gamma / max(m2, denom_floor)
        beta_new = (n_eff - gamma) / rss
        beta_new = float(max(beta_new, denom_floor))

        if (
            abs(alpha_new - alpha) / max(alpha, 1e-30) < 1e-6
            and abs(beta_new - beta_noise) / max(beta_noise, 1e-30) < 1e-6
        ):
            alpha, beta_noise = alpha_new, beta_new
            break

        alpha, beta_noise = alpha_new, beta_new

        if progress and (it % 10 == 0):
            pen_dbg = alpha / beta_noise
            print(f"[evidence] it={it} pen={pen_dbg:.6g} gamma={gamma:.3g}")

    lam = float(alpha / beta_noise)
    gamma_final = float(gamma)

    pen_total = float(lam + max(float(ridge), 0.0))
    df_eff = float(np.sum(d / (d + pen_total)))

    invS = 1.0 / (d + pen_total)
    Q2 = Q * Q
    diag_inv = Q2 @ invS
    denom_factor = 1.0 - pen_total * diag_inv

    qt_XY = Q.T @ XY
    qt_XY *= invS[:, None]
    beta_all = Q @ qt_XY

    ok = good & np.isfinite(denom_factor) & (np.abs(denom_factor) > 1e-12)
    beta[ok, :] = beta_all[ok, :] / denom_factor[ok, None]

    return beta, lam, gamma_final, df_eff


def identify_independent_signals(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    pen_total: float,
    df_eff: float,
    fdr: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    S = np.asarray(S, dtype=np.float64)
    XY = np.asarray(XY, dtype=np.float64)

    n = int(S.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32), {"n_selected": 0, "steps": []}

    if XY.ndim != 2 or XY.shape[0] != n or XY.shape[1] < 2:
        raise ValueError("XY must be (n, 1+n_ctrl) with n_ctrl>0 aligned to S.")
    if not (0.0 < float(fdr) < 1.0):
        raise ValueError("fdr must be in (0,1).")

    n_ctrl = int(XY.shape[1] - 1)
    n_df = max(1, min(n, int(np.rint(df_eff))))
    R_TEST = n_df
    MAX_SIGNALS = n_df
    EIG_FLOOR = 1e-12
    DIAG_FLOOR = 1e-30

    diagS_full = np.diag(S).astype(np.float64, copy=False)
    diagS_full = np.where(diagS_full > DIAG_FLOOR, diagS_full, DIAG_FLOOR)

    remaining = np.ones(n, dtype=bool)
    leads: List[int] = []
    steps: List[Dict[str, Any]] = []

    def _pinv_sym(A: np.ndarray) -> np.ndarray:
        invA = np.linalg.pinv(A, rcond=1e-12)
        return 0.5 * (invA + invA.T)

    def _top_eigs_psd_dense(M: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        d, Q = np.linalg.eigh(0.5 * (M + M.T))
        d = np.maximum(d, 0.0)
        order = np.argsort(d)[::-1]
        d = d[order]
        Q = Q[:, order]
        keep = np.flatnonzero(d > EIG_FLOOR)
        if keep.size == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((M.shape[0], 0), dtype=np.float64)
        kk = min(int(k), int(keep.size))
        return d[:kk], Q[:, :kk]

    for step in range(MAX_SIGNALS):
        C = np.flatnonzero(remaining)
        nC = int(C.size)
        if nC == 0:
            break

        if len(leads) == 0:
            S_CC = S[np.ix_(C, C)]
            XY_C = XY[C, :]

            k = min(R_TEST, nC)
            if nC <= 2000:
                dR, QR = _top_eigs_psd_dense(S_CC, k)
            else:
                def mv0(x):
                    return S_CC @ x

                Aop0 = LinearOperator((nC, nC), matvec=mv0, dtype=np.float64)
                kk = min(k, nC - 1) if nC > 1 else 1
                vals, vecs = eigsh(Aop0, k=kk, which="LA")
                vals = np.maximum(vals, 0.0)
                order = np.argsort(vals)[::-1]
                dR, QR = vals[order], vecs[:, order]
                keep = np.flatnonzero(dR > EIG_FLOOR)
                dR, QR = dR[keep], QR[:, keep]
                if dR.size > k:
                    dR, QR = dR[:k], QR[:, :k]

            R = int(dR.size)
            if R == 0:
                steps.append({"step": step, "n_remaining": nC, "n_tested": 0, "pvals": np.array([]), "qvals": np.array([])})
                break

            U = QR.T @ XY_C
            T = (U * U) / dR[:, None]
            T0 = T[:, 0]
            Tc = T[:, 1:]

            counts = np.sum(Tc >= T0[:, None], axis=1).astype(np.int64, copy=False)
            pvals = (1.0 + counts.astype(np.float64)) / (1.0 + float(n_ctrl))

            reject, qvals, _aS, _aB = multipletests(pvals, alpha=float(fdr), method="fdr_bh")
            sel = np.flatnonzero(np.asarray(reject, dtype=bool))

            rec: Dict[str, Any] = {
                "step": step,
                "n_remaining": nC,
                "n_tested": R,
                "pvals": pvals,
                "qvals": qvals,
                "selected_components_local": sel,
                "leads_before": leads.copy(),
            }
            steps.append(rec)

            if sel.size == 0:
                break

            best = int(sel[np.lexsort((-T0[sel], pvals[sel]))][0])

            diag_res = np.diag(S_CC).astype(np.float64, copy=False)
            diag_res = np.where(diag_res > DIAG_FLOOR, diag_res, DIAG_FLOOR)

            rho = (np.sqrt(dR[best]) * QR[:, best]) / np.sqrt(diag_res)
            lead_local = int(np.argmax(np.abs(rho)))
            lead_global = int(C[lead_local])

            rec.update(
                {
                    "chosen_component_local": best,
                    "chosen_component_eig": float(dR[best]),
                    "chosen_component_p": float(pvals[best]),
                    "chosen_component_q": float(qvals[best]),
                    "lead_index": lead_global,
                    "lead_index_local_in_C": lead_local,
                }
            )

            leads.append(lead_global)
            remaining[lead_global] = False
            continue

        L = np.asarray(leads, dtype=np.int64)
        S_LL = S[np.ix_(L, L)]
        invSLL = _pinv_sym(S_LL)

        S_CL = S[np.ix_(C, L)]
        S_LC = S_CL.T
        S_CC = S[np.ix_(C, C)]
        XY_L = XY[L, :]
        XY_C = XY[C, :]

        k = min(R_TEST, nC)
        if nC <= 2000:
            A = S_CC - (S_CL @ (invSLL @ S_LC))
            dR, QR = _top_eigs_psd_dense(A, k)
        else:
            def mv(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                t = S_LC @ x
                t2 = invSLL @ t
                return (S_CC @ x) - (S_CL @ t2)

            Aop = LinearOperator((nC, nC), matvec=mv, dtype=np.float64)
            kk = min(k, nC - 1) if nC > 1 else 1
            vals, vecs = eigsh(Aop, k=kk, which="LA")
            vals = np.maximum(vals, 0.0)
            order = np.argsort(vals)[::-1]
            dR, QR = vals[order], vecs[:, order]
            keep = np.flatnonzero(dR > EIG_FLOOR)
            dR, QR = dR[keep], QR[:, keep]
            if dR.size > k:
                dR, QR = dR[:k], QR[:, :k]

        R = int(dR.size)
        if R == 0:
            steps.append({"step": step, "n_remaining": nC, "n_tested": 0, "pvals": np.array([]), "qvals": np.array([]), "leads_before": leads.copy()})
            break

        U1 = QR.T @ XY_C
        Aproj = QR.T @ S_CL
        M = invSLL @ XY_L
        U = U1 - (Aproj @ M)

        T = (U * U) / dR[:, None]
        T0 = T[:, 0]
        Tc = T[:, 1:]

        counts = np.sum(Tc >= T0[:, None], axis=1).astype(np.int64, copy=False)
        pvals = (1.0 + counts.astype(np.float64)) / (1.0 + float(n_ctrl))

        reject, qvals, _aS, _aB = multipletests(pvals, alpha=float(fdr), method="fdr_bh")
        sel = np.flatnonzero(np.asarray(reject, dtype=bool))

        rec = {
            "step": step,
            "n_remaining": nC,
            "n_tested": R,
            "pvals": pvals,
            "qvals": qvals,
            "selected_components_local": sel,
            "leads_before": leads.copy(),
        }
        steps.append(rec)

        if sel.size == 0:
            break

        best = int(sel[np.lexsort((-T0[sel], pvals[sel]))][0])

        A = invSLL @ S_LC
        diag_res = np.diag(S_CC).astype(np.float64, copy=False) - np.sum(S_CL * A.T, axis=1)
        diag_res = np.where(diag_res > DIAG_FLOOR, diag_res, DIAG_FLOOR)

        rho = (np.sqrt(dR[best]) * QR[:, best]) / np.sqrt(diag_res)
        lead_local = int(np.argmax(np.abs(rho)))
        lead_global = int(C[lead_local])

        rec.update(
            {
                "chosen_component_local": best,
                "chosen_component_eig": float(dR[best]),
                "chosen_component_p": float(pvals[best]),
                "chosen_component_q": float(qvals[best]),
                "lead_index": lead_global,
                "lead_index_local_in_C": lead_local,
            }
        )

        leads.append(lead_global)
        remaining[lead_global] = False

    out = np.full(n, -1, dtype=np.int32)
    if len(leads) == 0:
        info = {
            "procedure": "sequential_conditional_eigensignals_option3",
            "n_selected": 0,
            "lead_pos": np.array([], dtype=np.int64),
            "steps": steps,
            "n_ctrl": n_ctrl,
            "R_TEST": int(R_TEST),
            "MAX_SIGNALS": int(MAX_SIGNALS),
            "fdr": float(fdr),
            "pen_total_unused": float(pen_total),
            "df_eff_unused": float(df_eff),
        }
        return out, info

    lead_pos = np.asarray(leads, dtype=np.int64)

    denom = np.sqrt(diagS_full[:, None] * diagS_full[lead_pos][None, :])
    corr = S[:, lead_pos] / denom
    best = np.argmax(corr, axis=1).astype(np.int64, copy=False)
    out[:] = lead_pos[best].astype(np.int32, copy=False)

    for lp in lead_pos:
        out[int(lp)] = int(lp)

    info = {
        "procedure": "sequential_conditional_eigensignals_option3",
        "n_selected": int(len(leads)),
        "lead_pos": lead_pos,
        "steps": steps,
        "n_ctrl": n_ctrl,
        "R_TEST": int(R_TEST),
        "MAX_SIGNALS": int(MAX_SIGNALS),
        "fdr": float(fdr),
        "pen_total_unused": float(pen_total),
        "df_eff_unused": float(df_eff),
    }
    return out, info


def _empirical_factor_pvals(U: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.asarray(denom, dtype=np.float64), 1e-12)
    T = (U * U) / denom[:, None]
    T0 = T[:, 0]
    Tc = T[:, 1:]
    n_ctrl = Tc.shape[1]
    counts = np.sum(Tc >= T0[:, None], axis=1).astype(np.int64, copy=False)
    return (1.0 + counts.astype(np.float64)) / (1.0 + float(n_ctrl))


def _assign_by_correlation(S: np.ndarray, lead_pos: np.ndarray) -> np.ndarray:
    n = S.shape[0]
    out = np.full(n, -1, dtype=np.int32)
    if lead_pos.size == 0:
        return out
    diagS = np.maximum(np.diag(S).astype(np.float64), 1e-30)
    denom = np.sqrt(diagS[:, None] * diagS[lead_pos][None, :])
    corr = S[:, lead_pos] / denom
    best = np.argmax(corr, axis=1).astype(np.int64, copy=False)
    out[:] = lead_pos[best].astype(np.int32, copy=False)
    for lp in lead_pos:
        out[int(lp)] = int(lp)
    return out


def identify_independent_signals_wls_stepwise(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    fdr: float = 0.1,
    max_signals: int = 10,
) -> np.ndarray:
    n = S.shape[0]
    n_ctrl = XY.shape[1] - 1
    remaining = np.ones(n, dtype=bool)
    leads: List[int] = []

    for _ in range(max_signals):
        C = np.flatnonzero(remaining)
        if C.size == 0:
            break

        S_CC = S[np.ix_(C, C)]
        XY_C = XY[C, :]

        if len(leads) > 0:
            L = np.asarray(leads, dtype=np.int64)
            S_LL = S[np.ix_(L, L)]
            invSLL = np.linalg.pinv(0.5 * (S_LL + S_LL.T), rcond=1e-12)
            S_CL = S[np.ix_(C, L)]
            XY_L = XY[L, :]
            A = invSLL @ S_CL.T
            XY_C = XY_C - S_CL @ (invSLL @ XY_L)
            diag_res = np.diag(S_CC) - np.sum(S_CL * A.T, axis=1)
        else:
            diag_res = np.diag(S_CC)

        diag_res = np.maximum(diag_res.astype(np.float64), 1e-12)
        T = (XY_C * XY_C) / diag_res[:, None]
        pvals = (1.0 + np.sum(T[:, 1:] >= T[:, [0]], axis=1)) / (1.0 + float(n_ctrl))
        reject, _, _, _ = multipletests(pvals, alpha=float(fdr), method="fdr_bh")
        sel = np.flatnonzero(reject)
        if sel.size == 0:
            break

        best = int(sel[np.argmin(pvals[sel])])
        lead = int(C[best])
        leads.append(lead)
        remaining[lead] = False

    return _assign_by_correlation(S, np.asarray(leads, dtype=np.int64))


def _component_based_signal_assignment(
    S: np.ndarray,
    XY: np.ndarray,
    components: np.ndarray,
    *,
    fdr: float = 0.1,
    positive_only: bool = False,
) -> np.ndarray:
    n = S.shape[0]
    out = np.full(n, -1, dtype=np.int32)
    if components.size == 0:
        return out

    V = np.asarray(components, dtype=np.float64)
    if V.ndim != 2:
        return out
    # normalize components in S-metric
    SV = S @ V
    denom = np.sqrt(np.maximum(np.sum(V * SV, axis=0), 1e-12))
    V = V / denom[None, :]

    U = V.T @ XY
    pvals = _empirical_factor_pvals(U, np.ones(V.shape[1], dtype=np.float64))
    reject, _, _, _ = multipletests(pvals, alpha=float(fdr), method="fdr_bh")
    keep = np.flatnonzero(reject)
    if keep.size == 0:
        return out

    Vsel = V[:, keep]
    score = Vsel if positive_only else np.abs(Vsel)
    assign = np.argmax(score, axis=1).astype(np.int32)
    out[:] = assign
    return out


def identify_independent_signals_factor_analysis(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    fdr: float = 0.1,
    n_components: int = 10,
) -> np.ndarray:
    try:
        from sklearn.decomposition import FactorAnalysis
    except Exception:
        return np.full(S.shape[0], -1, dtype=np.int32)

    k = min(n_components, max(1, S.shape[0] - 1))
    X = 0.5 * (S + S.T)
    fac = FactorAnalysis(n_components=k, random_state=0)
    fac.fit(X)
    components = fac.components_.T
    return _component_based_signal_assignment(S, XY, components, fdr=fdr, positive_only=False)


def identify_independent_signals_nmf(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    fdr: float = 0.1,
    n_components: int = 10,
) -> np.ndarray:
    try:
        from sklearn.decomposition import NMF
    except Exception:
        return np.full(S.shape[0], -1, dtype=np.int32)

    k = min(n_components, max(1, S.shape[0] - 1))
    X = 0.5 * (S + S.T)
    X = X - np.min(X)
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=500)
    W = nmf.fit_transform(X)
    return _component_based_signal_assignment(S, XY, W, fdr=fdr, positive_only=True)


def identify_independent_signals_lda(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    fdr: float = 0.1,
    n_components: int = 10,
) -> np.ndarray:
    try:
        from sklearn.decomposition import LatentDirichletAllocation
    except Exception:
        return np.full(S.shape[0], -1, dtype=np.int32)

    k = min(n_components, max(1, S.shape[0] - 1))
    X = 0.5 * (S + S.T)
    X = X - np.min(X)
    X = np.maximum(X, 1e-6)
    lda = LatentDirichletAllocation(n_components=k, random_state=0, learning_method="batch", max_iter=50)
    W = lda.fit_transform(X)
    return _component_based_signal_assignment(S, XY, W, fdr=fdr, positive_only=True)


def identify_independent_signals_multi_component_per_step(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    fdr: float = 0.1,
    max_signals: int = 10,
    r_test: int = 10,
) -> np.ndarray:
    n = S.shape[0]
    remaining = np.ones(n, dtype=bool)
    leads: List[int] = []
    DIAG_FLOOR = 1e-30

    for _ in range(max_signals):
        C = np.flatnonzero(remaining)
        if C.size == 0:
            break

        S_CC = S[np.ix_(C, C)]
        XY_C = XY[C, :]
        if len(leads) > 0:
            L = np.asarray(leads, dtype=np.int64)
            S_LL = S[np.ix_(L, L)]
            invSLL = np.linalg.pinv(0.5 * (S_LL + S_LL.T), rcond=1e-12)
            S_CL = S[np.ix_(C, L)]
            XY_L = XY[L, :]
            A = S_CC - S_CL @ (invSLL @ S_CL.T)
            XY_C = XY_C - S_CL @ (invSLL @ XY_L)
        else:
            A = S_CC

        d, Q = np.linalg.eigh(0.5 * (A + A.T))
        order = np.argsort(d)[::-1]
        d = np.maximum(d[order], 0.0)
        Q = Q[:, order]
        keep = np.flatnonzero(d > 1e-12)
        if keep.size == 0:
            break
        k = min(r_test, keep.size)
        d = d[:k]
        Q = Q[:, :k]

        U = Q.T @ XY_C
        pvals = _empirical_factor_pvals(U, d)
        reject, _, _, _ = multipletests(pvals, alpha=float(fdr), method="fdr_bh")
        sel = np.flatnonzero(reject)
        if sel.size == 0:
            break

        diag_res = np.maximum(np.diag(S_CC).astype(np.float64), DIAG_FLOOR)
        new_leads = []
        for j in sel:
            rho = (np.sqrt(d[j]) * Q[:, j]) / np.sqrt(diag_res)
            lead_local = int(np.argmax(np.abs(rho)))
            lead_global = int(C[lead_local])
            if remaining[lead_global]:
                new_leads.append(lead_global)
        if not new_leads:
            break
        for lg in new_leads:
            leads.append(lg)
            remaining[lg] = False
        if len(leads) >= max_signals:
            break

    return _assign_by_correlation(S, np.asarray(leads, dtype=np.int64))


def identify_independent_signals_by_direction(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    fdr: float = 0.1,
    max_signals: int = 10,
    r_test: int = 10,
) -> np.ndarray:
    n = S.shape[0]
    remaining = np.ones(n, dtype=bool)
    directions: List[np.ndarray] = []

    for _ in range(max_signals):
        C = np.flatnonzero(remaining)
        if C.size == 0:
            break

        S_CC = S[np.ix_(C, C)]
        XY_C = XY[C, :]
        d, Q = np.linalg.eigh(0.5 * (S_CC + S_CC.T))
        order = np.argsort(d)[::-1]
        d = np.maximum(d[order], 0.0)
        Q = Q[:, order]
        keep = np.flatnonzero(d > 1e-12)
        if keep.size == 0:
            break
        k = min(r_test, keep.size)
        d = d[:k]
        Q = Q[:, :k]

        U = Q.T @ XY_C
        pvals = _empirical_factor_pvals(U, d)
        reject, _, _, _ = multipletests(pvals, alpha=float(fdr), method="fdr_bh")
        sel = np.flatnonzero(reject)
        if sel.size == 0:
            break
        best = int(sel[np.argmin(pvals[sel])])

        vec = np.zeros(n, dtype=np.float64)
        vec[C] = Q[:, best]
        directions.append(vec)

        lead_local = int(np.argmax(np.abs(Q[:, best])))
        remaining[int(C[lead_local])] = False

    out = np.full(n, -1, dtype=np.int32)
    if not directions:
        return out
    D = np.column_stack(directions)
    score = np.abs(D)
    out[:] = np.argmax(score, axis=1).astype(np.int32)
    return out


def _fit_wls_no_ridge(S: np.ndarray, XY: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    XY = np.asarray(XY, dtype=np.float64)
    invS = np.linalg.pinv(0.5 * (S + S.T), rcond=1e-12)
    return invS @ XY


def _fit_ridge_with_focal_penalty(
    S: np.ndarray,
    XY: np.ndarray,
    *,
    pen_total: float,
) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    XY = np.asarray(XY, dtype=np.float64)
    d, Q = np.linalg.eigh(0.5 * (S + S.T))
    d = np.maximum(d, 0.0)
    invS = 1.0 / np.maximum(d + float(pen_total), 1e-20)
    qt_XY = Q.T @ XY
    qt_XY *= invS[:, None]
    beta_all = Q @ qt_XY
    return beta_all


def _fit_susie_alpha_proxy(S: np.ndarray, XY: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    XY = np.asarray(XY, dtype=np.float64)
    diagS = np.maximum(np.diag(S).astype(np.float64), 1e-20)
    z = XY / np.sqrt(diagS)[:, None]
    z2 = z * z
    z2 = z2 - np.max(z2, axis=0, keepdims=True)
    exp_z2 = np.exp(z2)
    denom = np.maximum(np.sum(exp_z2, axis=0, keepdims=True), 1e-30)
    alpha = exp_z2 / denom
    return alpha


def _build_conditional_scores_table(
    *,
    groups: np.ndarray,
    cell_ids_str: List[str],
    counts: np.ndarray,
    v_raw: np.ndarray,
    mat_ctrl_raw: np.ndarray,
    v_var_ratio_c2t: np.ndarray,
    include_ctrl_score: bool,
    independent_signal_main: np.ndarray,
    independent_cols: Dict[str, np.ndarray],
) -> pd.DataFrame:
    df_scores, _, _ = build_scdrs_score_table(
        index=groups,
        v_raw_score=v_raw,
        mat_ctrl_raw_score=mat_ctrl_raw,
        v_var_ratio_c2t=v_var_ratio_c2t,
        include_ctrl_norm_scores=include_ctrl_score,
    )
    df_scores.insert(0, "cell_ids", cell_ids_str)
    df_scores.insert(1, "metacell_size", counts.astype(np.int32, copy=False))
    df_scores.insert(2, "independent_signal", independent_signal_main.astype(np.int32, copy=False))
    insert_at = 3
    for k, v in independent_cols.items():
        df_scores.insert(insert_at, k, v.astype(np.int32, copy=False))
        insert_at += 1
    return df_scores


def run_conditional_analysis_and_save(
    metacell_data,
    Z: np.ndarray,
    Z_ctrl: np.ndarray,
    weights: np.ndarray,
    v_var_ratio_c2t: np.ndarray,
    *,
    out_folder: Path,
    score_basename: str,
    cond_ridge: float,
    include_ctrl_score: bool,
    ablation: bool = False,
) -> float:
    t_cond0 = time.perf_counter()

    metacell_groups, metacell_expression, metacell_cell_ids, metacell_sizes = metacell_data
    groups = np.asarray(metacell_groups)
    metacells = np.asarray(metacell_expression)
    cell_ids_str = list(metacell_cell_ids)
    counts = np.asarray(metacell_sizes)

    sqrt_w = np.sqrt(weights.astype(np.float64, copy=False))
    ones_w = sqrt_w
    Y_all = np.column_stack([Z, Z_ctrl.T])
    Yw = Y_all * sqrt_w[:, None]

    Mw = metacells.T.astype(np.float64) * sqrt_w[:, None]

    print("[main] Building intercept-projected Gram...")
    S, XY, s11, y0cTy0c = build_projected_gram_matrices(Mw=Mw, ones_w=ones_w, Yw=Yw)
    n_eff = max(int(Mw.shape[0] - 1), 1)

    beta_pred, lam, gamma, df_eff = fit_conditional_effects_with_evidence_shrinkage(
        S=S,
        XY=XY,
        y0cTy0c=y0cTy0c,
        n_eff=n_eff,
        ridge=cond_ridge,
        progress=True,
    )
    pen_total = float(lam + max(float(cond_ridge), 0.0))
    print(f"[main] Evidence: lam={lam:.6g} gamma={gamma:.6g} df_eff(total)={df_eff:.6g} pen_total={pen_total:.6g}")

    v_raw = beta_pred[:, 0]
    mat_ctrl_raw = beta_pred[:, 1:]

    df_scores, _, _ = build_scdrs_score_table(
        index=groups,
        v_raw_score=v_raw,
        mat_ctrl_raw_score=mat_ctrl_raw,
        v_var_ratio_c2t=v_var_ratio_c2t,
        include_ctrl_norm_scores=include_ctrl_score,
    )

    n_df = max(1, min(S.shape[0], int(np.floor(df_eff))))
    print("[independent/main] Calculating multi-component-per-step independent signals...")
    independent_signal_multi = identify_independent_signals_multi_component_per_step(
        S=S,
        XY=XY,
        fdr=0.1,
        max_signals=n_df,
        r_test=n_df,
    )

    independent_cols: Dict[str, np.ndarray] = {}
    if ablation:
        independent_cols["independent_signal_wls"] = identify_independent_signals_wls_stepwise(
            S=S,
            XY=XY,
            fdr=0.1,
            max_signals=n_df,
        )
        independent_cols["independent_signal_factor"] = identify_independent_signals_factor_analysis(
            S=S,
            XY=XY,
            fdr=0.1,
            n_components=n_df,
        )
        independent_cols["independent_signal_nmf"] = identify_independent_signals_nmf(
            S=S,
            XY=XY,
            fdr=0.1,
            n_components=n_df,
        )
        independent_cols["independent_signal_lda"] = identify_independent_signals_lda(
            S=S,
            XY=XY,
            fdr=0.1,
            n_components=n_df,
        )
        independent_cols["independent_signal_direction"] = identify_independent_signals_by_direction(
            S=S,
            XY=XY,
            fdr=0.1,
            max_signals=n_df,
            r_test=n_df,
        )

    df_scores = _build_conditional_scores_table(
        groups=groups,
        cell_ids_str=cell_ids_str,
        counts=counts,
        v_raw=v_raw,
        mat_ctrl_raw=mat_ctrl_raw,
        v_var_ratio_c2t=v_var_ratio_c2t,
        include_ctrl_score=include_ctrl_score,
        independent_signal_main=independent_signal_multi,
        independent_cols=independent_cols,
    )

    ctrl_cols = [c for c in df_scores.columns if str(c).startswith("ctrl_norm_score_")]
    out_file = out_folder / f"{score_basename}.conditional.tagging_score.gz"
    if include_ctrl_score and ctrl_cols:
        keep_cols = [c for c in df_scores.columns if c not in ctrl_cols]
        df_scores.loc[:, keep_cols].to_csv(out_file, compression="gzip", sep="\t")
        out_file_full = out_folder / f"{score_basename}.conditional_score_full.gz"
        df_scores.to_csv(out_file_full, compression="gzip", sep="\t")
        print(f"[main] Saved conditional tagging scores (without ctrl columns) -> {out_file}")
        print(f"[main] Saved conditional full scores (with ctrl columns) -> {out_file_full}")
    else:
        df_scores.to_csv(out_file, compression="gzip", sep="\t")
        print(f"[main] Saved conditional tagging scores -> {out_file}")

    if ablation:
        # (1) WLS / no ridge
        beta_wls = _fit_wls_no_ridge(S, XY)
        df_wls = _build_conditional_scores_table(
            groups=groups,
            cell_ids_str=cell_ids_str,
            counts=counts,
            v_raw=beta_wls[:, 0],
            mat_ctrl_raw=beta_wls[:, 1:],
            v_var_ratio_c2t=v_var_ratio_c2t,
            include_ctrl_score=include_ctrl_score,
            independent_signal_main=independent_signal_multi,
            independent_cols=independent_cols,
        )
        out_wls = out_folder / f"{score_basename}.conditional_wls.tagging_score.gz"
        df_wls.to_csv(out_wls, compression="gzip", sep="\t")

        # (2) Ridge applies to focal as well
        beta_ridge_focal = _fit_ridge_with_focal_penalty(S, XY, pen_total=pen_total)
        df_ridge_focal = _build_conditional_scores_table(
            groups=groups,
            cell_ids_str=cell_ids_str,
            counts=counts,
            v_raw=beta_ridge_focal[:, 0],
            mat_ctrl_raw=beta_ridge_focal[:, 1:],
            v_var_ratio_c2t=v_var_ratio_c2t,
            include_ctrl_score=include_ctrl_score,
            independent_signal_main=independent_signal_multi,
            independent_cols=independent_cols,
        )
        out_ridge_focal = out_folder / f"{score_basename}.conditional_ridge_focal.tagging_score.gz"
        df_ridge_focal.to_csv(out_ridge_focal, compression="gzip", sep="\t")

        # (3) SuSiE-like alpha proxy (alpha used as test quantity)
        alpha_susie = _fit_susie_alpha_proxy(S, XY)
        df_susie = _build_conditional_scores_table(
            groups=groups,
            cell_ids_str=cell_ids_str,
            counts=counts,
            v_raw=alpha_susie[:, 0],
            mat_ctrl_raw=alpha_susie[:, 1:],
            v_var_ratio_c2t=v_var_ratio_c2t,
            include_ctrl_score=include_ctrl_score,
            independent_signal_main=independent_signal_multi,
            independent_cols=independent_cols,
        )
        out_susie = out_folder / f"{score_basename}.conditional_susie.tagging_score.gz"
        df_susie.to_csv(out_susie, compression="gzip", sep="\t")

    conditional_scoring_seconds = time.perf_counter() - t_cond0
    return conditional_scoring_seconds
