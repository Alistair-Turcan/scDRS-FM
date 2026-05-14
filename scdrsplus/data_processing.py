from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import scdrs.pp as pp

try:
    import magic
    import scprep
except Exception:
    magic = None
    scprep = None


def ensure_csr_inplace(adata: sc.AnnData) -> sc.AnnData:
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
    else:
        adata.X = sp.csr_matrix(adata.X)
    return adata


def cast_x_float32_inplace(adata: sc.AnnData) -> sc.AnnData:
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)
        adata.X.eliminate_zeros()
    else:
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32, copy=False)
    return adata


def compute_leiden_metacells_avg_band(
    adata: sc.AnnData,
    avg_size_low: float = 9.0,
    avg_size_high: float = 11.0,
    what: str = "__x__",
    random_seed: int = 0,
    n_neighbors: int = 30,
    n_pcs: int = 50,
    hvg_n_top_genes: int = 2000,
    start_resolution: float = 100.0,
    res_min: float = 1.0,
    res_max: float = 1000.0,
    max_search_iter: int = 100,
    min_metacell_size: int = 5,
    scale_band_over_n: int = 10_000,
):
    n_cells = int(adata.n_obs)
    if n_cells == 0:
        raise ValueError("adata has 0 cells")
    if not (avg_size_low > 0 and avg_size_high >= avg_size_low):
        raise ValueError("Require 0 < avg_size_low <= avg_size_high")

    if n_cells > int(scale_band_over_n):
        scale = n_cells / float(scale_band_over_n)
        avg_size_low = float(avg_size_low) * scale
        avg_size_high = float(avg_size_high) * scale
        print(
            f"[leiden-metacells] Scaling target avg-size band by {scale:.3g} "
            f"(n_cells={n_cells:,} > {scale_band_over_n:,}) -> "
            f"[{avg_size_low:.2f}, {avg_size_high:.2f}]"
        )
    else:
        print(
            f"[leiden-metacells] Target avg-size band: [{avg_size_low:.2f}, {avg_size_high:.2f}] "
            f"(n_cells={n_cells:,})"
        )

    layer = None if what == "__x__" else what

    if "highly_variable" not in adata.var or int(adata.var["highly_variable"].sum()) < 50:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=hvg_n_top_genes,
            flavor="seurat",
            layer=layer,
            inplace=True,
        )

    if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < n_pcs:
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            use_highly_variable=True,
            svd_solver="randomized",
            random_state=random_seed,
            layer=layer,
        )

    if "connectivities" not in adata.obsp:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep="X_pca",
        )

    conn = adata.obsp["connectivities"]

    use_leidenalg = True
    try:
        import leidenalg as la
        from scanpy._utils import get_igraph_from_adjacency

        g = get_igraph_from_adjacency(conn, directed=False)

        def run_leiden(res: float) -> np.ndarray:
            part = la.find_partition(
                g,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=float(res),
                seed=int(random_seed),
            )
            return np.asarray(part.membership, dtype=np.int32)
    except Exception:
        use_leidenalg = False

        def run_leiden(res: float) -> np.ndarray:
            key = "__leiden_tmp__"
            sc.tl.leiden(adata, resolution=float(res), key_added=key, random_state=random_seed)
            labs = adata.obs[key].to_numpy()
            del adata.obs[key]
            codes, _ = pd.factorize(labs, sort=True)
            return codes.astype(np.int32, copy=False)

    cache: Dict[float, Tuple[float, int, np.ndarray]] = {}

    def eval_res(res: float):
        res = float(res)
        if res in cache:
            return cache[res]
        labs = run_leiden(res)
        k = int(np.unique(labs).size)
        mean_size = n_cells / max(1, k)
        cache[res] = (mean_size, k, labs)
        return cache[res]

    def in_band(mean_size: float) -> bool:
        return (mean_size >= avg_size_low) and (mean_size <= avg_size_high)

    res0 = float(np.clip(start_resolution, res_min, res_max))
    mean0, _, _ = eval_res(res0)

    if in_band(mean0):
        best_res = res0
    else:
        lo = hi = res0
        mean_lo = mean_hi = mean0

        if mean0 > avg_size_high:
            hi = lo
            mean_hi = mean0
            while mean_hi > avg_size_high and hi < res_max:
                lo = hi
                mean_lo = mean_hi
                hi = min(res_max, hi * 2.0)
                mean_hi, _, _ = eval_res(hi)
                if in_band(mean_hi):
                    break
        else:
            lo = hi
            mean_lo = mean0
            while mean_lo < avg_size_low and lo > res_min:
                hi = lo
                mean_hi = mean_lo
                lo = max(res_min, lo / 2.0)
                mean_lo, _, _ = eval_res(lo)
                if in_band(mean_lo):
                    break

        band_candidates = [(r, cache[r][0]) for r in cache.keys() if in_band(cache[r][0])]
        if band_candidates:
            mid = 0.5 * (avg_size_low + avg_size_high)
            best_res = min(band_candidates, key=lambda x: abs(x[1] - mid))[0]
        else:
            mid_target = 0.5 * (avg_size_low + avg_size_high)
            lo, hi = (min(lo, hi), max(lo, hi))

            best_res = None
            best_score = np.inf

            def score(mean_size: float) -> float:
                if mean_size < avg_size_low:
                    return avg_size_low - mean_size
                if mean_size > avg_size_high:
                    return mean_size - avg_size_high
                return abs(mean_size - mid_target)

            for r, (ms, _, _) in cache.items():
                sc_ = score(ms)
                if sc_ < best_score:
                    best_score = sc_
                    best_res = r

            for _ in range(max_search_iter):
                mid_res = 0.5 * (lo + hi)
                ms, _, _ = eval_res(mid_res)

                sc_ = score(ms)
                if sc_ < best_score:
                    best_score = sc_
                    best_res = mid_res

                if in_band(ms):
                    best_res = mid_res
                    break

                if ms > avg_size_high:
                    lo = mid_res
                elif ms < avg_size_low:
                    hi = mid_res

                if (hi - lo) < 1e-6:
                    break

    mean_best, k_best, labels_best = eval_res(float(best_res))

    sizes = np.bincount(labels_best, minlength=int(labels_best.max()) + 1)
    metacell = labels_best.copy()
    small = np.where(sizes < min_metacell_size)[0]
    if small.size:
        for c in small:
            metacell[metacell == c] = -1

    keep = metacell >= 0
    if np.any(keep):
        new_codes, _ = pd.factorize(metacell[keep], sort=True)
        out = np.full(n_cells, -1, dtype=np.int32)
        out[keep] = new_codes.astype(np.int32, copy=False)
        metacell = out

    n_outliers = int(np.sum(metacell < 0))
    n_metacells = int(np.unique(metacell[metacell >= 0]).size)

    print(
        f"[leiden-metacells] solver={'leidenalg' if use_leidenalg else 'scanpy'} "
        f"resolution={float(best_res):.6g} clusters={k_best:,} mean_size={mean_best:.2f} "
        f"-> metacells={n_metacells:,} outliers={n_outliers:,}"
    )

    adata.obs["metacell"] = metacell
    return n_metacells, float(best_res), float(mean_best)


def load_and_basic_process(
    h5ad_file: str,
    *,
    flag_filter: bool,
    flag_raw_count: bool,
) -> sc.AnnData:
    print("[main] Loading:", h5ad_file)
    adata = sc.read_h5ad(h5ad_file)
    ensure_csr_inplace(adata)

    if flag_filter:
        print("[main] Filtering cells/genes...")
        sc.pp.filter_cells(adata, min_genes=250)
        sc.pp.filter_genes(adata, min_cells=50)

    if flag_raw_count:
        print("[main] Normalizing + log1p...")
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)

    return adata


def scdrs_preprocess(
    adata: sc.AnnData,
    *,
    cov_file: Optional[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    print("[main] scDRS preprocess...")
    df_cov = pd.read_csv(cov_file, sep="\t", index_col=0) if cov_file else None
    pp.preprocess(adata, cov=df_cov, n_mean_bin=20, n_var_bin=20, copy=False)
    print("[main] Preprocessed.")

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[adata.var_names].copy()
    df_gene["gene"] = df_gene.index
    df_gene = df_gene.drop_duplicates(subset="gene")
    df_gene = df_gene.loc[adata.var_names]
    return df_gene, df_cov


def compute_metacells(adata: sc.AnnData) -> float:
    t0 = time.perf_counter()
    print("[main] Preparing UN-IMPUTED X for metacells (cast to float32)...")
    cast_x_float32_inplace(adata)

    print("[main] Computing metacells (leiden)...")
    n_metacells, best_res, _ = compute_leiden_metacells_avg_band(adata)
    print(f"[main] Created {n_metacells} metacells (resolution={best_res})")
    return time.perf_counter() - t0


def run_imputation(adata: sc.AnnData, *, imputation: Optional[str]) -> float:
    t0 = time.perf_counter()
    if imputation is None:
        return 0.0

    imp = imputation.lower()
    if imp in ("none", "null", "no", "off"):
        return 0.0

    if imp == "magic":
        if magic is None or scprep is None:
            raise ImportError("Packages 'magic' and 'scprep' are required for imputation='magic'.")
        print("[main] MAGIC imputation...")
        X = adata.X
        if sp.issparse(X):
            X = X.tocsr()
        X = scprep.normalize.library_size_normalize(X)
        magic_op = magic.MAGIC(n_jobs=-1, t="auto", random_state=0, verbose=1)
        adata.X = magic_op.fit_transform(X, genes="all_genes")
        print("[main] MAGIC done.")
    else:
        raise ValueError(f"Unknown imputation={imputation} (expected magic|none)")

    return time.perf_counter() - t0


def apply_covariate_correction(adata: sc.AnnData, df_cov: Optional[pd.DataFrame]) -> float:
    t0 = time.perf_counter()
    if df_cov is None:
        return 0.0

    print("[main] Applying covariate correction...")
    cell_list = list(adata.obs_names)
    gene_list_all = list(adata.var_names)
    cov_list = list(adata.uns["SCDRS_PARAM"]["COV_MAT"].columns)

    cov_mat = adata.uns["SCDRS_PARAM"]["COV_MAT"].loc[cell_list, cov_list].values
    cov_beta = adata.uns["SCDRS_PARAM"]["COV_BETA"].loc[gene_list_all, cov_list].values.T
    gene_mean = adata.uns["SCDRS_PARAM"]["COV_GENE_MEAN"].loc[gene_list_all].values

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.X = X + cov_mat @ cov_beta + gene_mean
    print("[main] Covariate correction done.")
    return time.perf_counter() - t0


def aggregate_expression_by_metacell(adata: sc.AnnData):
    """Aggregate cell-level expression into metacell averages once before per-trait runs."""
    if "metacell" not in adata.obs:
        raise RuntimeError("adata.obs['metacell'] not found. Did metacell assignment run?")

    print("[main] Aggregating expression into metacells (dropping metacell<0 outliers)...")
    metacell_ids = np.asarray(adata.obs["metacell"].to_numpy())
    valid = metacell_ids >= 0
    if not np.any(valid):
        raise RuntimeError("No valid metacells found (all metacell IDs < 0).")

    groups = np.unique(metacell_ids[valid])
    groups = np.sort(groups)
    id_to_row = {mid: i for i, mid in enumerate(groups)}
    group_idx = np.fromiter((id_to_row[mid] for mid in metacell_ids[valid]), dtype=np.int64)

    n_metacells = len(groups)
    n_valid = int(valid.sum())

    rows = group_idx
    cols = np.arange(n_valid, dtype=np.int64)
    data = np.ones(n_valid, dtype=np.float64)
    selector = sp.csr_matrix((data, (rows, cols)), shape=(n_metacells, n_valid))

    X = adata.X
    if sp.issparse(X):
        X_valid = X[valid].tocsr()
        sums = (selector @ X_valid).toarray()
    else:
        X_valid = np.asarray(X[valid], dtype=np.float64)
        sums = selector @ X_valid

    counts = np.asarray(selector.sum(axis=1)).ravel()
    metacell_expression = sums / counts[:, None]

    obs_valid = np.asarray(adata.obs_names[valid])
    cell_ids = [""] * n_metacells
    for i in range(n_metacells):
        cell_ids[i] = ",".join(obs_valid[rows == i])

    print(
        f"[main] Aggregated metacells: {metacell_expression.shape[0]:,} x {metacell_expression.shape[1]:,} "
        f"(min={int(counts.min())}, median={int(np.median(counts))}, max={int(counts.max())})"
    )

    return groups, metacell_expression, cell_ids, counts
