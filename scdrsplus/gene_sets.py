from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

import scdrs.util as util
from scdrs.method import _select_ctrl_geneset


def read_gs(path: str) -> list[str]:
    with open(path, "r") as f:
        _ = f.readline()
        genes = f.readline().split("\t")[1].split(",")
    return [g.strip() for g in genes if g.strip()]


def genes_to_weight_dict(genes: list[str]) -> dict[str, float]:
    outd: dict[str, float] = {}
    for g in genes:
        parts = g.split(":")
        gene = parts[0]
        w = float(parts[1]) if len(parts) == 2 else 1.0
        outd[gene] = w
    return outd


def filter_genes(gene_dict: dict[str, float], adata: sc.AnnData) -> dict[str, float]:
    keep = set(adata.var_names)
    return {g: w for g, w in gene_dict.items() if g in keep}


def iter_ctrl_sets(dic_ctrl_list, dic_ctrl_weight) -> Iterator[Tuple[list[str], list[float]]]:
    if isinstance(dic_ctrl_list, dict):
        for k in dic_ctrl_list.keys():
            yield dic_ctrl_list[k], dic_ctrl_weight[k]
    else:
        for genes, w in zip(dic_ctrl_list, dic_ctrl_weight):
            yield genes, w


def build_gene_set_and_controls(
    adata: sc.AnnData,
    df_gene: pd.DataFrame,
    *,
    gs_file: str,
    h5ad_species: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("[main] Reading gene set:", gs_file)
    gene_dict = genes_to_weight_dict(read_gs(gs_file))

    if h5ad_species == "mouse":
        print("[main] Mapping human -> mouse homologs...")
        gene_dict_map = util.load_homolog_mapping("human", "mouse")
        gene_dict = {gene_dict_map.get(g, g): w for g, w in gene_dict.items()}

    gene_dict = filter_genes(gene_dict, adata)
    print(f"[main] Gene set after filtering to adata.var_names: {len(gene_dict)} genes")

    gene_list = list(gene_dict.keys())
    gene_weight = list(gene_dict.values())

    print("[main] Selecting control gene sets (n_ctrl=1000)...")
    dic_ctrl_list, dic_ctrl_weight = _select_ctrl_geneset(
        df_gene, gene_list, gene_weight, "mean_var", 1000, 20, 0
    )
    n_ctrl_sets = len(dic_ctrl_list) if not isinstance(dic_ctrl_list, dict) else len(dic_ctrl_list.keys())
    print(f"[main] Control sets: {n_ctrl_sets}")

    var_names = adata.var_names
    var_pos = {g: i for i, g in enumerate(var_names)}

    Z = np.zeros(len(var_names), dtype=np.float64)
    for g, w in gene_dict.items():
        Z[var_pos[g]] = float(w)

    Z_ctrl = np.zeros((n_ctrl_sets, len(var_names)), dtype=np.float64)
    for k, (ctrl_genes, ctrl_w) in enumerate(iter_ctrl_sets(dic_ctrl_list, dic_ctrl_weight)):
        for g, wgt in zip(ctrl_genes, ctrl_w):
            j = var_pos.get(g)
            if j is not None:
                Z_ctrl[k, j] = float(wgt)

    weights = 1.0 / np.sqrt(df_gene["var_tech"].values.astype(np.float64, copy=False) + 1e-2)
    return Z, Z_ctrl, weights


def compute_v_var_ratio_c2t(df_gene: pd.DataFrame, Z: np.ndarray, Z_ctrl: np.ndarray) -> np.ndarray:
    var = df_gene["var"].values.astype(np.float64, copy=False)
    var_tech = df_gene["var_tech"].values.astype(np.float64, copy=False)

    idx = np.flatnonzero(Z != 0.0)
    if idx.size == 0:
        raise ValueError("Disease gene set is empty after filtering.")
    base = 1.0 / np.sqrt(var_tech[idx] + 1e-2)
    w_d = base * Z[idx]
    w_d = w_d / w_d.sum()
    denom = (var[idx] * (w_d**2)).sum()

    n_ctrl = Z_ctrl.shape[0]
    out = np.ones(n_ctrl, dtype=np.float64)

    for i in range(n_ctrl):
        idxc = np.flatnonzero(Z_ctrl[i] != 0.0)
        basec = 1.0 / np.sqrt(var_tech[idxc] + 1e-2)
        w_c = basec * Z_ctrl[i, idxc]
        w_c = w_c / w_c.sum()
        out[i] = (var[idxc] * (w_c**2)).sum() / denom

    return out
