#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scdrs_fm.conditional_analysis import run_conditional_analysis_and_save
from scdrs_fm.data_processing import (
    apply_covariate_correction,
    compute_metacells,
    load_and_basic_process,
    run_imputation,
    scdrs_preprocess,
    aggregate_expression_by_metacell,
)
from scdrs_fm.gene_sets import build_gene_set_and_controls, compute_v_var_ratio_c2t
from scdrs_fm.marginal_analysis import (
    run_marginal_analysis,
    save_marginal_results,
    save_marginal_results_split_ctrl,
)


def normalize_optional_path(s: str) -> Optional[str]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "" or ss.lower() in {"none", "null", "-", "na"}:
        return None
    return ss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_scdrs_fm.py",
        description="Metacell conditional tagging + marginal scDRS-like scores (+ sequential conditional independent signals).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("h5ad_file", help="Input .h5ad")
    p.add_argument("cov_file", help="Covariate file (tsv). Use '-' or 'none' to disable.")
    p.add_argument("out", help="Output directory")
    p.add_argument("gs_dir", help="Directory containing gs_split files")
    p.add_argument("traits", nargs="+", help="Trait gene set filenames within gs_dir")

    p.add_argument("--h5ad_species", default="human", choices=["human", "mouse"], help="Species of h5ad expression")
    p.add_argument("--flag_raw_count", action="store_true", help="If set: normalize_per_cell + log1p before scoring")
    p.add_argument("--flag_filter", action="store_true", help="If set: filter cells/genes before scoring")
    p.add_argument(
        "--imputation",
        default="magic",
        choices=["magic", "none", "scvi", "dca", "alra"],
        help="Imputation applied after metacell assignment",
    )
    p.add_argument(
        "--include_ctrl_score",
        action="store_true",
        help="If set: include ctrl_norm_score_* columns in output tables",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="If set: run ablation outputs (extra independent-signal methods + regression variants).",
    )
    return p.parse_args()


def run_pipeline(
    h5ad_file: str,
    cov_file: Optional[str],
    out_dir: str,
    gs_dir: str,
    traits: List[str],
    *,
    h5ad_species: str = "human",
    flag_raw_count: bool = False,
    flag_filter: bool = False,
    imputation: str = "magic",
    include_ctrl_score: bool = False,
    ablation: bool = False,
    cond_ridge: float = 1e-10,
) -> None:
    out_folder = Path(out_dir)
    out_folder.mkdir(parents=True, exist_ok=True)

    print("[main] Args:")
    print(f"  h5ad_file          = {h5ad_file}")
    print(f"  cov_file           = {cov_file}")
    print(f"  out                = {out_folder}")
    print(f"  gs_dir             = {gs_dir}")
    print(f"  traits             = {traits}")
    print(f"  h5ad_species       = {h5ad_species}")
    print(f"  flag_raw_count     = {flag_raw_count}")
    print(f"  flag_filter        = {flag_filter}")
    print(f"  imputation         = {imputation}")
    print(f"  include_ctrl_score = {include_ctrl_score}")
    print(f"  ablation           = {ablation}")

    t_total0 = time.perf_counter()
    timings_global: Dict[str, float] = {}

    t0 = time.perf_counter()
    adata = load_and_basic_process(
        h5ad_file,
        flag_filter=flag_filter,
        flag_raw_count=flag_raw_count,
    )
    timings_global["load_and_basic_process_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_gene, df_cov = scdrs_preprocess(adata, cov_file=cov_file)
    timings_global["scdrs_preprocess_s"] = time.perf_counter() - t0

    timings_global["metacell_assign_s"] = compute_metacells(adata)
    timings_global["imputation_s"] = run_imputation(adata, imputation=imputation)
    timings_global["covariate_correction_s"] = apply_covariate_correction(adata, df_cov)

    t0 = time.perf_counter()
    metacell_data = aggregate_expression_by_metacell(adata)
    timings_global["metacell_aggregation_s"] = time.perf_counter() - t0

    per_trait: List[Dict[str, Any]] = []

    for trait in traits:
        print("\n" + "=" * 80)
        print(f"[main] TRAIT: {trait}")
        print("=" * 80)

        trait_rec: Dict[str, Any] = {"trait": trait}
        t_trait0 = time.perf_counter()

        gs_file = str(Path(gs_dir) / trait)
        score_basename = Path(trait).name

        t0 = time.perf_counter()
        Z, Z_ctrl, weights = build_gene_set_and_controls(
            adata,
            df_gene,
            gs_file=gs_file,
            h5ad_species=h5ad_species,
        )
        trait_rec["build_gene_set_and_controls_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        v_var_ratio_c2t = compute_v_var_ratio_c2t(df_gene, Z, Z_ctrl)
        trait_rec["compute_v_var_ratio_c2t_s"] = time.perf_counter() - t0

        df_marginal, marginal_seconds = run_marginal_analysis(
            adata=adata,
            Z=Z,
            Z_ctrl=Z_ctrl,
            weights=weights,
            v_var_ratio_c2t=v_var_ratio_c2t,
            include_ctrl_score=include_ctrl_score,
        )
        trait_rec["marginal_scoring_s"] = marginal_seconds

        if "metacell" in adata.obs:
            df_marginal["metacell"] = adata.obs.loc[df_marginal.index, "metacell"].to_numpy()

        if include_ctrl_score:
            trait_rec["save_marginal_s"] = save_marginal_results_split_ctrl(
                df_marginal,
                out_folder=out_folder,
                score_basename=score_basename,
            )
        else:
            trait_rec["save_marginal_s"] = save_marginal_results(
                df_marginal,
                out_folder=out_folder,
                score_basename=score_basename,
            )

        conditional_seconds = run_conditional_analysis_and_save(
            metacell_data=metacell_data,
            Z=Z,
            Z_ctrl=Z_ctrl,
            weights=weights,
            v_var_ratio_c2t=v_var_ratio_c2t,
            out_folder=out_folder,
            score_basename=score_basename,
            cond_ridge=cond_ridge,
            include_ctrl_score=include_ctrl_score,
            ablation=ablation,
        )
        trait_rec["conditional_scoring_s"] = conditional_seconds

        trait_rec["trait_total_s"] = time.perf_counter() - t_trait0
        per_trait.append(trait_rec)

    total_seconds = time.perf_counter() - t_total0

    print("\n================ GLOBAL TIMINGS (one-time) ================")
    print(f"Load + basic process:      {timings_global['load_and_basic_process_s']:,.3f} s")
    print(f"scDRS preprocess:          {timings_global['scdrs_preprocess_s']:,.3f} s")
    print(f"Metacell assign:           {timings_global['metacell_assign_s']:,.3f} s")
    print(f"Imputation:                {timings_global['imputation_s']:,.3f} s")
    print(f"Covariate correction:      {timings_global['covariate_correction_s']:,.3f} s")
    print(f"Metacell aggregation:      {timings_global['metacell_aggregation_s']:,.3f} s")
    print("-----------------------------------------------------------")

    print("\n================ PER-TRAIT TIMINGS ================")
    if per_trait:
        header = ("trait", "build_gs+ctrl", "var_ratio", "marginal", "save_marginal", "conditional", "trait_total")
        print(
            f"{header[0]:<40}  {header[1]:>10}  {header[2]:>9}  {header[3]:>9}  "
            f"{header[4]:>12}  {header[5]:>11}  {header[6]:>11}"
        )
        for r in per_trait:
            print(
                f"{r['trait']:<40}  "
                f"{r.get('build_gene_set_and_controls_s', 0.0):>10.3f}  "
                f"{r.get('compute_v_var_ratio_c2t_s', 0.0):>9.3f}  "
                f"{r.get('marginal_scoring_s', 0.0):>9.3f}  "
                f"{r.get('save_marginal_s', 0.0):>12.3f}  "
                f"{r.get('conditional_scoring_s', 0.0):>11.3f}  "
                f"{r.get('trait_total_s', 0.0):>11.3f}"
            )
    else:
        print("(no traits)")

    print("---------------------------------------------------")
    print(f"TOTAL (everything): {total_seconds:,.3f} s")
    print("===================================================\n")


def main() -> None:
    args = parse_args()
    run_pipeline(
        h5ad_file=args.h5ad_file,
        cov_file=normalize_optional_path(args.cov_file),
        out_dir=args.out,
        gs_dir=args.gs_dir,
        traits=args.traits,
        h5ad_species=args.h5ad_species,
        flag_raw_count=bool(args.flag_raw_count),
        flag_filter=bool(args.flag_filter),
        imputation=args.imputation,
        include_ctrl_score=bool(args.include_ctrl_score),
        ablation=bool(args.ablation),
    )


if __name__ == "__main__":
    main()
