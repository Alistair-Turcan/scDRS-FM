#!/usr/bin/env python3
from __future__ import annotations

import argparse

from scdrs_fm.decompose_gradients import run_decomposition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_decompose_gradients.py",
        description="Run phenotype-gradient decomposition across marg-marg, marg-cond, and cond-cond combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--traits", nargs="+", required=True)
    p.add_argument("--trait_dir", required=True)
    p.add_argument("--pheno_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--adata", required=True)
    p.add_argument("--phenotypes", nargs="+", required=True)
    p.add_argument("--additional_phenotypes", nargs="*", default=[])
    p.add_argument("--cell_type_col", default="")
    p.add_argument("--cell_type", default="")
    p.add_argument("--n_controls", type=int, default=1000)
    p.add_argument("--include_negative_phenotypes", action="store_true")
    p.add_argument("--univ_fdr_thresh", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_decomposition(
        adata_file=args.adata,
        trait_dir=args.trait_dir,
        pheno_dir=args.pheno_dir,
        out_dir=args.out_dir,
        traits=args.traits,
        phenotypes=args.phenotypes,
        additional_phenotypes=args.additional_phenotypes,
        cell_type_col=args.cell_type_col,
        cell_type=args.cell_type,
        n_controls=int(args.n_controls),
        include_negative_phenotypes=bool(args.include_negative_phenotypes),
        univ_fdr_thresh=float(args.univ_fdr_thresh),
    )


if __name__ == "__main__":
    main()
