"""
Step 1: Compute stock-level alpha parquets for each lambda value.

Usage:
    python 1_compute_alphas.py --split train
    python 1_compute_alphas.py --split test --lambda 0.015873  # single lambda for test
"""
import argparse
import os

import polars as pl
from config import SPLITS, FACTOR_GROUPS, alphas_path
from pipeline import FactorOptimizer


def main():
    parser = argparse.ArgumentParser(description="Compute alphas for each factor group")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    parser.add_argument("--group", dest="group", type=str, default=None,
                        help="Single factor group (default: use all groups)")
    args = parser.parse_args()

    split = SPLITS[args.split]
    groups = [args.group] if args.group else FACTOR_GROUPS
    optimizer = FactorOptimizer()

    print(f"Split: {args.split} ({split['start']} to {split['end']})")
    print(f"Computing alphas for {len(groups)} factor group(s)...\n")

    # Load factor returns (small — wide format, ~3700 rows)
    returns = optimizer.load_factor_returns(split["start"], split["end"])

    for group in groups:
        print(f"  Group={group}...", flush=True)

        factor_alphas = optimizer.compute_alphas(returns, group)
        
        # We need to filter back the dataframe because `returns` is padded now 
        factor_alphas = factor_alphas.filter(
            (pl.col("date") >= split["start"]) & (pl.col("date") <= split["end"])
        )

        filtered = optimizer.map_to_assets(factor_alphas, split["start"], split["end"])

        out = alphas_path(args.split, group)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        filtered.write_parquet(out)

        print(f"    ✓ {filtered.height:,} rows → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
