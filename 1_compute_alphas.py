"""
Step 1: Compute stock-level alpha parquets for each lambda value.

Usage:
    python 1_compute_alphas.py --split train
    python 1_compute_alphas.py --split test --lambda 0.015873  # single lambda for test
"""
import argparse
import os

from config import SPLITS, LAMBDA_GRID, alphas_path
from pipeline import LambdaOptimizer


def main():
    parser = argparse.ArgumentParser(description="Compute alphas for each lambda")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--lambda", dest="lamb", type=float, default=None,
                        help="Single lambda value (default: use full grid)")
    args = parser.parse_args()

    split = SPLITS[args.split]
    lambdas = [args.lamb] if args.lamb else LAMBDA_GRID
    optimizer = LambdaOptimizer()

    print(f"Split: {args.split} ({split['start']} to {split['end']})")
    print(f"Computing alphas for {len(lambdas)} lambda value(s)...\n")

    # Load factor returns (small — wide format, ~3700 rows)
    returns = optimizer.load_factor_returns(split["start"], split["end"])

    for lamb in lambdas:
        hl = int(0.693 / lamb)
        print(f"  λ={lamb:.6f} (half-life {hl}d)...", flush=True)

        factor_alphas = optimizer.compute_alphas(returns, lamb)
        filtered = optimizer.map_to_assets(factor_alphas, split["start"], split["end"])

        out = alphas_path(args.split, lamb)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        filtered.write_parquet(out)

        print(f"    ✓ {filtered.height:,} rows → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
