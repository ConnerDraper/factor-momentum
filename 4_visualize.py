"""
Step 4: Visualize cumulative returns for a single backtest.
Use after step 3 to inspect the optimal lambda result on test data.

Usage:
    python 4_visualize.py --split test --lambda 0.015873
    python 4_visualize.py --split train --lambda 0.015873
"""
import argparse
import os

import polars as pl
import sf_quant.data as sfd
import matplotlib.pyplot as plt
import seaborn as sns

from config import SPLITS, GAMMA, signal_name, weights_dir, split_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize backtest results")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--lambda", dest="lamb", type=float, required=True,
                        help="Lambda value to visualize")
    args = parser.parse_args()

    split = SPLITS[args.split]
    hl = int(0.693 / args.lamb)
    sig = signal_name(args.lamb)

    # Load weights
    w_path = f"{weights_dir(args.split, args.lamb)}/*.parquet"
    print(f"Loading weights for λ={args.lamb:.6f} (HL={hl}d)...")
    weights = pl.read_parquet(w_path)

    # Load asset returns
    print("Loading asset returns...")
    returns = (
        sfd.load_assets(
            start=split["start"], end=split["end"],
            columns=["date", "barrid", "return"],
            in_universe=True,
        )
        .sort("date", "barrid")
        .select(
            "date", "barrid",
            pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
        )
    )

    # Compute portfolio returns
    portfolio_returns = (
        weights.join(returns, on=["date", "barrid"], how="left")
        .group_by("date")
        .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
        .sort("date")
    )

    # Summary statistics
    mean_ret = portfolio_returns["return"].mean() * 252 * 100
    vol = portfolio_returns["return"].std() * (252 ** 0.5) * 100
    sharpe = mean_ret / vol if vol > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  λ = {args.lamb:.6f}  (half-life: {hl} days)")
    print(f"  Split: {args.split}")
    print(f"  Mean Return:  {mean_ret:.2f}%")
    print(f"  Volatility:   {vol:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"{'='*50}")

    # Cumulative returns chart
    cumulative = portfolio_returns.select(
        "date",
        pl.col("return").log1p().cum_sum().mul(100).alias("cumulative_return"),
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=cumulative.to_pandas(), x="date", y="cumulative_return")
    plt.title(f"Factor Momentum — {args.split} set (λ={args.lamb:.6f}, HL={hl}d, γ={GAMMA})",
              fontsize=13, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("Cumulative Log Returns (%)")
    plt.grid(True, alpha=0.3)

    out_path = f"{split_dir(args.split)}/{sig}_chart.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
