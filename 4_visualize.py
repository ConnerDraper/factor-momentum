"""
Step 4: Visualize cumulative returns for a single backtest.
Use after step 3 to inspect the optimal lambda result on test data.

Usage:
    python 4_visualize.py --split test --lambda 0.015873
    python 4_visualize.py --split train --lambda 0.015873
"""
import argparse
import os
import sys

import polars as pl
import sf_quant.data as sfd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import SPLITS, GAMMA, signal_name, weights_dir, split_dir


class Tee:
    """Write output to both stdout and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_academic_style():
    """Configure matplotlib for a clean, professional academic style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
    })


def main():
    parser = argparse.ArgumentParser(description="Visualize backtest results")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--lambda", dest="lamb", type=float, required=True,
                        help="Lambda value to visualize")
    args = parser.parse_args()

    split = SPLITS[args.split]
    hl = int(0.693 / args.lamb)
    sig = signal_name(args.lamb)
    
    # Setup logging
    log_path = f"{split_dir(args.split)}/backtest_report_{sig}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Tee(log_path)

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

    print("\nBacktest Summary")
    print("-" * 30)
    print(f"Signal:       {sig}")
    print(f"Split:        {args.split} ({split['start']} to {split['end']})")
    print(f"Lambda:       {args.lamb:.6f} (HL: {hl} days)")
    print("-" * 30)
    print(f"Mean Return:  {mean_ret:.2f}%")
    print(f"Volatility:   {vol:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print("-" * 30)
    
    # Restore stdout (close logging)
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to {log_path}")

    # Cumulative returns chart
    cumulative = portfolio_returns.select(
        "date",
        pl.col("return").log1p().cum_sum().mul(100).alias("cumulative_return"),
    ).to_pandas()

    set_academic_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Main plot
    ax.plot(cumulative["date"], cumulative["cumulative_return"], 
            color="#003366", linewidth=1.2, label="Factor Momentum")
    ax.fill_between(cumulative["date"], cumulative["cumulative_return"], 0, 
                    color="#003366", alpha=0.08)
    
    # Reference line
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    # Titles & Labels
    ax.set_title(f"Cumulative Return — {args.split.capitalize()} Set (λ={args.lamb:.6f})", loc="left")
    ax.set_ylabel("Cumulative Return (%)")
    
    # Stats Box
    stats_text = (
        f"Sharpe: {sharpe:.2f}\n"
        f"Return: {mean_ret:.1f}%\n"
        f"Vol:    {vol:.1f}%"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    # Formatting
    ax.yaxis.grid(True)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(8))  # Limit x-axis ticks

    out_path = f"{split_dir(args.split)}/{sig}_chart.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()
