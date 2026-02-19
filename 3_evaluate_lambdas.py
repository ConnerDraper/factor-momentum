"""
Step 3: Load backtest results, compute performance metrics, and plot lambda comparison.
Run this AFTER all Slurm jobs from step 2 have completed.

Usage:
    python 3_evaluate_lambdas.py --split train
"""
import argparse
import os
import sys

import polars as pl
import sf_quant.data as sfd
import matplotlib.pyplot as plt

from config import SPLITS, LAMBDA_GRID, GAMMA, signal_name, weights_dir, split_dir


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
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
    })


def main():
    parser = argparse.ArgumentParser(description="Evaluate lambda performance")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    args = parser.parse_args()

    split = SPLITS[args.split]
    
    # Setup logging
    log_path = f"{split_dir(args.split)}/lambda_evaluation_report.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Tee(log_path)

    # Load asset returns (once)
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

    # Evaluate each lambda
    results_data = []

    print("\nLambda Performance Summary:")
    print(f"{'Lambda':<10} {'Half-Life':<12} {'Mean Ret %':<12} {'Vol %':<10} {'Sharpe':<8}")
    print("-" * 55)

    for lamb in LAMBDA_GRID:
        hl = int(0.693 / lamb)
        w_dir = weights_dir(args.split, lamb)
        weights_path = f"{w_dir}/*.parquet"

        try:
            weights = pl.read_parquet(weights_path)
        except Exception:
            no_data = f"{lamb:.6f}".ljust(10) + f"{hl}d".ljust(12) + "NO DATA"
            print(no_data)
            continue

        portfolio_returns = (
            weights.join(returns, on=["date", "barrid"], how="left")
            .group_by("date")
            .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
            .sort("date")
        )

        mean_ret = portfolio_returns["return"].mean() * 252 * 100
        vol = portfolio_returns["return"].std() * (252 ** 0.5) * 100
        sharpe = mean_ret / vol if vol > 0 else 0.0

        print(f"{lamb:<10.6f} {hl:>3d}d {'':<7} {mean_ret:>6.2f}      {vol:>5.2f}     {sharpe:>5.3f}")

        results_data.append({
            "lambda": lamb,
            "half_life_days": hl,
            "mean_return": mean_ret,
            "volatility": vol,
            "sharpe": sharpe,
        })

    if not results_data:
        print("\nNo results found. Make sure Slurm jobs have completed.")
        return

    df = pl.DataFrame(results_data).sort("half_life_days")

    # Find optimal lambda
    best = df.filter(pl.col("sharpe") == pl.col("sharpe").max())
    opt_lambda = best["lambda"][0]
    opt_hl = best["half_life_days"][0]
    opt_sharpe = best["sharpe"][0]

    print("-" * 55)
    print(f"Optimal:   λ={opt_lambda:.6f} (HL={opt_hl}d)")
    print(f"Sharpe:    {opt_sharpe:.3f}")
    
    # Restore stdout (close logging)
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to {log_path}")

    # Plot
    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    half_lives = df["half_life_days"].to_list()
    hl_labels = [f"{hl}d" for hl in half_lives]

    # Plot 1: Mean Return
    ax1.plot(half_lives, df["mean_return"].to_list(), "o-", color="#004488", label="Mean Return")
    ax1.axvline(opt_hl, color="#444444", linestyle=":", alpha=0.6, label=f"Opt (HL={opt_hl}d)")
    ax1.set_xlabel("Half-Life (Days)")
    ax1.set_ylabel("Annualized Mean Return (%)")
    ax1.set_title("Return vs. Half-Life")
    ax1.set_xticks(half_lives)
    ax1.set_xticklabels(hl_labels, rotation=45)
    ax1.yaxis.grid(True)
    
    # Plot 2: Sharpe Ratio
    ax2.plot(half_lives, df["sharpe"].to_list(), "o-", color="#006644", label="Sharpe Ratio")
    ax2.axvline(opt_hl, color="#444444", linestyle=":", alpha=0.6, label=f"Opt (HL={opt_hl}d)")
    ax2.set_xlabel("Half-Life (Days)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio vs. Half-Life")
    ax2.set_xticks(half_lives)
    ax2.set_xticklabels(hl_labels, rotation=45)
    ax2.yaxis.grid(True)

    plt.tight_layout()

    out_path = f"{split_dir(args.split)}/lambda_performance.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()
