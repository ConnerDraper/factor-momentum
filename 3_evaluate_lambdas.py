"""
Step 3: Load backtest results, compute performance metrics, and plot lambda comparison.
Run this AFTER all Slurm jobs from step 2 have completed.

Usage:
    python 3_evaluate_lambdas.py --split train
"""
import argparse
import os

import polars as pl
import sf_quant.data as sfd
import matplotlib.pyplot as plt

from config import SPLITS, LAMBDA_GRID, GAMMA, signal_name, weights_dir, split_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate lambda performance")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    args = parser.parse_args()

    split = SPLITS[args.split]

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

    for lamb in LAMBDA_GRID:
        hl = int(0.693 / lamb)
        w_dir = weights_dir(args.split, lamb)
        weights_path = f"{w_dir}/*.parquet"

        try:
            weights = pl.read_parquet(weights_path)
        except Exception:
            print(f"  ✗ λ={lamb:.6f} (HL={hl:>4d}d): no weights found, skipping")
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

        print(f"  ✓ λ={lamb:.6f} (HL={hl:>4d}d)  "
              f"Mean={mean_ret:>7.2f}%  Vol={vol:>6.2f}%  Sharpe={sharpe:>5.3f}")

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

    print(f"\n{'='*60}")
    print(f"Optimal lambda: {opt_lambda:.6f} (half-life: {opt_hl} days)")
    print(f"Sharpe ratio:   {opt_sharpe:.3f}")
    print(f"{'='*60}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    half_lives = df["half_life_days"].to_list()
    hl_labels = [f"{hl}d" for hl in half_lives]

    ax1.plot(half_lives, df["mean_return"].to_list(), "o-", color="#2563eb", linewidth=2, markersize=8)
    ax1.axvline(opt_hl, color="#dc2626", linestyle="--", alpha=0.5, label=f"Optimal (HL={opt_hl}d)")
    ax1.set_xlabel("Half-Life (trading days)")
    ax1.set_ylabel("Annualized Mean Return (%)")
    ax1.set_title("Mean Return vs. Lambda")
    ax1.set_xticks(half_lives)
    ax1.set_xticklabels(hl_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(half_lives, df["sharpe"].to_list(), "o-", color="#059669", linewidth=2, markersize=8)
    ax2.axvline(opt_hl, color="#dc2626", linestyle="--", alpha=0.5, label=f"Optimal (HL={opt_hl}d)")
    ax2.set_xlabel("Half-Life (trading days)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe Ratio vs. Lambda")
    ax2.set_xticks(half_lives)
    ax2.set_xticklabels(hl_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Factor Momentum: Lambda Grid Search — {args.split} set (γ={GAMMA})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = f"{split_dir(args.split)}/lambda_performance.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
