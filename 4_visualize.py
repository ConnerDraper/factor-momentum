"""
Step 4: Visualize cumulative returns for backtests.
Plots all valid static lambdas and the new dynamic optimal portfolio,
and shows a trajectory of the selected lambda over time.

Usage:
    python 4_visualize.py --split test
    python 4_visualize.py --split train
"""
import argparse
import os
import sys

import polars as pl
import numpy as np
import sf_quant.data as sfd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
    })


def main():
    parser = argparse.ArgumentParser(description="Visualize backtest results")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    args = parser.parse_args()

    split = SPLITS[args.split]
    
    log_path = f"{split_dir(args.split)}/backtest_report_all.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Tee(log_path)

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

    portfolios = []
    
    for lamb in LAMBDA_GRID:
        w_path = f"{weights_dir(args.split, lamb)}/*.parquet"
        try:
            weights = pl.read_parquet(w_path)
            hl_days = int(0.693 / lamb)
            label = f"HL={hl_days}d"
        except Exception:
            continue
            
        pf_ret = (
            weights.join(returns, on=["date", "barrid"], how="left")
            .group_by("date")
            .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
            .sort("date")
            .with_columns(pl.lit(label).alias("name"))
        )
        portfolios.append(pf_ret)
        
    dyn_path = f"{split_dir(args.split)}/weights/dynamic_optimal/{GAMMA}/*.parquet"
    try:
        dyn_weights = pl.read_parquet(dyn_path)
        dyn_ret = (
            dyn_weights.join(returns, on=["date", "barrid"], how="left")
            .group_by("date")
            .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
            .sort("date")
            .with_columns(pl.lit("Dynamic Optimal").alias("name"))
        )
        portfolios.append(dyn_ret)
        has_dynamic = True
    except Exception:
        print("Warning: dynamic_optimal weights not found.")
        has_dynamic = False

    if not portfolios:
        print("No weights found.")
        return

    all_returns = pl.concat(portfolios)
    
    print("\nBacktest Summary")
    print("-" * 50)
    print(f"{'Portfolio':<18} {'Mean Ret %':<12} {'Vol %':<10} {'Sharpe':<8}")
    print("-" * 50)
    
    summary_stats = []
    cumulative_dfs = []
    
    for name, df in all_returns.group_by("name"):
        df = df.sort("date").fill_null(0.0)
        
        mean_ret = df["return"].mean() * 252 * 100
        vol = df["return"].std() * (252 ** 0.5) * 100
        sharpe = mean_ret / vol if vol > 0 else 0.0
        
        print(f"{name[0]:<18} {mean_ret:>6.2f}      {vol:>5.2f}     {sharpe:>5.3f}")
        
        cum_ret = df.select(
            "date",
            pl.col("return").log1p().cum_sum().mul(100).alias("cumulative"),
            pl.col("return").alias("raw_return"),
            pl.lit(name[0]).alias("name")
        )
        
        cum_ret = cum_ret.with_columns(
            pl.col("cumulative").cum_max().alias("cum_max")
        ).with_columns(
            (pl.col("cumulative") - pl.col("cum_max")).alias("drawdown")
        )
        
        cumulative_dfs.append(cum_ret)
        summary_stats.append({"name": name[0], "sharpe": sharpe})
        
    print("-" * 50)
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to {log_path}")

    set_academic_style()
    
    traj_path = f"{split_dir(args.split)}/dynamic_optimal_trajectory.parquet"
    if has_dynamic and os.path.exists(traj_path):
        fig, axes = plt.subplots(4, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        ax1, ax2, ax3, ax4 = axes
        has_traj = True
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        has_traj = False

    cum_all = pl.concat(cumulative_dfs).to_pandas()
    
    for name, group in cum_all.groupby("name"):
        if name == "Dynamic Optimal":
            ax1.plot(group["date"], group["cumulative"], 
                    color="#003366", linewidth=2.0, label="Dynamic Optimal", zorder=10)
        else:
            ax1.plot(group["date"], group["cumulative"], 
                    linewidth=1.0, alpha=0.4, label=name)
    
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_title(f"Cumulative Return — {args.split.capitalize()} Set", loc="left")
    ax1.set_ylabel("Cumulative Log Return (%)")
    ax1.yaxis.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    if has_traj:
        traj_df = pl.read_parquet(traj_path)
        traj_df = traj_df.with_columns(
            (0.693 / pl.col("center_lambda")).cast(pl.Int32).alias("center_hl"),
            (0.693 / pl.col("winner_1")).cast(pl.Int32).alias("w1_hl"),
            (0.693 / pl.col("winner_3")).cast(pl.Int32).alias("w3_hl"),
        ).to_pandas().dropna()
        
        ax2.fill_between(traj_df["date"], traj_df["w1_hl"], traj_df["w3_hl"], 
                         color="#003366", alpha=0.2, label="Adjacent Winners Window")
        
        ax2.plot(traj_df["date"], traj_df["center_hl"], 
                 color="#003366", linewidth=1.5, label="Center Target HL")
                 
        ax2.set_title("Optimal Lambda Trajectory (3-Bar Smoothing)", loc="left")
        ax2.set_ylabel("Half-Life (Days)")
        ax2.set_yscale('log')
        ax2.yaxis.grid(True, which="both", alpha=0.3)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        # Plot 3: Drawdown
        for name, group in cum_all.groupby("name"):
            if name == "Dynamic Optimal":
                ax3.fill_between(group["date"], group["drawdown"], 0,
                                 color="#d9534f", alpha=0.9, label="Dynamic Optimal", zorder=10)
            else:
                ax3.plot(group["date"], group["drawdown"], 
                         linewidth=0.8, alpha=0.3, label=name)
                         
        ax3.set_title("Drawdown", loc="left")
        ax3.set_ylabel("Drawdown (%)")
        ax3.yaxis.grid(True)
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        # Plot 4: Rolling IC (Rank Corr between weights and forward return for the dynamic optimal)
        if has_dynamic:
            try:
                # Group by date to compute the spearman rank correlation 
                # This could be intensive, but simple approx using pearson on raw returns is faster here
                # Or we compute a rolling 252 return
                # Instead of raw IC which requires cross-sectional asset data, we'll plot rolling 1-year Sharpe
                dyn_group = cum_all[cum_all["name"] == "Dynamic Optimal"].copy()
                dyn_group.set_index("date", inplace=True)
                
                # Rolling 252-day annualized Sharpe Ratio
                rolling_mean = dyn_group["raw_return"].rolling(252).mean() * 252 * 100
                rolling_vol = dyn_group["raw_return"].rolling(252).std() * np.sqrt(252) * 100
                rolling_sharpe = rolling_mean / rolling_vol
                
                ax4.plot(dyn_group.index, rolling_sharpe, color="#5bc0de", linewidth=1.5)
                ax4.axhline(0, color="black", linewidth=0.8, alpha=0.5)
                ax4.axhline(rolling_sharpe.mean(), color="#003366", linestyle="--", linewidth=1.0, label="Mean Sharpe")
                
                ax4.set_title("Rolling 1-Year Sharpe Ratio (Dynamic Optimal)", loc="left")
                ax4.set_ylabel("Sharpe Ratio")
                ax4.yaxis.grid(True)
                ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            except Exception as e:
                print(f"Failed to plot rolling Sharpe: {e}")

    plt.tight_layout()
    out_path = f"{split_dir(args.split)}/backtest_performance_all.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()

