"""
Step 3: Construct the dynamic optimal factor momentum portfolio.
Run this AFTER all Slurm jobs from step 2 have completed.

Usage:
    python 3_backtest_optimal.py --split train
    python 3_backtest_optimal.py --split test
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
import sf_quant.data as sfd

from config import SPLITS, LAMBDA_GRID, GAMMA, TRAILING_PERF_SPAN_DAYS, signal_name, weights_dir, split_dir


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


def main():
    parser = argparse.ArgumentParser(description="Construct dynamic optimal lambda portfolio")
    parser.add_argument("--split", required=True, choices=list(SPLITS.keys()))
    args = parser.parse_args()

    split = SPLITS[args.split]
    
    # Setup logging
    log_path = f"{split_dir(args.split)}/dynamic_optimal_report.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = Tee(log_path)

    print(f"Constructing dynamic optimal portfolio for {args.split} split...")
    
    # 1. Load asset returns
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

    # 2. Load weights and compute daily portfolio returns for each lambda
    lambda_daily_returns = []
    lambda_weights_dict = {}
    valid_lambdas = []

    for lamb in LAMBDA_GRID:
        w_dir = weights_dir(args.split, lamb)
        weights_path = f"{w_dir}/*.parquet"

        try:
            weights = pl.read_parquet(weights_path)
            valid_lambdas.append(lamb)
        except Exception:
            continue

        lambda_weights_dict[lamb] = weights

        portfolio_returns = (
            weights.join(returns, on=["date", "barrid"], how="left")
            .group_by("date")
            .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
            .sort("date")
            .with_columns(pl.lit(lamb).alias("lambda"))
            .select("date", "lambda", "return")
        )
        lambda_daily_returns.append(portfolio_returns)
        
    if not valid_lambdas:
        print("Error: No lambda weights found. Run Step 2 first.")
        return

    df_returns = pl.concat(lambda_daily_returns)

    df_returns_wide = df_returns.pivot(
        on="lambda",
        index="date",
        values="return"
    ).sort("date")
    
    sorted_lambdas = sorted(valid_lambdas)
    str_lambdas = [str(lamb) for lamb in sorted_lambdas]
    
    # 3. Calculate Trailing Performance (1xK Vector)
    span = TRAILING_PERF_SPAN_DAYS 
    
    df_shifted = df_returns_wide.with_columns(
        [pl.col(str_lamb).shift(1) for str_lamb in str_lambdas]
    )
    
    pd_shifted = df_shifted.to_pandas().set_index("date")[str_lambdas]
    
    ewma_mean = pd_shifted.ewm(span=span, min_periods=252).mean()
    ewma_vol = pd_shifted.ewm(span=span, min_periods=252).std().replace(0, np.nan)
    ewma_perf = ewma_mean / ewma_vol
    
    # 4. Identify Optimal Lambda Neighborhood and 5. Construct Dynamic Portfolio Weights
    dates = df_returns_wide["date"].to_list()
    daily_optimal_lambdas = []
    
    warmup_days = 0
    for date in dates:
        if date not in ewma_perf.index or np.isnan(ewma_perf.loc[date].iloc[0]):
            warmup_days += 1
            median_idx = len(sorted_lambdas) // 2
            daily_optimal_lambdas.append({
                "date": date,
                "winner_1": sorted_lambdas[median_idx-1],
                "winner_2": sorted_lambdas[median_idx],
                "winner_3": sorted_lambdas[median_idx+1],
                "center_lambda": sorted_lambdas[median_idx]
            })
            continue
            
        perf_vector = ewma_perf.loc[date].values
        
        best_avg = -np.inf
        best_center_idx = 1
        
        for i in range(1, len(perf_vector) - 1):
            avg_perf = (perf_vector[i-1] + perf_vector[i] + perf_vector[i+1]) / 3.0
            if avg_perf > best_avg:
                best_avg = avg_perf
                best_center_idx = i
                
        daily_optimal_lambdas.append({
            "date": date,
            "winner_1": sorted_lambdas[best_center_idx - 1],
            "winner_2": sorted_lambdas[best_center_idx],
            "winner_3": sorted_lambdas[best_center_idx + 1],
            "center_lambda": sorted_lambdas[best_center_idx]
        })
        
    print(f"Info: Defaulted to median static lambda for {warmup_days} warmup days.")
        
    df_opt_lambdas = pl.DataFrame(daily_optimal_lambdas)
    
    dfs_to_join = []
    base_weights = lambda_weights_dict[sorted_lambdas[0]].select("date", "barrid")
    
    wide_weights = base_weights
    for lamb in sorted_lambdas:
        w = lambda_weights_dict[lamb].rename({"weight": f"weight_{lamb}"})
        wide_weights = wide_weights.join(w, on=["date", "barrid"], how="left")
        
    wide_weights = wide_weights.join(df_opt_lambdas, on="date", how="left")
    
    exprs = []
    combined_weight = pl.lit(0.0)
    
    for i in range(1, len(sorted_lambdas) - 1):
        center_lamb = sorted_lambdas[i]
        w1_col = pl.col(f"weight_{sorted_lambdas[i-1]}")
        w2_col = pl.col(f"weight_{sorted_lambdas[i]}")
        w3_col = pl.col(f"weight_{sorted_lambdas[i+1]}")
        
        avg_w = (w1_col.fill_null(0.0) + w2_col.fill_null(0.0) + w3_col.fill_null(0.0)) / 3.0
        combined_weight = pl.when(pl.col("center_lambda") == center_lamb).then(avg_w).otherwise(combined_weight)
        
    final_weights = wide_weights.with_columns(
        combined_weight.alias("weight")
    ).select("date", "barrid", "weight")
    
    # 6. Save
    out_sig = "dynamic_optimal"
    out_dir = f"{split_dir(args.split)}/weights/{out_sig}/{GAMMA}"
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = f"{out_dir}/weights.parquet"
    final_weights.write_parquet(out_path)
    print(f"✓ Dynamic weights saved to {out_path}")
    
    traj_path = f"{split_dir(args.split)}/dynamic_optimal_trajectory.parquet"
    df_opt_lambdas.write_parquet(traj_path)
    
    print("\nDone. You can now use 4_visualize.py to compare.")
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
