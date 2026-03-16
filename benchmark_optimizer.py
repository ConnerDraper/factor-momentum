"""
Benchmark sf-quant optimizer refactor: old (full N×N covariance) vs new (factored).

Loads pre-computed alphas, picks a reference date, subsamples N barrids,
and times both code paths. Outputs a console table and a log-log plot.

Usage:
    nohup uv run python benchmark_optimizer.py > benchmark.log 2>&1 &
"""

import time
import datetime as dt

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# New (refactored) code — imported from current sf-quant
import sf_quant.backtester as sfb
import sf_quant.optimizer as sfo

# Old (v0.1.23 baseline) code — inlined in _baseline.py
from _baseline import backtest_sequential_old

# ── Configuration ───────────────────────────────────────────────────
ALPHAS_PATH = "results/full/alphas/factor_momentum_style.parquet"
GAMMA = 50
CONSTRAINTS = [sfo.constraints.ZeroBeta(), sfo.constraints.ZeroInvestment()]
N_VALUES = [100, 200, 400, 800, 1600, 3200]  # N_max appended at runtime
N_REPEATS = 3
SEED = 42
OUTPUT_PLOT = "results/benchmark_optimizer.png"
OUTPUT_TABLE = "results/benchmark_results.txt"


def pick_reference_date(df: pl.DataFrame) -> dt.date:
    """Pick the date with the most assets as the reference date."""
    counts = df.group_by("date").len().sort("len", descending=True)
    best = counts.row(0)
    print(f"Reference date: {best[0]} ({best[1]:,} assets)")
    return best[0]


def subsample_data(df: pl.DataFrame, barrids: list[str], date_: dt.date) -> pl.DataFrame:
    """Filter to a single date and a specific set of barrids."""
    return df.filter(
        (pl.col("date") == date_) & pl.col("barrid").is_in(barrids)
    )


def time_backtest(fn, data, constraints, gamma, n_repeats):
    """Time a backtest function over n_repeats, return list of elapsed times."""
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(data=data, constraints=constraints, gamma=gamma)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, result


def set_academic_style():
    """Configure matplotlib for a clean, professional academic style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_results_table(results, output_path):
    """Save a formatted results table to a text file."""
    lines = []
    lines.append("Optimizer Benchmark Results")
    lines.append("Old (v0.1.23, full Σ) vs New (factored B·F·B'+D)")
    lines.append("")
    
    header = f"{'N':>6}  {'Old (s)':>10}  {'New (s)':>10}  {'Speedup':>8}  {'Match':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for r in results:
        match_str = "Yes" if r["weights_match"] else "No"
        lines.append(
            f"{r['n']:>6}  {r['old_median']:>10.4f}  {r['new_median']:>10.4f}  "
            f"{r['speedup']:>7.2f}x  {match_str:>6}"
        )
    
    lines.append("-" * len(header))
    lines.append("")

    # Summary stats
    max_speedup = max(r["speedup"] for r in results)
    min_speedup = min(r["speedup"] for r in results)
    all_match = all(r["weights_match"] for r in results)
    lines.append(f"Speedup range: {min_speedup:.2f}x – {max_speedup:.2f}x")
    lines.append(f"All weights match: {'Yes' if all_match else 'No'}")
    lines.append(f"Repeats per (version, N): {N_REPEATS}")
    
    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text + "\n")
    print(f"Table saved to {output_path}")


def main():
    print("=" * 70)
    print("SF-QUANT OPTIMIZER BENCHMARK")
    print("Old (v0.1.23, full Σ) vs New (optimizer-refactor, factored B·F·B'+D)")
    print("=" * 70)

    # Load pre-computed alphas
    print(f"\nLoading alphas from {ALPHAS_PATH}...")
    alphas_df = pl.read_parquet(ALPHAS_PATH)
    print(f"  {alphas_df.height:,} rows, {alphas_df.width} columns")
    print(f"  Columns: {alphas_df.columns}")

    # Pick reference date
    ref_date = pick_reference_date(alphas_df)

    # Get all available barrids on the reference date
    ref_data = alphas_df.filter(pl.col("date") == ref_date)
    all_barrids = sorted(ref_data["barrid"].unique().to_list())
    n_max = len(all_barrids)
    print(f"  N_max on reference date: {n_max:,}")

    # Build N values (add N_max)
    n_values = [n for n in N_VALUES if n < n_max] + [n_max]
    print(f"\nN values to test: {n_values}")
    print(f"Repeats per (version, N): {N_REPEATS}")

    # Results storage
    results = []
    rng = np.random.default_rng(SEED)

    print(f"\n{'N':>6} | {'Old (s)':>10} | {'New (s)':>10} | {'Speedup':>8} | {'Match':>5}")
    print("-" * 55)

    for n in n_values:
        # Subsample barrids
        if n >= n_max:
            sample_barrids = all_barrids
        else:
            indices = rng.choice(n_max, size=n, replace=False)
            sample_barrids = [all_barrids[i] for i in sorted(indices)]

        data_slice = subsample_data(ref_data, sample_barrids, ref_date)

        # Time OLD backtest
        old_times, old_result = time_backtest(
            backtest_sequential_old, data_slice, CONSTRAINTS, GAMMA, N_REPEATS
        )

        # Time NEW backtest
        new_times, new_result = time_backtest(
            sfb.backtest_sequential, data_slice, CONSTRAINTS, GAMMA, N_REPEATS
        )

        # Compare weights
        old_w = old_result.sort("barrid")["weight"].to_numpy()
        new_w = new_result.sort("barrid")["weight"].to_numpy()
        weights_match = np.allclose(old_w, new_w, atol=1e-6)

        old_median = np.median(old_times)
        new_median = np.median(new_times)
        speedup = old_median / new_median if new_median > 0 else float("inf")

        results.append({
            "n": n,
            "old_times": old_times,
            "new_times": new_times,
            "old_median": old_median,
            "new_median": new_median,
            "speedup": speedup,
            "weights_match": weights_match,
        })

        match_str = "✓" if weights_match else "✗"
        print(f"{n:>6} | {old_median:>10.4f} | {new_median:>10.4f} | {speedup:>7.2f}x | {match_str:>5}")

    # ── Save table ──────────────────────────────────────────────────
    save_results_table(results, OUTPUT_TABLE)

    # ── Plot ────────────────────────────────────────────────────────
    set_academic_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = [r["n"] for r in results]
    old_medians = [r["old_median"] for r in results]
    new_medians = [r["new_median"] for r in results]
    old_mins = [min(r["old_times"]) for r in results]
    old_maxs = [max(r["old_times"]) for r in results]
    new_mins = [min(r["new_times"]) for r in results]
    new_maxs = [max(r["new_times"]) for r in results]

    # Error bars: [median - min, max - median]
    old_err = [
        [m - lo for m, lo in zip(old_medians, old_mins)],
        [hi - m for m, hi in zip(old_medians, old_maxs)],
    ]
    new_err = [
        [m - lo for m, lo in zip(new_medians, new_mins)],
        [hi - m for m, hi in zip(new_medians, new_maxs)],
    ]

    ax.errorbar(ns, old_medians, yerr=old_err, fmt="o-", capsize=4,
                color="#c0392b", linewidth=1.5, markersize=6, label="v0.1.23 (full Σ)")
    ax.errorbar(ns, new_medians, yerr=new_err, fmt="s-", capsize=4,
                color="#2874a6", linewidth=1.5, markersize=6, label="Refactored (B·F·B′+D)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of assets)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Optimizer Benchmark: Full Covariance vs Factor Decomposition", loc="left")

    # Clean tick labels — explicit values instead of log-scale gridlines
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    # Light horizontal grid only (y-axis major ticks)
    ax.yaxis.grid(True, alpha=0.15, linewidth=0.6)
    ax.xaxis.grid(False)

    ax.legend(frameon=False, fontsize=10)

    # Annotate speedups
    for r in results:
        ax.annotate(
            f"{r['speedup']:.1f}×",
            xy=(r["n"], r["new_median"]),
            xytext=(0, -16),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color="#2874a6",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"Plot saved to {OUTPUT_PLOT}")
    print("Done.")


if __name__ == "__main__":
    main()
