# Factor Momentum

Backtesting an EWMA-based factor momentum strategy using Barra factor returns.

## Strategy

For each factor *i*, we compute a vol-adjusted momentum signal:

```
signal_i = EWMA(r_i) / sqrt(EWMA(r_i²))
```

Signals are z-scored cross-sectionally, then converted to stock-level alphas via factor exposures:

```
alpha_stock = Σ (exposure_i × IC × risk_{i,t-1} × score_{i,t-1})
```

Instead of using a static hyperparameter **λ** (the EWMA decay rate), we dynamically optimize it over time using a 3-year EWMA of the trailing portfolio Sharpe/return to continually weight the most performant adjacent 3 lambda portfolios.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_compute_alphas.py` | Compute & save stock-level alpha parquets for each λ across the grid |
| 2 | `2_backtest.py` | Submit MVO backtest jobs to Slurm for each static λ |
| 3 | `3_backtest_optimal.py` | Determine optimal λ trajectory via 3-yr EWMA and compute dynamic weights |
| 4 | `4_visualize.py` | Plot cumulative returns for all λs + dynamic optimal portfolio and trajectory |

Shared logic lives in `pipeline.py`. 

Dates, variables like `TRAILING_PERF_SPAN_DAYS` (3-year EWMA window), risk configurations like `GAMMA` and `IC`, and the testing `LAMBDA_GRID` are consolidated in `config.py`.

## Usage

```bash
# 1. Compute Alphas (Full Split: 1995–2025)
srun --mem=32G --cpus-per-task=8 --time=01:00:00 \
    uv run python 1_compute_alphas.py --split full

# 2. Submit Jobs and compute static weights
# 💡 Tip: Run this step inside `tmux` so you can close your SSH connection while Slurm jobs run!
> tmux
> uv run python 2_backtest.py --split full --clean
> # Press Ctrl+B, then D to detach
# ⏳ wait for Slurm

# 3. Compute the dynamic optimal target
uv run python 3_backtest_optimal.py --split full

# 4. View results & trajectory
uv run python 4_visualize.py --split full
```

## Output Structure

```
results/
└── full/
    ├── alphas/            
    ├── weights/           # Slurm MVO static weights + dynamic_optimal/
    └── backtest_performance_all.png
```