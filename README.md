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

The key hyperparameter is **λ** (the EWMA decay rate), which we optimize via grid search over half-lives from 1 to 12 months.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_compute_alphas.py` | Compute & save stock-level alpha parquets for each λ |
| 2 | `2_backtest.py` | Submit MVO backtest jobs to Slurm |
| 3 | `3_evaluate_lambdas.py` | Plot mean return & Sharpe vs. λ, pick optimal λ |
| 4 | `4_visualize.py` | Cumulative returns chart for a single backtest |

Shared logic lives in `pipeline.py`; dates, parameters, and paths are in `config.py`.

## Usage

```bash
# Train: find optimal lambda (1995–2010)
python 1_compute_alphas.py --split train
python 2_backtest.py --split train
# ⏳ wait for Slurm
python 3_evaluate_lambdas.py --split train

# Test: validate with optimal lambda (2010–2025)
python 1_compute_alphas.py --split test --lambda <optimal>
python 2_backtest.py --split test --lambda <optimal>
# ⏳ wait for Slurm
python 4_visualize.py --split test --lambda <optimal>
```

## Output Structure

```
results/
├── train/
│   ├── alphas/            # alpha parquets per lambda
│   ├── weights/           # MVO weights from Slurm
│   └── lambda_performance.png
└── test/
    ├── alphas/
    ├── weights/
    └── *_chart.png
```