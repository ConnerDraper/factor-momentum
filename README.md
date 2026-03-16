# Factor Momentum

Backtesting a Top-Down to Bottom-Up factor momentum strategy using Barra factor returns.

## Strategy

For each factor *i* in a predefined group (Style, Industry, All), we convert simple returns to log returns and compute a vol-adjusted momentum signal over an 11-to-1 month lookback window (231 trading days, skip 21):

```
signal_i = RollingSum(log(1 + r_i)) / RollingStd(log(1 + r_i))
```

Signals are z-scored cross-sectionally to produce factor-level z-scores. These are then mapped to asset-level z-scores via Barra factor exposures:

```
z_asset = Σ (exposure_i × z_factor_i)
```

Asset-level z-scores are converted to alphas using Grinold's fundamental law with stock-specific (idiosyncratic) risk:

```
alpha_asset = IC × σ_idio × z_asset
```

By separating Style and Industry factors, we can independently investigate how Style Momentum vs Industry Momentum (and All-Factor Momentum) perform over the backtest.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_compute_alphas.py` | Compute & save stock-level alpha parquets for each factor group |
| 2 | `2_backtest.py` | Submit MVO backtest jobs to Slurm for each static factor group |
| 3 | `3_visualize.py` | Plot cumulative returns, drawdown, and rolling Sharpe for factor groups |

Shared logic lives in `pipeline.py`. 

Dates, variables like `WINDOW_LENGTH` and `SKIP_LENGTH` for the momentum signal, risk configurations like `GAMMA` and `IC`, and the defined `STYLE_FACTORS` and `INDUSTRY_FACTORS` are consolidated in `config.py`.

## Usage

```bash
# 1. Compute Alphas (Full Split: 1995–2025)
# 💡 Tip: Run this step using `nohup` since predicting alphas over 30 years takes a while.
nohup uv run python 1_compute_alphas.py --split full > compute.log 2>&1 &
# ⏳ wait for alphas to finish generating

# 2. Submit Jobs and compute static weights
# 💡 Tip: Run this step using `nohup` so the job submission loop safely runs
# in the background. If you run it via standard shell or tmux, detaching or 
# losing SSH connection may kill the python script and block jobs from submitting.
nohup uv run python 2_backtest.py --split full --clean > submission.log 2>&1 &
# ⏳ wait for Slurm (you can safely close your terminal)

# 3. View results
uv run python 3_visualize.py --split full
```

## Output Structure

```
results/
└── full/
    ├── alphas/            
    ├── weights/           # Slurm MVO static weights
    └── backtest_performance_all.png
```