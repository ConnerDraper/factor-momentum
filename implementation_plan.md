# Factor Momentum Strategy Update

This plan outlines the steps to change the factor momentum strategy from the current cross-sectional alpha-decay (EWMA based on lambda) to a t-12 to t-2 factor momentum strategy, adjusted by factor volatility. It also details the separation of style and industry factors in the backtest.

## User Review Required
We need to finalize the exact definition of "t-12 to t-2" depending on the data frequency (e.g., if daily, this would mean trailing 252 days to 21 days). We also need to see how style and industry factors are identified in the current dataset since BARRA is not available.

## Proposed Changes

### Configuration ([config.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/config.py))
- Remove `LAMBDA_GRID` and [LambdaOptimizer](file:///Users/connerdraper/Work/Silverfund/factor-momentum/pipeline.py#11-157) references.
- Add lists for `STYLE_FACTORS` and `INDUSTRY_FACTORS`. Since actual data isn't present locally, we will create placeholder lists that you can flesh out when you run on the cluster (e.g. typical BARRA names).
- Note: Daily frequency is assumed based on old configuration. We'll define `WINDOW_LENGTH = 231` (11 months) and `SKIP_LENGTH = 21` (1 month) to capture the t-12 to t-2 month window. 

### Pipeline Logic ([pipeline.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/pipeline.py))
- Rename [LambdaOptimizer](file:///Users/connerdraper/Work/Silverfund/factor-momentum/pipeline.py#11-157) to `FactorOptimizer`.
- Implementing the **Top-Down to Bottom-Up** mapping workflow for calculating alphas:

  **1. Signal (Factor Momentum Scores):**
  - Calculate Vol-adjusted returns for the Barra factors based on `factor_group` ("All", "Style", or "Industry").
  - To filter factors: Check columns of loaded returns against lists defined in [config.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/config.py) (`STYLE_FACTORS`, `INDUSTRY_FACTORS`). Any column not matching the requested group is ignored.
  - Calculate momentum ($$S_k$$): Replace EWMA with a rolling window for t-12 to t-2 momentum and volatility:
    - Return from t-12 to t-2: `pl.col(c).rolling_sum(window_size=WINDOW_LENGTH).shift(SKIP_LENGTH)`
    - Factor Volatility ($$\sigma_k$$): `pl.col(c).rolling_std(window_size=WINDOW_LENGTH).shift(SKIP_LENGTH)`
    - $$S_k = \text{Momentum} / \sigma_k$$

  **2. Z-Score:** Cross-sectionally Z-score the $$S_k$$ values for the selected factors.
  
  **3. Factor Alpha:** Apply $$IC_k \cdot \sigma_k \cdot S_k$$ to compute the vector of factor alphas ($$\alpha_{factor}$$).
  
  **4. Projection (Map to Stocks):** Multiply the Barra Exposure Matrix ($$X$$) by the calculated factor alphas. The total alpha for any stock $$i$$ is $$\alpha_{stock, i} = \sum_{k} X_{ik} \cdot \alpha_{factor, k}$$. This step remains similar to the current implementation, but now explicitly operates only on the selected Style or Industry subsets.

### Scripts
- [1_compute_alphas.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/1_compute_alphas.py): Iterate over `factor_group` in `["Style", "Industry", "All"]` instead of different `lamb` values.
- [2_backtest.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/2_backtest.py) and [3_backtest_optimal.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/3_backtest_optimal.py): Update to support backtesting each factor group separately.
- Update [test_alignment.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/test_alignment.py) and others if needed to adapt to the new [config.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/config.py) parameter structure.

## Verification Plan
### Automated Tests
- Run [test_alignment.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/test_alignment.py) and verify all tests pass with the new logic.
- Run [1_compute_alphas.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/1_compute_alphas.py) and [2_backtest.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/2_backtest.py) to ensure shape and types remain correct.
