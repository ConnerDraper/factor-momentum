# Factor Rotation Pipeline (BARRA Framework) Implementation

This plan details the technical specifications for implementing the factor momentum strategy (Top-Down to Bottom-Up) using the BARRA framework. It defines the mathematical steps and how they map to our Python pipeline files.

## Technical Specification & Pipeline Logic

The following theoretical steps will be built into the `FactorOptimizer` class within [pipeline.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/pipeline.py).

**1. Data Inputs & Configuration**
- **Factor Payoffs ($F$):** A ($K \times T$) matrix of historical factor returns. In [config.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/config.py), we will define `STYLE_FACTORS` and `INDUSTRY_FACTORS` lists (using placeholder Barra names for now) to allow filtering the factor groups.
- **Exposure Matrix ($X$):** An ($N \times K$) matrix of current stock exposures to factors. We will load these for the target year.
- **Factor Covariance ($\Sigma_f$):** A ($K \times K$) matrix of factor variances/correlations (handled by the MVO step).

**2. Calculate Momentum Signal ($S$)**
- **Input:** Factor Payoffs ($F$), filtered by the selected `factor_group` ("All", "Style", or "Industry").
- **Operation:**
  a. Convert simple returns to log returns: $r_{log} = \ln(1 + r)$
  b. Calculate Rolling Sum: `Signal` = Sum($r_{log}$) over a lookback window from $t-12$ to $t-2$.
     - We will use `WINDOW_LENGTH = 231` (11 months) and `SKIP_LENGTH = 21` (1 month) based on daily data.

**3. Volatility Scaling (Risk Equalization)**
- **Input:** Factor Payoffs ($F$) and Momentum Signal.
- **Operation:**
  a. Calculate Rolling Std Dev: `Vol` = StdDev($F$) over the same $t-12$ to $t-2$ window.
  b. Scale: `Scaled_Signal = Signal / Vol`.

**4. Cross-Sectional Z-Score ($S_k$)**
- **Input:** `Scaled_Signal` (Vector of $K$ factors).
- **Operation:**
  a. For the current timestep, calculate the mean and standard deviation across all $K$ factors in the selected group.
  b. $Z = (\text{Scaled\_Signal} - \text{Mean}) / \text{StdDev}$.
- **Output:** Z-Scored Factor Momentum ($S_k$).

**5. Alpha Mapping / Factor Alpha to Asset Alpha**
- **Factor Alpha ($\alpha_f$):** First calculate the factor's alpha: $\alpha_{factor, k} = IC_k \cdot \sigma_k \cdot S_k$, where $IC_k$ is the Information Coefficient and $\sigma_k$ is the factor's annualized volatility.
- **Asset Alpha ($\alpha_{stock}$):** Multiply the Exposure Matrix ($X$) by Factor Alphas ($\alpha_f$).
  - Formula: $\alpha_{stock, i} = \sum_{k} X_{ik} \cdot \alpha_{factor, k}$
  - In [pipeline.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/pipeline.py), map the stock exposures against the computed `factor_alphas`.

**6. Portfolio Construction**
- Forward the computed asset alphas ($\alpha_{stock}$) to the backtest scripts.
- The `sf_backtester` will run a Constrained Mean-Variance Optimization (MVO) natively, constrained by Zero-Investment ( weights = 0 ) and Zero-Beta.

## Python Scripts & Execution Changes

- [config.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/config.py): Add `STYLE_FACTORS`, `INDUSTRY_FACTORS`, `WINDOW_LENGTH=231`, `SKIP_LENGTH=21`.
- [pipeline.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/pipeline.py): Implement the `FactorOptimizer` class with the math specified above (from Signal to Asset Alpha).
- [1_compute_alphas.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/1_compute_alphas.py): Replace `lambdas` loop with a loop over `["Style", "Industry", "All"]` and save separated alpha parquets.
- [2_backtest.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/2_backtest.py) / [3_backtest_optimal.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/3_backtest_optimal.py): Update runs to execute over the defined factor subsets independently.

## Verification Plan
1. **Automated Tests:** Update and run [test_alignment.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/test_alignment.py) to ensure shape output of the new pipeline logic is valid.
2. **Dry Run / Output Checking:** Run [1_compute_alphas.py](file:///Users/connerdraper/Work/Silverfund/factor-momentum/1_compute_alphas.py) against the `train` split and verify the alpha structures match the optimizer constraints.
