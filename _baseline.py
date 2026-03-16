"""
Baseline backtest from sf-quant v0.1.23 (before the optimizer-refactor).

This module contains the OLD code path that builds the full N×N covariance matrix
and passes it to _quadratic_program as `w^T Σ w`. It is used by benchmark_optimizer.py
to compare against the refactored (factored) code path.

Data-loading functions (construct_covariance_matrix, load_assets_by_date, etc.)
are imported from the CURRENT sf-quant, since those were NOT changed in the refactor.
Only the optimizer formulation and backtest loop differ.
"""

import numpy as np
import cvxpy as cp
import polars as pl
import tqdm

from sf_quant.data.covariance_matrix import construct_covariance_matrix
from sf_quant.optimizer.constraints import Constraint, _construct_constraints


# ————————————————————————————————————————————————————————————————————
# Old _quadratic_program: takes full covariance_matrix (N×N)
# ————————————————————————————————————————————————————————————————————
def _quadratic_program_old(
    alphas: np.ndarray,
    covariance_matrix: np.ndarray,
    gamma: float,
    constraints: list[cp.Constraint],
) -> np.ndarray:
    n_assets = len(alphas)
    weights = cp.Variable(n_assets)
    constraints = [constraint(weights) for constraint in constraints]

    portfolio_return = weights.T @ alphas
    portfolio_variance = weights.T @ covariance_matrix @ weights

    objective = cp.Maximize(portfolio_return - 0.5 * gamma * portfolio_variance)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="OSQP")

    return weights.value


# ————————————————————————————————————————————————————————————————————
# Old mve_optimizer: takes full covariance_matrix (N×N)
# ————————————————————————————————————————————————————————————————————
def mve_optimizer_old(
    ids: list[str],
    alphas: np.ndarray,
    covariance_matrix: np.ndarray,
    constraints: list[Constraint],
    gamma: float = 2,
    betas: np.ndarray | None = None,
) -> pl.DataFrame:
    constraints = _construct_constraints(constraints, betas=betas)
    optimal_weights = _quadratic_program_old(
        alphas=alphas,
        covariance_matrix=covariance_matrix,
        gamma=gamma,
        constraints=constraints,
    )
    return pl.DataFrame({"barrid": ids, "weight": optimal_weights})


# ————————————————————————————————————————————————————————————————————
# Old backtest_sequential: builds full N×N covariance matrix per date
# ————————————————————————————————————————————————————————————————————
def backtest_sequential_old(
    data: pl.DataFrame, constraints: list[Constraint], gamma: float = 2
) -> pl.DataFrame:
    """Old v0.1.23 backtest: construct_covariance_matrix → mve_optimizer(Σ)."""
    dates = data["date"].unique().sort().to_list()
    portfolio_list = []

    for date_ in tqdm.tqdm(dates, "Running OLD backtest"):
        subset = data.filter(pl.col("date").eq(date_)).sort("barrid")
        barrids = subset["barrid"].to_list()
        alphas = subset["alpha"].to_numpy()

        betas = (
            subset["predicted_beta"].to_numpy()
            if "predicted_beta" in subset.columns
            else None
        )

        covariance_matrix = (
            construct_covariance_matrix(date_, barrids).drop("barrid").to_numpy()
        )

        portfolio = mve_optimizer_old(
            ids=barrids,
            alphas=alphas,
            covariance_matrix=covariance_matrix,
            gamma=gamma,
            constraints=constraints,
            betas=betas,
        )

        portfolio = portfolio.with_columns(pl.lit(date_).alias("date")).select(
            "date", "barrid", "weight"
        )
        portfolio_list.append(portfolio)

    return pl.concat(portfolio_list).sort("barrid", "date")
