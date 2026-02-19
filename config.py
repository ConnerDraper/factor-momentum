"""
Shared configuration for the factor momentum pipeline.
"""
import datetime as dt

# ── Date splits ─────────────────────────────────────────────────────────────
SPLITS = {
    "train": {"start": dt.date(1995, 1, 1), "end": dt.date(2010, 1, 1)},
    "test":  {"start": dt.date(2010, 1, 1), "end": dt.date(2025, 1, 1)},
}

# ── Strategy parameters ────────────────────────────────────────────────────
GAMMA = 50
IC = 0.05

LAMBDA_GRID = [
    0.693/21,   # HL = 1 mo
    0.693/42,   # HL = 2 mo
    0.693/63,   # HL = 3 mo
    0.693/126,  # HL = 6 mo
    0.693/189,  # HL = 9 mo
    0.693/252,  # HL = 12 mo
    # 0.693/378,  # HL = 18 mo
    # 0.693/504,  # HL = 24 mo
    # 0.693/756,  # HL = 36 mo
    # 0.693/1008, # HL = 48 mo
]

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/home/connerd4/silverfund/momentum2"
FACTORS_PATH = "/home/connerd4/groups/grp_quant/database/development/factors/factors_*.parquet"
EXPOSURES_PATH = "/home/connerd4/groups/grp_quant/database/development/exposures/exposures_*.parquet"
BYU_EMAIL = "connerd4@byu.edu"

# ── Backtest constraints ───────────────────────────────────────────────────
CONSTRAINTS = ["ZeroBeta", "ZeroInvestment"]


def signal_name(lamb: float) -> str:
    """Consistent signal name for a given lambda value."""
    return f"factor_momentum_lambda_{lamb:.6f}"


def split_dir(split: str) -> str:
    """Path to the results directory for a given split."""
    return f"{PROJECT_ROOT}/results/{split}"


def alphas_path(split: str, lamb: float) -> str:
    """Path to the alpha parquet for a given split and lambda."""
    return f"{split_dir(split)}/alphas/{signal_name(lamb)}.parquet"


def weights_dir(split: str, lamb: float) -> str:
    """Path to the weights directory for a given split and lambda."""
    return f"{split_dir(split)}/weights/{signal_name(lamb)}/{GAMMA}"
