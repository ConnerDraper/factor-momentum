"""
Shared configuration for the factor momentum pipeline.
"""
import datetime as dt

# ── Date splits ─────────────────────────────────────────────────────────────
SPLITS = {
    "full": {"start": dt.date(1995, 1, 1), "end": dt.date(2025, 1, 1)},
    "train": {"start": dt.date(1995, 1, 1), "end": dt.date(2010, 1, 1)},
    "test": {"start": dt.date(2010, 1, 1), "end": dt.date(2025, 1, 1)},
}

# ── Strategy parameters ────────────────────────────────────────────────────
GAMMA = 50          # Risk aversion parameter for Mean-Variance Optimization (MVO)
IC = 0.05           # Assumed Information Coefficient for alpha generation

# Window parameters for Momentum Signal calculation
WINDOW_LENGTH = 231
SKIP_LENGTH = 21

# Factor groupings (based on Barra USE4 nominal names)
STYLE_FACTORS = [
    "BETA", "DIVYILD", "EARNQLTY", "EARNYILD", "GROWTH", "LEVERAGE", 
    "LIQUIDTY", "LTREVRSL", "MGMTQLTY", "MIDCAP", "MOMENTUM", "PROFIT", 
    "PROSPECT", "RESVOL", "SIZE", "VALUE"
]

INDUSTRY_FACTORS = [
    "AERODEF", "AIRLINES", "ALUMSTEL", "APPAREL", "AUTO", "BANKS", 
    "BEVTOB", "BIOLIFE", "BLDGPROD", "CHEM", "CNSTENG", "CNSTMACH", 
    "CNSTMATL", "COMMEQP", "COMPELEC", "COMSVCS", "CONGLOM", "CONTAINR", 
    "DISTRIB", "DIVFIN", "ELECEQP", "ELECUTIL", "FOODPROD", "FOODRET", 
    "GASUTIL", "HLTHEQP", "HLTHSVCS", "HOMEBLDG", "HOUSEDUR", "INDMACH", 
    "INSURNCE", "INTERNET", "LEISPROD", "LEISSVCS", "LIFEINS", "MEDIA", 
    "MGDHLTH", "MULTUTIL", "OILGSCON", "OILGSDRL", "OILGSEQP", "OILGSEXP", 
    "PAPER", "PHARMA", "PRECMTLS", "PSNLPROD", "REALEST", "RESTAUR", 
    "ROADRAIL", "SEMICOND", "SEMIEQP", "SOFTWARE", "SPLTYRET", "SPTYCHEM", 
    "SPTYSTOR", "TELECOM", "TRADECO", "TRANSPRT", "WIRELESS"
]

FACTOR_GROUPS = ["Style", "Industry", "All"]

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/home/connerd4/silverfund/momentum"
FACTORS_PATH = "/home/connerd4/groups/grp_quant/database/research/factors/factors_*.parquet"
EXPOSURES_PATH = "/home/connerd4/groups/grp_quant/database/research/exposures/exposures_*.parquet"
BYU_EMAIL = "connerd4@byu.edu"

# ── Backtest constraints ───────────────────────────────────────────────────
# Active portfolio constraints explicitly passed to the sf_backtester runner
CONSTRAINTS = ["ZeroBeta", "ZeroInvestment"]


def signal_name(factor_group: str) -> str:
    """Consistent signal name for a given factor group."""
    return f"factor_momentum_{factor_group.lower()}"


def split_dir(split: str) -> str:
    """Path to the results directory for a given split."""
    return f"{PROJECT_ROOT}/results/{split}"


def alphas_path(split: str, factor_group: str) -> str:
    """Path to the alpha parquet for a given split and factor group."""
    return f"{split_dir(split)}/alphas/{signal_name(factor_group)}.parquet"


def weights_dir(split: str, factor_group: str) -> str:
    """Path to the weights directory for a given split and factor group."""
    return f"{split_dir(split)}/weights/{signal_name(factor_group)}/{GAMMA}"
