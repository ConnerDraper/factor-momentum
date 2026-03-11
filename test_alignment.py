import datetime as dt
import polars as pl
from pipeline import FactorOptimizer
from config import IC

start = dt.date(2005, 1, 1)
end = dt.date(2005, 1, 20)

opt = FactorOptimizer()
group = "All"

returns = opt.load_factor_returns(start, end).collect()
alphas = opt.compute_alphas(returns.lazy(), group)

# Filter back down because FactorOptimizer loads extra history for rolling metrics
returns = returns.filter((pl.col("date") >= start) & (pl.col("date") <= end))
alphas = alphas.filter((pl.col("date") >= start) & (pl.col("date") <= end))

df = returns.join(alphas, on="date", suffix="_alpha")
print("Data aligned by date:")
if "BETA" in df.columns:
    print(df.select(["date", "BETA", "BETA_alpha"]).head(10))
else:
    cols = ["date"] + [c for c in df.columns if c != "date"][:2]
    print(df.select(cols).head(10))
