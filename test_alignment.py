import datetime as dt
import polars as pl
from pipeline import LambdaOptimizer
from config import IC

start = dt.date(2005, 1, 1)
end = dt.date(2005, 1, 20)

opt = LambdaOptimizer()
lamb = 0.003667

returns = opt.load_factor_returns(start, end).collect()
alphas = opt.compute_alphas(returns.lazy(), lamb)

df = returns.join(alphas, on="date", suffix="_alpha")
print("Data aligned by date:")
print(df.select(["date", "Beta", "Beta_alpha"]).head(10))
