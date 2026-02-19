import polars as pl
import sf_quant.data as sfd
import datetime as dt

print("Checking sfd.load_assets behavior for a few stocks...")
start = dt.date(2005, 1, 1)
end = dt.date(2005, 1, 10)

df = sfd.load_assets(
    start=start, end=end,
    columns=["date", "barrid", "price", "return"],
    in_universe=True,
)

# Pick one stock
sample = df.filter(pl.col("barrid") == df["barrid"][0]).sort("date")
print(sample)

# Check return calculation: is return[t] = price[t]/price[t-1] - 1 ?
sample = sample.with_columns(
    (pl.col("price") / pl.col("price").shift(1) - 1).mul(100).alias("calc_return")
)
print("\nComparing 'return' vs calculated from price:")
print(sample.select("date", "price", "return", "calc_return"))
