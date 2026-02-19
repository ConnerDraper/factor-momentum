"""
Shared pipeline logic for factor momentum.
All heavy computation lives here; numbered scripts are thin callers.
"""
import datetime as dt
import polars as pl
import sf_quant.data as sfd
from config import FACTORS_PATH, EXPOSURES_PATH, IC


class LambdaOptimizer:
    """Compute factor momentum alphas, map to stocks, and submit backtests."""

    def load_factor_returns(self, start: dt.date, end: dt.date) -> pl.LazyFrame:
        """Load factor returns for the given date range."""
        return pl.scan_parquet(FACTORS_PATH).filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )

    def compute_alphas(self, returns: pl.LazyFrame, lamb: float) -> pl.DataFrame:
        """
        Compute alphas from factor returns using EWMA momentum signal.
            signal_i = EWMA(returns_i) / sqrt(EWMA(returns_i²)) for each factor i
        Normalize signals cross-sectionally, then:
            alpha_{i,t} = IC * risk_{i,t-1} * score_{i,t-1}
        """
        df = returns.collect().sort("date")
        factor_cols = [c for c in df.columns if c != "date"]

        # EWMA of returns (momentum signal) and EWMA of r² (variance proxy)
        ewma_ret = df.select(
            "date",
            *[pl.col(c).ewm_mean(alpha=lamb).alias(c) for c in factor_cols]
        )
        ewma_vol = df.select(
            "date",
            *[pl.col(c).pow(2).ewm_mean(alpha=lamb).sqrt().alias(c) for c in factor_cols]
        )

        # Vol-adjusted signal: EWMA(return) / sqrt(EWMA(return²))
        signal = pl.DataFrame({
            "date": ewma_ret["date"],
            **{c: ewma_ret[c] / ewma_vol[c] for c in factor_cols}
        })

        # Cross-sectional z-score (across factors per day), clipped to [-3, 3]
        signal_vals = signal.select(factor_cols)
        row_mean = signal_vals.mean_horizontal()
        sq_diffs = signal_vals.select(
            *[(pl.col(c) - row_mean).pow(2).alias(c) for c in factor_cols]
        )
        row_std = sq_diffs.mean_horizontal().sqrt()

        scores = pl.DataFrame({
            c: ((signal[c] - row_mean) / row_std).clip(-3.0, 3.0)
            for c in factor_cols
        })

        # Alpha = IC * risk_{t-1} * score_{t-1}
        # Risk proxy = sqrt(EWMA(r²)), lagged by 1 day
        risk_lagged = ewma_vol.select(factor_cols).shift(1)
        scores_lagged = scores.shift(1)

        alphas = pl.DataFrame({
            "date": df["date"],
            **{c: IC * risk_lagged[c] * scores_lagged[c] for c in factor_cols}
        }).drop_nulls()

        return alphas

    def load_exposures(self, start: dt.date, end: dt.date) -> pl.LazyFrame:
        """Load Barra factor exposures per stock for the given date range."""
        return pl.scan_parquet(EXPOSURES_PATH).filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )

    def load_assets(self, start: dt.date, end: dt.date) -> pl.LazyFrame:
        """Load stock-level data (price, predicted_beta) for universe filtering."""
        return sfd.load_assets(
            start=start, end=end,
            columns=["date", "barrid", "price", "predicted_beta"],
            in_universe=True,
        ).lazy()

    def map_to_assets(
        self,
        factor_alphas: pl.DataFrame,
        start: dt.date,
        end: dt.date,
        min_price: float = 5.0,
    ) -> pl.DataFrame:
        """
        Map factor alphas → stock alphas, join with assets, and filter universe.
        Processes one year at a time to avoid OOM on large exposure data.

        Returns df with columns: date, barrid, predicted_beta, alpha.
        """
        factor_cols = [c for c in factor_alphas.columns if c != "date"]

        # Build dot-product expressions: sum(exposure_i * alpha_i)
        alpha_terms = [
            (pl.col(c) * pl.col(f"{c}_alpha")).alias(f"{c}_term")
            for c in factor_cols
        ]

        # Load assets once (small: only 4 columns)
        assets = self.load_assets(start, end).collect().sort(["barrid", "date"])
        assets = assets.with_columns(
            pl.col("price").shift(1).over("barrid").alias("prev_price")
        )

        # Process year by year
        years = range(start.year, end.year + 1)
        chunks = []

        for year in years:
            year_alphas = factor_alphas.filter(
                (pl.col("date").dt.year() == year)
            )
            if year_alphas.height == 0:
                continue

            # Load just this year's exposures
            path = EXPOSURES_PATH.replace("*", str(year))
            try:
                exp_year = pl.read_parquet(path)
            except FileNotFoundError:
                continue

            # Join exposures × factor alphas, compute dot product
            merged = exp_year.join(year_alphas, on="date", suffix="_alpha")
            stock_alphas = (
                merged.select("date", "barrid", *alpha_terms)
                .with_columns(
                    pl.sum_horizontal([f"{c}_term" for c in factor_cols]).alias("alpha")
                )
                .select("date", "barrid", "alpha")
            )

            # Join with assets and filter
            year_assets = assets.filter(pl.col("date").dt.year() == year)
            filtered = (
                stock_alphas.join(year_assets, on=["date", "barrid"], how="inner")
                .filter(
                    (pl.col("prev_price") > min_price) &
                    pl.col("predicted_beta").is_not_null() &
                    pl.col("alpha").is_not_null()
                )
                .select("date", "barrid", "predicted_beta", "alpha")
            )
            chunks.append(filtered)

            # Free memory
            del exp_year, merged, stock_alphas, filtered

        return pl.concat(chunks).sort("date", "barrid")

