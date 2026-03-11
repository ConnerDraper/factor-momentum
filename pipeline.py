"""
Shared pipeline logic for factor momentum.
All heavy computation lives here; numbered scripts are thin callers.
"""
import datetime as dt
import numpy as np
import polars as pl
import sf_quant.data as sfd
from config import (
    FACTORS_PATH, EXPOSURES_PATH, IC, 
    STYLE_FACTORS, INDUSTRY_FACTORS, WINDOW_LENGTH, SKIP_LENGTH
)

# Configure Silverfund Data path
#sfd.env(root="/home/connerd4", database="groups/grp_quant/database/research")


class FactorOptimizer:
    """Compute factor momentum alphas, map to stocks, and submit backtests."""

    def load_factor_returns(self, start: dt.date, end: dt.date) -> pl.LazyFrame:
        """
        Load factor returns for the given date range. 
        Pad start date by ~400 calendar days to allow rolling calculations (252 trading days).
        """
        padded_start = start - dt.timedelta(days=400)
        return pl.scan_parquet(FACTORS_PATH).filter(
            (pl.col("date") >= padded_start) & (pl.col("date") <= end)
        )

    def compute_alphas(self, returns: pl.LazyFrame, factor_group: str) -> pl.DataFrame:
        """
        Compute alphas from factor returns using Top-Down to Bottom-Up strategy.
        """
        df = returns.collect().sort("date")
        
        # Determine factor columns based on group
        if factor_group == "Style":
            factor_cols = [c for c in df.columns if c.split("_")[-1] in STYLE_FACTORS]
        elif factor_group == "Industry":
            factor_cols = [c for c in df.columns if c.split("_")[-1] in INDUSTRY_FACTORS]
        elif factor_group == "All":
            factor_cols = [c for c in df.columns if c.split("_")[-1] in STYLE_FACTORS or c.split("_")[-1] in INDUSTRY_FACTORS]
        else:
            raise ValueError(f"Unknown factor group: {factor_group}")
            
        factor_cols = [c for c in factor_cols if c in df.columns]

        # 1. Calculate Rolling Sum (Momentum Signal)
        signal = df.select(
            "date",
            *[pl.col(c).rolling_sum(window_size=WINDOW_LENGTH).shift(SKIP_LENGTH).alias(c) for c in factor_cols]
        )

        # 3. Volatility Scaling (Risk Equalization)
        vol = df.select(
            "date",
            *[pl.col(c).rolling_std(window_size=WINDOW_LENGTH).shift(SKIP_LENGTH).alias(c) for c in factor_cols]
        )

        scaled_signal = pl.DataFrame({
            "date": signal["date"],
            **{c: signal[c] / vol[c] for c in factor_cols}
        })

        # 4. Cross-Sectional Z-Score
        signal_vals = scaled_signal.select(factor_cols)
        
        # Calculate row mean ignoring nulls/NaNs
        row_mean = signal_vals.mean_horizontal()
        
        # Calculate squared differences
        sq_diffs = signal_vals.select(
            *[(pl.col(c) - row_mean).pow(2).alias(c) for c in factor_cols]
        )
        
        # Calculate row std ignoring nulls/NaNs
        row_std = sq_diffs.mean_horizontal().sqrt()

        # Z-score and clip, then fill remaining NaNs with 0 (neutral view)
        scores = pl.DataFrame({
            c: ((scaled_signal[c] - row_mean) / row_std).clip(-3.0, 3.0).fill_nan(0.0).fill_null(0.0) for c in factor_cols
        })

        # 5. Factor Alpha Mapping
        alphas = pl.DataFrame({
            "date": df["date"],
            **{c: (IC * vol[c].fill_nan(0.0).fill_null(0.0) * np.sqrt(252) * scores[c]).fill_nan(0.0).fill_null(0.0) for c in factor_cols}
        }).drop_nulls(subset=["date"])

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
            (pl.col(c).fill_null(0.0) * pl.col(f"{c}_alpha")).alias(f"{c}_term")
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

