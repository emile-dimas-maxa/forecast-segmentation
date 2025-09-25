"""Portfolio metrics transformations."""

import pandas as pd
import numpy as np
from src.config.segmentation import SegmentationConfig


def calculate_portfolio_metrics(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Calculate portfolio-level metrics and percentiles."""
    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    # Filter for accounts with sufficient history
    df = df[df["months_of_history"] >= config.min_months_history]

    # Calculate portfolio totals by month
    portfolio_totals = df.groupby("month").agg({"rolling_total_volume_12m": "sum", "rolling_eom_volume_12m": "sum"}).reset_index()

    portfolio_totals.columns = ["month", "total_portfolio_volume", "total_portfolio_eom_volume"]

    # Merge back to main dataframe
    df = df.merge(portfolio_totals, on="month", how="left")

    # Calculate cumulative portfolio percentages
    df["cumulative_overall_portfolio_pct"] = df.groupby("month")["rolling_total_volume_12m"].transform(
        lambda x: x.rank(method="min", ascending=False).cumsum() / len(x)
    )

    df["cumulative_eom_portfolio_pct"] = df.groupby("month")["rolling_eom_volume_12m"].transform(
        lambda x: x.rank(method="min", ascending=False).cumsum() / len(x)
    )

    # Handle division by zero
    df["cumulative_overall_portfolio_pct"] = df["cumulative_overall_portfolio_pct"].fillna(1.0)
    df["cumulative_eom_portfolio_pct"] = df["cumulative_eom_portfolio_pct"].fillna(1.0)

    return df
