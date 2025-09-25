"""Rolling window feature transformations."""

import pandas as pd
import numpy as np
from src.config.segmentation import SegmentationConfig


def create_rolling_features(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Create rolling window features for volume, EOM, and activity metrics."""
    if config is None:
        config = SegmentationConfig()

    df = df.sort_values(["dim_value", "month"]).copy()

    # Create rolling windows (12-month lookback, excluding current month)
    rolling_12m = (
        df.groupby("dim_value")
        .rolling(window=12, min_periods=1)
        .agg(
            {
                "monthly_total": ["sum", "mean", "std"],
                "eom_amount": ["sum", "mean", "max", "std"],
                "non_eom_total": ["sum", "mean"],
                "max_monthly_transaction": "max",
                "monthly_transactions": ["mean", "std"],
                "day_dispersion": "mean",
                "quarter_end_amount": "sum",
                "year_end_amount": "sum",
            }
        )
        .shift(1)
    )  # Shift to exclude current month

    # Flatten column names
    rolling_12m.columns = [f"rolling_{col[0]}_{col[1]}" for col in rolling_12m.columns]
    rolling_12m = rolling_12m.reset_index(level=0, drop=True)

    # Activity counters
    df["has_activity"] = (df["monthly_total"] > 0).astype(int)
    df["has_eom_activity"] = (df["eom_amount"] > 0).astype(int)
    df["has_zero_eom"] = (df["eom_amount"] == 0).astype(int)

    activity_rolling = (
        df.groupby("dim_value")
        .rolling(window=12, min_periods=1)
        .agg({"has_activity": "sum", "has_eom_activity": "sum", "has_zero_eom": "sum"})
        .shift(1)
    )

    activity_rolling.columns = [f"rolling_{col}_months" for col in activity_rolling.columns]
    activity_rolling = activity_rolling.reset_index(level=0, drop=True)

    # Combine rolling features
    result = pd.concat([df, rolling_12m, activity_rolling], axis=1)

    # Add derived rolling metrics
    result["rolling_avg_nonzero_eom"] = result.groupby("dim_value")["eom_amount"].transform(
        lambda x: x.rolling(window=12, min_periods=1)
        .apply(lambda vals: vals[vals > 0].mean() if len(vals[vals > 0]) > 0 else 0)
        .shift(1)
    )

    # Lagged values
    for lag in [1, 3, 12]:
        result[f"eom_amount_{lag}m_ago"] = result.groupby("dim_value")["eom_amount"].shift(lag)

    # Moving averages
    result["eom_ma3"] = (
        result.groupby("dim_value")["eom_amount"].rolling(window=3, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )

    # Months of history and activity tracking
    result["months_of_history"] = result.groupby("dim_value").cumcount() + 1

    # Total historical EOM count (expanding window)
    result["total_nonzero_eom_count"] = (
        result.groupby("dim_value")["has_eom_activity"].expanding().sum().shift(1).reset_index(level=0, drop=True)
    )

    # Months since last EOM
    result["months_since_last_eom"] = (
        result.groupby("dim_value")
        .apply(lambda x: x["has_eom_activity"].eq(0).groupby(x["has_eom_activity"].ne(0).cumsum()).cumsum())
        .reset_index(level=0, drop=True)
    )

    # Fill NaN values
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)

    # Rename columns to match SQL
    result = result.rename(
        columns={
            "rolling_monthly_total_sum": "rolling_total_volume_12m",
            "rolling_monthly_total_mean": "rolling_avg_monthly_volume",
            "rolling_eom_amount_sum": "rolling_eom_volume_12m",
            "rolling_eom_amount_max": "rolling_max_eom",
            "rolling_eom_amount_std": "rolling_std_eom",
            "rolling_non_eom_total_sum": "rolling_non_eom_volume_12m",
            "rolling_non_eom_total_mean": "rolling_avg_non_eom",
            "rolling_max_monthly_transaction_max": "rolling_max_transaction",
            "rolling_monthly_transactions_mean": "rolling_avg_transactions",
            "rolling_monthly_transactions_std": "rolling_std_transactions",
            "rolling_day_dispersion_mean": "rolling_avg_day_dispersion",
            "rolling_quarter_end_amount_sum": "rolling_quarter_end_volume",
            "rolling_year_end_amount_sum": "rolling_year_end_volume",
            "rolling_has_activity_months": "active_months_12m",
            "rolling_has_eom_activity_months": "rolling_nonzero_eom_months",
            "rolling_has_zero_eom_months": "rolling_zero_eom_months",
            "rolling_monthly_total_std": "rolling_std_monthly",
        }
    )

    return result
