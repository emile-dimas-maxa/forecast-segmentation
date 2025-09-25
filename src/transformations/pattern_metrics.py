"""Pattern metrics calculation transformations."""

import pandas as pd
import numpy as np
from src.config.segmentation import SegmentationConfig


def calculate_pattern_metrics(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Calculate EOM and general timeseries pattern metrics."""
    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    # EOM pattern metrics
    df["eom_concentration"] = np.where(
        df["rolling_total_volume_12m"] > 0, df["rolling_eom_volume_12m"] / df["rolling_total_volume_12m"], 0
    )

    df["eom_predictability"] = np.where(
        df["rolling_avg_nonzero_eom"] > 0, np.maximum(0, 1 - (df["rolling_std_eom"] / df["rolling_avg_nonzero_eom"])), 0
    )

    df["eom_frequency"] = np.where(
        df["months_of_history"] > 1, df["rolling_nonzero_eom_months"] / np.minimum(df["months_of_history"] - 1, 12), 0
    )

    df["eom_zero_ratio"] = np.where(
        df["months_of_history"] > 1, df["rolling_zero_eom_months"] / np.minimum(df["months_of_history"] - 1, 12), 0
    )

    # EOM spike ratio
    df["eom_spike_ratio"] = np.where(
        (df["rolling_avg_non_eom"] > 0) & (df["rolling_avg_nonzero_eom"] > 0),
        df["rolling_avg_nonzero_eom"] / df["rolling_avg_non_eom"],
        np.where(
            (df["rolling_avg_nonzero_eom"] > 0) & (df["rolling_avg_non_eom"] == 0),
            999,  # Infinite spike
            1,
        ),
    )

    # EOM coefficient of variation
    df["eom_cv"] = np.where(df["rolling_avg_nonzero_eom"] > 0, df["rolling_std_eom"] / df["rolling_avg_nonzero_eom"], 0)

    # General timeseries metrics
    df["monthly_cv"] = np.where(
        df["rolling_avg_monthly_volume"] > 0, df["rolling_std_monthly"] / df["rolling_avg_monthly_volume"], 0
    )

    df["transaction_regularity"] = np.where(
        df["rolling_avg_transactions"] > 0, np.maximum(0, 1 - (df["rolling_std_transactions"] / df["rolling_avg_transactions"])), 0
    )

    df["activity_rate"] = np.where(
        df["months_of_history"] > 0, df["active_months_12m"] / np.minimum(df["months_of_history"], 12), 0
    )

    # Seasonality indicators
    df["quarter_end_concentration"] = np.where(
        df["rolling_total_volume_12m"] > 0, df["rolling_quarter_end_volume"] / df["rolling_total_volume_12m"], 0
    )

    df["year_end_concentration"] = np.where(
        df["rolling_total_volume_12m"] > 0, df["rolling_year_end_volume"] / df["rolling_total_volume_12m"], 0
    )

    # Transaction dispersion
    df["transaction_dispersion"] = np.where(df["rolling_avg_day_dispersion"] > 0, df["rolling_avg_day_dispersion"], 0)

    # EOM history flag
    df["has_eom_history"] = (df["total_nonzero_eom_count"] > 0).astype(int)

    # Months inactive
    df["months_inactive"] = np.where(
        df["monthly_total"] > 0,
        0,
        df.groupby("dim_value")["monthly_total"].transform(lambda x: (x == 0).groupby((x != 0).cumsum()).cumsum()),
    )

    # EOM periodicity detection
    def detect_eom_periodicity(row):
        if row["total_nonzero_eom_count"] < 3 or row["months_of_history"] < 12:
            return "INSUFFICIENT_DATA"

        count = row["total_nonzero_eom_count"]
        history = row["months_of_history"]
        nonzero_months = row["rolling_nonzero_eom_months"]

        if nonzero_months >= 10:
            return "MONTHLY"
        elif count % 3 == 0 and count >= history // 4:
            return "QUARTERLY"
        elif count % 6 == 0 and count >= history // 7:
            return "SEMIANNUAL"
        elif count == history // 12:
            return "ANNUAL"
        else:
            return "IRREGULAR"

    df["eom_periodicity"] = df.apply(detect_eom_periodicity, axis=1)

    return df
