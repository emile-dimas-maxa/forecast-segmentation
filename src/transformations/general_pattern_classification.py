"""General timeseries pattern classification."""

import pandas as pd
import numpy as np
from src.config.segmentation import SegmentationConfig


def classify_general_patterns(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Classify general timeseries behavioral patterns."""
    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    def classify_pattern(row):
        # INACTIVE: No recent activity
        if row["months_inactive"] >= config.inactive_months:
            return "INACTIVE"

        # EMERGING: Too new to classify
        if row["months_of_history"] <= config.emerging_months:
            return "EMERGING"

        # HIGHLY_SEASONAL: Strong quarterly/yearly patterns
        if row["quarter_end_concentration"] >= config.ts_seasonal_concentration_threshold or row["year_end_concentration"] >= 0.5:
            return "HIGHLY_SEASONAL"

        # INTERMITTENT: Sporadic overall activity
        if row["activity_rate"] <= config.ts_intermittent_threshold or row["active_months_12m"] <= 4:
            return "INTERMITTENT"

        # VOLATILE: Regular but unpredictable amounts
        if (
            row["monthly_cv"] >= config.ts_high_volatility_threshold
            and row["transaction_regularity"] >= config.ts_medium_regularity_threshold
        ):
            return "VOLATILE"

        # MODERATELY_VOLATILE: Some variability
        if (
            row["monthly_cv"] >= config.ts_medium_volatility_threshold
            and row["transaction_regularity"] >= config.ts_medium_regularity_threshold
        ):
            return "MODERATELY_VOLATILE"

        # STABLE: Regular and predictable
        if (
            row["monthly_cv"] < config.ts_medium_volatility_threshold
            and row["transaction_regularity"] >= config.ts_high_regularity_threshold
        ):
            return "STABLE"

        # CONCENTRATED: Activity clustered in time
        if row["transaction_dispersion"] < 5 and row["transaction_regularity"] >= config.ts_medium_regularity_threshold:
            return "CONCENTRATED"

        # DISTRIBUTED: Activity spread throughout month
        if row["transaction_dispersion"] >= 8 and row["transaction_regularity"] >= config.ts_medium_regularity_threshold:
            return "DISTRIBUTED"

        return "MIXED"

    df["general_pattern"] = df.apply(classify_pattern, axis=1)

    return df
