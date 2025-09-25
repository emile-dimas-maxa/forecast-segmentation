"""Importance classification transformations."""

import pandas as pd
import numpy as np
from src.config.segmentation import SegmentationConfig


def classify_importance(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Classify overall and EOM importance tiers."""
    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    # Overall importance tier
    def classify_overall_importance(row):
        if (
            row["cumulative_overall_portfolio_pct"] <= config.overall_critical_percentile
            or row["rolling_avg_monthly_volume"] >= config.critical_monthly_avg_threshold
            or row["rolling_max_transaction"] >= config.critical_max_transaction_threshold
        ):
            return "CRITICAL"
        elif (
            row["cumulative_overall_portfolio_pct"] <= config.overall_high_percentile
            or row["rolling_avg_monthly_volume"] >= config.high_monthly_avg_threshold
            or row["rolling_max_transaction"] >= config.high_max_transaction_threshold
        ):
            return "HIGH"
        elif (
            row["cumulative_overall_portfolio_pct"] <= config.overall_medium_percentile
            or row["rolling_avg_monthly_volume"] >= config.medium_monthly_avg_threshold
            or row["rolling_max_transaction"] >= config.medium_max_transaction_threshold
        ):
            return "MEDIUM"
        else:
            return "LOW"

    df["overall_importance_tier"] = df.apply(classify_overall_importance, axis=1)

    # EOM importance tier
    def classify_eom_importance(row):
        if (
            row["cumulative_eom_portfolio_pct"] <= config.eom_critical_percentile
            or row["rolling_avg_nonzero_eom"] >= config.critical_eom_monthly_threshold
            or row["rolling_max_eom"] >= config.critical_max_eom_threshold
        ):
            return "CRITICAL"
        elif (
            row["cumulative_eom_portfolio_pct"] <= config.eom_high_percentile
            or row["rolling_avg_nonzero_eom"] >= config.high_eom_monthly_threshold
            or row["rolling_max_eom"] >= config.high_max_eom_threshold
        ):
            return "HIGH"
        elif (
            row["cumulative_eom_portfolio_pct"] <= config.eom_medium_percentile
            or (row["rolling_avg_nonzero_eom"] >= config.medium_eom_monthly_threshold and row["total_nonzero_eom_count"] >= 3)
            or row["rolling_max_eom"] >= config.medium_max_eom_threshold
        ):
            return "MEDIUM"
        elif row["rolling_eom_volume_12m"] > 0 or row["total_nonzero_eom_count"] > 0:
            return "LOW"
        else:
            return "NONE"

    df["eom_importance_tier"] = df.apply(classify_eom_importance, axis=1)

    # Importance scores
    df["overall_importance_score"] = np.where(
        df["total_portfolio_volume"] > 0, df["rolling_total_volume_12m"] / df["total_portfolio_volume"], 0
    )

    df["eom_importance_score"] = np.where(
        df["total_portfolio_eom_volume"] > 0, df["rolling_eom_volume_12m"] / df["total_portfolio_eom_volume"], 0
    )

    # EOM risk flag
    df["eom_risk_flag"] = (
        (df["rolling_total_volume_12m"] >= config.eom_risk_volume_threshold)
        & (df["has_eom_history"] == 0)
        & (df["months_of_history"] >= config.eom_risk_min_months)
    ).astype(int)

    return df
