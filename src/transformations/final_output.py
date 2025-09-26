"""Final output transformation with recommendations and growth metrics."""

import time

import numpy as np
import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig


def create_final_output(df: pd.DataFrame, config: SegmentationConfig = None, target_month: str = "2025-07-01") -> pd.DataFrame:
    """Create final output with combined priority, recommendations, and growth metrics."""
    start_time = time.time()
    initial_rows = len(df)

    logger.debug("Starting final output generation")
    logger.debug("Input shape: {} rows × {} columns", initial_rows, len(df.columns))
    logger.debug("Target month: {}", target_month)

    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    # Calculate growth metrics
    df["eom_yoy_growth"] = np.where(
        df["eom_amount_12m_ago"] > 0,
        (df["eom_amount"] - df["eom_amount_12m_ago"]) / df["eom_amount_12m_ago"],
        np.where((df["eom_amount_12m_ago"] == 0) & (df["eom_amount"] > 0), 999.99, 0),
    )

    df["eom_mom_growth"] = df.groupby("dim_value")["eom_amount"].pct_change().fillna(0)

    # Combined priority (1-10 scale)
    def calculate_priority(row):
        # Highest priority: Critical importance with complex patterns
        if (row["overall_importance_tier"] == "CRITICAL" or row["eom_importance_tier"] == "CRITICAL") and (
            row["eom_pattern"] in ["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE"]
            or row["general_pattern"] in ["VOLATILE", "INTERMITTENT"]
        ):
            return 10

        if (row["overall_importance_tier"] == "CRITICAL" or row["eom_importance_tier"] == "CRITICAL") and (
            row["eom_pattern"] == "INTERMITTENT_DORMANT" or row["general_pattern"] == "HIGHLY_SEASONAL"
        ):
            return 9

        if row["overall_importance_tier"] == "CRITICAL" or row["eom_importance_tier"] == "CRITICAL":
            return 8

        # High importance with complex patterns
        if (row["overall_importance_tier"] == "HIGH" or row["eom_importance_tier"] == "HIGH") and (
            row["eom_pattern"] in ["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE"]
            or row["general_pattern"] in ["VOLATILE", "INTERMITTENT"]
        ):
            return 7

        if row["overall_importance_tier"] == "HIGH" or row["eom_importance_tier"] == "HIGH":
            return 6

        # EOM risk cases or medium importance with complex patterns
        if row["eom_risk_flag"] == 1 or (
            row["overall_importance_tier"] == "MEDIUM" and row["general_pattern"] in ["VOLATILE", "INTERMITTENT"]
        ):
            return 5

        # Medium importance
        if row["overall_importance_tier"] == "MEDIUM" or row["eom_importance_tier"] == "MEDIUM":
            return 4

        # Low importance but active
        if row["general_pattern"] not in ["INACTIVE", "EMERGING"]:
            return 3

        # Emerging patterns
        if row["general_pattern"] == "EMERGING" or row["eom_pattern"] == "EMERGING":
            return 2

        return 1

    df["combined_priority"] = df.apply(calculate_priority, axis=1)

    # Recommended forecasting method
    def get_recommendation(row):
        if row["eom_pattern"] == "NO_EOM":
            return "Zero_EOM_Forecast"

        if row["eom_pattern"] in ["CONTINUOUS_STABLE", "CONTINUOUS_VOLATILE"]:
            if row["eom_importance_tier"] in ["CRITICAL", "HIGH"] and row["eom_pattern"] == "CONTINUOUS_VOLATILE":
                return "XGBoost_EOM_Focus"
            elif row["eom_importance_tier"] in ["CRITICAL", "HIGH"]:
                return "SARIMA_EOM"
            else:
                return "Simple_MA_EOM"

        if row["eom_pattern"] in ["INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]:
            if row["overall_importance_tier"] in ["CRITICAL", "HIGH"]:
                return "Croston_Method"
            else:
                return "Zero_Inflated_Model"

        if row["general_pattern"] == "HIGHLY_SEASONAL":
            if row["overall_importance_tier"] in ["CRITICAL", "HIGH"]:
                return "Seasonal_Decomposition"
            else:
                return "Seasonal_Naive"

        if row["eom_pattern"] in ["RARE_RECENT", "RARE_STALE"]:
            if row["general_pattern"] in ["VOLATILE", "MODERATELY_VOLATILE"]:
                return "Ensemble_Method"
            elif row["general_pattern"] == "STABLE":
                return "Prophet"
            else:
                return "Historical_Average"

        # General patterns without significant EOM
        if row["general_pattern"] == "VOLATILE":
            return "XGBoost_Full_Series"
        elif row["general_pattern"] == "STABLE":
            return "Linear_Trend"
        elif row["general_pattern"] == "CONCENTRATED":
            return "Peak_Detection_Model"
        elif row["general_pattern"] == "DISTRIBUTED":
            return "Daily_Decomposition"
        elif row["general_pattern"] == "INACTIVE":
            return "Zero_Forecast"
        elif row["general_pattern"] == "EMERGING":
            return "Conservative_MA"

        return "Historical_Average"

    df["recommended_method"] = df.apply(get_recommendation, axis=1)

    # Forecast complexity (1-5)
    def get_complexity(row):
        if (
            row["eom_pattern"] in ["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE"]
            or row["general_pattern"] in ["VOLATILE", "INTERMITTENT"]
        ) and (row["overall_importance_tier"] in ["CRITICAL", "HIGH"] or row["eom_importance_tier"] in ["CRITICAL", "HIGH"]):
            return 5

        if (row["eom_pattern"] == "INTERMITTENT_DORMANT" or row["general_pattern"] == "HIGHLY_SEASONAL") and row[
            "overall_importance_tier"
        ] in ["CRITICAL", "HIGH"]:
            return 4

        if (
            row["eom_pattern"] in ["CONTINUOUS_STABLE", "CONTINUOUS_VOLATILE"]
            or row["general_pattern"] == "STABLE"
            or row["overall_importance_tier"] in ["CRITICAL", "HIGH"]
        ):
            return 3

        if row["overall_importance_tier"] == "MEDIUM" or row["eom_importance_tier"] == "MEDIUM":
            return 2

        return 1

    df["forecast_complexity"] = df.apply(get_complexity, axis=1)

    # Segment names
    df["full_segment_name"] = (
        df["overall_importance_tier"] + "_" + df["general_pattern"] + "__" + df["eom_importance_tier"] + "EOM_" + df["eom_pattern"]
    )

    df["segment_name"] = np.where(
        df["overall_importance_tier"] == df["eom_importance_tier"],
        df["overall_importance_tier"] + "_" + df["general_pattern"] + "_" + df["eom_pattern"],
        df["overall_importance_tier"] + "/" + df["eom_importance_tier"] + "EOM_" + df["general_pattern"] + "_" + df["eom_pattern"],
    )

    # Filter for target month and select final columns
    result = df[df["month"] == target_month].copy() if target_month else df.copy()

    # Select and rename columns to match SQL output
    final_columns = {
        "dim_value": "dim_value",
        "month": "forecast_month",
        "year": "year",
        "month_num": "month_num",
        "eom_amount": "target_eom_amount",
        "overall_importance_tier": "overall_importance_tier",
        "eom_importance_tier": "eom_importance_tier",
        "overall_importance_score": "overall_importance_score",
        "eom_importance_score": "eom_importance_score",
        "eom_risk_flag": "eom_risk_flag",
        "has_eom_history": "has_eom_history",
        "regularity_score": "eom_regularity_score",
        "stability_score": "eom_stability_score",
        "recency_score": "eom_recency_score",
        "concentration_score": "eom_concentration_score",
        "volume_importance_score": "eom_volume_score",
        "eom_pattern": "eom_pattern_primary",
        "eom_pattern_confidence": "eom_pattern_confidence_pct",
        "classification_entropy": "pattern_uncertainty",
        "general_pattern": "general_pattern",
        "segment_name": "segment_name",
        "full_segment_name": "full_segment_name",
        "combined_priority": "combined_priority",
        "recommended_method": "recommended_method",
        "forecast_complexity": "forecast_complexity",
        "rolling_total_volume_12m": "total_volume_12m",
        "rolling_eom_volume_12m": "eom_volume_12m",
        "rolling_non_eom_volume_12m": "non_eom_volume_12m",
        "rolling_avg_monthly_volume": "avg_monthly_volume",
        "rolling_max_transaction": "max_transaction",
        "rolling_max_eom": "max_eom_transaction",
        "eom_yoy_growth": "eom_yoy_growth",
        "eom_mom_growth": "eom_mom_growth",
        "is_quarter_end": "is_quarter_end",
        "is_year_end": "is_year_end",
        "has_nonzero_eom": "current_month_has_eom",
    }

    # Add probability columns
    prob_cols = {
        f"prob_{pattern}": f"prob_{pattern}_pct"
        for pattern in [
            "continuous_stable",
            "continuous_volatile",
            "intermittent_active",
            "intermittent_dormant",
            "rare_recent",
            "rare_stale",
        ]
    }
    final_columns.update(prob_cols)

    # Select available columns
    available_columns = {k: v for k, v in final_columns.items() if k in result.columns}
    result = result[list(available_columns.keys())].rename(columns=available_columns)

    # Round numeric columns
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(5)

    # Convert percentage columns
    pct_columns = [col for col in result.columns if "pct" in col.lower()]
    for col in pct_columns:
        if col in result.columns:
            result[col] = (result[col] * 100).round(1)

    elapsed_time = time.time() - start_time
    logger.debug(
        "Final output generation completed in {:.2f}s - {} rows × {} columns", elapsed_time, len(result), len(result.columns)
    )

    # Log summary statistics
    if "combined_priority_score" in result.columns:
        avg_priority = result["combined_priority_score"].mean()
        logger.debug("Average combined priority score: {:.3f}", avg_priority)

    if "recommendation" in result.columns:
        rec_counts = result["recommendation"].value_counts()
        logger.debug("Recommendation distribution: {}", rec_counts.to_dict())

    result = result.sort_values(["combined_priority", "dim_value"], ascending=[False, True])
    return result
