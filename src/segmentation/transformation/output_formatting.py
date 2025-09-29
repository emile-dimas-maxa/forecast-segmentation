"""
Final column selection and output formatting
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from loguru import logger

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.utils import log_transformation


@log_transformation
def select_final_columns(config: SegmentationConfig, df: DataFrame) -> DataFrame:
    """
    Select and format final output columns
    """
    logger.debug(f"Selecting final columns with target_forecast_month={config.target_forecast_month}")

    # Apply target month filter if specified
    if config.target_forecast_month:
        df = df.filter(F.col("month") == config.target_forecast_month)
        logger.info(f"Filtered to forecast month: {config.target_forecast_month}")

    # Apply importance filter if specified
    if config.filter_low_importance:
        df = df.filter(~F.col("eom_importance_tier").isin(["LOW", "NONE"]))
        logger.info("Filtered out LOW/NONE importance tiers")

    # Select final columns
    df = df.select(
        # Identifiers
        "dim_value",
        F.col("month").alias("forecast_month"),
        "year",
        "month_num",
        # Target variable
        F.col("eom_amount").alias("target_eom_amount"),
        # Dual importance metrics
        "overall_importance_tier",
        "eom_importance_tier",
        F.round("overall_importance_score", 5).alias("overall_importance_score"),
        F.round("eom_importance_score", 5).alias("eom_importance_score"),
        "eom_risk_flag",
        "has_eom_history",
        # Smooth scores
        F.round("regularity_score", 1).alias("eom_regularity_score"),
        F.round("stability_score", 1).alias("eom_stability_score"),
        F.round("recency_score", 1).alias("eom_recency_score"),
        F.round("concentration_score", 1).alias("eom_concentration_score"),
        F.round("volume_importance_score", 1).alias("eom_volume_score"),
        # Pattern probabilities
        F.col("eom_pattern").alias("eom_pattern_primary"),
        F.round(F.col("eom_pattern_confidence") * 100, 1).alias("eom_pattern_confidence_pct"),
        F.round(F.col("prob_continuous_stable") * 100, 1).alias("prob_continuous_stable_pct"),
        F.round(F.col("prob_continuous_volatile") * 100, 1).alias("prob_continuous_volatile_pct"),
        F.round(F.col("prob_intermittent_active") * 100, 1).alias("prob_intermittent_active_pct"),
        F.round(F.col("prob_intermittent_dormant") * 100, 1).alias("prob_intermittent_dormant_pct"),
        F.round(F.col("prob_rare_recent") * 100, 1).alias("prob_rare_recent_pct"),
        F.round(F.col("prob_rare_stale") * 100, 1).alias("prob_rare_stale_pct"),
        F.round("classification_entropy", 3).alias("pattern_uncertainty"),
        # General timeseries pattern
        "general_pattern",
        # Combined metrics
        "segment_name",
        "full_segment_name",
        "combined_priority",
        "recommended_method",
        "forecast_complexity",
        # Volume metrics
        F.col("rolling_total_volume_12m").alias("total_volume_12m"),
        F.col("rolling_eom_volume_12m").alias("eom_volume_12m"),
        F.col("rolling_non_eom_volume_12m").alias("non_eom_volume_12m"),
        F.col("rolling_avg_monthly_volume").alias("avg_monthly_volume"),
        F.col("rolling_max_transaction").alias("max_transaction"),
        F.col("rolling_max_eom").alias("max_eom_transaction"),
        # Raw EOM metrics
        F.round("eom_concentration", 3).alias("eom_concentration"),
        F.round("eom_predictability", 3).alias("eom_predictability"),
        F.round("eom_frequency", 3).alias("eom_frequency"),
        F.round("eom_zero_ratio", 3).alias("eom_zero_ratio"),
        F.round("eom_cv", 3).alias("eom_cv"),
        # General timeseries pattern features
        F.round("monthly_cv", 3).alias("monthly_cv"),
        F.round("transaction_regularity", 3).alias("transaction_regularity"),
        F.round("activity_rate", 3).alias("activity_rate"),
        F.round("transaction_dispersion", 2).alias("transaction_dispersion"),
        F.round("quarter_end_concentration", 3).alias("quarter_end_concentration"),
        F.round("year_end_concentration", 3).alias("year_end_concentration"),
        # Activity indicators
        "active_months_12m",
        "total_nonzero_eom_count",
        "months_inactive",
        "months_of_history",
        # Portfolio percentiles
        F.round("cumulative_overall_portfolio_pct", 4).alias("cumulative_overall_portfolio_pct"),
        F.round("cumulative_eom_portfolio_pct", 4).alias("cumulative_eom_portfolio_pct"),
        # Lagged values for forecasting
        F.coalesce("eom_amount_1m_ago", F.lit(0)).alias("lag_1m_eom"),
        F.coalesce("eom_amount_3m_ago", F.lit(0)).alias("lag_3m_eom"),
        F.coalesce("eom_amount_12m_ago", F.lit(0)).alias("lag_12m_eom"),
        F.round("eom_ma3", 2).alias("eom_ma3"),
        # Growth metrics
        F.when(F.col("months_of_history") >= 12, F.round("eom_yoy_growth", 3)).alias("eom_yoy_growth"),
        F.round("eom_mom_growth", 3).alias("eom_mom_growth"),
        # Calendar features
        "is_quarter_end",
        "is_year_end",
        F.col("month_num").alias("month_of_year"),
        # Current month status
        F.when(F.col("eom_amount") == 0, 1).otherwise(0).alias("is_zero_eom"),
        F.col("has_nonzero_eom").alias("current_month_has_eom"),
        # Raw rolling features (for model training)
        F.col("active_months_12m").alias("raw_rf__active_months_12m"),
        F.col("rolling_avg_day_dispersion").alias("raw_rf__rolling_avg_day_dispersion"),
        F.col("rolling_avg_monthly_volume").alias("raw_rf__rolling_avg_monthly_volume"),
        F.col("rolling_avg_non_eom").alias("raw_rf__rolling_avg_non_eom"),
        F.col("rolling_avg_nonzero_eom").alias("raw_rf__rolling_avg_nonzero_eom"),
        F.col("rolling_avg_transactions").alias("raw_rf__rolling_avg_transactions"),
        F.col("rolling_eom_volume_12m").alias("raw_rf__rolling_eom_volume_12m"),
        F.col("rolling_max_eom").alias("raw_rf__rolling_max_eom"),
        F.col("rolling_max_transaction").alias("raw_rf__rolling_max_transaction"),
        F.col("rolling_non_eom_volume_12m").alias("raw_rf__rolling_non_eom_volume_12m"),
        F.col("rolling_nonzero_eom_months").alias("raw_rf__rolling_nonzero_eom_months"),
        F.col("rolling_quarter_end_volume").alias("raw_rf__rolling_quarter_end_volume"),
        F.col("rolling_std_eom").alias("raw_rf__rolling_std_eom"),
        F.col("rolling_std_monthly").alias("raw_rf__rolling_std_monthly"),
        F.col("rolling_std_transactions").alias("raw_rf__rolling_std_transactions"),
        F.col("rolling_total_volume_12m").alias("raw_rf__rolling_total_volume_12m"),
        F.col("rolling_year_end_volume").alias("raw_rf__rolling_year_end_volume"),
        F.col("rolling_zero_eom_months").alias("raw_rf__rolling_zero_eom_months"),
        F.col("total_nonzero_eom_count").alias("raw_rf__total_nonzero_eom_count"),
        F.col("eom_amount_12m_ago").alias("raw_rf__eom_amount_12m_ago"),
        F.col("eom_amount_3m_ago").alias("raw_rf__eom_amount_3m_ago"),
        F.col("eom_amount_1m_ago").alias("raw_rf__eom_amount_1m_ago"),
        F.col("eom_ma3").alias("raw_rf__eom_ma3"),
        F.col("months_of_history").alias("raw_rf__months_of_history"),
        F.col("months_since_last_eom").alias("raw_rf__months_since_last_eom"),
        # Raw pattern metrics
        F.col("eom_concentration").alias("raw_pm__eom_concentration"),
        F.col("eom_predictability").alias("raw_pm__eom_predictability"),
        F.col("eom_frequency").alias("raw_pm__eom_frequency"),
        F.col("eom_zero_ratio").alias("raw_pm__eom_zero_ratio"),
        F.col("eom_spike_ratio").alias("raw_pm__eom_spike_ratio"),
        F.col("eom_cv").alias("raw_pm__eom_cv"),
        F.col("monthly_cv").alias("raw_pm__monthly_cv"),
        F.col("transaction_regularity").alias("raw_pm__transaction_regularity"),
        F.col("activity_rate").alias("raw_pm__activity_rate"),
        F.col("quarter_end_concentration").alias("raw_pm__quarter_end_concentration"),
        F.col("year_end_concentration").alias("raw_pm__year_end_concentration"),
        F.col("transaction_dispersion").alias("raw_pm__transaction_dispersion"),
        F.col("has_eom_history").alias("raw_pm__has_eom_history"),
        F.col("months_inactive").alias("raw_pm__months_inactive"),
        F.col("eom_periodicity").alias("raw_pm__eom_periodicity"),
    )

    # Order by priority
    df = df.order_by(F.col("combined_priority").desc(), "dim_value", "forecast_month")

    return df
