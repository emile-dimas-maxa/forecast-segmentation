from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def create_aggregated_features(
    df: DataFrame,
    forecast_month: str | None = None,
    keep_individual_eom_tiers: list[str] | None = None,
    keep_individual_overall_tiers: list[str] | None = None,
    others_in_suffix: str = "::IN",
    others_out_suffix: str = "::OUT",
    aggregated_in_name: str = "others::IN",
    aggregated_out_name: str = "others::OUT",
    include_aggregation_metadata: bool = True,
) -> DataFrame:
    """
    Transform segmentation pipeline output into aggregated features

    Rules:
    - Keep all important dim_values as individual entries with full feature sets
    - Aggregate less important dim_values into "others::IN" and "others::OUT" categories with:
      * SUM: target_eom_amount, cumulative_overall_portfolio_pct, cumulative_eom_portfolio_pct
      * NULL: all other feature fields (pattern scores, probabilities, etc.)
      * Special label: eom_pattern = "AGGREGATED_OTHERS"

    Args:
        df: Output DataFrame from segmentation pipeline
        forecast_month: Specific forecast month to process (YYYY-MM-DD format). If None, uses latest month.
        keep_individual_eom_tiers: Importance tiers that should be kept as individual dim_values (EOM)
        keep_individual_overall_tiers: Importance tiers that should be kept as individual dim_values (Overall)
        others_in_suffix: Suffix for IN aggregation
        others_out_suffix: Suffix for OUT aggregation
        aggregated_in_name: Name for aggregated IN category
        aggregated_out_name: Name for aggregated OUT category
        include_aggregation_metadata: Whether to include aggregation metadata in output

    Returns:
        Aggregated DataFrame with:
        - Individual important dim_values: full feature sets preserved
        - "Others" categories: only target amounts and portfolio percentiles summed, rest NULL
        - Special eom_pattern labels to distinguish aggregated vs individual entries
    """
    logger.debug("Creating aggregated features from segmentation output")

    # Set default values for mutable arguments
    if keep_individual_eom_tiers is None:
        keep_individual_eom_tiers = ["CRITICAL", "HIGH", "MEDIUM"]
    if keep_individual_overall_tiers is None:
        keep_individual_overall_tiers = ["CRITICAL"]

    original_columns = [
        "dim_value",
        "forecast_month",
        "year",
        "month_num",
        "target_eom_amount",
        "overall_importance_tier",
        "eom_importance_tier",
        "overall_importance_score",
        "eom_importance_score",
        "eom_risk_flag",
        "has_eom_history",
        "eom_regularity_score",
        "eom_stability_score",
        "eom_recency_score",
        "eom_concentration_score",
        "eom_volume_score",
        "eom_pattern_primary",
        "eom_pattern_confidence_pct",
        "prob_continuous_stable_pct",
        "prob_continuous_volatile_pct",
        "prob_intermittent_active_pct",
        "prob_intermittent_dormant_pct",
        "prob_rare_recent_pct",
        "prob_rare_stale_pct",
        "pattern_uncertainty",
        "general_pattern",
        "segment_name",
        "full_segment_name",
        "combined_priority",
        "recommended_method",
        "forecast_complexity",
        "total_volume_12m",
        "eom_volume_12m",
        "non_eom_volume_12m",
        "avg_monthly_volume",
        "max_transaction",
        "max_eom_transaction",
        "eom_concentration",
        "eom_predictability",
        "eom_frequency",
        "eom_zero_ratio",
        "eom_cv",
        "monthly_cv",
        "transaction_regularity",
        "activity_rate",
        "transaction_dispersion",
        "quarter_end_concentration",
        "year_end_concentration",
        "active_months_12m",
        "total_nonzero_eom_count",
        "months_inactive",
        "months_of_history",
        "cumulative_overall_portfolio_pct",
        "cumulative_eom_portfolio_pct",
        "lag_1m_eom",
        "lag_3m_eom",
        "lag_12m_eom",
        "eom_ma3",
        "eom_yoy_growth",
        "eom_mom_growth",
        "is_quarter_end",
        "is_year_end",
        "month_of_year",
        "is_zero_eom",
        "current_month_has_eom",
        "raw_rf__active_months_12m",
        "raw_rf__rolling_avg_day_dispersion",
        "raw_rf__rolling_avg_monthly_volume",
        "raw_rf__rolling_avg_non_eom",
        "raw_rf__rolling_avg_nonzero_eom",
        "raw_rf__rolling_avg_transactions",
        "raw_rf__rolling_eom_volume_12m",
        "raw_rf__rolling_max_eom",
        "raw_rf__rolling_max_transaction",
        "raw_rf__rolling_non_eom_volume_12m",
        "raw_rf__rolling_nonzero_eom_months",
        "raw_rf__rolling_quarter_end_volume",
        "raw_rf__rolling_std_eom",
        "raw_rf__rolling_std_monthly",
        "raw_rf__rolling_std_transactions",
        "raw_rf__rolling_total_volume_12m",
        "raw_rf__rolling_year_end_volume",
        "raw_rf__rolling_zero_eom_months",
        "raw_rf__total_nonzero_eom_count",
        "raw_rf__eom_amount_12m_ago",
        "raw_rf__eom_amount_3m_ago",
        "raw_rf__eom_amount_1m_ago",
        "raw_rf__eom_ma3",
        "raw_rf__months_of_history",
        "raw_rf__months_since_last_eom",
        "raw_pm__eom_concentration",
        "raw_pm__eom_predictability",
        "raw_pm__eom_frequency",
        "raw_pm__eom_zero_ratio",
        "raw_pm__eom_spike_ratio",
        "raw_pm__eom_cv",
        "raw_pm__monthly_cv",
        "raw_pm__transaction_regularity",
        "raw_pm__activity_rate",
        "raw_pm__quarter_end_concentration",
        "raw_pm__year_end_concentration",
        "raw_pm__transaction_dispersion",
        "raw_pm__has_eom_history",
        "raw_pm__months_inactive",
        "raw_pm__eom_periodicity",
    ]

    # Step 1: Select date
    mapping_forecast_month = (
        forecast_month
        or (df.select(F.max("forecast_month").alias("latest_month")).to_pandas().rename(columns=str.lower).iloc[0]["latest_month"])
    )

    log_str = (
        f"Using specified forecast month for mapping: {forecast_month}"
        if forecast_month is not None
        else f"Using latest forecast month for mapping: {mapping_forecast_month}"
    )

    logger.info(log_str)

    # Step 2: Create mapping
    mapping_df = (
        df.filter(F.col("forecast_month") == mapping_forecast_month)
        .select("dim_value", "eom_importance_tier", "overall_importance_tier")
        .distinct()
    )

    mapping_df = mapping_df.with_column(
        "should_aggregate",
        F.when(
            (F.col("eom_importance_tier").isin(keep_individual_eom_tiers))
            | (F.col("overall_importance_tier").isin(keep_individual_overall_tiers)),
            F.lit(False),
        ).otherwise(F.lit(True)),
    )

    # Step 3: Apply the mapping
    df = df.join(mapping_df.select("dim_value", "should_aggregate"), on="dim_value", how="left")

    # Step 3: Create the aggregated dim_value
    df = df.with_column(
        "aggregated_dim_value",
        F.when(
            ~F.col("should_aggregate"),
            F.col("dim_value"),
        )
        .when(
            F.col("dim_value").like(f"%{others_in_suffix}"),
            F.concat(F.lit(aggregated_in_name)),
        )
        .otherwise(F.concat(F.lit(aggregated_out_name))),
    )

    # Step 4: Group by forecast_month and aggregated_dim_value
    logger.debug("Grouping data by forecast_month and aggregated_dim_value")

    # Split processing: individual dim_values vs aggregated "others"
    individual_df = df.filter(~F.col("should_aggregate"))
    others_df = df.filter(F.col("should_aggregate"))

    # Process individual dim_values (keep all original values unchanged)
    individual_aggregated = individual_df.select(original_columns)
    logger.info(f"Created {individual_aggregated.count()} individual aggregated rows")

    # Process "others" categories (only sum specific fields, nulls for rest)
    others_aggregated = (
        others_df.with_column("dim_value", F.col("aggregated_dim_value"))
        .group_by(["forecast_month", "dim_value"])
        .agg(
            # Core identifiers (take first/max as they should be consistent within group)
            F.max("year").alias("year"),
            F.max("month_num").alias("month_num"),
            F.max("month_of_year").alias("month_of_year"),
            F.max("is_quarter_end").alias("is_quarter_end"),
            F.max("is_year_end").alias("is_year_end"),
            # Target variable - SUM of amounts (main aggregation)
            F.sum("target_eom_amount").alias("target_eom_amount"),
            # Portfolio percentiles - SUM (as requested)
            F.sum("cumulative_overall_portfolio_pct").alias("cumulative_overall_portfolio_pct"),
            F.sum("cumulative_eom_portfolio_pct").alias("cumulative_eom_portfolio_pct"),
            # Special EOM pattern label for aggregated others
            F.lit("AGGREGATED_OTHERS").alias("eom_pattern_primary"),
            # Set all other feature columns to NULL
            *[
                F.lit(None).alias(col)
                for col in original_columns
                if col
                not in [
                    "dim_value",
                    "forecast_month",
                    "year",
                    "month_num",
                    "month_of_year",
                    "is_quarter_end",
                    "is_year_end",
                    "target_eom_amount",
                    "cumulative_overall_portfolio_pct",
                    "cumulative_eom_portfolio_pct",
                    "eom_pattern_primary",
                ]
            ],
        )
    )

    logger.info(f"Created {others_aggregated.count()} others aggregated rows")
    logger.info("Unioning individual and others aggregated rows")
    df = individual_aggregated.union(others_aggregated)
    logger.info(f"Created aggregated features from segmentation output with {df.count()} rows")
    return df


def run_feature_pipeline(df: DataFrame, forecast_month: str | None = None) -> DataFrame:
    logger.info("Running feature pipeline")
    df = create_aggregated_features(df=df, forecast_month=forecast_month)
    logger.info("Feature pipeline completed")
    return df
