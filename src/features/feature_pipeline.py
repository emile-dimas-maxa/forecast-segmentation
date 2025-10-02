"""
Feature pipeline for aggregating segmentation output
Creates aggregated time series by grouping less important dim_values into "others" categories
"""

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.features.config import FeatureConfig
from src.segmentation.transformation.utils import log_transformation


@log_transformation
def create_aggregated_features(
    df: DataFrame,
    forecast_month: str | None = None,
    keep_individual_eom_tiers: list[str] = None,
    keep_individual_overall_tiers: list[str] = None,
    others_in_suffix: str = "::IN",
    others_out_suffix: str = "::OUT",
    aggregated_in_name: str = "others::IN",
    aggregated_out_name: str = "others::OUT",
    include_aggregation_metadata: bool = True,
) -> DataFrame:
    """
    Transform segmentation pipeline output into aggregated features

    Rules:
    - Keep all important dim_values as individual entries
    - Only aggregate less important dim_values into "others::IN" and "others::OUT" categories

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
        Aggregated DataFrame with individual important dim_values and aggregated "others" categories
    """
    # Set defaults for optional parameters
    if keep_individual_eom_tiers is None:
        keep_individual_eom_tiers = ["CRITICAL", "HIGH", "MEDIUM"]
    if keep_individual_overall_tiers is None:
        keep_individual_overall_tiers = ["CRITICAL"]

    logger.debug("Creating aggregated features from segmentation output")

    # Step 1: Select the latest mapping - determine which forecast month to use for mapping
    if forecast_month is not None:
        mapping_forecast_month = forecast_month
        logger.info(f"Using specified forecast month for mapping: {forecast_month}")
    else:
        # Get the latest forecast month for mapping
        mapping_forecast_month = (
            df.select(F.max("forecast_month").alias("latest_month")).to_pandas().rename(columns=str.lower).iloc[0]["latest_month"]
        )
        logger.info(f"Using latest forecast month for mapping: {mapping_forecast_month}")

    # Step 2: Create mapping from the selected forecast month to determine which dim_values should be aggregated
    # Filter to mapping month to get the importance tiers for each dim_value
    mapping_df = (
        df.filter(F.col("forecast_month") == mapping_forecast_month)
        .select("dim_value", "eom_importance_tier", "overall_importance_tier")
        .distinct()
    )

    # Create the aggregation mapping
    mapping_df = mapping_df.with_column(
        "should_aggregate",
        F.when(
            # Keep as individual if high importance
            (F.col("eom_importance_tier").isin(keep_individual_eom_tiers))
            | (F.col("overall_importance_tier").isin(keep_individual_overall_tiers)),
            F.lit(False),
        ).otherwise(F.lit(True)),
    )

    # Step 3: Apply the mapping to the unfiltered dataframe
    df = df.join(mapping_df.select("dim_value", "should_aggregate"), on="dim_value", how="left")

    # Step 3 (continued): Create aggregated_dim_value column using the mapping
    # Use concat to build the aggregated names to avoid :: literal issues
    df = df.with_column(
        "aggregated_dim_value",
        F.when(
            ~F.col("should_aggregate"),
            F.col("dim_value"),  # Keep original dim_value for important ones
        )
        .when(
            # Aggregate less important ones based on suffix
            F.col("dim_value").like("%::IN"),
            F.concat(F.lit("others::IN")),
        )
        .otherwise(
            # All other less important ones go to OUT
            F.concat(F.lit("others::OUT"))
        ),
    )

    # Step 4: Group by forecast_month and aggregated_dim_value
    # This will keep individual important dim_values as separate groups
    # and aggregate only the "others::IN" and "others::OUT" categories
    logger.debug("Grouping data by forecast_month and aggregated_dim_value")

    aggregated_df = df.group_by(["forecast_month", "aggregated_dim_value"]).agg(
        # Core identifiers (take first/max as they should be consistent within group)
        F.max("year").alias("year"),
        F.max("month_num").alias("month_num"),
        F.max("month_of_year").alias("month_of_year"),
        F.max("is_quarter_end").alias("is_quarter_end"),
        F.max("is_year_end").alias("is_year_end"),
        # Target variable - SUM of amounts (main aggregation)
        F.sum("target_eom_amount").alias("target_eom_amount"),
        # Importance metrics - take weighted average by target amount or max for categorical
        F.max("overall_importance_tier").alias("overall_importance_tier"),  # Most important tier in group
        F.max("eom_importance_tier").alias("eom_importance_tier"),  # Most important tier in group
        F.sum(F.col("overall_importance_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_overall_importance_score"),
        F.sum(F.col("eom_importance_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_importance_score"),
        F.sum(F.abs(F.col("target_eom_amount"))).alias("total_abs_amount"),  # For weighted average calculation
        # Risk indicators - max (if any dim_value has risk, group has risk)
        F.max("eom_risk_flag").alias("eom_risk_flag"),
        F.max("has_eom_history").alias("has_eom_history"),
        # Pattern scores - weighted average by absolute amount
        F.sum(F.col("eom_regularity_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_regularity_score"),
        F.sum(F.col("eom_stability_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_stability_score"),
        F.sum(F.col("eom_recency_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_recency_score"),
        F.sum(F.col("eom_concentration_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_concentration_score"),
        F.sum(F.col("eom_volume_score") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_volume_score"),
        # Pattern probabilities - weighted average
        F.sum(F.col("prob_continuous_stable_pct") * F.abs(F.col("target_eom_amount"))).alias("weighted_prob_continuous_stable_pct"),
        F.sum(F.col("prob_continuous_volatile_pct") * F.abs(F.col("target_eom_amount"))).alias(
            "weighted_prob_continuous_volatile_pct"
        ),
        F.sum(F.col("prob_intermittent_active_pct") * F.abs(F.col("target_eom_amount"))).alias(
            "weighted_prob_intermittent_active_pct"
        ),
        F.sum(F.col("prob_intermittent_dormant_pct") * F.abs(F.col("target_eom_amount"))).alias(
            "weighted_prob_intermittent_dormant_pct"
        ),
        F.sum(F.col("prob_rare_recent_pct") * F.abs(F.col("target_eom_amount"))).alias("weighted_prob_rare_recent_pct"),
        F.sum(F.col("prob_rare_stale_pct") * F.abs(F.col("target_eom_amount"))).alias("weighted_prob_rare_stale_pct"),
        F.sum(F.col("pattern_uncertainty") * F.abs(F.col("target_eom_amount"))).alias("weighted_pattern_uncertainty"),
        # Combined metrics - max priority, most common pattern/method
        F.max("combined_priority").alias("combined_priority"),
        F.max("forecast_complexity").alias("forecast_complexity"),
        # Volume metrics - SUM for totals, weighted average for rates/ratios
        F.sum("total_volume_12m").alias("total_volume_12m"),
        F.sum("eom_volume_12m").alias("eom_volume_12m"),
        F.sum("non_eom_volume_12m").alias("non_eom_volume_12m"),
        F.sum("avg_monthly_volume").alias("total_avg_monthly_volume"),  # Sum of averages
        F.max("max_transaction").alias("max_transaction"),
        F.max("max_eom_transaction").alias("max_eom_transaction"),
        # Pattern metrics - weighted averages
        F.sum(F.col("eom_concentration") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_concentration"),
        F.sum(F.col("eom_predictability") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_predictability"),
        F.sum(F.col("eom_frequency") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_frequency"),
        F.sum(F.col("eom_zero_ratio") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_zero_ratio"),
        F.sum(F.col("eom_cv") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_cv"),
        F.sum(F.col("monthly_cv") * F.abs(F.col("target_eom_amount"))).alias("weighted_monthly_cv"),
        F.sum(F.col("transaction_regularity") * F.abs(F.col("target_eom_amount"))).alias("weighted_transaction_regularity"),
        F.sum(F.col("activity_rate") * F.abs(F.col("target_eom_amount"))).alias("weighted_activity_rate"),
        # Activity indicators - SUM for counts, weighted average for ratios
        F.sum("active_months_12m").alias("total_active_months_12m"),
        F.sum("total_nonzero_eom_count").alias("total_nonzero_eom_count"),
        F.sum("months_of_history").alias("total_months_of_history"),
        # Portfolio percentiles - weighted sum
        F.sum(F.col("cumulative_overall_portfolio_pct") * F.abs(F.col("target_eom_amount"))).alias(
            "weighted_cumulative_overall_portfolio_pct"
        ),
        F.sum(F.col("cumulative_eom_portfolio_pct") * F.abs(F.col("target_eom_amount"))).alias(
            "weighted_cumulative_eom_portfolio_pct"
        ),
        # Lagged values - SUM for amounts
        F.sum("lag_1m_eom").alias("lag_1m_eom"),
        F.sum("lag_3m_eom").alias("lag_3m_eom"),
        F.sum("lag_12m_eom").alias("lag_12m_eom"),
        F.sum("eom_ma3").alias("eom_ma3"),
        # Growth metrics - weighted average
        F.sum(F.col("eom_yoy_growth") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_yoy_growth"),
        F.sum(F.col("eom_mom_growth") * F.abs(F.col("target_eom_amount"))).alias("weighted_eom_mom_growth"),
        # Status indicators
        F.when(F.sum("target_eom_amount") == 0, 1).otherwise(0).alias("is_zero_eom"),
        F.max("current_month_has_eom").alias("current_month_has_eom"),
        # Count of original dim_values in this aggregated group
        F.count("*").alias("dim_value_count"),
        F.count_distinct("dim_value").alias("unique_dim_value_count"),
    )

    # Step 5: Calculate final weighted averages for aggregated categories
    # Individual dim_values will have their original values preserved
    logger.debug("Computing final weighted averages for aggregated categories")

    aggregated_df = aggregated_df.with_columns(
        [
            "overall_importance_score",
            "eom_importance_score",
            "eom_regularity_score",
            "eom_stability_score",
            "eom_recency_score",
            "eom_concentration_score",
            "eom_volume_score",
            "prob_continuous_stable_pct",
            "prob_continuous_volatile_pct",
            "prob_intermittent_active_pct",
            "prob_intermittent_dormant_pct",
            "prob_rare_recent_pct",
            "prob_rare_stale_pct",
            "pattern_uncertainty",
            "eom_concentration",
            "eom_predictability",
            "eom_frequency",
            "eom_zero_ratio",
            "eom_cv",
            "monthly_cv",
            "transaction_regularity",
            "activity_rate",
            "cumulative_overall_portfolio_pct",
            "cumulative_eom_portfolio_pct",
            "eom_yoy_growth",
            "eom_mom_growth",
        ],
        [
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_overall_importance_score") / F.col("total_abs_amount")).otherwise(
                0
            ),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_importance_score") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_regularity_score") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_stability_score") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_recency_score") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_concentration_score") / F.col("total_abs_amount")).otherwise(
                0
            ),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_volume_score") / F.col("total_abs_amount")).otherwise(0),
            F.when(
                F.col("total_abs_amount") > 0, F.col("weighted_prob_continuous_stable_pct") / F.col("total_abs_amount")
            ).otherwise(0),
            F.when(
                F.col("total_abs_amount") > 0, F.col("weighted_prob_continuous_volatile_pct") / F.col("total_abs_amount")
            ).otherwise(0),
            F.when(
                F.col("total_abs_amount") > 0, F.col("weighted_prob_intermittent_active_pct") / F.col("total_abs_amount")
            ).otherwise(0),
            F.when(
                F.col("total_abs_amount") > 0, F.col("weighted_prob_intermittent_dormant_pct") / F.col("total_abs_amount")
            ).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_prob_rare_recent_pct") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_prob_rare_stale_pct") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_pattern_uncertainty") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_concentration") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_predictability") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_frequency") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_zero_ratio") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_cv") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_monthly_cv") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_transaction_regularity") / F.col("total_abs_amount")).otherwise(
                0
            ),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_activity_rate") / F.col("total_abs_amount")).otherwise(0),
            F.when(
                F.col("total_abs_amount") > 0, F.col("weighted_cumulative_overall_portfolio_pct") / F.col("total_abs_amount")
            ).otherwise(0),
            F.when(
                F.col("total_abs_amount") > 0, F.col("weighted_cumulative_eom_portfolio_pct") / F.col("total_abs_amount")
            ).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_yoy_growth") / F.col("total_abs_amount")).otherwise(0),
            F.when(F.col("total_abs_amount") > 0, F.col("weighted_eom_mom_growth") / F.col("total_abs_amount")).otherwise(0),
        ],
    )

    # Step 6: Determine dominant patterns for categorical fields
    logger.debug("Determining dominant categorical patterns")

    # For categorical fields, we'll need to determine the most common pattern within each group
    # This is more complex and would require window functions, so for now we'll use a simpler approach
    # by selecting the pattern from the highest volume contributor

    # Add derived fields
    aggregated_df = aggregated_df.with_columns(
        [
            "dim_value",  # Use aggregated_dim_value as the main identifier
            "avg_monthly_volume",  # Convert total back to average
            "segment_name",  # Create aggregated segment name
            "recommended_method",  # Determine recommended method based on patterns
        ],
        [
            F.col("aggregated_dim_value"),
            F.col("total_avg_monthly_volume") / F.greatest(F.col("unique_dim_value_count"), F.lit(1)),
            F.concat(F.col("overall_importance_tier"), F.lit("_AGGREGATED_"), F.col("eom_importance_tier")),
            F.when(F.col("eom_volume_12m") == 0, "Zero_EOM_Forecast")
            .when(F.col("prob_continuous_stable_pct") > 50, "Simple_MA_EOM")
            .when(F.col("prob_continuous_volatile_pct") > 30, "XGBoost_EOM_Focus")
            .when(F.col("prob_intermittent_active_pct") > 30, "Croston_Method")
            .otherwise("Historical_Average"),
        ],
    )

    # Step 7: Clean up intermediate columns and select final output
    final_columns = [
        # Core identifiers
        "forecast_month",
        "dim_value",
        "year",
        "month_num",
        "month_of_year",
        "is_quarter_end",
        "is_year_end",
        # Target variable
        "target_eom_amount",
        # Importance metrics
        "overall_importance_tier",
        "eom_importance_tier",
        "overall_importance_score",
        "eom_importance_score",
        "eom_risk_flag",
        "has_eom_history",
        # Pattern scores
        "eom_regularity_score",
        "eom_stability_score",
        "eom_recency_score",
        "eom_concentration_score",
        "eom_volume_score",
        # Pattern probabilities
        "prob_continuous_stable_pct",
        "prob_continuous_volatile_pct",
        "prob_intermittent_active_pct",
        "prob_intermittent_dormant_pct",
        "prob_rare_recent_pct",
        "prob_rare_stale_pct",
        "pattern_uncertainty",
        # Combined metrics
        "segment_name",
        "combined_priority",
        "recommended_method",
        "forecast_complexity",
        # Volume metrics
        "total_volume_12m",
        "eom_volume_12m",
        "non_eom_volume_12m",
        "avg_monthly_volume",
        "max_transaction",
        "max_eom_transaction",
        # Pattern metrics
        "eom_concentration",
        "eom_predictability",
        "eom_frequency",
        "eom_zero_ratio",
        "eom_cv",
        "monthly_cv",
        "transaction_regularity",
        "activity_rate",
        # Activity indicators
        "total_active_months_12m",
        "total_nonzero_eom_count",
        "total_months_of_history",
        # Portfolio percentiles
        "cumulative_overall_portfolio_pct",
        "cumulative_eom_portfolio_pct",
        # Lagged values
        "lag_1m_eom",
        "lag_3m_eom",
        "lag_12m_eom",
        "eom_ma3",
        # Growth metrics
        "eom_yoy_growth",
        "eom_mom_growth",
        # Status indicators
        "is_zero_eom",
        "current_month_has_eom",
        # Add aggregation metadata if requested
    ] + (["dim_value_count", "unique_dim_value_count"] if include_aggregation_metadata else [])

    result_df = aggregated_df.select(*final_columns)

    # Order by forecast_month and target_eom_amount (descending)
    result_df = result_df.order_by("forecast_month", F.col("target_eom_amount").desc())

    logger.info("Feature aggregation complete. Aggregated from individual dim_values to grouped categories.")

    return result_df


def run_feature_pipeline(
    segmentation_df: DataFrame,
    forecast_month: str | None = None,
    keep_individual_eom_tiers: list[str] = None,
    keep_individual_overall_tiers: list[str] = None,
    others_in_suffix: str = "::IN",
    others_out_suffix: str = "::OUT",
    aggregated_in_name: str = "others::IN",
    aggregated_out_name: str = "others::OUT",
    include_aggregation_metadata: bool = True,
) -> DataFrame:
    """
    Main entry point for the feature pipeline

    Args:
        segmentation_df: Output from segmentation pipeline
        forecast_month: Specific forecast month to process (YYYY-MM-DD format). If None, uses latest month.
        keep_individual_eom_tiers: Importance tiers that should be kept as individual dim_values (EOM)
        keep_individual_overall_tiers: Importance tiers that should be kept as individual dim_values (Overall)
        others_in_suffix: Suffix for IN aggregation
        others_out_suffix: Suffix for OUT aggregation
        aggregated_in_name: Name for aggregated IN category
        aggregated_out_name: Name for aggregated OUT category
        include_aggregation_metadata: Whether to include aggregation metadata in output

    Returns:
        Aggregated feature DataFrame with individual important dim_values and aggregated "others" categories
    """
    logger.info("=" * 80)
    logger.info("Starting Feature Aggregation Pipeline")
    logger.info("=" * 80)

    # Apply feature aggregation
    result_df = create_aggregated_features(
        df=segmentation_df,
        forecast_month=forecast_month,
        keep_individual_eom_tiers=keep_individual_eom_tiers,
        keep_individual_overall_tiers=keep_individual_overall_tiers,
        others_in_suffix=others_in_suffix,
        others_out_suffix=others_out_suffix,
        aggregated_in_name=aggregated_in_name,
        aggregated_out_name=aggregated_out_name,
        include_aggregation_metadata=include_aggregation_metadata,
    )

    logger.info("=" * 80)
    logger.info("Completed Feature Aggregation Pipeline Successfully")
    logger.info("=" * 80)

    return result_df
