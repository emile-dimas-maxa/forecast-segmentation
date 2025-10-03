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

    # Get the actual column names from the DataFrame (they are uppercase)
    actual_columns = df.columns
    logger.debug(f"Actual DataFrame columns: {actual_columns}")

    # Use the actual column names from the DataFrame
    original_columns = actual_columns

    # Step 1: Select date
    mapping_forecast_month = (
        forecast_month
        or (df.select(F.max("FORECAST_MONTH").alias("latest_month")).to_pandas().rename(columns=str.lower).iloc[0]["latest_month"])
    )

    log_str = (
        f"Using specified forecast month for mapping: {forecast_month}"
        if forecast_month is not None
        else f"Using latest forecast month for mapping: {mapping_forecast_month}"
    )

    logger.info(log_str)

    # Step 2: Create mapping
    mapping_df = (
        df.filter(F.col("FORECAST_MONTH") == mapping_forecast_month)
        .select("DIM_VALUE", "EOM_IMPORTANCE_TIER", "OVERALL_IMPORTANCE_TIER")
        .distinct()
    )

    mapping_df = mapping_df.with_column(
        "should_aggregate",
        F.when(
            (F.col("EOM_IMPORTANCE_TIER").isin(keep_individual_eom_tiers))
            | (F.col("OVERALL_IMPORTANCE_TIER").isin(keep_individual_overall_tiers)),
            F.lit(False),
        ).otherwise(F.lit(True)),
    )

    # Step 3: Apply the mapping
    df = df.join(mapping_df.select("DIM_VALUE", "should_aggregate"), on="DIM_VALUE", how="left")

    # Step 3: Create the aggregated dim_value
    df = df.with_column(
        "aggregated_dim_value",
        F.when(
            ~F.col("should_aggregate"),
            F.col("DIM_VALUE"),
        )
        .when(
            F.col("DIM_VALUE").like(f"%{others_in_suffix}"),
            F.concat(F.lit(aggregated_in_name)),
        )
        .otherwise(F.concat(F.lit(aggregated_out_name))),
    )

    # Step 4: Group by FORECAST_MONTH and aggregated_dim_value
    logger.debug("Grouping data by FORECAST_MONTH and aggregated_dim_value")

    # Split processing: individual dim_values vs aggregated "others"
    individual_df = df.filter(~F.col("should_aggregate"))
    others_df = df.filter(F.col("should_aggregate"))

    # Process individual dim_values (keep all original values unchanged)
    # Apply identical type casting as others DataFrame for successful union
    individual_aggregated = individual_df.select(
        *original_columns,
    )
    logger.info(f"Created {individual_aggregated.count()} individual aggregated rows")

    # Process "others" categories (only sum specific fields, nulls for rest)
    # Aggregate the data first
    others_aggregated_data = (
        others_df.with_column("DIM_VALUE", F.col("aggregated_dim_value"))
        .group_by(["FORECAST_MONTH", "DIM_VALUE"])
        .agg(
            # Core identifiers (take first/max as they should be consistent within group)
            F.max("YEAR").alias("agg_year"),
            F.max("MONTH_NUM").alias("agg_month_num"),
            F.max("MONTH_OF_YEAR").alias("agg_month_of_year"),
            F.max("IS_QUARTER_END").alias("agg_is_quarter_end"),
            F.max("IS_YEAR_END").alias("agg_is_year_end"),
            # Target variable - SUM of amounts (main aggregation)
            F.sum("TARGET_EOM_AMOUNT").alias("agg_target_eom_amount"),
            # Portfolio percentiles - SUM (as requested)
            F.sum("CUMULATIVE_OVERALL_PORTFOLIO_PCT").alias("agg_cumulative_overall_portfolio_pct"),
            F.sum("CUMULATIVE_EOM_PORTFOLIO_PCT").alias("agg_cumulative_eom_portfolio_pct"),
        )
    )

    # Now create others_aggregated with exact same column structure as individual DataFrame
    # Create a mapping of original column names to their aggregated equivalents or null values
    column_mappings = {}

    for col in original_columns:
        if col == "DIM_VALUE":
            column_mappings[col] = F.col("DIM_VALUE")
        elif col == "FORECAST_MONTH":
            column_mappings[col] = F.col("FORECAST_MONTH")
        elif col == "YEAR":
            column_mappings[col] = F.col("agg_year")
        elif col == "MONTH_NUM":
            column_mappings[col] = F.col("agg_month_num")
        elif col == "MONTH_OF_YEAR":
            column_mappings[col] = F.col("agg_month_of_year")
        elif col == "IS_QUARTER_END":
            column_mappings[col] = F.col("agg_is_quarter_end")
        elif col == "IS_YEAR_END":
            column_mappings[col] = F.col("agg_is_year_end")
        elif col == "TARGET_EOM_AMOUNT":
            column_mappings[col] = F.col("agg_target_eom_amount")
        elif col == "CUMULATIVE_OVERALL_PORTFOLIO_PCT":
            column_mappings[col] = F.col("agg_cumulative_overall_portfolio_pct")
        elif col == "CUMULATIVE_EOM_PORTFOLIO_PCT":
            column_mappings[col] = F.col("agg_cumulative_eom_portfolio_pct")
        elif col == "EOM_PATTERN_PRIMARY":
            column_mappings[col] = F.lit("AGGREGATED_OTHERS")
        else:
            # Set all other columns to NULL
            column_mappings[col] = F.lit(None)

    # Create the select statement
    others_aggregated = others_aggregated_data.select(*[column_mappings[col].alias(col) for col in original_columns])

    logger.info("Columns in the non aggregated dataframe:")
    logger.info(individual_aggregated.columns)
    logger.info("Columns in the aggregated dataframe:")
    logger.info(others_aggregated.columns)

    logger.info(f"Created {others_aggregated.count()} others aggregated rows")
    logger.info("Unioning individual and others aggregated rows")
    df = individual_aggregated.select(*original_columns).union(others_aggregated.select(*original_columns))
    logger.info(f"Created aggregated features from segmentation output with {df.count()} rows")
    return df


def run_feature_pipeline(df: DataFrame, forecast_month: str | None = None) -> DataFrame:
    logger.info("Running feature pipeline")
    df = create_aggregated_features(df=df, forecast_month=forecast_month)
    logger.info("Feature pipeline completed")
    return df
