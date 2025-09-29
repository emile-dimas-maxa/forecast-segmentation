"""
EOM Forecasting Segmentation Pipeline
Modular transformation functions for Snowpark DataFrames
"""

from typing import Optional, Callable, Any
from functools import wraps
import time
from datetime import date

import snowflake.snowpark as snowpark
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F
from snowflake.snowpark import types as T
from snowflake.snowpark.window import Window
from loguru import logger

from src.segmentation.config import SegmentationConfig


def log_transformation(func: Callable) -> Callable:
    """
    Decorator to log transformation execution details

    Args:
        func: The transformation function to wrap

    Returns:
        Wrapped function with logging
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Extract meaningful information for logging
        func_name = func.__name__

        # Get DataFrame from args (usually first argument after config)
        df_arg_position = 1 if len(args) > 1 and isinstance(args[1], DataFrame) else 0
        if df_arg_position < len(args) and isinstance(args[df_arg_position], DataFrame):
            input_df = args[df_arg_position]
            try:
                input_count = input_df.count()
                logger.info(f"Starting {func_name} | Input rows: {input_count:,}")
            except:
                logger.info(f"Starting {func_name} | Input DataFrame provided")
        else:
            logger.info(f"Starting {func_name}")

        # Execute the transformation
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log output information
            if isinstance(result, DataFrame):
                try:
                    output_count = result.count()
                    logger.success(f"Completed {func_name} | Output rows: {output_count:,} | Execution time: {execution_time:.2f}s")
                except:
                    logger.success(f"Completed {func_name} | Execution time: {execution_time:.2f}s")
            else:
                logger.success(f"Completed {func_name} | Execution time: {execution_time:.2f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func_name} | Error: {str(e)} | Execution time: {execution_time:.2f}s")
            raise

    return wrapper


class SegmentationPipeline:
    """Pipeline for EOM forecasting segmentation using static methods"""

    @staticmethod
    def run_full_pipeline(session: Session, config: SegmentationConfig, source_df: Optional[DataFrame] = None) -> DataFrame:
        """
        Run the complete segmentation pipeline

        Args:
            session: Snowpark session
            config: Configuration object
            source_df: Optional source DataFrame. If not provided, reads from config.source_table

        Returns:
            Final segmented DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Full Segmentation Pipeline")
        logger.info(f"Configuration: start_date={config.start_date}, end_date={config.end_date}")
        logger.info("=" * 80)

        # Step 1: Load and prepare base data
        if source_df is None:
            df = SegmentationPipeline.load_source_data(session, config)
        else:
            df = source_df

        df = SegmentationPipeline.prepare_base_data(config, df)

        # Step 1.5: Apply clipping if configured
        if config.daily_amount_clip_threshold is not None:
            df = SegmentationPipeline.apply_daily_clipping(config, df)

        # Step 2: Monthly aggregations
        df = SegmentationPipeline.create_monthly_aggregates(config, df)

        # Step 3: Rolling features
        df = SegmentationPipeline.calculate_rolling_features(config, df)

        # Step 4: Portfolio metrics
        df = SegmentationPipeline.calculate_portfolio_metrics(config, df)

        # Step 5: Pattern metrics
        df = SegmentationPipeline.calculate_pattern_metrics(config, df)

        # Step 6: Importance classification
        df = SegmentationPipeline.classify_importance(config, df)

        # Step 7: EOM pattern classification (smooth scoring)
        df = SegmentationPipeline.calculate_eom_smooth_scores(config, df)
        df = SegmentationPipeline.calculate_pattern_distances(config, df)
        df = SegmentationPipeline.calculate_pattern_probabilities(config, df)
        df = SegmentationPipeline.classify_eom_patterns(config, df)

        # Step 8: General pattern classification
        df = SegmentationPipeline.classify_general_patterns(config, df)

        # Step 9: Final classification and recommendations
        df = SegmentationPipeline.create_final_classification(config, df)

        # Step 10: Growth metrics
        df = SegmentationPipeline.calculate_growth_metrics(config, df)

        # Final selection and filtering
        df = SegmentationPipeline.select_final_columns(config, df)

        logger.info("=" * 80)
        logger.info("Completed Full Segmentation Pipeline Successfully")
        logger.info("=" * 80)

        return df

    @staticmethod
    @log_transformation
    def load_source_data(session: Session, config: SegmentationConfig) -> DataFrame:
        """Load source data from Snowflake table"""
        logger.debug(f"Loading data from table: {config.source_table}")
        return session.table(config.source_table)

    @staticmethod
    @log_transformation
    def prepare_base_data(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 1: Data preparation - add time-based features
        """
        # Filter by date range
        df = df.filter(
            (F.col("date") >= F.lit(config.start_date)) & (F.col("date") <= F.coalesce(F.lit(config.end_date), F.current_date()))
        )

        # Add month and quarter identifiers
        df = df.with_columns(
            [
                "month",
                "quarter",
                "year",
                "month_num",
                "quarter_num",
                "day_of_month",
            ],
            [
                F.date_trunc("month", F.col("date")),
                F.date_trunc("quarter", F.col("date")),
                F.year(F.col("date")),
                F.month(F.col("date")),
                F.quarter(F.col("date")),
                F.dayofmonth(F.col("date")),
            ],
        )

        # Calculate business days from EOM
        window_spec = Window.partition_by("dim_value", F.date_trunc("month", F.col("date"))).order_by(F.col("date").desc())

        df = df.with_column(
            "days_from_eom", F.when(F.col("is_last_work_day_of_month") == True, 0).otherwise(-F.row_number().over(window_spec))
        )

        return df

    @staticmethod
    @log_transformation
    def apply_daily_clipping(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 1.5: Apply clipping to daily amounts below threshold
        Clips small daily amounts to zero to reduce noise
        Provides detailed analysis of clipping impact
        """
        threshold = config.daily_amount_clip_threshold
        logger.info(f"Applying daily clipping with threshold: {threshold:,.2f}")

        # Store original amount for analysis
        df = df.with_column("original_amount", F.col("amount"))

        # Create clipping indicators
        df = df.with_columns(
            [
                "is_clipped",
                F.when((F.abs(F.col("amount")) > 0) & (F.abs(F.col("amount")) < threshold), 1).otherwise(0),
                "clipped_amount",
                F.when((F.abs(F.col("amount")) > 0) & (F.abs(F.col("amount")) < threshold), F.col("amount")).otherwise(0),
            ]
        )

        # Apply clipping
        df = df.with_column(
            "amount", F.when((F.abs(F.col("amount")) > 0) & (F.abs(F.col("amount")) < threshold), 0).otherwise(F.col("amount"))
        )

        # Perform clipping analysis if enabled
        if config.clip_analysis_enabled:
            SegmentationPipeline._analyze_clipping_impact(config, df)

        return df

    @staticmethod
    def _analyze_clipping_impact(config: SegmentationConfig, df: DataFrame) -> None:
        """
        Analyze the impact of clipping on the data
        """
        logger.info("Performing detailed clipping analysis...")

        try:
            # Overall clipping statistics
            overall_stats = df.agg(
                # Total transactions
                F.count("*").alias("total_transactions"),
                F.count(F.when(F.col("original_amount") != 0, 1)).alias("total_nonzero_transactions"),
                # Clipped transactions
                F.sum("is_clipped").alias("clipped_transactions"),
                F.count_distinct(F.when(F.col("is_clipped") == 1, F.col("dim_value"))).alias("affected_dim_values"),
                # Amount statistics
                F.sum(F.abs("original_amount")).alias("total_original_amount"),
                F.sum(F.abs("clipped_amount")).alias("total_clipped_amount"),
                F.sum(F.abs("amount")).alias("total_after_clipping"),
                # EOM specific
                F.sum(F.when(F.col("is_last_work_day_of_month") & (F.col("is_clipped") == 1), 1)).alias("clipped_eom_transactions"),
                F.sum(F.when(F.col("is_last_work_day_of_month"), F.abs("clipped_amount"))).alias("clipped_eom_amount"),
            ).collect()[0]

            # Calculate percentages
            pct_transactions_clipped = (
                (overall_stats["clipped_transactions"] / overall_stats["total_nonzero_transactions"] * 100)
                if overall_stats["total_nonzero_transactions"] > 0
                else 0
            )
            pct_amount_clipped = (
                (overall_stats["total_clipped_amount"] / overall_stats["total_original_amount"] * 100)
                if overall_stats["total_original_amount"] > 0
                else 0
            )

            # Log overall statistics
            logger.info("=" * 60)
            logger.info("CLIPPING ANALYSIS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Clipping Threshold: {config.daily_amount_clip_threshold:,.2f}")
            logger.info(f"Total Transactions: {overall_stats['total_transactions']:,}")
            logger.info(f"Non-zero Transactions: {overall_stats['total_nonzero_transactions']:,}")
            logger.info(f"Clipped Transactions: {overall_stats['clipped_transactions']:,} ({pct_transactions_clipped:.2f}%)")
            logger.info(f"Affected Dim Values: {overall_stats['affected_dim_values']:,}")
            logger.info(f"Total Original Amount: {overall_stats['total_original_amount']:,.2f}")
            logger.info(f"Total Clipped Amount: {overall_stats['total_clipped_amount']:,.2f} ({pct_amount_clipped:.2f}%)")
            logger.info(f"Total After Clipping: {overall_stats['total_after_clipping']:,.2f}")
            logger.info(f"Clipped EOM Transactions: {overall_stats['clipped_eom_transactions']:,}")
            logger.info(f"Clipped EOM Amount: {overall_stats['clipped_eom_amount']:,.2f}")

            # Analysis by dim_value
            dim_value_stats = (
                df.group_by("dim_value")
                .agg(
                    F.count("*").alias("total_days"),
                    F.sum("is_clipped").alias("clipped_days"),
                    F.sum(F.abs("original_amount")).alias("original_total"),
                    F.sum(F.abs("clipped_amount")).alias("clipped_total"),
                    F.avg(F.when(F.col("is_clipped") == 1, F.abs("clipped_amount"))).alias("avg_clipped_amount"),
                    F.max(F.when(F.col("is_clipped") == 1, F.abs("clipped_amount"))).alias("max_clipped_amount"),
                    F.sum(F.when(F.col("is_last_work_day_of_month") & (F.col("is_clipped") == 1), 1)).alias("clipped_eom_days"),
                )
                .filter(F.col("clipped_days") > 0)
            )

            # Get top affected dim_values
            top_affected = dim_value_stats.order_by(F.col("clipped_total").desc()).limit(20).to_pandas()

            if not top_affected.empty:
                logger.info("\n" + "=" * 60)
                logger.info("TOP 20 AFFECTED DIM_VALUES BY CLIPPED AMOUNT")
                logger.info("=" * 60)
                for _, row in top_affected.iterrows():
                    pct_clipped = (row["clipped_total"] / row["original_total"] * 100) if row["original_total"] > 0 else 0
                    logger.info(
                        f"  {row['dim_value']}: "
                        f"Clipped {row['clipped_days']:,} days, "
                        f"Amount: {row['clipped_total']:,.2f} ({pct_clipped:.1f}%), "
                        f"Avg: {row['avg_clipped_amount']:,.2f}, "
                        f"Max: {row['max_clipped_amount']:,.2f}, "
                        f"EOM: {row['clipped_eom_days']}"
                    )

            # Distribution analysis
            distribution_stats = (
                df.filter(F.col("is_clipped") == 1)
                .select(F.abs("clipped_amount").alias("amount"))
                .agg(
                    F.min("amount").alias("min_clipped"),
                    F.expr("percentile_approx(amount, 0.25)").alias("q1_clipped"),
                    F.expr("percentile_approx(amount, 0.50)").alias("median_clipped"),
                    F.expr("percentile_approx(amount, 0.75)").alias("q3_clipped"),
                    F.max("amount").alias("max_clipped"),
                    F.avg("amount").alias("mean_clipped"),
                    F.stddev("amount").alias("std_clipped"),
                )
                .collect()[0]
            )

            logger.info("\n" + "=" * 60)
            logger.info("CLIPPED AMOUNT DISTRIBUTION")
            logger.info("=" * 60)
            logger.info(f"Min:    {distribution_stats['min_clipped']:,.2f}")
            logger.info(f"Q1:     {distribution_stats['q1_clipped']:,.2f}")
            logger.info(f"Median: {distribution_stats['median_clipped']:,.2f}")
            logger.info(f"Q3:     {distribution_stats['q3_clipped']:,.2f}")
            logger.info(f"Max:    {distribution_stats['max_clipped']:,.2f}")
            logger.info(f"Mean:   {distribution_stats['mean_clipped']:,.2f}")
            logger.info(f"StdDev: {distribution_stats['std_clipped']:,.2f}")

            # Time-based analysis
            time_stats = (
                df.group_by("month")
                .agg(
                    F.sum("is_clipped").alias("clipped_transactions"),
                    F.sum(F.abs("clipped_amount")).alias("clipped_amount"),
                    F.count_distinct(F.when(F.col("is_clipped") == 1, F.col("dim_value"))).alias("affected_dim_values"),
                )
                .order_by("month")
            )

            # Get recent months
            recent_months = time_stats.order_by(F.col("month").desc()).limit(6).to_pandas()

            if not recent_months.empty:
                logger.info("\n" + "=" * 60)
                logger.info("RECENT 6 MONTHS CLIPPING SUMMARY")
                logger.info("=" * 60)
                for _, row in recent_months.iterrows():
                    logger.info(
                        f"  {row['month'].strftime('%Y-%m')}: "
                        f"Transactions: {row['clipped_transactions']:,}, "
                        f"Amount: {row['clipped_amount']:,.2f}, "
                        f"Dim Values: {row['affected_dim_values']}"
                    )

            # Warning for high impact scenarios
            if pct_amount_clipped > 5:
                logger.warning(f"⚠️ High clipping impact: {pct_amount_clipped:.2f}% of total amount clipped")
            if overall_stats["clipped_eom_transactions"] > 100:
                logger.warning(f"⚠️ Significant EOM impact: {overall_stats['clipped_eom_transactions']:,} EOM transactions clipped")

            # Save detailed analysis to a table if significant clipping
            if overall_stats["clipped_transactions"] > 0:
                analysis_summary = df.select(
                    "dim_value", "date", "original_amount", "amount", "is_clipped", "clipped_amount", "is_last_work_day_of_month"
                ).filter(F.col("is_clipped") == 1)

                # Save to a temporary table for further analysis if needed
                table_name = f"clipping_analysis_{date.today().strftime('%Y%m%d')}"
                analysis_summary.write.mode("overwrite").save_as_table(table_name)
                logger.info(f"\nDetailed clipping analysis saved to table: {table_name}")

        except Exception as e:
            logger.warning(f"Could not complete full clipping analysis: {str(e)}")
            logger.info("Continuing with pipeline execution...")

    @staticmethod
    @log_transformation
    def create_monthly_aggregates(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 2: Create monthly aggregations
        """
        logger.debug(f"Aggregating with pre_eom_days={config.pre_eom_days}, early_month_days={config.early_month_days}")

        agg_df = df.group_by(["dim_value", "month", "year", "month_num"]).agg(
            [
                # Total monthly amounts
                F.coalesce(F.sum("amount"), F.lit(0)).alias("monthly_total"),
                F.count(F.when(F.col("amount") != 0, 1)).alias("monthly_transactions"),
                F.coalesce(F.avg(F.when(F.col("amount") != 0, F.col("amount"))), F.lit(0)).alias("monthly_avg_amount"),
                F.coalesce(F.stddev_pop(F.when(F.col("amount") != 0, F.col("amount"))), F.lit(0)).alias("monthly_std_amount"),
                # EOM specific amounts
                F.coalesce(F.sum(F.when(F.col("is_last_work_day_of_month"), F.col("amount")).otherwise(0)), F.lit(0)).alias(
                    "eom_amount"
                ),
                F.count(F.when((F.col("is_last_work_day_of_month")) & (F.col("amount") != 0), 1)).alias("eom_transaction_count"),
                # Non-EOM amounts
                F.coalesce(F.sum(F.when(~F.col("is_last_work_day_of_month"), F.col("amount")).otherwise(0)), F.lit(0)).alias(
                    "non_eom_total"
                ),
                F.count(F.when((~F.col("is_last_work_day_of_month")) & (F.col("amount") != 0), 1)).alias("non_eom_transactions"),
                F.coalesce(
                    F.avg(F.when((~F.col("is_last_work_day_of_month")) & (F.col("amount") != 0), F.col("amount"))), F.lit(0)
                ).alias("non_eom_avg"),
                # Pre-EOM signals
                F.coalesce(
                    F.sum(F.when(F.col("days_from_eom").between(-config.pre_eom_days, -1), F.col("amount")).otherwise(0)), F.lit(0)
                ).alias("pre_eom_5d_total"),
                F.count(F.when((F.col("days_from_eom").between(-config.pre_eom_days, -1)) & (F.col("amount") != 0), 1)).alias(
                    "pre_eom_5d_count"
                ),
                # Early month signal
                F.coalesce(
                    F.sum(F.when(F.col("day_of_month") <= config.early_month_days, F.col("amount")).otherwise(0)), F.lit(0)
                ).alias("early_month_total"),
                # Mid month signal
                F.coalesce(
                    F.sum(
                        F.when(
                            F.col("day_of_month").between(config.early_month_days, config.mid_month_end_day), F.col("amount")
                        ).otherwise(0)
                    ),
                    F.lit(0),
                ).alias("mid_month_total"),
                # Maximum single transaction
                F.coalesce(F.max("amount"), F.lit(0)).alias("max_monthly_transaction"),
                # Transaction distribution within month
                F.coalesce(F.stddev_pop("day_of_month"), F.lit(0)).alias("day_dispersion"),
                # Quarter-end indicator
                F.max(F.when(F.col("month_num").isin([3, 6, 9, 12]), 1).otherwise(0)).alias("is_quarter_end"),
                # Year-end indicator
                F.max(F.when(F.col("month_num") == 12, 1).otherwise(0)).alias("is_year_end"),
                # EOM activity flag
                F.max(F.when((F.col("is_last_work_day_of_month")) & (F.col("amount") > 0), 1).otherwise(0)).alias(
                    "has_nonzero_eom"
                ),
                # Quarterly totals for seasonality detection
                F.sum(F.when(F.col("month_num").isin([3, 6, 9, 12]), F.col("amount")).otherwise(0)).alias("quarter_end_amount"),
                F.sum(F.when(F.col("month_num") == 12, F.col("amount")).otherwise(0)).alias("year_end_amount"),
            ]
        )

        return agg_df

    @staticmethod
    @log_transformation
    def calculate_rolling_features(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 3: Calculate rolling window features
        """
        logger.debug(f"Calculating rolling features with window={config.rolling_window_months} months")

        # Define window specifications
        window_12m = Window.partition_by("dim_value").order_by("month").rows_between(-12, -1)
        window_3m = Window.partition_by("dim_value").order_by("month").rows_between(-3, -1)
        window_unbounded = Window.partition_by("dim_value").order_by("month").rows_between(Window.unbounded_preceding, -1)

        # Overall volume metrics (12-month rolling)
        df = df.with_columns(
            [
                "rolling_total_volume_12m",
                F.sum("monthly_total").over(window_12m),
                "rolling_avg_monthly_volume",
                F.avg("monthly_total").over(window_12m),
                "rolling_max_transaction",
                F.max("max_monthly_transaction").over(window_12m),
                # EOM-specific metrics
                "rolling_eom_volume_12m",
                F.sum("eom_amount").over(window_12m),
                "rolling_avg_nonzero_eom",
                F.avg(F.when(F.col("eom_amount") > 0, F.col("eom_amount"))).over(window_12m),
                "rolling_max_eom",
                F.max("eom_amount").over(window_12m),
                "rolling_std_eom",
                F.stddev_pop("eom_amount").over(window_12m),
                # Non-EOM metrics
                "rolling_non_eom_volume_12m",
                F.sum("non_eom_total").over(window_12m),
                "rolling_avg_non_eom",
                F.avg("non_eom_total").over(window_12m),
                # Frequency counts
                "rolling_nonzero_eom_months",
                F.sum(F.when(F.col("eom_amount") > 0, 1).otherwise(0)).over(window_12m),
                "rolling_zero_eom_months",
                F.sum(F.when(F.col("eom_amount") == 0, 1).otherwise(0)).over(window_12m),
                "active_months_12m",
                F.sum(F.when(F.col("monthly_total") > 0, 1).otherwise(0)).over(window_12m),
                # Volatility metrics
                "rolling_std_monthly",
                F.stddev_pop("monthly_total").over(window_12m),
                # Seasonality metrics
                "rolling_quarter_end_volume",
                F.sum("quarter_end_amount").over(window_12m),
                "rolling_year_end_volume",
                F.sum("year_end_amount").over(window_12m),
                # Transaction regularity
                "rolling_avg_transactions",
                F.avg("monthly_transactions").over(window_12m),
                "rolling_std_transactions",
                F.stddev_pop("monthly_transactions").over(window_12m),
                # Day dispersion
                "rolling_avg_day_dispersion",
                F.avg("day_dispersion").over(window_12m),
                # Total counts (expanding window)
                "total_nonzero_eom_count",
                F.sum(F.when(F.col("eom_amount") > 0, 1).otherwise(0)).over(window_unbounded),
                # Lagged values
                "eom_amount_12m_ago",
                F.lag("eom_amount", 12).over(Window.partition_by("dim_value").order_by("month")),
                "eom_amount_3m_ago",
                F.lag("eom_amount", 3).over(Window.partition_by("dim_value").order_by("month")),
                "eom_amount_1m_ago",
                F.lag("eom_amount", 1).over(Window.partition_by("dim_value").order_by("month")),
                # Moving averages
                "eom_ma3",
                F.avg("eom_amount").over(window_3m),
                # Months of history
                "months_of_history",
                F.row_number().over(Window.partition_by("dim_value").order_by("month")),
            ]
        )

        # Calculate months since last EOM
        window_eom = Window.partition_by("dim_value", "has_nonzero_eom").order_by(F.col("month").desc())
        df = df.with_column(
            "months_since_last_eom", F.when(F.col("has_nonzero_eom") == 1, 0).otherwise(F.row_number().over(window_eom) - 1)
        )

        # Coalesce null values to 0
        rolling_columns = [
            "rolling_total_volume_12m",
            "rolling_avg_monthly_volume",
            "rolling_max_transaction",
            "rolling_eom_volume_12m",
            "rolling_avg_nonzero_eom",
            "rolling_max_eom",
            "rolling_std_eom",
            "rolling_non_eom_volume_12m",
            "rolling_avg_non_eom",
            "rolling_nonzero_eom_months",
            "rolling_zero_eom_months",
            "active_months_12m",
            "rolling_std_monthly",
            "rolling_quarter_end_volume",
            "rolling_year_end_volume",
            "rolling_avg_transactions",
            "rolling_std_transactions",
            "rolling_avg_day_dispersion",
            "total_nonzero_eom_count",
        ]

        for col in rolling_columns:
            df = df.with_column(col, F.coalesce(F.col(col), F.lit(0)))

        return df

    @staticmethod
    @log_transformation
    def calculate_portfolio_metrics(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 4: Calculate portfolio-level metrics for relative importance
        """
        logger.debug(f"Calculating portfolio metrics with min_months_history={config.min_months_history}")

        # Filter for sufficient history
        df = df.filter(F.col("months_of_history") >= config.min_months_history)

        # Calculate portfolio totals per month
        window_portfolio = Window.partition_by("month")

        df = df.with_columns(
            [
                "total_portfolio_volume",
                F.sum("rolling_total_volume_12m").over(window_portfolio),
                "total_portfolio_eom_volume",
                F.sum("rolling_eom_volume_12m").over(window_portfolio),
            ]
        )

        # Calculate cumulative portfolio percentages
        window_cumulative_overall = (
            Window.partition_by("month")
            .order_by(F.col("rolling_total_volume_12m").desc())
            .rows_between(Window.unbounded_preceding, Window.current_row)
        )
        window_cumulative_eom = (
            Window.partition_by("month")
            .order_by(F.col("rolling_eom_volume_12m").desc())
            .rows_between(Window.unbounded_preceding, Window.current_row)
        )

        df = df.with_columns(
            [
                "cumulative_overall_portfolio_pct",
                F.sum("rolling_total_volume_12m").over(window_cumulative_overall)
                / F.nullif(F.col("total_portfolio_volume"), F.lit(0)),
                "cumulative_eom_portfolio_pct",
                F.sum("rolling_eom_volume_12m").over(window_cumulative_eom)
                / F.nullif(F.col("total_portfolio_eom_volume"), F.lit(0)),
            ]
        )

        # Handle nulls
        df = df.with_columns(
            [
                "cumulative_overall_portfolio_pct",
                F.coalesce(F.col("cumulative_overall_portfolio_pct"), F.lit(0)),
                "cumulative_eom_portfolio_pct",
                F.coalesce(F.col("cumulative_eom_portfolio_pct"), F.lit(0)),
            ]
        )

        return df

    @staticmethod
    @log_transformation
    def calculate_pattern_metrics(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 5: Calculate pattern metrics for classification
        """
        logger.debug("Calculating pattern metrics for behavioral classification")

        # EOM concentration
        df = df.with_column(
            "eom_concentration",
            F.when(
                F.col("rolling_total_volume_12m") > 0, F.col("rolling_eom_volume_12m") / F.col("rolling_total_volume_12m")
            ).otherwise(0),
        )

        # EOM predictability
        df = df.with_column(
            "eom_predictability",
            F.when(
                F.col("rolling_avg_nonzero_eom") > 0,
                F.greatest(F.lit(0), 1 - (F.col("rolling_std_eom") / F.col("rolling_avg_nonzero_eom"))),
            ).otherwise(0),
        )

        # EOM frequency
        df = df.with_column(
            "eom_frequency",
            F.when(
                F.col("months_of_history") > 1,
                F.coalesce(F.col("rolling_nonzero_eom_months") / F.least(F.col("months_of_history") - 1, F.lit(12)), F.lit(0)),
            ).otherwise(0),
        )

        # EOM zero ratio
        df = df.with_column(
            "eom_zero_ratio",
            F.when(
                F.col("months_of_history") > 1,
                F.coalesce(F.col("rolling_zero_eom_months") / F.least(F.col("months_of_history") - 1, F.lit(12)), F.lit(0)),
            ).otherwise(0),
        )

        # EOM spike ratio
        df = df.with_column(
            "eom_spike_ratio",
            F.when(
                (F.col("rolling_avg_non_eom") > 0) & (F.col("rolling_avg_nonzero_eom") > 0),
                F.col("rolling_avg_nonzero_eom") / F.col("rolling_avg_non_eom"),
            )
            .when((F.col("rolling_avg_nonzero_eom") > 0) & (F.col("rolling_avg_non_eom") == 0), 999)
            .otherwise(1),
        )

        # EOM CV
        df = df.with_column(
            "eom_cv",
            F.when(
                F.col("rolling_avg_nonzero_eom") > 0,
                F.col("rolling_std_eom") / F.nullif(F.col("rolling_avg_nonzero_eom"), F.lit(0)),
            ).otherwise(0),
        )

        # Monthly CV
        df = df.with_column(
            "monthly_cv",
            F.when(
                F.col("rolling_avg_monthly_volume") > 0, F.col("rolling_std_monthly") / F.col("rolling_avg_monthly_volume")
            ).otherwise(0),
        )

        # Transaction regularity
        df = df.with_column(
            "transaction_regularity",
            F.when(
                F.col("rolling_avg_transactions") > 0,
                F.greatest(F.lit(0), 1 - (F.col("rolling_std_transactions") / F.col("rolling_avg_transactions"))),
            ).otherwise(0),
        )

        # Activity rate
        df = df.with_column(
            "activity_rate",
            F.when(
                F.col("months_of_history") > 0, F.col("active_months_12m") / F.least(F.col("months_of_history"), F.lit(12))
            ).otherwise(0),
        )

        # Quarter end concentration
        df = df.with_column(
            "quarter_end_concentration",
            F.when(
                F.col("rolling_total_volume_12m") > 0, F.col("rolling_quarter_end_volume") / F.col("rolling_total_volume_12m")
            ).otherwise(0),
        )

        # Year end concentration
        df = df.with_column(
            "year_end_concentration",
            F.when(
                F.col("rolling_total_volume_12m") > 0, F.col("rolling_year_end_volume") / F.col("rolling_total_volume_12m")
            ).otherwise(0),
        )

        # Transaction dispersion
        df = df.with_column(
            "transaction_dispersion",
            F.when(F.col("rolling_avg_day_dispersion") > 0, F.col("rolling_avg_day_dispersion")).otherwise(0),
        )

        # Has EOM history flag
        df = df.with_column("has_eom_history", F.when(F.col("total_nonzero_eom_count") > 0, 1).otherwise(0))

        # Months inactive
        window_inactive = Window.partition_by("dim_value", F.col("monthly_total") > 0).order_by(F.col("month").desc())
        df = df.with_column(
            "months_inactive", F.when(F.col("monthly_total") > 0, 0).otherwise(F.row_number().over(window_inactive) - 1)
        )

        # EOM periodicity detection
        df = df.with_column(
            "eom_periodicity",
            F.when(
                (F.col("total_nonzero_eom_count") >= 3) & (F.col("months_of_history") >= 12),
                F.when(F.col("rolling_nonzero_eom_months") >= 10, "MONTHLY")
                .when(
                    (F.col("total_nonzero_eom_count") % 3 == 0)
                    & (F.col("total_nonzero_eom_count") >= F.floor(F.col("months_of_history") / 4)),
                    "QUARTERLY",
                )
                .when(
                    (F.col("total_nonzero_eom_count") % 6 == 0)
                    & (F.col("total_nonzero_eom_count") >= F.floor(F.col("months_of_history") / 7)),
                    "SEMIANNUAL",
                )
                .when(F.col("total_nonzero_eom_count") == F.floor(F.col("months_of_history") / 12), "ANNUAL")
                .otherwise("IRREGULAR"),
            ).otherwise("INSUFFICIENT_DATA"),
        )

        return df

    @staticmethod
    @log_transformation
    def classify_importance(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 6: Dual importance classification (overall vs EOM)
        """
        logger.debug(f"Classifying importance with overall_critical_percentile={config.overall_critical_percentile}")

        # Overall importance tier
        df = df.with_column(
            "overall_importance_tier",
            F.when(
                (F.col("cumulative_overall_portfolio_pct") <= config.overall_critical_percentile)
                | (F.col("rolling_avg_monthly_volume") >= config.critical_monthly_avg_threshold)
                | (F.col("rolling_max_transaction") >= config.critical_max_transaction_threshold),
                "CRITICAL",
            )
            .when(
                (F.col("cumulative_overall_portfolio_pct") <= config.overall_high_percentile)
                | (F.col("rolling_avg_monthly_volume") >= config.high_monthly_avg_threshold)
                | (F.col("rolling_max_transaction") >= config.high_max_transaction_threshold),
                "HIGH",
            )
            .when(
                (F.col("cumulative_overall_portfolio_pct") <= config.overall_medium_percentile)
                | (F.col("rolling_avg_monthly_volume") >= config.medium_monthly_avg_threshold)
                | (F.col("rolling_max_transaction") >= config.medium_max_transaction_threshold),
                "MEDIUM",
            )
            .otherwise("LOW"),
        )

        # EOM importance tier
        df = df.with_column(
            "eom_importance_tier",
            F.when(
                (F.col("cumulative_eom_portfolio_pct") <= config.eom_critical_percentile)
                | (F.col("rolling_avg_nonzero_eom") >= config.critical_eom_monthly_threshold)
                | (F.col("rolling_max_eom") >= config.critical_max_eom_threshold),
                "CRITICAL",
            )
            .when(
                (F.col("cumulative_eom_portfolio_pct") <= config.eom_high_percentile)
                | (F.col("rolling_avg_nonzero_eom") >= config.high_eom_monthly_threshold)
                | (F.col("rolling_max_eom") >= config.high_max_eom_threshold),
                "HIGH",
            )
            .when(
                (F.col("cumulative_eom_portfolio_pct") <= config.eom_medium_percentile)
                | (
                    (F.col("rolling_avg_nonzero_eom") >= config.medium_eom_monthly_threshold)
                    & (F.col("total_nonzero_eom_count") >= 3)
                )
                | (F.col("rolling_max_eom") >= config.medium_max_eom_threshold),
                "MEDIUM",
            )
            .when((F.col("rolling_eom_volume_12m") > 0) | (F.col("total_nonzero_eom_count") > 0), "LOW")
            .otherwise("NONE"),
        )

        # Importance scores
        df = df.with_column(
            "overall_importance_score",
            F.when(
                F.col("total_portfolio_volume") > 0, F.col("rolling_total_volume_12m") / F.col("total_portfolio_volume")
            ).otherwise(0),
        )

        df = df.with_column(
            "eom_importance_score",
            F.when(
                F.col("total_portfolio_eom_volume") > 0, F.col("rolling_eom_volume_12m") / F.col("total_portfolio_eom_volume")
            ).otherwise(0),
        )

        # EOM risk flag
        df = df.with_column(
            "eom_risk_flag",
            F.when(
                (F.col("rolling_total_volume_12m") >= config.eom_risk_volume_threshold)
                & (F.col("has_eom_history") == 0)
                & (F.col("months_of_history") >= config.eom_risk_min_months),
                1,
            ).otherwise(0),
        )

        return df

    @staticmethod
    @log_transformation
    def calculate_eom_smooth_scores(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 7a: Calculate smooth scores for EOM patterns
        """
        logger.debug("Calculating smooth EOM pattern scores")

        # Regularity score: Sigmoid function for smooth transition
        df = df.with_column("regularity_score", 100 * (1 / (1 + F.exp(-config.sigmoid_steepness * (F.col("eom_frequency") - 0.5)))))

        # Stability score: Inverse exponential decay based on CV
        df = df.with_column("stability_score", 100 * F.exp(-config.stability_decay_rate * F.greatest(F.col("eom_cv"), F.lit(0))))

        # Recency score: Exponential time decay
        df = df.with_column(
            "recency_score",
            F.when(F.col("has_eom_history") == 0, 0)
            .when(F.col("has_nonzero_eom") == 1, 100)
            .when(F.col("months_since_last_eom") == 1, 80)
            .when(F.col("months_since_last_eom") == 2, 64)
            .when(F.col("months_since_last_eom") == 3, 51)
            .otherwise(
                100
                * F.pow(F.lit(config.recency_decay_rate), F.greatest(F.lit(4), F.least(F.lit(24), F.col("months_since_last_eom"))))
            ),
        )

        # Concentration score: Logistic curve
        df = df.with_column(
            "concentration_score", 100 * (1 / (1 + F.exp(-config.concentration_steepness * (F.col("eom_concentration") - 0.5))))
        )

        # Volume importance score: Asymptotic growth
        df = df.with_column(
            "volume_importance_score",
            F.when(
                F.col("total_portfolio_eom_volume") > 0,
                100 * (1 - F.exp(-config.volume_growth_rate * F.col("eom_importance_score"))),
            ).otherwise(0),
        )

        return df

    @staticmethod
    @log_transformation
    def calculate_pattern_distances(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 7b: Calculate distances to pattern archetypes
        """
        logger.debug("Calculating distances to EOM pattern archetypes")

        # CONTINUOUS_STABLE: high regularity (90), high stability (80), medium recency (50)
        df = df.with_column(
            "dist_continuous_stable",
            F.sqrt(
                F.pow(90 - F.col("regularity_score"), 2)
                + F.pow(80 - F.col("stability_score"), 2)
                + F.pow(50 - F.col("recency_score"), 2) * 0.5
            ),
        )

        # CONTINUOUS_VOLATILE: high regularity (90), low stability (20), medium recency (50)
        df = df.with_column(
            "dist_continuous_volatile",
            F.sqrt(
                F.pow(90 - F.col("regularity_score"), 2)
                + F.pow(20 - F.col("stability_score"), 2)
                + F.pow(50 - F.col("recency_score"), 2) * 0.5
            ),
        )

        # INTERMITTENT_ACTIVE: medium regularity (50), medium stability (50), high recency (90)
        df = df.with_column(
            "dist_intermittent_active",
            F.sqrt(
                F.pow(50 - F.col("regularity_score"), 2)
                + F.pow(50 - F.col("stability_score"), 2)
                + F.pow(90 - F.col("recency_score"), 2)
            ),
        )

        # INTERMITTENT_DORMANT: medium regularity (50), medium stability (50), low recency (20)
        df = df.with_column(
            "dist_intermittent_dormant",
            F.sqrt(
                F.pow(50 - F.col("regularity_score"), 2)
                + F.pow(50 - F.col("stability_score"), 2)
                + F.pow(20 - F.col("recency_score"), 2)
            ),
        )

        # RARE_RECENT: low regularity (15), any stability (50), high recency (85)
        df = df.with_column(
            "dist_rare_recent",
            F.sqrt(
                F.pow(15 - F.col("regularity_score"), 2)
                + F.pow(50 - F.col("stability_score"), 2) * 0.3
                + F.pow(85 - F.col("recency_score"), 2)
            ),
        )

        # RARE_STALE: low regularity (15), any stability (50), low recency (15)
        df = df.with_column(
            "dist_rare_stale",
            F.sqrt(
                F.pow(15 - F.col("regularity_score"), 2)
                + F.pow(50 - F.col("stability_score"), 2) * 0.3
                + F.pow(15 - F.col("recency_score"), 2)
            ),
        )

        # NO_EOM: all zeros
        df = df.with_column(
            "dist_no_eom",
            F.sqrt(
                F.pow(0 - F.col("regularity_score"), 2)
                + F.pow(50 - F.col("stability_score"), 2)
                + F.pow(0 - F.col("recency_score"), 2)
            ),
        )

        # EMERGING: special case for new series
        df = df.with_column("dist_emerging", F.when(F.col("months_of_history") <= 3, 0).otherwise(999))

        return df

    @staticmethod
    @log_transformation
    def calculate_pattern_probabilities(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 7c: Convert distances to probabilities using softmax
        """
        logger.debug(f"Converting distances to probabilities with temperature={config.pattern_temperature}")

        # Calculate softmax denominator
        df = df.with_column(
            "softmax_denominator",
            F.exp(-F.col("dist_continuous_stable") / config.pattern_temperature)
            + F.exp(-F.col("dist_continuous_volatile") / config.pattern_temperature)
            + F.exp(-F.col("dist_intermittent_active") / config.pattern_temperature)
            + F.exp(-F.col("dist_intermittent_dormant") / config.pattern_temperature)
            + F.exp(-F.col("dist_rare_recent") / config.pattern_temperature)
            + F.exp(-F.col("dist_rare_stale") / config.pattern_temperature)
            + F.when(F.col("has_eom_history") == 0, F.exp(-F.col("dist_no_eom") / config.pattern_temperature)).otherwise(0)
            + F.when(F.col("months_of_history") <= 3, F.exp(-F.col("dist_emerging") / config.pattern_temperature)).otherwise(0),
        )

        # Calculate probabilities for each pattern
        df = df.with_column(
            "prob_continuous_stable",
            F.when(F.col("months_of_history") <= 3, 0).otherwise(
                F.exp(-F.col("dist_continuous_stable") / config.pattern_temperature) / F.col("softmax_denominator")
            ),
        )

        df = df.with_column(
            "prob_continuous_volatile",
            F.when(F.col("months_of_history") <= 3, 0).otherwise(
                F.exp(-F.col("dist_continuous_volatile") / config.pattern_temperature) / F.col("softmax_denominator")
            ),
        )

        df = df.with_column(
            "prob_intermittent_active",
            F.when(F.col("months_of_history") <= 3, 0).otherwise(
                F.exp(-F.col("dist_intermittent_active") / config.pattern_temperature) / F.col("softmax_denominator")
            ),
        )

        df = df.with_column(
            "prob_intermittent_dormant",
            F.when(F.col("months_of_history") <= 3, 0).otherwise(
                F.exp(-F.col("dist_intermittent_dormant") / config.pattern_temperature) / F.col("softmax_denominator")
            ),
        )

        df = df.with_column(
            "prob_rare_recent",
            F.when(F.col("months_of_history") <= 3, 0).otherwise(
                F.exp(-F.col("dist_rare_recent") / config.pattern_temperature) / F.col("softmax_denominator")
            ),
        )

        df = df.with_column(
            "prob_rare_stale",
            F.when(F.col("months_of_history") <= 3, 0).otherwise(
                F.exp(-F.col("dist_rare_stale") / config.pattern_temperature) / F.col("softmax_denominator")
            ),
        )

        df = df.with_column(
            "prob_no_eom",
            F.when(
                F.col("has_eom_history") == 0,
                F.exp(-F.col("dist_no_eom") / config.pattern_temperature) / F.col("softmax_denominator"),
            ).otherwise(0),
        )

        df = df.with_column(
            "prob_emerging",
            F.when(
                F.col("months_of_history") <= 3,
                F.exp(-F.col("dist_emerging") / config.pattern_temperature) / F.col("softmax_denominator"),
            ).otherwise(0),
        )

        return df

    @staticmethod
    @log_transformation
    def classify_eom_patterns(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 7d: Final EOM pattern classification
        """
        logger.debug("Classifying final EOM patterns")

        # Find primary classification (highest probability)
        df = df.with_column(
            "eom_pattern",
            F.when(F.col("months_of_history") <= 3, "EMERGING")
            .when((F.col("has_eom_history") == 0) & (F.col("months_of_history") >= 6), "NO_EOM")
            .otherwise(
                F.when(
                    F.greatest(
                        F.col("prob_continuous_stable"),
                        F.col("prob_continuous_volatile"),
                        F.col("prob_intermittent_active"),
                        F.col("prob_intermittent_dormant"),
                        F.col("prob_rare_recent"),
                        F.col("prob_rare_stale"),
                    )
                    == F.col("prob_continuous_stable"),
                    "CONTINUOUS_STABLE",
                )
                .when(
                    F.greatest(
                        F.col("prob_continuous_stable"),
                        F.col("prob_continuous_volatile"),
                        F.col("prob_intermittent_active"),
                        F.col("prob_intermittent_dormant"),
                        F.col("prob_rare_recent"),
                        F.col("prob_rare_stale"),
                    )
                    == F.col("prob_continuous_volatile"),
                    "CONTINUOUS_VOLATILE",
                )
                .when(
                    F.greatest(
                        F.col("prob_continuous_stable"),
                        F.col("prob_continuous_volatile"),
                        F.col("prob_intermittent_active"),
                        F.col("prob_intermittent_dormant"),
                        F.col("prob_rare_recent"),
                        F.col("prob_rare_stale"),
                    )
                    == F.col("prob_intermittent_active"),
                    "INTERMITTENT_ACTIVE",
                )
                .when(
                    F.greatest(
                        F.col("prob_continuous_stable"),
                        F.col("prob_continuous_volatile"),
                        F.col("prob_intermittent_active"),
                        F.col("prob_intermittent_dormant"),
                        F.col("prob_rare_recent"),
                        F.col("prob_rare_stale"),
                    )
                    == F.col("prob_intermittent_dormant"),
                    "INTERMITTENT_DORMANT",
                )
                .when(
                    F.greatest(
                        F.col("prob_continuous_stable"),
                        F.col("prob_continuous_volatile"),
                        F.col("prob_intermittent_active"),
                        F.col("prob_intermittent_dormant"),
                        F.col("prob_rare_recent"),
                        F.col("prob_rare_stale"),
                    )
                    == F.col("prob_rare_recent"),
                    "RARE_RECENT",
                )
                .otherwise("RARE_STALE")
            ),
        )

        # Pattern confidence
        df = df.with_column(
            "eom_pattern_confidence",
            F.greatest(
                F.when((F.col("months_of_history") <= 3) | (F.col("has_eom_history") == 0), 1.0).otherwise(0),
                F.col("prob_continuous_stable"),
                F.col("prob_continuous_volatile"),
                F.col("prob_intermittent_active"),
                F.col("prob_intermittent_dormant"),
                F.col("prob_rare_recent"),
                F.col("prob_rare_stale"),
            ),
        )

        # Classification entropy (uncertainty)
        df = df.with_column(
            "classification_entropy",
            -(
                F.coalesce(F.col("prob_continuous_stable") * F.log(F.nullif(F.col("prob_continuous_stable"), F.lit(0))), F.lit(0))
                + F.coalesce(
                    F.col("prob_continuous_volatile") * F.log(F.nullif(F.col("prob_continuous_volatile"), F.lit(0))), F.lit(0)
                )
                + F.coalesce(
                    F.col("prob_intermittent_active") * F.log(F.nullif(F.col("prob_intermittent_active"), F.lit(0))), F.lit(0)
                )
                + F.coalesce(
                    F.col("prob_intermittent_dormant") * F.log(F.nullif(F.col("prob_intermittent_dormant"), F.lit(0))), F.lit(0)
                )
                + F.coalesce(F.col("prob_rare_recent") * F.log(F.nullif(F.col("prob_rare_recent"), F.lit(0))), F.lit(0))
                + F.coalesce(F.col("prob_rare_stale") * F.log(F.nullif(F.col("prob_rare_stale"), F.lit(0))), F.lit(0))
            ),
        )

        # High risk flag
        df = df.with_column(
            "eom_high_risk_flag",
            F.when(
                (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]))
                & (F.col("stability_score") < 30)
                & (F.col("concentration_score") >= 50),
                1,
            ).otherwise(0),
        )

        return df

    @staticmethod
    @log_transformation
    def classify_general_patterns(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 8: General timeseries pattern classification
        """
        logger.debug("Classifying general timeseries patterns")

        df = df.with_column(
            "general_pattern",
            F.when(F.col("months_inactive") >= config.inactive_months, "INACTIVE")
            .when(F.col("months_of_history") <= config.emerging_months, "EMERGING")
            .when(
                (F.col("quarter_end_concentration") >= config.ts_seasonal_concentration_threshold)
                | (F.col("year_end_concentration") >= 0.5),
                "HIGHLY_SEASONAL",
            )
            .when((F.col("activity_rate") <= config.ts_intermittent_threshold) | (F.col("active_months_12m") <= 4), "INTERMITTENT")
            .when(
                (F.col("monthly_cv") >= config.ts_high_volatility_threshold)
                & (F.col("transaction_regularity") >= config.ts_medium_regularity_threshold),
                "VOLATILE",
            )
            .when(
                (F.col("monthly_cv") >= config.ts_medium_volatility_threshold)
                & (F.col("transaction_regularity") >= config.ts_medium_regularity_threshold),
                "MODERATELY_VOLATILE",
            )
            .when(
                (F.col("monthly_cv") < config.ts_medium_volatility_threshold)
                & (F.col("transaction_regularity") >= config.ts_high_regularity_threshold),
                "STABLE",
            )
            .when(
                (F.col("transaction_dispersion") < 5) & (F.col("transaction_regularity") >= config.ts_medium_regularity_threshold),
                "CONCENTRATED",
            )
            .when(
                (F.col("transaction_dispersion") >= 8) & (F.col("transaction_regularity") >= config.ts_medium_regularity_threshold),
                "DISTRIBUTED",
            )
            .otherwise("MIXED"),
        )

        return df

    @staticmethod
    @log_transformation
    def create_final_classification(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 9: Combined priority and recommendations
        """
        logger.debug("Creating final classification and recommendations")

        # Combined priority (1-10 scale)
        df = df.with_column(
            "combined_priority",
            F.when(
                ((F.col("overall_importance_tier") == "CRITICAL") | (F.col("eom_importance_tier") == "CRITICAL"))
                & (
                    (F.col("eom_pattern").isin(["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]))
                    | (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))
                ),
                10,
            )
            .when(
                ((F.col("overall_importance_tier") == "CRITICAL") | (F.col("eom_importance_tier") == "CRITICAL"))
                & ((F.col("eom_pattern") == "SEASONAL_EOM") | (F.col("general_pattern") == "HIGHLY_SEASONAL")),
                9,
            )
            .when((F.col("overall_importance_tier") == "CRITICAL") | (F.col("eom_importance_tier") == "CRITICAL"), 8)
            .when(
                ((F.col("overall_importance_tier") == "HIGH") | (F.col("eom_importance_tier") == "HIGH"))
                & (
                    (F.col("eom_pattern").isin(["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]))
                    | (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))
                ),
                7,
            )
            .when((F.col("overall_importance_tier") == "HIGH") | (F.col("eom_importance_tier") == "HIGH"), 6)
            .when(
                (F.col("eom_risk_flag") == 1)
                | ((F.col("overall_importance_tier") == "MEDIUM") & (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))),
                5,
            )
            .when((F.col("overall_importance_tier") == "MEDIUM") | (F.col("eom_importance_tier") == "MEDIUM"), 4)
            .when(~F.col("general_pattern").isin(["INACTIVE", "EMERGING"]), 3)
            .when((F.col("general_pattern") == "EMERGING") | (F.col("eom_pattern") == "EMERGING"), 2)
            .otherwise(1),
        )

        # Recommended forecasting method
        df = df.with_column(
            "recommended_method",
            F.when(F.col("eom_pattern") == "NO_EOM", "Zero_EOM_Forecast")
            .when(
                F.col("eom_pattern").isin(["CONTINUOUS_STABLE", "CONTINUOUS_VOLATILE"]),
                F.when(
                    (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"])) & (F.col("eom_pattern") == "CONTINUOUS_VOLATILE"),
                    "XGBoost_EOM_Focus",
                )
                .when(F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]), "SARIMA_EOM")
                .otherwise("Simple_MA_EOM"),
            )
            .when(
                (F.col("eom_pattern") == "INTERMITTENT_ACTIVE") | (F.col("general_pattern") == "INTERMITTENT"),
                F.when(F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"]), "Croston_Method").otherwise(
                    "Zero_Inflated_Model"
                ),
            )
            .when(
                F.col("general_pattern") == "HIGHLY_SEASONAL",
                F.when(F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"]), "Seasonal_Decomposition").otherwise(
                    "Seasonal_Naive"
                ),
            )
            .when(F.col("general_pattern") == "VOLATILE", "XGBoost_Full_Series")
            .when(F.col("general_pattern") == "STABLE", "Linear_Trend")
            .when(F.col("general_pattern") == "INACTIVE", "Zero_Forecast")
            .when(F.col("general_pattern") == "EMERGING", "Conservative_MA")
            .otherwise("Historical_Average"),
        )

        # Forecast complexity
        df = df.with_column(
            "forecast_complexity",
            F.when(
                (
                    (F.col("eom_pattern").isin(["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]))
                    | (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))
                )
                & (
                    (F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"]))
                    | (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]))
                ),
                5,
            )
            .when(
                (F.col("general_pattern") == "HIGHLY_SEASONAL") & (F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"])), 4
            )
            .when(
                (F.col("eom_pattern").isin(["CONTINUOUS_STABLE"]))
                | (F.col("general_pattern") == "STABLE")
                | (F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"])),
                3,
            )
            .when((F.col("overall_importance_tier") == "MEDIUM") | (F.col("eom_importance_tier") == "MEDIUM"), 2)
            .otherwise(1),
        )

        # Segment names
        df = df.with_column(
            "full_segment_name",
            F.concat(
                F.col("overall_importance_tier"),
                F.lit("_"),
                F.col("general_pattern"),
                F.lit("__"),
                F.col("eom_importance_tier"),
                F.lit("EOM_"),
                F.col("eom_pattern"),
            ),
        )

        df = df.with_column(
            "segment_name",
            F.concat(
                F.when(
                    F.col("overall_importance_tier") == F.col("eom_importance_tier"), F.col("overall_importance_tier")
                ).otherwise(F.concat(F.col("overall_importance_tier"), F.lit("/"), F.col("eom_importance_tier"), F.lit("EOM"))),
                F.lit("_"),
                F.col("general_pattern"),
                F.lit("_"),
                F.col("eom_pattern"),
            ),
        )

        return df

    @staticmethod
    @log_transformation
    def calculate_growth_metrics(config: SegmentationConfig, df: DataFrame) -> DataFrame:
        """
        Step 10: Calculate growth metrics
        """
        logger.debug("Calculating growth metrics")

        # Year-over-year growth
        df = df.with_column(
            "eom_yoy_growth",
            F.when(
                F.col("eom_amount_12m_ago") > 0, (F.col("eom_amount") - F.col("eom_amount_12m_ago")) / F.col("eom_amount_12m_ago")
            )
            .when((F.col("eom_amount_12m_ago") == 0) & (F.col("eom_amount") > 0), 999.99)
            .otherwise(0),
        )

        # Month-over-month growth
        window_mom = Window.partition_by("dim_value").order_by("month")
        df = df.with_column(
            "eom_mom_growth",
            F.when(
                F.lag("eom_amount", 1).over(window_mom) > 0,
                (F.col("eom_amount") - F.lag("eom_amount", 1).over(window_mom)) / F.lag("eom_amount", 1).over(window_mom),
            ).otherwise(0),
        )

        return df

    @staticmethod
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
