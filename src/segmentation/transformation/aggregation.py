"""
Monthly aggregation and EOM clipping functions
"""

from datetime import date

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def create_monthly_aggregates(df: DataFrame) -> DataFrame:
    """
    Step 2: Create monthly aggregations
    """
    logger.debug("Creating monthly aggregations")

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
            # Pre-EOM signals (using default 5 days)
            F.coalesce(F.sum(F.when(F.col("days_from_eom").between(-5, -1), F.col("amount")).otherwise(0)), F.lit(0)).alias(
                "pre_eom_5d_total"
            ),
            F.count(F.when((F.col("days_from_eom").between(-5, -1)) & (F.col("amount") != 0), 1)).alias("pre_eom_5d_count"),
            # Early month signal (using default 10 days)
            F.coalesce(F.sum(F.when(F.col("day_of_month") <= 10, F.col("amount")).otherwise(0)), F.lit(0)).alias(
                "early_month_total"
            ),
            # Mid month signal (using default 10-20 days)
            F.coalesce(
                F.sum(F.when(F.col("day_of_month").between(10, 20), F.col("amount")).otherwise(0)),
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
            F.max(F.when((F.col("is_last_work_day_of_month")) & (F.col("amount") > 0), 1).otherwise(0)).alias("has_nonzero_eom"),
            # Quarterly totals for seasonality detection
            F.sum(F.when(F.col("month_num").isin([3, 6, 9, 12]), F.col("amount")).otherwise(0)).alias("quarter_end_amount"),
            F.sum(F.when(F.col("month_num") == 12, F.col("amount")).otherwise(0)).alias("year_end_amount"),
        ]
    )

    return agg_df


@log_transformation
def apply_eom_clipping(df: DataFrame, daily_amount_clip_threshold: float) -> DataFrame:
    """
    Step 2.5: Apply clipping to EOM amounts below threshold
    Clips small EOM amounts to zero to reduce noise
    This is applied after monthly aggregation to focus only on EOM amounts

    Args:
        df: Input DataFrame
        daily_amount_clip_threshold: Threshold for clipping small EOM amounts
    """
    threshold = daily_amount_clip_threshold
    logger.info(f"Applying EOM clipping with threshold: {threshold:,.2f}")

    # Store original EOM amount for analysis
    df = df.with_column("original_eom_amount", F.col("eom_amount"))

    # Create clipping indicators
    df = df.with_columns(
        [
            "is_eom_clipped",
            "clipped_eom_amount",
        ],
        [
            F.when((F.abs(F.col("eom_amount")) > 0) & (F.abs(F.col("eom_amount")) < threshold), 1).otherwise(0),
            F.when((F.abs(F.col("eom_amount")) > 0) & (F.abs(F.col("eom_amount")) < threshold), F.col("eom_amount")).otherwise(0),
        ],
    )

    # Apply clipping to EOM amount only
    df = df.with_column(
        "eom_amount",
        F.when((F.abs(F.col("eom_amount")) > 0) & (F.abs(F.col("eom_amount")) < threshold), 0).otherwise(F.col("eom_amount")),
    )

    # Perform clipping analysis
    _analyze_eom_clipping_impact(df, threshold)

    return df


def _analyze_eom_clipping_impact(df: DataFrame, threshold: float) -> None:
    """
    Analyze the impact of EOM clipping on the aggregated monthly data
    """
    logger.info("Performing detailed EOM clipping analysis...")

    try:
        # Overall EOM clipping statistics - simplified approach
        total_records = df.count()
        clipped_records = df.filter(F.col("is_eom_clipped") == 1).count()
        nonzero_eom_records = df.filter(F.col("original_eom_amount") != 0).count()
        affected_dim_values = df.filter(F.col("is_eom_clipped") == 1).select("dim_value").distinct().count()

        # Amount statistics
        amount_stats = (
            df.agg(
                F.sum(F.abs("original_eom_amount")).alias("total_original_eom_amount"),
                F.sum(F.abs("clipped_eom_amount")).alias("total_clipped_eom_amount"),
                F.sum(F.abs("eom_amount")).alias("total_after_clipping_eom_amount"),
            )
            .to_pandas()
            .rename(columns=str.lower)
            .iloc[0]
        )

        # Calculate percentages
        pct_records_clipped = (clipped_records / nonzero_eom_records * 100) if nonzero_eom_records > 0 else 0
        pct_amount_clipped = (
            (amount_stats["total_clipped_eom_amount"] / amount_stats["total_original_eom_amount"] * 100)
            if amount_stats["total_original_eom_amount"] > 0
            else 0
        )

        # Log overall statistics
        logger.info("=" * 60)
        logger.info("EOM CLIPPING ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Clipping Threshold: {threshold:,.2f}")
        logger.info(f"Total Monthly Records: {total_records:,}")
        logger.info(f"Non-zero EOM Records: {nonzero_eom_records:,}")
        logger.info(f"Clipped EOM Records: {clipped_records:,} ({pct_records_clipped:.2f}%)")
        logger.info(f"Affected Dim Values: {affected_dim_values:,}")
        logger.info(f"Total Original EOM Amount: {amount_stats['total_original_eom_amount']:,.2f}")
        logger.info(f"Total Clipped EOM Amount: {amount_stats['total_clipped_eom_amount']:,.2f} ({pct_amount_clipped:.2f}%)")
        logger.info(f"Total After Clipping: {amount_stats['total_after_clipping_eom_amount']:,.2f}")

        # Analysis by dim_value
        dim_value_stats = (
            df.group_by("dim_value")
            .agg(
                F.count("*").alias("total_months"),
                F.sum("is_eom_clipped").alias("clipped_months"),
                F.sum(F.abs("original_eom_amount")).alias("original_total"),
                F.sum(F.abs("clipped_eom_amount")).alias("clipped_total"),
                F.avg(F.when(F.col("is_eom_clipped") == 1, F.abs("clipped_eom_amount"))).alias("avg_clipped_amount"),
                F.max(F.when(F.col("is_eom_clipped") == 1, F.abs("clipped_eom_amount"))).alias("max_clipped_amount"),
            )
            .filter(F.col("clipped_months") > 0)
        )

        # Get top affected dim_values
        top_affected = dim_value_stats.order_by(F.col("clipped_total").desc()).limit(20).to_pandas().rename(columns=str.lower)

        if not top_affected.empty:
            logger.info("\n" + "=" * 60)
            logger.info("TOP 20 AFFECTED DIM_VALUES BY CLIPPED EOM AMOUNT")
            logger.info("=" * 60)
            for _, row in top_affected.iterrows():
                pct_clipped = (row["clipped_total"] / row["original_total"] * 100) if row["original_total"] > 0 else 0
                logger.info(
                    f"  {row['dim_value']}: "
                    f"Clipped {row['clipped_months']:,} months, "
                    f"Amount: {row['clipped_total']:,.2f} ({pct_clipped:.1f}%), "
                    f"Avg: {row['avg_clipped_amount']:,.2f}, "
                    f"Max: {row['max_clipped_amount']:,.2f}"
                )

        # Distribution analysis - simplified without percentiles
        distribution_stats = (
            df.filter(F.col("is_eom_clipped") == 1)
            .select(F.abs("clipped_eom_amount").alias("amount"))
            .agg(
                F.min("amount").alias("min_clipped"),
                F.max("amount").alias("max_clipped"),
                F.avg("amount").alias("mean_clipped"),
                F.stddev("amount").alias("std_clipped"),
            )
            .to_pandas()
            .rename(columns=str.lower)
            .iloc[0]
        )

        logger.info("\n" + "=" * 60)
        logger.info("CLIPPED EOM AMOUNT DISTRIBUTION")
        logger.info("=" * 60)
        logger.info(f"Min:    {distribution_stats['min_clipped']:,.2f}")
        logger.info(f"Max:    {distribution_stats['max_clipped']:,.2f}")
        logger.info(f"Mean:   {distribution_stats['mean_clipped']:,.2f}")
        logger.info(f"StdDev: {distribution_stats['std_clipped']:,.2f}")

        # Monthly EOM clipping analysis
        monthly_eom_stats = (
            df.group_by("month")
            .agg(
                F.sum("is_eom_clipped").alias("clipped_eom_records"),
                F.sum(F.abs("clipped_eom_amount")).alias("total_clipped_eom_amount"),
                F.avg(F.when(F.col("is_eom_clipped") == 1, F.abs("clipped_eom_amount"))).alias("avg_clipped_eom_amount"),
                F.min(F.when(F.col("is_eom_clipped") == 1, F.abs("clipped_eom_amount"))).alias("min_clipped_eom_amount"),
                F.max(F.when(F.col("is_eom_clipped") == 1, F.abs("clipped_eom_amount"))).alias("max_clipped_eom_amount"),
                F.count_distinct(F.when(F.col("is_eom_clipped") == 1, F.col("dim_value"))).alias("affected_dim_values"),
            )
            .filter(F.col("clipped_eom_records") > 0)
            .order_by("month")
        )

        # Get recent months with clipping
        recent_eom_months = monthly_eom_stats.order_by(F.col("month").desc()).limit(12).to_pandas().rename(columns=str.lower)

        if not recent_eom_months.empty:
            logger.info("\n" + "=" * 60)
            logger.info("RECENT MONTHS EOM CLIPPING SUMMARY")
            logger.info("=" * 60)
            for _, row in recent_eom_months.iterrows():
                logger.info(
                    f"  {row['month'].strftime('%Y-%m')}: "
                    f"Records: {int(row['clipped_eom_records']):,}, "
                    f"Total: {row['total_clipped_eom_amount']:,.2f}, "
                    f"Avg: {row['avg_clipped_eom_amount']:,.2f}, "
                    f"Min: {row['min_clipped_eom_amount']:,.2f}, "
                    f"Max: {row['max_clipped_eom_amount']:,.2f}, "
                    f"Dim Values: {int(row['affected_dim_values'])}"
                )

            # Calculate sum by month and 12-month rolling average
            monthly_sums = recent_eom_months["total_clipped_eom_amount"]
            twelve_month_avg_sum = monthly_sums.mean()

            logger.info("\n" + "=" * 60)
            logger.info("MONTHLY SUMS AND ROLLING AVERAGE")
            logger.info("=" * 60)
            logger.info("Monthly Clipped EOM Amounts:")
            for _, row in recent_eom_months.iterrows():
                logger.info(f"  {row['month'].strftime('%Y-%m')}: {row['total_clipped_eom_amount']:,.2f}")

            logger.info(f"\n12-Month Rolling Average of Monthly Sums: {twelve_month_avg_sum:,.2f}")

            # Overall monthly statistics - simplified approach
            total_amount_stats = recent_eom_months["total_clipped_eom_amount"].agg(["sum", "mean", "min", "max"])
            avg_amount_stats = recent_eom_months["avg_clipped_eom_amount"].agg(["mean", "min", "max"])
            records_stats = recent_eom_months["clipped_eom_records"].agg(["sum", "mean", "min", "max"])

            logger.info("\n" + "=" * 60)
            logger.info("MONTHLY EOM CLIPPING STATISTICS SUMMARY")
            logger.info("=" * 60)
            logger.info("Total Amount Clipped Across All Months:")
            logger.info(f"  Sum: {total_amount_stats['sum']:,.2f}")
            logger.info(f"  Monthly Avg: {total_amount_stats['mean']:,.2f}")
            logger.info(f"  Monthly Min: {total_amount_stats['min']:,.2f}")
            logger.info(f"  Monthly Max: {total_amount_stats['max']:,.2f}")

            logger.info("Average Clipped EOM Amount Per Record:")
            logger.info(f"  Overall Avg: {avg_amount_stats['mean']:,.2f}")
            logger.info(f"  Monthly Min Avg: {avg_amount_stats['min']:,.2f}")
            logger.info(f"  Monthly Max Avg: {avg_amount_stats['max']:,.2f}")

            logger.info("Clipped Records Per Month:")
            logger.info(f"  Total: {int(records_stats['sum']):,}")
            logger.info(f"  Monthly Avg: {records_stats['mean']:,.1f}")
            logger.info(f"  Monthly Min: {int(records_stats['min']):,}")
            logger.info(f"  Monthly Max: {int(records_stats['max']):,}")

            # Additional insights
            logger.info("\nTrend Analysis:")
            if len(recent_eom_months) >= 2:
                latest_month_sum = recent_eom_months.iloc[0]["total_clipped_eom_amount"]
                previous_month_sum = recent_eom_months.iloc[1]["total_clipped_eom_amount"]
                month_over_month_change = (
                    ((latest_month_sum - previous_month_sum) / previous_month_sum * 100) if previous_month_sum > 0 else 0
                )
                logger.info(f"  Month-over-Month Change: {month_over_month_change:+.1f}%")

                if latest_month_sum > twelve_month_avg_sum * 1.2:
                    logger.warning(f"  ⚠️ Latest month ({latest_month_sum:,.2f}) is 20%+ above 12-month average")
                elif latest_month_sum < twelve_month_avg_sum * 0.8:
                    logger.info(f"  ✓ Latest month ({latest_month_sum:,.2f}) is below 12-month average")

        # Warning for high impact scenarios
        if pct_amount_clipped > 5:
            logger.warning(f"⚠️ High EOM clipping impact: {pct_amount_clipped:.2f}% of total EOM amount clipped")
        if clipped_records > 100:
            logger.warning(f"⚠️ Significant EOM impact: {clipped_records:,} EOM records clipped")

        # Save detailed analysis to a table if significant clipping
        if clipped_records > 0:
            analysis_summary = df.select(
                "dim_value", "month", "original_eom_amount", "eom_amount", "is_eom_clipped", "clipped_eom_amount"
            ).filter(F.col("is_eom_clipped") == 1)

            # Save to a temporary table for further analysis if needed
            table_name = f"eom_clipping_analysis_{date.today().strftime('%Y%m%d')}"
            analysis_summary.write.mode("overwrite").save_as_table(table_name)
            logger.info(f"\nDetailed EOM clipping analysis saved to table: {table_name}")

    except Exception as e:
        logger.warning(f"Could not complete full EOM clipping analysis: {str(e)}")
        logger.info("Continuing with pipeline execution...")
