"""
Monthly aggregation functions
"""

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def create_monthly_aggregates(
    df: DataFrame,
    pre_eom_days: int = 5,
    early_month_days: int = 10,
    mid_month_end_day: int = 20,
) -> DataFrame:
    """
    Step 2: Create monthly aggregations

    Args:
        df: Input DataFrame
        pre_eom_days: Days before EOM to consider for pre-EOM signals
        early_month_days: First N days of month for early month signal
        mid_month_end_day: End day for mid-month period
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
            # Pre-EOM signals (configurable days)
            F.coalesce(
                F.sum(F.when(F.col("days_from_eom").between(-pre_eom_days, -1), F.col("amount")).otherwise(0)), F.lit(0)
            ).alias("pre_eom_5d_total"),
            F.count(F.when((F.col("days_from_eom").between(-pre_eom_days, -1)) & (F.col("amount") != 0), 1)).alias(
                "pre_eom_5d_count"
            ),
            # Early month signal (configurable days)
            F.coalesce(F.sum(F.when(F.col("day_of_month") <= early_month_days, F.col("amount")).otherwise(0)), F.lit(0)).alias(
                "early_month_total"
            ),
            # Mid month signal (configurable range)
            F.coalesce(
                F.sum(F.when(F.col("day_of_month").between(early_month_days, mid_month_end_day), F.col("amount")).otherwise(0)),
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
