"""
Rolling window feature calculations
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from loguru import logger

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def calculate_rolling_features(
    df: DataFrame,
    rolling_window_months: int = 12,
    ma_window_short: int = 3,
) -> DataFrame:
    """
    Step 3: Calculate rolling window features

    Args:
        df: Input DataFrame
        rolling_window_months: Rolling window for feature calculation
        ma_window_short: Short-term moving average window (months)
    """
    logger.debug(f"Calculating rolling features with window={rolling_window_months} months")

    # Define window specifications
    window_12m = Window.partition_by("dim_value").order_by("month").rows_between(-rolling_window_months, -1)
    window_3m = Window.partition_by("dim_value").order_by("month").rows_between(-ma_window_short, -1)
    window_unbounded = Window.partition_by("dim_value").order_by("month").rows_between(Window.UNBOUNDED_PRECEDING, -1)

    # Overall volume metrics (12-month rolling)
    df = df.with_columns(
        [
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
            "eom_amount_12m_ago",
            "eom_amount_3m_ago",
            "eom_amount_1m_ago",
            "eom_ma3",
            "months_of_history",
        ],
        [
            F.sum("monthly_total").over(window_12m),
            F.avg("monthly_total").over(window_12m),
            F.max("max_monthly_transaction").over(window_12m),
            F.sum("eom_amount").over(window_12m),
            F.avg(F.when(F.col("eom_amount") > 0, F.col("eom_amount"))).over(window_12m),
            F.max("eom_amount").over(window_12m),
            F.stddev_pop("eom_amount").over(window_12m),
            F.sum("non_eom_total").over(window_12m),
            F.avg("non_eom_total").over(window_12m),
            F.sum(F.when(F.col("eom_amount") > 0, 1).otherwise(0)).over(window_12m),
            F.sum(F.when(F.col("eom_amount") == 0, 1).otherwise(0)).over(window_12m),
            F.sum(F.when(F.col("monthly_total") > 0, 1).otherwise(0)).over(window_12m),
            F.stddev_pop("monthly_total").over(window_12m),
            F.sum("quarter_end_amount").over(window_12m),
            F.sum("year_end_amount").over(window_12m),
            F.avg("monthly_transactions").over(window_12m),
            F.stddev_pop("monthly_transactions").over(window_12m),
            F.avg("day_dispersion").over(window_12m),
            F.sum(F.when(F.col("eom_amount") > 0, 1).otherwise(0)).over(window_unbounded),
            F.lag("eom_amount", 12).over(Window.partition_by("dim_value").order_by("month")),
            F.lag("eom_amount", 3).over(Window.partition_by("dim_value").order_by("month")),
            F.lag("eom_amount", 1).over(Window.partition_by("dim_value").order_by("month")),
            F.avg("eom_amount").over(window_3m),
            F.row_number().over(Window.partition_by("dim_value").order_by("month")),
        ],
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
