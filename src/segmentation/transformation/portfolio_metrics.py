"""
Portfolio-level metrics calculations
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from loguru import logger

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.utils import log_transformation


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
            "total_portfolio_eom_volume",
        ],
        [
            F.sum("rolling_total_volume_12m").over(window_portfolio),
            F.sum("rolling_eom_volume_12m").over(window_portfolio),
        ],
    )

    # Calculate cumulative portfolio percentages
    window_cumulative_overall = (
        Window.partition_by("month")
        .order_by(F.col("rolling_total_volume_12m").desc())
        .rows_between(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    )
    window_cumulative_eom = (
        Window.partition_by("month")
        .order_by(F.col("rolling_eom_volume_12m").desc())
        .rows_between(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    )

    df = df.with_columns(
        [
            "cumulative_overall_portfolio_pct",
            "cumulative_eom_portfolio_pct",
        ],
        [
            F.when(
                F.col("total_portfolio_volume") != 0,
                F.sum("rolling_total_volume_12m").over(window_cumulative_overall) / F.col("total_portfolio_volume"),
            ).otherwise(0),
            F.when(
                F.col("total_portfolio_eom_volume") != 0,
                F.sum("rolling_eom_volume_12m").over(window_cumulative_eom) / F.col("total_portfolio_eom_volume"),
            ).otherwise(0),
        ],
    )

    # Handle nulls
    df = df.with_columns(
        [
            "cumulative_overall_portfolio_pct",
            "cumulative_eom_portfolio_pct",
        ],
        [
            F.coalesce(F.col("cumulative_overall_portfolio_pct"), F.lit(0)),
            F.coalesce(F.col("cumulative_eom_portfolio_pct"), F.lit(0)),
        ],
    )

    return df
