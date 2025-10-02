"""
Growth metrics calculations
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from loguru import logger

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def calculate_growth_metrics(df: DataFrame) -> DataFrame:
    """
    Step 10: Calculate growth metrics
    """
    logger.debug("Calculating growth metrics")

    # Year-over-year growth
    df = df.with_column(
        "eom_yoy_growth",
        F.when(F.col("eom_amount_12m_ago") > 0, (F.col("eom_amount") - F.col("eom_amount_12m_ago")) / F.col("eom_amount_12m_ago"))
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
