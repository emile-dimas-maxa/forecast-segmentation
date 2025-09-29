"""
Data preparation and loading functions
"""

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from loguru import logger

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.utils import log_transformation


@log_transformation
def load_source_data(session: Session, config: SegmentationConfig) -> DataFrame:
    """Load source data from Snowflake table"""
    logger.debug(f"Loading data from table: {config.source_table}")
    return session.table(config.source_table)


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
