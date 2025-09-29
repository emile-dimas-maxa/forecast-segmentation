"""
General timeseries pattern classification
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from loguru import logger

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.utils import log_transformation


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
