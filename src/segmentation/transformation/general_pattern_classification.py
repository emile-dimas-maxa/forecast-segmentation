"""
General timeseries pattern classification
"""

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def classify_general_patterns(
    df: DataFrame,
    inactive_months: int = 3,
    emerging_months: int = 3,
    ts_seasonal_concentration_threshold: float = 0.25,
    ts_year_end_concentration_threshold: float = 0.5,
    ts_intermittent_threshold: float = 0.30,
    ts_high_volatility_threshold: float = 0.50,
    ts_medium_volatility_threshold: float = 0.25,
    ts_high_regularity_threshold: float = 0.70,
    ts_medium_regularity_threshold: float = 0.40,
) -> DataFrame:
    """
    Step 8: General timeseries pattern classification

    Args:
        df: Input DataFrame
        inactive_months: Months of inactivity for INACTIVE classification
        emerging_months: Max months for EMERGING classification
        ts_seasonal_concentration_threshold: Quarterly concentration for seasonality
        ts_year_end_concentration_threshold: Year-end concentration for seasonality
        ts_intermittent_threshold: Intermittent activity threshold
        ts_high_volatility_threshold: High volatility (CV) threshold
        ts_medium_volatility_threshold: Medium volatility (CV) threshold
        ts_high_regularity_threshold: High transaction regularity threshold
        ts_medium_regularity_threshold: Medium transaction regularity threshold
    """
    logger.debug("Classifying general timeseries patterns")

    df = df.with_column(
        "general_pattern",
        F.when(F.col("months_inactive") >= inactive_months, "INACTIVE")
        .when(F.col("months_of_history") <= emerging_months, "EMERGING")
        .when(
            (F.col("quarter_end_concentration") >= ts_seasonal_concentration_threshold)
            | (F.col("year_end_concentration") >= ts_year_end_concentration_threshold),
            "HIGHLY_SEASONAL",
        )
        .when((F.col("activity_rate") <= ts_intermittent_threshold) | (F.col("active_months_12m") <= 4), "INTERMITTENT")
        .when(
            (F.col("monthly_cv") >= ts_high_volatility_threshold)
            & (F.col("transaction_regularity") >= ts_medium_regularity_threshold),
            "VOLATILE",
        )
        .when(
            (F.col("monthly_cv") >= ts_medium_volatility_threshold)
            & (F.col("transaction_regularity") >= ts_medium_regularity_threshold),
            "MODERATELY_VOLATILE",
        )
        .when(
            (F.col("monthly_cv") < ts_medium_volatility_threshold)
            & (F.col("transaction_regularity") >= ts_high_regularity_threshold),
            "STABLE",
        )
        .when(
            (F.col("transaction_dispersion") < 5) & (F.col("transaction_regularity") >= ts_medium_regularity_threshold),
            "CONCENTRATED",
        )
        .when(
            (F.col("transaction_dispersion") >= 8) & (F.col("transaction_regularity") >= ts_medium_regularity_threshold),
            "DISTRIBUTED",
        )
        .otherwise("MIXED"),
    )

    return df
