"""
Importance tier classification functions
"""

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def classify_importance(
    df: DataFrame,
    critical_volume_threshold: float = 100_000_000_000,
    high_volume_threshold: float = 5_000_000_000,
    medium_volume_threshold: float = 1_000_000_000,
    critical_monthly_avg_threshold: float = 1_000_000_000,
    high_monthly_avg_threshold: float = 416_666_667,
    medium_monthly_avg_threshold: float = 83_333_333,
    critical_eom_threshold: float = 50_000_000_000,
    high_eom_threshold: float = 2_500_000_000,
    medium_eom_threshold: float = 500_000_000,
) -> DataFrame:
    """
    Step 6: Dual importance classification (overall vs EOM)

    Args:
        df: Input DataFrame
        critical_volume_threshold: Critical overall volume (12 months)
        high_volume_threshold: High overall volume (12 months)
        medium_volume_threshold: Medium overall volume (12 months)
        critical_monthly_avg_threshold: Critical monthly average
        high_monthly_avg_threshold: High monthly average
        medium_monthly_avg_threshold: Medium monthly average
        critical_eom_threshold: Critical EOM volume (12 months)
        high_eom_threshold: High EOM volume (12 months)
        medium_eom_threshold: Medium EOM volume (12 months)
    """
    logger.debug("Classifying importance based on volume thresholds")

    # Overall importance tier
    df = df.with_column(
        "overall_importance_tier",
        F.when(
            (F.col("rolling_total_volume_12m") >= critical_volume_threshold)
            | (F.col("rolling_avg_monthly_volume") >= critical_monthly_avg_threshold),
            "CRITICAL",
        )
        .when(
            (F.col("rolling_total_volume_12m") >= high_volume_threshold)
            | (F.col("rolling_avg_monthly_volume") >= high_monthly_avg_threshold),
            "HIGH",
        )
        .when(
            (F.col("rolling_total_volume_12m") >= medium_volume_threshold)
            | (F.col("rolling_avg_monthly_volume") >= medium_monthly_avg_threshold),
            "MEDIUM",
        )
        .otherwise("LOW"),
    )

    # EOM importance tier
    df = df.with_column(
        "eom_importance_tier",
        F.when(
            F.col("rolling_eom_volume_12m") >= critical_eom_threshold,
            "CRITICAL",
        )
        .when(
            F.col("rolling_eom_volume_12m") >= high_eom_threshold,
            "HIGH",
        )
        .when(
            (F.col("rolling_eom_volume_12m") >= medium_eom_threshold) & (F.col("total_nonzero_eom_count") >= 3),
            "MEDIUM",
        )
        .when((F.col("rolling_eom_volume_12m") > 0) | (F.col("total_nonzero_eom_count") > 0), "LOW")
        .otherwise("NONE"),
    )

    # Importance scores
    df = df.with_column(
        "overall_importance_score",
        F.when(F.col("total_portfolio_volume") > 0, F.col("rolling_total_volume_12m") / F.col("total_portfolio_volume")).otherwise(
            0
        ),
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
