"""
Importance tier classification functions
"""

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.utils import log_transformation


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
            | ((F.col("rolling_avg_nonzero_eom") >= config.medium_eom_monthly_threshold) & (F.col("total_nonzero_eom_count") >= 3))
            | (F.col("rolling_max_eom") >= config.medium_max_eom_threshold),
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
