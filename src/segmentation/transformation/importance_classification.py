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
    # Overall importance thresholds (based on total volume)
    critical_volume_threshold: float = 100_000_000_000,
    high_volume_threshold: float = 5_000_000_000,
    medium_volume_threshold: float = 1_000_000_000,
    # Monthly average thresholds
    critical_monthly_avg_threshold: float = 1_000_000_000,
    high_monthly_avg_threshold: float = 500_000_000,
    medium_monthly_avg_threshold: float = 100_000_000,
    # Max single transaction thresholds
    critical_max_transaction_threshold: float = 50_000_000,
    high_max_transaction_threshold: float = 10_000_000,
    medium_max_transaction_threshold: float = 5_000_000,
    # EOM importance thresholds (based on EOM-specific volume)
    critical_eom_volume_threshold: float = 50_000_000_000,
    high_eom_volume_threshold: float = 50_000_000_000,
    medium_eom_volume_threshold: float = 50_000_000_000,
    # EOM monthly average thresholds
    critical_eom_monthly_threshold: float = 50_000_000_000,
    high_eom_monthly_threshold: float = 50_000_000_000,
    medium_eom_monthly_threshold: float = 50_000_000_000,
    # Max single EOM transaction thresholds
    critical_max_eom_threshold: float = 100_000_000,
    high_max_eom_threshold: float = 50_000_000,
    medium_max_eom_threshold: float = 10_000_000,
    # Portfolio percentile thresholds for OVERALL importance
    overall_critical_percentile: float = 0.2,
    overall_high_percentile: float = 0.4,
    overall_medium_percentile: float = 0.8,
    # Portfolio percentile thresholds for EOM importance
    eom_critical_percentile: float = 0.3,
    eom_high_percentile: float = 0.6,
    eom_medium_percentile: float = 0.95,
    # EOM risk parameters
    eom_risk_volume_threshold: float = 100_000,
    eom_risk_min_months: int = 6,
) -> DataFrame:
    """
    Step 6: Dual importance classification (overall vs EOM)

    Uses both portfolio percentile thresholds AND absolute volume thresholds
    to match the SQL implementation exactly.

    Args:
        df: Input DataFrame with portfolio metrics
        critical_volume_threshold: Critical overall volume (12 months)
        high_volume_threshold: High overall volume (12 months)
        medium_volume_threshold: Medium overall volume (12 months)
        critical_monthly_avg_threshold: Critical monthly average
        high_monthly_avg_threshold: High monthly average
        medium_monthly_avg_threshold: Medium monthly average
        critical_max_transaction_threshold: Critical single transaction
        high_max_transaction_threshold: High single transaction
        medium_max_transaction_threshold: Medium single transaction
        critical_eom_volume_threshold: Critical EOM volume (12 months)
        high_eom_volume_threshold: High EOM volume (12 months)
        medium_eom_volume_threshold: Medium EOM volume (12 months)
        critical_eom_monthly_threshold: Critical EOM monthly average
        high_eom_monthly_threshold: High EOM monthly average
        medium_eom_monthly_threshold: Medium EOM monthly average
        critical_max_eom_threshold: Critical single EOM transaction
        high_max_eom_threshold: High single EOM transaction
        medium_max_eom_threshold: Medium single EOM transaction
        overall_critical_percentile: Top percentile for overall critical (0.2 = top 20%)
        overall_high_percentile: Top percentile for overall high (0.4 = top 40%)
        overall_medium_percentile: Top percentile for overall medium (0.8 = top 80%)
        eom_critical_percentile: Top percentile for EOM critical (0.3 = top 30%)
        eom_high_percentile: Top percentile for EOM high (0.6 = top 60%)
        eom_medium_percentile: Top percentile for EOM medium (0.95 = top 95%)
        eom_risk_volume_threshold: Volume threshold for EOM risk flag
        eom_risk_min_months: Min months history for risk assessment
    """
    logger.debug("Classifying importance based on portfolio percentiles AND volume thresholds")

    # Overall importance tier (using OVERALL-specific thresholds)
    # Matches SQL: Uses both percentile AND absolute thresholds
    df = df.with_column(
        "overall_importance_tier",
        F.when(
            (F.col("cumulative_overall_portfolio_pct") <= overall_critical_percentile)
            | (F.col("rolling_avg_monthly_volume") >= critical_monthly_avg_threshold)
            | (F.col("rolling_max_transaction") >= critical_max_transaction_threshold),
            "CRITICAL",
        )
        .when(
            (F.col("cumulative_overall_portfolio_pct") <= overall_high_percentile)
            | (F.col("rolling_avg_monthly_volume") >= high_monthly_avg_threshold)
            | (F.col("rolling_max_transaction") >= high_max_transaction_threshold),
            "HIGH",
        )
        .when(
            (F.col("cumulative_overall_portfolio_pct") <= overall_medium_percentile)
            | (F.col("rolling_avg_monthly_volume") >= medium_monthly_avg_threshold)
            | (F.col("rolling_max_transaction") >= medium_max_transaction_threshold),
            "MEDIUM",
        )
        .otherwise("LOW"),
    )

    # EOM importance tier (using EOM-specific thresholds)
    # Matches SQL: Uses both percentile AND absolute thresholds
    df = df.with_column(
        "eom_importance_tier",
        F.when(
            (F.col("cumulative_eom_portfolio_pct") <= eom_critical_percentile)
            | (F.col("rolling_avg_nonzero_eom") >= critical_eom_monthly_threshold)
            | (F.col("rolling_max_eom") >= critical_max_eom_threshold),
            "CRITICAL",
        )
        .when(
            (F.col("cumulative_eom_portfolio_pct") <= eom_high_percentile)
            | (F.col("rolling_avg_nonzero_eom") >= high_eom_monthly_threshold)
            | (F.col("rolling_max_eom") >= high_max_eom_threshold),
            "HIGH",
        )
        .when(
            (F.col("cumulative_eom_portfolio_pct") <= eom_medium_percentile)
            | ((F.col("rolling_avg_nonzero_eom") >= medium_eom_monthly_threshold) & (F.col("total_nonzero_eom_count") >= 3))
            | (F.col("rolling_max_eom") >= medium_max_eom_threshold),
            "MEDIUM",
        )
        .when(
            (F.col("rolling_eom_volume_12m") > 0) | (F.col("total_nonzero_eom_count") > 0),
            "LOW",
        )
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

    # EOM risk flag (high volume accounts with no EOM history)
    df = df.with_column(
        "eom_risk_flag",
        F.when(
            (F.col("rolling_total_volume_12m") >= eom_risk_volume_threshold)
            & (F.col("has_eom_history") == 0)
            & (F.col("months_of_history") >= eom_risk_min_months),
            1,
        ).otherwise(0),
    )

    return df
