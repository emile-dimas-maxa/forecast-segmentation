"""
Pattern metrics for behavioral classification
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from loguru import logger

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def calculate_pattern_metrics(
    df: DataFrame,
    pre_eom_signal_window: int = 6,
    pre_eom_days: int = 5,
    early_month_days: int = 10,
    mid_month_end_day: int = 20,
) -> DataFrame:
    """
    Step 5: Calculate pattern metrics for classification

    Args:
        df: Input DataFrame
        pre_eom_signal_window: Pre-EOM signal rolling window (months)
        pre_eom_days: Days before EOM to consider for pre-EOM signals
        early_month_days: First N days of month for early month signal
        mid_month_end_day: End day for mid-month period
    """
    logger.debug("Calculating pattern metrics for behavioral classification")

    # EOM concentration
    df = df.with_column(
        "eom_concentration",
        F.when(
            F.col("rolling_total_volume_12m") > 0, F.col("rolling_eom_volume_12m") / F.col("rolling_total_volume_12m")
        ).otherwise(0),
    )

    # EOM predictability
    df = df.with_column(
        "eom_predictability",
        F.when(
            F.col("rolling_avg_nonzero_eom") > 0,
            F.expr("GREATEST(0, 1 - (rolling_std_eom / rolling_avg_nonzero_eom))"),
        ).otherwise(0),
    )

    # EOM frequency
    df = df.with_column(
        "eom_frequency",
        F.when(
            F.col("months_of_history") > 1,
            F.coalesce(F.expr("rolling_nonzero_eom_months / LEAST(months_of_history - 1, 12)"), F.lit(0)),
        ).otherwise(0),
    )

    # EOM zero ratio
    df = df.with_column(
        "eom_zero_ratio",
        F.when(
            F.col("months_of_history") > 1,
            F.coalesce(F.expr("rolling_zero_eom_months / LEAST(months_of_history - 1, 12)"), F.lit(0)),
        ).otherwise(0),
    )

    # EOM spike ratio
    df = df.with_column(
        "eom_spike_ratio",
        F.when(
            (F.col("rolling_avg_non_eom") > 0) & (F.col("rolling_avg_nonzero_eom") > 0),
            F.col("rolling_avg_nonzero_eom") / F.col("rolling_avg_non_eom"),
        )
        .when((F.col("rolling_avg_nonzero_eom") > 0) & (F.col("rolling_avg_non_eom") == 0), 999)
        .otherwise(1),
    )

    # EOM CV
    df = df.with_column(
        "eom_cv",
        F.when(
            F.col("rolling_avg_nonzero_eom") > 0,
            F.col("rolling_std_eom") / F.col("rolling_avg_nonzero_eom"),
        ).otherwise(0),
    )

    # Monthly CV
    df = df.with_column(
        "monthly_cv",
        F.when(
            F.col("rolling_avg_monthly_volume") > 0, F.col("rolling_std_monthly") / F.col("rolling_avg_monthly_volume")
        ).otherwise(0),
    )

    # Transaction regularity
    df = df.with_column(
        "transaction_regularity",
        F.when(
            F.col("rolling_avg_transactions") > 0,
            F.expr("GREATEST(0, 1 - (rolling_std_transactions / rolling_avg_transactions))"),
        ).otherwise(0),
    )

    # Activity rate
    df = df.with_column(
        "activity_rate",
        F.when(F.col("months_of_history") > 0, F.expr("active_months_12m / LEAST(months_of_history, 12)")).otherwise(0),
    )

    # Quarter end concentration
    df = df.with_column(
        "quarter_end_concentration",
        F.when(
            F.col("rolling_total_volume_12m") > 0, F.col("rolling_quarter_end_volume") / F.col("rolling_total_volume_12m")
        ).otherwise(0),
    )

    # Year end concentration
    df = df.with_column(
        "year_end_concentration",
        F.when(
            F.col("rolling_total_volume_12m") > 0, F.col("rolling_year_end_volume") / F.col("rolling_total_volume_12m")
        ).otherwise(0),
    )

    # Transaction dispersion
    df = df.with_column(
        "transaction_dispersion",
        F.when(F.col("rolling_avg_day_dispersion") > 0, F.col("rolling_avg_day_dispersion")).otherwise(0),
    )

    # Has EOM history flag
    df = df.with_column("has_eom_history", F.when(F.col("total_nonzero_eom_count") > 0, 1).otherwise(0))

    # Months inactive
    window_inactive = Window.partition_by("dim_value", F.col("monthly_total") > 0).order_by(F.col("month").desc())
    df = df.with_column(
        "months_inactive", F.when(F.col("monthly_total") > 0, 0).otherwise(F.row_number().over(window_inactive) - 1)
    )

    # EOM periodicity detection
    df = df.with_column(
        "eom_periodicity",
        F.when(
            (F.col("total_nonzero_eom_count") >= 3) & (F.col("months_of_history") >= 12),
            F.when(F.col("rolling_nonzero_eom_months") >= 10, "MONTHLY")
            .when(
                (F.col("total_nonzero_eom_count") % 3 == 0)
                & (F.col("total_nonzero_eom_count") >= F.floor(F.col("months_of_history") / 4)),
                "QUARTERLY",
            )
            .when(
                (F.col("total_nonzero_eom_count") % 6 == 0)
                & (F.col("total_nonzero_eom_count") >= F.floor(F.col("months_of_history") / 7)),
                "SEMIANNUAL",
            )
            .when(F.col("total_nonzero_eom_count") == F.floor(F.col("months_of_history") / 12), "ANNUAL")
            .otherwise("IRREGULAR"),
        ).otherwise("INSUFFICIENT_DATA"),
    )

    return df
