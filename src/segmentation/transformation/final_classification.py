"""
Final classification and recommendations
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from loguru import logger

from src.segmentation.transformation.utils import log_transformation


@log_transformation
def create_final_classification(df: DataFrame) -> DataFrame:
    """
    Step 9: Combined priority and recommendations
    """
    logger.debug("Creating final classification and recommendations")

    # Combined priority (1-10 scale)
    df = df.with_column(
        "combined_priority",
        F.when(
            ((F.col("overall_importance_tier") == "CRITICAL") | (F.col("eom_importance_tier") == "CRITICAL"))
            & (
                (F.col("eom_pattern").isin(["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]))
                | (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))
            ),
            10,
        )
        .when(
            ((F.col("overall_importance_tier") == "CRITICAL") | (F.col("eom_importance_tier") == "CRITICAL"))
            & ((F.col("eom_pattern") == "SEASONAL_EOM") | (F.col("general_pattern") == "HIGHLY_SEASONAL")),
            9,
        )
        .when((F.col("overall_importance_tier") == "CRITICAL") | (F.col("eom_importance_tier") == "CRITICAL"), 8)
        .when(
            ((F.col("overall_importance_tier") == "HIGH") | (F.col("eom_importance_tier") == "HIGH"))
            & (
                (F.col("eom_pattern").isin(["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]))
                | (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))
            ),
            7,
        )
        .when((F.col("overall_importance_tier") == "HIGH") | (F.col("eom_importance_tier") == "HIGH"), 6)
        .when(
            (F.col("eom_risk_flag") == 1)
            | ((F.col("overall_importance_tier") == "MEDIUM") & (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))),
            5,
        )
        .when((F.col("overall_importance_tier") == "MEDIUM") | (F.col("eom_importance_tier") == "MEDIUM"), 4)
        .when(~F.col("general_pattern").isin(["INACTIVE", "EMERGING"]), 3)
        .when((F.col("general_pattern") == "EMERGING") | (F.col("eom_pattern") == "EMERGING"), 2)
        .otherwise(1),
    )

    # Recommended forecasting method
    df = df.with_column(
        "recommended_method",
        F.when(F.col("eom_pattern") == "NO_EOM", "Zero_EOM_Forecast")
        .when(
            F.col("eom_pattern").isin(["CONTINUOUS_STABLE", "CONTINUOUS_VOLATILE"]),
            F.when(
                (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"])) & (F.col("eom_pattern") == "CONTINUOUS_VOLATILE"),
                "XGBoost_EOM_Focus",
            )
            .when(F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]), "SARIMA_EOM")
            .otherwise("Simple_MA_EOM"),
        )
        .when(
            (F.col("eom_pattern") == "INTERMITTENT_ACTIVE") | (F.col("general_pattern") == "INTERMITTENT"),
            F.when(F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"]), "Croston_Method").otherwise("Zero_Inflated_Model"),
        )
        .when(
            F.col("general_pattern") == "HIGHLY_SEASONAL",
            F.when(F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"]), "Seasonal_Decomposition").otherwise(
                "Seasonal_Naive"
            ),
        )
        .when(F.col("general_pattern") == "VOLATILE", "XGBoost_Full_Series")
        .when(F.col("general_pattern") == "STABLE", "Linear_Trend")
        .when(F.col("general_pattern") == "INACTIVE", "Zero_Forecast")
        .when(F.col("general_pattern") == "EMERGING", "Conservative_MA")
        .otherwise("Historical_Average"),
    )

    # Forecast complexity
    df = df.with_column(
        "forecast_complexity",
        F.when(
            (
                (F.col("eom_pattern").isin(["CONTINUOUS_VOLATILE", "INTERMITTENT_ACTIVE", "INTERMITTENT_DORMANT"]))
                | (F.col("general_pattern").isin(["VOLATILE", "INTERMITTENT"]))
            )
            & (
                (F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"]))
                | (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]))
            ),
            5,
        )
        .when((F.col("general_pattern") == "HIGHLY_SEASONAL") & (F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"])), 4)
        .when(
            (F.col("eom_pattern").isin(["CONTINUOUS_STABLE"]))
            | (F.col("general_pattern") == "STABLE")
            | (F.col("overall_importance_tier").isin(["CRITICAL", "HIGH"])),
            3,
        )
        .when((F.col("overall_importance_tier") == "MEDIUM") | (F.col("eom_importance_tier") == "MEDIUM"), 2)
        .otherwise(1),
    )

    # Segment names
    df = df.with_column(
        "full_segment_name",
        F.concat(
            F.col("overall_importance_tier"),
            F.lit("_"),
            F.col("general_pattern"),
            F.lit("__"),
            F.col("eom_importance_tier"),
            F.lit("EOM_"),
            F.col("eom_pattern"),
        ),
    )

    df = df.with_column(
        "segment_name",
        F.concat(
            F.when(F.col("overall_importance_tier") == F.col("eom_importance_tier"), F.col("overall_importance_tier")).otherwise(
                F.concat(F.col("overall_importance_tier"), F.lit("/"), F.col("eom_importance_tier"), F.lit("EOM"))
            ),
            F.lit("_"),
            F.col("general_pattern"),
            F.lit("_"),
            F.col("eom_pattern"),
        ),
    )

    return df
