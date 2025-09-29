"""
EOM pattern classification functions - smooth scores, distances, probabilities
"""

from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F
from loguru import logger

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.utils import log_transformation


@log_transformation
def calculate_eom_smooth_scores(config: SegmentationConfig, df: DataFrame) -> DataFrame:
    """
    Step 7a: Calculate smooth scores for EOM patterns
    """
    logger.debug("Calculating smooth EOM pattern scores")

    # Regularity score: Sigmoid function for smooth transition
    df = df.with_column("regularity_score", F.expr(f"100 * (1 / (1 + EXP(-{config.sigmoid_steepness} * (eom_frequency - 0.5))))"))

    # Stability score: Inverse exponential decay based on CV
    df = df.with_column("stability_score", F.expr(f"100 * EXP(-{config.stability_decay_rate} * GREATEST(eom_cv, 0))"))

    # Recency score: Exponential time decay
    df = df.with_column(
        "recency_score",
        F.when(F.col("has_eom_history") == 0, 0)
        .when(F.col("has_nonzero_eom") == 1, 100)
        .when(F.col("months_since_last_eom") == 1, 80)
        .when(F.col("months_since_last_eom") == 2, 64)
        .when(F.col("months_since_last_eom") == 3, 51)
        .otherwise(F.expr(f"100 * POWER({config.recency_decay_rate}, GREATEST(4, LEAST(24, months_since_last_eom)))")),
    )

    # Concentration score: Logistic curve
    df = df.with_column(
        "concentration_score", F.expr(f"100 * (1 / (1 + EXP(-{config.concentration_steepness} * (eom_concentration - 0.5))))")
    )

    # Volume importance score: Asymptotic growth
    df = df.with_column(
        "volume_importance_score",
        F.when(
            F.col("total_portfolio_eom_volume") > 0,
            F.expr(f"100 * (1 - EXP(-{config.volume_growth_rate} * eom_importance_score))"),
        ).otherwise(0),
    )

    return df


@log_transformation
def calculate_pattern_distances(config: SegmentationConfig, df: DataFrame) -> DataFrame:
    """
    Step 7b: Calculate distances to pattern archetypes
    """
    logger.debug("Calculating distances to EOM pattern archetypes")

    # CONTINUOUS_STABLE: high regularity (90), high stability (80), medium recency (50)
    df = df.with_column(
        "dist_continuous_stable",
        F.sqrt(
            F.pow(90 - F.col("regularity_score"), 2)
            + F.pow(80 - F.col("stability_score"), 2)
            + F.pow(50 - F.col("recency_score"), 2) * 0.5
        ),
    )

    # CONTINUOUS_VOLATILE: high regularity (90), low stability (20), medium recency (50)
    df = df.with_column(
        "dist_continuous_volatile",
        F.sqrt(
            F.pow(90 - F.col("regularity_score"), 2)
            + F.pow(20 - F.col("stability_score"), 2)
            + F.pow(50 - F.col("recency_score"), 2) * 0.5
        ),
    )

    # INTERMITTENT_ACTIVE: medium regularity (50), medium stability (50), high recency (90)
    df = df.with_column(
        "dist_intermittent_active",
        F.sqrt(
            F.pow(50 - F.col("regularity_score"), 2)
            + F.pow(50 - F.col("stability_score"), 2)
            + F.pow(90 - F.col("recency_score"), 2)
        ),
    )

    # INTERMITTENT_DORMANT: medium regularity (50), medium stability (50), low recency (20)
    df = df.with_column(
        "dist_intermittent_dormant",
        F.sqrt(
            F.pow(50 - F.col("regularity_score"), 2)
            + F.pow(50 - F.col("stability_score"), 2)
            + F.pow(20 - F.col("recency_score"), 2)
        ),
    )

    # RARE_RECENT: low regularity (15), any stability (50), high recency (85)
    df = df.with_column(
        "dist_rare_recent",
        F.sqrt(
            F.pow(15 - F.col("regularity_score"), 2)
            + F.pow(50 - F.col("stability_score"), 2) * 0.3
            + F.pow(85 - F.col("recency_score"), 2)
        ),
    )

    # RARE_STALE: low regularity (15), any stability (50), low recency (15)
    df = df.with_column(
        "dist_rare_stale",
        F.sqrt(
            F.pow(15 - F.col("regularity_score"), 2)
            + F.pow(50 - F.col("stability_score"), 2) * 0.3
            + F.pow(15 - F.col("recency_score"), 2)
        ),
    )

    # NO_EOM: all zeros
    df = df.with_column(
        "dist_no_eom",
        F.sqrt(
            F.pow(0 - F.col("regularity_score"), 2) + F.pow(50 - F.col("stability_score"), 2) + F.pow(0 - F.col("recency_score"), 2)
        ),
    )

    # EMERGING: special case for new series
    df = df.with_column("dist_emerging", F.when(F.col("months_of_history") <= 3, 0).otherwise(999))

    return df


@log_transformation
def calculate_pattern_probabilities(config: SegmentationConfig, df: DataFrame) -> DataFrame:
    """
    Step 7c: Convert distances to probabilities using softmax
    """
    logger.debug(f"Converting distances to probabilities with temperature={config.pattern_temperature}")

    # Calculate softmax denominator
    df = df.with_column(
        "softmax_denominator",
        F.exp(-F.col("dist_continuous_stable") / config.pattern_temperature)
        + F.exp(-F.col("dist_continuous_volatile") / config.pattern_temperature)
        + F.exp(-F.col("dist_intermittent_active") / config.pattern_temperature)
        + F.exp(-F.col("dist_intermittent_dormant") / config.pattern_temperature)
        + F.exp(-F.col("dist_rare_recent") / config.pattern_temperature)
        + F.exp(-F.col("dist_rare_stale") / config.pattern_temperature)
        + F.when(F.col("has_eom_history") == 0, F.exp(-F.col("dist_no_eom") / config.pattern_temperature)).otherwise(0)
        + F.when(F.col("months_of_history") <= 3, F.exp(-F.col("dist_emerging") / config.pattern_temperature)).otherwise(0),
    )

    # Calculate probabilities for each pattern
    df = df.with_column(
        "prob_continuous_stable",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_continuous_stable") / config.pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_continuous_volatile",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_continuous_volatile") / config.pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_intermittent_active",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_intermittent_active") / config.pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_intermittent_dormant",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_intermittent_dormant") / config.pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_rare_recent",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_rare_recent") / config.pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_rare_stale",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_rare_stale") / config.pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_no_eom",
        F.when(
            F.col("has_eom_history") == 0,
            F.exp(-F.col("dist_no_eom") / config.pattern_temperature) / F.col("softmax_denominator"),
        ).otherwise(0),
    )

    df = df.with_column(
        "prob_emerging",
        F.when(
            F.col("months_of_history") <= 3,
            F.exp(-F.col("dist_emerging") / config.pattern_temperature) / F.col("softmax_denominator"),
        ).otherwise(0),
    )

    return df


@log_transformation
def classify_eom_patterns(config: SegmentationConfig, df: DataFrame) -> DataFrame:
    """
    Step 7d: Final EOM pattern classification
    """
    logger.debug("Classifying final EOM patterns")

    # Find primary classification (highest probability)
    df = df.with_column(
        "eom_pattern",
        F.when(F.col("months_of_history") <= 3, "EMERGING")
        .when((F.col("has_eom_history") == 0) & (F.col("months_of_history") >= 6), "NO_EOM")
        .otherwise(
            F.when(
                F.greatest(
                    F.col("prob_continuous_stable"),
                    F.col("prob_continuous_volatile"),
                    F.col("prob_intermittent_active"),
                    F.col("prob_intermittent_dormant"),
                    F.col("prob_rare_recent"),
                    F.col("prob_rare_stale"),
                )
                == F.col("prob_continuous_stable"),
                "CONTINUOUS_STABLE",
            )
            .when(
                F.greatest(
                    F.col("prob_continuous_stable"),
                    F.col("prob_continuous_volatile"),
                    F.col("prob_intermittent_active"),
                    F.col("prob_intermittent_dormant"),
                    F.col("prob_rare_recent"),
                    F.col("prob_rare_stale"),
                )
                == F.col("prob_continuous_volatile"),
                "CONTINUOUS_VOLATILE",
            )
            .when(
                F.greatest(
                    F.col("prob_continuous_stable"),
                    F.col("prob_continuous_volatile"),
                    F.col("prob_intermittent_active"),
                    F.col("prob_intermittent_dormant"),
                    F.col("prob_rare_recent"),
                    F.col("prob_rare_stale"),
                )
                == F.col("prob_intermittent_active"),
                "INTERMITTENT_ACTIVE",
            )
            .when(
                F.greatest(
                    F.col("prob_continuous_stable"),
                    F.col("prob_continuous_volatile"),
                    F.col("prob_intermittent_active"),
                    F.col("prob_intermittent_dormant"),
                    F.col("prob_rare_recent"),
                    F.col("prob_rare_stale"),
                )
                == F.col("prob_intermittent_dormant"),
                "INTERMITTENT_DORMANT",
            )
            .when(
                F.greatest(
                    F.col("prob_continuous_stable"),
                    F.col("prob_continuous_volatile"),
                    F.col("prob_intermittent_active"),
                    F.col("prob_intermittent_dormant"),
                    F.col("prob_rare_recent"),
                    F.col("prob_rare_stale"),
                )
                == F.col("prob_rare_recent"),
                "RARE_RECENT",
            )
            .otherwise("RARE_STALE")
        ),
    )

    # Pattern confidence
    df = df.with_column(
        "eom_pattern_confidence",
        F.greatest(
            F.when((F.col("months_of_history") <= 3) | (F.col("has_eom_history") == 0), 1.0).otherwise(0),
            F.col("prob_continuous_stable"),
            F.col("prob_continuous_volatile"),
            F.col("prob_intermittent_active"),
            F.col("prob_intermittent_dormant"),
            F.col("prob_rare_recent"),
            F.col("prob_rare_stale"),
        ),
    )

    # Classification entropy (uncertainty)
    df = df.with_column(
        "classification_entropy",
        F.expr("""
            -(
                CASE WHEN prob_continuous_stable > 0 THEN prob_continuous_stable * LN(prob_continuous_stable) ELSE 0 END +
                CASE WHEN prob_continuous_volatile > 0 THEN prob_continuous_volatile * LN(prob_continuous_volatile) ELSE 0 END +
                CASE WHEN prob_intermittent_active > 0 THEN prob_intermittent_active * LN(prob_intermittent_active) ELSE 0 END +
                CASE WHEN prob_intermittent_dormant > 0 THEN prob_intermittent_dormant * LN(prob_intermittent_dormant) ELSE 0 END +
                CASE WHEN prob_rare_recent > 0 THEN prob_rare_recent * LN(prob_rare_recent) ELSE 0 END +
                CASE WHEN prob_rare_stale > 0 THEN prob_rare_stale * LN(prob_rare_stale) ELSE 0 END
            )
        """),
    )

    # High risk flag
    df = df.with_column(
        "eom_high_risk_flag",
        F.when(
            (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]))
            & (F.col("stability_score") < 30)
            & (F.col("concentration_score") >= 50),
            1,
        ).otherwise(0),
    )

    return df
