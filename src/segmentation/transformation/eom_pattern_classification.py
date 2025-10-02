"""
EOM pattern classification functions - smooth scores, distances, probabilities
"""

from dataclasses import dataclass

from loguru import logger
from snowflake.snowpark import DataFrame
from snowflake.snowpark import functions as F

from src.segmentation.transformation.utils import log_transformation


@dataclass
class ArchetypeConfig:
    """Configuration for pattern archetypes (centroids) for Snowpark implementation"""

    # CONTINUOUS_STABLE: high regularity, high stability, medium recency
    continuous_stable: tuple[float, float, float] = (90, 80, 50)

    # CONTINUOUS_VOLATILE: high regularity, low stability, medium recency
    continuous_volatile: tuple[float, float, float] = (90, 20, 50)

    # INTERMITTENT_ACTIVE: medium regularity, medium stability, high recency
    intermittent_active: tuple[float, float, float] = (50, 50, 90)

    # INTERMITTENT_DORMANT: medium regularity, medium stability, low recency
    intermittent_dormant: tuple[float, float, float] = (50, 50, 20)

    # RARE_RECENT: low regularity, any stability, high recency
    rare_recent: tuple[float, float, float] = (15, 50, 85)

    # RARE_STALE: low regularity, any stability, low recency
    rare_stale: tuple[float, float, float] = (15, 50, 15)

    # NO_EOM: all zeros
    no_eom: tuple[float, float, float] = (0, 50, 0)

    # Weights for distance calculation [regularity, stability, recency]
    weights: tuple[float, float, float] = (1.0, 1.0, 0.5)

    # Special weight for stability in rare patterns (lower = less importance)
    rare_stability_weight: float = 0.3

    # Pattern classification parameters
    pattern_temperature: float = 20.0  # Softmax temperature (higher = softer classifications)
    emerging_months_threshold: int = 3  # Max months for EMERGING classification
    no_eom_min_months: int = 6  # Min months history for NO_EOM classification


@log_transformation
def calculate_eom_smooth_scores(
    df: DataFrame,
) -> DataFrame:
    """
    Step 7a: Calculate smooth scores for EOM patterns

    Args:
        df: Input DataFrame
    """
    logger.debug("Calculating smooth EOM pattern scores")

    # Regularity score: Sigmoid function for smooth transition (using default steepness of 10)
    df = df.with_column("regularity_score", F.expr("100 * (1 / (1 + EXP(-10 * (eom_frequency - 0.5))))"))

    # Stability score: Inverse exponential decay based on CV (using default decay rate of 2)
    df = df.with_column("stability_score", F.expr("100 * EXP(-2 * GREATEST(eom_cv, 0))"))

    # Recency score: Exponential time decay (using default decay rate of 0.8)
    df = df.with_column(
        "recency_score",
        F.when(F.col("has_eom_history") == 0, 0)
        .when(F.col("has_nonzero_eom") == 1, 100)
        .when(F.col("months_since_last_eom") == 1, 80)
        .when(F.col("months_since_last_eom") == 2, 64)
        .when(F.col("months_since_last_eom") == 3, 51)
        .otherwise(F.expr("100 * POWER(0.8, GREATEST(4, LEAST(24, months_since_last_eom)))")),
    )

    # Concentration score: Logistic curve (using default steepness of 8)
    df = df.with_column("concentration_score", F.expr("100 * (1 / (1 + EXP(-5 * (eom_concentration - 0.5))))"))

    # Volume importance score: Asymptotic growth (using default growth rate of 5)
    df = df.with_column(
        "volume_importance_score",
        F.when(
            F.col("total_portfolio_eom_volume") > 0,
            F.expr("100 * (1 - EXP(-5 * eom_importance_score))"),
        ).otherwise(0),
    )

    return df


@log_transformation
def calculate_pattern_distances(df: DataFrame, archetype_config: ArchetypeConfig = None) -> DataFrame:
    """
    Step 7b: Calculate distances to pattern archetypes

    Args:
        df: Input DataFrame with smooth scores
        archetype_config: Configuration for pattern archetypes (centroids)
    """
    logger.debug("Calculating distances to EOM pattern archetypes")

    if archetype_config is None:
        archetype_config = ArchetypeConfig()

    # Extract weights
    w_reg, w_stab, w_rec = archetype_config.weights

    # CONTINUOUS_STABLE
    reg, stab, rec = archetype_config.continuous_stable
    df = df.with_column(
        "dist_continuous_stable",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * w_stab
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # CONTINUOUS_VOLATILE
    reg, stab, rec = archetype_config.continuous_volatile
    df = df.with_column(
        "dist_continuous_volatile",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * w_stab
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # INTERMITTENT_ACTIVE
    reg, stab, rec = archetype_config.intermittent_active
    df = df.with_column(
        "dist_intermittent_active",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * w_stab
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # INTERMITTENT_DORMANT
    reg, stab, rec = archetype_config.intermittent_dormant
    df = df.with_column(
        "dist_intermittent_dormant",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * w_stab
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # RARE_RECENT
    reg, stab, rec = archetype_config.rare_recent
    df = df.with_column(
        "dist_rare_recent",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * archetype_config.rare_stability_weight
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # RARE_STALE
    reg, stab, rec = archetype_config.rare_stale
    df = df.with_column(
        "dist_rare_stale",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * archetype_config.rare_stability_weight
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # NO_EOM
    reg, stab, rec = archetype_config.no_eom
    df = df.with_column(
        "dist_no_eom",
        F.sqrt(
            F.pow(reg - F.col("regularity_score"), 2) * w_reg
            + F.pow(stab - F.col("stability_score"), 2) * w_stab
            + F.pow(rec - F.col("recency_score"), 2) * w_rec
        ),
    )

    # EMERGING: special case for new series
    df = df.with_column("dist_emerging", F.when(F.col("months_of_history") <= 3, 0).otherwise(999))

    return df


@log_transformation
def calculate_pattern_probabilities(df: DataFrame, archetype_config: ArchetypeConfig = None) -> DataFrame:
    """
    Step 7c: Convert distances to probabilities using softmax

    Args:
        df: Input DataFrame with pattern distances
        archetype_config: Configuration for pattern archetypes
    """
    logger.debug("Converting distances to probabilities using softmax")

    if archetype_config is None:
        archetype_config = ArchetypeConfig()

    # Calculate softmax denominator (using configurable temperature)
    # Higher temperature = softer, less confident classifications
    pattern_temperature = archetype_config.pattern_temperature
    df = df.with_column(
        "softmax_denominator",
        F.exp(-F.col("dist_continuous_stable") / pattern_temperature)
        + F.exp(-F.col("dist_continuous_volatile") / pattern_temperature)
        + F.exp(-F.col("dist_intermittent_active") / pattern_temperature)
        + F.exp(-F.col("dist_intermittent_dormant") / pattern_temperature)
        + F.exp(-F.col("dist_rare_recent") / pattern_temperature)
        + F.exp(-F.col("dist_rare_stale") / pattern_temperature)
        + F.when(F.col("has_eom_history") == 0, F.exp(-F.col("dist_no_eom") / pattern_temperature)).otherwise(0)
        + F.when(F.col("months_of_history") <= 3, F.exp(-F.col("dist_emerging") / pattern_temperature)).otherwise(0),
    )

    # Calculate probabilities for each pattern
    df = df.with_column(
        "prob_continuous_stable",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_continuous_stable") / pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_continuous_volatile",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_continuous_volatile") / pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_intermittent_active",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_intermittent_active") / pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_intermittent_dormant",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_intermittent_dormant") / pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_rare_recent",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_rare_recent") / pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_rare_stale",
        F.when(F.col("months_of_history") <= 3, 0).otherwise(
            F.exp(-F.col("dist_rare_stale") / pattern_temperature) / F.col("softmax_denominator")
        ),
    )

    df = df.with_column(
        "prob_no_eom",
        F.when(
            F.col("has_eom_history") == 0,
            F.exp(-F.col("dist_no_eom") / pattern_temperature) / F.col("softmax_denominator"),
        ).otherwise(0),
    )

    df = df.with_column(
        "prob_emerging",
        F.when(
            F.col("months_of_history") <= 3,
            F.exp(-F.col("dist_emerging") / pattern_temperature) / F.col("softmax_denominator"),
        ).otherwise(0),
    )

    return df


@log_transformation
def classify_eom_patterns(
    df: DataFrame,
    archetype_config: ArchetypeConfig = None,
    eom_high_risk_stability_threshold: float = 30.0,
    eom_high_risk_concentration_threshold: float = 50.0,
) -> DataFrame:
    """
    Step 7d: Final EOM pattern classification

    Args:
        df: Input DataFrame with pattern probabilities
        archetype_config: Configuration for pattern archetypes
        eom_high_risk_stability_threshold: Stability score threshold for high risk flag (lower = higher risk)
        eom_high_risk_concentration_threshold: Concentration score threshold for high risk flag
    """
    logger.debug("Classifying final EOM patterns")

    if archetype_config is None:
        archetype_config = ArchetypeConfig()

    # Find primary classification (highest probability)
    df = df.with_column(
        "eom_pattern",
        F.when(F.col("months_of_history") <= archetype_config.emerging_months_threshold, "EMERGING")
        .when((F.col("has_eom_history") == 0) & (F.col("months_of_history") >= archetype_config.no_eom_min_months), "NO_EOM")
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
            F.when(
                (F.col("months_of_history") <= archetype_config.emerging_months_threshold) | (F.col("has_eom_history") == 0), 1.0
            ).otherwise(0),
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

    # High risk flag for high-value volatile accounts
    df = df.with_column(
        "eom_high_risk_flag",
        F.when(
            (F.col("eom_importance_tier").isin(["CRITICAL", "HIGH"]))
            & (F.col("stability_score") < eom_high_risk_stability_threshold)
            & (F.col("concentration_score") >= eom_high_risk_concentration_threshold),
            1,
        ).otherwise(0),
    )

    return df
