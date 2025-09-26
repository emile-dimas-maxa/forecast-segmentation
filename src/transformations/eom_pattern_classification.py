"""EOM pattern classification using smooth scoring approach."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig


@dataclass
class ArchetypeCharacteristics:
    """Characteristics defining a pattern archetype."""

    regularity_score: float
    stability_score: float
    recency_score: float
    regularity_weight: float = 1.0
    stability_weight: float = 1.0
    recency_weight: float = 1.0


@dataclass
class EOMArchetype:
    """Definition of an EOM pattern archetype."""

    name: str
    characteristics: ArchetypeCharacteristics
    requires_eom_history: bool = True
    max_months_of_history: int | None = None
    min_months_of_history: int | None = None


class EOMArchetypes:
    """Container for all EOM pattern archetypes."""

    def __init__(self):
        self.archetypes = {
            "continuous_stable": EOMArchetype(
                name="CONTINUOUS_STABLE",
                characteristics=ArchetypeCharacteristics(
                    regularity_score=90, stability_score=80, recency_score=50, recency_weight=0.5
                ),
            ),
            "continuous_volatile": EOMArchetype(
                name="CONTINUOUS_VOLATILE",
                characteristics=ArchetypeCharacteristics(
                    regularity_score=90, stability_score=20, recency_score=50, recency_weight=0.5
                ),
            ),
            "intermittent_active": EOMArchetype(
                name="INTERMITTENT_ACTIVE",
                characteristics=ArchetypeCharacteristics(regularity_score=50, stability_score=50, recency_score=90),
            ),
            "intermittent_dormant": EOMArchetype(
                name="INTERMITTENT_DORMANT",
                characteristics=ArchetypeCharacteristics(regularity_score=50, stability_score=50, recency_score=20),
            ),
            "rare_recent": EOMArchetype(
                name="RARE_RECENT",
                characteristics=ArchetypeCharacteristics(
                    regularity_score=15, stability_score=50, recency_score=85, stability_weight=0.3
                ),
            ),
            "rare_stale": EOMArchetype(
                name="RARE_STALE",
                characteristics=ArchetypeCharacteristics(
                    regularity_score=15, stability_score=50, recency_score=15, stability_weight=0.3
                ),
            ),
            "no_eom": EOMArchetype(
                name="NO_EOM",
                characteristics=ArchetypeCharacteristics(regularity_score=0, stability_score=50, recency_score=0),
                requires_eom_history=False,
            ),
            "emerging": EOMArchetype(
                name="EMERGING",
                characteristics=ArchetypeCharacteristics(
                    regularity_score=0,  # Not used for emerging
                    stability_score=0,  # Not used for emerging
                    recency_score=0,  # Not used for emerging
                ),
                max_months_of_history=3,
            ),
        }

    def get_archetype(self, name: str) -> EOMArchetype:
        """Get archetype by name."""
        return self.archetypes[name]

    def get_all_archetypes(self) -> dict[str, EOMArchetype]:
        """Get all archetypes."""
        return self.archetypes

    def get_applicable_archetypes(self, has_eom_history: bool, months_of_history: int) -> list[str]:
        """Get list of applicable archetype names based on data conditions."""
        applicable = []

        # Special case: if emerging (months <= 3), only emerging and potentially no_eom are applicable
        if months_of_history <= 3:
            applicable.append("emerging")
            if not has_eom_history:
                applicable.append("no_eom")
            return applicable

        # Special case: if no EOM history and enough data, include no_eom
        if not has_eom_history and months_of_history >= 6:
            applicable.append("no_eom")
            return applicable

        # For regular cases with EOM history and sufficient data
        for key, archetype in self.archetypes.items():
            # Skip emerging and no_eom as they're handled above
            if archetype.name in ["EMERGING", "NO_EOM"]:
                continue

            # Check history requirements
            if archetype.requires_eom_history and not has_eom_history:
                continue

            # Check months of history constraints
            if archetype.max_months_of_history and months_of_history > archetype.max_months_of_history:
                continue

            if archetype.min_months_of_history and months_of_history < archetype.min_months_of_history:
                continue

            applicable.append(key)

        return applicable


def calculate_archetype_distance(
    regularity_score: float, stability_score: float, recency_score: float, archetype: EOMArchetype
) -> float:
    """Calculate distance to an archetype."""
    chars = archetype.characteristics

    # Special case for emerging pattern
    if archetype.name == "EMERGING":
        return 0  # Will be handled separately based on months_of_history

    distance = np.sqrt(
        ((chars.regularity_score - regularity_score) ** 2) * chars.regularity_weight
        + ((chars.stability_score - stability_score) ** 2) * chars.stability_weight
        + ((chars.recency_score - recency_score) ** 2) * chars.recency_weight
    )

    return distance


def classify_eom_patterns(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Classify EOM patterns using smooth scoring and distance-based approach."""
    start_time = time.time()
    initial_rows = len(df)

    logger.debug("Starting EOM pattern classification")
    logger.debug("Input shape: {} rows × {} columns", initial_rows, len(df.columns))

    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    # Initialize archetypes
    archetypes = EOMArchetypes()
    logger.debug(
        "Initialized {} archetypes: {}", len(archetypes.get_all_archetypes()), list(archetypes.get_all_archetypes().keys())
    )

    # Calculate smooth scores
    logger.debug("Calculating smooth scores (regularity, stability, recency, concentration, volume importance)")
    df["regularity_score"] = 100 * (1 / (1 + np.exp(-10 * (df["eom_frequency"] - 0.5))))
    df["stability_score"] = 100 * np.exp(-2 * np.maximum(df["eom_cv"], 0))

    # Recency score
    def calculate_recency_score(row):
        if row["has_eom_history"] == 0:
            return 0
        elif row["has_nonzero_eom"] == 1:
            return 100
        elif row["months_since_last_eom"] == 1:
            return 80
        elif row["months_since_last_eom"] == 2:
            return 64
        elif row["months_since_last_eom"] == 3:
            return 51
        else:
            return 100 * (0.8 ** max(4, min(24, row["months_since_last_eom"])))

    df["recency_score"] = df.apply(calculate_recency_score, axis=1)

    df["concentration_score"] = 100 * (1 / (1 + np.exp(-5 * (df["eom_concentration"] - 0.5))))

    df["volume_importance_score"] = np.where(
        df["total_portfolio_eom_volume"] > 0, 100 * (1 - np.exp(-5 * df["eom_importance_score"])), 0
    )

    # Calculate distances to pattern archetypes using the new archetype system
    temperature = 20.0

    logger.debug("Calculating distances to {} archetypes with temperature={}", len(archetypes.get_all_archetypes()), temperature)

    # Calculate distances for each archetype
    for archetype_key, archetype in archetypes.get_all_archetypes().items():
        if archetype.name == "EMERGING":
            # Special handling for emerging pattern
            df[f"dist_{archetype_key}"] = np.where(df["months_of_history"] <= 3, 0, 999)
        else:
            # Use the new distance calculation function
            def calc_distance(row, arch=archetype):
                return calculate_archetype_distance(row["regularity_score"], row["stability_score"], row["recency_score"], arch)

            df[f"dist_{archetype_key}"] = df.apply(calc_distance, axis=1)

    # Calculate softmax probabilities using archetype system
    def calculate_softmax_denominator(row):
        terms = []

        for archetype_key in archetypes.get_all_archetypes():
            # Check if this archetype is applicable for this row
            applicable_archetypes = archetypes.get_applicable_archetypes(
                bool(row["has_eom_history"]), int(row["months_of_history"])
            )

            if archetype_key in applicable_archetypes:
                terms.append(np.exp(-row[f"dist_{archetype_key}"] / temperature))

        return sum(terms) if terms else 1.0  # Avoid division by zero

    df["softmax_denominator"] = df.apply(calculate_softmax_denominator, axis=1)

    # Calculate pattern probabilities for each archetype
    for archetype_key, archetype in archetypes.get_all_archetypes().items():
        prob_col = f"prob_{archetype_key}"
        dist_col = f"dist_{archetype_key}"

        if archetype.name == "NO_EOM":
            df[prob_col] = np.where(df["has_eom_history"] == 0, np.exp(-df[dist_col] / temperature) / df["softmax_denominator"], 0)
        elif archetype.name == "EMERGING":
            df[prob_col] = np.where(
                df["months_of_history"] <= 3, np.exp(-df[dist_col] / temperature) / df["softmax_denominator"], 0
            )
        else:
            # Regular archetypes - exclude if emerging (months <= 3)
            df[prob_col] = np.where(
                df["months_of_history"] <= 3, 0, np.exp(-df[dist_col] / temperature) / df["softmax_denominator"]
            )

    # Primary classification using archetype system
    def get_primary_pattern(row):
        if row["months_of_history"] <= 3:
            return "EMERGING"
        elif row["has_eom_history"] == 0 and row["months_of_history"] >= 6:
            return "NO_EOM"
        else:
            # Get probabilities for all regular archetypes (excluding NO_EOM and EMERGING)
            probs = {}
            for archetype_key, archetype in archetypes.get_all_archetypes().items():
                if archetype.name not in ["NO_EOM", "EMERGING"]:
                    probs[archetype.name] = row[f"prob_{archetype_key}"]

            return max(probs, key=probs.get) if probs else "UNKNOWN"

    df["eom_pattern"] = df.apply(get_primary_pattern, axis=1)

    # Log pattern distribution
    pattern_counts = df["eom_pattern"].value_counts()
    logger.debug("Pattern distribution: {}", pattern_counts.to_dict())

    # Pattern confidence - get max probability across all archetypes
    prob_columns = [f"prob_{key}" for key in archetypes.get_all_archetypes()]
    df["eom_pattern_confidence"] = df[prob_columns].max(axis=1)

    avg_confidence = df["eom_pattern_confidence"].mean()
    logger.debug("Average pattern confidence: {:.3f}", avg_confidence)

    # Classification entropy - calculate for regular archetypes only (excluding NO_EOM and EMERGING)
    def calculate_entropy(row):
        probs = []
        for archetype_key, archetype in archetypes.get_all_archetypes().items():
            if archetype.name not in ["NO_EOM", "EMERGING"]:
                probs.append(row[f"prob_{archetype_key}"])

        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p)
        return entropy

    df["classification_entropy"] = df.apply(calculate_entropy, axis=1)
    avg_entropy = df["classification_entropy"].mean()
    logger.debug("Average classification entropy: {:.3f}", avg_entropy)

    # High risk flag
    df["eom_high_risk_flag"] = (
        df["eom_importance_tier"].isin(["CRITICAL", "HIGH"]) & (df["stability_score"] < 30) & (df["concentration_score"] >= 50)
    ).astype(int)

    high_risk_count = df["eom_high_risk_flag"].sum()
    logger.debug("High risk entities identified: {} ({:.1f}%)", high_risk_count, 100 * high_risk_count / len(df))

    elapsed_time = time.time() - start_time
    logger.debug("EOM pattern classification completed in {:.2f}s - {} rows × {} columns", elapsed_time, len(df), len(df.columns))

    return df
