"""
Pandas-based EOM pattern classification for interactive Streamlit app
Replicates the Snowpark logic for real-time archetype tuning
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.segmentation.config import SegmentationConfig


@dataclass
class ArchetypeConfig:
    """Configuration for pattern archetypes (centroids)"""

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


class PandasEOMClassifier:
    """Pandas-based EOM pattern classifier with configurable archetypes"""

    def __init__(self, config: SegmentationConfig, archetype_config: ArchetypeConfig = None):
        self.config = config
        self.archetype_config = archetype_config or ArchetypeConfig()

    def calculate_smooth_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate smooth scores for EOM patterns (pandas version)"""
        df = df.copy()

        # Regularity score: Sigmoid function for smooth transition
        df["regularity_score"] = 100 * (1 / (1 + np.exp(-self.config.sigmoid_steepness * (df["eom_frequency"] - 0.5))))

        # Stability score: Inverse exponential decay based on CV
        df["stability_score"] = 100 * np.exp(-self.config.stability_decay_rate * np.maximum(df["eom_cv"], 0))

        # Recency score: Exponential time decay
        def calculate_recency(row):
            if row.get("has_eom_history", 1) == 0:
                return 0
            if row.get("has_nonzero_eom", 0) == 1:
                return 100
            months_since = row.get("months_since_last_eom", 999)
            if months_since == 1:
                return 80
            elif months_since == 2:
                return 64
            elif months_since == 3:
                return 51
            else:
                return 100 * (self.config.recency_decay_rate ** max(4, min(24, months_since)))

        df["recency_score"] = df.apply(calculate_recency, axis=1)

        # Concentration score: Logistic curve
        df["concentration_score"] = 100 * (1 / (1 + np.exp(-self.config.concentration_steepness * (df["eom_concentration"] - 0.5))))

        # Volume importance score: Asymptotic growth
        total_portfolio_eom = df.get("total_portfolio_eom_volume", pd.Series([1] * len(df))).iloc[0] if len(df) > 0 else 1
        df["volume_importance_score"] = np.where(
            total_portfolio_eom > 0, 100 * (1 - np.exp(-self.config.volume_growth_rate * df["eom_importance_score"])), 0
        )

        return df

    def calculate_pattern_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distances to pattern archetypes (pandas version)"""
        df = df.copy()

        # Get archetype centroids
        archetypes = {
            "continuous_stable": self.archetype_config.continuous_stable,
            "continuous_volatile": self.archetype_config.continuous_volatile,
            "intermittent_active": self.archetype_config.intermittent_active,
            "intermittent_dormant": self.archetype_config.intermittent_dormant,
            "rare_recent": self.archetype_config.rare_recent,
            "rare_stale": self.archetype_config.rare_stale,
            "no_eom": self.archetype_config.no_eom,
        }

        weights = self.archetype_config.weights

        # Calculate distances for each archetype
        for pattern_name, (reg_target, stab_target, rec_target) in archetypes.items():
            # Apply weights to distance calculation
            reg_diff = (df["regularity_score"] - reg_target) * weights[0]
            stab_diff = (df["stability_score"] - stab_target) * weights[1]
            rec_diff = (df["recency_score"] - rec_target) * weights[2]

            # Special weight handling for rare patterns (stability less important)
            if "rare" in pattern_name:
                stab_diff *= 0.3

            df[f"dist_{pattern_name}"] = np.sqrt(reg_diff**2 + stab_diff**2 + rec_diff**2)

        # Special case for emerging pattern
        df["dist_emerging"] = np.where(df["months_of_history"] <= 3, 0, 999)

        return df

    def calculate_pattern_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert distances to probabilities using softmax (pandas version)"""
        df = df.copy()

        # Distance columns
        distance_cols = [
            "dist_continuous_stable",
            "dist_continuous_volatile",
            "dist_intermittent_active",
            "dist_intermittent_dormant",
            "dist_rare_recent",
            "dist_rare_stale",
        ]

        # Calculate softmax denominator
        softmax_terms = []
        for col in distance_cols:
            softmax_terms.append(np.exp(-df[col] / self.config.pattern_temperature))

        # Add conditional terms
        no_eom_term = np.where(df.get("has_eom_history", 1) == 0, np.exp(-df["dist_no_eom"] / self.config.pattern_temperature), 0)
        emerging_term = np.where(df["months_of_history"] <= 3, np.exp(-df["dist_emerging"] / self.config.pattern_temperature), 0)

        softmax_terms.extend([no_eom_term, emerging_term])
        df["softmax_denominator"] = sum(softmax_terms)

        # Calculate probabilities
        pattern_names = [
            "continuous_stable",
            "continuous_volatile",
            "intermittent_active",
            "intermittent_dormant",
            "rare_recent",
            "rare_stale",
        ]

        for pattern in pattern_names:
            df[f"prob_{pattern}"] = np.where(
                df["months_of_history"] <= 3,
                0,
                np.exp(-df[f"dist_{pattern}"] / self.config.pattern_temperature) / df["softmax_denominator"],
            )

        df["prob_no_eom"] = np.where(
            df.get("has_eom_history", 1) == 0,
            np.exp(-df["dist_no_eom"] / self.config.pattern_temperature) / df["softmax_denominator"],
            0,
        )

        df["prob_emerging"] = np.where(
            df["months_of_history"] <= 3,
            np.exp(-df["dist_emerging"] / self.config.pattern_temperature) / df["softmax_denominator"],
            0,
        )

        return df

    def classify_eom_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final EOM pattern classification (pandas version)"""
        df = df.copy()

        # Pattern probability columns
        prob_cols = [
            "prob_continuous_stable",
            "prob_continuous_volatile",
            "prob_intermittent_active",
            "prob_intermittent_dormant",
            "prob_rare_recent",
            "prob_rare_stale",
        ]

        def get_primary_pattern(row):
            if row["months_of_history"] <= 3:
                return "EMERGING"
            if row.get("has_eom_history", 1) == 0 and row["months_of_history"] >= 6:
                return "NO_EOM"

            # Find pattern with highest probability
            max_prob = 0
            max_pattern = "RARE_STALE"  # default

            pattern_mapping = {
                "prob_continuous_stable": "CONTINUOUS_STABLE",
                "prob_continuous_volatile": "CONTINUOUS_VOLATILE",
                "prob_intermittent_active": "INTERMITTENT_ACTIVE",
                "prob_intermittent_dormant": "INTERMITTENT_DORMANT",
                "prob_rare_recent": "RARE_RECENT",
                "prob_rare_stale": "RARE_STALE",
            }

            for prob_col, pattern_name in pattern_mapping.items():
                if row[prob_col] > max_prob:
                    max_prob = row[prob_col]
                    max_pattern = pattern_name

            return max_pattern

        df["eom_pattern_primary"] = df.apply(get_primary_pattern, axis=1)

        # Pattern confidence (highest probability)
        df["eom_pattern_confidence"] = df[prob_cols].max(axis=1)

        # Handle special cases
        mask_emerging = df["months_of_history"] <= 3
        mask_no_eom = df.get("has_eom_history", 1) == 0
        df.loc[mask_emerging | mask_no_eom, "eom_pattern_confidence"] = 1.0

        # Classification entropy (uncertainty)
        def calculate_entropy(row):
            probs = [row[col] for col in prob_cols if row[col] > 0]
            if not probs:
                return 0
            return -sum(p * np.log(p) for p in probs if p > 0)

        df["classification_entropy"] = df.apply(calculate_entropy, axis=1)

        return df

    def classify_general_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """General timeseries pattern classification (pandas version)"""
        df = df.copy()

        def get_general_pattern(row):
            if row.get("months_inactive", 0) >= self.config.inactive_months:
                return "INACTIVE"
            if row["months_of_history"] <= self.config.emerging_months:
                return "EMERGING"
            if (
                row.get("quarter_end_concentration", 0) >= self.config.ts_seasonal_concentration_threshold
                or row.get("year_end_concentration", 0) >= 0.5
            ):
                return "HIGHLY_SEASONAL"
            if row.get("activity_rate", 1) <= self.config.ts_intermittent_threshold or row.get("active_months_12m", 12) <= 4:
                return "INTERMITTENT"
            if (
                row.get("monthly_cv", 0) >= self.config.ts_high_volatility_threshold
                and row.get("transaction_regularity", 0) >= self.config.ts_medium_regularity_threshold
            ):
                return "VOLATILE"
            if (
                row.get("monthly_cv", 0) >= self.config.ts_medium_volatility_threshold
                and row.get("transaction_regularity", 0) >= self.config.ts_medium_regularity_threshold
            ):
                return "MODERATELY_VOLATILE"
            if (
                row.get("monthly_cv", 0) < self.config.ts_medium_volatility_threshold
                and row.get("transaction_regularity", 0) >= self.config.ts_high_regularity_threshold
            ):
                return "STABLE"
            if (
                row.get("transaction_dispersion", 10) < 5
                and row.get("transaction_regularity", 0) >= self.config.ts_medium_regularity_threshold
            ):
                return "CONCENTRATED"
            if (
                row.get("transaction_dispersion", 0) >= 8
                and row.get("transaction_regularity", 0) >= self.config.ts_medium_regularity_threshold
            ):
                return "DISTRIBUTED"
            return "MIXED"

        df["general_pattern"] = df.apply(get_general_pattern, axis=1)
        return df

    def classify_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Importance tier classification (pandas version)"""
        df = df.copy()

        # Overall importance tier
        def get_overall_importance(row):
            if (
                row.get("cumulative_overall_portfolio_pct", 1) <= self.config.overall_critical_percentile
                or row.get("rolling_avg_monthly_volume", 0) >= self.config.critical_monthly_avg_threshold
                or row.get("rolling_max_transaction", 0) >= self.config.critical_max_transaction_threshold
            ):
                return "CRITICAL"
            elif (
                row.get("cumulative_overall_portfolio_pct", 1) <= self.config.overall_high_percentile
                or row.get("rolling_avg_monthly_volume", 0) >= self.config.high_monthly_avg_threshold
                or row.get("rolling_max_transaction", 0) >= self.config.high_max_transaction_threshold
            ):
                return "HIGH"
            elif (
                row.get("cumulative_overall_portfolio_pct", 1) <= self.config.overall_medium_percentile
                or row.get("rolling_avg_monthly_volume", 0) >= self.config.medium_monthly_avg_threshold
                or row.get("rolling_max_transaction", 0) >= self.config.medium_max_transaction_threshold
            ):
                return "MEDIUM"
            else:
                return "LOW"

        df["overall_importance_tier"] = df.apply(get_overall_importance, axis=1)

        # EOM importance tier
        def get_eom_importance(row):
            if (
                row.get("cumulative_eom_portfolio_pct", 1) <= self.config.eom_critical_percentile
                or row.get("rolling_avg_nonzero_eom", 0) >= self.config.critical_eom_monthly_threshold
                or row.get("rolling_max_eom", 0) >= self.config.critical_max_eom_threshold
            ):
                return "CRITICAL"
            elif (
                row.get("cumulative_eom_portfolio_pct", 1) <= self.config.eom_high_percentile
                or row.get("rolling_avg_nonzero_eom", 0) >= self.config.high_eom_monthly_threshold
                or row.get("rolling_max_eom", 0) >= self.config.high_max_eom_threshold
            ):
                return "HIGH"
            elif (
                row.get("cumulative_eom_portfolio_pct", 1) <= self.config.eom_medium_percentile
                or (
                    row.get("rolling_avg_nonzero_eom", 0) >= self.config.medium_eom_monthly_threshold
                    and row.get("total_nonzero_eom_count", 0) >= 3
                )
                or row.get("rolling_max_eom", 0) >= self.config.medium_max_eom_threshold
            ):
                return "MEDIUM"
            elif row.get("rolling_eom_volume_12m", 0) > 0 or row.get("total_nonzero_eom_count", 0) > 0:
                return "LOW"
            else:
                return "NONE"

        df["eom_importance_tier"] = df.apply(get_eom_importance, axis=1)

        return df

    def run_full_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the complete EOM classification pipeline"""
        df = self.calculate_smooth_scores(df)
        df = self.calculate_pattern_distances(df)
        df = self.calculate_pattern_probabilities(df)
        df = self.classify_eom_patterns(df)
        df = self.classify_importance(df)

        # Create segment names (EOM-focused)
        df["segment_name"] = df.apply(lambda row: f"{row['eom_importance_tier']}_EOM_{row['eom_pattern']}", axis=1)

        return df

    def update_archetypes(self, new_archetype_config: ArchetypeConfig):
        """Update archetype configuration for real-time tuning"""
        self.archetype_config = new_archetype_config
