"""
Configuration for feature pipeline
"""

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """
    Configuration for feature aggregation pipeline
    """

    # Importance tiers that should be kept as individual dim_values
    keep_individual_eom_tiers: list[str] = None
    keep_individual_overall_tiers: list[str] = None

    # Custom aggregation rules
    others_in_suffix: str = "::IN"
    others_out_suffix: str = "::OUT"
    aggregated_in_name: str = "others::IN"
    aggregated_out_name: str = "others::OUT"

    # Output options
    include_aggregation_metadata: bool = True

    def __post_init__(self):
        """Set default values after initialization"""
        if self.keep_individual_eom_tiers is None:
            self.keep_individual_eom_tiers = ["CRITICAL", "HIGH", "MEDIUM"]

        if self.keep_individual_overall_tiers is None:
            self.keep_individual_overall_tiers = ["CRITICAL"]
