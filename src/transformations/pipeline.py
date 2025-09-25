"""Main pipeline orchestrator for EOM forecasting transformations."""

import pandas as pd
from typing import Optional
from src.config.segmentation import SegmentationConfig
from .base_preparation import prepare_base_data
from .monthly_aggregation import create_monthly_aggregates
from .rolling_features import create_rolling_features
from .portfolio_metrics import calculate_portfolio_metrics
from .pattern_metrics import calculate_pattern_metrics
from .importance_classification import classify_importance
from .eom_pattern_classification import classify_eom_patterns
from .general_pattern_classification import classify_general_patterns
from .final_output import create_final_output


class EOMForecastingPipeline:
    """Pipeline for EOM forecasting feature engineering."""

    def __init__(self, config: Optional[SegmentationConfig] = None):
        """Initialize pipeline with configuration."""
        self.config = config or SegmentationConfig()

    def transform(self, df: pd.DataFrame, target_month: Optional[str] = "2025-07-01") -> pd.DataFrame:
        """
        Execute the complete transformation pipeline.

        Args:
            df: Input DataFrame with columns: dim_value, date, amount, is_last_work_day_of_month
            target_month: Target month for final output filtering (format: 'YYYY-MM-DD')

        Returns:
            Transformed DataFrame with all features and classifications
        """
        # Step 1: Base data preparation
        df = prepare_base_data(df, self.config)

        # Step 2: Monthly aggregations
        df = create_monthly_aggregates(df, self.config)

        # Step 3: Rolling window features
        df = create_rolling_features(df, self.config)

        # Step 4: Portfolio metrics
        df = calculate_portfolio_metrics(df, self.config)

        # Step 5: Pattern metrics
        df = calculate_pattern_metrics(df, self.config)

        # Step 6: Importance classification
        df = classify_importance(df, self.config)

        # Step 7: EOM pattern classification
        df = classify_eom_patterns(df, self.config)

        # Step 8: General pattern classification
        df = classify_general_patterns(df, self.config)

        # Step 9: Final output with recommendations
        df = create_final_output(df, self.config, target_month)

        return df

    def transform_step_by_step(self, df: pd.DataFrame) -> dict:
        """
        Execute pipeline step by step, returning intermediate results.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with results from each transformation step
        """
        results = {}

        # Step 1: Base preparation
        results["base_data"] = prepare_base_data(df, self.config)

        # Step 2: Monthly aggregation
        results["monthly_aggregates"] = create_monthly_aggregates(results["base_data"], self.config)

        # Step 3: Rolling features
        results["rolling_features"] = create_rolling_features(results["monthly_aggregates"], self.config)

        # Step 4: Portfolio metrics
        results["portfolio_metrics"] = calculate_portfolio_metrics(results["rolling_features"], self.config)

        # Step 5: Pattern metrics
        results["pattern_metrics"] = calculate_pattern_metrics(results["portfolio_metrics"], self.config)

        # Step 6: Importance classification
        results["importance_classification"] = classify_importance(results["pattern_metrics"], self.config)

        # Step 7: EOM patterns
        results["eom_patterns"] = classify_eom_patterns(results["importance_classification"], self.config)

        # Step 8: General patterns
        results["general_patterns"] = classify_general_patterns(results["eom_patterns"], self.config)

        # Step 9: Final output
        results["final_output"] = create_final_output(results["general_patterns"], self.config)

        return results


def run_pipeline(
    df: pd.DataFrame, config: Optional[SegmentationConfig] = None, target_month: Optional[str] = "2025-07-01"
) -> pd.DataFrame:
    """
    Convenience function to run the complete pipeline.

    Args:
        df: Input DataFrame with columns: dim_value, date, amount, is_last_work_day_of_month
        config: Optional configuration object
        target_month: Target month for final output filtering

    Returns:
        Transformed DataFrame with all features and classifications
    """
    pipeline = EOMForecastingPipeline(config)
    return pipeline.transform(df, target_month)
