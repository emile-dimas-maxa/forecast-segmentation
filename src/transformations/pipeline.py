"""Main pipeline orchestrator for EOM forecasting transformations."""

from __future__ import annotations

import time

import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig

from src.transformations.base_preparation import prepare_base_data
from src.transformations.eom_pattern_classification import classify_eom_patterns
from src.transformations.final_output import create_final_output
from src.transformations.general_pattern_classification import classify_general_patterns
from src.transformations.importance_classification import classify_importance
from src.transformations.monthly_aggregation import create_monthly_aggregates
from src.transformations.pattern_metrics import calculate_pattern_metrics
from src.transformations.portfolio_metrics import calculate_portfolio_metrics
from src.transformations.rolling_features import create_rolling_features


class EOMForecastingPipeline:
    """Pipeline for EOM forecasting feature engineering."""

    def __init__(self, config: SegmentationConfig | None = None):
        """Initialize pipeline with configuration."""
        self.config = config or SegmentationConfig()
        logger.info("Initialized EOM Forecasting Pipeline with configuration: {}", self.config.__class__.__name__)

    def transform(self, df: pd.DataFrame, target_month: str | None = "2025-07-01") -> pd.DataFrame:
        """
        Execute the complete transformation pipeline.

        Args:
            df: Input DataFrame with columns: dim_value, date, amount, is_last_work_day_of_month
            target_month: Target month for final output filtering (format: 'YYYY-MM-DD')

        Returns:
            Transformed DataFrame with all features and classifications
        """
        start_time = time.time()
        initial_rows = len(df)
        initial_cols = len(df.columns)

        logger.info("Starting EOM forecasting pipeline transformation")
        logger.info("Input data shape: {} rows × {} columns", initial_rows, initial_cols)
        logger.info("Target month: {}", target_month)

        # Step 1: Base data preparation
        step_start = time.time()
        logger.debug("Step 1/9: Starting base data preparation")
        df = prepare_base_data(df, self.config)
        logger.info(
            "Step 1/9: Base data preparation completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 2: Monthly aggregations
        step_start = time.time()
        logger.debug("Step 2/9: Starting monthly aggregations")
        df = create_monthly_aggregates(df, self.config)
        logger.info(
            "Step 2/9: Monthly aggregations completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 3: Rolling window features
        step_start = time.time()
        logger.debug("Step 3/9: Starting rolling window features")
        df = create_rolling_features(df, self.config)
        logger.info(
            "Step 3/9: Rolling features completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 4: Portfolio metrics
        step_start = time.time()
        logger.debug("Step 4/9: Starting portfolio metrics calculation")
        df = calculate_portfolio_metrics(df, self.config)
        logger.info(
            "Step 4/9: Portfolio metrics completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 5: Pattern metrics
        step_start = time.time()
        logger.debug("Step 5/9: Starting pattern metrics calculation")
        df = calculate_pattern_metrics(df, self.config)
        logger.info(
            "Step 5/9: Pattern metrics completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 6: Importance classification
        step_start = time.time()
        logger.debug("Step 6/9: Starting importance classification")
        df = classify_importance(df, self.config)
        logger.info(
            "Step 6/9: Importance classification completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 7: EOM pattern classification
        step_start = time.time()
        logger.debug("Step 7/9: Starting EOM pattern classification")
        df = classify_eom_patterns(df, self.config)
        logger.info(
            "Step 7/9: EOM pattern classification completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 8: General pattern classification
        step_start = time.time()
        logger.debug("Step 8/9: Starting general pattern classification")
        df = classify_general_patterns(df, self.config)
        logger.info(
            "Step 8/9: General pattern classification completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 9: Final output with recommendations
        step_start = time.time()
        logger.debug("Step 9/9: Starting final output generation")
        df = create_final_output(df, self.config, target_month)
        logger.info(
            "Step 9/9: Final output completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        total_time = time.time() - start_time
        logger.success("Pipeline transformation completed successfully in {:.2f}s", total_time)
        logger.info(
            "Final output: {} rows × {} columns ({}% rows retained)",
            len(df),
            len(df.columns),
            round(100 * len(df) / initial_rows, 1),
        )

        return df

    def transform_step_by_step(self, df: pd.DataFrame) -> dict:
        """
        Execute pipeline step by step, returning intermediate results.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with results from each transformation step
        """
        start_time = time.time()
        initial_rows = len(df)

        logger.info("Starting step-by-step EOM forecasting pipeline transformation")
        logger.info("Input data shape: {} rows × {} columns", initial_rows, len(df.columns))

        results = {}

        # Step 1: Base preparation
        step_start = time.time()
        logger.debug("Step 1/9: Base data preparation")
        results["base_data"] = prepare_base_data(df, self.config)
        logger.info(
            "Step 1/9: Base preparation - {} rows × {} columns ({:.2f}s)",
            len(results["base_data"]),
            len(results["base_data"].columns),
            time.time() - step_start,
        )

        # Step 2: Monthly aggregation
        step_start = time.time()
        logger.debug("Step 2/9: Monthly aggregation")
        results["monthly_aggregates"] = create_monthly_aggregates(results["base_data"], self.config)
        logger.info(
            "Step 2/9: Monthly aggregation - {} rows × {} columns ({:.2f}s)",
            len(results["monthly_aggregates"]),
            len(results["monthly_aggregates"].columns),
            time.time() - step_start,
        )

        # Step 3: Rolling features
        step_start = time.time()
        logger.debug("Step 3/9: Rolling features")
        results["rolling_features"] = create_rolling_features(results["monthly_aggregates"], self.config)
        logger.info(
            "Step 3/9: Rolling features - {} rows × {} columns ({:.2f}s)",
            len(results["rolling_features"]),
            len(results["rolling_features"].columns),
            time.time() - step_start,
        )

        # Step 4: Portfolio metrics
        step_start = time.time()
        logger.debug("Step 4/9: Portfolio metrics")
        results["portfolio_metrics"] = calculate_portfolio_metrics(results["rolling_features"], self.config)
        logger.info(
            "Step 4/9: Portfolio metrics - {} rows × {} columns ({:.2f}s)",
            len(results["portfolio_metrics"]),
            len(results["portfolio_metrics"].columns),
            time.time() - step_start,
        )

        # Step 5: Pattern metrics
        step_start = time.time()
        logger.debug("Step 5/9: Pattern metrics")
        results["pattern_metrics"] = calculate_pattern_metrics(results["portfolio_metrics"], self.config)
        logger.info(
            "Step 5/9: Pattern metrics - {} rows × {} columns ({:.2f}s)",
            len(results["pattern_metrics"]),
            len(results["pattern_metrics"].columns),
            time.time() - step_start,
        )

        # Step 6: Importance classification
        step_start = time.time()
        logger.debug("Step 6/9: Importance classification")
        results["importance_classification"] = classify_importance(results["pattern_metrics"], self.config)
        logger.info(
            "Step 6/9: Importance classification - {} rows × {} columns ({:.2f}s)",
            len(results["importance_classification"]),
            len(results["importance_classification"].columns),
            time.time() - step_start,
        )

        # Step 7: EOM patterns
        step_start = time.time()
        logger.debug("Step 7/9: EOM patterns")
        results["eom_patterns"] = classify_eom_patterns(results["importance_classification"], self.config)
        logger.info(
            "Step 7/9: EOM patterns - {} rows × {} columns ({:.2f}s)",
            len(results["eom_patterns"]),
            len(results["eom_patterns"].columns),
            time.time() - step_start,
        )

        # Step 8: General patterns
        step_start = time.time()
        logger.debug("Step 8/9: General patterns")
        results["general_patterns"] = classify_general_patterns(results["eom_patterns"], self.config)
        logger.info(
            "Step 8/9: General patterns - {} rows × {} columns ({:.2f}s)",
            len(results["general_patterns"]),
            len(results["general_patterns"].columns),
            time.time() - step_start,
        )

        # Step 9: Final output
        step_start = time.time()
        logger.debug("Step 9/9: Final output")
        results["final_output"] = create_final_output(results["general_patterns"], self.config)
        logger.info(
            "Step 9/9: Final output - {} rows × {} columns ({:.2f}s)",
            len(results["final_output"]),
            len(results["final_output"].columns),
            time.time() - step_start,
        )

        total_time = time.time() - start_time
        logger.success("Step-by-step pipeline completed successfully in {:.2f}s", total_time)
        logger.info("All intermediate results stored in dictionary with {} keys", len(results))

        return results


def run_pipeline(
    df: pd.DataFrame, config: SegmentationConfig | None = None, target_month: str | None = "2025-07-01"
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
    logger.info("Running EOM forecasting pipeline via convenience function")
    pipeline = EOMForecastingPipeline(config)
    return pipeline.transform(df, target_month)
