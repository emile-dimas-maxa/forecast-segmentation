"""Main pipeline orchestrator for EOM forecasting transformations."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig

from src.transformations.amount_clipping import clip_small_amounts
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

    def __init__(self, config: SegmentationConfig | None = None, save_intermediate: bool = False, output_dir: str | None = None):
        """Initialize pipeline with configuration.

        Args:
            config: Configuration object for the pipeline
            save_intermediate: Whether to save intermediate results to files
            output_dir: Directory to save intermediate results (defaults to 'intermediate_results')
        """
        self.config = config or SegmentationConfig()
        self.save_intermediate = save_intermediate
        self.output_dir = Path(output_dir) if output_dir else Path("intermediate_results")

        if self.save_intermediate:
            self.output_dir.mkdir(exist_ok=True)
            logger.info("Intermediate results will be saved to: {}", self.output_dir.absolute())

        logger.info("Initialized EOM Forecasting Pipeline with configuration: {}", self.config.__class__.__name__)

    def _save_intermediate_result(self, df: pd.DataFrame, step_name: str, step_number: int, suffix: str = "") -> None:
        """Save intermediate result to file if enabled.

        Args:
            df: DataFrame to save
            step_name: Name of the transformation step
            step_number: Step number in the pipeline
            suffix: Optional suffix to add to filename
        """
        if not self.save_intermediate:
            return

        filename = f"step_{step_number:02d}_{step_name}{suffix}.parquet"
        filepath = self.output_dir / filename

        try:
            # Try Parquet first, fallback to CSV
            try:
                df.to_parquet(filepath, index=False)
                logger.debug("Saved intermediate result: {} ({} rows × {} columns)", filename, len(df), len(df.columns))
            except ImportError:
                # Fallback to CSV if Parquet is not available
                csv_filepath = filepath.with_suffix(".csv")
                df.to_csv(csv_filepath, index=False)
                logger.debug(
                    "Saved intermediate result as CSV: {} ({} rows × {} columns)", csv_filepath.name, len(df), len(df.columns)
                )
        except Exception as e:
            logger.warning("Failed to save intermediate result {}: {}", filename, e)

    def list_intermediate_results(self) -> list[Path]:
        """List all saved intermediate result files.

        Returns:
            List of Path objects for saved intermediate files
        """
        if not self.save_intermediate or not self.output_dir.exists():
            return []

        # Look for both parquet and csv files
        parquet_files = list(self.output_dir.glob("step_*.parquet"))
        csv_files = list(self.output_dir.glob("step_*.csv"))
        return sorted(parquet_files + csv_files)

    def load_intermediate_result(self, step_number: int) -> pd.DataFrame | None:
        """Load a specific intermediate result by step number.

        Args:
            step_number: Step number to load (0-9)

        Returns:
            DataFrame if found, None otherwise
        """
        if not self.save_intermediate:
            logger.warning("Intermediate saving is not enabled")
            return None

        # Look for both parquet and csv files
        parquet_pattern = f"step_{step_number:02d}_*.parquet"
        csv_pattern = f"step_{step_number:02d}_*.csv"

        parquet_files = list(self.output_dir.glob(parquet_pattern))
        csv_files = list(self.output_dir.glob(csv_pattern))

        files = parquet_files + csv_files

        if not files:
            logger.warning("No intermediate result found for step {}", step_number)
            return None

        filepath = files[0]  # Take the first match
        try:
            if filepath.suffix == ".parquet":
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath)

            logger.info("Loaded intermediate result: {} ({} rows × {} columns)", filepath.name, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.error("Failed to load intermediate result {}: {}", filepath.name, e)
            return None

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

        # Save input data if intermediate saving is enabled
        if self.save_intermediate:
            self._save_intermediate_result(df, "input_data", 0)

        # Step 1: Base data preparation
        step_start = time.time()
        logger.debug("Step 1/10: Starting base data preparation")
        df = prepare_base_data(df, self.config)
        self._save_intermediate_result(df, "base_preparation", 1)
        logger.info(
            "Step 1/10: Base data preparation completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 2: Amount clipping
        step_start = time.time()
        logger.debug("Step 2/10: Starting amount clipping")
        df = clip_small_amounts(df, self.config)
        self._save_intermediate_result(df, "amount_clipping", 2)

        # Save clipping impact analysis if available
        if hasattr(df, "attrs") and "clipping_impact_analysis" in df.attrs:
            impact_df = df.attrs["clipping_impact_analysis"]
            if len(impact_df) > 0:
                self._save_intermediate_result(impact_df, "amount_clipping_impact", 2, suffix="_impact")

        logger.info(
            "Step 2/10: Amount clipping completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 3: Monthly aggregations
        step_start = time.time()
        logger.debug("Step 3/10: Starting monthly aggregations")
        df = create_monthly_aggregates(df, self.config)
        self._save_intermediate_result(df, "monthly_aggregation", 3)
        logger.info(
            "Step 3/10: Monthly aggregations completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 4: Rolling window features
        step_start = time.time()
        logger.debug("Step 4/10: Starting rolling window features")
        df = create_rolling_features(df, self.config)
        self._save_intermediate_result(df, "rolling_features", 4)
        logger.info(
            "Step 4/10: Rolling features completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 5: Portfolio metrics
        step_start = time.time()
        logger.debug("Step 5/10: Starting portfolio metrics calculation")
        df = calculate_portfolio_metrics(df, self.config)
        self._save_intermediate_result(df, "portfolio_metrics", 5)
        logger.info(
            "Step 5/10: Portfolio metrics completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 6: Pattern metrics
        step_start = time.time()
        logger.debug("Step 6/10: Starting pattern metrics calculation")
        df = calculate_pattern_metrics(df, self.config)
        self._save_intermediate_result(df, "pattern_metrics", 6)
        logger.info(
            "Step 6/10: Pattern metrics completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 7: Importance classification
        step_start = time.time()
        logger.debug("Step 7/10: Starting importance classification")
        df = classify_importance(df, self.config)
        self._save_intermediate_result(df, "importance_classification", 7)
        logger.info(
            "Step 7/10: Importance classification completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 8: EOM pattern classification
        step_start = time.time()
        logger.debug("Step 8/10: Starting EOM pattern classification")
        df = classify_eom_patterns(df, self.config)
        self._save_intermediate_result(df, "eom_pattern_classification", 8)
        logger.info(
            "Step 8/10: EOM pattern classification completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 9: General pattern classification
        step_start = time.time()
        logger.debug("Step 9/10: Starting general pattern classification")
        df = classify_general_patterns(df, self.config)
        self._save_intermediate_result(df, "general_pattern_classification", 9)
        logger.info(
            "Step 9/10: General pattern classification completed in {:.2f}s - Shape: {} rows × {} columns",
            time.time() - step_start,
            len(df),
            len(df.columns),
        )

        # Step 10: Final output with recommendations
        step_start = time.time()
        logger.debug("Step 10/10: Starting final output generation")
        df = create_final_output(df, self.config, target_month)
        self._save_intermediate_result(df, "final_output", 10)
        logger.info(
            "Step 10/10: Final output completed in {:.2f}s - Shape: {} rows × {} columns",
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

        # Log intermediate results summary
        if self.save_intermediate:
            saved_files = self.list_intermediate_results()
            logger.info("Saved {} intermediate result files to: {}", len(saved_files), self.output_dir.absolute())
            for filepath in saved_files:
                logger.debug("  - {}", filepath.name)

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
        logger.debug("Step 1/10: Base data preparation")
        results["base_data"] = prepare_base_data(df, self.config)
        logger.info(
            "Step 1/10: Base preparation - {} rows × {} columns ({:.2f}s)",
            len(results["base_data"]),
            len(results["base_data"].columns),
            time.time() - step_start,
        )

        # Step 2: Amount clipping
        step_start = time.time()
        logger.debug("Step 2/10: Amount clipping")
        results["amount_clipping"] = clip_small_amounts(results["base_data"], self.config)
        logger.info(
            "Step 2/10: Amount clipping - {} rows × {} columns ({:.2f}s)",
            len(results["amount_clipping"]),
            len(results["amount_clipping"].columns),
            time.time() - step_start,
        )

        # Step 3: Monthly aggregation
        step_start = time.time()
        logger.debug("Step 3/10: Monthly aggregation")
        results["monthly_aggregates"] = create_monthly_aggregates(results["amount_clipping"], self.config)
        logger.info(
            "Step 3/10: Monthly aggregation - {} rows × {} columns ({:.2f}s)",
            len(results["monthly_aggregates"]),
            len(results["monthly_aggregates"].columns),
            time.time() - step_start,
        )

        # Step 4: Rolling features
        step_start = time.time()
        logger.debug("Step 4/10: Rolling features")
        results["rolling_features"] = create_rolling_features(results["monthly_aggregates"], self.config)
        logger.info(
            "Step 4/10: Rolling features - {} rows × {} columns ({:.2f}s)",
            len(results["rolling_features"]),
            len(results["rolling_features"].columns),
            time.time() - step_start,
        )

        # Step 5: Portfolio metrics
        step_start = time.time()
        logger.debug("Step 5/10: Portfolio metrics")
        results["portfolio_metrics"] = calculate_portfolio_metrics(results["rolling_features"], self.config)
        logger.info(
            "Step 5/10: Portfolio metrics - {} rows × {} columns ({:.2f}s)",
            len(results["portfolio_metrics"]),
            len(results["portfolio_metrics"].columns),
            time.time() - step_start,
        )

        # Step 6: Pattern metrics
        step_start = time.time()
        logger.debug("Step 6/10: Pattern metrics")
        results["pattern_metrics"] = calculate_pattern_metrics(results["portfolio_metrics"], self.config)
        logger.info(
            "Step 6/10: Pattern metrics - {} rows × {} columns ({:.2f}s)",
            len(results["pattern_metrics"]),
            len(results["pattern_metrics"].columns),
            time.time() - step_start,
        )

        # Step 7: Importance classification
        step_start = time.time()
        logger.debug("Step 7/10: Importance classification")
        results["importance_classification"] = classify_importance(results["pattern_metrics"], self.config)
        logger.info(
            "Step 7/10: Importance classification - {} rows × {} columns ({:.2f}s)",
            len(results["importance_classification"]),
            len(results["importance_classification"].columns),
            time.time() - step_start,
        )

        # Step 8: EOM patterns
        step_start = time.time()
        logger.debug("Step 8/10: EOM patterns")
        results["eom_patterns"] = classify_eom_patterns(results["importance_classification"], self.config)
        logger.info(
            "Step 8/10: EOM patterns - {} rows × {} columns ({:.2f}s)",
            len(results["eom_patterns"]),
            len(results["eom_patterns"].columns),
            time.time() - step_start,
        )

        # Step 9: General patterns
        step_start = time.time()
        logger.debug("Step 9/10: General patterns")
        results["general_patterns"] = classify_general_patterns(results["eom_patterns"], self.config)
        logger.info(
            "Step 9/10: General patterns - {} rows × {} columns ({:.2f}s)",
            len(results["general_patterns"]),
            len(results["general_patterns"].columns),
            time.time() - step_start,
        )

        # Step 10: Final output
        step_start = time.time()
        logger.debug("Step 10/10: Final output")
        results["final_output"] = create_final_output(results["general_patterns"], self.config)
        logger.info(
            "Step 10/10: Final output - {} rows × {} columns ({:.2f}s)",
            len(results["final_output"]),
            len(results["final_output"].columns),
            time.time() - step_start,
        )

        total_time = time.time() - start_time
        logger.success("Step-by-step pipeline completed successfully in {:.2f}s", total_time)
        logger.info("All intermediate results stored in dictionary with {} keys", len(results))

        return results


def run_pipeline(
    df: pd.DataFrame,
    config: SegmentationConfig | None = None,
    target_month: str | None = "2025-07-01",
    save_intermediate: bool = False,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Convenience function to run the complete pipeline.

    Args:
        df: Input DataFrame with columns: dim_value, date, amount, is_last_work_day_of_month
        config: Optional configuration object
        target_month: Target month for final output filtering
        save_intermediate: Whether to save intermediate results to files
        output_dir: Directory to save intermediate results (defaults to 'intermediate_results')

    Returns:
        Transformed DataFrame with all features and classifications
    """
    logger.info("Running EOM forecasting pipeline via convenience function")
    pipeline = EOMForecastingPipeline(config, save_intermediate, output_dir)
    return pipeline.transform(df, target_month)
