"""
Grid Search for Segmented Forecasting Models

This module implements a grid search functionality that:
1. Tests multiple combinations of models per segment
2. Performs backtesting using expanding windows
3. Uses first x months for model selection
4. Uses last n-x months for final performance testing
"""

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.forecast.evaluate import calculate_regression_metrics, evaluate_forecast
from src.forecast.segmented import SegmentedForecastModel
from src.splitter import TimeSeriesBacktest


@dataclass
class GridSearchConfig:
    """Configuration for grid search"""

    # Data configuration
    feature_file_path: str = "outputs/feature_df.csv"
    segment_col: str = "segment_name"
    target_col: str = "target_eom_amount"  # Based on feature pipeline output
    date_col: str = "forecast_month"
    dimensions: list[str] = field(default_factory=lambda: ["dim_value"])  # Adjust based on your data

    # Grid search configuration - based on number of predictions
    test_predictions: int = 6  # Number of predictions for test set (from n-x to n)
    validation_predictions: int = 6  # Number of predictions for validation set (from n-x-y to n-x)

    # Backtesting configuration
    forecast_horizon: int = 1
    input_steps: int = 12  # Use 12 months of history for training
    expanding_window: bool = True
    stride: int = 1
    min_backtest_iterations: int = 3

    # Model configurations to test
    model_configs: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {
            "arima": [
                {"type": "arima", "params": {"order": (1, 1, 1)}},
                {"type": "arima", "params": {"order": (2, 1, 1)}},
                {"type": "arima", "params": {"order": (1, 1, 2)}},
            ],
            "moving_average": [
                {"type": "moving_average", "params": {"window": 3}},
                {"type": "moving_average", "params": {"window": 6}},
                {"type": "moving_average", "params": {"window": 12}},
            ],
            "null": [
                {"type": "null", "params": {}},
            ],
        }
    )

    # Model-to-segment mapping (optional - if None, tests all models on all segments)
    model_segment_mapping: dict[str, list[str]] | None = None  # {"model_name": ["segment1", "segment2"]}

    # Evaluation metrics (primary metric used for selection)
    primary_metric: str = "mae"  # mae, rmse, mape, r2

    # Output configuration
    output_dir: str = "outputs/grid_search"
    save_detailed_results: bool = True


# Results are now returned as dictionaries instead of a class
def create_empty_results() -> dict:
    """Create empty results dictionary"""
    return {
        "selection_results": [],
        "test_results": [],
        "best_config": None,
        "best_model": None,
        "selection_metrics": None,
        "test_metrics": None,
    }


def load_data(config: GridSearchConfig) -> pd.DataFrame:
    """Load feature data"""
    logger.info(f"Loading data from {config.feature_file_path}")
    df = pd.read_csv(config.feature_file_path)

    # Convert date column to datetime
    df[config.date_col] = pd.to_datetime(df[config.date_col])

    logger.info(f"Loaded {len(df)} rows with {df[config.segment_col].nunique()} segments")
    logger.info(f"Date range: {df[config.date_col].min()} to {df[config.date_col].max()}")

    return df


def generate_model_combinations(config: GridSearchConfig, segments: list[str]) -> list[dict[str, dict[str, Any]]]:
    """Generate model combinations respecting model-to-segment mapping"""
    logger.info("Generating model combinations...")

    if config.model_segment_mapping is None:
        # Original behavior: test all models on all segments
        logger.info("No model-segment mapping specified, testing all models on all segments")

        all_configs = []
        for configs in config.model_configs.values():
            all_configs.extend(configs)

        combinations = []
        for combo in itertools.product(all_configs, repeat=len(segments)):
            model_mapping = {segment: model_config for segment, model_config in zip(segments, combo, strict=True)}
            combinations.append(model_mapping)

        logger.info(f"Generated {len(combinations)} model combinations for {len(segments)} segments")
        return combinations

    else:
        # New behavior: respect model-to-segment mapping
        logger.info("Using model-segment mapping to generate combinations")

        # Validate mapping
        mapped_segments = set()
        for model_name, segment_list in config.model_segment_mapping.items():
            if model_name not in config.model_configs:
                raise ValueError(f"Model '{model_name}' in mapping not found in model_configs")
            mapped_segments.update(segment_list)

        # Check for segments not in mapping
        unmapped_segments = set(segments) - mapped_segments
        if unmapped_segments:
            logger.info(f"Segments not in mapping will use fallback model: {sorted(unmapped_segments)}")

        # Generate combinations based on mapping
        segment_model_options = {}
        fallback_model = {"type": "moving_average", "params": {"window": 3}}

        for segment in segments:
            segment_model_options[segment] = []

            # Find which models can be applied to this segment
            for model_name, segment_list in config.model_segment_mapping.items():
                if segment in segment_list:
                    segment_model_options[segment].extend(config.model_configs[model_name])

            # If no models specified for this segment, use fallback
            if not segment_model_options[segment]:
                segment_model_options[segment] = [fallback_model]

        # Generate all combinations
        combinations = []
        segment_names = list(segments)
        segment_options = [segment_model_options[seg] for seg in segment_names]

        for combo in itertools.product(*segment_options):
            model_mapping = {segment: model_config for segment, model_config in zip(segment_names, combo, strict=True)}
            combinations.append(model_mapping)

        logger.info(f"Generated {len(combinations)} model combinations based on segment mapping")

        # Log mapping summary
        for segment in segments:
            num_options = len(segment_model_options[segment])
            logger.info(f"  {segment}: {num_options} model options")

        return combinations


def split_predictions_for_grid_search(config: GridSearchConfig, all_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split predictions into validation and test sets based on prediction count"""

    # Sort predictions by date to ensure correct ordering
    all_predictions = all_predictions.sort_values(config.date_col).reset_index(drop=True)

    n_predictions = len(all_predictions)

    if n_predictions < config.test_predictions + config.validation_predictions:
        raise ValueError(
            f"Not enough predictions: need at least {config.test_predictions + config.validation_predictions} "
            f"predictions, got {n_predictions}"
        )

    # Test set: from (n-x) to n
    test_start_idx = n_predictions - config.test_predictions
    test_predictions = all_predictions.iloc[test_start_idx:].copy()

    # Validation set: from (n-x-y) to (n-x)
    validation_start_idx = n_predictions - config.test_predictions - config.validation_predictions
    validation_end_idx = n_predictions - config.test_predictions
    validation_predictions = all_predictions.iloc[validation_start_idx:validation_end_idx].copy()

    logger.info(f"Total predictions: {n_predictions}")
    logger.info(
        f"Validation set: {len(validation_predictions)} predictions (indices {validation_start_idx} to {validation_end_idx - 1})"
    )
    logger.info(f"Test set: {len(test_predictions)} predictions (indices {test_start_idx} to {n_predictions - 1})")

    return validation_predictions, test_predictions


def run_full_backtest(config: GridSearchConfig, model_mapping: dict[str, dict[str, Any]], data: pd.DataFrame) -> pd.DataFrame:
    """Run full backtesting on the entire dataset and return all predictions"""

    # Create splitter
    splitter = TimeSeriesBacktest(
        forecast_horizon=config.forecast_horizon,
        input_steps=config.input_steps,
        expanding_window=config.expanding_window,
        stride=config.stride,
        date_column=config.date_col,
        min_backtest_iterations=config.min_backtest_iterations,
    )

    # Check if we have enough data for backtesting
    try:
        n_splits = splitter.number_of_splits(data)
        if n_splits < config.min_backtest_iterations:
            raise ValueError(f"Not enough data for backtesting: {n_splits} splits available, need {config.min_backtest_iterations}")
    except Exception as e:
        raise ValueError(f"Error checking splits: {e}") from e

    all_predictions = []

    # Perform backtesting
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data)):
        try:
            train_df = data.iloc[train_idx].copy()
            test_df = data.iloc[test_idx].copy()

            # Create and fit model
            model = SegmentedForecastModel(
                segment_col=config.segment_col,
                target_col=config.target_col,
                dimensions=config.dimensions,
                model_mapping=model_mapping,
                fallback_model={"type": "moving_average", "params": {"window": 3}},
            )

            model.fit(train_df)

            # Generate predictions
            predictions = model.predict(context=None, model_input=test_df)

            # Add actual values and metadata
            predictions["actual"] = test_df[config.target_col].values
            predictions["fold"] = fold_idx

            # Ensure we have the date column for sorting
            if config.date_col not in predictions.columns:
                predictions[config.date_col] = test_df[config.date_col].values

            all_predictions.append(predictions)

        except Exception as e:
            logger.warning(f"Error in fold {fold_idx}: {e}")
            continue

    if not all_predictions:
        raise ValueError("No successful predictions generated during backtesting")

    # Combine all predictions and sort by date
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_predictions = combined_predictions.sort_values(config.date_col).reset_index(drop=True)

    return combined_predictions


def evaluate_predictions(config: GridSearchConfig, predictions: pd.DataFrame, period_name: str) -> dict[str, Any]:
    """Evaluate predictions using the existing evaluation function"""

    if len(predictions) == 0:
        logger.warning(f"No predictions to evaluate for {period_name}")
        return None

    try:
        # Use the existing evaluation function
        evaluation_df = evaluate_forecast(
            data=predictions,
            dim_values_col=config.dimensions[0] if config.dimensions else "dim_value",
            actual_col="actual",
            prediction_col="prediction",
            segment_col=config.segment_col,
        )

        # Extract overall metrics (you can modify this based on what metrics you want to focus on)
        overall_metrics = evaluation_df[evaluation_df[config.segment_col] == "OVERALL"]

        if len(overall_metrics) == 0:
            # Fallback to simple metrics if no overall metrics available
            metrics = calculate_regression_metrics(predictions["actual"], predictions["prediction"])
        else:
            # Use the first overall metric row (you might want to specify direction here)
            metrics = overall_metrics.iloc[0].to_dict()

        result = {
            "metrics": metrics,
            "evaluation_df": evaluation_df,
            "n_predictions": len(predictions),
            "period": period_name,
        }

        return result

    except Exception as e:
        logger.warning(f"Error evaluating predictions for {period_name}: {e}")
        # Fallback to simple metrics
        metrics = calculate_regression_metrics(predictions["actual"], predictions["prediction"])
        return {
            "metrics": metrics,
            "evaluation_df": None,
            "n_predictions": len(predictions),
            "period": period_name,
        }


def run_selection_phase(
    config: GridSearchConfig, full_data: pd.DataFrame, segments: list[str]
) -> tuple[dict[str, Any], list[dict], pd.DataFrame]:
    """Run model selection phase using full backtesting approach"""
    logger.info("=" * 60)
    logger.info("Starting Model Selection Phase")
    logger.info("=" * 60)

    model_combinations = generate_model_combinations(config, segments)

    # Limit combinations for practical reasons (can be very large)
    max_combinations = 100  # Adjust based on computational resources
    if len(model_combinations) > max_combinations:
        logger.warning(f"Too many combinations ({len(model_combinations)}), sampling {max_combinations}")
        import random

        model_combinations = random.sample(model_combinations, max_combinations)

    selection_results = []

    for i, model_mapping in enumerate(model_combinations):
        logger.info(f"Evaluating combination {i + 1}/{len(model_combinations)}")

        try:
            # Run full backtesting
            all_predictions = run_full_backtest(config, model_mapping, full_data)

            # Split predictions into validation and test sets
            validation_predictions, test_predictions = split_predictions_for_grid_search(config, all_predictions)

            # Evaluate on validation set for selection
            result = evaluate_predictions(config, validation_predictions, "validation")

            if result is not None:
                # Store the model mapping and test predictions for later use
                result["model_mapping"] = model_mapping
                result["test_predictions"] = test_predictions
                selection_results.append(result)
                logger.info(f"  {config.primary_metric}: {result['metrics'][config.primary_metric]:.4f}")

        except Exception as e:
            logger.warning(f"Error evaluating combination {i + 1}: {e}")
            continue

    if not selection_results:
        raise ValueError("No successful model evaluations in selection phase")

    # Find best model based on primary metric
    if config.primary_metric in ["mae", "mse", "rmse", "mape"]:
        # Lower is better
        best_result = min(selection_results, key=lambda x: x["metrics"][config.primary_metric])
    else:
        # Higher is better (e.g., r2)
        best_result = max(selection_results, key=lambda x: x["metrics"][config.primary_metric])

    logger.info(f"Best model selected with {config.primary_metric}: {best_result['metrics'][config.primary_metric]:.4f}")

    # Extract test predictions for the best model
    best_test_predictions = best_result["test_predictions"]

    return best_result, selection_results, best_test_predictions


def run_test_phase(config: GridSearchConfig, test_predictions: pd.DataFrame) -> dict[str, Any]:
    """Run final test phase using pre-computed predictions"""
    logger.info("=" * 60)
    logger.info("Starting Test Phase")
    logger.info("=" * 60)

    result = evaluate_predictions(config, test_predictions, "test")

    if result is None:
        raise ValueError("Failed to evaluate best model on test data")

    logger.info(f"Test {config.primary_metric}: {result['metrics'][config.primary_metric]:.4f}")

    return result


def save_results(config: GridSearchConfig, results: dict):
    """Save grid search results"""
    logger.info("Saving results...")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = Path(config.output_dir) / "config.json"
    with open(config_path, "w") as f:
        # Convert dataclass to dict for JSON serialization
        config_dict = {
            "feature_file_path": config.feature_file_path,
            "segment_col": config.segment_col,
            "target_col": config.target_col,
            "date_col": config.date_col,
            "dimensions": config.dimensions,
            "test_predictions": config.test_predictions,
            "validation_predictions": config.validation_predictions,
            "forecast_horizon": config.forecast_horizon,
            "input_steps": config.input_steps,
            "expanding_window": config.expanding_window,
            "stride": config.stride,
            "min_backtest_iterations": config.min_backtest_iterations,
            "model_configs": config.model_configs,
            "model_segment_mapping": config.model_segment_mapping,
            "primary_metric": config.primary_metric,
        }
        json.dump(config_dict, f, indent=2)

    # Save best model configuration
    best_config_path = Path(config.output_dir) / "best_model_config.json"
    with open(best_config_path, "w") as f:
        json.dump(results["best_config"], f, indent=2)

    # Save detailed results if requested
    if config.save_detailed_results:
        # Selection results
        selection_df = pd.DataFrame(
            [
                {
                    "combination_id": i,
                    "period": "selection",
                    **result["metrics"],
                    "n_folds": result["n_folds"],
                    "n_predictions": result["n_predictions"],
                }
                for i, result in enumerate(results["selection_results"])
            ]
        )
        selection_df.to_csv(Path(config.output_dir) / "selection_results.csv", index=False)

        # Test results
        if results["test_results"]:
            test_df = pd.DataFrame(
                [
                    {
                        "period": "test",
                        **results["test_metrics"],
                        "n_folds": results["test_results"][0]["n_folds"],
                        "n_predictions": results["test_results"][0]["n_predictions"],
                    }
                ]
            )
            test_df.to_csv(Path(config.output_dir) / "test_results.csv", index=False)

    # Save summary
    summary = {
        "selection_metrics": results["selection_metrics"],
        "test_metrics": results["test_metrics"],
        "best_model_config": results["best_config"],
        "n_combinations_tested": len(results["selection_results"]),
    }

    summary_path = Path(config.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {config.output_dir}")


def print_summary(config: GridSearchConfig, results: dict):
    """Print summary of results"""
    logger.info("=" * 80)
    logger.info("Grid Search Summary")
    logger.info("=" * 80)

    logger.info(f"Combinations tested: {len(results['selection_results'])}")
    logger.info(f"Primary metric: {config.primary_metric}")

    if results["selection_metrics"]:
        logger.info("\nSelection Phase Results:")
        for metric, value in results["selection_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

    if results["test_metrics"]:
        logger.info("\nTest Phase Results:")
        for metric, value in results["test_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

    if results["best_config"]:
        logger.info("\nBest Model Configuration:")
        for segment, model_config in results["best_config"].items():
            logger.info(f"  {segment}: {model_config['type']} {model_config.get('params', {})}")

    logger.info("=" * 80)


def run_grid_search(config: GridSearchConfig) -> dict:
    """Run complete grid search"""
    logger.info("=" * 80)
    logger.info("Starting Segmented Model Grid Search")
    logger.info("=" * 80)

    # Initialize results
    results = create_empty_results()

    # Load data
    df = load_data(config)

    # Get unique segments
    segments = sorted(df[config.segment_col].unique())
    logger.info(f"Found segments: {segments}")

    # Run selection phase (this does full backtesting and splits predictions)
    best_result, selection_results, best_test_predictions = run_selection_phase(config, df, segments)
    results["selection_results"] = selection_results
    results["best_config"] = best_result["model_mapping"]
    results["selection_metrics"] = best_result["metrics"]

    # Run test phase using pre-computed predictions
    test_result = run_test_phase(config, best_test_predictions)
    results["test_results"] = [test_result]
    results["test_metrics"] = test_result["metrics"]

    # Save results
    save_results(config, results)

    # Print summary
    print_summary(config, results)

    return results


# Backward compatibility wrapper (optional)
class SegmentedModelGridSearch:
    """Backward compatibility wrapper for the functional grid search"""

    def __init__(self, config: GridSearchConfig):
        self.config = config

    def run(self) -> dict:
        """Run grid search using functional approach"""
        return run_grid_search(self.config)


def main():
    """Main function to run grid search"""

    # Create configuration
    config = GridSearchConfig(
        feature_file_path="outputs/feature_df.csv",
        segment_col="segment_name",  # Adjust based on your data
        target_col="target_eom_amount",  # Adjust based on your actual target column
        date_col="forecast_month",
        dimensions=["dim_value"],  # Adjust based on your data structure
        test_predictions=6,
        validation_predictions=6,
        input_steps=12,
        min_backtest_iterations=3,
        primary_metric="mae",
    )

    # Run grid search
    results = run_grid_search(config)

    return results


if __name__ == "__main__":
    main()
