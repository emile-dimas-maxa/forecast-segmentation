"""
Core grid search functionality.
"""

import random
from typing import Any

from tqdm import tqdm
from loguru import logger

from src.grid_search.backtest import run_full_backtest
from src.grid_search.evaluation import evaluate_predictions
from src.grid_search.io import print_summary, save_results
from src.grid_search.utils import create_empty_results, generate_model_combinations, split_predictions_for_grid_search


def run_selection_phase(
    full_data,
    segments: list[str],
    segment_col: str,
    target_col: str,
    date_col: str,
    dimensions: list[str],
    test_predictions: int,
    validation_predictions: int,
    forecast_horizon: int,
    input_steps: int,
    expanding_window: bool,
    stride: int,
    min_backtest_iterations: int,
    model_configs: dict[str, list[dict[str, Any]]],
    model_segment_mapping: dict[str, list[str]],
    unmapped_strategy: str,
    primary_metric: str,
) -> tuple[dict[str, Any], list[dict], Any]:
    """Run model selection phase using full backtesting approach"""
    logger.info("=" * 60)
    logger.info("Starting Model Selection Phase")
    logger.info("=" * 60)

    model_combinations = generate_model_combinations(segments, model_configs, model_segment_mapping, unmapped_strategy)

    max_combinations = 1  # Adjust based on computational resources
    if len(model_combinations) > max_combinations:
        logger.warning(f"Too many combinations ({len(model_combinations)}), sampling {max_combinations}")
        model_combinations = random.sample(model_combinations, max_combinations)

    print(model_combinations)

    selection_results = []

    for i, model_mapping in tqdm(enumerate(model_combinations)):
        logger.info(f"Evaluating combination {i + 1}/{len(model_combinations)}")

        try:
            all_predictions = run_full_backtest(
                model_mapping,
                full_data,
                segment_col,
                target_col,
                dimensions,
                forecast_horizon,
                input_steps,
                expanding_window,
                stride,
                min_backtest_iterations,
                date_col,
            )

            # Split predictions into validation and test sets
            validation_predictions, test_predictions = split_predictions_for_grid_search(
                all_predictions, date_col, test_predictions, validation_predictions
            )

            # Evaluate on validation set for selection
            result = evaluate_predictions(validation_predictions, "validation", dimensions, segment_col, date_col)

            if result is not None:
                # Store the model mapping and test predictions for later use
                result["model_mapping"] = model_mapping
                result["test_predictions"] = test_predictions
                selection_results.append(result)
                logger.info(f"  {primary_metric}: {result['metrics'][primary_metric]:.4f}")

        except Exception as e:
            logger.warning(f"Error evaluating combination {i + 1}: {e}")
            raise e

    if not selection_results:
        raise ValueError("No successful model evaluations in selection phase")

    # Find best model based on primary metric
    if primary_metric in ["mae", "mse", "rmse", "mape"]:
        # Lower is better
        best_result = min(selection_results, key=lambda x: x["metrics"][primary_metric])
    else:
        # Higher is better (e.g., r2)
        best_result = max(selection_results, key=lambda x: x["metrics"][primary_metric])

    logger.info(f"Best model selected with {primary_metric}: {best_result['metrics'][primary_metric]:.4f}")

    # Extract test predictions for the best model
    best_test_predictions = best_result["test_predictions"]

    return best_result, selection_results, best_test_predictions


def run_test_phase(test_predictions, dimensions: list[str], segment_col: str, date_col: str, primary_metric: str) -> dict[str, Any]:
    """Run final test phase using pre-computed predictions"""
    logger.info("=" * 60)
    logger.info("Starting Test Phase")
    logger.info("=" * 60)

    result = evaluate_predictions(test_predictions, "test", dimensions, segment_col, date_col)

    if result is None:
        raise ValueError("Failed to evaluate best model on test data")

    logger.info(f"Test {primary_metric}: {result['metrics'][primary_metric]:.4f}")

    return result


def run_grid_search(
    data,
    segment_col: str,
    target_col: str,
    date_col: str,
    dimensions: list[str],
    test_predictions: int = 6,
    validation_predictions: int = 6,
    forecast_horizon: int = 1,
    input_steps: int = 12,
    expanding_window: bool = True,
    stride: int = 1,
    min_backtest_iterations: int = 3,
    model_configs: dict[str, list[dict[str, Any]]] = None,
    model_segment_mapping: dict[str, list[str]] = None,
    unmapped_strategy: str = "all_models",
    primary_metric: str = "mae",
    output_dir: str = "outputs/grid_search",
    save_detailed_results: bool = True,
) -> dict:
    """
    Run complete grid search

    Args:
        data: DataFrame with the feature data
        segment_col: Column name for segments
        target_col: Column name for target variable
        date_col: Column name for date
        dimensions: List of dimension columns
        test_predictions: Number of predictions for test set
        validation_predictions: Number of predictions for validation set
        forecast_horizon: Forecast horizon for backtesting
        input_steps: Number of input steps for training
        expanding_window: Whether to use expanding window
        stride: Stride for backtesting
        min_backtest_iterations: Minimum number of backtest iterations
        model_configs: Dictionary of model configurations to test
        model_segment_mapping: Optional mapping of models to segments
        unmapped_strategy: Strategy for unmapped segments ("all_models" or "fallback")
        primary_metric: Primary metric for model selection
        output_dir: Directory to save results
        save_detailed_results: Whether to save detailed results

    Returns:
        Dictionary with grid search results
    """
    logger.info("=" * 80)
    logger.info("Starting Segmented Model Grid Search")
    logger.info("=" * 80)

    # Set default model configs if not provided
    if model_configs is None:
        raise ValueError("model_configs is required")

    results = create_empty_results()

    segments = sorted(data[segment_col].unique())
    logger.info(f"Found segments: {segments}")

    best_result, selection_results, best_test_predictions = run_selection_phase(
        data,
        segments,
        segment_col,
        target_col,
        date_col,
        dimensions,
        test_predictions,
        validation_predictions,
        forecast_horizon,
        input_steps,
        expanding_window,
        stride,
        min_backtest_iterations,
        model_configs,
        model_segment_mapping,
        unmapped_strategy,
        primary_metric,
    )
    results["selection_results"] = selection_results
    results["best_config"] = best_result["model_mapping"]
    results["selection_metrics"] = best_result["metrics"]

    # Run test phase using pre-computed predictions
    test_result = run_test_phase(best_test_predictions, dimensions, segment_col, date_col, primary_metric)
    results["test_results"] = [test_result]
    results["test_metrics"] = test_result["metrics"]

    # Save results
    save_results(
        results,
        output_dir,
        save_detailed_results,
        segment_col,
        target_col,
        date_col,
        dimensions,
        test_predictions,
        validation_predictions,
        forecast_horizon,
        input_steps,
        expanding_window,
        stride,
        min_backtest_iterations,
        model_configs,
        model_segment_mapping,
        primary_metric,
    )

    # Print summary
    print_summary(results, primary_metric)

    return results
