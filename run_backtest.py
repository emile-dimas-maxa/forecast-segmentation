"""
Comprehensive backtesting script for segmented forecasting models.

This script consolidates all backtesting functionality into a single, clean interface:
- Loads data from feature_df.csv
- Generates model combinations (all segments, segment-specific, forced mappings)
- Runs time series backtesting with configurable parameters
- Evaluates predictions with train/validation/test splits
- Saves results with caching to avoid recomputation
- Provides comprehensive reporting

Usage:
    python run_backtest.py [--config CONFIG_FILE] [--data DATA_FILE] [--output OUTPUT_FILE]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.forecast.evaluate import evaluate_forecast_simple
from src.new_forecast import NewSegmentedForecastModel
from src.splitter import TimeSeriesBacktest


def load_feature_data(data_path: str) -> pd.DataFrame:
    """Load and prepare feature data for backtesting."""
    logger.info(f"Loading data from {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df["forecast_month"] = pd.to_datetime(df["forecast_month"])

    logger.info(f"Loaded {len(df)} rows with {df['eom_pattern_primary'].nunique()} segments")
    logger.info(f"Date range: {df['forecast_month'].min()} to {df['forecast_month'].max()}")
    logger.info(f"Segments: {df['eom_pattern_primary'].unique().tolist()}")

    return df


def generate_model_combinations(
    data: pd.DataFrame, sample_size: int = 100, fixed_segments: dict[str, dict] = None
) -> list[dict[str, Any]]:
    """Generate model combinations for backtesting based on actual data segments.

    Args:
        data: DataFrame containing the data with segments
        sample_size: Number of combinations to sample (default: 100). If None, returns all combinations.
        fixed_segments: Dict mapping segment names to fixed model configs. These segments won't be included in combinations.
    """
    import itertools
    import random

    # Define all available models
    all_models = [
        {"name": "null", "description": "Null model (baseline)"},
        {"name": "moving_average", "window": 3, "description": "Moving Average (window=3)"},
        {"name": "arima", "order": [1, 1, 1], "description": "ARIMA(1,1,1)"},
        {"name": "net_arima", "order": [1, 1, 1], "description": "Net ARIMA (forecasts net value)"},
        {"name": "net_moving_average", "window": 3, "description": "Net Moving Average (forecasts net value)"},
        {"name": "direction_arima", "order": [1, 1, 1], "description": "Direction ARIMA (forecasts credit/debit separately)"},
        {
            "name": "direction_moving_average",
            "window": 3,
            "description": "Direction Moving Average (forecasts credit/debit separately)",
        },
        {"name": "xgboost", "description": "XGBoost (gradient boosting)"},
        {"name": "random_forest", "description": "Random Forest (ensemble method)"},
    ]

    # Extract segments dynamically from the data
    all_segments = sorted(data["eom_pattern_primary"].unique().tolist())
    logger.info(f"Found {len(all_segments)} segments in data: {all_segments}")

    # Handle fixed segments
    if fixed_segments is None:
        fixed_segments = {}

    # Separate fixed and variable segments
    variable_segments = [seg for seg in all_segments if seg not in fixed_segments]
    fixed_segment_names = [seg for seg in all_segments if seg in fixed_segments]

    logger.info(f"Fixed segments ({len(fixed_segment_names)}): {fixed_segment_names}")
    logger.info(f"Variable segments ({len(variable_segments)}): {variable_segments}")

    combinations = []

    # Generate all possible combinations of models for variable segments only
    # This creates 7^M combinations (7 models for each of M variable segments)
    for model_combination in itertools.product(all_models, repeat=len(variable_segments)):
        # Create segment-specific config
        segment_config = {}

        # Add fixed segments
        for segment, fixed_config in fixed_segments.items():
            segment_config[segment] = fixed_config

        # Add variable segments
        for i, segment in enumerate(variable_segments):
            model_config = model_combination[i]
            segment_config[segment] = {k: v for k, v in model_config.items() if k != "description"}

        # Create a unique name for this combination
        # Include both fixed and variable segments in the name
        all_model_names = []
        for segment in all_segments:
            if segment in fixed_segments:
                model_name = fixed_segments[segment]["name"]
            else:
                var_index = variable_segments.index(segment)
                model_name = model_combination[var_index]["name"]
            all_model_names.append(f"{segment.lower()}_{model_name}")

        combination_name = "_".join(all_model_names)

        # Create description
        description_parts = []
        for segment in all_segments:
            if segment in fixed_segments:
                model_desc = fixed_segments[segment].get("description", f"Fixed model for {segment}")
            else:
                var_index = variable_segments.index(segment)
                model_desc = model_combination[var_index]["description"]
            description_parts.append(f"{model_desc} for {segment}")

        description = ", ".join(description_parts)

        combinations.append(
            {
                "name": combination_name,
                "model_mapping": "segment_specific",
                "config": segment_config,
                "description": description,
            }
        )

    # Sample combinations if requested
    if sample_size is not None and len(combinations) > sample_size:
        logger.info(f"Sampling {sample_size} combinations from {len(combinations)} total combinations")
        combinations = random.sample(combinations, sample_size)
    else:
        logger.info(f"Using all {len(combinations)} combinations")

    # Note: ML models (XGBoost, Random Forest) are available but may have data compatibility issues
    # They can be enabled by uncommenting the following lines if data preprocessing is improved
    # try:
    #     import xgboost
    #     ml_models = [
    #         {"name": "xgboost", "description": "XGBoost"},
    #         {"name": "random_forest", "description": "Random Forest"},
    #     ]
    #
    #     for model_config in ml_models:
    #         combinations.append({
    #             "name": f"all_{model_config['name']}",
    #             "model_mapping": "all_segments",
    #             "config": {k: v for k, v in model_config.items() if k != "description"},
    #             "description": f"{model_config['description']} for all segments",
    #         })
    # except ImportError:
    #     logger.warning("XGBoost not available, skipping ML models")

    logger.info(f"Generated {len(combinations)} model combinations")
    return combinations


def create_model_mapping(combination: dict[str, Any], segments: list[str]) -> dict[str, dict[str, Any]]:
    """Create model mapping based on combination type."""

    if combination["model_mapping"] == "all_segments":
        # All segments use the same model
        return {segment: combination["config"] for segment in segments}

    elif combination["model_mapping"] == "segment_specific":
        # Use specific mapping, with fallback to null for missing segments
        model_mapping = {}
        for segment in segments:
            if segment in combination["config"]:
                model_mapping[segment] = combination["config"][segment]
            else:
                # Fallback to null model for unmapped segments
                model_mapping[segment] = {"name": "null"}
        return model_mapping

    else:
        raise ValueError(f"Unknown model mapping type: {combination['model_mapping']}")


def split_predictions(
    predictions: pd.DataFrame, test_size: int = 1, val_size: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split predictions into train/validation/test sets chronologically."""

    n_total = len(predictions)
    test_size = min(test_size, n_total // 3)
    val_size = min(val_size, (n_total - test_size) // 2)

    # Sort by date to ensure chronological order
    predictions = predictions.sort_values("forecast_month").reset_index(drop=True)

    # Split chronologically: test (last), validation (second last), train (first)
    test_pred = predictions.tail(test_size)
    val_pred = predictions.iloc[-(test_size + val_size) : -test_size] if val_size > 0 else pd.DataFrame()
    train_pred = predictions.iloc[: -(test_size + val_size)] if val_size > 0 else predictions.iloc[:-test_size]

    return train_pred, val_pred, test_pred


def evaluate_predictions(predictions: pd.DataFrame, period_name: str) -> dict[str, Any]:
    """Evaluate predictions using the comprehensive evaluation function."""

    if len(predictions) == 0:
        logger.warning(f"No predictions to evaluate for {period_name}")
        return {"error": "No predictions available"}

    try:
        # Use the comprehensive evaluation function
        results = evaluate_forecast_simple(
            predictions, actual_col="actual", prediction_col="prediction", segment_col="eom_pattern_primary", date_col="date"
        )

        # Extract key metrics from the detailed results
        overall_avg = results[results["level"].str.contains("OVERALL_AVG")]

        if len(overall_avg) > 0:
            # Get the first row (they should be the same for net metrics)
            overall_metrics = overall_avg.iloc[0]
            overall_mae = overall_metrics.get("abs_net_error", float("inf"))
        else:
            overall_mae = float("inf")

        return {
            "overall_mae": overall_mae,
            "n_predictions": len(predictions),
            "n_segments": predictions["eom_pattern_primary"].nunique(),
            "detailed_results": results.to_dict("records") if len(results) <= 10 else "Too many detailed results to include",
        }

    except Exception as e:
        logger.warning(f"Evaluation failed for {period_name}: {e}")
        return {"error": str(e)}


def run_backtest_for_combination(
    combination: dict[str, Any], data: pd.DataFrame, segments: list[str], backtest_params: dict[str, Any]
) -> dict[str, Any]:
    """Run backtest for a single model combination."""

    start_time = time.time()

    try:
        # Create model mapping
        model_mapping = create_model_mapping(combination, segments)

        # Create segmented model
        segmented_model = NewSegmentedForecastModel(
            segment_col="eom_pattern_primary",
            target_col="target_eom_amount",
            date_col="forecast_month",
            dimensions=["dim_value"],
            model_mapping=model_mapping,
            fallback_model={"name": "null"},
        )

        # Create time series splitter
        splitter = TimeSeriesBacktest(
            forecast_horizon=backtest_params["forecast_horizon"],
            input_steps=backtest_params["input_steps"],
            expanding_window=backtest_params["expanding_window"],
            stride=backtest_params["stride"],
            date_column="forecast_month",
            min_backtest_iterations=backtest_params["min_backtest_iterations"],
        )

        # Check if we have enough data
        n_splits = splitter.number_of_splits(data)
        if n_splits < backtest_params["min_backtest_iterations"]:
            raise ValueError(f"Not enough data: {n_splits} splits available, need {backtest_params['min_backtest_iterations']}")

        logger.info(f"Running {n_splits} backtest splits for {combination['name']}")

        # Run backtesting
        all_predictions = []
        successful_folds = 0

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data)):
            try:
                train_df = data.iloc[train_idx].copy()
                test_df = data.iloc[test_idx].copy()

                logger.info(f"  Fold {fold_idx}: Train={len(train_df)}, Test={len(test_df)}")

                # Fit model
                segmented_model.fit(train_df)

                # Generate predictions
                predictions = segmented_model.predict(test_df, forecast_horizon=backtest_params["forecast_horizon"])

                # Check if predictions are valid
                if predictions is None or len(predictions) == 0:
                    logger.warning(f"  Fold {fold_idx}: No predictions generated")
                    continue

                # Add actual values and metadata
                predictions["actual"] = test_df["target_eom_amount"].values
                predictions["fold"] = fold_idx
                predictions["forecast_month"] = test_df["forecast_month"].values

                # Check for valid predictions (not all NaN)
                valid_preds = predictions.dropna(subset=["prediction"])
                if len(valid_preds) == 0:
                    logger.warning(f"  Fold {fold_idx}: All predictions are NaN")
                    continue

                all_predictions.append(predictions)
                successful_folds += 1
                logger.info(f"  Fold {fold_idx}: ✓ {len(valid_preds)} valid predictions")

            except Exception as e:
                logger.warning(f"  Fold {fold_idx} failed: {e}")
                continue

        if not all_predictions:
            raise ValueError("No successful predictions generated")

        # Combine predictions
        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        # Split into train/validation/test
        train_pred, val_pred, test_pred = split_predictions(
            combined_predictions, test_size=backtest_params.get("test_size", 1), val_size=backtest_params.get("val_size", 1)
        )

        # Evaluate each split
        train_metrics = evaluate_predictions(train_pred, "train") if len(train_pred) > 0 else {}
        val_metrics = evaluate_predictions(val_pred, "validation") if len(val_pred) > 0 else {}
        test_metrics = evaluate_predictions(test_pred, "test") if len(test_pred) > 0 else {}

        end_time = time.time()

        result = {
            "combination_id": f"{combination['name']}_{hash(str(combination['config']))}",
            "name": combination["name"],
            "description": combination["description"],
            "model_mapping": combination["model_mapping"],
            "config": combination["config"],
            "successful_folds": successful_folds,
            "total_folds": n_splits,
            "execution_time": end_time - start_time,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "status": "success",
        }

        test_mae = test_metrics.get("overall_mae", "N/A")
        if isinstance(test_mae, (int, float)):
            logger.info(f"✓ {combination['name']}: Test MAE={test_mae:.2f}")
        else:
            logger.info(f"✓ {combination['name']}: Test MAE={test_mae}")

        return result

    except Exception as e:
        end_time = time.time()
        logger.error(f"✗ {combination['name']}: {e}")

        return {
            "combination_id": f"{combination['name']}_{hash(str(combination['config']))}",
            "name": combination["name"],
            "description": combination["description"],
            "model_mapping": combination["model_mapping"],
            "config": combination["config"],
            "successful_folds": 0,
            "total_folds": 0,
            "execution_time": end_time - start_time,
            "train_metrics": {},
            "validation_metrics": {},
            "test_metrics": {},
            "status": "failed",
            "error": str(e),
        }


def load_existing_results(results_path: str) -> dict[str, Any]:
    """Load existing results from JSON file."""
    if Path(results_path).exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return {}


def save_results(results: dict[str, Any], results_path: str) -> None:
    """Save results to JSON file."""
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    """Main function to run comprehensive backtesting."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run comprehensive backtesting for segmented forecasting models")
    parser.add_argument("--data", default="dataset/feature_df.csv", help="Path to feature data CSV file")
    parser.add_argument("--output", default="outputs/backtest_results.json", help="Path to output results JSON file")
    parser.add_argument("--config", help="Path to custom configuration JSON file")
    parser.add_argument("--min-iterations", type=int, default=3, help="Minimum number of backtest iterations")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Forecast horizon")
    parser.add_argument("--input-steps", type=int, default=6, help="Number of input steps for training")
    parser.add_argument("--test-size", type=int, default=1, help="Size of test set for evaluation")
    parser.add_argument("--val-size", type=int, default=1, help="Size of validation set for evaluation")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of model combinations to sample (default: 100, use 0 for all combinations)",
    )
    parser.add_argument(
        "--fix-segments",
        type=str,
        help='JSON string defining fixed segments. Example: \'{"CONTINUOUS_STABLE": {"name": "arima", "order": [1,1,1]}, "NO_EOM": {"name": "null"}}\'',
    )

    args = parser.parse_args()

    # Configuration
    data_path = args.data
    results_path = args.output

    # Backtest parameters
    backtest_params = {
        "forecast_horizon": args.forecast_horizon,
        "input_steps": args.input_steps,
        "expanding_window": True,
        "stride": 1,
        "min_backtest_iterations": args.min_iterations,
        "test_size": args.test_size,
        "val_size": args.val_size,
    }

    # Load custom config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            backtest_params.update(custom_config.get("backtest_params", {}))

    logger.info("Starting comprehensive backtesting...")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {results_path}")
    logger.info(f"Backtest parameters: {backtest_params}")

    # Load data
    data = load_feature_data(data_path)
    segments = data["eom_pattern_primary"].unique().tolist()

    # Load existing results
    existing_results = load_existing_results(results_path)
    logger.info(f"Found {len(existing_results)} existing results")

    # Parse fixed segments
    fixed_segments = {
        "NO_EOM": {"name": "null"},
        "RARE_STALE": {"name": "null"},
        "AGGREGATED_OTHERS": {"name": "null"},
    }

    if args.fix_segments:
        import json

        try:
            user_fixed_segments = json.loads(args.fix_segments)
            # Merge user-defined fixed segments with default ones
            fixed_segments.update(user_fixed_segments)
            logger.info(f"Fixed segments: {fixed_segments}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for fixed segments: {e}")
            return
    else:
        logger.info(f"Using default fixed segments: {fixed_segments}")

    # Generate model combinations
    sample_size = None if args.sample_size == 0 else args.sample_size
    combinations = generate_model_combinations(data, sample_size, fixed_segments)

    # Run backtests
    for i, combination in enumerate(combinations, 1):
        combination_id = f"{combination['name']}_{hash(str(combination['config']))}"

        # Skip if already computed
        if combination_id in existing_results:
            logger.info(f"Skipping {combination['name']} (already computed)")
            continue

        logger.info(f"Running test {i}/{len(combinations)}: {combination['name']}")

        result = run_backtest_for_combination(combination, data, segments, backtest_params)
        existing_results[combination_id] = result

        # Save intermediate results
        save_results(existing_results, results_path)
        logger.info("Saved intermediate results")

    # Print summary
    logger.info("=" * 80)
    logger.info("BACKTESTING SUMMARY")
    logger.info("=" * 80)

    successful_results = {k: v for k, v in existing_results.items() if v.get("status") == "success"}
    logger.info(f"Total combinations: {len(existing_results)}")
    logger.info(f"Successful: {len(successful_results)}")

    if successful_results:
        logger.info("\n🏆 BEST PERFORMING MODELS (by Test MAE):")

        # Sort by test MAE
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["test_metrics"].get("overall_mae", float("inf")))

        for i, (_, result) in enumerate(sorted_results, 1):
            test_mae = result["test_metrics"].get("overall_mae", "N/A")
            val_mae = result["validation_metrics"].get("overall_mae", "N/A")
            if isinstance(test_mae, (int, float)) and isinstance(val_mae, (int, float)):
                logger.info(f"  {i}. {result['name']}: Test MAE={test_mae:.2f}, Val MAE={val_mae:.2f}")
            else:
                logger.info(f"  {i}. {result['name']}: Test MAE={test_mae}, Val MAE={val_mae}")

        logger.info(f"\n💾 Results saved to: {results_path}")
    else:
        logger.warning("No successful results found!")

    logger.info("Backtesting completed! 🎉")


if __name__ == "__main__":
    main()
