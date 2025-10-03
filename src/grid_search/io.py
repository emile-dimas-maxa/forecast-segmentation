"""
Input/Output functions for grid search results.
"""

import json
from pathlib import Path

import pandas as pd
from loguru import logger


def save_results(
    results: dict,
    output_dir: str,
    save_detailed_results: bool,
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
    model_configs: dict,
    model_segment_mapping: dict,
    primary_metric: str,
):
    """Save grid search results"""
    logger.info("Saving results...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        # Convert parameters to dict for JSON serialization
        config_dict = {
            "segment_col": segment_col,
            "target_col": target_col,
            "date_col": date_col,
            "dimensions": dimensions,
            "test_predictions": test_predictions,
            "validation_predictions": validation_predictions,
            "forecast_horizon": forecast_horizon,
            "input_steps": input_steps,
            "expanding_window": expanding_window,
            "stride": stride,
            "min_backtest_iterations": min_backtest_iterations,
            "model_configs": model_configs,
            "model_segment_mapping": model_segment_mapping,
            "primary_metric": primary_metric,
        }
        json.dump(config_dict, f, indent=2)

    # Save best model configuration
    best_config_path = Path(output_dir) / "best_model_config.json"
    with open(best_config_path, "w") as f:
        json.dump(results["best_config"], f, indent=2)

    # Save detailed results if requested
    if save_detailed_results:
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
        selection_df.to_csv(Path(output_dir) / "selection_results.csv", index=False)

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
            test_df.to_csv(Path(output_dir) / "test_results.csv", index=False)

        # Save granular evaluation results
        logger.info("Saving granular evaluation results...")

        # Save selection phase granular results for all combinations
        selection_granular_results = []
        for i, result in enumerate(results["selection_results"]):
            if "evaluation_df" in result:
                eval_df = result["evaluation_df"].copy()
                eval_df["combination_id"] = i
                eval_df["period"] = "selection"
                selection_granular_results.append(eval_df)

        if selection_granular_results:
            selection_granular_df = pd.concat(selection_granular_results, ignore_index=True)
            selection_granular_df.to_csv(Path(output_dir) / "selection_granular_results.csv", index=False)
            logger.info(f"Saved {len(selection_granular_df)} selection granular result rows")

        # Save test phase granular results (best model only)
        if results["test_results"] and "evaluation_df" in results["test_results"][0]:
            test_granular_df = results["test_results"][0]["evaluation_df"].copy()
            test_granular_df["period"] = "test"
            test_granular_df["is_best_model"] = True
            test_granular_df.to_csv(Path(output_dir) / "test_granular_results.csv", index=False)
            logger.info(f"Saved {len(test_granular_df)} test granular result rows")

        # Save best model granular results with model configuration
        if results["selection_results"] and results["best_config"]:
            # Find the best model result
            best_model_result = None
            for result in results["selection_results"]:
                if result.get("model_mapping") == results["best_config"]:
                    best_model_result = result
                    break

            if best_model_result and "evaluation_df" in best_model_result:
                best_granular_df = best_model_result["evaluation_df"].copy()
                best_granular_df["period"] = "selection_best"
                best_granular_df["is_best_model"] = True

                # Add model configuration details
                for segment, model_config in results["best_config"].items():
                    mask = best_granular_df["segment"] == segment
                    best_granular_df.loc[mask, "model_type"] = model_config["type"]
                    best_granular_df.loc[mask, "model_params"] = str(model_config.get("params", {}))

                best_granular_df.to_csv(Path(output_dir) / "best_model_granular_results.csv", index=False)
                logger.info(f"Saved {len(best_granular_df)} best model granular result rows")

    # Save summary
    summary = {
        "selection_metrics": results["selection_metrics"],
        "test_metrics": results["test_metrics"],
        "best_model_config": results["best_config"],
        "n_combinations_tested": len(results["selection_results"]),
    }

    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def print_summary(results: dict, primary_metric: str):
    """Print summary of results"""
    logger.info("=" * 80)
    logger.info("Grid Search Summary")
    logger.info("=" * 80)

    logger.info(f"Combinations tested: {len(results['selection_results'])}")
    logger.info(f"Primary metric: {primary_metric}")

    if results["selection_metrics"]:
        logger.info("\nSelection Phase Results:")
        for metric, value in results["selection_metrics"].items():
            if isinstance(value, (int, float, complex)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    if results["test_metrics"]:
        logger.info("\nTest Phase Results:")
        for metric, value in results["test_metrics"].items():
            if isinstance(value, (int, float, complex)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    if results["best_config"]:
        logger.info("\nBest Model Configuration:")
        for segment, model_config in results["best_config"].items():
            logger.info(f"  {segment}: {model_config['type']} {model_config.get('params', {})}")

    logger.info("=" * 80)
