"""
Evaluation functions for grid search results.
"""

from typing import Any

import pandas as pd
from loguru import logger

from src.forecast.evaluate import evaluate_forecast_simple


def evaluate_predictions(
    predictions: pd.DataFrame, period_name: str, dimensions: list[str], segment_col: str, date_col: str = "forecast_month"
) -> dict[str, Any]:
    """Evaluate predictions using the existing evaluation function"""

    if len(predictions) == 0:
        logger.warning(f"No predictions to evaluate for {period_name}")
        return None

    try:
        # Use the simplified evaluation function
        evaluation_df = evaluate_forecast_simple(
            data=predictions,
            dim_values_col=dimensions[0] if dimensions else "dim_value",
            actual_col="actual",
            prediction_col="prediction",
            segment_col=segment_col,
            date_col=date_col,
        )

        # Get overall AVERAGE NET metrics for primary evaluation (most important for grid search)
        overall_avg_net = evaluation_df[(evaluation_df["level"] == "OVERALL_AVG") & (evaluation_df["direction"] == "NET")]

        if len(overall_avg_net) == 0:
            raise ValueError(f"No overall average NET metrics found for {period_name}")

        # Calculate number of folds from the predictions data
        n_folds = len(predictions["fold"].unique()) if "fold" in predictions.columns else 1

        # Use average abs_net_error as the primary metric for grid search optimization
        primary_metrics = {
            "mae": overall_avg_net["abs_net_error"].iloc[0],  # Use average absolute net error as MAE equivalent
            "abs_net_error": overall_avg_net["abs_net_error"].iloc[0],
            "actual_total": overall_avg_net["actual_total"].iloc[0],
            "forecast_total": overall_avg_net["forecast_total"].iloc[0],
        }

        result = {
            "metrics": primary_metrics,
            "evaluation_df": evaluation_df,
            "n_predictions": len(predictions),
            "n_folds": n_folds,
            "period": period_name,
        }

        return result

    except Exception as e:
        logger.error(f"Error evaluating predictions for {period_name}: {e}")
        raise e
