"""
Backtesting functionality for grid search.
"""

from typing import Any

import pandas as pd
from loguru import logger

from src.forecast.segmented import SegmentedForecastModel
from src.splitter import TimeSeriesBacktest


def run_full_backtest(
    model_mapping: dict[str, dict[str, Any]],
    data: pd.DataFrame,
    segment_col: str,
    target_col: str,
    dimensions: list[str],
    forecast_horizon: int,
    input_steps: int,
    expanding_window: bool,
    stride: int,
    min_backtest_iterations: int,
    date_col: str,
) -> pd.DataFrame:
    """Run full backtesting on the entire dataset and return all predictions"""

    # Create splitter
    splitter = TimeSeriesBacktest(
        forecast_horizon=forecast_horizon,
        input_steps=input_steps,
        expanding_window=expanding_window,
        stride=stride,
        date_column=date_col,
        min_backtest_iterations=min_backtest_iterations,
    )

    # Check if we have enough data for backtesting
    try:
        n_splits = splitter.number_of_splits(data)
        if n_splits < min_backtest_iterations:
            raise ValueError(f"Not enough data for backtesting: {n_splits} splits available, need {min_backtest_iterations}")
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
                segment_col=segment_col,
                target_col=target_col,
                dimensions=dimensions,
                model_mapping=model_mapping,
                fallback_model={"type": "moving_average", "params": {"window": 3}},
            )

            model.fit(train_df)

            # Generate predictions
            predictions = model.predict(context=None, model_input=test_df)

            # Add actual values and metadata
            # Use the target column that's already in predictions (from model output)
            # instead of overwriting with potentially misaligned values
            if target_col in predictions.columns:
                predictions["actual"] = predictions[target_col]
            else:
                # Fallback: ensure proper alignment by using index-based join
                predictions = predictions.reset_index(drop=True)
                test_df_reset = test_df.reset_index(drop=True)
                if len(predictions) == len(test_df_reset):
                    predictions["actual"] = test_df_reset[target_col].values
                else:
                    raise ValueError(
                        f"Prediction length ({len(predictions)}) doesn't match test data length ({len(test_df_reset)})"
                    )

            predictions["fold"] = fold_idx

            # Ensure we have the date column for sorting
            if date_col not in predictions.columns:
                predictions[date_col] = test_df[date_col].values

            all_predictions.append(predictions)

        except Exception as e:
            logger.warning(f"Error in fold {fold_idx}: {e}")
            raise e

    if not all_predictions:
        raise ValueError("No successful predictions generated during backtesting")

    # Combine all predictions and sort by date
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_predictions = combined_predictions.sort_values(date_col).reset_index(drop=True)

    return combined_predictions
