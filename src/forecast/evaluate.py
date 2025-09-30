"""Evaluation functions for segmented forecasting models."""

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


def calculate_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary of regression metrics
    """
    # Remove NaN values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "mean_error": np.nan,
            "std_error": np.nan,
            "count": 0,
        }

    # Calculate metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)

    # Handle MAPE calculation (avoid division by zero)
    try:
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    except (ValueError, ZeroDivisionError):
        mape = np.nan

    # R-squared
    try:
        r2 = r2_score(y_true_clean, y_pred_clean)
    except ValueError:
        r2 = np.nan

    # Error statistics
    errors = y_pred_clean - y_true_clean
    mean_error = errors.mean()
    std_error = errors.std()

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "mean_error": mean_error,
        "std_error": std_error,
        "count": len(y_true_clean),
    }


def prepare_segment_data(
    data: pd.DataFrame, dim_values_col: str = "dim_value", actual_col: str = "actual", prediction_col: str = "prediction"
) -> pd.DataFrame:
    """
    Prepare data with credit/debit classifications based on dim_value patterns.

    Args:
        data: DataFrame with predictions and actuals
        dim_values_col: Column name for dim_values (contains ::IN/::OUT suffixes)
        actual_col: Column name for actual values
        prediction_col: Column name for predicted values

    Returns:
        DataFrame with additional classification columns
    """
    df = data.copy()

    # Add transaction type classifications based on dim_value suffix patterns
    df["is_credit"] = df[dim_values_col].str.endswith("::IN", na=False)
    df["is_debit"] = df[dim_values_col].str.endswith("::OUT", na=False)

    # Calculate base segment by removing ::IN/::OUT suffixes
    df["base_segment"] = df[dim_values_col].str.replace("::IN$|::OUT$", "", regex=True)

    return df


def evaluate_forecast(
    data: pd.DataFrame,
    dim_values_col: str = "dim_value",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    segment_col: str = "segmentation",
) -> pd.DataFrame:
    """
    Prepare data for evaluation.
    """
    df = data.copy()

    cols = [segment_col, "direction", actual_col, prediction_col]

    df["direction"] = df[dim_values_col].str.contains("::IN").map({True: "CREDIT", False: "DEBIT"})

    df_segment_direction = df.groupby([segment_col, "direction"]).agg({actual_col: "sum", prediction_col: "sum"}).reset_index()
    df_segment_net = (
        df_segment_direction.groupby(segment_col)
        .agg(
            {
                "actual": lambda x: x[x == "CREDIT"].sum() - x[x == "DEBIT"].sum(),
                "prediction": lambda x: x[x == "CREDIT"].sum() - x[x == "DEBIT"].sum(),
            }
        )
        .reset_index()
        .assign(direction="NET")
    )

    df_overall_direction = (
        df.groupby("direction").agg({actual_col: "sum", prediction_col: "sum"}).reset_index().assign(**{segment_col: "OVERALL"})
    )
    df_overall_net = (
        df_overall_direction.groupby([segment_col, "direction"])
        .agg(
            {
                actual_col: lambda x: x[x == "CREDIT"].sum() - x[x == "DEBIT"].sum(),
                prediction_col: lambda x: x[x == "CREDIT"].sum() - x[x == "DEBIT"].sum(),
            }
        )
        .reset_index()
        .assign(direction="NET")
    )

    return pd.concat([df_segment_direction[cols], df_segment_net[cols], df_overall_direction[cols], df_overall_net[cols]]).apply(
        lambda x: calculate_regression_metrics(x[actual_col], x[prediction_col]), axis=1
    )


def print_evaluation_summary(evaluation_df: pd.DataFrame) -> None:
    """
    Print evaluation summary.
    """
    print(evaluation_df.to_string(index=False))
