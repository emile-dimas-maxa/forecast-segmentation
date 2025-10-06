"""Evaluation functions for segmented forecasting models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
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
    Prepare data with credit/debit classifications based on dim_values patterns.

    Args:
        data: DataFrame with predictions and actuals
        dim_values_col: Column name for dim_values (contains ::IN/::OUT suffixes)
        actual_col: Column name for actual values
        prediction_col: Column name for predicted values

    Returns:
        DataFrame with additional classification columns
    """
    df = data.copy()

    # Add transaction type classifications based on dim_values suffix patterns
    df["is_credit"] = df[dim_values_col].str.endswith("::IN", na=False)
    df["is_debit"] = df[dim_values_col].str.endswith("::OUT", na=False)

    return df


def evaluate_forecast(
    data: pd.DataFrame,
    dim_values_col: str = "dim_value",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    segment_col: str = "eom_pattern_primary",
    date_col: str = "forecast_month",
) -> pd.DataFrame:
    """
    Evaluate forecast by calculating metrics for each segment and direction combination.
    """
    df = data.copy().assign(
        direction=lambda x: x[dim_values_col].str.endswith("::IN", na=False).map({True: "CREDIT", False: "DEBIT"})
    )

    base_cols = ["direction", segment_col, date_col, actual_col, prediction_col]

    # Step 1: Create signed_actual column
    df_signed = df.assign(
        signed_actual=lambda _d: _d.apply(
            lambda x: x[actual_col] if x["direction"] == "CREDIT" else -x[actual_col],
            axis=1,
        )
    )

    # Step 2: Create signed_prediction column
    df_signed = df_signed.assign(
        signed_prediction=lambda _d: _d.apply(
            lambda x: x[prediction_col] if x["direction"] == "CREDIT" else -x[prediction_col],
            axis=1,
        )
    )

    # Step 3: Group by segment and date, summing signed values
    df_grouped = df_signed.groupby([segment_col, date_col]).agg({"signed_actual": "sum", "signed_prediction": "sum"})

    # Step 4: Reset index
    df_reset = df_grouped.reset_index()

    # Step 5: Rename columns to original names
    df_renamed = df_reset.rename(columns={"signed_actual": actual_col, "signed_prediction": prediction_col})

    # Step 6: Add direction column with value "NET"
    df_with_direction = df_renamed.assign(direction="NET")

    # Step 7: Select only base columns
    df_segment_net = df_with_direction[base_cols]

    df_granular = df.assign(direction=np.nan)[base_cols]

    df_all_directions = pd.concat([df[base_cols], df_segment_net, df_granular])

    scores_summary = []

    # segement/ date
    segment_date_scores = (
        df_all_directions.groupby([segment_col, "direction", date_col])
        .apply(
            lambda x: pd.Series(
                {
                    **calculate_regression_metrics(x[actual_col], x[prediction_col]),
                    actual_col: x[actual_col].sum(),
                    prediction_col: x[prediction_col].sum(),
                }
            )
        )
        .reset_index()
    )

    all_cols = list(segment_date_scores.columns)

    segment_scores = (
        segment_date_scores.groupby([segment_col, "direction"])
        .agg(
            {
                actual_col: lambda x: x.mean(),
                prediction_col: lambda x: x.mean(),
                "mae": lambda x: x.mean(),
                "mse": lambda x: x.mean(),
                "rmse": lambda x: x.mean(),
                "mape": lambda x: x.mean(),
                "r2": lambda x: x.mean(),
                "mean_error": lambda x: x.mean(),
                "std_error": lambda x: x.mean(),
                "count": "sum",
            }
        )
        .reset_index()
        .assign(**{date_col: np.nan})[all_cols]
    )

    scores_summary.append(segment_date_scores)
    scores_summary.append(segment_scores)

    overall_date_scores = (
        df_all_directions.groupby(["direction", date_col])
        .apply(
            lambda x: pd.Series(
                {
                    **calculate_regression_metrics(x[actual_col], x[prediction_col]),
                    actual_col: x[actual_col].sum(),
                    prediction_col: x[prediction_col].sum(),
                }
            )
        )
        .reset_index()
        .assign(**{segment_col: "OVERALL"})
    )[all_cols]

    overall_scores = (
        overall_date_scores.groupby([segment_col, "direction"])
        .agg(
            {
                actual_col: lambda x: x.mean(),
                prediction_col: lambda x: x.mean(),
                "mae": lambda x: x.mean(),
                "mse": lambda x: x.mean(),
                "rmse": lambda x: x.mean(),
                "mape": lambda x: x.mean(),
                "r2": lambda x: x.mean(),
                "mean_error": lambda x: x.mean(),
                "std_error": lambda x: x.mean(),
                "count": "sum",
            }
        )
        .reset_index()
        .assign(**{date_col: np.nan})[all_cols]
    )
    scores_summary.append(overall_date_scores)
    scores_summary.append(overall_scores)

    return pd.concat(scores_summary)
