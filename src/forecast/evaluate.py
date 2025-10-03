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
    Evaluate forecast by calculating metrics for each segment and direction combination.
    """
    df = data.copy()
    df["direction"] = df[dim_values_col].str.contains("::IN").map({True: "CREDIT", False: "DEBIT"})

    results = []

    # Calculate metrics for each segment-direction combination
    for segment in df[segment_col].unique():
        segment_data = df[df[segment_col] == segment]

        for direction in ["CREDIT", "DEBIT"]:
            direction_data = segment_data[segment_data["direction"] == direction]
            if len(direction_data) > 0:
                metrics = calculate_regression_metrics(direction_data[actual_col], direction_data[prediction_col])
                metrics.update({segment_col: segment, "direction": direction})
                results.append(metrics)

        # Calculate NET metrics (CREDIT - DEBIT) for each segment
        if len(segment_data) > 0:
            # Create net values by subtracting DEBIT from CREDIT for each data point
            segment_pivot = segment_data.pivot_table(
                index=segment_data.index, columns="direction", values=[actual_col, prediction_col], fill_value=0
            )

            if "CREDIT" in segment_pivot.columns.get_level_values(1) and "DEBIT" in segment_pivot.columns.get_level_values(1):
                net_actual = segment_pivot[(actual_col, "CREDIT")] - segment_pivot[(actual_col, "DEBIT")]
                net_pred = segment_pivot[(prediction_col, "CREDIT")] - segment_pivot[(prediction_col, "DEBIT")]

                # Only calculate if we have non-zero net values
                if len(net_actual.dropna()) > 0:
                    metrics = calculate_regression_metrics(net_actual, net_pred)
                    metrics.update({segment_col: segment, "direction": "NET"})
                    results.append(metrics)

    # Calculate overall metrics
    for direction in ["CREDIT", "DEBIT"]:
        direction_data = df[df["direction"] == direction]
        if len(direction_data) > 0:
            metrics = calculate_regression_metrics(direction_data[actual_col], direction_data[prediction_col])
            metrics.update({segment_col: "OVERALL", "direction": direction})
            results.append(metrics)

    # Calculate overall NET metrics
    if len(df) > 0:
        overall_pivot = df.pivot_table(index=df.index, columns="direction", values=[actual_col, prediction_col], fill_value=0)

        if "CREDIT" in overall_pivot.columns.get_level_values(1) and "DEBIT" in overall_pivot.columns.get_level_values(1):
            net_actual = overall_pivot[(actual_col, "CREDIT")] - overall_pivot[(actual_col, "DEBIT")]
            net_pred = overall_pivot[(prediction_col, "CREDIT")] - overall_pivot[(prediction_col, "DEBIT")]

            if len(net_actual.dropna()) > 0:
                metrics = calculate_regression_metrics(net_actual, net_pred)
                metrics.update({segment_col: "OVERALL", "direction": "NET"})
                results.append(metrics)

    return pd.DataFrame(results)


def evaluate_forecast_simple(
    data: pd.DataFrame,
    dim_values_col: str = "dim_value",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    segment_col: str = "segmentation",
    date_col: str = "forecast_month",
) -> pd.DataFrame:
    """
    Simplified evaluation that computes credit/debit/net totals by segment, date, and overall.

    Returns a clean table with levels (segment-date, segment-average, overall-date, overall-average),
    directions (credit/debit/net), actual totals, forecast totals, and absolute net error.
    """
    df = data.copy()

    # Classify direction based on dim_value
    df["direction"] = df[dim_values_col].str.contains("::IN").map({True: "CREDIT", False: "DEBIT"})

    results = []

    # 1. SEGMENT-DATE LEVEL: Process each segment-date combination
    for segment in df[segment_col].unique():
        segment_data = df[df[segment_col] == segment]

        for date in segment_data[date_col].unique():
            date_data = segment_data[segment_data[date_col] == date]

            # Calculate totals by direction for this segment-date
            date_totals = date_data.groupby("direction").agg({actual_col: "sum", prediction_col: "sum"}).reset_index()

            # Add segment and date info
            date_totals["level"] = f"{segment}_{date}"
            date_totals["level_type"] = "segment_date"
            date_totals["segment"] = segment
            date_totals["date"] = date

            # Rename columns for clarity
            date_totals = date_totals.rename(columns={actual_col: "actual_total", prediction_col: "forecast_total"})

            results.append(date_totals)

            # Calculate NET for this segment-date
            credit_row = date_totals[date_totals["direction"] == "CREDIT"]
            debit_row = date_totals[date_totals["direction"] == "DEBIT"]

            if len(credit_row) > 0 and len(debit_row) > 0:
                net_actual = credit_row["actual_total"].iloc[0] - debit_row["actual_total"].iloc[0]
                net_forecast = credit_row["forecast_total"].iloc[0] - debit_row["forecast_total"].iloc[0]
            elif len(credit_row) > 0:
                net_actual = credit_row["actual_total"].iloc[0]
                net_forecast = credit_row["forecast_total"].iloc[0]
            elif len(debit_row) > 0:
                net_actual = -debit_row["actual_total"].iloc[0]
                net_forecast = -debit_row["forecast_total"].iloc[0]
            else:
                net_actual = 0
                net_forecast = 0

            net_row = pd.DataFrame(
                [
                    {
                        "level": f"{segment}_{date}",
                        "level_type": "segment_date",
                        "segment": segment,
                        "date": date,
                        "direction": "NET",
                        "actual_total": net_actual,
                        "forecast_total": net_forecast,
                    }
                ]
            )
            results.append(net_row)

    # 2. OVERALL-DATE LEVEL: Process each date across all segments
    for date in df[date_col].unique():
        date_data = df[df[date_col] == date]

        # Calculate totals by direction for this date
        date_totals = date_data.groupby("direction").agg({actual_col: "sum", prediction_col: "sum"}).reset_index()

        # Add date info
        date_totals["level"] = f"OVERALL_{date}"
        date_totals["level_type"] = "overall_date"
        date_totals["segment"] = "OVERALL"
        date_totals["date"] = date

        # Rename columns for clarity
        date_totals = date_totals.rename(columns={actual_col: "actual_total", prediction_col: "forecast_total"})

        results.append(date_totals)

        # Calculate NET for this date
        credit_row = date_totals[date_totals["direction"] == "CREDIT"]
        debit_row = date_totals[date_totals["direction"] == "DEBIT"]

        if len(credit_row) > 0 and len(debit_row) > 0:
            net_actual = credit_row["actual_total"].iloc[0] - debit_row["actual_total"].iloc[0]
            net_forecast = credit_row["forecast_total"].iloc[0] - debit_row["forecast_total"].iloc[0]
        elif len(credit_row) > 0:
            net_actual = credit_row["actual_total"].iloc[0]
            net_forecast = credit_row["forecast_total"].iloc[0]
        elif len(debit_row) > 0:
            net_actual = -debit_row["actual_total"].iloc[0]
            net_forecast = -debit_row["forecast_total"].iloc[0]
        else:
            net_actual = 0
            net_forecast = 0

        net_row = pd.DataFrame(
            [
                {
                    "level": f"OVERALL_{date}",
                    "level_type": "overall_date",
                    "segment": "OVERALL",
                    "date": date,
                    "direction": "NET",
                    "actual_total": net_actual,
                    "forecast_total": net_forecast,
                }
            ]
        )
        results.append(net_row)

    # Combine all date-level results
    date_level_df = pd.concat(results, ignore_index=True)
    date_level_df["abs_net_error"] = abs(date_level_df["actual_total"] - date_level_df["forecast_total"])

    # 3. SEGMENT-AVERAGE LEVEL: Average errors across dates for each segment
    segment_avg_results = []
    for segment in df[segment_col].unique():
        segment_date_data = date_level_df[(date_level_df["segment"] == segment) & (date_level_df["level_type"] == "segment_date")]

        for direction in ["CREDIT", "DEBIT", "NET"]:
            direction_data = segment_date_data[segment_date_data["direction"] == direction]
            if len(direction_data) > 0:
                avg_abs_error = direction_data["abs_net_error"].mean()
                total_actual = direction_data["actual_total"].sum()
                total_forecast = direction_data["forecast_total"].sum()

                segment_avg_results.append(
                    {
                        "level": f"{segment}_AVG",
                        "level_type": "segment_average",
                        "segment": segment,
                        "date": "AVERAGE",
                        "direction": direction,
                        "actual_total": total_actual,
                        "forecast_total": total_forecast,
                        "abs_net_error": avg_abs_error,
                    }
                )

    # 4. OVERALL-AVERAGE LEVEL: Average errors across dates for overall
    overall_avg_results = []
    overall_date_data = date_level_df[date_level_df["level_type"] == "overall_date"]

    for direction in ["CREDIT", "DEBIT", "NET"]:
        direction_data = overall_date_data[overall_date_data["direction"] == direction]
        if len(direction_data) > 0:
            avg_abs_error = direction_data["abs_net_error"].mean()
            total_actual = direction_data["actual_total"].sum()
            total_forecast = direction_data["forecast_total"].sum()

            overall_avg_results.append(
                {
                    "level": "OVERALL_AVG",
                    "level_type": "overall_average",
                    "segment": "OVERALL",
                    "date": "AVERAGE",
                    "direction": direction,
                    "actual_total": total_actual,
                    "forecast_total": total_forecast,
                    "abs_net_error": avg_abs_error,
                }
            )

    # Combine all results
    avg_df = pd.concat([pd.DataFrame(segment_avg_results), pd.DataFrame(overall_avg_results)], ignore_index=True)

    # Combine date-level and average-level results
    final_df = pd.concat([date_level_df, avg_df], ignore_index=True)

    # Reorder columns for clarity
    final_df = final_df[["level", "level_type", "segment", "date", "direction", "actual_total", "forecast_total", "abs_net_error"]]

    return final_df


def print_evaluation_summary(evaluation_df: pd.DataFrame) -> None:
    """
    Print evaluation summary.
    """
    print(evaluation_df.to_string(index=False))
