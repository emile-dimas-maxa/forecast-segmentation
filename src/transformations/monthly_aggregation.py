"""Monthly aggregation transformations."""

import time

import numpy as np
import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig


def create_monthly_aggregates(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Create monthly aggregations with EOM and non-EOM breakdowns."""
    start_time = time.time()
    initial_rows = len(df)

    logger.debug("Starting monthly aggregation")
    logger.debug("Input shape: {} rows × {} columns", initial_rows, len(df.columns))

    if config is None:
        config = SegmentationConfig()

    # Group by dim_value and month
    agg_funcs = {
        "amount": [
            ("monthly_total", "sum"),
            ("monthly_avg_amount", lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0),
            ("monthly_std_amount", lambda x: x[x != 0].std() if len(x[x != 0]) > 1 else 0),
            ("max_monthly_transaction", "max"),
        ],
        "day_of_month": [("day_dispersion", "std")],
        "month_num": [("month_num", "first")],
        "year": [("year", "first")],
        "quarter_num": [("is_quarter_end", lambda x: 1 if x.iloc[0] in [3, 6, 9, 12] else 0)],
    }

    monthly = (
        df.groupby(["dim_value", "month"])
        .agg(
            {
                **{col: funcs for col, funcs in agg_funcs.items()},
                # Transaction counts
                "amount": [
                    ("monthly_transactions", lambda x: (x != 0).sum()),
                ],
            }
        )
        .reset_index()
    )

    # Flatten column names
    monthly.columns = [f"{col[1]}" if col[1] else col[0] for col in monthly.columns]

    # EOM-specific calculations
    eom_data = (
        df[df["is_last_work_day_of_month"]]
        .groupby(["dim_value", "month"])
        .agg({"amount": [("eom_amount", "sum"), ("eom_transaction_count", lambda x: (x != 0).sum())]})
        .reset_index()
    )
    eom_data.columns = [f"{col[1]}" if col[1] else col[0] for col in eom_data.columns]

    # Non-EOM calculations
    non_eom_data = (
        df[~df["is_last_work_day_of_month"]]
        .groupby(["dim_value", "month"])
        .agg(
            {
                "amount": [
                    ("non_eom_total", "sum"),
                    ("non_eom_transactions", lambda x: (x != 0).sum()),
                    ("non_eom_avg", lambda x: x[x != 0].mean() if len(x[x != 0]) > 0 else 0),
                ]
            }
        )
        .reset_index()
    )
    non_eom_data.columns = [f"{col[1]}" if col[1] else col[0] for col in non_eom_data.columns]

    # Pre-EOM signals
    pre_eom_data = (
        df[(df["days_from_eom"] >= -config.pre_eom_days) & (df["days_from_eom"] <= -1)]
        .groupby(["dim_value", "month"])
        .agg({"amount": [("pre_eom_5d_total", "sum"), ("pre_eom_5d_count", lambda x: (x != 0).sum())]})
        .reset_index()
    )
    pre_eom_data.columns = [f"{col[1]}" if col[1] else col[0] for col in pre_eom_data.columns]

    # Early/mid month signals
    early_month = df[df["day_of_month"] <= config.early_month_days].groupby(["dim_value", "month"])["amount"].sum().reset_index()
    early_month.columns = ["dim_value", "month", "early_month_total"]

    mid_month = (
        df[(df["day_of_month"] >= config.early_month_days) & (df["day_of_month"] <= config.mid_month_end_day)]
        .groupby(["dim_value", "month"])["amount"]
        .sum()
        .reset_index()
    )
    mid_month.columns = ["dim_value", "month", "mid_month_total"]

    # Quarterly/yearly amounts
    quarter_end = df[df["month_num"].isin([3, 6, 9, 12])].groupby(["dim_value", "month"])["amount"].sum().reset_index()
    quarter_end.columns = ["dim_value", "month", "quarter_end_amount"]

    year_end = df[df["month_num"] == 12].groupby(["dim_value", "month"])["amount"].sum().reset_index()
    year_end.columns = ["dim_value", "month", "year_end_amount"]

    # Merge all components
    result = monthly
    for data in [eom_data, non_eom_data, pre_eom_data, early_month, mid_month, quarter_end, year_end]:
        result = result.merge(data, on=["dim_value", "month"], how="left")

    # Fill missing values and add derived fields
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)

    result["is_year_end"] = (result["month_num"] == 12).astype(int)
    result["has_nonzero_eom"] = (result["eom_amount"] > 0).astype(int)
    result["day_dispersion"] = result["day_dispersion"].fillna(0)

    elapsed_time = time.time() - start_time
    logger.debug("Monthly aggregation completed in {:.2f}s - {} rows × {} columns", elapsed_time, len(result), len(result.columns))
    logger.debug("Unique entities after aggregation: {}", result["dim_value"].nunique())

    return result
