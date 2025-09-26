"""Base data preparation transformations."""

import time

import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig


def prepare_base_data(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Prepare base data with calendar features and EOM indicators."""
    start_time = time.time()
    initial_rows = len(df)

    logger.debug("Starting base data preparation")
    logger.debug("Input shape: {} rows × {} columns", initial_rows, len(df.columns))

    if config is None:
        config = SegmentationConfig()

    # Filter date range and select columns
    start_date_ts = pd.Timestamp(config.start_date)
    end_date_ts = pd.Timestamp(config.end_date)
    df = df[(df["date"] >= start_date_ts) & (df["date"] <= end_date_ts)].copy()
    logger.debug(
        "Filtered date range {}-{}: {} rows retained ({:.1f}%)",
        config.start_date,
        config.end_date,
        len(df),
        100 * len(df) / initial_rows,
    )

    # Add calendar features
    df["month"] = df["date"].dt.to_period("M")
    df["quarter"] = df["date"].dt.to_period("Q")
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["quarter_num"] = df["date"].dt.quarter
    df["day_of_month"] = df["date"].dt.day

    # Calculate days from EOM
    df = df.sort_values(["dim_value", "date"])
    df["days_from_eom"] = (
        df.groupby(["dim_value", "month"])
        .apply(
            lambda x: x["is_last_work_day_of_month"]
            .map(lambda eom: 0 if eom else None)
            .fillna(method="bfill")
            .fillna(-len(x) + x.index - x.index.min())
        )
        .values
    )

    elapsed_time = time.time() - start_time
    logger.debug("Base data preparation completed in {:.2f}s - {} rows × {} columns", elapsed_time, len(df), len(df.columns))

    return df.reset_index(drop=True)
