"""Base data preparation transformations."""

import pandas as pd

from src.config.segmentation import SegmentationConfig


def prepare_base_data(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """Prepare base data with calendar features and EOM indicators."""
    if config is None:
        config = SegmentationConfig()

    # Filter date range and select columns
    df = df[(df["date"] >= config.start_date) & (df["date"] <= config.end_date)].copy()

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

    return df.reset_index(drop=True)
