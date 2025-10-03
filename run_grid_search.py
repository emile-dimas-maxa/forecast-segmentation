#!/usr/bin/env python3
"""
Entry point script for running grid search from the project root.
"""

from src.grid_search.config import GridSearchConfig
from src.grid_search.core import run_grid_search
from src.grid_search.main import main
import pandas as pd
from src.grid_search.utils import load_data
from pathlib import Path

output_dir = "outputs"
segmentation_file_name = "segmentation_df"
feature_file_name = "feature_df"
grid_search_file_name = "grid_search_results"
date_col = "forecast_month"
segment_col = "eom_pattern_primary"

if __name__ == "__main__":
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    feature_df = load_data(f"{output_dir}/{feature_file_name}.csv", date_col, segment_col)

    model_configs = {
        # "arima": [
        #     {"type": "arima", "params": {"order": (1, 1, 1)}},
        #     {"type": "arima", "params": {"order": (2, 1, 1)}},
        # ],
        "moving_average": [
            {"type": "moving_average", "params": {"window": 3}},
            {"type": "moving_average", "params": {"window": 6}},
        ],
        "null": [
            {"type": "null", "params": {}},
        ],
    }

    model_segment_mapping = {
        "null": ["RARE_STALE", "NO_EOM", "EMERGING", "AGGREGATED_OTHERS"],
    }

    config = GridSearchConfig(
        feature_file_path="outputs/feature_df.csv",
        segment_col="eom_pattern_primary",
        target_col="target_eom_amount",
        date_col="forecast_month",
        dimensions=["dim_value"],
        test_predictions=6,
        validation_predictions=6,
        input_steps=12,
        min_backtest_iterations=3,
        primary_metric="mae",
        model_configs=model_configs,
        model_segment_mapping=model_segment_mapping,
        output_dir=f"{output_dir}/{grid_search_file_name}",
        save_detailed_results=True,
    )

    results = run_grid_search(
        data=feature_df,
        segment_col="eom_pattern_primary",
        target_col="target_eom_amount",
        date_col="forecast_month",
        dimensions=["dim_value"],
        test_predictions=6,
        validation_predictions=6,
        input_steps=12,
        min_backtest_iterations=3,
        primary_metric="mae",
        model_configs=model_configs,
        model_segment_mapping=model_segment_mapping,
        output_dir=f"{output_dir}/{grid_search_file_name}",
        save_detailed_results=True,
    )
