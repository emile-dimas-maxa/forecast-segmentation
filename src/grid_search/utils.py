"""
Utility functions for grid search functionality.
"""

import itertools
from typing import Any

import pandas as pd
from loguru import logger

from .config import GridSearchConfig


def create_empty_results() -> dict:
    """Create empty results dictionary"""
    return {
        "selection_results": [],
        "test_results": [],
        "best_config": None,
        "best_model": None,
        "selection_metrics": None,
        "test_metrics": None,
    }


def load_data(path: str, date_col: str, segment_col: str) -> pd.DataFrame:
    """Load feature data"""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    df[date_col] = pd.to_datetime(df[date_col])

    logger.info(f"Loaded {len(df)} rows with {df[segment_col].nunique()} segments")
    logger.info(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    return df


def generate_model_combinations(
    segments: list[str],
    model_configs: dict[str, list[dict[str, Any]]],
    model_segment_mapping: dict[str, list[str]] = None,
    unmapped_strategy: str = "all_models",  # "all_models" or "fallback"
) -> list[dict[str, dict[str, Any]]]:
    """
    Generate model combinations respecting model-to-segment mapping

    Args:
        segments: List of segment names
        model_configs: Dictionary of model configurations
        model_segment_mapping: Optional mapping of models to specific segments
        unmapped_strategy: Strategy for unmapped segments:
            - "all_models": Use all available models (enables random combinations)
            - "fallback": Use a single fallback model
    """
    logger.info("Generating model combinations...")

    if model_segment_mapping is None:
        return _generate_all_combinations(segments, model_configs)
    else:
        return _generate_mapped_combinations(segments, model_configs, model_segment_mapping, unmapped_strategy)


def _generate_all_combinations(
    segments: list[str], model_configs: dict[str, list[dict[str, Any]]]
) -> list[dict[str, dict[str, Any]]]:
    """Generate combinations testing all models on all segments"""
    logger.info("No model-segment mapping specified, testing all models on all segments")

    # Flatten all model configs into a single list
    all_configs = [config for configs in model_configs.values() for config in configs]

    # Generate all possible combinations
    combinations = [
        {segment: model_config for segment, model_config in zip(segments, combo, strict=True)}
        for combo in itertools.product(all_configs, repeat=len(segments))
    ]

    logger.info(f"Generated {len(combinations)} model combinations for {len(segments)} segments")
    return combinations


def _generate_mapped_combinations(
    segments: list[str],
    model_configs: dict[str, list[dict[str, Any]]],
    model_segment_mapping: dict[str, list[str]],
    unmapped_strategy: str = "all_models",
) -> list[dict[str, dict[str, Any]]]:
    """Generate combinations respecting model-to-segment mapping"""
    logger.info("Using model-segment mapping to generate combinations")

    # Validate mapping
    for model_name in model_segment_mapping:
        if model_name not in model_configs:
            raise ValueError(f"Model '{model_name}' in mapping not found in model_configs")

    # Split segments into mapped and unmapped
    mapped_segments = {seg for segs in model_segment_mapping.values() for seg in segs}
    unmapped_segments = [seg for seg in segments if seg not in mapped_segments]
    mapped_segment_list = [seg for seg in segments if seg in mapped_segments]

    # Generate combinations for unmapped segments
    if unmapped_segments:
        if unmapped_strategy == "all_models":
            unmapped_combinations = _generate_all_combinations(unmapped_segments, model_configs)
        else:
            # Use fallback for all unmapped segments
            fallback_config = {"type": "moving_average", "params": {"window": 3}}
            unmapped_combinations = [{seg: fallback_config for seg in unmapped_segments}]
        logger.info(f"Unmapped segments using {unmapped_strategy}: {sorted(unmapped_segments)}")
    else:
        unmapped_combinations = [{}]  # Empty dict for merging

    # Generate combinations for mapped segments
    if mapped_segment_list:
        mapped_options = {
            seg: [
                config for model_name, segs in model_segment_mapping.items() if seg in segs for config in model_configs[model_name]
            ]
            for seg in mapped_segment_list
        }
        mapped_combinations = [
            {seg: model for seg, model in zip(mapped_segment_list, combo, strict=True)}
            for combo in itertools.product(*[mapped_options[seg] for seg in mapped_segment_list])
        ]
    else:
        mapped_combinations = [{}]  # Empty dict for merging

    # Combine mapped and unmapped combinations
    combinations = [
        {**mapped_combo, **unmapped_combo} for mapped_combo in mapped_combinations for unmapped_combo in unmapped_combinations
    ]

    logger.info(f"Generated {len(combinations)} combinations")
    return combinations


def split_predictions_for_grid_search(
    all_predictions: pd.DataFrame,
    date_col: str,
    test_predictions: int,
    validation_predictions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split predictions into validation and test sets based on date ranges"""

    # Sort predictions by date to ensure correct ordering
    all_predictions = all_predictions.sort_values(date_col).reset_index(drop=True)

    # Get unique dates and ensure we have enough
    unique_dates = sorted(all_predictions[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < test_predictions + validation_predictions:
        raise ValueError(f"Not enough unique dates: need at least {test_predictions + validation_predictions} dates, got {n_dates}")

    # Define date ranges for test and validation sets
    # Test set: last test_predictions dates
    test_start_date = unique_dates[-test_predictions]
    test_preds = all_predictions[all_predictions[date_col] >= test_start_date].copy()

    # Validation set: validation_predictions dates before test set
    validation_start_date = unique_dates[-(test_predictions + validation_predictions)]
    validation_end_date = unique_dates[-test_predictions - 1]  # Last date before test set
    validation_preds = all_predictions[
        (all_predictions[date_col] >= validation_start_date) & (all_predictions[date_col] <= validation_end_date)
    ].copy()

    logger.info(f"Total predictions: {len(all_predictions)} across {n_dates} unique dates")
    logger.info(f"Validation set: {len(validation_preds)} predictions from {validation_start_date} to {validation_end_date}")
    logger.info(f"Test set: {len(test_preds)} predictions from {test_start_date} onwards")

    return validation_preds, test_preds
