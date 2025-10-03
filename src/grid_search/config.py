"""
Configuration classes for grid search functionality.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GridSearchConfig:
    """Configuration for grid search"""

    # Data configuration
    feature_file_path: str = "outputs/feature_df.csv"
    segment_col: str = "eom_pattern_primary"
    target_col: str = "target_eom_amount"  # Based on feature pipeline output
    date_col: str = "forecast_month"
    dimensions: list[str] = field(default_factory=lambda: ["dim_value"])  # Adjust based on your data

    # Grid search configuration - based on number of predictions
    test_predictions: int = 6  # Number of predictions for test set (from n-x to n)
    validation_predictions: int = 6  # Number of predictions for validation set (from n-x-y to n-x)

    # Backtesting configuration
    forecast_horizon: int = 1
    input_steps: int = 12  # Use 12 months of history for training
    expanding_window: bool = True
    stride: int = 1
    min_backtest_iterations: int = 3

    # Model configurations to test
    model_configs: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {
            "arima": [
                {"type": "arima", "params": {"order": (1, 1, 1)}},
                {"type": "arima", "params": {"order": (2, 1, 1)}},
                {"type": "arima", "params": {"order": (1, 1, 2)}},
            ],
            "moving_average": [
                {"type": "moving_average", "params": {"window": 3}},
                {"type": "moving_average", "params": {"window": 6}},
                {"type": "moving_average", "params": {"window": 12}},
            ],
            "null": [
                {"type": "null", "params": {}},
            ],
        }
    )

    # Model-to-segment mapping (optional - if None, tests all models on all segments)
    model_segment_mapping: dict[str, list[str]] | None = None  # {"model_name": ["segment1", "segment2"]}

    # Evaluation metrics (primary metric used for selection)
    primary_metric: str = "mae"  # mae, rmse, mape, r2

    # Output configuration
    output_dir: str = "outputs/grid_search"
    save_detailed_results: bool = True
