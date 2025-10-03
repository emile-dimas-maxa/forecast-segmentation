"""
Main entry point for grid search functionality.
"""

from .config import GridSearchConfig
from .core import run_grid_search
from .utils import load_data


def main():
    """Main function to run grid search"""

    # Create configuration for data loading
    config = GridSearchConfig(
        feature_file_path="outputs/feature_df.csv",
        segment_col="eom_pattern_primary",
        target_col="target_eom_amount",
        date_col="forecast_month",
        dimensions=["dim_value"],
    )

    # Load data
    data = load_data(config)

    # Run grid search with parameters
    results = run_grid_search(
        data=data,
        segment_col="eom_pattern_primary",
        target_col="target_eom_amount",
        date_col="forecast_month",
        dimensions=["dim_value"],
        test_predictions=6,
        validation_predictions=6,
        input_steps=12,
        min_backtest_iterations=3,
        primary_metric="mae",
        model_configs=config.model_configs,
        model_segment_mapping=config.model_segment_mapping,
        output_dir="outputs/grid_search",
        save_detailed_results=True,
    )

    return results


if __name__ == "__main__":
    main()
