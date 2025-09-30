"""Utility functions for model configuration and sample data generation."""

import numpy as np
import pandas as pd


def create_example_config():
    """Create example configuration for segmented forecasting."""
    return {
        "segment_col": "segment",
        "target_col": "sales",
        "model_mapping": {
            "segment_1": {"type": "arima", "params": {"order": (2, 1, 1)}},
            "segment_2": {"type": "moving_average", "params": {"window": 5}},
            "segment_3": {"type": "null", "params": {}},  # Always forecast null/NaN
        },
        "fallback_model": {"type": "moving_average", "params": {"window": 3}},
    }


def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)

    data = []
    for segment in ["segment_1", "segment_2", "segment_3"]:
        for region in ["North", "South"]:
            for product in ["A", "B"]:
                dates = pd.date_range("2023-01-01", periods=100, freq="D")
                if segment == "segment_1":
                    # Trending data for ARIMA
                    trend = np.linspace(100, 200, 100)
                    noise = np.random.normal(0, 10, 100)
                    base_sales = trend + noise
                else:
                    # More stationary data for moving average
                    base = 150 if segment == "segment_2" else 120
                    base_sales = base + np.random.normal(0, 15, 100)

                # Add regional and product effects
                region_effect = 20 if region == "North" else -10
                product_effect = 15 if product == "A" else -5
                sales = base_sales + region_effect + product_effect

                segment_data = pd.DataFrame(
                    {"date": dates, "segment": segment, "region": region, "product": product, "sales": sales}
                )
                data.append(segment_data)

    return pd.concat(data, ignore_index=True)
