"""Moving average model implementation."""

import numpy as np
import pandas as pd

from .base import BaseSegmentModel


class MovingAverageModel(BaseSegmentModel):
    """Moving average model implementation."""

    def __init__(self, window: int = 3):
        self.window = window
        self.last_values = {}  # Store last values for each dimension combination
        self.dimensions = None
        self.target_col = None

    def fit(self, data: pd.DataFrame, target_col: str = "target", dimensions: list[str] = None) -> "MovingAverageModel":
        """Fit moving average model (store last values)."""
        self.target_col = target_col
        self.dimensions = dimensions or []
        self.last_values = {}

        if not self.dimensions:
            # Treat all data as one time series
            y = data[target_col].dropna()
            self.last_values["__all__"] = y.tail(self.window).values
        else:
            # Store last values for each dimension combination
            for dim_values, group_data in data.groupby(self.dimensions):
                # Convert single value to tuple for consistency
                if not isinstance(dim_values, tuple):
                    dim_values = (dim_values,)

                y = group_data[target_col].dropna()
                if len(y) > 0:  # Only store if we have data
                    self.last_values[dim_values] = y.tail(self.window).values

        return self

    def predict(self, data: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """Generate moving average predictions."""
        if not self.last_values:
            raise ValueError("Model must be fitted before prediction")

        results = []

        if not self.dimensions:
            # Single time series prediction
            last_vals = self.last_values.get("__all__")
            if last_vals is None:
                raise ValueError("Model was not successfully fitted")

            predictions = []
            current_values = last_vals.copy()

            for _ in range(steps):
                pred = np.mean(current_values)
                predictions.append(pred)
                # Update rolling window
                current_values = np.append(current_values[1:], pred)

            # Create result DataFrame
            result = data.iloc[-steps:].copy() if len(data) >= steps else data.copy()
            if len(result) < steps:
                # Extend with NaN rows if needed
                additional_rows = steps - len(result)
                last_row = result.iloc[-1:] if len(result) > 0 else pd.DataFrame([{}] * 1, columns=data.columns)
                for _ in range(additional_rows):
                    result = pd.concat([result, last_row], ignore_index=True)

            result = result.iloc[-steps:].copy()
            result["prediction"] = predictions
            results.append(result)
        else:
            # Multiple time series predictions
            for dim_values, group_data in data.groupby(self.dimensions):
                # Convert single value to tuple for consistency
                if not isinstance(dim_values, tuple):
                    dim_values = (dim_values,)

                last_vals = self.last_values.get(dim_values)
                if last_vals is None:
                    print(f"Warning: No fitted values for dimensions {dim_values}, skipping prediction")
                    continue

                predictions = []
                current_values = last_vals.copy()

                for _ in range(steps):
                    pred = np.mean(current_values)
                    predictions.append(pred)
                    # Update rolling window
                    current_values = np.append(current_values[1:], pred)

                # Create result DataFrame
                result = group_data.iloc[-steps:].copy() if len(group_data) >= steps else group_data.copy()
                if len(result) < steps:
                    # Extend with NaN rows if needed
                    additional_rows = steps - len(result)
                    last_row = result.iloc[-1:] if len(result) > 0 else pd.DataFrame([{}] * 1, columns=group_data.columns)
                    for _ in range(additional_rows):
                        result = pd.concat([result, last_row], ignore_index=True)

                result = result.iloc[-steps:].copy()
                result["prediction"] = predictions

                # Ensure dimension columns are preserved
                for i, dim_col in enumerate(self.dimensions):
                    result[dim_col] = dim_values[i] if len(dim_values) > i else dim_values[0]

                results.append(result)

        if not results:
            raise ValueError("No successful predictions were generated")

        return pd.concat(results, ignore_index=True)
