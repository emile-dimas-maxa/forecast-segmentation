from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.forecast.models.base import BaseSegmentModel


class MovingAverageModel(BaseSegmentModel):
    """Efficient moving average model with historical data storage."""

    def __init__(self, window: int = 3):
        self.window = window
        self.dimensions: list[str] | None = None
        self.target_col: str | None = None
        self.historical_data: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, target_col: str = "target", dimensions: list[str] = None) -> "MovingAverageModel":
        """Fit model by storing the last window values for each dimension group."""
        self.target_col = target_col
        self.dimensions = dimensions or []
        self.historical_data.clear()

        # Store historical data efficiently
        if not self.dimensions:
            # Single time series
            values = data[target_col].dropna()
            if len(values) >= self.window:
                self.historical_data[tuple()] = deque(values.tail(self.window), maxlen=self.window)
        else:
            # Multiple time series - store last window values per group
            for group_key, group in data.groupby(self.dimensions):
                values = group[target_col].dropna()
                if len(values) >= self.window:
                    key = group_key if isinstance(group_key, tuple) else (group_key,)
                    self.historical_data[key] = deque(values.tail(self.window), maxlen=self.window)

        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """Generate predictions using stored historical data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        def _create_prediction_rows(template_row: pd.Series, prediction: float, steps: int) -> pd.DataFrame:
            """Create prediction rows efficiently."""
            result = pd.DataFrame([template_row] * steps)
            result["prediction"] = prediction
            return result

        def _predict_group(group: pd.DataFrame, group_key: tuple = tuple()) -> pd.DataFrame:
            """Generate predictions for a group using stored historical data."""
            stored_values = self.historical_data.get(group_key)

            # Use np.nan if insufficient historical data
            if not stored_values or len(stored_values) < self.window:
                prediction = np.nan
            else:
                prediction = sum(stored_values) / len(stored_values)

            template_row = group.iloc[-1] if not group.empty else pd.Series()
            return _create_prediction_rows(template_row, prediction, steps)

        if not self.dimensions:
            # Single time series
            result = _predict_group(data)
        else:
            # Multiple time series
            results = []
            for group_key, group in data.groupby(self.dimensions):
                key = group_key if isinstance(group_key, tuple) else (group_key,)
                group_result = _predict_group(group, key)
                if not group_result.empty:
                    results.append(group_result)

            result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

        return result

    def update_historical_data(self, new_data: pd.DataFrame) -> None:
        """Update stored historical data with new observations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating historical data")

        if not self.dimensions:
            # Single time series
            values = new_data[self.target_col].dropna()
            for value in values:
                self.historical_data[tuple()].append(value)
        else:
            # Multiple time series
            for group_key, group in new_data.groupby(self.dimensions):
                key = group_key if isinstance(group_key, tuple) else (group_key,)
                values = group[self.target_col].dropna()
                for value in values:
                    self.historical_data[key].append(value)
