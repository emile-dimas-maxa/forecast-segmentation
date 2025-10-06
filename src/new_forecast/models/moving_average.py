from collections import defaultdict, deque
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from src.new_forecast.models.base import BaseForecastModel


class MovingAverageModel(BaseForecastModel):
    """Moving Average model implementation."""

    name: Literal["moving_average"] = Field(description="Name of the model", default="moving_average")
    window: int = Field(description="Window size for moving average", default=3)
    dimensions: list[str] = Field(description="Dimensions of the moving average model", default=[])
    target_col: str = Field(description="Target column of the moving average model", default="target")
    date_col: str = Field(description="Date column of the moving average model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the moving average model", default=1)
    historical_data: dict[tuple, deque] | None = Field(description="Historical data for moving average", default=None)

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "MovingAverageModel":
        """Fit the model to the data"""
        # Initialize historical data storage
        self.historical_data = defaultdict(lambda: deque(maxlen=self.window))

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        def _store_historical_data(group):
            values = group[self.target_col].dropna()
            if len(values) >= self.window:
                key = group.name if isinstance(group.name, tuple) else (group.name,) if self.dimensions else tuple()
                self.historical_data[key] = deque(values.tail(self.window), maxlen=self.window)
            return group

        data_grouped.apply(_store_historical_data)

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data"""
        if not self.historical_data:
            raise ValueError(f"Moving Average model must be fitted before prediction")

        def _predict_group(group):
            """Generate predictions for a group using stored historical data."""
            key = group.name if isinstance(group.name, tuple) else (group.name,) if self.dimensions else tuple()
            stored_values = self.historical_data.get(key)

            # Use np.nan if insufficient historical data
            if not stored_values or len(stored_values) < self.window:
                prediction = np.nan
            else:
                prediction = sum(stored_values) / len(stored_values)

            # Create prediction DataFrame for each row in the group
            result = pd.DataFrame([prediction] * len(group), columns=["prediction"])
            return result

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        results = data_grouped.apply(_predict_group)

        if self.dimensions:
            results = results.reset_index()

        # Ensure we have the same number of predictions as input rows
        if len(results) != len(data):
            # If we have fewer predictions than input rows, repeat the last prediction
            if len(results) < len(data):
                last_pred = results.iloc[-1]["prediction"] if len(results) > 0 else np.nan
                additional_preds = pd.DataFrame([last_pred] * (len(data) - len(results)), columns=["prediction"])
                results = pd.concat([results, additional_preds], ignore_index=True)
            # If we have more predictions than input rows, take only the first N
            elif len(results) > len(data):
                results = results.iloc[: len(data)]

        return results
