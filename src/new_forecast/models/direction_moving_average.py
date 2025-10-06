from collections import defaultdict, deque
from typing import Literal

import pandas as pd
from pydantic import Field

from src.new_forecast.models.base import BaseForecastModel


class DirectionMovingAverageModel(BaseForecastModel):
    """Direction-aware Moving Average model that performs data transformations before fitting."""

    name: Literal["direction_moving_average"] = Field(description="Name of the model", default="direction_moving_average")
    window: int = Field(description="Window size for moving average", default=3)
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    historical_data: dict[str, deque] | None = Field(description="Historical data for each direction", default=None)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by creating direction column and aggregating by direction and date."""
        # Create direction column by splitting dim_value on "::" and taking second part
        data = data.copy()
        data["direction"] = data["dim_value"].str.split("::").str[1]

        # Group by direction and date, sum the target column
        transformed_data = data.groupby(["direction", self.date_col])[self.target_col].sum().reset_index()

        return transformed_data

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "DirectionMovingAverageModel":
        """Fit the model to the data with transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)

        # Initialize historical data storage
        self.historical_data = defaultdict(lambda: deque(maxlen=self.window))

        def _fit_direction(direction_data):
            """Fit moving average for a specific direction."""
            direction = direction_data["direction"].iloc[0]
            direction_data = direction_data.sort_values(self.date_col)

            values = direction_data[self.target_col].dropna()
            if len(values) >= self.window:
                self.historical_data[direction] = deque(values.tail(self.window), maxlen=self.window)

            return direction_data

        transformed_data.groupby("direction").apply(_fit_direction)

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with transformations."""
        if not self.historical_data:
            raise ValueError(f"Direction Moving Average model must be fitted before prediction")

        # Transform the data
        transformed_data = self._transform_data(data)

        def _predict_direction(direction_data):
            """Generate predictions for a specific direction."""
            direction = direction_data["direction"].iloc[0]
            stored_values = self.historical_data.get(direction)

            # Use NaN if insufficient historical data
            if not stored_values or len(stored_values) < self.window:
                prediction = pd.DataFrame([float("nan")] * self.forecast_horizon, columns=["prediction"])
            else:
                # Calculate moving average
                avg_value = sum(stored_values) / len(stored_values)
                prediction = pd.DataFrame([avg_value] * self.forecast_horizon, columns=["prediction"])

            # Add direction column to prediction
            prediction["direction"] = direction
            return prediction

        # Group by direction and predict
        results = transformed_data.groupby("direction").apply(_predict_direction)

        if results.index.nlevels > 1:
            results = results.reset_index(level=0, drop=True)
        results = results.reset_index(drop=True)

        # Add date column for predictions
        max_date = data[self.date_col].max()
        results[self.date_col] = max_date + pd.DateOffset(months=self.forecast_horizon)

        # Add dim_value column by reconstructing it from direction
        results["dim_value"] = "::" + results["direction"]

        return pd.concat([data, results], ignore_index=True)
