from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
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
    historical_data: dict[str, list] | None = Field(description="Historical data for each direction", default=None)

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
        self.historical_data = {}

        data_grouped = transformed_data.groupby("direction")

        self.historical_data = data_grouped.apply(
            lambda x: x.sort_values(self.date_col)[self.target_col].dropna().tail(self.window).values
        ).to_dict()

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with transformations."""
        if not self.historical_data:
            logger.warning(self.historical_data)
            raise ValueError("Direction Moving Average model must be fitted before prediction")

        # Transform the data
        transformed_data = self._transform_data(data)

        def _predict_direction(direction_data):
            """Generate predictions for a specific direction."""
            direction = direction_data["direction"].iloc[0]
            stored_values = self.historical_data.get(direction, [])

            # Use 0.0 if insufficient historical data (matching moving_average.py logic)
            prediction = 0.0 if len(stored_values) < self.window else sum(stored_values) / len(stored_values)

            # Create prediction DataFrame for each row in the group
            result = pd.DataFrame([prediction] * len(direction_data), columns=["prediction"])
            result["direction"] = direction
            return result

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

        return pd.concat([data, results], ignore_index=True)
