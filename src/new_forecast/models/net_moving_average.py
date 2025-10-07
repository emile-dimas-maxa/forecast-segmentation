from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field

from src.new_forecast.models.base import BaseForecastModel


class NetMovingAverageModel(BaseForecastModel):
    """Net Moving Average model that calculates net (credit - debit) before fitting."""

    name: Literal["net_moving_average"] = Field(description="Name of the model", default="net_moving_average")
    window: int = Field(description="Window size for moving average", default=3)
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    historical_data: dict[str, list] | None = Field(description="Historical data for net series", default=None)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by creating direction column, calculating net, and aggregating by date."""
        # Create direction column by splitting dim_value on "::" and taking second part
        data = data.copy()
        data["direction"] = data["dim_value"].str.split("::").str[1]

        # Pivot to get credit and debit as separate columns
        pivot_data = data.pivot_table(
            index=self.date_col, columns="direction", values=self.target_col, aggfunc="sum", fill_value=0
        ).reset_index()

        # Calculate net (credit - debit)
        if "IN" in pivot_data.columns and "OUT" in pivot_data.columns:
            pivot_data["net"] = pivot_data["IN"] - pivot_data["OUT"]
        elif "IN" in pivot_data.columns:
            pivot_data["net"] = pivot_data["IN"]
        elif "OUT" in pivot_data.columns:
            pivot_data["net"] = -pivot_data["OUT"]
        else:
            pivot_data["net"] = 0

        # Melt back to long format
        net_data = pivot_data[[self.date_col, "net"]].copy()
        net_data["direction"] = "net"
        net_data = net_data.rename(columns={"net": self.target_col})

        return net_data

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "NetMovingAverageModel":
        """Fit the model to the data with net transformations."""
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
        """Predict the data with net transformations."""
        if not self.historical_data:
            logger.warning(self.historical_data)
            raise ValueError("Net Moving Average model must be fitted before prediction")

        # Transform the data
        transformed_data = self._transform_data(data)

        def _predict_net(net_data):
            """Generate predictions for the net series."""
            stored_values = self.historical_data.get("net", [])

            # Use 0.0 if insufficient historical data (matching moving_average.py logic)
            prediction = 0.0 if len(stored_values) < self.window else sum(stored_values) / len(stored_values)

            # Create prediction DataFrame for each row in the group
            result = pd.DataFrame([prediction] * len(net_data), columns=["prediction"])
            result["direction"] = "net"
            return result

        # Group by direction and predict
        results = transformed_data.groupby("direction").apply(_predict_net)

        if results.index.nlevels > 1:
            results = results.reset_index(level=0, drop=True)
        results = results.reset_index(drop=True)

        # Add date column for predictions
        max_date = data[self.date_col].max()
        results[self.date_col] = max_date + pd.DateOffset(months=self.forecast_horizon)

        # Add dim_value column for net
        results["dim_value"] = "net"

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
