from typing import Literal

import pandas as pd
from pydantic import Field
from statsmodels.tsa.arima.model import ARIMA

from src.new_forecast.models.base import BaseForecastModel


class NetArimaModel(BaseForecastModel):
    """Net ARIMA model that calculates net (credit - debit) before fitting."""

    name: Literal["net_arima"] = Field(description="Name of the model", default="net_arima")
    order: tuple = Field(description="Order of the ARIMA model", default=(1, 1, 1))
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="date")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    models: dict[str, ARIMA] | None = Field(description="Fitted ARIMA models for each net series", default=None)

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
        net_data = pivot_data[["date", "net"]].copy()
        net_data["direction"] = "net"
        net_data = net_data.rename(columns={"net": self.target_col})

        return net_data

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "NetArimaModel":
        """Fit the model to the data with net transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)

        # Initialize models dictionary
        self.models = {}

        def _fit_net(net_data):
            """Fit ARIMA for the net series."""
            net_data = net_data.sort_values(self.date_col)
            arima_model = ARIMA(net_data[self.target_col], order=self.order).fit()
            self.models["net"] = arima_model
            return net_data

        transformed_data.groupby("direction").apply(_fit_net)

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with net transformations."""
        if not self.models:
            raise ValueError("Model must be fitted before prediction")

        # Transform the data
        transformed_data = self._transform_data(data)

        def _predict_net(net_data):
            """Generate predictions for the net series."""
            arima_model = self.models.get("net")

            if arima_model is None:
                # If no model, return NaN
                prediction = pd.DataFrame([float("nan")] * self.forecast_horizon, columns=["prediction"])
            else:
                # Generate forecast
                forecast = arima_model.forecast(steps=self.forecast_horizon)
                prediction = pd.DataFrame(forecast, columns=["prediction"])

            # Add direction column to prediction
            prediction["direction"] = "net"
            return prediction

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

        return pd.concat([data, results], ignore_index=True)
