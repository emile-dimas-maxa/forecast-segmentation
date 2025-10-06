from typing import Literal
from pydantic import Field, model_validator
from src.new_forecast.models.base import BaseForecastModel
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA


class ArimaModel(BaseForecastModel):
    """ARIMA model implementation."""

    name: Literal["arima"] = Field(description="Name of the model", default="arima")
    order: tuple = Field(description="Order of the ARIMA model", default=(1, 1, 1))
    dimensions: list[str] = Field(description="Dimensions of the ARIMA model", default=[])
    target_col: str = Field(description="Target column of the ARIMA model", default="target")
    date_col: str = Field(description="Date column of the ARIMA model", default="date")
    forecast_horizon: int = Field(description="Forecast horizon of the ARIMA model", default=1)
    models: dict[tuple, ARIMA] | None = Field(description="Models of the ARIMA model", default=None)

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "ArimaModel":
        """Fit the model to the data"""
        # manage multiple time series using groupby and apply

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        self.models = data_grouped.apply(
            lambda x: ARIMA(x.sort_values(self.date_col)[self.target_col], order=self.order).fit()
        ).to_dict()
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data"""
        results = []

        if not self.models:
            raise ValueError("Model must be fitted before prediction")

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        results = data_grouped.apply(lambda x: self.models[x.name].forecast(steps=len(x)).to_frame(name="prediction"))
        if self.dimensions:
            results = results.reset_index()

        # Ensure we have the same number of predictions as input rows
        if len(results) != len(data):
            # If we have fewer predictions than input rows, repeat the last prediction
            if len(results) < len(data):
                last_pred = results.iloc[-1]["prediction"] if len(results) > 0 else 0.0
                additional_preds = pd.DataFrame([last_pred] * (len(data) - len(results)), columns=["prediction"])
                results = pd.concat([results, additional_preds], ignore_index=True)
            # If we have more predictions than input rows, take only the first N
            elif len(results) > len(data):
                results = results.iloc[: len(data)]

        return results
