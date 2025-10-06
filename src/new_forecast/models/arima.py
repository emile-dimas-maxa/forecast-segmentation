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
    date_col: str = Field(description="Date column of the ARIMA model", default="forecast_month")
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
            raise ValueError(f"ARIMA model must be fitted before prediction")

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        def _predict_group(group):
            """Generate predictions for a group using ARIMA model."""
            key = group.name if isinstance(group.name, tuple) else (group.name,) if self.dimensions else tuple()
            model = self.models.get(key)

            if model is None or len(group) == 0:
                prediction = pd.DataFrame([float("nan")] * len(group), columns=["prediction"])
            else:
                try:
                    forecast_result = model.forecast(steps=len(group))
                    # Handle both scalar and array results
                    if hasattr(forecast_result, "to_frame"):
                        prediction = forecast_result.to_frame(name="prediction")
                    elif hasattr(forecast_result, "__len__") and len(forecast_result) > 0:
                        # Handle array-like results
                        if len(forecast_result) == len(group):
                            prediction = pd.DataFrame(forecast_result, columns=["prediction"])
                        else:
                            # Repeat the last value if lengths don't match
                            last_val = forecast_result[-1] if len(forecast_result) > 0 else float("nan")
                            prediction = pd.DataFrame([last_val] * len(group), columns=["prediction"])
                    else:
                        # Handle scalar results
                        prediction = pd.DataFrame([float(forecast_result)] * len(group), columns=["prediction"])
                except Exception as e:
                    # If forecasting fails, use NaN predictions
                    prediction = pd.DataFrame([float("nan")] * len(group), columns=["prediction"])

            return prediction

        results = data_grouped.apply(_predict_group)
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
