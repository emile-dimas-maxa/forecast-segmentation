import warnings
from typing import Literal

import pandas as pd
from pydantic import Field
from statsmodels.tsa.arima.model import ARIMA
from loguru import logger
from src.new_forecast.models.base import BaseForecastModel


def fit_arima_robust(series: pd.Series, order: tuple, min_samples: int = 10) -> ARIMA | None:
    """
    Fit ARIMA model with robust error handling for insufficient data.

    Args:
        series: Time series data to fit
        order: ARIMA order (p, d, q)
        min_samples: Minimum number of samples required to fit ARIMA

    Returns:
        Fitted ARIMA model or None if insufficient data
    """
    # Remove NaN values
    series_clean = series.dropna()

    # Check if we have enough data
    if len(series_clean) < min_samples:
        warnings.warn(f"Insufficient data for ARIMA fitting: {len(series_clean)} samples < {min_samples} required", stacklevel=2)
        return None

    # Check if we have enough data for the specific ARIMA order
    p, d, q = order
    min_required = max(p + d + q + 1, min_samples)

    if len(series_clean) < min_required:
        warnings.warn(
            f"Insufficient data for ARIMA({p},{d},{q}): {len(series_clean)} samples < {min_required} required", stacklevel=2
        )
        return None

    try:
        # Try to fit the ARIMA model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings
            model = ARIMA(series_clean, order=order).fit()
        return model
    except Exception as e:
        warnings.warn(f"ARIMA fitting failed: {str(e)}", stacklevel=2)
        return None


class ArimaModel(BaseForecastModel):
    """ARIMA model implementation."""

    name: Literal["arima"] = Field(description="Name of the model", default="arima")
    order: tuple = Field(description="Order of the ARIMA model", default=(1, 1, 1))
    dimensions: list[str] = Field(description="Dimensions of the ARIMA model", default=[])
    target_col: str = Field(description="Target column of the ARIMA model", default="target")
    date_col: str = Field(description="Date column of the ARIMA model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the ARIMA model", default=1)
    min_samples: int = Field(description="Minimum number of samples required to fit ARIMA", default=10)
    models: dict[tuple, ARIMA | None] | None = Field(description="Models of the ARIMA model", default=None)

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "ArimaModel":
        """Fit the model to the data"""
        # manage multiple time series using groupby and apply
        logger.info(f"Fitting ARIMA model with dimensions: {self.dimensions}")
        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        def fit_single_arima(group):
            """Fit ARIMA for a single group with robust error handling."""
            series = group.sort_values(self.date_col)[self.target_col]
            return fit_arima_robust(series, self.order, self.min_samples)

        self.models = data_grouped.apply(fit_single_arima).to_dict()
        logger.info(f"Fitted ARIMA model with dimensions: {self.dimensions}")
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data"""
        results = []

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

        def _predict_group(group):
            """Generate predictions for a group using ARIMA model."""
            key = group.name if isinstance(group.name, tuple) else (group.name,) if self.dimensions else tuple()
            model = self.models.get(key)

            if model is None or len(group) == 0:
                # Model failed to fit due to insufficient data or other issues
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
                except Exception:
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
