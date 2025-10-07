import warnings
from typing import Literal

import pandas as pd
from loguru import logger
from pydantic import Field
from statsmodels.tsa.arima.model import ARIMA

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


class DirectionArimaModel(BaseForecastModel):
    """Direction-aware ARIMA model that performs data transformations before fitting."""

    name: Literal["direction_arima"] = Field(description="Name of the model", default="direction_arima")
    order: tuple = Field(description="Order of the ARIMA model", default=(1, 1, 1))
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    min_samples: int = Field(description="Minimum number of samples required to fit ARIMA", default=10)
    models: dict[str, ARIMA | None] | None = Field(description="Fitted ARIMA models for each direction", default=None)

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
    ) -> "DirectionArimaModel":
        """Fit the model to the data with transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)
        logger.info(f"Fitting Direction ARIMA model with dimensions: {self.dimensions}")

        # Initialize models dictionary
        self.models = {}

        # Fit ARIMA model for each direction using groupby
        def _fit_direction(direction_data):
            direction = direction_data["direction"].iloc[0]
            direction_data = direction_data.sort_values(self.date_col)
            series = direction_data[self.target_col]
            arima_model = fit_arima_robust(series, self.order, self.min_samples)
            self.models[direction] = arima_model
            return direction_data

        transformed_data.groupby("direction").apply(_fit_direction)
        logger.info(f"Fitted Direction ARIMA model with dimensions: {self.dimensions}")

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)

        def _predict_direction(direction_data):
            """Generate predictions for a specific direction."""
            direction = direction_data["direction"].iloc[0]
            arima_model = self.models.get(direction)

            if arima_model is None:
                # Model failed to fit due to insufficient data or other issues
                prediction = pd.DataFrame([float("nan")] * self.forecast_horizon, columns=["prediction"])
            else:
                try:
                    # Generate forecast
                    forecast = arima_model.forecast(steps=self.forecast_horizon)
                    # Handle both scalar and array results
                    if hasattr(forecast, "__len__") and len(forecast) > 0:
                        prediction = pd.DataFrame(forecast, columns=["prediction"])
                    else:
                        # Handle scalar results
                        prediction = pd.DataFrame([float(forecast)] * self.forecast_horizon, columns=["prediction"])
                except Exception:
                    # If forecasting fails, use NaN predictions
                    prediction = pd.DataFrame([float("nan")] * self.forecast_horizon, columns=["prediction"])

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
