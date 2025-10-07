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


class NetArimaModel(BaseForecastModel):
    """Net ARIMA model that calculates net (credit - debit) before fitting."""

    name: Literal["net_arima"] = Field(description="Name of the model", default="net_arima")
    order: tuple = Field(description="Order of the ARIMA model", default=(1, 1, 1))
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    min_samples: int = Field(description="Minimum number of samples required to fit ARIMA", default=10)
    models: dict[str, ARIMA | None] | None = Field(description="Fitted ARIMA models for each net series", default=None)

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
    ) -> "NetArimaModel":
        """Fit the model to the data with net transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)
        logger.info(f"Fitting Net ARIMA model with dimensions: {self.dimensions}")

        # Initialize models dictionary
        self.models = {}

        def _fit_net(net_data):
            """Fit ARIMA for the net series."""
            net_data = net_data.sort_values(self.date_col)
            series = net_data[self.target_col]
            arima_model = fit_arima_robust(series, self.order, self.min_samples)
            self.models["net"] = arima_model
            return net_data

        transformed_data.groupby("direction").apply(_fit_net)
        logger.info(f"Fitted Net ARIMA model with dimensions: {self.dimensions}")

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with net transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)

        def _predict_net(net_data):
            """Generate predictions for the net series."""
            arima_model = self.models.get("net")

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
