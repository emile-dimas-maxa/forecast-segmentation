import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.forecast.models.base import BaseSegmentModel


class ARIMAModel(BaseSegmentModel):
    """ARIMA model implementation."""

    def __init__(self, order: tuple = (1, 1, 1), **kwargs):
        if ARIMA is None:
            raise ImportError("statsmodels is required for ARIMA model. Install with: pip install statsmodels")
        self.order = order
        self.kwargs = kwargs
        self.fitted_models = {}
        self.dimensions = None
        self.target_col = None

    def fit(self, data: pd.DataFrame, target_col: str = "target", dimensions: list[str] = None) -> "ARIMAModel":
        self.target_col = target_col
        self.dimensions = dimensions or []
        self.fitted_models = {}

        if not self.dimensions:
            # Treat all data as one time series
            y = data[target_col].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y, order=self.order, **self.kwargs)
                self.fitted_models["__all__"] = model.fit()
        else:
            # Fit separate models for each dimension combination
            for dim_values, group_data in data.groupby(self.dimensions):
                # Convert single value to tuple for consistency
                if not isinstance(dim_values, tuple):
                    dim_values = (dim_values,)

                y = group_data[target_col].dropna()
                if len(y) > 0:  # Only fit if we have data
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            model = ARIMA(y, order=self.order, **self.kwargs)
                            self.fitted_models[dim_values] = model.fit()
                        except Exception as e:
                            print(f"Warning: Failed to fit ARIMA model for dimensions {dim_values}: {e}")
                            # Store None to indicate failed fit
                            self.fitted_models[dim_values] = None

        return self

    def predict(self, data: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """Generate ARIMA predictions."""
        if not self.fitted_models:
            raise ValueError("Model must be fitted before prediction")

        results = []

        if not self.dimensions:
            # Single time series prediction
            fitted_model = self.fitted_models.get("__all__")
            if fitted_model is None:
                raise ValueError("Model was not successfully fitted")

            forecast = fitted_model.forecast(steps=steps)

            # Create result DataFrame with same structure as input
            result = data.iloc[-steps:].copy() if len(data) >= steps else data.copy()
            if len(result) < steps:
                # Extend with NaN rows if needed
                additional_rows = steps - len(result)
                last_row = result.iloc[-1:] if len(result) > 0 else pd.DataFrame([{}] * 1, columns=data.columns)
                for _ in range(additional_rows):
                    result = pd.concat([result, last_row], ignore_index=True)

            result = result.iloc[-steps:].copy()
            # Handle both scalar and array forecasts
            if hasattr(forecast, "values"):
                forecast_values = forecast.values
                if forecast_values.ndim == 0:  # 0-dimensional array (scalar)
                    forecast_values = [forecast_values.item()]
                elif forecast_values.ndim == 1 and len(forecast_values) == 1 and steps > 1:
                    # Repeat single value for multiple steps
                    forecast_values = [forecast_values[0]] * steps
            else:
                # Handle scalar values
                forecast_values = [forecast] * steps if np.isscalar(forecast) else forecast

            result["prediction"] = forecast_values
            results.append(result)
        else:
            # Multiple time series predictions
            for dim_values, group_data in data.groupby(self.dimensions):
                # Convert single value to tuple for consistency
                if not isinstance(dim_values, tuple):
                    dim_values = (dim_values,)

                fitted_model = self.fitted_models.get(dim_values)
                if fitted_model is None:
                    print(f"Warning: No fitted model for dimensions {dim_values}, skipping prediction")
                    continue

                try:
                    forecast = fitted_model.forecast(steps=steps)

                    # Create result DataFrame with same structure as input
                    result = group_data.iloc[-steps:].copy() if len(group_data) >= steps else group_data.copy()
                    if len(result) < steps:
                        # Extend with NaN rows if needed
                        additional_rows = steps - len(result)
                        last_row = result.iloc[-1:] if len(result) > 0 else pd.DataFrame([{}] * 1, columns=group_data.columns)
                        for _ in range(additional_rows):
                            result = pd.concat([result, last_row], ignore_index=True)

                    result = result.iloc[-steps:].copy()
                    # Handle both scalar and array forecasts
                    if hasattr(forecast, "values"):
                        forecast_values = forecast.values
                        if forecast_values.ndim == 0:  # 0-dimensional array (scalar)
                            forecast_values = [forecast_values.item()]
                        elif forecast_values.ndim == 1 and len(forecast_values) == 1 and steps > 1:
                            # Repeat single value for multiple steps
                            forecast_values = [forecast_values[0]] * steps
                    else:
                        # Handle scalar values
                        forecast_values = [forecast] * steps if np.isscalar(forecast) else forecast

                    result["prediction"] = forecast_values

                    # Ensure dimension columns are preserved
                    for i, dim_col in enumerate(self.dimensions):
                        result[dim_col] = dim_values[i] if len(dim_values) > i else dim_values[0]

                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to predict for dimensions {dim_values}: {e}")

        if not results:
            raise ValueError("No successful predictions were generated")

        return pd.concat(results, ignore_index=True)
