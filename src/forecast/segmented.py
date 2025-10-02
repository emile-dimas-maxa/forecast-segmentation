"""MLflow custom pyfunc model for segmented forecasting."""

from typing import Any

import pandas as pd

from src.forecast.models.arima import ARIMAModel
from src.forecast.models.base import BaseSegmentModel
from src.forecast.models.moving_average import MovingAverageModel
from src.forecast.models.null import NullModel

from mlflow.pyfunc import PythonModel


class SegmentedForecastModel(PythonModel):
    def __init__(
        self,
        segment_col: str = "segment",
        target_col: str = "target",
        dimensions: list[str] = None,
        model_mapping: dict[str, dict[str, Any]] | None = None,
        fallback_model: dict[str, Any] | None = None,
    ):
        """
        Initialize segmented forecast model.

        Args:
            segment_col: Column name containing segment identifiers
            target_col: Column name containing target values to forecast
            dimensions: List of column names that define individual time series within each segment.
                       If empty/None, each segment is treated as one time series.
                       If provided, each unique combination of dimension values within a segment defines a separate time series.
            model_mapping: Dict mapping segment values to model configs
                          Format: {segment_value: {"type": "arima", "params": {...}}}
            fallback_model: Default model config for unseen segments
                           Format: {"type": "moving_average", "params": {...}}
        """
        self.segment_col = segment_col
        self.target_col = target_col
        self.dimensions = dimensions or []
        self.model_mapping = model_mapping or {}
        self.fallback_model = fallback_model or {"type": "moving_average", "params": {"window": 3}}
        self.fitted_models = {}

        # Model registry
        self._model_registry = {"arima": ARIMAModel, "moving_average": MovingAverageModel, "null": NullModel}

    def _create_model(self, model_config: dict[str, Any]) -> BaseSegmentModel:
        model_type = model_config["type"]
        params = model_config.get("params", {})

        if model_type not in self._model_registry:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self._model_registry.keys())}")

        return self._model_registry[model_type](**params)

    def fit(self, data: pd.DataFrame) -> "SegmentedForecastModel":
        """
        Fit models for each segment.

        Args:
            data: DataFrame with segment_col, target_col, and other features

        Returns:
            Self for method chaining
        """
        if self.segment_col not in data.columns:
            raise ValueError(f"Segment column '{self.segment_col}' not found in data")
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")

        self.fitted_models = {}

        for segment in data[self.segment_col].unique():
            segment_data = data[data[self.segment_col] == segment].copy()

            if segment in self.model_mapping:
                model_config = self.model_mapping[segment]
            else:
                model_config = self.fallback_model
                print(f"Using fallback model for segment '{segment}'")

            model = self._create_model(model_config)
            model.fit(segment_data, self.target_col, self.dimensions)
            self.fitted_models[segment] = model

        return self

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        """
        Generate predictions for each segment.

        Args:
            context: MLflow context (unused)
            model_input: DataFrame with segment data
            params: Additional parameters (e.g., {"steps": 5})

        Returns:
            DataFrame with predictions concatenated across segments
        """
        return self.transform(model_input, params)

    def transform(self, data: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Transform data by generating predictions for each segment.

        Args:
            data: DataFrame with segment data
            params: Additional parameters (e.g., {"steps": 5})

        Returns:
            DataFrame with predictions concatenated across segments
        """
        if not self.fitted_models:
            raise ValueError("Model must be fitted before transform/predict")

        params = params or {}
        steps = params.get("steps", 1)

        results = []

        for segment in data[self.segment_col].unique():
            segment_data = data[data[self.segment_col] == segment].copy()

            if segment in self.fitted_models:
                model = self.fitted_models[segment]
            else:
                print(f"Segment '{segment}' not seen during training, using fallback model")
                model = self._create_model(self.fallback_model)
                model.fit(segment_data, self.target_col, self.dimensions)

            segment_predictions = model.predict(segment_data, steps=steps)
            segment_predictions[self.segment_col] = segment
            results.append(segment_predictions)

        return pd.concat(results, ignore_index=True)
