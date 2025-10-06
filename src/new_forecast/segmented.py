"""Segmented forecasting model for the new forecast models."""

from typing import Any

import pandas as pd
from loguru import logger

from src.new_forecast.models import (
    ArimaModel,
    BaseForecastModel,
    DirectionArimaModel,
    DirectionMovingAverageModel,
    DirectionRandomForestModel,
    DirectionXGBoostModel,
    MovingAverageModel,
    NetArimaModel,
    NetMovingAverageModel,
    NetRandomForestModel,
    NetXGBoostModel,
    NullModel,
    RandomForestModel,
    XGBoostModel,
)


# Create a union type for all forecast models using discriminator
ForecastModelUnion = (
    ArimaModel
    | DirectionArimaModel
    | DirectionMovingAverageModel
    | DirectionRandomForestModel
    | DirectionXGBoostModel
    | MovingAverageModel
    | NetArimaModel
    | NetMovingAverageModel
    | NetRandomForestModel
    | NetXGBoostModel
    | NullModel
    | RandomForestModel
    | XGBoostModel
)


class NewSegmentedForecastModel:
    """Segmented forecast model that works with the new forecast model architecture."""

    def __init__(
        self,
        segment_col: str = "segment",
        target_col: str = "target",
        date_col: str = "forecast_month",
        dimensions: list[str] = None,
        model_mapping: dict[str, dict[str, Any]] | None = None,
        fallback_model: dict[str, Any] | None = None,
    ):
        """
        Initialize segmented forecast model.

        Args:
            segment_col: Column name containing segment identifiers
            target_col: Column name containing target values to forecast
            date_col: Column name containing date values
            dimensions: List of column names that define individual time series within each segment.
                       If empty/None, each segment is treated as one time series.
                       If provided, each unique combination of dimension values within a segment defines a separate time series.
            model_mapping: Dict mapping segment values to model configs
                          Format: {segment_value: {"name": "arima", "order": (1,1,1), ...}}
            fallback_model: Default model config for unseen segments
                           Format: {"name": "moving_average", "window": 3, ...}
        """
        self.segment_col = segment_col
        self.target_col = target_col
        self.date_col = date_col
        self.dimensions = dimensions or []
        self.model_mapping = model_mapping or {}
        self.fallback_model = fallback_model or {"name": "moving_average", "window": 3}
        self.fitted_models = {}

        # Model registry using discriminator pattern
        self._model_registry = {
            "arima": ArimaModel,
            "direction_arima": DirectionArimaModel,
            "direction_moving_average": DirectionMovingAverageModel,
            "direction_random_forest": DirectionRandomForestModel,
            "direction_xgboost": DirectionXGBoostModel,
            "moving_average": MovingAverageModel,
            "net_arima": NetArimaModel,
            "net_moving_average": NetMovingAverageModel,
            "net_random_forest": NetRandomForestModel,
            "net_xgboost": NetXGBoostModel,
            "null": NullModel,
            "random_forest": RandomForestModel,
            "xgboost": XGBoostModel,
        }

    def _create_model(self, model_config: dict[str, Any]) -> BaseForecastModel:
        """Create a model instance from configuration using discriminator pattern."""
        # Create a temporary config with common parameters
        config_data = model_config.copy()
        config_data.update(
            {
                "target_col": self.target_col,
                "date_col": self.date_col,
                "dimensions": self.dimensions,
            }
        )

        # Use the model's name as discriminator to create the appropriate model instance
        model_name = config_data["name"]

        if model_name not in self._model_registry:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(self._model_registry.keys())}")

        # Get the model class and create instance with validated parameters
        model_class = self._model_registry[model_name]

        # Create model instance with Pydantic validation
        # The discriminator is handled by the model's 'name' field
        return model_class(**config_data)

    def _create_model_with_discriminator(self, model_config: dict[str, Any]) -> BaseForecastModel:
        """Create a model instance using Pydantic's discriminator pattern."""
        # Create a temporary config with common parameters
        config_data = model_config.copy()
        config_data.update(
            {
                "target_col": self.target_col,
                "date_col": self.date_col,
                "dimensions": self.dimensions,
            }
        )

        # Use Pydantic's model_validate to handle discriminator automatically
        # This leverages the 'name' field as the discriminator
        try:
            # Try to parse as a union type using the discriminator
            # The BaseForecastModel's name field acts as the discriminator
            return BaseForecastModel.model_validate(config_data)
        except Exception:
            # Fallback to manual registry lookup
            model_name = config_data["name"]
            if model_name not in self._model_registry:
                raise ValueError(f"Unknown model name: {model_name}. Available: {list(self._model_registry.keys())}") from None

            model_class = self._model_registry[model_name]
            return model_class(**config_data)

    def fit(self, data: pd.DataFrame) -> "NewSegmentedForecastModel":
        """
        Fit models for each segment.

        Args:
            data: DataFrame with segment_col, target_col, date_col, and other features

        Returns:
            Self for method chaining
        """
        if self.segment_col not in data.columns:
            raise ValueError(f"Segment column '{self.segment_col}' not found in data")
        if self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        if self.date_col not in data.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")

        self.fitted_models = {}

        def _fit_segment(segment_data):
            """Fit model for a specific segment."""
            segment = segment_data[self.segment_col].iloc[0]

            if segment in self.model_mapping:
                model_config = self.model_mapping[segment]
            else:
                model_config = self.fallback_model
                logger.info(f"Using fallback model for segment '{segment}'")

            model = self._create_model_with_discriminator(model_config)
            model.fit(segment_data)
            self.fitted_models[segment] = model

            return segment_data

        data.groupby(self.segment_col).apply(_fit_segment)

        return self

    def predict(self, data: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Generate predictions for each segment.

        Args:
            data: DataFrame with segment data
            forecast_horizon: Number of steps to forecast ahead

        Returns:
            DataFrame with predictions concatenated across segments
        """
        if not self.fitted_models:
            raise ValueError("Model must be fitted before prediction")

        def _predict_segment(segment_data):
            """Generate predictions for a specific segment."""
            segment = segment_data[self.segment_col].iloc[0]

            if segment in self.fitted_models:
                model = self.fitted_models[segment]
                # Update forecast horizon if different from model default
                if hasattr(model, "forecast_horizon"):
                    model.forecast_horizon = forecast_horizon
            else:
                logger.info(f"Segment '{segment}' not seen during training, using fallback model")
                model = self._create_model_with_discriminator(self.fallback_model)
                model.forecast_horizon = forecast_horizon
                model.fit(segment_data)

            segment_predictions = model.predict(segment_data)

            # Ensure the segment column is properly set
            if self.segment_col not in segment_predictions.columns:
                segment_predictions[self.segment_col] = segment
            else:
                # If it exists, make sure all values are set to the segment
                segment_predictions[self.segment_col] = segment

            return segment_predictions

        results = data.groupby(self.segment_col).apply(_predict_segment)

        if results.index.nlevels > 1:
            results = results.reset_index(level=0, drop=True)
        results = results.reset_index(drop=True)

        return results

    def get_model_info(self) -> dict[str, dict[str, Any]]:
        """
        Get information about fitted models.

        Returns:
            Dictionary mapping segment names to model information
        """
        info = {}
        for segment, model in self.fitted_models.items():
            info[segment] = {
                "model_name": model.name,
                "model_type": type(model).__name__,
                "parameters": model.model_dump() if hasattr(model, "model_dump") else {},
            }
        return info

    def get_available_models(self) -> list[str]:
        """
        Get list of available model names.

        Returns:
            List of available model names
        """
        return list(self._model_registry.keys())
