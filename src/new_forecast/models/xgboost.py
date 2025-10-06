from typing import Literal

import pandas as pd
from pydantic import Field
from xgboost import XGBRegressor

from src.new_forecast.models.base import BaseForecastModel


class XGBoostModel(BaseForecastModel):
    """XGBoost model implementation."""

    name: Literal["xgboost"] = Field(description="Name of the model", default="xgboost")
    n_estimators: int = Field(description="Number of boosting rounds", default=100)
    max_depth: int = Field(description="Maximum tree depth", default=6)
    learning_rate: float = Field(description="Boosting learning rate", default=0.1)
    dimensions: list[str] = Field(description="Dimensions of the model", default=[])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    models: dict[tuple, XGBRegressor] | None = Field(description="Fitted XGBoost models", default=None)

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for XGBoost model."""
        features_data = data.copy()

        # Sort by date to ensure proper feature creation
        features_data = features_data.sort_values(self.date_col)

        # Create time-based features
        features_data["year"] = features_data[self.date_col].dt.year
        features_data["month"] = features_data[self.date_col].dt.month
        features_data["day"] = features_data[self.date_col].dt.day
        features_data["dayofweek"] = features_data[self.date_col].dt.dayofweek
        features_data["dayofyear"] = features_data[self.date_col].dt.dayofyear

        # Create lag features
        for lag in [1, 2, 3, 6, 12]:
            features_data[f"lag_{lag}"] = features_data.groupby(self.dimensions)[self.target_col].shift(lag)

        # Create rolling statistics
        for window in [3, 6, 12]:
            features_data[f"rolling_mean_{window}"] = (
                features_data.groupby(self.dimensions)[self.target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            features_data[f"rolling_std_{window}"] = (
                features_data.groupby(self.dimensions)[self.target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

        # Handle categorical columns by converting to numeric
        categorical_cols = features_data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in [self.target_col, self.date_col] + self.dimensions:
                # Convert categorical to numeric using label encoding
                features_data[col] = pd.Categorical(features_data[col]).codes

        return features_data

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "XGBoostModel":
        """Fit the model to the data"""
        # Create features
        features_data = self._create_features(data)

        # Remove rows with NaN values (from lag features)
        features_data = features_data.dropna()

        # Initialize models dictionary
        self.models = {}

        # Prepare feature columns (exclude target and date columns)
        feature_cols = [col for col in features_data.columns if col not in [self.target_col, self.date_col] + self.dimensions]

        data_grouped = features_data.groupby(self.dimensions) if self.dimensions else features_data

        def _fit_group(group):
            """Fit XGBoost model for a group"""
            if len(group) < 2:  # Need at least 2 samples for training
                return group

            X = group[feature_cols]
            y = group[self.target_col]

            model = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42
            )
            model.fit(X, y)

            key = group.name if isinstance(group.name, tuple) else (group.name,) if self.dimensions else tuple()
            self.models[key] = model

            return group

        data_grouped.apply(_fit_group)

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data"""
        if not self.models:
            raise ValueError(f"XGBoost model must be fitted before prediction")

        # Create features
        features_data = self._create_features(data)

        # Prepare feature columns
        feature_cols = [col for col in features_data.columns if col not in [self.target_col, self.date_col] + self.dimensions]

        def _predict_group(group):
            """Generate predictions for a group"""
            key = group.name if isinstance(group.name, tuple) else (group.name,) if self.dimensions else tuple()
            model = self.models.get(key)

            if model is None or len(group) == 0:
                # Return NaN predictions for each row in the group
                prediction = pd.DataFrame([float("nan")] * len(group), columns=["prediction"])
            else:
                # Generate predictions for each row in the group
                X = group[feature_cols]
                predictions = model.predict(X)
                prediction = pd.DataFrame(predictions, columns=["prediction"])

            return prediction

        data_grouped = features_data.groupby(self.dimensions) if self.dimensions else features_data

        results = data_grouped.apply(_predict_group)

        if self.dimensions:
            results = results.reset_index()

        return results
