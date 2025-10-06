from typing import Literal

import pandas as pd
from pydantic import Field
from sklearn.ensemble import RandomForestRegressor

from src.new_forecast.models.base import BaseForecastModel


class RandomForestModel(BaseForecastModel):
    """Random Forest model implementation."""

    name: Literal["random_forest"] = Field(description="Name of the model", default="random_forest")
    n_estimators: int = Field(description="Number of trees in the forest", default=100)
    max_depth: int = Field(description="Maximum depth of the tree", default=None)
    min_samples_split: int = Field(description="Minimum number of samples required to split an internal node", default=2)
    min_samples_leaf: int = Field(description="Minimum number of samples required to be at a leaf node", default=1)
    dimensions: list[str] = Field(description="Dimensions of the model", default=[])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="date")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    models: dict[tuple, RandomForestRegressor] | None = Field(description="Fitted Random Forest models", default=None)

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for Random Forest model."""
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
    ) -> "RandomForestModel":
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
            """Fit Random Forest model for a group"""
            if len(group) < 2:  # Need at least 2 samples for training
                return group

            X = group[feature_cols]
            y = group[self.target_col]

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
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
            raise ValueError("Model must be fitted before prediction")

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
