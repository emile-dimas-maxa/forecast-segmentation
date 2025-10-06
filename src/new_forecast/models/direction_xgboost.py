from typing import Literal

import pandas as pd
from pydantic import Field
from xgboost import XGBRegressor

from src.new_forecast.models.base import BaseForecastModel


class DirectionXGBoostModel(BaseForecastModel):
    """Direction-aware XGBoost model that performs data transformations before fitting."""

    name: Literal["direction_xgboost"] = Field(description="Name of the model", default="direction_xgboost")
    n_estimators: int = Field(description="Number of boosting rounds", default=100)
    max_depth: int = Field(description="Maximum tree depth", default=6)
    learning_rate: float = Field(description="Boosting learning rate", default=0.1)
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    models: dict[str, XGBRegressor] | None = Field(description="Fitted XGBoost models for each direction", default=None)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by creating direction column and aggregating by direction and date."""
        # Create direction column by splitting dim_value on "::" and taking second part
        data = data.copy()
        data["direction"] = data["dim_value"].str.split("::").str[1]

        # Group by direction and date, sum the target column
        transformed_data = data.groupby(["direction", self.date_col])[self.target_col].sum().reset_index()

        return transformed_data

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
            features_data[f"lag_{lag}"] = features_data.groupby("direction")[self.target_col].shift(lag)

        # Create rolling statistics
        for window in [3, 6, 12]:
            features_data[f"rolling_mean_{window}"] = (
                features_data.groupby("direction")[self.target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            features_data[f"rolling_std_{window}"] = (
                features_data.groupby("direction")[self.target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

        return features_data

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "DirectionXGBoostModel":
        """Fit the model to the data with transformations."""
        # Transform the data
        transformed_data = self._transform_data(data)

        # Create features
        features_data = self._create_features(transformed_data)

        # Remove rows with NaN values (from lag features)
        features_data = features_data.dropna()

        # Initialize models dictionary
        self.models = {}

        # Prepare feature columns (exclude target, date, and direction columns)
        feature_cols = [col for col in features_data.columns if col not in [self.target_col, self.date_col, "direction"]]

        def _fit_direction(direction_data):
            """Fit XGBoost for a specific direction"""
            if len(direction_data) < 2:  # Need at least 2 samples for training
                return direction_data

            direction = direction_data["direction"].iloc[0]
            X = direction_data[feature_cols]
            y = direction_data[self.target_col]

            model = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42
            )
            model.fit(X, y)
            self.models[direction] = model

            return direction_data

        features_data.groupby("direction").apply(_fit_direction)

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with transformations."""
        if not self.models:
            raise ValueError(f"Direction XGBoost model must be fitted before prediction")

        # Transform the data
        transformed_data = self._transform_data(data)

        # Create features
        features_data = self._create_features(transformed_data)

        # Prepare feature columns
        feature_cols = [col for col in features_data.columns if col not in [self.target_col, self.date_col, "direction"]]

        def _predict_direction(direction_data):
            """Generate predictions for a specific direction"""
            direction = direction_data["direction"].iloc[0]
            model = self.models.get(direction)

            if model is None or len(direction_data) == 0:
                prediction = pd.DataFrame([float("nan")] * self.forecast_horizon, columns=["prediction"])
            else:
                # Use the last row for prediction (most recent features)
                last_row = direction_data.iloc[-1:][feature_cols]
                predictions = []

                for _ in range(self.forecast_horizon):
                    pred_result = model.predict(last_row)
                    # Handle both scalar and array results
                    if hasattr(pred_result, "__len__") and len(pred_result) > 0:
                        pred = pred_result[0]
                    else:
                        pred = float(pred_result)
                    predictions.append(pred)
                    # Update features for next prediction (simplified)
                    last_row = last_row.copy()

                prediction = pd.DataFrame(predictions, columns=["prediction"])

            # Add direction column to prediction
            prediction["direction"] = direction
            return prediction

        # Group by direction and predict
        results = features_data.groupby("direction").apply(_predict_direction)

        if results.index.nlevels > 1:
            results = results.reset_index(level=0, drop=True)
        results = results.reset_index(drop=True)

        # Add date column for predictions
        max_date = data[self.date_col].max()
        results[self.date_col] = max_date + pd.DateOffset(months=self.forecast_horizon)

        # Add dim_value column by reconstructing it from direction
        results["dim_value"] = "::" + results["direction"]

        return pd.concat([data, results], ignore_index=True)
