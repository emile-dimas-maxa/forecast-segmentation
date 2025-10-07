from typing import Literal

import pandas as pd
from pydantic import Field
from sklearn.ensemble import RandomForestRegressor

from src.new_forecast.models.base import BaseForecastModel


class NetRandomForestModel(BaseForecastModel):
    """Net Random Forest model that calculates net (credit - debit) before fitting."""

    name: Literal["net_random_forest"] = Field(description="Name of the model", default="net_random_forest")
    n_estimators: int = Field(description="Number of trees in the forest", default=100)
    max_depth: int = Field(description="Maximum depth of the tree", default=None)
    min_samples_split: int = Field(description="Minimum number of samples required to split an internal node", default=2)
    min_samples_leaf: int = Field(description="Minimum number of samples required to be at a leaf node", default=1)
    dimensions: list[str] = Field(description="Dimensions of the model", default=["dim_value"])
    target_col: str = Field(description="Target column of the model", default="target")
    date_col: str = Field(description="Date column of the model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the model", default=1)
    models: dict[str, RandomForestRegressor] | None = Field(description="Fitted Random Forest model for net series", default=None)

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
            features_data[f"lag_{lag}"] = features_data[self.target_col].shift(lag)

        # Create rolling statistics
        for window in [3, 6, 12]:
            features_data[f"rolling_mean_{window}"] = features_data[self.target_col].rolling(window=window, min_periods=1).mean()
            features_data[f"rolling_std_{window}"] = features_data[self.target_col].rolling(window=window, min_periods=1).std()

        return features_data

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "NetRandomForestModel":
        """Fit the model to the data with net transformations."""
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

        def _fit_net(net_data):
            """Fit Random Forest for the net series"""
            if len(net_data) < 2:  # Need at least 2 samples for training
                return net_data

            X = net_data[feature_cols]
            y = net_data[self.target_col]

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
            model.fit(X, y)
            self.models["net"] = model

            return net_data

        features_data.groupby("direction").apply(_fit_net)

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data with net transformations."""
        if not self.models:
            raise ValueError(f"Net Random Forest model must be fitted before prediction")

        # Transform the data
        transformed_data = self._transform_data(data)

        # Create features
        features_data = self._create_features(transformed_data)

        # Prepare feature columns
        feature_cols = [col for col in features_data.columns if col not in [self.target_col, self.date_col, "direction"]]

        def _predict_net(net_data):
            """Generate predictions for the net series"""
            model = self.models.get("net")

            if model is None or len(net_data) == 0:
                prediction = pd.DataFrame([float("nan")] * self.forecast_horizon, columns=["prediction"])
            else:
                # Use the last row for prediction (most recent features)
                last_row = net_data.iloc[-1:][feature_cols]
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
            prediction["direction"] = "net"
            return prediction

        # Group by direction and predict
        results = features_data.groupby("direction").apply(_predict_net)

        if results.index.nlevels > 1:
            results = results.reset_index(level=0, drop=True)
        results = results.reset_index(drop=True)

        # Add date column for predictions
        max_date = data[self.date_col].max()
        results[self.date_col] = max_date + pd.DateOffset(months=self.forecast_horizon)

        # Add dim_value column for net
        results["dim_value"] = "net"

        return pd.concat([data, results], ignore_index=True)
