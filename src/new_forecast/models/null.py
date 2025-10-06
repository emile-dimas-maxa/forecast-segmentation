from typing import Literal

import pandas as pd
from pydantic import Field

from src.new_forecast.models.base import BaseForecastModel


class NullModel(BaseForecastModel):
    """Null model that always predicts 0."""

    name: Literal["null"] = Field(description="Name of the model", default="null")
    dimensions: list[str] = Field(description="Dimensions of the null model", default=[])
    target_col: str = Field(description="Target column of the null model", default="target")
    date_col: str = Field(description="Date column of the null model", default="forecast_month")
    forecast_horizon: int = Field(description="Forecast horizon of the null model", default=1)

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "NullModel":
        """Fit the model to the data (no fitting needed for null model)"""
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data (always returns 0)"""

        def _predict_group(group):
            """Generate predictions for a group (always 0)"""
            # Return predictions for each row in the group
            predictions = pd.DataFrame([0.0] * len(group), columns=["prediction"])
            return predictions

        data_grouped = data.groupby(self.dimensions) if self.dimensions else data

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
