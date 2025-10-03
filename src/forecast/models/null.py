import pandas as pd

from src.forecast.models.base import BaseSegmentModel


class NullModel(BaseSegmentModel):
    """Null model implementation that always forecasts zero values."""

    def __init__(self):
        """Initialize the null model."""
        self.dimensions = None
        self.target_col = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, target_col: str = "target", dimensions: list[str] = None) -> "NullModel":
        """
        Fit null model (no actual fitting needed, just store metadata).

        Args:
            data: DataFrame containing the time series data
            target_col: Column name containing target values to forecast
            dimensions: List of column names that define individual time series.
                       If empty/None, all data is treated as one time series.
                       If provided, each unique combination of dimension values defines a separate time series.
        """
        self.target_col = target_col
        self.dimensions = dimensions or []
        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """
        Generate null predictions (always returns 0).

        Args:
            data: DataFrame with input data
            steps: Number of steps to forecast

        Returns:
            DataFrame with zero predictions for each time series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        results = []

        if not self.dimensions:
            # Single time series prediction
            result = data.iloc[-steps:].copy() if len(data) >= steps else data.copy()
            if len(result) < steps:
                # Extend with additional rows if needed
                additional_rows = steps - len(result)
                last_row = result.iloc[-1:] if len(result) > 0 else pd.DataFrame([{}] * 1, columns=data.columns)
                for _ in range(additional_rows):
                    result = pd.concat([result, last_row], ignore_index=True)

            result = result.iloc[-steps:].copy()
            result["prediction"] = 0.0  # Always predict zero
            results.append(result)
        else:
            # Multiple time series predictions
            for dim_values, group_data in data.groupby(self.dimensions):
                # Convert single value to tuple for consistency
                if not isinstance(dim_values, tuple):
                    dim_values = (dim_values,)

                # Create result DataFrame
                result = group_data.iloc[-steps:].copy() if len(group_data) >= steps else group_data.copy()
                if len(result) < steps:
                    # Extend with additional rows if needed
                    additional_rows = steps - len(result)
                    last_row = result.iloc[-1:] if len(result) > 0 else pd.DataFrame([{}] * 1, columns=group_data.columns)
                    for _ in range(additional_rows):
                        result = pd.concat([result, last_row], ignore_index=True)

                result = result.iloc[-steps:].copy()
                result["prediction"] = 0.0  # Always predict zero

                # Ensure dimension columns are preserved
                for i, dim_col in enumerate(self.dimensions):
                    result[dim_col] = dim_values[i] if len(dim_values) > i else dim_values[0]

                results.append(result)

        if not results:
            raise ValueError("No predictions were generated")

        return pd.concat(results, ignore_index=True)
