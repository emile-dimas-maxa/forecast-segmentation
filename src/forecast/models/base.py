from abc import ABC, abstractmethod

import pandas as pd


class BaseSegmentModel(ABC):
    """Base class for segment-specific models."""

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_col: str = "target", dimensions: list[str] = None) -> "BaseSegmentModel":
        """
        Fit the model to the data.

        Args:
            data: DataFrame containing the time series data
            target_col: Column name containing target values to forecast
            dimensions: List of column names that define individual time series.
                       If empty/None, all data is treated as one time series.
                       If provided, each unique combination of dimension values defines a separate time series.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, steps: int = 1) -> pd.DataFrame:
        """Generate predictions."""
        pass
