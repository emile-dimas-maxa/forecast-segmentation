from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class BaseForecastModel(BaseModel, ABC):
    """Base model for all forecast models"""

    name: Literal[
        "arima",
        "direction_arima",
        "direction_moving_average",
        "direction_random_forest",
        "direction_xgboost",
        "moving_average",
        "net_arima",
        "net_moving_average",
        "net_random_forest",
        "net_xgboost",
        "null",
        "random_forest",
        "xgboost",
    ] = Field(description="Name of the model")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseModel":
        """Fit the model to the data"""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the data"""
        pass
