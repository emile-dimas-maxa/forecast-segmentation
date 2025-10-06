from typing import Annotated, Union

from pydantic import Field
from src.new_forecast.models.arima import ArimaModel
from src.new_forecast.models.base import BaseForecastModel
from src.new_forecast.models.direction_arima import DirectionArimaModel
from src.new_forecast.models.direction_moving_average import DirectionMovingAverageModel
from src.new_forecast.models.direction_random_forest import DirectionRandomForestModel
from src.new_forecast.models.direction_xgboost import DirectionXGBoostModel
from src.new_forecast.models.moving_average import MovingAverageModel
from src.new_forecast.models.net_arima import NetArimaModel
from src.new_forecast.models.net_moving_average import NetMovingAverageModel
from src.new_forecast.models.net_random_forest import NetRandomForestModel
from src.new_forecast.models.net_xgboost import NetXGBoostModel
from src.new_forecast.models.null import NullModel
from src.new_forecast.models.random_forest import RandomForestModel
from src.new_forecast.models.xgboost import XGBoostModel


UNION_MODELS = Union[
    ArimaModel,
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
]

ModelType = Annotated[
    UNION_MODELS,
    Field(discriminator="name"),
]
