from typing import Any, Generator, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field


from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from pydantic import BaseModel
from snowflake import snowpark


class BaseSplitter(BaseModel, ABC):
    @abstractmethod
    def split(self, dataset: Union[pd.DataFrame, snowpark.DataFrame], engine: str) -> tuple[str, str]:
        pass


class TimeSeriesBacktestError(Exception):
    """
    Custom exception for errors related to time series backtesting operations.
    Provides detailed error messages for various validation failures.
    """


class TimeSeriesBacktest(BaseSplitter):
    """
    Split the dataset into train and test sets based on a time series backtest.
    If the timeseries is multi-dimensional, make sure all the timeseries have the same dates.
    dates are fetched globally, not per timeseries and then the split is done on the global dates.
    """

    model_config = ConfigDict(extra="forbid")
    method: Literal["timeseries-backtest"] = Field(default="timeseries-backtest")
    forecast_horizon: int = Field(default=1, ge=1)
    input_steps: int = Field(default=1, ge=1)
    expanding_window: bool = Field(
        default=False,
        description="If True, training data includes all previous data points. If False, uses fixed-size training windows.",
    )
    stride: int = Field(default=1, ge=1)
    ascending: bool = Field(
        default=True,
        description="If True, splits start from earliest date and move forward. If False, splits start from latest date and move backward.",
    )
    date_column: str = Field(default="datetime")
    min_backtest_iterations: int = Field(default=1, ge=1)

    def number_of_splits(self, dataset: pd.DataFrame) -> int:
        """
        Calculate the number of possible train-test splits based on dataset size and parameters.
        formula: (number_of_dates - forecast_horizon - input_steps) // stride + 1

        Args:
            dataset: A pandas DataFrame containing the time series data

        Returns:
            int: The number of possible train-test splits
        """
        number_of_dates = len(dataset[self.date_column].unique())
        return (number_of_dates - self.input_steps - self.forecast_horizon) // self.stride + 1

    def split(self, dataset: pd.DataFrame) -> Generator[tuple[pd.Index, pd.Index], None, None]:
        """
        Split the dataset into train-test pairs for time series backtesting.

        Args:
            dataset: A pandas DataFrame containing the time series data

        Returns:
            A generator of tuples (train_idx, test_idx) containing indices for training and testing

        Raises:
            TimeSeriesBacktestError: If the dataset doesn't meet the requirements for backtesting
        """

        self._pre_verify(dataset=dataset)

        indices = np.arange(len(dataset))
        unique_dates = np.sort(dataset[self.date_column].unique())
        number_of_dates = len(unique_dates)

        if self.ascending:
            window_start = 0
            window_end = self.input_steps + self.forecast_horizon
            step = self.stride
        else:
            window_start = number_of_dates - (self.input_steps + self.forecast_horizon)
            window_end = number_of_dates
            step = -self.stride

        train_window_start = 0

        for _ in range(self.number_of_splits(dataset)):
            if not self.expanding_window:
                train_window_start = window_start

            train_date_indices = indices[train_window_start : window_start + self.input_steps]
            test_date_indices = indices[window_end - self.forecast_horizon : window_end]

            train_dates = unique_dates[train_date_indices]
            test_dates = unique_dates[test_date_indices]

            train_idx = dataset[dataset[self.date_column].isin(train_dates)].index
            test_idx = dataset[dataset[self.date_column].isin(test_dates)].index

            yield train_idx, test_idx

            window_start += step
            window_end += step

    def _pre_verify(self, dataset: Any) -> None:
        """
        Verify that the dataset meets the requirements for time series backtesting.

        Args:
            dataset: A pandas DataFrame containing the time series data

        Raises:
            TimeSeriesBacktestError: If the dataset doesn't meet the requirements
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TimeSeriesBacktestError("Dataset must be a local DataFrame")

        # verify that the datetime column is present:
        if self.date_column not in dataset.columns:
            raise TimeSeriesBacktestError(f"The datetime column {self.date_column} is not present in the dataset")

        number_of_splits = self.number_of_splits(dataset=dataset)
        if number_of_splits < self.min_backtest_iterations:
            raise ValueError(
                (
                    "Not enough dates to perform a time series backtest.\n"
                    f"- Minimum number of iterations is {self.min_backtest_iterations}\n"
                    f"- Number of possible iterations is {number_of_splits}\n"
                    f"- Forecast horizon is {self.forecast_horizon}\n"
                    f"- Input steps are {self.input_steps}\n"
                    f"- Stride is {self.stride}\n"
                    "- Formula: (number_of_dates - forecast_horizon - input_steps) // stride + 1",
                ),
            )
