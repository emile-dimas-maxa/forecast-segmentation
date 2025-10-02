from pydantic import BaseModel, Field
from datetime import date
from enum import Enum


class ForecastMethod(str, Enum):
    """Available forecasting methods"""

    ZERO = "zero"
    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    WEIGHTED_MOVING_AVERAGE = "weighted_moving_average"
    ETS = "ets"
    ARIMA = "arima"
    SARIMA = "sarima"
    XGBOOST_INDIVIDUAL = "xgboost_individual"
    XGBOOST_GLOBAL = "xgboost_global"
    CROSTON = "croston"
    ENSEMBLE = "ensemble"
    SEGMENT_AGGREGATE = "segment_aggregate"


class EvaluationLevel(str, Enum):
    """Levels at which to evaluate forecasts"""

    DIM_VALUE = "dim_value"
    SEGMENT = "segment"
    OVERALL = "overall"


class ForecastingConfig(BaseModel):
    """Configuration for forecasting pipeline"""

    # Backtesting parameters
    train_start_date: date = Field(description="Start date for training data")
    train_end_date: date = Field(description="End date for training data")
    test_start_date: date = Field(description="Start date for test data")
    test_end_date: date = Field(description="End date for test data")

    # Forecast horizon
    forecast_horizon: int = Field(default=1, description="Number of months to forecast ahead")
    min_history_months: int = Field(default=12, description="Minimum months of history required for forecasting")

    # Methods to test
    methods_to_test: list[ForecastMethod] = Field(
        default=[
            ForecastMethod.ZERO,
            ForecastMethod.NAIVE,
            ForecastMethod.MOVING_AVERAGE,
            ForecastMethod.WEIGHTED_MOVING_AVERAGE,
            ForecastMethod.ARIMA,
        ],
        description="List of forecasting methods to test",
    )

    # Evaluation levels
    evaluation_levels: list[EvaluationLevel] = Field(
        default=[EvaluationLevel.DIM_VALUE, EvaluationLevel.SEGMENT, EvaluationLevel.OVERALL],
        description="Levels at which to evaluate forecasts",
    )

    # Method-specific configurations

    # Moving Average
    ma_windows: list[int] = Field(default=[3, 6, 12], description="Window sizes for moving average")
    ma_min_periods: int = Field(default=2, description="Minimum periods required for MA calculation")

    # Weighted Moving Average
    wma_weights: list[float] | None = Field(
        default=None, description="Weights for WMA (if None, uses exponentially decreasing weights)"
    )
    wma_window: int = Field(default=6, description="Window size for WMA")

    # ARIMA
    arima_auto_select: bool = Field(default=True, description="Auto-select ARIMA parameters")
    arima_max_p: int = Field(default=3, description="Maximum p for ARIMA")
    arima_max_d: int = Field(default=2, description="Maximum d for ARIMA")
    arima_max_q: int = Field(default=3, description="Maximum q for ARIMA")
    arima_seasonal: bool = Field(default=False, description="Include seasonal component")
    arima_m: int = Field(default=12, description="Seasonal period")

    # SARIMA
    sarima_seasonal_order: tuple | None = Field(default=(1, 1, 1, 12), description="Seasonal order (P, D, Q, m) for SARIMA")

    # XGBoost
    xgb_n_estimators: int = Field(default=100, description="Number of trees for XGBoost")
    xgb_max_depth: int = Field(default=5, description="Max depth for XGBoost")
    xgb_learning_rate: float = Field(default=0.1, description="Learning rate for XGBoost")
    xgb_subsample: float = Field(default=0.8, description="Subsample ratio for XGBoost")
    xgb_colsample_bytree: float = Field(default=0.8, description="Column sample ratio for XGBoost")
    xgb_use_lag_features: bool = Field(default=True, description="Include lag features")
    xgb_use_rolling_features: bool = Field(default=True, description="Include rolling features")
    xgb_use_date_features: bool = Field(default=True, description="Include date features")
    xgb_feature_columns: list[str] | None = Field(
        default=None, description="Specific feature columns to use (if None, uses all available)"
    )

    # Croston's method
    croston_alpha: float = Field(default=0.1, description="Smoothing parameter for Croston")

    # Ensemble
    ensemble_methods: list[ForecastMethod] | None = Field(
        default=None, description="Methods to include in ensemble (if None, uses top performing methods)"
    )
    ensemble_weights: dict[str, float] | None = Field(
        default=None, description="Weights for ensemble methods (if None, uses equal weights)"
    )
    ensemble_selection_metric: str = Field(default="mae", description="Metric to use for selecting ensemble methods")

    # Segment aggregation
    segment_agg_method: ForecastMethod = Field(
        default=ForecastMethod.MOVING_AVERAGE, description="Method to use for segment-level aggregation forecasting"
    )
    segment_agg_then_distribute: bool = Field(default=True, description="Whether to distribute segment forecast to dim_values")

    # Error metrics
    calculate_mae: bool = Field(default=True, description="Calculate Mean Absolute Error")
    calculate_rmse: bool = Field(default=True, description="Calculate Root Mean Square Error")
    calculate_mape: bool = Field(default=True, description="Calculate Mean Absolute Percentage Error")
    calculate_smape: bool = Field(default=True, description="Calculate Symmetric MAPE")
    calculate_mase: bool = Field(default=False, description="Calculate Mean Absolute Scaled Error")
    calculate_directional_accuracy: bool = Field(default=True, description="Calculate directional accuracy")

    # Performance thresholds
    mape_cap: float = Field(default=200.0, description="Cap MAPE at this value to handle division by zero")
    zero_threshold: float = Field(default=1e-6, description="Values below this are considered zero")

    # Parallel processing
    use_parallel: bool = Field(default=True, description="Use parallel processing where possible")
    n_parallel_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")

    # Output options
    save_forecasts: bool = Field(default=True, description="Save individual forecasts")
    save_errors: bool = Field(default=True, description="Save error metrics")
    output_table_prefix: str = Field(default="forecast_backtest", description="Prefix for output tables")

    class Config:
        """Pydantic config"""

        use_enum_values = True

    def get_ma_window_for_segment(self, segment_name: str) -> int:
        """Get appropriate MA window based on segment characteristics"""
        if "CONTINUOUS" in segment_name:
            return min(self.ma_windows)  # Use shorter window for continuous
        elif "INTERMITTENT" in segment_name:
            return max(self.ma_windows)  # Use longer window for intermittent
        else:
            return self.ma_windows[len(self.ma_windows) // 2]  # Use middle window

    def get_xgb_features(self) -> list[str]:
        """Get list of features to use for XGBoost"""
        if self.xgb_feature_columns:
            return self.xgb_feature_columns

        features = []

        # Lag features
        if self.xgb_use_lag_features:
            features.extend(["lag_1m_eom", "lag_3m_eom", "lag_12m_eom", "eom_ma3"])

        # Rolling features
        if self.xgb_use_rolling_features:
            features.extend(
                [
                    "raw_rf__rolling_avg_monthly_volume",
                    "raw_rf__rolling_avg_nonzero_eom",
                    "raw_rf__rolling_std_eom",
                    "raw_rf__rolling_nonzero_eom_months",
                    "raw_rf__active_months_12m",
                ]
            )

        # Date features
        if self.xgb_use_date_features:
            features.extend(["month_of_year", "is_quarter_end", "is_year_end"])

        # Pattern metrics
        features.extend(["raw_pm__eom_frequency", "raw_pm__eom_concentration", "raw_pm__eom_cv", "raw_pm__activity_rate"])

        return features
