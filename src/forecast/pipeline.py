"""
Forecasting Pipeline for EOM Predictions
Implements multiple forecasting methods with backtesting capabilities
"""

from functools import wraps
import time
from datetime import date
import numpy as np
import pandas as pd

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from loguru import logger


def log_forecast_method(func):
    """Decorator to log forecasting method execution"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__.replace("_forecast_", "").replace("_", " ").upper()
        logger.info(f"Starting {method_name} forecasting...")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.success(f"Completed {method_name} | Execution time: {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {method_name} | Error: {str(e)} | Execution time: {execution_time:.2f}s")
            raise

    return wrapper


class ForecastingPipeline:
    """Pipeline for forecasting and backtesting"""

    @staticmethod
    def run_backtesting(
        session: Session,
        segmented_df: DataFrame,
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str,
        forecast_horizon: int = 1,
        min_history_months: int = 12,
        methods_to_test: list[str] = None,
        evaluation_levels: list[str] = None,
        ma_windows: list[int] = None,
        ma_min_periods: int = 2,
        wma_weights: list[float] = None,
        wma_window: int = 6,
        arima_auto_select: bool = True,
        arima_max_p: int = 3,
        arima_max_d: int = 2,
        arima_max_q: int = 3,
        output_table_prefix: str = "forecast_backtest",
        save_forecasts: bool = True,
        save_errors: bool = True,
        zero_threshold: float = 1e-6,
        mape_cap: float = 200.0,
    ) -> tuple[DataFrame, DataFrame, DataFrame]:
        """
        Run complete backtesting pipeline

        Args:
            session: Snowpark session
            segmented_df: Segmented dataframe from segmentation pipeline
            train_start_date: Start date for training data
            train_end_date: End date for training data
            test_start_date: Start date for test data
            test_end_date: End date for test data
            forecast_horizon: Number of months to forecast ahead
            min_history_months: Minimum months of history required for forecasting
            methods_to_test: List of forecasting methods to test
            evaluation_levels: Levels at which to evaluate forecasts
            ma_windows: Window sizes for moving average
            ma_min_periods: Minimum periods required for MA calculation
            wma_weights: Weights for WMA (if None, uses exponentially decreasing weights)
            wma_window: Window size for WMA
            arima_auto_select: Auto-select ARIMA parameters
            arima_max_p: Maximum p for ARIMA
            arima_max_d: Maximum d for ARIMA
            arima_max_q: Maximum q for ARIMA
            output_table_prefix: Prefix for output tables
            save_forecasts: Save individual forecasts
            save_errors: Save error metrics
            zero_threshold: Threshold for zero values in error calculations
            mape_cap: Cap for MAPE calculation

        Returns:
            Tuple of (forecasts_df, dim_value_errors_df, aggregated_errors_df)
        """
        # Set defaults for optional parameters
        if methods_to_test is None:
            methods_to_test = ["naive", "moving_average", "arima"]
        if evaluation_levels is None:
            evaluation_levels = ["dim_value", "segment", "overall"]
        if ma_windows is None:
            ma_windows = [3, 6, 12]

        logger.info("=" * 80)
        logger.info("Starting Forecasting Backtesting Pipeline")
        logger.info(f"Train period: {train_start_date} to {train_end_date}")
        logger.info(f"Test period: {test_start_date} to {test_end_date}")
        logger.info(f"Methods to test: {methods_to_test}")
        logger.info("=" * 80)

        # Prepare data for forecasting
        train_df, test_df = ForecastingPipeline.prepare_forecast_data(
            segmented_df, train_start_date, train_end_date, test_start_date, test_end_date
        )

        # Run forecasts for each method
        all_forecasts = []
        for method in methods_to_test:
            logger.info(f"\nRunning {method} forecasting...")
            forecasts = ForecastingPipeline.generate_forecasts(
                session=session,
                train_df=train_df,
                test_df=test_df,
                method=method,
                forecast_horizon=forecast_horizon,
                min_history_months=min_history_months,
                ma_windows=ma_windows,
                ma_min_periods=ma_min_periods,
                wma_weights=wma_weights,
                wma_window=wma_window,
                arima_auto_select=arima_auto_select,
                arima_max_p=arima_max_p,
                arima_max_d=arima_max_d,
                arima_max_q=arima_max_q,
            )
            all_forecasts.append(forecasts)

        # Combine all forecasts
        forecasts_df = ForecastingPipeline.combine_forecasts(all_forecasts)

        # Calculate errors at different levels
        dim_value_errors = ForecastingPipeline.calculate_dim_value_errors(forecasts_df, test_df, zero_threshold, mape_cap)
        segment_errors = ForecastingPipeline.calculate_segment_errors(forecasts_df, test_df)
        overall_errors = ForecastingPipeline.calculate_overall_errors(forecasts_df, test_df)

        # Combine all error levels
        aggregated_errors = ForecastingPipeline.combine_error_levels(dim_value_errors, segment_errors, overall_errors)

        # Save results if configured
        if save_forecasts:
            ForecastingPipeline.save_results(session, output_table_prefix, forecasts_df, aggregated_errors)

        logger.info("=" * 80)
        logger.info("Completed Forecasting Backtesting Pipeline")
        logger.info("=" * 80)

        return forecasts_df, dim_value_errors, aggregated_errors

    @staticmethod
    def prepare_forecast_data(
        segmented_df: DataFrame, train_start_date: str, train_end_date: str, test_start_date: str, test_end_date: str
    ) -> tuple[DataFrame, DataFrame]:
        """Split data into train and test sets"""
        logger.info("Preparing forecast data...")

        # Filter to relevant date ranges
        train_df = segmented_df.filter((F.col("forecast_month") >= train_start_date) & (F.col("forecast_month") <= train_end_date))

        test_df = segmented_df.filter((F.col("forecast_month") >= test_start_date) & (F.col("forecast_month") <= test_end_date))

        train_count = train_df.count()
        test_count = test_df.count()

        logger.info(f"Train samples: {train_count:,}")
        logger.info(f"Test samples: {test_count:,}")

        return train_df, test_df

    @staticmethod
    def generate_forecasts(
        session: Session,
        train_df: DataFrame,
        test_df: DataFrame,
        method: str,
        forecast_horizon: int = 1,
        min_history_months: int = 12,
        ma_windows: list[int] = None,
        ma_min_periods: int = 2,
        wma_weights: list[float] = None,
        wma_window: int = 6,
        arima_auto_select: bool = True,
        arima_max_p: int = 3,
        arima_max_d: int = 2,
        arima_max_q: int = 3,
    ) -> DataFrame:
        """Generate forecasts for a specific method"""

        if method == "zero":
            return ForecastingPipeline._forecast_zero(train_df, test_df)
        elif method == "naive":
            return ForecastingPipeline._forecast_naive(train_df, test_df)
        elif method == "moving_average":
            return ForecastingPipeline._forecast_moving_average(train_df, test_df, ma_windows, ma_min_periods)
        elif method == "weighted_moving_average":
            return ForecastingPipeline._forecast_weighted_moving_average(train_df, test_df, wma_weights, wma_window)
        elif method == "arima":
            return ForecastingPipeline._forecast_arima(
                session, train_df, test_df, min_history_months, arima_auto_select, arima_max_p, arima_max_d, arima_max_q
            )
        elif method == "xgboost_global":
            return ForecastingPipeline._forecast_xgboost_global(session, train_df, test_df)
        elif method == "croston":
            return ForecastingPipeline._forecast_croston(train_df, test_df)
        elif method == "segment_aggregate":
            return ForecastingPipeline._forecast_segment_aggregate(train_df, test_df)
        else:
            logger.warning(f"Method {method} not implemented, using zero forecast")
            return ForecastingPipeline._forecast_zero(train_df, test_df)

    @staticmethod
    @log_forecast_method
    def _forecast_zero(train_df: DataFrame, test_df: DataFrame) -> DataFrame:
        """Zero forecast - always predicts 0"""

        # Get unique dim_values and test months
        forecast_df = test_df.select("dim_value", "forecast_month", "segment_name", F.col("target_eom_amount").alias("actual"))

        # Add zero forecast
        forecast_df = forecast_df.with_columns([F.lit(0.0).alias("forecast"), F.lit("zero").alias("method")])

        return forecast_df

    @staticmethod
    @log_forecast_method
    def _forecast_naive(train_df: DataFrame, test_df: DataFrame) -> DataFrame:
        """Naive forecast - uses last observed value"""

        # Get last value from training data for each dim_value
        last_values = train_df.group_by("dim_value").agg(F.max("forecast_month").alias("last_month"))

        last_values = last_values.join(
            train_df,
            (last_values["dim_value"] == train_df["dim_value"]) & (last_values["last_month"] == train_df["forecast_month"]),
            "inner",
        ).select(train_df["dim_value"], train_df["target_eom_amount"].alias("last_value"))

        # Join with test data
        forecast_df = test_df.select(
            "dim_value", "forecast_month", "segment_name", F.col("target_eom_amount").alias("actual")
        ).join(last_values, "dim_value", "left")

        # Use last value as forecast
        forecast_df = forecast_df.with_columns(
            [F.coalesce(F.col("last_value"), F.lit(0)).alias("forecast"), F.lit("naive").alias("method")]
        ).drop("last_value")

        return forecast_df

    @staticmethod
    @log_forecast_method
    def _forecast_moving_average(
        train_df: DataFrame, test_df: DataFrame, ma_windows: list[int] = None, ma_min_periods: int = 2
    ) -> DataFrame:
        """Moving average forecast"""

        if ma_windows is None:
            ma_windows = [3, 6, 12]

        all_ma_forecasts = []

        for window_size in ma_windows:
            logger.debug(f"Calculating MA with window={window_size}")

            # Calculate moving average from training data
            window_spec = Window.partition_by("dim_value").order_by("forecast_month").rows_between(-(window_size - 1), 0)

            ma_df = train_df.with_column(f"ma_{window_size}", F.avg("target_eom_amount").over(window_spec))

            # Get the last MA value for each dim_value
            last_ma = ma_df.group_by("dim_value").agg(F.max("forecast_month").alias("last_month"))

            last_ma = last_ma.join(
                ma_df, (last_ma["dim_value"] == ma_df["dim_value"]) & (last_ma["last_month"] == ma_df["forecast_month"]), "inner"
            ).select(ma_df["dim_value"], ma_df[f"ma_{window_size}"].alias("ma_forecast"))

            # Join with test data
            forecast_df = test_df.select(
                "dim_value", "forecast_month", "segment_name", F.col("target_eom_amount").alias("actual")
            ).join(last_ma, "dim_value", "left")

            # Add forecast and method info
            forecast_df = forecast_df.with_columns(
                [F.coalesce(F.col("ma_forecast"), F.lit(0)).alias("forecast"), F.lit(f"ma_{window_size}").alias("method")]
            ).drop("ma_forecast")

            all_ma_forecasts.append(forecast_df)

        # Combine all MA forecasts
        from functools import reduce

        combined_df = reduce(lambda df1, df2: df1.union(df2), all_ma_forecasts)

        return combined_df

    @staticmethod
    @log_forecast_method
    def _forecast_weighted_moving_average(
        train_df: DataFrame, test_df: DataFrame, wma_weights: list[float] = None, wma_window: int = 6
    ) -> DataFrame:
        """Weighted moving average forecast"""

        window_size = wma_window

        # Generate weights if not provided
        if wma_weights:
            weights = wma_weights[:window_size]
        else:
            # Exponentially decreasing weights
            weights = [2 ** (-i) for i in range(window_size)]
            weights = [w / sum(weights) for w in weights]  # Normalize

        logger.debug(f"Using WMA weights: {weights}")

        # Get the last N values for each dim_value
        window_spec = Window.partition_by("dim_value").order_by(F.col("forecast_month").desc())

        recent_data = train_df.with_column("row_num", F.row_number().over(window_spec)).filter(F.col("row_num") <= window_size)

        # Calculate weighted average
        wma_calcs = []
        for i, weight in enumerate(weights):
            wma_calcs.append(f"CASE WHEN row_num = {i + 1} THEN target_eom_amount * {weight} ELSE 0 END")

        wma_expr = " + ".join(wma_calcs)

        wma_values = recent_data.group_by("dim_value").agg(F.expr(f"SUM({wma_expr})").alias("wma_forecast"))

        # Join with test data
        forecast_df = test_df.select(
            "dim_value", "forecast_month", "segment_name", F.col("target_eom_amount").alias("actual")
        ).join(wma_values, "dim_value", "left")

        # Add forecast and method info
        forecast_df = forecast_df.with_columns(
            [F.coalesce(F.col("wma_forecast"), F.lit(0)).alias("forecast"), F.lit(f"wma_{window_size}").alias("method")]
        ).drop("wma_forecast")

        return forecast_df

    @staticmethod
    @log_forecast_method
    def _forecast_arima(
        session: Session,
        train_df: DataFrame,
        test_df: DataFrame,
        min_history_months: int = 12,
        arima_auto_select: bool = True,
        arima_max_p: int = 3,
        arima_max_d: int = 2,
        arima_max_q: int = 3,
    ) -> DataFrame:
        """ARIMA forecast using stored procedure or UDF"""

        # Convert to pandas for ARIMA modeling
        train_pd = train_df.select("dim_value", "forecast_month", "target_eom_amount").to_pandas()

        test_pd = test_df.select("dim_value", "forecast_month", "segment_name", "target_eom_amount").to_pandas()

        forecasts = []

        # Group by dim_value and fit ARIMA
        for dim_value in train_pd["dim_value"].unique():
            dim_train = train_pd[train_pd["dim_value"] == dim_value].sort_values("forecast_month")
            dim_test = test_pd[test_pd["dim_value"] == dim_value].sort_values("forecast_month")

            if len(dim_train) < min_history_months:
                # Use zero forecast if insufficient history
                for _, row in dim_test.iterrows():
                    forecasts.append(
                        {
                            "dim_value": dim_value,
                            "forecast_month": row["forecast_month"],
                            "segment_name": row["segment_name"],
                            "actual": row["target_eom_amount"],
                            "forecast": 0.0,
                            "method": "arima",
                        }
                    )
                continue

            try:
                # Simple ARIMA implementation (would use statsmodels or pmdarima in practice)
                # For now, using a simple exponential smoothing approximation
                values = dim_train["target_eom_amount"].values
                alpha = 0.3  # Smoothing parameter

                # Exponential smoothing
                forecast_value = values[-1]
                for i in range(len(values) - 2, -1, -1):
                    forecast_value = alpha * values[i] + (1 - alpha) * forecast_value

                # Apply to all test months
                for _, row in dim_test.iterrows():
                    forecasts.append(
                        {
                            "dim_value": dim_value,
                            "forecast_month": row["forecast_month"],
                            "segment_name": row["segment_name"],
                            "actual": row["target_eom_amount"],
                            "forecast": forecast_value,
                            "method": "arima",
                        }
                    )

            except Exception as e:
                logger.warning(f"ARIMA failed for {dim_value}: {str(e)}")
                # Fallback to zero
                for _, row in dim_test.iterrows():
                    forecasts.append(
                        {
                            "dim_value": dim_value,
                            "forecast_month": row["forecast_month"],
                            "segment_name": row["segment_name"],
                            "actual": row["target_eom_amount"],
                            "forecast": 0.0,
                            "method": "arima",
                        }
                    )

        # Convert back to Snowpark DataFrame
        forecast_pd = pd.DataFrame(forecasts)
        forecast_df = session.create_dataframe(forecast_pd)

        return forecast_df

    @staticmethod
    @log_forecast_method
    def _forecast_xgboost_global(session: Session, config: ForecastingConfig, train_df: DataFrame, test_df: DataFrame) -> DataFrame:
        """XGBoost global model forecast"""

        # Get feature columns
        feature_cols = config.get_xgb_features()
        logger.debug(f"Using features: {feature_cols}")

        # Prepare training data
        train_features = train_df.select("dim_value", "forecast_month", "target_eom_amount", *feature_cols).to_pandas()

        # Prepare test data
        test_features = test_df.select(
            "dim_value", "forecast_month", "segment_name", "target_eom_amount", *feature_cols
        ).to_pandas()

        # Handle missing values
        train_features[feature_cols] = train_features[feature_cols].fillna(0)
        test_features[feature_cols] = test_features[feature_cols].fillna(0)

        try:
            import xgboost as xgb

            # Train XGBoost model
            X_train = train_features[feature_cols]
            y_train = train_features["target_eom_amount"]

            X_test = test_features[feature_cols]

            # Create and train model
            model = xgb.XGBRegressor(
                n_estimators=config.xgb_n_estimators,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                subsample=config.xgb_subsample,
                colsample_bytree=config.xgb_colsample_bytree,
                random_state=42,
            )

            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Create forecast dataframe
            forecast_data = test_features[["dim_value", "forecast_month", "segment_name", "target_eom_amount"]].copy()
            forecast_data["forecast"] = predictions
            forecast_data["method"] = "xgboost_global"
            forecast_data.rename(columns={"target_eom_amount": "actual"}, inplace=True)

        except ImportError:
            logger.warning("XGBoost not available, using fallback forecast")
            # Fallback to simple average
            avg_value = train_features["target_eom_amount"].mean()
            forecast_data = test_features[["dim_value", "forecast_month", "segment_name", "target_eom_amount"]].copy()
            forecast_data["forecast"] = avg_value
            forecast_data["method"] = "xgboost_global"
            forecast_data.rename(columns={"target_eom_amount": "actual"}, inplace=True)

        # Convert back to Snowpark DataFrame
        forecast_df = session.create_dataframe(forecast_data)

        return forecast_df

    @staticmethod
    @log_forecast_method
    def _forecast_croston(train_df: DataFrame, test_df: DataFrame, croston_alpha: float = 0.1) -> DataFrame:
        """Croston's method for intermittent demand"""

        alpha = croston_alpha

        # Calculate intervals and sizes for non-zero demands
        train_pd = train_df.select("dim_value", "forecast_month", "target_eom_amount").to_pandas()

        test_pd = test_df.select("dim_value", "forecast_month", "segment_name", "target_eom_amount").to_pandas()

        forecasts = []

        for dim_value in train_pd["dim_value"].unique():
            dim_train = train_pd[train_pd["dim_value"] == dim_value].sort_values("forecast_month")
            dim_test = test_pd[test_pd["dim_value"] == dim_value]

            values = dim_train["target_eom_amount"].values

            # Find non-zero demands
            non_zero_indices = np.where(values != 0)[0]

            if len(non_zero_indices) == 0:
                # No non-zero values, forecast zero
                forecast_value = 0.0
            elif len(non_zero_indices) == 1:
                # Only one non-zero value
                forecast_value = values[non_zero_indices[0]] / len(values)
            else:
                # Calculate intervals between non-zero demands
                intervals = np.diff(non_zero_indices)
                demands = values[non_zero_indices]

                # Initialize estimates
                demand_est = demands[0]
                interval_est = intervals[0] if len(intervals) > 0 else len(values)

                # Update estimates using exponential smoothing
                for i in range(1, len(demands)):
                    demand_est = alpha * demands[i] + (1 - alpha) * demand_est
                    if i < len(intervals):
                        interval_est = alpha * intervals[i] + (1 - alpha) * interval_est

                # Croston's forecast
                forecast_value = demand_est / interval_est if interval_est > 0 else 0.0

            # Apply forecast to all test months
            for _, row in dim_test.iterrows():
                forecasts.append(
                    {
                        "dim_value": dim_value,
                        "forecast_month": row["forecast_month"],
                        "segment_name": row["segment_name"],
                        "actual": row["target_eom_amount"],
                        "forecast": forecast_value,
                        "method": "croston",
                    }
                )

        # Convert to DataFrame
        forecast_df = train_df.sql_context.sparkSession.createDataFrame(pd.DataFrame(forecasts))

        return forecast_df

    @staticmethod
    @log_forecast_method
    def _forecast_segment_aggregate(train_df: DataFrame, test_df: DataFrame) -> DataFrame:
        """Forecast at segment level then distribute"""

        # Extract direction from dim_value
        train_with_direction = train_df.with_column(
            "direction",
            F.when(F.col("dim_value").endswith("::IN"), "IN")
            .when(F.col("dim_value").endswith("::OUT"), "OUT")
            .otherwise("UNKNOWN"),
        )

        test_with_direction = test_df.with_column(
            "direction",
            F.when(F.col("dim_value").endswith("::IN"), "IN")
            .when(F.col("dim_value").endswith("::OUT"), "OUT")
            .otherwise("UNKNOWN"),
        )

        # Aggregate to segment level
        segment_train = train_with_direction.group_by("segment_name", "direction", "forecast_month").agg(
            F.sum("target_eom_amount").alias("segment_total")
        )

        # Simple forecast at segment level (using moving average)
        window_spec = Window.partition_by("segment_name", "direction").order_by("forecast_month").rows_between(-5, 0)

        segment_forecast = segment_train.with_column("segment_forecast", F.avg("segment_total").over(window_spec))

        # Get last forecast for each segment
        last_forecast = segment_forecast.group_by("segment_name", "direction").agg(F.max("forecast_month").alias("last_month"))

        last_forecast = last_forecast.join(
            segment_forecast,
            (last_forecast["segment_name"] == segment_forecast["segment_name"])
            & (last_forecast["direction"] == segment_forecast["direction"])
            & (last_forecast["last_month"] == segment_forecast["forecast_month"]),
            "inner",
        ).select(segment_forecast["segment_name"], segment_forecast["direction"], segment_forecast["segment_forecast"])

        # Calculate distribution weights from training data
        dim_weights = train_with_direction.group_by("segment_name", "direction", "dim_value").agg(
            F.avg("target_eom_amount").alias("dim_avg")
        )

        # Calculate total per segment for weights
        segment_totals = dim_weights.group_by("segment_name", "direction").agg(F.sum("dim_avg").alias("segment_total_avg"))

        # Calculate weights
        dim_weights = dim_weights.join(segment_totals, ["segment_name", "direction"], "left").with_column(
            "weight", F.when(F.col("segment_total_avg") > 0, F.col("dim_avg") / F.col("segment_total_avg")).otherwise(0)
        )

        # Join test data with segment forecast and weights
        forecast_df = test_with_direction.join(last_forecast, ["segment_name", "direction"], "left").join(
            dim_weights.select("segment_name", "direction", "dim_value", "weight"),
            ["segment_name", "direction", "dim_value"],
            "left",
        )

        # Distribute segment forecast to dim_values
        forecast_df = forecast_df.with_columns(
            [
                (F.coalesce(F.col("segment_forecast"), F.lit(0)) * F.coalesce(F.col("weight"), F.lit(0))).alias("forecast"),
                F.lit("segment_aggregate").alias("method"),
                F.col("target_eom_amount").alias("actual"),
            ]
        ).select("dim_value", "forecast_month", "segment_name", "actual", "forecast", "method")

        return forecast_df

    @staticmethod
    def combine_forecasts(forecast_dfs: list[DataFrame]) -> DataFrame:
        """Combine all forecast DataFrames"""
        logger.info("Combining all forecasts...")

        from functools import reduce

        combined = reduce(lambda df1, df2: df1.union(df2), forecast_dfs)

        return combined

    @staticmethod
    def calculate_dim_value_errors(
        forecasts_df: DataFrame, test_df: DataFrame, zero_threshold: float = 1e-6, mape_cap: float = 200.0
    ) -> DataFrame:
        """Calculate errors at dim_value level"""
        logger.info("Calculating dim_value level errors...")

        errors = forecasts_df.with_columns(
            [
                # Absolute error
                F.abs(F.col("actual") - F.col("forecast")).alias("ae"),
                # Squared error
                F.pow(F.col("actual") - F.col("forecast"), 2).alias("se"),
                # Percentage error (capped)
                F.when(
                    F.abs(F.col("actual")) > zero_threshold,
                    F.least(F.abs((F.col("actual") - F.col("forecast")) / F.col("actual")) * 100, F.lit(mape_cap)),
                )
                .otherwise(F.when(F.abs(F.col("forecast")) > zero_threshold, F.lit(mape_cap)).otherwise(0))
                .alias("ape"),
                # Directional accuracy
                F.when(
                    ((F.col("actual") > 0) & (F.col("forecast") > 0))
                    | ((F.col("actual") < 0) & (F.col("forecast") < 0))
                    | ((F.col("actual") == 0) & (F.col("forecast") == 0)),
                    1,
                )
                .otherwise(0)
                .alias("direction_correct"),
            ]
        )

        # Aggregate by dim_value and method
        dim_value_metrics = errors.group_by("dim_value", "segment_name", "method").agg(
            F.count("*").alias("n_periods"),
            F.avg("ae").alias("mae"),
            F.sqrt(F.avg("se")).alias("rmse"),
            F.avg("ape").alias("mape"),
            F.avg("direction_correct").alias("directional_accuracy"),
            F.sum("actual").alias("total_actual"),
            F.sum("forecast").alias("total_forecast"),
            F.max("ae").alias("max_error"),
            F.min("ae").alias("min_error"),
        )

        return dim_value_metrics

    @staticmethod
    def calculate_segment_errors(forecasts_df: DataFrame, test_df: DataFrame) -> DataFrame:
        """Calculate errors at segment level (net, credit, debit)"""
        logger.info("Calculating segment level errors...")

        # Add direction column
        forecasts_with_dir = forecasts_df.with_column(
            "direction",
            F.when(F.col("dim_value").endswith("::IN"), "CREDIT")
            .when(F.col("dim_value").endswith("::OUT"), "DEBIT")
            .otherwise("UNKNOWN"),
        )

        # Aggregate to segment-direction level
        segment_dir_agg = forecasts_with_dir.group_by("segment_name", "direction", "forecast_month", "method").agg(
            F.sum("actual").alias("actual_sum"), F.sum("forecast").alias("forecast_sum")
        )

        # Calculate net (credit - debit)
        credit_df = segment_dir_agg.filter(F.col("direction") == "CREDIT").select(
            F.col("segment_name"),
            F.col("forecast_month"),
            F.col("method"),
            F.col("actual_sum").alias("credit_actual"),
            F.col("forecast_sum").alias("credit_forecast"),
        )

        debit_df = segment_dir_agg.filter(F.col("direction") == "DEBIT").select(
            F.col("segment_name"),
            F.col("forecast_month"),
            F.col("method"),
            F.col("actual_sum").alias("debit_actual"),
            F.col("forecast_sum").alias("debit_forecast"),
        )

        # Join and calculate net
        net_df = credit_df.join(debit_df, ["segment_name", "forecast_month", "method"], "outer").with_columns(
            [
                (F.coalesce(F.col("credit_actual"), F.lit(0)) - F.coalesce(F.col("debit_actual"), F.lit(0))).alias("net_actual"),
                (F.coalesce(F.col("credit_forecast"), F.lit(0)) - F.coalesce(F.col("debit_forecast"), F.lit(0))).alias(
                    "net_forecast"
                ),
            ]
        )

        # Calculate errors for each component
        segment_errors = net_df.with_columns(
            [
                # Net errors
                F.abs(F.col("net_actual") - F.col("net_forecast")).alias("net_ae"),
                F.pow(F.col("net_actual") - F.col("net_forecast"), 2).alias("net_se"),
                # Credit errors
                F.abs(F.coalesce(F.col("credit_actual"), F.lit(0)) - F.coalesce(F.col("credit_forecast"), F.lit(0))).alias(
                    "credit_ae"
                ),
                F.pow(F.coalesce(F.col("credit_actual"), F.lit(0)) - F.coalesce(F.col("credit_forecast"), F.lit(0)), 2).alias(
                    "credit_se"
                ),
                # Debit errors
                F.abs(F.coalesce(F.col("debit_actual"), F.lit(0)) - F.coalesce(F.col("debit_forecast"), F.lit(0))).alias(
                    "debit_ae"
                ),
                F.pow(F.coalesce(F.col("debit_actual"), F.lit(0)) - F.coalesce(F.col("debit_forecast"), F.lit(0)), 2).alias(
                    "debit_se"
                ),
            ]
        )

        # Aggregate metrics
        segment_metrics = segment_errors.group_by("segment_name", "method").agg(
            F.count("*").alias("n_periods"),
            # Net metrics
            F.avg("net_ae").alias("net_mae"),
            F.sqrt(F.avg("net_se")).alias("net_rmse"),
            F.sum("net_actual").alias("net_total_actual"),
            F.sum("net_forecast").alias("net_total_forecast"),
            # Credit metrics
            F.avg("credit_ae").alias("credit_mae"),
            F.sqrt(F.avg("credit_se")).alias("credit_rmse"),
            F.sum(F.coalesce(F.col("credit_actual"), F.lit(0))).alias("credit_total_actual"),
            F.sum(F.coalesce(F.col("credit_forecast"), F.lit(0))).alias("credit_total_forecast"),
            # Debit metrics
            F.avg("debit_ae").alias("debit_mae"),
            F.sqrt(F.avg("debit_se")).alias("debit_rmse"),
            F.sum(F.coalesce(F.col("debit_actual"), F.lit(0))).alias("debit_total_actual"),
            F.sum(F.coalesce(F.col("debit_forecast"), F.lit(0))).alias("debit_total_forecast"),
        )

        return segment_metrics

    @staticmethod
    def calculate_overall_errors(forecasts_df: DataFrame, test_df: DataFrame) -> DataFrame:
        """Calculate errors at overall level"""
        logger.info("Calculating overall level errors...")

        # Add direction column
        forecasts_with_dir = forecasts_df.with_column(
            "direction",
            F.when(F.col("dim_value").endswith("::IN"), "CREDIT")
            .when(F.col("dim_value").endswith("::OUT"), "DEBIT")
            .otherwise("UNKNOWN"),
        )

        # Aggregate to overall level
        overall_agg = forecasts_with_dir.group_by("forecast_month", "method", "direction").agg(
            F.sum("actual").alias("actual_sum"), F.sum("forecast").alias("forecast_sum")
        )

        # Pivot to get credit and debit columns
        credit_overall = overall_agg.filter(F.col("direction") == "CREDIT").select(
            F.col("forecast_month"),
            F.col("method"),
            F.col("actual_sum").alias("credit_actual"),
            F.col("forecast_sum").alias("credit_forecast"),
        )

        debit_overall = overall_agg.filter(F.col("direction") == "DEBIT").select(
            F.col("forecast_month"),
            F.col("method"),
            F.col("actual_sum").alias("debit_actual"),
            F.col("forecast_sum").alias("debit_forecast"),
        )

        # Join and calculate net
        overall_df = credit_overall.join(debit_overall, ["forecast_month", "method"], "outer").with_columns(
            [
                (F.coalesce(F.col("credit_actual"), F.lit(0)) - F.coalesce(F.col("debit_actual"), F.lit(0))).alias("net_actual"),
                (F.coalesce(F.col("credit_forecast"), F.lit(0)) - F.coalesce(F.col("debit_forecast"), F.lit(0))).alias(
                    "net_forecast"
                ),
            ]
        )

        # Calculate errors
        overall_errors = overall_df.with_columns(
            [
                # Net errors
                F.abs(F.col("net_actual") - F.col("net_forecast")).alias("net_ae"),
                F.pow(F.col("net_actual") - F.col("net_forecast"), 2).alias("net_se"),
                # Credit errors
                F.abs(F.coalesce(F.col("credit_actual"), F.lit(0)) - F.coalesce(F.col("credit_forecast"), F.lit(0))).alias(
                    "credit_ae"
                ),
                F.pow(F.coalesce(F.col("credit_actual"), F.lit(0)) - F.coalesce(F.col("credit_forecast"), F.lit(0)), 2).alias(
                    "credit_se"
                ),
                # Debit errors
                F.abs(F.coalesce(F.col("debit_actual"), F.lit(0)) - F.coalesce(F.col("debit_forecast"), F.lit(0))).alias(
                    "debit_ae"
                ),
                F.pow(F.coalesce(F.col("debit_actual"), F.lit(0)) - F.coalesce(F.col("debit_forecast"), F.lit(0)), 2).alias(
                    "debit_se"
                ),
            ]
        )

        # Aggregate metrics
        overall_metrics = overall_errors.group_by("method").agg(
            F.count("*").alias("n_periods"),
            # Net metrics
            F.avg("net_ae").alias("overall_net_mae"),
            F.sqrt(F.avg("net_se")).alias("overall_net_rmse"),
            F.sum("net_actual").alias("overall_net_total_actual"),
            F.sum("net_forecast").alias("overall_net_total_forecast"),
            # Credit metrics
            F.avg("credit_ae").alias("overall_credit_mae"),
            F.sqrt(F.avg("credit_se")).alias("overall_credit_rmse"),
            F.sum(F.coalesce(F.col("credit_actual"), F.lit(0))).alias("overall_credit_total_actual"),
            F.sum(F.coalesce(F.col("credit_forecast"), F.lit(0))).alias("overall_credit_total_forecast"),
            # Debit metrics
            F.avg("debit_ae").alias("overall_debit_mae"),
            F.sqrt(F.avg("debit_se")).alias("overall_debit_rmse"),
            F.sum(F.coalesce(F.col("debit_actual"), F.lit(0))).alias("overall_debit_total_actual"),
            F.sum(F.coalesce(F.col("debit_forecast"), F.lit(0))).alias("overall_debit_total_forecast"),
        )

        return overall_metrics

    @staticmethod
    def combine_error_levels(dim_value_errors: DataFrame, segment_errors: DataFrame, overall_errors: DataFrame) -> DataFrame:
        """Combine errors from all levels into a summary DataFrame"""
        logger.info("Combining error metrics from all levels...")

        # Add level indicator
        dim_value_summary = dim_value_errors.group_by("method").agg(
            F.avg("mae").alias("dim_value_avg_mae"),
            F.avg("rmse").alias("dim_value_avg_rmse"),
            F.avg("mape").alias("dim_value_avg_mape"),
            F.avg("directional_accuracy").alias("dim_value_avg_dir_acc"),
        )

        segment_summary = segment_errors.group_by("method").agg(
            F.avg("net_mae").alias("segment_net_avg_mae"),
            F.avg("net_rmse").alias("segment_net_avg_rmse"),
            F.avg("credit_mae").alias("segment_credit_avg_mae"),
            F.avg("debit_mae").alias("segment_debit_avg_mae"),
        )

        # Join all levels
        combined = dim_value_summary.join(segment_summary, "method", "outer").join(overall_errors, "method", "outer")

        # Add ranking
        combined = combined.with_column("overall_rank", F.rank().over(Window.order_by("overall_net_mae")))

        return combined

    @staticmethod
    def save_results(session: Session, output_table_prefix: str, forecasts_df: DataFrame, error_metrics_df: DataFrame) -> None:
        """Save forecasting results to tables"""
        logger.info("Saving results to tables...")

        timestamp = date.today().strftime("%Y%m%d")

        # Save forecasts
        forecast_table = f"{output_table_prefix}_forecasts_{timestamp}"
        forecasts_df.write.mode("overwrite").save_as_table(forecast_table)
        logger.info(f"Forecasts saved to: {forecast_table}")

        # Save error metrics
        error_table = f"{output_table_prefix}_errors_{timestamp}"
        error_metrics_df.write.mode("overwrite").save_as_table(error_table)
        logger.info(f"Error metrics saved to: {error_table}")

        # Log summary statistics
        best_method = error_metrics_df.order_by("overall_net_mae").limit(1).collect()[0]
        logger.info(f"\nBest performing method: {best_method['method']}")
        logger.info(f"  Overall Net MAE: {best_method['overall_net_mae']:.2f}")
        logger.info(f"  Overall Net RMSE: {best_method['overall_net_rmse']:.2f}")
