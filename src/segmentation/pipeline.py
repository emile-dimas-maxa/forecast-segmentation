"""
EOM Forecasting Segmentation Pipeline
Modular transformation functions for Snowpark DataFrames
"""

from loguru import logger
from snowflake.snowpark import DataFrame, Session

from src.segmentation.transformation.aggregation import apply_eom_clipping, create_monthly_aggregates
from src.segmentation.transformation.data_preparation import load_source_data, prepare_base_data
from src.segmentation.transformation.eom_pattern_classification import (
    calculate_eom_smooth_scores,
    calculate_pattern_distances,
    calculate_pattern_probabilities,
    classify_eom_patterns,
)
from src.segmentation.transformation.final_classification import create_final_classification
from src.segmentation.transformation.general_pattern_classification import classify_general_patterns
from src.segmentation.transformation.growth_metrics import calculate_growth_metrics
from src.segmentation.transformation.importance_classification import classify_importance
from src.segmentation.transformation.output_formatting import select_final_columns
from src.segmentation.transformation.pattern_metrics import calculate_pattern_metrics
from src.segmentation.transformation.portfolio_metrics import calculate_portfolio_metrics
from src.segmentation.transformation.rolling_features import calculate_rolling_features


class SegmentationPipeline:
    """Pipeline for EOM forecasting segmentation using static methods"""

    @staticmethod
    def run_full_pipeline(
        session: Session,
        source_df: DataFrame | None = None,
        source_table: str | None = None,
        # Basic configuration
        start_date: str = "2022-01-01",
        end_date: str | None = None,
        min_months_history: int = 3,
        rolling_window_months: int = 12,
        min_transactions: int = 6,
        # Window sizes for rolling calculations
        ma_window_short: int = 3,
        pre_eom_signal_window: int = 6,
        pre_eom_days: int = 5,
        # Calendar day thresholds
        early_month_days: int = 10,
        mid_month_end_day: int = 20,
        # Overall importance thresholds (based on total volume)
        critical_volume_threshold: float = 100_000_000_000,
        high_volume_threshold: float = 5_000_000_000,
        medium_volume_threshold: float = 1_000_000_000,
        # Monthly average thresholds (derived from annual)
        critical_monthly_avg_threshold: float = 1_000_000_000,
        high_monthly_avg_threshold: float = 500_000_000,
        medium_monthly_avg_threshold: float = 100_000_000,
        # Max single transaction thresholds
        critical_max_transaction_threshold: float = 50_000_000,
        high_max_transaction_threshold: float = 10_000_000,
        medium_max_transaction_threshold: float = 5_000_000,
        # EOM importance thresholds (based on EOM-specific volume)
        critical_eom_volume_threshold: float = 50_000_000_000,
        high_eom_volume_threshold: float = 50_000_000_000,
        medium_eom_volume_threshold: float = 50_000_000_000,
        # EOM monthly average thresholds
        critical_eom_monthly_threshold: float = 50_000_000_000,
        high_eom_monthly_threshold: float = 50_000_000_000,
        medium_eom_monthly_threshold: float = 50_000_000_000,
        # Max single EOM transaction thresholds
        critical_max_eom_threshold: float = 100_000_000,
        high_max_eom_threshold: float = 50_000_000,
        medium_max_eom_threshold: float = 10_000_000,
        # Portfolio percentile thresholds for OVERALL importance
        overall_critical_percentile: float = 0.2,
        overall_high_percentile: float = 0.4,
        overall_medium_percentile: float = 0.8,
        # Portfolio percentile thresholds for EOM importance
        eom_critical_percentile: float = 0.3,
        eom_high_percentile: float = 0.6,
        eom_medium_percentile: float = 0.95,
        # EOM-specific pattern thresholds
        eom_concentration_threshold: float = 0.70,
        eom_predictability_threshold: float = 0.60,
        eom_frequency_threshold: float = 0.50,
        eom_zero_ratio_threshold: float = 0.30,
        eom_cv_threshold: float = 1.0,
        monthly_cv_threshold: float = 0.50,
        transaction_regularity_threshold: float = 0.40,
        activity_rate_threshold: float = 0.60,
        # General timeseries pattern thresholds
        ts_high_volatility_threshold: float = 0.50,
        ts_medium_volatility_threshold: float = 0.25,
        ts_high_regularity_threshold: float = 0.70,
        ts_medium_regularity_threshold: float = 0.40,
        ts_intermittent_threshold: float = 0.30,
        ts_seasonal_concentration_threshold: float = 0.25,
        ts_year_end_concentration_threshold: float = 0.5,
        # Other thresholds
        inactive_months: int = 3,
        emerging_months: int = 3,
        eom_risk_volume_threshold: float = 100_000,
        eom_risk_min_months: int = 6,
        # EOM clipping threshold (new feature not in SQL)
        daily_amount_clip_threshold: float = 1_000_000,
        # Output filtering
        target_forecast_month: str | None = None,
        filter_low_importance: bool = False,
    ) -> DataFrame:
        """
        Run the complete segmentation pipeline

        Args:
            session: Snowpark session
            source_df: Optional source DataFrame. If not provided, reads from source_table
            source_table: Source table name (required if source_df not provided)
            start_date: Analysis start date
            end_date: Analysis end date (None = current date)
            min_months_history: Minimum months of history required
            rolling_window_months: Rolling window for feature calculation
            min_transactions: Minimum non-zero transactions to include series
            ma_window_short: Short-term moving average window (months)
            pre_eom_signal_window: Pre-EOM signal rolling window (months)
            pre_eom_days: Days before EOM to consider for pre-EOM signals
            early_month_days: First N days of month for early month signal
            mid_month_end_day: End day for mid-month period
            critical_volume_threshold: Critical overall volume (12 months)
            high_volume_threshold: High overall volume (12 months)
            medium_volume_threshold: Medium overall volume (12 months)
            critical_monthly_avg_threshold: Critical monthly average
            high_monthly_avg_threshold: High monthly average
            medium_monthly_avg_threshold: Medium monthly average
            critical_eom_threshold: Critical EOM volume (12 months)
            high_eom_threshold: High EOM volume (12 months)
            medium_eom_threshold: Medium EOM volume (12 months)
            eom_concentration_threshold: EOM concentration threshold
            eom_predictability_threshold: EOM predictability threshold
            eom_frequency_threshold: EOM frequency threshold
            eom_zero_ratio_threshold: EOM zero ratio threshold
            eom_cv_threshold: EOM coefficient of variation threshold
            monthly_cv_threshold: Monthly CV threshold
            transaction_regularity_threshold: Transaction regularity threshold
            activity_rate_threshold: Activity rate threshold
            daily_amount_clip_threshold: Daily amount clipping threshold
            target_forecast_month: Target forecast month to filter to
            filter_low_importance: Whether to filter out low importance tiers

        Returns:
            Final segmented DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Full Segmentation Pipeline")
        logger.info(f"Configuration: start_date={start_date}, end_date={end_date}")
        logger.info("=" * 80)

        # Step 1: Load and prepare base data
        df = prepare_base_data(
            df=source_df or load_source_data(session, source_table),
            start_date=start_date,
            end_date=end_date,
            min_months_history=min_months_history,
            min_transactions=min_transactions,
        )

        # Step 2: Monthly aggregations
        df = create_monthly_aggregates(
            df=df,
            pre_eom_days=pre_eom_days,
            early_month_days=early_month_days,
            mid_month_end_day=mid_month_end_day,
        )

        # Step 2.5: Apply EOM clipping if configured
        df = apply_eom_clipping(df=df, daily_amount_clip_threshold=daily_amount_clip_threshold)

        # Step 3: Rolling features
        df = calculate_rolling_features(
            df=df,
            rolling_window_months=rolling_window_months,
            ma_window_short=ma_window_short,
        )

        # Step 4: Portfolio metrics
        df = calculate_portfolio_metrics(df=df, min_months_history=min_months_history)

        # Step 5: Pattern metrics
        df = calculate_pattern_metrics(
            df=df,
            pre_eom_signal_window=pre_eom_signal_window,
            pre_eom_days=pre_eom_days,
            early_month_days=early_month_days,
            mid_month_end_day=mid_month_end_day,
        )

        # Step 6: Importance classification
        df = classify_importance(
            df=df,
            critical_volume_threshold=critical_volume_threshold,
            high_volume_threshold=high_volume_threshold,
            medium_volume_threshold=medium_volume_threshold,
            critical_monthly_avg_threshold=critical_monthly_avg_threshold,
            high_monthly_avg_threshold=high_monthly_avg_threshold,
            medium_monthly_avg_threshold=medium_monthly_avg_threshold,
            critical_max_transaction_threshold=critical_max_transaction_threshold,
            high_max_transaction_threshold=high_max_transaction_threshold,
            medium_max_transaction_threshold=medium_max_transaction_threshold,
            critical_eom_volume_threshold=critical_eom_volume_threshold,
            high_eom_volume_threshold=high_eom_volume_threshold,
            medium_eom_volume_threshold=medium_eom_volume_threshold,
            critical_eom_monthly_threshold=critical_eom_monthly_threshold,
            high_eom_monthly_threshold=high_eom_monthly_threshold,
            medium_eom_monthly_threshold=medium_eom_monthly_threshold,
            critical_max_eom_threshold=critical_max_eom_threshold,
            high_max_eom_threshold=high_max_eom_threshold,
            medium_max_eom_threshold=medium_max_eom_threshold,
            overall_critical_percentile=overall_critical_percentile,
            overall_high_percentile=overall_high_percentile,
            overall_medium_percentile=overall_medium_percentile,
            eom_critical_percentile=eom_critical_percentile,
            eom_high_percentile=eom_high_percentile,
            eom_medium_percentile=eom_medium_percentile,
            eom_risk_volume_threshold=eom_risk_volume_threshold,
            eom_risk_min_months=eom_risk_min_months,
        )

        # Step 7: EOM pattern classification (smooth scoring)
        df = calculate_eom_smooth_scores(
            df=df,
            eom_concentration_threshold=eom_concentration_threshold,
            eom_predictability_threshold=eom_predictability_threshold,
            eom_frequency_threshold=eom_frequency_threshold,
            eom_zero_ratio_threshold=eom_zero_ratio_threshold,
            eom_cv_threshold=eom_cv_threshold,
            monthly_cv_threshold=monthly_cv_threshold,
            transaction_regularity_threshold=transaction_regularity_threshold,
            activity_rate_threshold=activity_rate_threshold,
        )
        df = calculate_pattern_distances(df=df)
        df = calculate_pattern_probabilities(df=df)
        df = classify_eom_patterns(df=df)

        # Step 8: General pattern classification
        df = classify_general_patterns(
            df=df,
            inactive_months=inactive_months,
            emerging_months=emerging_months,
            ts_seasonal_concentration_threshold=ts_seasonal_concentration_threshold,
            ts_year_end_concentration_threshold=ts_year_end_concentration_threshold,
            ts_intermittent_threshold=ts_intermittent_threshold,
            ts_high_volatility_threshold=ts_high_volatility_threshold,
            ts_medium_volatility_threshold=ts_medium_volatility_threshold,
            ts_high_regularity_threshold=ts_high_regularity_threshold,
            ts_medium_regularity_threshold=ts_medium_regularity_threshold,
        )

        # Step 9: Final classification and recommendations
        df = create_final_classification(df=df)

        # Step 10: Growth metrics
        df = calculate_growth_metrics(df=df)

        # Final selection and filtering
        df = select_final_columns(
            df=df,
            target_forecast_month=target_forecast_month,
            filter_low_importance=filter_low_importance,
        )

        logger.info("=" * 80)
        logger.info("Completed Full Segmentation Pipeline Successfully")
        logger.info("=" * 80)

        return df
