"""
EOM Forecasting Segmentation Pipeline
Modular transformation functions for Snowpark DataFrames
"""

from loguru import logger
from snowflake.snowpark import DataFrame, Session
from src.segmentation.transformation.data_preparation import load_source_data, prepare_base_data
from src.segmentation.transformation.aggregation import create_monthly_aggregates, apply_eom_clipping
from src.segmentation.transformation.rolling_features import calculate_rolling_features
from src.segmentation.transformation.portfolio_metrics import calculate_portfolio_metrics
from src.segmentation.transformation.pattern_metrics import calculate_pattern_metrics
from src.segmentation.transformation.importance_classification import classify_importance
from src.segmentation.transformation.eom_pattern_classification import (
    calculate_eom_smooth_scores,
    calculate_pattern_distances,
    calculate_pattern_probabilities,
    classify_eom_patterns,
)
from src.segmentation.transformation.general_pattern_classification import classify_general_patterns
from src.segmentation.transformation.final_classification import create_final_classification
from src.segmentation.transformation.growth_metrics import calculate_growth_metrics
from src.segmentation.transformation.output_formatting import select_final_columns


class SegmentationPipeline:
    """Pipeline for EOM forecasting segmentation using static methods"""

    @staticmethod
    def run_full_pipeline(
        session: Session,
        source_df: DataFrame | None = None,
        source_table: str | None = None,
        start_date: str = "2022-01-01",
        end_date: str | None = None,
        min_months_history: int = 3,
        rolling_window_months: int = 12,
        min_transactions: int = 6,
        ma_window_short: int = 3,
        pre_eom_signal_window: int = 6,
        pre_eom_days: int = 5,
        early_month_days: int = 10,
        mid_month_end_day: int = 20,
        critical_volume_threshold: float = 100_000_000_000,
        high_volume_threshold: float = 5_000_000_000,
        medium_volume_threshold: float = 1_000_000_000,
        critical_monthly_avg_threshold: float = 1_000_000_000,
        high_monthly_avg_threshold: float = 416_666_667,
        medium_monthly_avg_threshold: float = 83_333_333,
        critical_eom_threshold: float = 50_000_000_000,
        high_eom_threshold: float = 2_500_000_000,
        medium_eom_threshold: float = 500_000_000,
        eom_concentration_threshold: float = 0.7,
        eom_predictability_threshold: float = 0.6,
        eom_frequency_threshold: float = 0.5,
        eom_zero_ratio_threshold: float = 0.3,
        eom_cv_threshold: float = 1.0,
        monthly_cv_threshold: float = 0.5,
        transaction_regularity_threshold: float = 0.4,
        activity_rate_threshold: float = 0.6,
        daily_amount_clip_threshold: float = 1_000_000,
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
        df = create_monthly_aggregates(df=df)

        # Step 2.5: Apply EOM clipping if configured
        df = apply_eom_clipping(df=df, daily_amount_clip_threshold=daily_amount_clip_threshold)

        # Step 3: Rolling features
        df = calculate_rolling_features(
            df=df,
            rolling_window_months=rolling_window_months,
            ma_window_short=ma_window_short,
        )

        # Step 4: Portfolio metrics
        df = calculate_portfolio_metrics(df=df)

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
            critical_eom_threshold=critical_eom_threshold,
            high_eom_threshold=high_eom_threshold,
            medium_eom_threshold=medium_eom_threshold,
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
        df = classify_general_patterns(df=df)

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
