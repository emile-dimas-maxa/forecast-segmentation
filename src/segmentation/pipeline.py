"""
EOM Forecasting Segmentation Pipeline
Modular transformation functions for Snowpark DataFrames
"""

from loguru import logger
from snowflake.snowpark import DataFrame, Session

from src.segmentation.transformation.aggregation import create_monthly_aggregates
from src.segmentation.transformation.data_preparation import load_source_data, prepare_base_data
from src.segmentation.transformation.eom_clipping import apply_eom_clipping
from src.segmentation.transformation.eom_pattern_classification import (
    ArchetypeConfig,
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
        # Pattern archetype configuration
        archetype_config: ArchetypeConfig | None = None,
        # EOM high risk flag thresholds
        eom_high_risk_stability_threshold: float = 30.0,
        eom_high_risk_concentration_threshold: float = 50.0,
        # Output filtering
        target_forecast_month: str | None = None,
        filter_low_importance: bool = False,
    ) -> DataFrame:
        """
        Run the complete EOM forecasting segmentation pipeline

        This pipeline processes time series data through multiple transformation steps:
        1. Data preparation and filtering
        2. Monthly aggregations with EOM/non-EOM breakdown
        3. EOM clipping (noise reduction)
        4. Rolling window feature calculations
        5. Portfolio-level metrics and percentiles
        6. Pattern metrics calculation
        7. Dual importance classification (overall + EOM)
        8. EOM pattern classification with smooth scoring
        9. General timeseries pattern classification
        10. Final classification and recommendations
        11. Growth metrics calculation
        12. Output formatting and filtering

        Args:
            session: Snowpark session for database operations
            source_df: Optional source DataFrame (if not provided, reads from source_table)
            source_table: Source table name (required if source_df not provided)

            # Basic Configuration
            start_date: Analysis start date (YYYY-MM-DD format)
            end_date: Analysis end date (None = current date)
            min_months_history: Minimum months of history required for analysis
            rolling_window_months: Rolling window size for feature calculations
            min_transactions: Minimum non-zero transactions to include a series

            # Window Sizes
            ma_window_short: Short-term moving average window (months)
            pre_eom_signal_window: Pre-EOM signal rolling window (months)
            pre_eom_days: Days before EOM to consider for pre-EOM signals

            # Calendar Thresholds
            early_month_days: First N days of month for early month signal
            mid_month_end_day: End day for mid-month period

            # Overall Importance Thresholds
            critical_volume_threshold: Critical overall volume threshold (12 months)
            high_volume_threshold: High overall volume threshold (12 months)
            medium_volume_threshold: Medium overall volume threshold (12 months)
            critical_monthly_avg_threshold: Critical monthly average threshold
            high_monthly_avg_threshold: High monthly average threshold
            medium_monthly_avg_threshold: Medium monthly average threshold
            critical_max_transaction_threshold: Critical single transaction threshold
            high_max_transaction_threshold: High single transaction threshold
            medium_max_transaction_threshold: Medium single transaction threshold

            # EOM Importance Thresholds
            critical_eom_volume_threshold: Critical EOM volume threshold (12 months)
            high_eom_volume_threshold: High EOM volume threshold (12 months)
            medium_eom_volume_threshold: Medium EOM volume threshold (12 months)
            critical_eom_monthly_threshold: Critical EOM monthly average threshold
            high_eom_monthly_threshold: High EOM monthly average threshold
            medium_eom_monthly_threshold: Medium EOM monthly average threshold
            critical_max_eom_threshold: Critical single EOM transaction threshold
            high_max_eom_threshold: High single EOM transaction threshold
            medium_max_eom_threshold: Medium single EOM transaction threshold

            # Portfolio Percentile Thresholds
            overall_critical_percentile: Top percentile for overall critical (0.2 = top 20%)
            overall_high_percentile: Top percentile for overall high (0.4 = top 40%)
            overall_medium_percentile: Top percentile for overall medium (0.8 = top 80%)
            eom_critical_percentile: Top percentile for EOM critical (0.3 = top 30%)
            eom_high_percentile: Top percentile for EOM high (0.6 = top 60%)
            eom_medium_percentile: Top percentile for EOM medium (0.95 = top 95%)

            # EOM Pattern Thresholds
            eom_concentration_threshold: EOM concentration threshold for pattern analysis
            eom_predictability_threshold: EOM predictability threshold
            eom_frequency_threshold: EOM frequency threshold
            eom_zero_ratio_threshold: EOM zero ratio threshold
            eom_cv_threshold: EOM coefficient of variation threshold
            monthly_cv_threshold: Monthly CV threshold
            transaction_regularity_threshold: Transaction regularity threshold
            activity_rate_threshold: Activity rate threshold

            # General Timeseries Pattern Thresholds
            ts_high_volatility_threshold: High volatility (CV) threshold
            ts_medium_volatility_threshold: Medium volatility (CV) threshold
            ts_high_regularity_threshold: High transaction regularity threshold
            ts_medium_regularity_threshold: Medium transaction regularity threshold
            ts_intermittent_threshold: Intermittent activity threshold
            ts_seasonal_concentration_threshold: Quarterly concentration for seasonality
            ts_year_end_concentration_threshold: Year-end concentration for seasonality

            # Other Classification Thresholds
            inactive_months: Months of inactivity for INACTIVE classification
            emerging_months: Max months for EMERGING classification
            eom_risk_volume_threshold: Volume threshold for EOM risk flag
            eom_risk_min_months: Min months history for risk assessment

            # EOM Clipping (Python-specific feature)
            daily_amount_clip_threshold: Threshold for clipping small EOM amounts to zero

            # Pattern Archetype Configuration
            archetype_config: Configuration for pattern archetype centroids and weights
            eom_high_risk_stability_threshold: Stability threshold for high risk flag
            eom_high_risk_concentration_threshold: Concentration threshold for high risk flag

            # Output Filtering
            target_forecast_month: Target forecast month to filter results to
            filter_low_importance: Whether to filter out low importance tiers

        Returns:
            DataFrame: Final segmented DataFrame with all classifications and metrics

        Raises:
            ValueError: If neither source_df nor source_table is provided

        Example:
            >>> from src.segmentation.pipeline import SegmentationPipeline
            >>> from src.segmentation.transformation.eom_pattern_classification import ArchetypeConfig
            >>>
            >>> # Basic usage with defaults
            >>> result = SegmentationPipeline.run_full_pipeline(
            ...     session=session,
            ...     source_table="my_table",
            ...     start_date="2022-01-01",
            ...     end_date="2024-12-31"
            ... )
            >>>
            >>> # Advanced usage with custom archetype configuration
            >>> custom_config = ArchetypeConfig(
            ...     pattern_temperature=15.0,  # More confident classifications
            ...     continuous_stable=(95, 85, 55)  # Stricter stable pattern
            ... )
            >>> result = SegmentationPipeline.run_full_pipeline(
            ...     session=session,
            ...     source_table="my_table",
            ...     archetype_config=custom_config,
            ...     eom_high_risk_stability_threshold=25.0
            ... )
        """
        # Input validation
        if source_df is None and source_table is None:
            raise ValueError("Either source_df or source_table must be provided")

        if archetype_config is None:
            archetype_config = ArchetypeConfig()

        logger.info("=" * 80)
        logger.info("Starting Full Segmentation Pipeline")
        logger.info(f"Configuration: start_date={start_date}, end_date={end_date}")
        logger.info(f"Archetype Config: temperature={archetype_config.pattern_temperature}")
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
            critical_monthly_avg_threshold=critical_monthly_avg_threshold,
            high_monthly_avg_threshold=high_monthly_avg_threshold,
            medium_monthly_avg_threshold=medium_monthly_avg_threshold,
            critical_max_transaction_threshold=critical_max_transaction_threshold,
            high_max_transaction_threshold=high_max_transaction_threshold,
            medium_max_transaction_threshold=medium_max_transaction_threshold,
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
        df = calculate_eom_smooth_scores(df=df)
        df = calculate_pattern_distances(df=df, archetype_config=archetype_config)
        df = calculate_pattern_probabilities(df=df, archetype_config=archetype_config)
        df = classify_eom_patterns(
            df=df,
            archetype_config=archetype_config,
            eom_high_risk_stability_threshold=eom_high_risk_stability_threshold,
            eom_high_risk_concentration_threshold=eom_high_risk_concentration_threshold,
        )

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
