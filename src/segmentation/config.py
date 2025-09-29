"""
Configuration module for EOM Forecasting Segmentation Pipeline
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import date


class SegmentationConfig(BaseModel):
    """Configuration for the segmentation pipeline"""

    # Basic configuration
    min_months_history: int = Field(default=3, description="Minimum months of history required")
    rolling_window_months: int = Field(default=12, description="Rolling window for feature calculation")
    min_transactions: int = Field(default=6, description="Minimum non-zero transactions to include series")
    start_date: date = Field(default=date(2022, 1, 1), description="Analysis start date")
    end_date: Optional[date] = Field(default=None, description="Analysis end date (None = current date)")

    # Window sizes for rolling calculations
    ma_window_short: int = Field(default=3, description="Short-term moving average window (months)")
    pre_eom_signal_window: int = Field(default=6, description="Pre-EOM signal rolling window (months)")
    pre_eom_days: int = Field(default=5, description="Days before EOM to consider for pre-EOM signals")

    # Calendar day thresholds
    early_month_days: int = Field(default=10, description="First N days of month for early month signal")
    mid_month_end_day: int = Field(default=20, description="End day for mid-month period (10-20)")

    # Overall importance thresholds (based on total volume)
    critical_volume_threshold: float = Field(default=100_000_000_000, description="Critical overall volume (12 months)")
    high_volume_threshold: float = Field(default=5_000_000_000, description="High overall volume (12 months)")
    medium_volume_threshold: float = Field(default=1_000_000_000, description="Medium overall volume (12 months)")

    # Monthly average thresholds (derived from annual)
    critical_monthly_avg_threshold: float = Field(default=1_000_000_000, description="Critical monthly average")
    high_monthly_avg_threshold: float = Field(default=500_000_000, description="High monthly average")
    medium_monthly_avg_threshold: float = Field(default=100_000_000, description="Medium monthly average")

    # Max single transaction thresholds
    critical_max_transaction_threshold: float = Field(default=50_000_000, description="Critical single transaction")
    high_max_transaction_threshold: float = Field(default=10_000_000, description="High single transaction")
    medium_max_transaction_threshold: float = Field(default=5_000_000, description="Medium single transaction")

    # EOM importance thresholds (based on EOM-specific volume)
    critical_eom_volume_threshold: float = Field(default=50_000_000_000, description="Critical EOM volume (12 months)")
    high_eom_volume_threshold: float = Field(default=50_000_000_000, description="High EOM volume (12 months)")
    medium_eom_volume_threshold: float = Field(default=50_000_000_000, description="Medium EOM volume (12 months)")

    # EOM monthly average thresholds
    critical_eom_monthly_threshold: float = Field(default=50_000_000_000, description="Critical EOM monthly average")
    high_eom_monthly_threshold: float = Field(default=50_000_000_000, description="High EOM monthly average")
    medium_eom_monthly_threshold: float = Field(default=50_000_000_000, description="Medium EOM monthly average")

    # Max single EOM transaction thresholds
    critical_max_eom_threshold: float = Field(default=100_000_000, description="Critical single EOM transaction")
    high_max_eom_threshold: float = Field(default=50_000_000, description="High single EOM transaction")
    medium_max_eom_threshold: float = Field(default=10_000_000, description="Medium single EOM transaction")

    # Portfolio percentile thresholds for OVERALL importance
    overall_critical_percentile: float = Field(default=0.2, description="Top percentile for overall critical")
    overall_high_percentile: float = Field(default=0.4, description="Top percentile for overall high")
    overall_medium_percentile: float = Field(default=0.8, description="Top percentile for overall medium")

    # Portfolio percentile thresholds for EOM importance
    eom_critical_percentile: float = Field(default=0.3, description="Top percentile for EOM critical")
    eom_high_percentile: float = Field(default=0.6, description="Top percentile for EOM high")
    eom_medium_percentile: float = Field(default=0.95, description="Top percentile for EOM medium")

    # EOM-specific pattern thresholds
    eom_high_concentration_threshold: float = Field(default=0.70, description="High EOM concentration")
    eom_medium_concentration_threshold: float = Field(default=0.40, description="Medium EOM concentration")
    eom_high_predictability_threshold: float = Field(default=0.70, description="High EOM predictability")
    eom_medium_predictability_threshold: float = Field(default=0.40, description="Medium EOM predictability")
    eom_high_frequency_threshold: float = Field(default=0.50, description="High EOM frequency")
    eom_intermittent_threshold: float = Field(default=0.30, description="Intermittent EOM threshold")

    # General timeseries pattern thresholds
    ts_high_volatility_threshold: float = Field(default=0.50, description="High volatility (CV)")
    ts_medium_volatility_threshold: float = Field(default=0.25, description="Medium volatility (CV)")
    ts_high_regularity_threshold: float = Field(default=0.70, description="High transaction regularity")
    ts_medium_regularity_threshold: float = Field(default=0.40, description="Medium transaction regularity")
    ts_intermittent_threshold: float = Field(default=0.30, description="Intermittent activity threshold")
    ts_seasonal_concentration_threshold: float = Field(default=0.25, description="Quarterly concentration for seasonality")

    # Other thresholds
    inactive_months: int = Field(default=3, description="Months of inactivity for INACTIVE")
    emerging_months: int = Field(default=3, description="Max months for EMERGING")
    eom_risk_volume_threshold: float = Field(default=100_000, description="Volume threshold for EOM risk flag")
    eom_risk_min_months: int = Field(default=6, description="Min months history for risk assessment")

    # Trend analysis thresholds
    trend_analysis_months: int = Field(default=6, description="Months required for trend analysis")
    trend_window_months: int = Field(default=5, description="Window for trend calculation (preceding months)")
    growth_threshold: float = Field(default=0.1, description="Growth rate threshold for trend classification")
    yoy_comparison_months: int = Field(default=12, description="Months for year-over-year comparison")

    # Clipping thresholds
    daily_amount_clip_threshold: Optional[float] = Field(
        default=None, description="Threshold below which daily amounts are clipped to 0 (None = no clipping)"
    )
    clip_analysis_enabled: bool = Field(default=True, description="Whether to perform detailed analysis of clipped values")

    # Smooth scoring parameters
    sigmoid_steepness: float = Field(default=10.0, description="Steepness of sigmoid functions for smooth scoring")
    stability_decay_rate: float = Field(default=2.0, description="Decay rate for stability score")
    recency_decay_rate: float = Field(default=0.8, description="Decay rate for recency score")
    concentration_steepness: float = Field(default=5.0, description="Steepness for concentration score")
    volume_growth_rate: float = Field(default=5.0, description="Growth rate for volume importance score")

    # Pattern distance calculation
    pattern_temperature: float = Field(default=20.0, description="Temperature for softmax in pattern classification")

    # Filtering options
    target_forecast_month: Optional[date] = Field(default=None, description="Specific month to forecast (for filtering)")
    filter_low_importance: bool = Field(default=False, description="Filter out LOW/NONE importance tiers")

    # Data source
    source_table: str = Field(
        default="int__t__cad_core_banking_regular_time_series_recorded", description="Source table for time series data"
    )

    class Config:
        """Pydantic config"""

        validate_assignment = True
        use_enum_values = True

    def get_snowflake_date_str(self, date_field: Optional[date]) -> str:
        """Convert date to Snowflake-compatible string"""
        if date_field is None:
            return "CURRENT_DATE"
        return f"'{date_field.isoformat()}'::DATE"
