"""Configuration parameters for EOM forecasting transformations."""

from datetime import date
from pydantic import BaseModel


class Config(BaseModel):
    """Configuration parameters for EOM forecasting feature engineering."""

    # Basic configuration
    min_months_history: int = 3
    rolling_window_months: int = 12
    min_transactions: int = 6
    start_date: date = date(2022, 1, 1)
    end_date: date = date.today()

    # Window sizes
    ma_window_short: int = 3
    pre_eom_signal_window: int = 6
    pre_eom_days: int = 5

    # Calendar thresholds
    early_month_days: int = 10
    mid_month_end_day: int = 20

    # Overall importance thresholds
    critical_volume_threshold: float = 100_000_000_000
    high_volume_threshold: float = 5_000_000_000
    medium_volume_threshold: float = 1_000_000_000

    critical_monthly_avg_threshold: float = 1_000_000_000
    high_monthly_avg_threshold: float = 500_000_000
    medium_monthly_avg_threshold: float = 100_000_000

    critical_max_transaction_threshold: float = 50_000_000
    high_max_transaction_threshold: float = 10_000_000
    medium_max_transaction_threshold: float = 5_000_000

    # EOM importance thresholds
    critical_eom_volume_threshold: float = 50_000_000_000
    high_eom_volume_threshold: float = 50_000_000_000
    medium_eom_volume_threshold: float = 50_000_000_000

    critical_eom_monthly_threshold: float = 50_000_000_000
    high_eom_monthly_threshold: float = 50_000_000_000
    medium_eom_monthly_threshold: float = 50_000_000_000

    critical_max_eom_threshold: float = 100_000_000
    high_max_eom_threshold: float = 50_000_000
    medium_max_eom_threshold: float = 10_000_000

    # Portfolio percentile thresholds
    overall_critical_percentile: float = 0.2
    overall_high_percentile: float = 0.4
    overall_medium_percentile: float = 0.8

    eom_critical_percentile: float = 0.3
    eom_high_percentile: float = 0.6
    eom_medium_percentile: float = 0.95

    # Pattern thresholds
    eom_high_concentration_threshold: float = 0.70
    eom_medium_concentration_threshold: float = 0.40
    eom_high_predictability_threshold: float = 0.70
    eom_medium_predictability_threshold: float = 0.40
    eom_high_frequency_threshold: float = 0.50
    eom_intermittent_threshold: float = 0.30

    ts_high_volatility_threshold: float = 0.50
    ts_medium_volatility_threshold: float = 0.25
    ts_high_regularity_threshold: float = 0.70
    ts_medium_regularity_threshold: float = 0.40
    ts_intermittent_threshold: float = 0.30
    ts_seasonal_concentration_threshold: float = 0.25

    # Other thresholds
    inactive_months: int = 3
    emerging_months: int = 3
    eom_risk_volume_threshold: float = 100_000
    eom_risk_min_months: int = 6

    # Trend analysis
    trend_analysis_months: int = 6
    trend_window_months: int = 5
    growth_threshold: float = 0.1
    yoy_comparison_months: int = 12
