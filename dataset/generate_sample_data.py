#!/usr/bin/env python3
"""
Sample Data Generator for Radio Forecast Segmentation

This script generates sample data that matches the expected output format of feature_df.csv
from the feature pipeline. The data includes both individual dim_values and aggregated
"others" categories as produced by the create_aggregated_features function.

The generated data simulates:
1. Individual important dim_values with full feature sets
2. Aggregated "others::IN" and "others::OUT" categories with only summed values
3. Realistic patterns for EOM (End of Month) forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any


def generate_sample_feature_data(
    num_individual_series: int = 50, start_date: str = "2023-01-01", end_date: str = "2025-09-01", seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample feature data that matches the expected feature_df.csv format.
    Creates complete time series for each dim_value across multiple months.

    Args:
        num_individual_series: Number of individual series to generate
        start_date: Start date for time series (YYYY-MM-DD format)
        end_date: End date for time series (YYYY-MM-DD format)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample feature data matching the expected schema
    """
    np.random.seed(seed)
    random.seed(seed)

    # Parse dates and generate month range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate list of months
    months = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        months.append(current.strftime("%Y-%m-%d"))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    num_months = len(months)

    # Define importance tiers and patterns
    importance_tiers = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
    eom_patterns = [
        "CONTINUOUS_STABLE",
        "CONTINUOUS_VOLATILE",
        "INTERMITTENT_ACTIVE",
        "INTERMITTENT_DORMANT",
        "RARE_RECENT",
        "RARE_STALE",
    ]
    general_patterns = ["STABLE", "VOLATILE", "INTERMITTENT", "HIGHLY_SEASONAL"]
    recommended_methods = ["ARIMA", "Exponential_Smoothing", "Moving_Average", "Conservative_MA", "Historical_Average"]

    # Generate individual series data
    individual_data = []

    for i in range(num_individual_series):
        # Generate realistic dim_value names
        dim_value = f"series_{i + 1:03d}"

        # Assign importance tiers (weighted towards lower importance for realism)
        overall_tier = np.random.choice(importance_tiers, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        eom_tier = np.random.choice(importance_tiers, p=[0.05, 0.15, 0.25, 0.35, 0.20])

        # Generate base characteristics for this series (consistent across time)
        tier_multipliers = {"CRITICAL": 1000000, "HIGH": 100000, "MEDIUM": 10000, "LOW": 1000, "NONE": 100}
        base_amount = tier_multipliers[overall_tier]

        # Generate EOM pattern and probabilities (consistent for the series)
        eom_pattern = np.random.choice(eom_patterns)
        pattern_confidence = np.random.uniform(0.3, 0.95)

        # Generate pattern probabilities (should sum to ~1.0)
        probs = np.random.dirichlet([1, 1, 1, 1, 1, 1]) * 100
        prob_continuous_stable_pct = probs[0]
        prob_continuous_volatile_pct = probs[1]
        prob_intermittent_active_pct = probs[2]
        prob_intermittent_dormant_pct = probs[3]
        prob_rare_recent_pct = probs[4]
        prob_rare_stale_pct = probs[5]

        # Generate general pattern (consistent for the series)
        general_pattern = np.random.choice(general_patterns)

        # Generate base smooth scores (consistent for the series)
        base_eom_regularity_score = np.random.uniform(0, 100)
        base_eom_stability_score = np.random.uniform(0, 100)
        base_eom_recency_score = np.random.uniform(0, 100)
        base_eom_concentration_score = np.random.uniform(0, 100)
        base_eom_volume_score = np.random.uniform(0, 100)

        # Generate base importance scores (consistent for the series)
        base_overall_importance_score = np.random.uniform(0.1, 1.0)
        base_eom_importance_score = np.random.uniform(0.1, 1.0)

        # Generate base flags (consistent for the series)
        base_eom_risk_flag = np.random.choice([True, False], p=[0.2, 0.8])

        # Generate base activity indicators (consistent for the series)
        base_active_months_12m = np.random.randint(1, 13)
        base_months_of_history = np.random.randint(3, 60)

        # Generate base raw EOM metrics (consistent for the series)
        base_eom_concentration = np.random.uniform(0, 1)
        base_eom_predictability = np.random.uniform(0, 1)
        base_eom_frequency = np.random.uniform(0, 1)
        base_eom_zero_ratio = np.random.uniform(0, 0.8)
        base_eom_cv = np.random.uniform(0, 2)

        # Generate base general timeseries pattern features (consistent for the series)
        base_monthly_cv = np.random.uniform(0, 1)
        base_transaction_regularity = np.random.uniform(0, 1)
        base_activity_rate = np.random.uniform(0, 1)
        base_transaction_dispersion = np.random.uniform(0, 10)
        base_quarter_end_concentration = np.random.uniform(0, 1)
        base_year_end_concentration = np.random.uniform(0, 1)

        # Generate base portfolio percentiles (consistent for the series)
        base_cumulative_overall_portfolio_pct = np.random.uniform(0, 1)
        base_cumulative_eom_portfolio_pct = np.random.uniform(0, 1)

        # Generate base combined metrics (consistent for the series)
        base_combined_priority = np.random.uniform(0, 100)
        base_recommended_method = np.random.choice(recommended_methods)
        base_forecast_complexity = np.random.randint(1, 6)

        # Generate base pattern uncertainty (consistent for the series)
        base_pattern_uncertainty = np.random.uniform(0, 2)

        # Generate time series for this dim_value
        for month_idx, forecast_month in enumerate(months):
            forecast_date = datetime.strptime(forecast_month, "%Y-%m-%d")
            year = forecast_date.year
            month_num = forecast_date.month
            month_of_year = month_num
            is_quarter_end = month_num in [3, 6, 9, 12]
            is_year_end = month_num == 12

            # Generate time-varying target EOM amount with some seasonality and trend
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month_num / 12)  # Seasonal variation
            trend_factor = 1 + 0.1 * (month_idx / num_months)  # Slight upward trend
            noise_factor = np.random.lognormal(0, 0.2)  # Random noise

            target_eom_amount = base_amount * seasonal_factor * trend_factor * noise_factor

            # Generate time-varying volume metrics
            total_volume_12m = target_eom_amount * np.random.uniform(10, 50)
            eom_volume_12m = target_eom_amount * np.random.uniform(0.1, 0.8)
            non_eom_volume_12m = total_volume_12m - eom_volume_12m
            avg_monthly_volume = total_volume_12m / 12
            max_transaction = target_eom_amount * np.random.uniform(0.5, 2.0)
            max_eom_transaction = target_eom_amount * np.random.uniform(0.3, 1.5)

            # Generate time-varying activity indicators
            active_months_12m = max(1, base_active_months_12m + np.random.randint(-2, 3))
            total_nonzero_eom_count = np.random.randint(0, active_months_12m)
            months_inactive = 12 - active_months_12m
            months_of_history = base_months_of_history + month_idx  # Growing history

            # Generate time-varying lagged values
            lag_1m_eom = target_eom_amount * np.random.uniform(0.5, 1.5)
            lag_3m_eom = target_eom_amount * np.random.uniform(0.3, 1.2)
            lag_12m_eom = target_eom_amount * np.random.uniform(0.1, 1.0)
            eom_ma3 = (lag_1m_eom + target_eom_amount + np.random.uniform(0.5, 1.5) * target_eom_amount) / 3

            # Generate time-varying growth metrics
            eom_yoy_growth = np.random.uniform(-0.5, 2.0) if months_of_history >= 12 else None
            eom_mom_growth = np.random.uniform(-0.3, 0.3)

            # Generate time-varying flags and indicators
            has_eom_history = total_nonzero_eom_count > 0
            is_zero_eom = 1 if target_eom_amount == 0 else 0
            current_month_has_eom = target_eom_amount > 0

            # Generate segment names (consistent for the series)
            full_segment_name = f"{overall_tier}_{general_pattern}__{eom_tier}EOM_{eom_pattern}"
            segment_name = f"{overall_tier}_{general_pattern}_{eom_pattern}"

            # Generate time-varying smooth scores (with some variation around base)
            eom_regularity_score = max(0, min(100, base_eom_regularity_score + np.random.normal(0, 5)))
            eom_stability_score = max(0, min(100, base_eom_stability_score + np.random.normal(0, 5)))
            eom_recency_score = max(0, min(100, base_eom_recency_score + np.random.normal(0, 5)))
            eom_concentration_score = max(0, min(100, base_eom_concentration_score + np.random.normal(0, 5)))
            eom_volume_score = max(0, min(100, base_eom_volume_score + np.random.normal(0, 5)))

            # Generate time-varying importance scores (with some variation around base)
            overall_importance_score = max(0.1, min(1.0, base_overall_importance_score + np.random.normal(0, 0.05)))
            eom_importance_score = max(0.1, min(1.0, base_eom_importance_score + np.random.normal(0, 0.05)))

            # Generate time-varying raw EOM metrics (with some variation around base)
            eom_concentration = max(0, min(1, base_eom_concentration + np.random.normal(0, 0.05)))
            eom_predictability = max(0, min(1, base_eom_predictability + np.random.normal(0, 0.05)))
            eom_frequency = max(0, min(1, base_eom_frequency + np.random.normal(0, 0.05)))
            eom_zero_ratio = max(0, min(0.8, base_eom_zero_ratio + np.random.normal(0, 0.05)))
            eom_cv = max(0, min(2, base_eom_cv + np.random.normal(0, 0.1)))

            # Generate time-varying general timeseries pattern features
            monthly_cv = max(0, min(1, base_monthly_cv + np.random.normal(0, 0.05)))
            transaction_regularity = max(0, min(1, base_transaction_regularity + np.random.normal(0, 0.05)))
            activity_rate = max(0, min(1, base_activity_rate + np.random.normal(0, 0.05)))
            transaction_dispersion = max(0, min(10, base_transaction_dispersion + np.random.normal(0, 0.5)))
            quarter_end_concentration = max(0, min(1, base_quarter_end_concentration + np.random.normal(0, 0.05)))
            year_end_concentration = max(0, min(1, base_year_end_concentration + np.random.normal(0, 0.05)))

            # Generate time-varying portfolio percentiles (with some variation around base)
            cumulative_overall_portfolio_pct = max(0, min(1, base_cumulative_overall_portfolio_pct + np.random.normal(0, 0.01)))
            cumulative_eom_portfolio_pct = max(0, min(1, base_cumulative_eom_portfolio_pct + np.random.normal(0, 0.01)))

            # Generate time-varying combined metrics (with some variation around base)
            combined_priority = max(0, min(100, base_combined_priority + np.random.normal(0, 2)))
            recommended_method = base_recommended_method  # Keep consistent
            forecast_complexity = base_forecast_complexity  # Keep consistent

            # Generate time-varying pattern uncertainty (with some variation around base)
            pattern_uncertainty = max(0, min(2, base_pattern_uncertainty + np.random.normal(0, 0.1)))

            # Generate raw rolling features (prefixed with raw_rf__)
            raw_rf_data = {
                "raw_rf__active_months_12m": active_months_12m,
                "raw_rf__rolling_avg_day_dispersion": np.random.uniform(0, 5),
                "raw_rf__rolling_avg_monthly_volume": avg_monthly_volume,
                "raw_rf__rolling_avg_non_eom": non_eom_volume_12m / 12,
                "raw_rf__rolling_avg_nonzero_eom": eom_volume_12m / max(1, total_nonzero_eom_count),
                "raw_rf__rolling_avg_transactions": np.random.uniform(10, 1000),
                "raw_rf__rolling_eom_volume_12m": eom_volume_12m,
                "raw_rf__rolling_max_eom": max_eom_transaction,
                "raw_rf__rolling_max_transaction": max_transaction,
                "raw_rf__rolling_non_eom_volume_12m": non_eom_volume_12m,
                "raw_rf__rolling_nonzero_eom_months": total_nonzero_eom_count,
                "raw_rf__rolling_quarter_end_volume": quarter_end_concentration * total_volume_12m,
                "raw_rf__rolling_std_eom": eom_cv * target_eom_amount,
                "raw_rf__rolling_std_monthly": monthly_cv * avg_monthly_volume,
                "raw_rf__rolling_std_transactions": np.random.uniform(0, 100),
                "raw_rf__rolling_total_volume_12m": total_volume_12m,
                "raw_rf__rolling_year_end_volume": year_end_concentration * total_volume_12m,
                "raw_rf__rolling_zero_eom_months": 12 - total_nonzero_eom_count,
                "raw_rf__total_nonzero_eom_count": total_nonzero_eom_count,
                "raw_rf__eom_amount_12m_ago": lag_12m_eom,
                "raw_rf__eom_amount_3m_ago": lag_3m_eom,
                "raw_rf__eom_amount_1m_ago": lag_1m_eom,
                "raw_rf__eom_ma3": eom_ma3,
                "raw_rf__months_of_history": months_of_history,
                "raw_rf__months_since_last_eom": np.random.randint(0, 12),
            }

            # Generate raw pattern metrics (prefixed with raw_pm__)
            raw_pm_data = {
                "raw_pm__eom_concentration": eom_concentration,
                "raw_pm__eom_predictability": eom_predictability,
                "raw_pm__eom_frequency": eom_frequency,
                "raw_pm__eom_zero_ratio": eom_zero_ratio,
                "raw_pm__eom_spike_ratio": np.random.uniform(0, 1),
                "raw_pm__eom_cv": eom_cv,
                "raw_pm__monthly_cv": monthly_cv,
                "raw_pm__transaction_regularity": transaction_regularity,
                "raw_pm__activity_rate": activity_rate,
                "raw_pm__quarter_end_concentration": quarter_end_concentration,
                "raw_pm__year_end_concentration": year_end_concentration,
                "raw_pm__transaction_dispersion": transaction_dispersion,
                "raw_pm__has_eom_history": has_eom_history,
                "raw_pm__months_inactive": months_inactive,
                "raw_pm__eom_periodicity": np.random.uniform(0, 1),
            }

            # Create the row data for this month
            row_data = {
                # Identifiers
                "dim_value": dim_value,
                "forecast_month": forecast_month,
                "year": year,
                "month_num": month_num,
                # Target variable
                "target_eom_amount": target_eom_amount,
                # Dual importance metrics
                "overall_importance_tier": overall_tier,
                "eom_importance_tier": eom_tier,
                "overall_importance_score": round(overall_importance_score, 5),
                "eom_importance_score": round(eom_importance_score, 5),
                "eom_risk_flag": base_eom_risk_flag,
                "has_eom_history": has_eom_history,
                # Smooth scores
                "eom_regularity_score": round(eom_regularity_score, 1),
                "eom_stability_score": round(eom_stability_score, 1),
                "eom_recency_score": round(eom_recency_score, 1),
                "eom_concentration_score": round(eom_concentration_score, 1),
                "eom_volume_score": round(eom_volume_score, 1),
                # Pattern probabilities
                "eom_pattern_primary": eom_pattern,
                "eom_pattern_confidence_pct": round(pattern_confidence * 100, 1),
                "prob_continuous_stable_pct": round(prob_continuous_stable_pct, 1),
                "prob_continuous_volatile_pct": round(prob_continuous_volatile_pct, 1),
                "prob_intermittent_active_pct": round(prob_intermittent_active_pct, 1),
                "prob_intermittent_dormant_pct": round(prob_intermittent_dormant_pct, 1),
                "prob_rare_recent_pct": round(prob_rare_recent_pct, 1),
                "prob_rare_stale_pct": round(prob_rare_stale_pct, 1),
                "pattern_uncertainty": round(pattern_uncertainty, 3),
                # General timeseries pattern
                "general_pattern": general_pattern,
                # Combined metrics
                "segment_name": segment_name,
                "full_segment_name": full_segment_name,
                "combined_priority": round(combined_priority, 2),
                "recommended_method": recommended_method,
                "forecast_complexity": forecast_complexity,
                # Volume metrics
                "total_volume_12m": total_volume_12m,
                "eom_volume_12m": eom_volume_12m,
                "non_eom_volume_12m": non_eom_volume_12m,
                "avg_monthly_volume": avg_monthly_volume,
                "max_transaction": max_transaction,
                "max_eom_transaction": max_eom_transaction,
                # Raw EOM metrics
                "eom_concentration": round(eom_concentration, 3),
                "eom_predictability": round(eom_predictability, 3),
                "eom_frequency": round(eom_frequency, 3),
                "eom_zero_ratio": round(eom_zero_ratio, 3),
                "eom_cv": round(eom_cv, 3),
                # General timeseries pattern features
                "monthly_cv": round(monthly_cv, 3),
                "transaction_regularity": round(transaction_regularity, 3),
                "activity_rate": round(activity_rate, 3),
                "transaction_dispersion": round(transaction_dispersion, 2),
                "quarter_end_concentration": round(quarter_end_concentration, 3),
                "year_end_concentration": round(year_end_concentration, 3),
                # Activity indicators
                "active_months_12m": active_months_12m,
                "total_nonzero_eom_count": total_nonzero_eom_count,
                "months_inactive": months_inactive,
                "months_of_history": months_of_history,
                # Portfolio percentiles
                "cumulative_overall_portfolio_pct": round(cumulative_overall_portfolio_pct, 4),
                "cumulative_eom_portfolio_pct": round(cumulative_eom_portfolio_pct, 4),
                # Lagged values for forecasting
                "lag_1m_eom": lag_1m_eom,
                "lag_3m_eom": lag_3m_eom,
                "lag_12m_eom": lag_12m_eom,
                "eom_ma3": round(eom_ma3, 2),
                # Growth metrics
                "eom_yoy_growth": round(eom_yoy_growth, 3) if eom_yoy_growth is not None else None,
                "eom_mom_growth": round(eom_mom_growth, 3),
                # Calendar features
                "is_quarter_end": is_quarter_end,
                "is_year_end": is_year_end,
                "month_of_year": month_of_year,
                # Current month status
                "is_zero_eom": is_zero_eom,
                "current_month_has_eom": current_month_has_eom,
            }

            # Add raw rolling features
            row_data.update(raw_rf_data)

            # Add raw pattern metrics
            row_data.update(raw_pm_data)

            individual_data.append(row_data)

    # Create individual DataFrame
    individual_df = pd.DataFrame(individual_data)

    # Generate aggregated "others" data for each month
    others_data = []

    for month_idx, forecast_month in enumerate(months):
        forecast_date = datetime.strptime(forecast_month, "%Y-%m-%d")
        year = forecast_date.year
        month_num = forecast_date.month
        month_of_year = month_num
        is_quarter_end = month_num in [3, 6, 9, 12]
        is_year_end = month_num == 12

        # Create "others::IN" category for this month
        others_in_row = {
            # Identifiers
            "dim_value": "others::IN",
            "forecast_month": forecast_month,
            "year": year,
            "month_num": month_num,
            # Target variable (sum of smaller amounts with some variation)
            "target_eom_amount": np.random.uniform(10000, 100000),
            # Dual importance metrics (NULL for aggregated)
            "overall_importance_tier": None,
            "eom_importance_tier": None,
            "overall_importance_score": None,
            "eom_importance_score": None,
            "eom_risk_flag": None,
            "has_eom_history": None,
            # Smooth scores (NULL for aggregated)
            "eom_regularity_score": None,
            "eom_stability_score": None,
            "eom_recency_score": None,
            "eom_concentration_score": None,
            "eom_volume_score": None,
            # Pattern probabilities (NULL for aggregated)
            "eom_pattern_primary": "AGGREGATED_OTHERS",
            "eom_pattern_confidence_pct": None,
            "prob_continuous_stable_pct": None,
            "prob_continuous_volatile_pct": None,
            "prob_intermittent_active_pct": None,
            "prob_intermittent_dormant_pct": None,
            "prob_rare_recent_pct": None,
            "prob_rare_stale_pct": None,
            "pattern_uncertainty": None,
            # General timeseries pattern (NULL for aggregated)
            "general_pattern": None,
            # Combined metrics (NULL for aggregated)
            "segment_name": None,
            "full_segment_name": None,
            "combined_priority": None,
            "recommended_method": None,
            "forecast_complexity": None,
            # Volume metrics (NULL for aggregated)
            "total_volume_12m": None,
            "eom_volume_12m": None,
            "non_eom_volume_12m": None,
            "avg_monthly_volume": None,
            "max_transaction": None,
            "max_eom_transaction": None,
            # Raw EOM metrics (NULL for aggregated)
            "eom_concentration": None,
            "eom_predictability": None,
            "eom_frequency": None,
            "eom_zero_ratio": None,
            "eom_cv": None,
            # General timeseries pattern features (NULL for aggregated)
            "monthly_cv": None,
            "transaction_regularity": None,
            "activity_rate": None,
            "transaction_dispersion": None,
            "quarter_end_concentration": None,
            "year_end_concentration": None,
            # Activity indicators (NULL for aggregated)
            "active_months_12m": None,
            "total_nonzero_eom_count": None,
            "months_inactive": None,
            "months_of_history": None,
            # Portfolio percentiles (summed for aggregated)
            "cumulative_overall_portfolio_pct": np.random.uniform(0.1, 0.3),
            "cumulative_eom_portfolio_pct": np.random.uniform(0.1, 0.3),
            # Lagged values (NULL for aggregated)
            "lag_1m_eom": None,
            "lag_3m_eom": None,
            "lag_12m_eom": None,
            "eom_ma3": None,
            # Growth metrics (NULL for aggregated)
            "eom_yoy_growth": None,
            "eom_mom_growth": None,
            # Calendar features
            "is_quarter_end": is_quarter_end,
            "is_year_end": is_year_end,
            "month_of_year": month_of_year,
            # Current month status (NULL for aggregated)
            "is_zero_eom": None,
            "current_month_has_eom": None,
        }

        # Add NULL values for all raw features
        for col in individual_df.columns:
            if col.startswith("raw_rf__") or col.startswith("raw_pm__"):
                others_in_row[col] = None

        others_data.append(others_in_row)

        # Create "others::OUT" category for this month
        others_out_row = others_in_row.copy()
        others_out_row["dim_value"] = "others::OUT"
        others_out_row["target_eom_amount"] = np.random.uniform(1000, 50000)
        others_out_row["cumulative_overall_portfolio_pct"] = np.random.uniform(0.01, 0.1)
        others_out_row["cumulative_eom_portfolio_pct"] = np.random.uniform(0.01, 0.1)

        others_data.append(others_out_row)

    # Create others DataFrame
    others_df = pd.DataFrame(others_data)

    # Combine individual and others DataFrames
    final_df = pd.concat([individual_df, others_df], ignore_index=True)

    # Sort by combined_priority (descending) and dim_value
    final_df = final_df.sort_values(["combined_priority", "dim_value"], ascending=[False, True], na_position="last")

    return final_df


def main():
    """Generate and save sample feature data."""
    print("Generating sample feature data...")

    # Generate the data
    df = generate_sample_feature_data(
        num_individual_series=20,  # Reduced for time series
        start_date="2023-01-01",
        end_date="2025-09-01",
        seed=42,
    )

    # Save to CSV
    output_file = "/Users/emile.dimas/dev/radio-forecast-segmentation/dataset/feature_df.csv"
    df.to_csv(output_file, index=False)

    print(f"Sample data generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_file}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nTime series summary:")
    print(f"Date range: {df['forecast_month'].min()} to {df['forecast_month'].max()}")
    print(f"Number of months: {df['forecast_month'].nunique()}")
    print(f"Number of individual series: {df[df['dim_value'].str.startswith('series_')]['dim_value'].nunique()}")
    print(f"Number of aggregated categories: {df[df['dim_value'].str.startswith('others::')]['dim_value'].nunique()}")

    print(f"\nSample of individual series (first 10 rows):")
    individual_data = df[df["dim_value"].str.startswith("series_")]
    print(
        individual_data[
            [
                "dim_value",
                "forecast_month",
                "overall_importance_tier",
                "eom_importance_tier",
                "eom_pattern_primary",
                "target_eom_amount",
            ]
        ].head(10)
    )

    print(f"\nSample of aggregated 'others' categories (first 10 rows):")
    others_data = df[df["dim_value"].str.startswith("others::")]
    print(
        others_data[
            [
                "dim_value",
                "forecast_month",
                "target_eom_amount",
                "cumulative_overall_portfolio_pct",
                "cumulative_eom_portfolio_pct",
                "eom_pattern_primary",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
