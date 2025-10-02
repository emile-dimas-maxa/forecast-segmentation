"""
Data loading and preparation utilities for Streamlit app
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from snowflake.snowpark import Session

    SNOWPARK_AVAILABLE = True
except ImportError:
    SNOWPARK_AVAILABLE = False
    Session = None


def generate_sample_timeseries_data(n_series: int = 100, n_months: int = 24) -> pd.DataFrame:
    """Generate sample time series data for demonstration purposes"""

    np.random.seed(42)  # For reproducibility

    data = []

    for series_id in range(n_series):
        # Generate different types of patterns
        pattern_type = np.random.choice(["stable", "volatile", "intermittent", "seasonal", "emerging"])

        base_volume = np.random.lognormal(15, 2)  # Log-normal distribution for realistic volumes

        for month in range(n_months):
            month_date = date(2022, 1, 1).replace(month=(month % 12) + 1, year=2022 + month // 12)

            # Generate pattern-specific behavior
            if pattern_type == "stable":
                monthly_volume = base_volume * (1 + np.random.normal(0, 0.1))
                eom_concentration = 0.7 + np.random.normal(0, 0.1)
                eom_frequency = 0.9 + np.random.normal(0, 0.05)
                monthly_cv = 0.15 + np.random.normal(0, 0.05)

            elif pattern_type == "volatile":
                monthly_volume = base_volume * (1 + np.random.normal(0, 0.4))
                eom_concentration = 0.6 + np.random.normal(0, 0.15)
                eom_frequency = 0.8 + np.random.normal(0, 0.1)
                monthly_cv = 0.6 + np.random.normal(0, 0.1)

            elif pattern_type == "intermittent":
                # Sometimes zero, sometimes high
                if np.random.random() < 0.4:
                    monthly_volume = 0
                    eom_concentration = 0
                else:
                    monthly_volume = base_volume * (1 + np.random.normal(0, 0.3))
                    eom_concentration = 0.5 + np.random.normal(0, 0.2)
                eom_frequency = 0.4 + np.random.normal(0, 0.1)
                monthly_cv = 0.8 + np.random.normal(0, 0.1)

            elif pattern_type == "seasonal":
                # Higher in Q4, lower in Q1
                seasonal_factor = 1.5 if month_date.month in [10, 11, 12] else 0.7 if month_date.month in [1, 2, 3] else 1.0
                monthly_volume = base_volume * seasonal_factor * (1 + np.random.normal(0, 0.2))
                eom_concentration = 0.8 + np.random.normal(0, 0.1)
                eom_frequency = 0.7 + np.random.normal(0, 0.1)
                monthly_cv = 0.3 + np.random.normal(0, 0.1)

            else:  # emerging
                # Only recent months have data
                if month < n_months - 6:
                    monthly_volume = 0
                    eom_concentration = 0
                    eom_frequency = 0
                else:
                    monthly_volume = base_volume * (1 + np.random.normal(0, 0.2))
                    eom_concentration = 0.6 + np.random.normal(0, 0.1)
                    eom_frequency = 0.5 + np.random.normal(0, 0.1)
                monthly_cv = 0.4 + np.random.normal(0, 0.1)

            # Ensure values are within reasonable bounds
            eom_concentration = np.clip(eom_concentration, 0, 1)
            eom_frequency = np.clip(eom_frequency, 0, 1)
            monthly_cv = np.clip(monthly_cv, 0, 2)
            monthly_volume = max(0, monthly_volume)

            # Calculate EOM amount
            eom_amount = monthly_volume * eom_concentration if monthly_volume > 0 else 0

            # Calculate derived metrics
            months_of_history = month + 1
            has_eom_history = (
                1
                if eom_amount > 0 or any(d["target_eom_amount"] > 0 for d in data if d["dim_value"] == f"series_{series_id}")
                else 0
            )
            months_since_last_eom = (
                1
                if eom_amount > 0
                else (
                    min(
                        [
                            i
                            for i, d in enumerate(reversed([d for d in data if d["dim_value"] == f"series_{series_id}"]))
                            if d["target_eom_amount"] > 0
                        ]
                        + [999]
                    )
                    + 1
                    if has_eom_history
                    else 999
                )
            )

            data.append(
                {
                    "dim_value": f"series_{series_id}",
                    "forecast_month": month_date,
                    "year": month_date.year,
                    "month_num": month_date.month,
                    "monthly_volume": monthly_volume,
                    "target_eom_amount": eom_amount,
                    "eom_concentration": eom_concentration,
                    "eom_frequency": eom_frequency,
                    "eom_cv": monthly_cv,
                    "monthly_cv": monthly_cv,
                    "months_of_history": months_of_history,
                    "has_eom_history": has_eom_history,
                    "has_nonzero_eom": 1 if eom_amount > 0 else 0,
                    "months_since_last_eom": months_since_last_eom,
                    "pattern_type_true": pattern_type,  # Ground truth for evaluation
                    "activity_rate": np.random.uniform(0.3, 1.0),
                    "transaction_regularity": np.random.uniform(0.2, 0.9),
                    "transaction_dispersion": np.random.uniform(2, 15),
                    "quarter_end_concentration": 0.3 if pattern_type == "seasonal" else np.random.uniform(0.1, 0.25),
                    "year_end_concentration": 0.6 if pattern_type == "seasonal" else np.random.uniform(0.1, 0.4),
                    "active_months_12m": min(12, months_of_history),
                    "months_inactive": 0 if monthly_volume > 0 else 1,
                }
            )

    df = pd.DataFrame(data)

    # Calculate portfolio-level metrics
    total_portfolio_volume = df.groupby("forecast_month")["monthly_volume"].sum().mean()
    total_portfolio_eom_volume = df.groupby("forecast_month")["target_eom_amount"].sum().mean()

    # Add importance scores and percentiles
    df["rolling_total_volume_12m"] = (
        df.groupby("dim_value")["monthly_volume"].rolling(12, min_periods=1).sum().reset_index(0, drop=True)
    )
    df["rolling_eom_volume_12m"] = (
        df.groupby("dim_value")["target_eom_amount"].rolling(12, min_periods=1).sum().reset_index(0, drop=True)
    )

    df["overall_importance_score"] = df["rolling_total_volume_12m"] / total_portfolio_volume
    df["eom_importance_score"] = df["rolling_eom_volume_12m"] / total_portfolio_eom_volume

    # Calculate percentiles for importance classification
    df["cumulative_overall_portfolio_pct"] = df["overall_importance_score"].rank(pct=True, ascending=False)
    df["cumulative_eom_portfolio_pct"] = df["eom_importance_score"].rank(pct=True, ascending=False)

    # Add more derived metrics
    df["rolling_avg_monthly_volume"] = df["rolling_total_volume_12m"] / 12
    df["rolling_max_transaction"] = (
        df.groupby("dim_value")["monthly_volume"].rolling(12, min_periods=1).max().reset_index(0, drop=True)
    )
    df["rolling_avg_nonzero_eom"] = (
        df.groupby("dim_value")["target_eom_amount"].rolling(12, min_periods=1).mean().reset_index(0, drop=True)
    )
    df["rolling_max_eom"] = df.groupby("dim_value")["target_eom_amount"].rolling(12, min_periods=1).max().reset_index(0, drop=True)
    df["total_nonzero_eom_count"] = df.groupby("dim_value")["has_nonzero_eom"].cumsum()
    df["total_portfolio_volume"] = total_portfolio_volume
    df["total_portfolio_eom_volume"] = total_portfolio_eom_volume

    return df


def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)

        # Convert date columns if they exist
        date_columns = ["forecast_month", "month", "date", "transaction_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}") from e


def load_snowflake_data(connection_params: dict, table_name: str, query: str | None = None) -> pd.DataFrame:
    """Load data from Snowflake using Snowpark"""
    if not SNOWPARK_AVAILABLE:
        raise ImportError("Snowpark is not available. Install with: pip install snowflake-snowpark-python")

    try:
        # Create Snowpark session
        session = Session.builder.configs(connection_params).create()

        # Execute query or load table
        df_snowpark = session.sql(query) if query else session.table(table_name)

        # Convert to pandas
        df = df_snowpark.to_pandas()

        # Close session
        session.close()

        return df
    except Exception as e:
        raise ValueError(f"Error loading from Snowflake: {e}") from e


def load_sample_data(cache_file: str = "sample_data.pkl") -> pd.DataFrame:
    """Load or generate sample data with caching"""
    cache_path = Path(__file__).parent / cache_file

    if cache_path.exists():
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"Error loading cached data: {e}")

    # Generate new data
    df = generate_sample_timeseries_data()

    # Cache for future use
    try:
        df.to_pickle(cache_path)
    except Exception as e:
        print(f"Error caching data: {e}")

    return df


def validate_data_format(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that the dataframe has required columns for EOM classification"""
    required_columns = [
        "dim_value",  # Series identifier
        "forecast_month",  # Time dimension
        "target_eom_amount",  # EOM amount (target variable)
    ]

    optional_columns = [
        "monthly_volume",
        "eom_concentration",
        "eom_frequency",
        "eom_cv",
        "months_of_history",
        "has_eom_history",
        "has_nonzero_eom",
        "months_since_last_eom",
    ]

    missing_required = [col for col in required_columns if col not in df.columns]
    missing_optional = [col for col in optional_columns if col not in df.columns]

    issues = []

    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")

    if missing_optional:
        issues.append(f"Missing optional columns (will be generated): {missing_optional}")

    # Check data types
    if "forecast_month" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["forecast_month"]):
        issues.append("'forecast_month' column should be datetime type")

    if "target_eom_amount" in df.columns and not pd.api.types.is_numeric_dtype(df["target_eom_amount"]):
        issues.append("'target_eom_amount' column should be numeric")

    is_valid = len(missing_required) == 0
    return is_valid, issues


def prepare_classification_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for classification by taking the latest month for each series"""
    # Get the latest month for each series
    latest_data = df.loc[df.groupby("dim_value")["forecast_month"].idxmax()].copy()

    # Ensure required columns exist with defaults
    required_cols = {
        "eom_frequency": 0.5,
        "eom_cv": 0.3,
        "eom_concentration": 0.5,
        "months_of_history": 12,
        "has_eom_history": 1,
        "has_nonzero_eom": 0,
        "months_since_last_eom": 1,
        "overall_importance_score": 0.01,
        "eom_importance_score": 0.01,
    }

    for col, default_val in required_cols.items():
        if col not in latest_data.columns:
            latest_data[col] = default_val
        else:
            latest_data[col] = latest_data[col].fillna(default_val)

    return latest_data


def save_human_labels(labels: dict[str, str], filename: str = "human_labels.json"):
    """Save human labels to file"""
    labels_path = Path(__file__).parent / filename

    # Load existing labels if they exist
    existing_labels = {}
    if labels_path.exists():
        try:
            with open(labels_path) as f:
                existing_labels = json.load(f)
        except Exception as e:
            print(f"Error loading existing labels: {e}")

    # Update with new labels
    existing_labels.update(labels)

    # Save updated labels
    try:
        with open(labels_path, "w") as f:
            json.dump(existing_labels, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving labels: {e}")
        return False


def load_human_labels(filename: str = "human_labels.json") -> dict[str, str]:
    """Load human labels from file"""
    labels_path = Path(__file__).parent / filename

    if not labels_path.exists():
        return {}

    try:
        with open(labels_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading labels: {e}")
        return {}


def calculate_evaluation_metrics(df: pd.DataFrame, human_labels: dict[str, str]) -> dict[str, float]:
    """Calculate evaluation metrics comparing predicted vs human labels"""

    # Filter to series that have human labels
    labeled_series = df[df["dim_value"].isin(human_labels.keys())].copy()

    if len(labeled_series) == 0:
        return {"error": "No labeled series found"}

    # Add human labels to dataframe
    labeled_series["human_label"] = labeled_series["dim_value"].map(human_labels)

    # Calculate accuracy
    correct_predictions = (labeled_series["eom_pattern"] == labeled_series["human_label"]).sum()
    total_predictions = len(labeled_series)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate per-class metrics
    unique_labels = set(labeled_series["human_label"].unique()) | set(labeled_series["eom_pattern"].unique())

    class_metrics = {}
    for label in unique_labels:
        true_positives = ((labeled_series["eom_pattern"] == label) & (labeled_series["human_label"] == label)).sum()
        false_positives = ((labeled_series["eom_pattern"] == label) & (labeled_series["human_label"] != label)).sum()
        false_negatives = ((labeled_series["eom_pattern"] != label) & (labeled_series["human_label"] == label)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": ((labeled_series["human_label"] == label).sum()),
        }

    # Calculate macro averages
    macro_precision = np.mean([metrics["precision"] for metrics in class_metrics.values()])
    macro_recall = np.mean([metrics["recall"] for metrics in class_metrics.values()])
    macro_f1 = np.mean([metrics["f1"] for metrics in class_metrics.values()])

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "n_labeled": total_predictions,
        "class_metrics": class_metrics,
    }


def get_timeseries_plot_data(df: pd.DataFrame, series_id: str) -> tuple[pd.DataFrame, dict]:
    """Get time series data for plotting a specific series"""
    series_data = df[df["dim_value"] == series_id].copy()
    series_data = series_data.sort_values("forecast_month")

    # Calculate summary statistics
    summary = {
        "total_months": len(series_data),
        "avg_monthly_volume": series_data["monthly_volume"].mean(),
        "avg_eom_amount": series_data["target_eom_amount"].mean(),
        "eom_frequency": (series_data["target_eom_amount"] > 0).mean(),
        "volatility_cv": series_data["monthly_volume"].std() / series_data["monthly_volume"].mean()
        if series_data["monthly_volume"].mean() > 0
        else 0,
        "pattern_type_true": series_data["pattern_type_true"].iloc[-1] if "pattern_type_true" in series_data.columns else "unknown",
    }

    return series_data, summary
