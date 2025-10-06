# Radio Forecast Segmentation Dataset

This dataset contains sample time series data that matches the expected output format of `feature_df.csv` from the feature pipeline in the radio forecast segmentation project.

## Files

- `feature_df.csv` - Sample feature data with 726 rows and 105 columns
- `generate_sample_data.py` - Python script to generate the sample data
- `README.md` - This documentation file

## Data Structure

The dataset contains complete time series for each dimension value across multiple months:

### Time Series Coverage
- **Date Range**: 2023-01-01 to 2025-09-01 (33 months)
- **Total Rows**: 726 (20 individual series × 33 months + 2 aggregated categories × 33 months)
- **Individual Series**: 20 unique series with full time series data
- **Aggregated Categories**: 2 categories (others::IN, others::OUT) with time series data

### 1. Individual Series (660 rows = 20 series × 33 months)
- **dim_value**: Unique identifier for each series (e.g., "series_001", "series_002", etc.)
- **Time Series**: Complete monthly data from 2023-01-01 to 2025-09-01
- **Full feature sets**: All 105 columns populated with realistic values
- **Importance tiers**: CRITICAL, HIGH, MEDIUM, LOW, NONE (weighted distribution)
- **EOM patterns**: CONTINUOUS_STABLE, CONTINUOUS_VOLATILE, INTERMITTENT_ACTIVE, INTERMITTENT_DORMANT, RARE_RECENT, RARE_STALE
- **General patterns**: STABLE, VOLATILE, INTERMITTENT, HIGHLY_SEASONAL
- **Time-varying features**: Target amounts, volume metrics, and activity indicators change over time
- **Consistent features**: Importance tiers, patterns, and segment names remain consistent for each series

### 2. Aggregated "Others" Categories (66 rows = 2 categories × 33 months)
- **others::IN**: Aggregated category for less important series with "::IN" suffix
- **others::OUT**: Aggregated category for less important series with "::OUT" suffix
- **Time Series**: Complete monthly data from 2023-01-01 to 2025-09-01
- **Limited data**: Only target amounts and portfolio percentiles are summed
- **NULL values**: All other feature fields are NULL
- **Special pattern**: `eom_pattern_primary` = "AGGREGATED_OTHERS"

## Key Columns

### Identifiers
- `dim_value` - Series identifier
- `forecast_month` - Target forecast month (2025-09-01)
- `year`, `month_num`, `month_of_year` - Time dimensions

### Target Variable
- `target_eom_amount` - End-of-month amount (target for forecasting)

### Importance Classification
- `overall_importance_tier` - Overall importance tier (CRITICAL, HIGH, MEDIUM, LOW, NONE)
- `eom_importance_tier` - EOM-specific importance tier
- `overall_importance_score`, `eom_importance_score` - Numerical importance scores

### EOM Pattern Analysis
- `eom_pattern_primary` - Primary EOM pattern classification
- `eom_pattern_confidence_pct` - Confidence in pattern classification
- `prob_*_pct` - Probabilities for each pattern type
- `pattern_uncertainty` - Classification entropy

### Smooth Scores
- `eom_regularity_score` - EOM regularity score (0-100)
- `eom_stability_score` - EOM stability score (0-100)
- `eom_recency_score` - EOM recency score (0-100)
- `eom_concentration_score` - EOM concentration score (0-100)
- `eom_volume_score` - EOM volume importance score (0-100)

### Volume Metrics
- `total_volume_12m` - Total volume over 12 months
- `eom_volume_12m` - EOM volume over 12 months
- `non_eom_volume_12m` - Non-EOM volume over 12 months
- `avg_monthly_volume` - Average monthly volume
- `max_transaction`, `max_eom_transaction` - Maximum transaction amounts

### Raw Features
- `raw_rf__*` - Raw rolling features (22 columns)
- `raw_pm__*` - Raw pattern metrics (15 columns)

### Portfolio Metrics
- `cumulative_overall_portfolio_pct` - Cumulative overall portfolio percentile
- `cumulative_eom_portfolio_pct` - Cumulative EOM portfolio percentile

### Forecasting Features
- `lag_1m_eom`, `lag_3m_eom`, `lag_12m_eom` - Lagged EOM amounts
- `eom_ma3` - 3-month moving average
- `eom_yoy_growth`, `eom_mom_growth` - Growth metrics

## Data Generation

The sample data is generated using the `generate_sample_data.py` script with the following characteristics:

- **Time Series Structure**: Complete monthly data from 2023-01-01 to 2025-09-01
- **Realistic distributions**: Importance tiers and patterns follow realistic business distributions
- **Time-varying features**: Target amounts, volume metrics, and activity indicators change over time with seasonality and trends
- **Consistent series characteristics**: Each series maintains consistent importance tiers, patterns, and segment names across time
- **Correlated features**: Related features are generated with appropriate correlations
- **Proper aggregation**: "Others" categories follow the exact aggregation rules from the feature pipeline
- **Reproducible**: Uses fixed random seed (42) for consistent results

## Usage

This dataset can be used for:
- Testing the forecasting pipeline with complete time series data
- Validating model performance across multiple time periods
- Understanding the expected data format and structure
- Development and testing of time series forecasting features
- Backtesting and cross-validation of forecasting models

## Column Count

Total: 105 columns
- Core features: 65 columns
- Raw rolling features: 22 columns (raw_rf__*)
- Raw pattern metrics: 15 columns (raw_pm__*)
- Calendar features: 3 columns

## Data Quality

- No missing values in individual series
- Proper NULL values in aggregated "others" categories
- Realistic value ranges and distributions
- Consistent data types and formats
- Complete time series coverage for all dimension values
- Time-varying features show realistic seasonality and trends
