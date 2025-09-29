# Automated Forecasting Scripts

This directory contains scripts for automated forecasting evaluation and comparison.

## Scripts Overview

### 1. `auto_forecast_compare.py`
Python script that automatically runs all forecasting methods and generates comprehensive comparison reports.

### 2. `run_auto_forecast.sh`
Bash wrapper script for easier command-line usage with environment variable validation.

### 3. `auto_config.json`
Default configuration file optimized for comprehensive forecasting evaluation.

## Quick Start

### Method 1: Using the CLI Command (Recommended)

```bash
# Set environment variables
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_username
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_WAREHOUSE=your_warehouse
export SNOWFLAKE_DATABASE=your_database
export SNOWFLAKE_SCHEMA=your_schema

# Run automated comparison via CLI
radio-forecast forecast auto \
  --segmented-table my_segmented_data \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --verbose
```

### Method 2: Using the Bash Script

```bash
# Set environment variables (same as above)

# Run the bash script
./scripts/run_auto_forecast.sh \
  --segmented-table my_segmented_data \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --verbose
```

### Method 3: Direct Python Script

```bash
# Set environment variables (same as above)

# Run Python script directly
python scripts/auto_forecast_compare.py \
  --segmented-table my_segmented_data \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --config scripts/auto_config.json \
  --verbose
```

## What the Automated Script Does

1. **Loads Configuration**: Uses provided config file or creates default configuration
2. **Connects to Snowflake**: Uses environment variables for connection
3. **Runs ALL Methods**: Tests all 12 available forecasting methods:
   - Baseline: zero, naive
   - Statistical: moving_average, weighted_moving_average, arima, sarima, ets
   - Machine Learning: xgboost_individual, xgboost_global
   - Specialized: croston, ensemble, segment_aggregate

4. **Evaluates at All Levels**:
   - **dim_value**: Individual counterparty_direction pairs
   - **segment**: Credit (::IN), Debit (::OUT), Net (Credit - Debit)
   - **overall**: Total across all segments

5. **Generates Comprehensive Reports**:
   - Overall method rankings by MAE, RMSE, MAPE, Directional Accuracy
   - Performance breakdown by evaluation level
   - Automated recommendations for best methods
   - Detailed text report saved to `reports/` directory

## Output

The script generates several outputs:

### Console Output
- Configuration summary with all methods being tested
- Real-time progress updates
- Comprehensive performance comparison tables
- Method rankings and recommendations

### Saved Results
- **Snowflake Tables**: Forecasts and error metrics saved with specified prefix
- **Text Report**: Detailed comparison report in `reports/` directory
- **Console Log**: All output displayed in rich, colored format

### Sample Output

```
🏆 Overall Method Rankings

┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Rank ┃ Method                 ┃ MAE     ┃ RMSE    ┃ MAPE    ┃ Dir. Accuracy ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 1    │ xgboost_global         │ 0.1234  │ 0.2345  │ 12.34%  │ 78.90%        │
│ 2    │ arima                  │ 0.1456  │ 0.2567  │ 14.56%  │ 76.54%        │
│ 3    │ ensemble               │ 0.1567  │ 0.2678  │ 15.67%  │ 75.43%        │
└──────┴────────────────────────┴─────────┴─────────┴─────────┴───────────────┘

🎯 Automated Recommendations:
• Best Overall (Lowest MAE): xgboost_global (MAE: 0.1234)
• Best Directional Accuracy: ensemble (78.90%)
• Most Balanced: arima
• Top 3 Recommendations: xgboost_global, arima, ensemble
```

## Configuration

### Environment Variables (Required)
```bash
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_username  
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_WAREHOUSE=your_warehouse
export SNOWFLAKE_DATABASE=your_database
export SNOWFLAKE_SCHEMA=your_schema
export SNOWFLAKE_ROLE=your_role  # Optional
```

### Custom Configuration File
Create a JSON file with your preferred settings:

```json
{
  "forecast_horizon": 1,
  "min_history_months": 12,
  "methods_to_test": ["naive", "arima", "xgboost_global"],
  "ma_windows": [3, 6, 12],
  "arima_auto_select": true,
  "xgb_n_estimators": 200,
  "calculate_mae": true,
  "calculate_rmse": true,
  "save_forecasts": true,
  "output_table_prefix": "my_custom_forecast"
}
```

## Advanced Usage

### Custom Output Prefix
```bash
radio-forecast forecast auto \
  --segmented-table my_data \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --output-prefix "quarterly_eval_2024q1"
```

### Using Custom Configuration
```bash
radio-forecast forecast auto \
  --segmented-table my_data \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --config my_custom_config.json
```

### Batch Processing
Create a batch script to run multiple evaluations:

```bash
#!/bin/bash

# Set environment variables once
export SNOWFLAKE_ACCOUNT=your_account
# ... other variables

# Run multiple evaluations
for quarter in "2024q1" "2024q2" "2024q3"; do
    radio-forecast forecast auto \
      --segmented-table "segmented_data_$quarter" \
      --train-start 2023-01-01 \
      --train-end 2023-12-31 \
      --test-start 2024-01-01 \
      --test-end 2024-03-31 \
      --output-prefix "eval_$quarter"
done
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```
   Error: Missing required environment variables: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER
   ```
   Solution: Set all required environment variables

2. **Table Not Found**
   ```
   Error: Table 'my_segmented_data' does not exist
   ```
   Solution: Ensure the segmented table exists and is accessible

3. **Insufficient Permissions**
   ```
   Error: Access denied for table creation
   ```
   Solution: Ensure your Snowflake user has CREATE TABLE permissions

4. **Memory Issues**
   ```
   Error: Out of memory during XGBoost training
   ```
   Solution: Reduce `xgb_n_estimators` in config or filter data

### Performance Tips

1. **Parallel Processing**: The script uses parallel processing by default
2. **Method Selection**: For faster runs, limit methods in config file
3. **Data Filtering**: Pre-filter your segmented data for specific segments
4. **Warehouse Size**: Use larger Snowflake warehouse for better performance

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Automated Forecasting Evaluation

on:
  schedule:
    - cron: '0 2 * * 1'  # Run every Monday at 2 AM

jobs:
  forecast-evaluation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run automated forecasting
        env:
          SNOWFLAKE_ACCOUNT: ${{ secrets.SNOWFLAKE_ACCOUNT }}
          SNOWFLAKE_USER: ${{ secrets.SNOWFLAKE_USER }}
          SNOWFLAKE_PASSWORD: ${{ secrets.SNOWFLAKE_PASSWORD }}
          SNOWFLAKE_WAREHOUSE: ${{ secrets.SNOWFLAKE_WAREHOUSE }}
          SNOWFLAKE_DATABASE: ${{ secrets.SNOWFLAKE_DATABASE }}
          SNOWFLAKE_SCHEMA: ${{ secrets.SNOWFLAKE_SCHEMA }}
        run: |
          poetry run python scripts/auto_forecast_compare.py \
            --segmented-table weekly_segmented_data \
            --train-start 2023-01-01 \
            --train-end 2023-12-31 \
            --test-start 2024-01-01 \
            --test-end 2024-03-31 \
            --output-prefix "weekly_auto_$(date +%Y%m%d)"
```

This automated approach gives you a complete, hands-off solution for comprehensive forecasting evaluation!
