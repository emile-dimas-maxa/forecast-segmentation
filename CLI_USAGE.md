# Radio Forecast Segmentation CLI

A comprehensive command-line interface for running segmentation and forecasting pipelines on time series data using Typer and Rich.

## Installation

1. Install dependencies using Poetry:
```bash
poetry install
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Usage

The CLI provides several commands for different operations:

### Main Commands

```bash
# Show help
python cli.py --help

# Show version
python cli.py --version

# Show configuration templates
python cli.py config

# List available forecasting methods
python cli.py methods
```

### Segmentation Pipeline

Run the segmentation pipeline to analyze and segment time series data:

```bash
# Basic usage with command line parameters
python cli.py segmentation run \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --min-months 3 \
  --rolling-window 12 \
  --output-table segmented_results

# Using a configuration file
python cli.py segmentation run \
  --config example_configs/segmentation_config.json \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --output-table segmented_results

# Save results to local CSV file
python cli.py segmentation run \
  --config example_configs/segmentation_config.json \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --save-to-file results/segmentation_output.csv
```

#### Segmentation Parameters

- `--config`: Path to JSON configuration file
- `--start-date`: Analysis start date (YYYY-MM-DD)
- `--end-date`: Analysis end date (YYYY-MM-DD), defaults to current date
- `--source-table`: Source table name (default: int__t__cad_core_banking_regular_time_series_recorded)
- `--min-months`: Minimum months of history required (default: 3)
- `--rolling-window`: Rolling window for feature calculation (default: 12)
- `--min-transactions`: Minimum non-zero transactions to include series (default: 6)
- `--output-table`: Output table name
- `--save-to-file`: Save results to local CSV file
- `--verbose`: Enable verbose logging

### Forecasting Pipeline

#### Run All Methods (Comprehensive Evaluation)

Run ALL available forecasting methods and evaluate at all levels:

```bash
# Run all methods with comprehensive evaluation
radio-forecast forecast run-all \
  --segmented-table segmented_results \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --detailed \
  --verbose

# Run all methods with summary output only
radio-forecast forecast run-all \
  --segmented-table segmented_results \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --summary
```

This command will:
- Test ALL 12 forecasting methods
- Evaluate at 3 levels: dim_value, segment (credit/debit/net), overall
- Show comprehensive performance comparison
- Rank methods by multiple metrics (MAE, RMSE, MAPE, Directional Accuracy)
- Provide recommendations for best methods

#### Run Specific Methods

Run the forecasting pipeline with specific methods:

```bash
# Basic usage
python cli.py forecast run \
  --segmented-table segmented_results \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --method naive \
  --method moving_average \
  --method arima \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA

# Using a configuration file
python cli.py forecast run \
  --config example_configs/forecasting_config.json \
  --segmented-table segmented_results \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA

# Advanced usage with custom parameters
python cli.py forecast run \
  --segmented-table segmented_results \
  --train-start 2023-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-03-31 \
  --method naive \
  --method moving_average \
  --method arima \
  --method xgboost_individual \
  --horizon 3 \
  --min-history 18 \
  --output-prefix my_forecast_backtest \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --verbose
```

#### Forecasting Parameters

- `--config`: Path to JSON configuration file
- `--segmented-table`: Table with segmented data (required)
- `--train-start`: Training start date (YYYY-MM-DD) (required)
- `--train-end`: Training end date (YYYY-MM-DD) (required)
- `--test-start`: Test start date (YYYY-MM-DD) (required)
- `--test-end`: Test end date (YYYY-MM-DD) (required)
- `--method`: Forecasting methods to test (can be used multiple times)
- `--horizon`: Number of months to forecast ahead (default: 1)
- `--min-history`: Minimum months of history required (default: 12)
- `--output-prefix`: Prefix for output tables (default: forecast_backtest)
- `--save-forecasts/--no-save-forecasts`: Save individual forecasts (default: True)
- `--save-errors/--no-save-errors`: Save error metrics (default: True)
- `--verbose`: Enable verbose logging

#### Compare Methods

Compare forecasting methods from saved results:

```bash
# Compare methods from results table
radio-forecast forecast compare \
  --results-table forecast_backtest_all_errors \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA

# Compare with specific metric and show top 5
radio-forecast forecast compare \
  --results-table forecast_backtest_all_errors \
  --metric rmse \
  --top-n 5 \
  --by-level \
  --account YOUR_SNOWFLAKE_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD
```

#### Evaluation Levels Explained

The forecasting evaluation happens at three levels:

1. **dim_value level**: Individual counterparty_direction pairs (e.g., "company_a::IN", "company_b::OUT")
2. **segment level**: Aggregated by credit/debit/net:
   - **Credit**: All dim_values ending with "::IN" 
   - **Debit**: All dim_values ending with "::OUT"
   - **Net**: Credit - Debit
3. **overall level**: Total across all segments and dim_values

## Environment Variables

You can set Snowflake connection parameters as environment variables to avoid passing them as command line arguments.

### Option 1: Export Environment Variables
```bash
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_USER=your_username
export SNOWFLAKE_PASSWORD=your_password
export SNOWFLAKE_WAREHOUSE=your_warehouse
export SNOWFLAKE_DATABASE=your_database
export SNOWFLAKE_SCHEMA=your_schema
export SNOWFLAKE_ROLE=your_role
```

### Option 2: Use a .env File (Recommended)
Create a `.env` file in the project root directory:

```env
# Snowflake Connection Parameters
SNOWFLAKE_ACCOUNT=your_account_name
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_ROLE=your_role

# Optional: Set default clipping threshold
DAILY_AMOUNT_CLIP_THRESHOLD=1000.0

# Optional: Enable/disable clipping analysis
CLIP_ANALYSIS_ENABLED=true
```

The CLI will automatically load these variables from the `.env` file when it starts. Make sure to add `.env` to your `.gitignore` file to avoid committing sensitive credentials.

## Configuration Files

### Segmentation Configuration

Create a JSON file with segmentation parameters:

```json
{
  "min_months_history": 3,
  "rolling_window_months": 12,
  "min_transactions": 6,
  "start_date": "2022-01-01",
  "end_date": null,
  "source_table": "int__t__cad_core_banking_regular_time_series_recorded",
  "critical_volume_threshold": 100000000000,
  "high_volume_threshold": 5000000000,
  "medium_volume_threshold": 1000000000
}
```

### Forecasting Configuration

Create a JSON file with forecasting parameters:

```json
{
  "train_start_date": "2023-01-01",
  "train_end_date": "2023-12-31",
  "test_start_date": "2024-01-01",
  "test_end_date": "2024-03-31",
  "forecast_horizon": 1,
  "min_history_months": 12,
  "methods_to_test": ["naive", "moving_average", "arima"],
  "save_forecasts": true,
  "save_errors": true,
  "output_table_prefix": "forecast_backtest"
}
```

## Available Forecasting Methods

- `zero`: Always predicts zero
- `naive`: Uses the last observed value
- `moving_average`: Simple moving average
- `weighted_moving_average`: Weighted moving average with exponential decay
- `ets`: Exponential smoothing (Error, Trend, Seasonality)
- `arima`: AutoRegressive Integrated Moving Average
- `sarima`: Seasonal ARIMA
- `xgboost_individual`: XGBoost trained on individual time series
- `xgboost_global`: XGBoost trained on all time series
- `croston`: Croston's method for intermittent demand
- `ensemble`: Ensemble of multiple methods
- `segment_aggregate`: Forecast at segment level then distribute

## Complete Workflow Example

1. **Run Segmentation**:
```bash
python cli.py segmentation run \
  --config example_configs/segmentation_config.json \
  --account YOUR_ACCOUNT \
  --user YOUR_USER \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --output-table my_segmented_data \
  --verbose
```

2. **Run Forecasting**:
```bash
python cli.py forecast run \
  --config example_configs/forecasting_config.json \
  --segmented-table my_segmented_data \
  --account YOUR_ACCOUNT \
  --user YOUR_USER \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA \
  --verbose
```

## Script Installation

You can also install the CLI as a script using Poetry:

```bash
poetry install
```

Then use it directly:

```bash
radio-forecast --help
radio-forecast segmentation run --help
radio-forecast forecast run --help
```

## Error Handling

The CLI includes comprehensive error handling:

- Connection errors to Snowflake are caught and displayed clearly
- Configuration validation errors are shown with specific details
- Pipeline execution errors are logged with full context
- All errors include suggestions for resolution

## Logging

Use the `--verbose` flag to enable detailed logging:

```bash
python cli.py segmentation run --verbose [other options]
```

This will show:
- Detailed progress information
- Transformation step timing
- Row counts at each stage
- Performance metrics
- Debug information for troubleshooting
