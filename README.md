# Radio Forecast Segmentation - Backtesting System

A comprehensive backtesting framework for segmented time series forecasting models with support for multiple model types, dynamic segment detection, and flexible combination generation.

## 🚀 Features

### **Model Types**
- **Time Series Models**: ARIMA, Moving Average, Null
- **Transformed Models**: Direction-aware (credit/debit), Net-aware (net value)
- **Machine Learning Models**: XGBoost, Random Forest
- **Segmented Forecasting**: Different models for different data segments

### **Backtesting Capabilities**
- **Dynamic Segment Detection**: Automatically detects segments from data
- **Comprehensive Combinations**: Generates all possible model-segment combinations
- **Fixed Segments**: Predefine models for specific segments
- **Sampling**: Test a subset of combinations for faster iteration
- **Chronological Splits**: Train/Validation/Test splits for robust evaluation
- **JSON Caching**: Skip already computed combinations

### **Evaluation Metrics**
- **Overall Metrics**: MAE, MSE, RMSE, MAPE
- **Segment-level Metrics**: Individual segment performance
- **Net/Credit/Debit Metrics**: Detailed breakdown by transaction type

## 📁 Project Structure

```
radio-forecast-segmentation/
├── src/
│   ├── new_forecast/
│   │   ├── models/           # Individual forecast models
│   │   │   ├── arima.py
│   │   │   ├── moving_average.py
│   │   │   ├── null.py
│   │   │   ├── xgboost.py
│   │   │   ├── random_forest.py
│   │   │   └── ...
│   │   └── segmented.py      # Segmented forecast model
│   └── forecast/
│       └── evaluate.py       # Evaluation functions
├── dataset/
│   └── feature_df.csv        # Sample data
├── outputs/
│   └── backtest_results.json # Backtest results
├── run_backtest.py           # Main backtesting script
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd radio-forecast-segmentation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # or if using poetry
   poetry install
   ```

3. **Activate virtual environment** (if using poetry):
   ```bash
   poetry shell
   ```

## 🚀 Quick Start

### **Basic Usage**

Run a quick backtest with default settings:
```bash
python run_backtest.py
```

### **Custom Configuration**

```bash
python run_backtest.py \
    --sample-size 50 \
    --min-iterations 5 \
    --test-size 2 \
    --val-size 2 \
    --data dataset/feature_df.csv \
    --output outputs/my_results.json
```

### **Fixed Segments**

Predefine models for specific segments:
```bash
python run_backtest.py \
    --fix-segments '{"CONTINUOUS_STABLE": {"name": "arima", "order": [1,1,1]}, "NO_EOM": {"name": "null"}}'
```

## 📊 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Path to feature data CSV file | `dataset/feature_df.csv` |
| `--output` | Path to output results JSON file | `outputs/backtest_results.json` |
| `--config` | Path to custom configuration JSON file | None |
| `--min-iterations` | Minimum number of backtest iterations | 3 |
| `--forecast-horizon` | Forecast horizon | 1 |
| `--input-steps` | Number of input steps for training | 6 |
| `--test-size` | Size of test set for evaluation | 1 |
| `--val-size` | Size of validation set for evaluation | 1 |
| `--sample-size` | Number of combinations to sample (0 = all) | 100 |
| `--fix-segments` | JSON string defining fixed segments | See below |

## 🔧 Configuration

### **Fixed Segments**

The system comes with default fixed segments:
- `NO_EOM` → `null` model
- `RARE_STALE` → `null` model
- `AGGREGATED_OTHERS` → `null` model

You can override these or add new ones:
```bash
--fix-segments '{"SEGMENT_NAME": {"name": "model_name", "param": "value"}}'
```

### **Model Parameters**

Different models support different parameters:

**ARIMA**:
```json
{"name": "arima", "order": [1, 1, 1]}
```

**Moving Average**:
```json
{"name": "moving_average", "window": 3}
```

**XGBoost**:
```json
{"name": "xgboost", "n_estimators": 100, "max_depth": 6}
```

**Random Forest**:
```json
{"name": "random_forest", "n_estimators": 100, "max_depth": 10}
```

## 📈 Understanding Results

### **Output Format**

Results are saved in JSON format with the following structure:
```json
{
  "combination_name": {
    "test_mae": 30636.84,
    "val_mae": 525.94,
    "test_mse": 1234567.89,
    "val_mse": 12345.67,
    "test_rmse": 1111.11,
    "val_rmse": 111.11,
    "test_mape": 0.15,
    "val_mape": 0.02,
    "config": {...},
    "description": "...",
    "timestamp": "2025-10-06T13:23:29"
  }
}
```

### **Performance Ranking**

The system automatically ranks models by Test MAE and displays the top performers:
```
🏆 BEST PERFORMING MODELS (by Test MAE):
  1. all_moving_average: Test MAE=30636.84, Val MAE=525.94
  2. all_arima: Test MAE=30706.33, Val MAE=537.60
  3. all_null: Test MAE=31570.27, Val MAE=644.45
```

## 🔍 Troubleshooting

### **Common Issues**

1. **"['date'] not in index" Error**:
   - This occurs with direction-aware and net-aware models
   - The data format may not be compatible with these models
   - These models expect `dim_value` format like `"account_1::IN"` but data has `"series_011"`

2. **"Model must be fitted before prediction" Error**:
   - Some models fail to fit due to insufficient data
   - This is normal behavior for models that can't learn from the available data

3. **Categorical Data Warnings**:
   - XGBoost and Random Forest models handle categorical data automatically
   - Warnings about categorical columns are normal and don't affect functionality

### **Data Requirements**

The system expects a CSV file with the following columns:
- `date`: Date column (will be converted to datetime)
- `eom_pattern_primary`: Segment column
- `dim_value`: Dimension column (for direction/net models)
- `target`: Target variable to forecast

## 🧪 Examples

### **Example 1: Quick Test**
```bash
python run_backtest.py --sample-size 5 --min-iterations 1
```

### **Example 2: Full Backtest**
```bash
python run_backtest.py --sample-size 0 --min-iterations 10
```

### **Example 3: Custom Fixed Segments**
```bash
python run_backtest.py \
    --fix-segments '{"CONTINUOUS_STABLE": {"name": "arima"}, "RARE_STALE": {"name": "null"}}' \
    --sample-size 20
```

### **Example 4: High Iterations**
```bash
python run_backtest.py \
    --min-iterations 20 \
    --test-size 3 \
    --val-size 3 \
    --sample-size 50
```

## 📚 Model Details

### **Time Series Models**
- **ARIMA**: AutoRegressive Integrated Moving Average
- **Moving Average**: Simple moving average with configurable window
- **Null**: Baseline model that always predicts zero

### **Transformed Models**
- **Direction-aware**: Splits data by IN/OUT direction and forecasts separately
- **Net-aware**: Calculates net value (IN - OUT) and forecasts it

### **Machine Learning Models**
- **XGBoost**: Gradient boosting with automatic feature engineering
- **Random Forest**: Ensemble method with automatic feature engineering

### **Segmented Model**
- **NewSegmentedForecastModel**: Applies different models to different segments
- Uses Pydantic discriminator pattern for model registration
- Supports groupby operations for efficient processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the command-line help: `python run_backtest.py --help`
3. Open an issue on GitHub

## 🎯 Roadmap

- [ ] Support for more model types
- [ ] Advanced feature engineering
- [ ] Real-time forecasting
- [ ] Model persistence and deployment
- [ ] Web interface for results visualization
