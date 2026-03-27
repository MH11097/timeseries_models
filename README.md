# Time Series Forecasting Model Comparison

Comparative framework for 6 time series forecasting models on the **Rossmann Store Sales** dataset (Kaggle).

## Models

| Model | Type | Fitting Strategy | Store Scope |
|-------|------|------------------|-------------|
| ARIMA | Statistical | Per-store | Type C (148 stores) |
| SARIMAX | Statistical | Per-store | Type C (148 stores) |
| Prophet | Statistical | Per-store | Type C (148 stores) |
| XGBoost | ML | Global (Store as feature) | All (1,115 stores) |
| RNN | Deep Learning | Global (Store as embedding) | All (1,115 stores) |
| LSTM | Deep Learning | Global (Store as embedding) | All (1,115 stores) |

> **Store type "c":** 3 per-store models chỉ train trên 148 stores type C để kiểm soát biến —
> cùng loại cửa hàng, cùng đặc tính kinh doanh → so sánh thuần model vs model.
> Global models (XGBoost, RNN, LSTM) vẫn dùng toàn bộ 1,115 stores vì chúng encode Store như feature.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) - 1,115 stores, ~1M rows, 2013-2015.

### Download

```bash
# Option 1: Kaggle CLI
kaggle competitions download -c rossmann-store-sales -p data/raw/
unzip data/raw/rossmann-store-sales.zip -d data/raw/

# Option 2: Manual
# Download from https://www.kaggle.com/c/rossmann-store-sales/data
# Place train.csv, store.csv, test.csv in data/raw/
```

## Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv /root/.venvs/timeseries_models --python 3.11
uv sync

# Activate
source /root/.venvs/timeseries_models/bin/activate
```

## Usage

### Train a model
```bash
python scripts/train.py --model arima
python scripts/train.py --model xgboost
python scripts/train.py --all
```

### Evaluate
```bash
python scripts/evaluate.py --model arima
python scripts/evaluate.py --model arima --cv expanding --n-splits 5
```

### Compare models
```bash
python scripts/compare.py
```

## Project Structure

```
timeseries_models/
├── configs/             # YAML configs (base + per-model)
├── data/raw/            # Dataset files (gitignored)
├── src/
│   ├── data/            # Data loading, preprocessing, features
│   ├── models/          # BaseModel ABC + 6 model implementations
│   ├── evaluation/      # Metrics, comparison, cross-validation
│   └── utils/           # Config, seed, visualization
├── notebooks/           # EDA, comparison, experiment templates
├── results/             # Experiment outputs (gitignored)
├── scripts/             # CLI entry points (train, evaluate, compare)
├── tests/               # Unit tests
└── docs/                # Project documentation
```

## How to Add a New Model

1. Create `src/models/your_model.py` extending `BaseModel`
2. Implement `train()`, `predict()`, `evaluate()`, `save()`, `load()`
3. Create `configs/your_model.yaml` with model-specific params
4. Register in model registry (`src/models/__init__.py`)
5. Run: `python scripts/train.py --model your_model`

## Evaluation

- **Primary metric:** RMSPE (Kaggle competition metric)
- **Secondary:** RMSE, MAE, MAPE
- **Cross-validation:** Walk-forward expanding window
