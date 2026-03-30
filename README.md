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

### Train tất cả models

```bash
python scripts/train.py --model all
```

### Train từng model

Mỗi model có config YAML riêng trong `configs/`. Override bất kỳ param nào qua `--set key=value`.

#### ARIMA

Per-store univariate, order (p,d,q) từ ACF/PACF analysis.

```bash
# Train với config mặc định (order=[1,1,1], trend='t', store_type='c')
python scripts/train.py --model arima

# Override order & trend
python scripts/train.py --model arima --set model.order=[2,1,1] --set model.trend=c
```

#### SARIMAX

Per-store với seasonal component + exogenous regressors.

```bash
# Train mặc định (order=[1,1,1], seasonal_order=[1,0,1,7], exog=[Promo, SchoolHoliday, StateHoliday])
python scripts/train.py --model sarimax

# Override seasonal order & maxiter
python scripts/train.py --model sarimax --set model.seasonal_order=[1,1,1,7] --set model.maxiter=300
```

#### Prophet

Per-store với German holidays + regressors (Promo, SchoolHoliday).

```bash
# Train mặc định (multiplicative seasonality, changepoint_prior_scale=0.04)
python scripts/train.py --model prophet

# Override seasonality mode & prior scales
python scripts/train.py --model prophet \
  --set model.seasonality_mode=additive \
  --set model.changepoint_prior_scale=0.1 \
  --set model.seasonality_prior_scale=1.0
```

#### XGBoost

Global model, Store là feature, 18 features sau ablation study.

```bash
# Train mặc định (n_estimators=500, max_depth=9, lr=0.05, early_stopping=50)
python scripts/train.py --model xgboost

# Override hyperparams
python scripts/train.py --model xgboost \
  --set model.max_depth=7 \
  --set model.learning_rate=0.03 \
  --set model.n_estimators=1000
```

#### RNN

Global sequence model, vanilla RNN cells.

```bash
# Train mặc định (hidden=64, layers=2, seq_len=30, epochs=50, H=1)
python scripts/train.py --model rnn

# Override architecture & forecast horizon
python scripts/train.py --model rnn \
  --set model.hidden_size=128 \
  --set model.num_layers=1 \
  --set model.seq_len=21 \
  --set model.forecast_horizon=30 \
  --set forecast_strategy=multioutput \
  --set use_log_sales=true
```

#### LSTM

Global sequence model, LSTM cells (better long-range dependencies).

```bash
# Train mặc định (hidden=64, layers=1, seq_len=21, epochs=100, H=30, log_sales, multioutput)
python scripts/train.py --model lstm

# Override training params
python scripts/train.py --model lstm \
  --set model.hidden_size=128 \
  --set model.learning_rate=0.001 \
  --set model.dropout=0.3 \
  --set model.patience=20
```

### Cross-Validation

Walk-forward CV: train trên quá khứ, test trên tương lai gần — mô phỏng đúng thực tế forecasting.

```bash
# Expanding window CV (train luôn bắt đầu từ ngày đầu tiên, mở rộng dần)
python scripts/evaluate.py --model arima --cv expanding --n-splits 5

# Sliding window CV (train chỉ lấy window gần nhất, cùng kích thước)
python scripts/evaluate.py --model xgboost --cv sliding --n-splits 5

# Giới hạn test mỗi fold chỉ 30 ngày đầu (eval_days)
python scripts/evaluate.py --model lstm --cv expanding --n-splits 3 --eval-days 30

# CV cho từng model khác
python scripts/evaluate.py --model sarimax --cv expanding --n-splits 3
python scripts/evaluate.py --model prophet --cv expanding --n-splits 3
python scripts/evaluate.py --model rnn --cv expanding --n-splits 3
```

> Kết quả CV lưu tại `results/{model}/cv/cv_results.json` gồm per-fold metrics + aggregated mean/std.

### So sánh models (Cross-Validation)

Chạy CV cho nhiều models cùng lúc, tạo bảng so sánh sorted by RMSPE.

```bash
# So sánh tất cả 6 models (3 folds, 30 eval days)
python scripts/compare_cv.py

# Chỉ so sánh subset models
python scripts/compare_cv.py --models arima,xgboost,lstm

# Tuỳ chỉnh CV params
python scripts/compare_cv.py --models arima,sarimax,prophet --n-splits 5 --eval-days 30 --cv expanding
```

> Kết quả lưu tại `results/comparison/` gồm `cv_comparison.csv`, `.json`, `.md`.

### Đánh giá model đã train

Load model pickle đã train trước đó, evaluate trên val/test set mà không cần train lại.

```bash
# Evaluate run mới nhất của model
python scripts/evaluate.py --model lstm

# Evaluate 1 run cụ thể
python scripts/evaluate.py --model xgboost --run-dir results/xgboost/250320_123456_md9_ne500_lr0.05
```

### Grid Search

Tìm hyperparams tối ưu bằng grid search qua tất cả tổ hợp.

```bash
# Grid search bằng inline JSON
python scripts/train.py grid-search --model xgboost \
  --grid '{"model.max_depth":[7,9,11],"model.learning_rate":[0.03,0.05,0.1]}'

# Grid search từ file JSON (khuyến nghị với grid phức tạp)
python scripts/train.py grid-search --model lstm \
  --grid-file configs/grid_lstm.json \
  --set use_log_sales=true

# Dry run — chỉ in danh sách tổ hợp, không train
python scripts/train.py grid-search --model rnn \
  --grid '{"model.hidden_size":[32,64,128],"model.num_layers":[1,2]}' \
  --dry-run
```

### Interactive mode

Không truyền `--model` → vào chế độ tương tác: chọn model + tham số bằng menu.

```bash
python scripts/train.py
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
