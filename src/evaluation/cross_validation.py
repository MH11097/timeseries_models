"""Walk-forward cross-validation for time series models."""

import time

import numpy as np
import pandas as pd

from src.data.preprocessor import (
    _encode_categoricals,
    _get_numeric_feature_cols,
    _handle_missing_values,
)
from sklearn.preprocessing import StandardScaler


def walk_forward_cv(
    model_class,
    config: dict,
    df: pd.DataFrame,
    n_splits: int = 5,
    expanding: bool = True,
    eval_days: int = None,
) -> dict:
    """Walk-forward expanding/sliding window cross-validation.

    Args:
        model_class: BaseModel subclass to instantiate per fold
        config: Model config dict
        df: Full DataFrame sorted by Date
        n_splits: Number of CV folds
        expanding: If True, expanding window; else sliding window
        eval_days: Limit evaluation to first N days of each test fold (default: None)

    Returns:
        Dict with per-fold metrics and aggregated mean/std
    """
    from src.evaluation.metrics import evaluate_all

    # sắp xếp tất cả ngày unique -> chia thành n_splits fold theo thứ tự thời gian
    dates = sorted(df["Date"].unique())
    total = len(dates)
    step = total // (n_splits + 1)

    fold_metrics = []
    for fold in range(n_splits):
        if expanding:
            # expanding: train từ đầu đến cutoff -> mô phỏng thực tế khi dữ liệu tích luỹ dần
            train_end_idx = step * (fold + 1)
            train_dates = dates[:train_end_idx]
        else:
            # sliding: train chỉ lấy cửa sổ gần nhất -> phù hợp khi pattern thay đổi theo thời gian
            train_start_idx = step * fold
            train_end_idx = step * (fold + 1)
            train_dates = dates[train_start_idx:train_end_idx]

        # test luôn là giai đoạn ngay sau train -> đảm bảo không có data leakage từ tương lai
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + step, total)
        test_dates = dates[test_start_idx:test_end_idx]

        if len(test_dates) == 0:
            continue

        train_df = df[df["Date"].isin(train_dates)].copy()
        test_df = df[df["Date"].isin(test_dates)].copy()

        # giới hạn theo số ngày unique, không phải số dòng
        # test_df = test_df.head(eval_days)
        # head(N) chỉ lấy N dòng đầu = 1 store duy nhất do sort (Store, Date)
        if eval_days is not None and eval_days > 0:
            first_n_dates = sorted(test_df["Date"].unique())[:eval_days]
            test_df = test_df[test_df["Date"].isin(first_n_dates)]
            # test_df = test_df.head(eval_days)

        # preprocessing per-fold để tránh data leakage:
        # - handle missing values và encode categoricals cho cả train/test
        # - fit scaler chỉ trên train_df, rồi transform cả 2 set
        train_df = _handle_missing_values(train_df)
        test_df = _handle_missing_values(test_df)
        train_df = _encode_categoricals(train_df)
        test_df = _encode_categoricals(test_df)

        # tree-based models (XGBoost) không cần scaling → đọc skip_scaling từ config
        # reuse cùng pattern với preprocessor.py:40
        skip_scaling = config.get("model", {}).get("skip_scaling", False)
        numeric_cols = _get_numeric_feature_cols(train_df)
        if numeric_cols and not skip_scaling:
            scaler = StandardScaler()
            train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
            test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

        model = model_class(config)
        start = time.time()
        model.train(train_df)
        train_time = time.time() - start

        predictions = model.predict(test_df)
        y_true = test_df["Sales"].values
        # khi use_log_sales=True: Sales trong test_df đang ở log1p space,
        # nhưng model.predict() đã expm1 về scale gốc → cần inverse y_true để so sánh cùng scale
        if config.get("use_log_sales", False):
            y_true = np.expm1(y_true.astype(float))
        metrics = evaluate_all(y_true, predictions)
        metrics["fold"] = fold
        metrics["training_time_seconds"] = round(train_time, 2)
        fold_metrics.append(metrics)

    # tổng hợp mean/std qua các fold -> mean cho biết hiệu suất trung bình, std cho biết mức ổn định
    metric_keys = ["rmspe", "rmse", "mae", "mape"]
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = round(float(np.mean(values)), 6)
            aggregated[f"{key}_std"] = round(float(np.std(values)), 6)

    return {
        "folds": fold_metrics,
        "aggregated": aggregated,
        "n_splits": len(fold_metrics),
    }
