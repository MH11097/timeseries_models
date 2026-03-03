"""Evaluation metrics for time series forecasting."""

import numpy as np


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error (Kaggle competition metric).

    Filters out zero actual values to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    # RMSPE chia cho y_true -> y_true=0 gây chia 0, phải loại bỏ trước khi tính
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    # metric chính của Kaggle Rossmann -> dùng để so sánh model và report kết quả
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Filters zero actual values."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all metrics and return as dict."""
    # tính đồng thời 4 metric -> nhìn đa chiều: RMSPE (Kaggle), RMSE (scale gốc), MAE (trực quan), MAPE (%)
    return {
        "rmspe": round(rmspe(y_true, y_pred), 6),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 6),
    }
