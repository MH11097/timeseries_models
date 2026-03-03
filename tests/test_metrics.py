"""Tests for evaluation metrics."""

import numpy as np

from src.evaluation.metrics import evaluate_all, mae, mape, rmse, rmspe


def test_rmspe_basic():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = rmspe(y_true, y_pred)
    assert 0 < result < 1


def test_rmspe_filters_zeros():
    y_true = np.array([0, 100, 200])
    y_pred = np.array([10, 110, 190])
    result = rmspe(y_true, y_pred)
    assert result > 0


def test_rmse():
    y_true = np.array([100, 200])
    y_pred = np.array([100, 200])
    assert rmse(y_true, y_pred) == 0.0


def test_mae():
    y_true = np.array([100, 200])
    y_pred = np.array([110, 210])
    assert mae(y_true, y_pred) == 10.0


def test_mape():
    y_true = np.array([100, 200])
    y_pred = np.array([90, 210])
    result = mape(y_true, y_pred)
    assert 0 < result < 1


def test_evaluate_all():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = evaluate_all(y_true, y_pred)
    assert "rmspe" in result
    assert "rmse" in result
    assert "mae" in result
    assert "mape" in result
