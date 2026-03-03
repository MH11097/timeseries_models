"""Visualization utilities for time series forecasting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    dates: pd.Series | None = None,
    save_path: str | None = None,
):
    """Plot actual vs predicted values."""
    # vẽ actual vs predicted trên cùng trục -> trực quan thấy model dự đoán sát hay lệch
    fig, ax = plt.subplots(figsize=(14, 5))
    x = dates if dates is not None else range(len(y_true))
    ax.plot(x, y_true, label="Actual", alpha=0.8)
    ax.plot(x, y_pred, label="Predicted", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date" if dates is not None else "Index")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmspe",
    x_col: str = "model_name",
    save_path: str | None = None,
):
    """Bar chart comparing models/experiments on a given metric."""
    # biểu đồ cột so sánh metric giữa các model -> sort tăng dần, model tốt nhất ở bên trái
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = results_df.sort_values(metric)
    sns.barplot(data=sorted_df, x=x_col, y=metric, ax=ax, palette="viridis")
    ax.set_title(f"Comparison - {metric.upper()}")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(metric.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: str | None = None,
):
    """Plot residual distribution and scatter."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter: residual vs predicted -> nếu có pattern (phễu, cong) thì model chưa bắt hết signal
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title} - Scatter")

    # histogram: phân phối residual -> lý tưởng là chuông đối xứng quanh 0
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title} - Distribution")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
