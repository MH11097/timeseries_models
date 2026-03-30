"""Hình minh họa Section 2.4 — Feature Engineering.

Output (results/figures/):
  24_lag_features.png            — Sales vs Sales_lag_7 cho 1 store (60 ngày)
  24_rolling_features.png        — Sales + rolling_mean_7 + rolling_mean_30 cho 1 store
  24_feature_correlation.png     — Heatmap tương quan top features vs Sales
  24_log_transform.png           — Phân phối Sales trước/sau log1p

Chạy:
    python scripts/figures/fig_24_features.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.loader import load_raw_data
from src.data.features import add_all_features, apply_log_transform
from src.utils.config import load_config

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# Store mẫu để minh họa lag/rolling (chọn store type C, có đủ dữ liệu)
SAMPLE_STORE = 262
WINDOW = 90  # số ngày hiển thị


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
config = load_config("xgboost")
df, _ = load_raw_data(config)
df = add_all_features(df)

# Lấy 1 store, chỉ phần train
train_end = config["split"]["train_end"]
store_df = df[(df["Store"] == SAMPLE_STORE) & (df["Date"] <= train_end)].sort_values("Date").copy()
# Lấy WINDOW ngày cuối để vẽ rõ
store_df = store_df.tail(WINDOW).reset_index(drop=True)

print(f"  Store {SAMPLE_STORE}: {len(store_df)} ngày | Features: {len(df.columns)} columns")
print("Plotting...\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Lag features — Sales vs Sales_lag_7
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(store_df["Date"], store_df["Sales"], linewidth=2, color="#2c3e50", label="Sales (hôm nay)", zorder=3)
if "Sales_lag_7" in store_df.columns:
    ax.plot(store_df["Date"], store_df["Sales_lag_7"], linewidth=1.5, color="#e74c3c",
            linestyle="--", alpha=0.8, label="Sales_lag_7 (tuần trước)")
if "Sales_lag_1" in store_df.columns:
    ax.plot(store_df["Date"], store_df["Sales_lag_1"], linewidth=1, color="#3498db",
            linestyle=":", alpha=0.6, label="Sales_lag_1 (hôm qua)")

ax.set_xlabel("Ngày")
ax.set_ylabel("Sales (€)")
ax.set_title(f"Lag Features — Store {SAMPLE_STORE} ({WINDOW} ngày cuối train)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
_save(fig, "24_lag_features.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Rolling features — Sales + rolling_mean_7 + rolling_mean_30
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(store_df["Date"], store_df["Sales"], linewidth=1.2, color="#bdc3c7",
        alpha=0.7, label="Sales (raw)")

if "Sales_rolling_mean_7" in store_df.columns:
    ax.plot(store_df["Date"], store_df["Sales_rolling_mean_7"], linewidth=2,
            color="#e74c3c", label="Rolling mean 7 ngày")
if "Sales_rolling_mean_30" in store_df.columns:
    ax.plot(store_df["Date"], store_df["Sales_rolling_mean_30"], linewidth=2,
            color="#2ecc71", label="Rolling mean 30 ngày")

# Vùng tô std
if "Sales_rolling_std_7" in store_df.columns and "Sales_rolling_mean_7" in store_df.columns:
    mean7 = store_df["Sales_rolling_mean_7"]
    std7 = store_df["Sales_rolling_std_7"]
    ax.fill_between(store_df["Date"], mean7 - std7, mean7 + std7,
                    alpha=0.1, color="#e74c3c", label="± 1 Std (7 ngày)")

ax.set_xlabel("Ngày")
ax.set_ylabel("Sales (€)")
ax.set_title(f"Rolling Statistics — Store {SAMPLE_STORE} (shift(1) tránh leakage)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
_save(fig, "24_rolling_features.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Feature correlation heatmap (top features vs Sales)
# ══════════════════════════════════════════════════════════════════════════════
# Tính correlation trên toàn bộ train set (sample để nhanh)
train_df = df[df["Date"] <= train_end].copy()
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
# Loại các cột không phải feature thực sự
exclude = {"Customers", "Open", "lag_364_valid", "Promo2SinceWeek", "Promo2SinceYear",
           "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"}
numeric_cols = [c for c in numeric_cols if c not in exclude and c != "Sales"]

# Tính correlation với Sales
corr_with_sales = train_df[numeric_cols + ["Sales"]].corr()["Sales"].drop("Sales").abs().sort_values(ascending=False)
top_features = corr_with_sales.head(15).index.tolist()

# Heatmap correlation matrix giữa top features
corr_matrix = train_df[top_features + ["Sales"]].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Ma trận tương quan — Top 15 features vs Sales", fontsize=13, fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
plt.setp(ax.get_yticklabels(), fontsize=9)
fig.tight_layout()
_save(fig, "24_feature_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Log transform effect
# ══════════════════════════════════════════════════════════════════════════════
sample_sales = train_df["Sales"].values
log_sales = np.log1p(sample_sales)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Log Transform: log1p(Sales) — Ổn định phương sai & gần Gaussian hơn",
             fontsize=13, fontweight="bold")

# Histogram raw
ax = axes[0, 0]
ax.hist(sample_sales, bins=80, color="#e74c3c", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_title(f"Sales (raw) — skewness={pd.Series(sample_sales).skew():.2f}")
ax.set_xlabel("Sales (€)")
ax.set_ylabel("Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Histogram log
ax = axes[0, 1]
ax.hist(log_sales, bins=80, color="#2ecc71", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_title(f"log1p(Sales) — skewness={pd.Series(log_sales).skew():.2f}")
ax.set_xlabel("log1p(Sales)")
ax.set_ylabel("Count")

# QQ-plot raw (dùng scatter vs normal quantiles)
ax = axes[1, 0]
from scipy import stats
stats.probplot(sample_sales[::10], dist="norm", plot=ax)
ax.set_title("Q-Q Plot — Sales (raw)")
ax.get_lines()[0].set_markersize(2)
ax.get_lines()[0].set_alpha(0.5)

# QQ-plot log
ax = axes[1, 1]
stats.probplot(log_sales[::10], dist="norm", plot=ax)
ax.set_title("Q-Q Plot — log1p(Sales)")
ax.get_lines()[0].set_markersize(2)
ax.get_lines()[0].set_alpha(0.5)

fig.tight_layout()
_save(fig, "24_log_transform.png")

print(f"\nDone. All feature engineering figures saved to: {OUT_DIR.resolve()}")
