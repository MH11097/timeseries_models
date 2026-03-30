"""Hình minh họa Section 2.3 — Tiền xử lý dữ liệu.

Output (results/figures/):
  23_train_test_split.png        — Timeline chia tập train/test theo thời gian
  23_encoding_example.png        — Ví dụ encoding biến phân loại (before/after)
  23_scaling_effect.png          — Phân phối feature trước/sau StandardScaler

Chạy:
    python scripts/figures/fig_23_preprocessing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_raw_data
from src.data.features import add_all_features
from src.utils.config import load_config

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
config = load_config("xgboost")
df, _ = load_raw_data(config)

print(f"  {len(df):,} rows, {df['Store'].nunique()} stores")
print("Plotting...\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Train/Test split timeline
# ══════════════════════════════════════════════════════════════════════════════
split_cfg = config["split"]
train_end = pd.Timestamp(split_cfg["train_end"])
test_start = pd.Timestamp(split_cfg["test_start"])
test_end = pd.Timestamp(split_cfg["test_end"])
data_start = df["Date"].min()
data_end = df["Date"].max()

# Tổng hợp Sales trung bình hàng ngày (tất cả stores)
daily_avg = df.groupby("Date")["Sales"].mean().reset_index()
daily_avg = daily_avg.sort_values("Date")

fig, ax = plt.subplots(figsize=(14, 5))

# Vẽ Sales trung bình
train_mask = daily_avg["Date"] <= train_end
test_mask = (daily_avg["Date"] >= test_start) & (daily_avg["Date"] <= test_end)

ax.plot(daily_avg[train_mask]["Date"], daily_avg[train_mask]["Sales"],
        color="#3498db", linewidth=0.8, alpha=0.7, label="Train period")
ax.plot(daily_avg[test_mask]["Date"], daily_avg[test_mask]["Sales"],
        color="#e74c3c", linewidth=1.2, alpha=0.9, label="Test period (30 ngày)")

# Vùng tô màu
ax.axvspan(data_start, train_end, alpha=0.08, color="#3498db")
ax.axvspan(test_start, test_end, alpha=0.15, color="#e74c3c")

# Đường phân cách
ax.axvline(train_end, color="#2c3e50", linestyle="--", linewidth=2, label=f"Train end: {train_end.strftime('%d/%m/%Y')}")
ax.axvline(test_start, color="#e74c3c", linestyle="--", linewidth=2, label=f"Test start: {test_start.strftime('%d/%m/%Y')}")

# Annotations
ax.annotate("TRAIN\n(~2.5 năm)", xy=(pd.Timestamp("2014-06-01"), daily_avg["Sales"].max() * 0.9),
            fontsize=14, fontweight="bold", color="#3498db", ha="center")
ax.annotate("TEST\n(30 ngày)", xy=(pd.Timestamp("2015-07-16"), daily_avg["Sales"].max() * 0.9),
            fontsize=14, fontweight="bold", color="#e74c3c", ha="center")

ax.set_xlabel("Ngày")
ax.set_ylabel("Sales trung bình (€)")
ax.set_title("Chia tập dữ liệu theo thời gian — Time-based Split", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
_save(fig, "23_train_test_split.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Encoding example (table)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
fig.suptitle("Encoding biến phân loại — Trước và Sau", fontsize=13, fontweight="bold")

# Before
before_data = [
    ["StateHoliday", "'0', 'a', 'b', 'c'", "str/int lẫn lộn"],
    ["StoreType", "'a', 'b', 'c', 'd'", "categorical"],
    ["Assortment", "'a', 'b', 'c'", "categorical"],
]
ax = axes[0]
ax.axis("off")
ax.set_title("Trước encoding", fontsize=11, fontweight="bold", color="#e74c3c")
table = ax.table(
    cellText=before_data,
    colLabels=["Biến", "Giá trị", "Kiểu"],
    cellLoc="center", loc="center", colWidths=[0.3, 0.4, 0.3],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
for j in range(3):
    table[0, j].set_facecolor("#e74c3c")
    table[0, j].set_text_props(color="white", fontweight="bold")

# After
after_data = [
    ["StateHoliday", "0, 1, 2, 3", "int (mapping cố định)"],
    ["StoreType", "0, 1, 2, 3", "int (label encoding)"],
    ["Assortment", "0, 1, 2", "int (label encoding)"],
]
ax = axes[1]
ax.axis("off")
ax.set_title("Sau encoding", fontsize=11, fontweight="bold", color="#2ecc71")
table = ax.table(
    cellText=after_data,
    colLabels=["Biến", "Giá trị", "Kiểu"],
    cellLoc="center", loc="center", colWidths=[0.3, 0.4, 0.3],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
for j in range(3):
    table[0, j].set_facecolor("#2ecc71")
    table[0, j].set_text_props(color="white", fontweight="bold")

fig.tight_layout()
_save(fig, "23_encoding_example.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Scaling effect (before/after StandardScaler)
# ══════════════════════════════════════════════════════════════════════════════
# Tạo features để có CompetitionDistance, lag features...
df_feat = add_all_features(df)

# Chọn 3 features với scale rất khác nhau
features_to_show = ["CompetitionDistance", "Sales_lag_1", "Sales_rolling_mean_7"]
available = [f for f in features_to_show if f in df_feat.columns]

fig, axes = plt.subplots(len(available), 2, figsize=(14, 4 * len(available)))
fig.suptitle("StandardScaler: Trước và Sau chuẩn hóa (z-score)", fontsize=13, fontweight="bold")

for i, col in enumerate(available):
    raw_vals = df_feat[col].dropna().values
    scaled_vals = StandardScaler().fit_transform(raw_vals.reshape(-1, 1)).flatten()

    # Sample để histogram nhanh hơn
    n_sample = min(200_000, len(raw_vals))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(raw_vals), n_sample, replace=False)

    # Before
    ax = axes[i, 0]
    ax.hist(raw_vals[idx], bins=60, color="#e74c3c", edgecolor="white", linewidth=0.3, alpha=0.8)
    ax.set_title(f"{col} — Raw (mean={raw_vals.mean():,.0f}, std={raw_vals.std():,.0f})", fontsize=10)
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # After
    ax = axes[i, 1]
    ax.hist(scaled_vals[idx], bins=60, color="#2ecc71", edgecolor="white", linewidth=0.3, alpha=0.8)
    ax.set_title(f"{col} — Scaled (mean≈0, std≈1)", fontsize=10)
    ax.set_ylabel("Count")

fig.tight_layout()
_save(fig, "23_scaling_effect.png")

print(f"\nDone. All preprocessing figures saved to: {OUT_DIR.resolve()}")
