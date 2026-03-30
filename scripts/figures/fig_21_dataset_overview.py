"""Hình minh họa Section 2.1 — Giới thiệu tập dữ liệu Rossmann.

Output:
  results/figures/21_store_type_distribution.png   — Phân bố loại cửa hàng (StoreType a/b/c/d)
  results/figures/21_assortment_distribution.png   — Phân bố mức đa dạng sản phẩm (Assortment a/b/c)
  results/figures/21_dataset_summary_table.png     — Bảng tổng quan dataset

Chạy:
    python scripts/figures/fig_21_dataset_overview.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.utils.config import load_config

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STYLE = dict(edgecolor="white", linewidth=0.5)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load data (không lọc Open, giữ nguyên toàn bộ) ──────────────────────────
print("Loading data...")
config = load_config()
data_cfg = config["data"]
raw_dir = Path(data_cfg["raw_dir"])

train_raw = pd.read_csv(raw_dir / data_cfg["train_file"], parse_dates=["Date"], low_memory=False)
store_df = pd.read_csv(raw_dir / data_cfg["store_file"])

n_rows = len(train_raw)
n_stores = train_raw["Store"].nunique()
date_min = train_raw["Date"].min()
date_max = train_raw["Date"].max()

# Merge để có StoreType, Assortment
merged = train_raw.merge(store_df, on="Store", how="left")

print(f"  {n_rows:,} rows, {n_stores} stores, {date_min.date()} → {date_max.date()}")

# ── Plot 1: StoreType distribution ───────────────────────────────────────────
print("\nPlotting...")

# Đếm số store (không phải số row) cho mỗi type
store_types = store_df["StoreType"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phân bố cửa hàng theo StoreType và Assortment", fontsize=13, fontweight="bold")

ax = axes[0]
colors_type = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
bars = ax.bar(store_types.index, store_types.values, color=colors_type[:len(store_types)], **STYLE)
for bar, val in zip(bars, store_types.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{val}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xlabel("StoreType")
ax.set_ylabel("Số cửa hàng")
ax.set_title(f"StoreType (tổng {n_stores} stores)")
ax.set_ylim(0, store_types.max() * 1.15)

# ── Plot 2: Assortment distribution ──────────────────────────────────────────
assortments = store_df["Assortment"].value_counts().sort_index()
assort_labels = {"a": "a (cơ bản)", "b": "b (mở rộng)", "c": "c (đầy đủ)"}

ax = axes[1]
colors_assort = ["#1abc9c", "#9b59b6", "#e67e22"]
bars = ax.bar(
    [assort_labels.get(k, k) for k in assortments.index],
    assortments.values,
    color=colors_assort[:len(assortments)],
    **STYLE,
)
for bar, val in zip(bars, assortments.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{val}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xlabel("Assortment")
ax.set_ylabel("Số cửa hàng")
ax.set_title(f"Assortment (tổng {n_stores} stores)")
ax.set_ylim(0, assortments.max() * 1.15)

fig.tight_layout()
_save(fig, "21_store_type_assortment.png")

# ── Plot 3: Dataset summary table ────────────────────────────────────────────
# Bảng tổng quan dữ liệu render thành hình
n_open = (merged["Open"] == 1).sum()
n_closed = (merged["Open"] == 0).sum()

summary_data = [
    ["Tổng số bản ghi", f"{n_rows:,}"],
    ["Số cửa hàng", f"{n_stores}"],
    ["Giai đoạn", f"{date_min.strftime('%d/%m/%Y')} — {date_max.strftime('%d/%m/%Y')}"],
    ["Ngày mở cửa (Open=1)", f"{n_open:,} ({n_open/n_rows*100:.1f}%)"],
    ["Ngày đóng cửa (Open=0)", f"{n_closed:,} ({n_closed/n_rows*100:.1f}%)"],
    ["Số biến train.csv", f"{len(train_raw.columns)}"],
    ["Số biến store.csv", f"{len(store_df.columns)}"],
    ["Sales trung bình (Open=1)", f"{merged[merged['Open']==1]['Sales'].mean():,.0f} €"],
    ["Sales trung vị (Open=1)", f"{merged[merged['Open']==1]['Sales'].median():,.0f} €"],
    ["Sales max", f"{merged['Sales'].max():,} €"],
]

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
table = ax.table(
    cellText=summary_data,
    colLabels=["Thông tin", "Giá trị"],
    cellLoc="left",
    loc="center",
    colWidths=[0.5, 0.4],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# Header style
for j in range(2):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Alternating row colors
for i in range(1, len(summary_data) + 1):
    for j in range(2):
        cell = table[i, j]
        cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

fig.suptitle("Tổng quan tập dữ liệu Rossmann Store Sales", fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout()
_save(fig, "21_dataset_summary_table.png")

print(f"\nDone. All figures saved to: {OUT_DIR.resolve()}")
