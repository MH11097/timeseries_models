"""Hình minh họa Section 3.2.1 — Bước 1: Kiểm định tính dừng (ADF + KPSS).

Chạy ADF + KPSS trên toàn bộ 148 cửa hàng loại C → xác định bậc sai phân d.
Kết quả mong đợi: ~54.7% cần d=1, ~44.6% đã dừng (d=0), ~0.7% cần d=2.

Output (results/figures/):
  321_stationarity_pie.png    — Biểu đồ tròn phân bố d gợi ý
  321_stationarity_table.png  — Bảng tổng hợp ADF/KPSS cho 10 stores đại diện

Chạy:
    python scripts/figures/fig_321_stationarity.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.stationarity import stationarity_summary, test_stationarity
from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== Bước 1: Kiểm định tính dừng ADF/KPSS ===\n")

config = load_config("arima")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)  # 148 stores type C
df = add_all_features(df)

# chỉ dùng phần train → không leak val/test vào phân tích
train_end = config["split"]["train_end"]
train_df = df[df["Date"] <= train_end].copy()

stores = sorted(train_df["Store"].unique())
print(f"Số cửa hàng type C: {len(stores)}")

# ── Chạy kiểm định ADF + KPSS cho từng store ─────────────────────────────────
print("Đang chạy kiểm định tính dừng...")
results = []
for sid in stores:
    sales = train_df[train_df["Store"] == sid].sort_values("Date")["Sales"].values.astype(float)
    res = test_stationarity(sales, store_id=sid)
    results.append(res)

summary_df = stationarity_summary(results)

# ── Thống kê phân bố d ───────────────────────────────────────────────────────
d_counts = summary_df["suggested_d"].value_counts().sort_index()
d_pcts = (d_counts / len(summary_df) * 100).round(1)
print(f"\nPhân bố bậc sai phân gợi ý (n={len(summary_df)}):")
for d_val in sorted(d_counts.index):
    print(f"  d={d_val}: {d_counts[d_val]} stores ({d_pcts[d_val]}%)")
print("Kết luận: Chọn d=1 làm cấu hình chung cho toàn bộ 148 cửa hàng.")

# ── Lưu CSV ───────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "321_stationarity_summary.csv"
summary_df.to_csv(csv_path, index=False)
print(f"  CSV: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Biểu đồ tròn — phân bố d
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

colors_map = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}  # xanh lá, xanh dương, đỏ
labels, sizes, colors = [], [], []
for d_val in sorted(d_counts.index):
    cnt = d_counts[d_val]
    pct = d_pcts[d_val]
    labels.append(f"d={d_val}\n{cnt} stores ({pct}%)")
    sizes.append(cnt)
    colors.append(colors_map.get(d_val, "#95a5a6"))

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, autopct="", startangle=90,
    textprops={"fontsize": 11}, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
ax.set_title("Phân bố bậc sai phân gợi ý — 148 cửa hàng type C\n(Kiểm định kép ADF + KPSS)",
             fontsize=13, fontweight="bold", pad=15)
_save(fig, "321_stationarity_pie.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Bảng thống kê ADF/KPSS cho 10 stores đại diện
# ══════════════════════════════════════════════════════════════════════════════
# chọn stores đại diện: 5 store d=0 đầu + 5 store d=1 đầu (theo store_id)
sample_d0 = summary_df[summary_df["suggested_d"] == 0].head(5)
sample_d1 = summary_df[summary_df["suggested_d"] == 1].head(5)
sample = pd.concat([sample_d0, sample_d1]).sort_values("store_id").reset_index(drop=True)

table_data = []
for _, row in sample.iterrows():
    table_data.append([
        int(row["store_id"]),
        f"{row['adf_stat']:.3f}",
        f"{row['adf_pvalue']:.4f}",
        "Có" if row["adf_stationary"] else "Không",
        f"{row['kpss_stat']:.3f}",
        f"{row['kpss_pvalue']:.4f}",
        "Có" if row["kpss_stationary"] else "Không",
        int(row["suggested_d"]),
    ])

col_labels = ["Store", "ADF stat", "ADF p-val", "ADF dừng?",
              "KPSS stat", "KPSS p-val", "KPSS dừng?", "d"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

table = ax.table(
    cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
    colWidths=[0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# header style — nền tối, chữ trắng
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# tô màu theo d: xanh lá cho d=0, xanh dương cho d=1
for i, (_, row) in enumerate(sample.iterrows()):
    color = "#eafaf1" if row["suggested_d"] == 0 else "#ebf5fb"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

fig.suptitle("Kết quả kiểm định ADF/KPSS — 10 stores đại diện",
             fontsize=12, fontweight="bold", y=0.95)
_save(fig, "321_stationarity_table.png")

print(f"\nDone. Stationarity figures saved to: {OUT_DIR.resolve()}")
