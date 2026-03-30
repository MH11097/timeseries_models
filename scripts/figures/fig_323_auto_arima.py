"""Hình minh họa Section 3.2.1 — Bước 3: Xác nhận bằng Auto-ARIMA (AIC stepwise).

Chạy pmdarima auto_arima trên từng cửa hàng đại diện → kiểm chứng phạm vi p, q.
Kết quả mong đợi: orders đa dạng, xác nhận cần grid search thay vì 1 cấu hình cố định.

Output (results/figures/):
  323_auto_arima_table.png  — Bảng kết quả: Store | Order (p,d,q) | AIC

Chạy:
    python scripts/figures/fig_323_auto_arima.py
"""

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_STORES = 30  # số store chạy auto_arima


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== Bước 3: Auto-ARIMA Discovery (AIC stepwise) ===\n")

config = load_config("arima")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, _, _, _ = preprocess(df, config)

stores = sorted(train_df["Store"].unique())[:N_STORES]
print(f"Chạy auto_arima trên {len(stores)} stores (max_p=5, max_d=2, max_q=5)...\n")

# ── Chạy auto_arima per store ─────────────────────────────────────────────────
# mỗi store có pattern riêng → order tối ưu có thể khác nhau đáng kể
results = []
order_counter = Counter()

for i, sid in enumerate(stores, 1):
    store_data = train_df[train_df["Store"] == sid].sort_values("Date")
    sales = store_data["Sales"].values.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                sales,
                max_p=5, max_d=2, max_q=5,
                seasonal=False, stepwise=True,
                information_criterion="aic",
                suppress_warnings=True, error_action="ignore",
            )
            order = model.order
            aic_val = model.aic()
            order_counter[order] += 1
            results.append({"Store": int(sid), "Order (p,d,q)": f"({order[0]}, {order[1]}, {order[2]})",
                            "AIC": round(aic_val, 2)})
            print(f"  [{i:2d}/{len(stores)}] Store {sid}: order={order}, AIC={aic_val:.2f}")
    except Exception as e:
        print(f"  [{i:2d}/{len(stores)}] Store {sid}: FAILED — {e}")
        results.append({"Store": int(sid), "Order (p,d,q)": "FAILED", "AIC": None})

# ── Tổng hợp kết quả ─────────────────────────────────────────────────────────
valid = [r for r in results if r["AIC"] is not None]
print(f"\nThành công: {len(valid)}/{len(stores)} stores")
print(f"Số orders khác nhau: {len(order_counter)}")
print(f"\nPhân phối order (top 5):")
for order, count in order_counter.most_common(5):
    pct = count / len(valid) * 100
    print(f"  {order}: {count} stores ({pct:.0f}%)")
print("Kết luận: Orders đa dạng → cần grid search mở rộng ±1 quanh top orders.")

# ── Lưu CSV ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
csv_path = OUT_DIR / "323_auto_arima_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"  CSV: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot: Bảng kết quả auto_arima
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = [[r["Store"], r["Order (p,d,q)"], r["AIC"] if r["AIC"] else "—"] for r in results]
col_labels = ["Store", "Order (p, d, q)", "AIC"]

fig_height = max(6, len(table_data) * 0.35 + 1.5)
fig, ax = plt.subplots(figsize=(8, fig_height))
ax.axis("off")

table = ax.table(
    cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
    colWidths=[0.2, 0.4, 0.3],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

# header style
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# xen kẽ màu nền cho dễ đọc
for i in range(len(table_data)):
    color = "#f8f9fa" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

fig.suptitle(f"Kết quả Auto-ARIMA (AIC stepwise) — {len(stores)} stores type C",
             fontsize=12, fontweight="bold", y=0.98)
_save(fig, "323_auto_arima_table.png")

print(f"\nDone. Auto-ARIMA figures saved to: {OUT_DIR.resolve()}")
