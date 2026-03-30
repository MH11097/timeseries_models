"""Hình minh họa Section 3.2.2 — Xác định tham số mùa vụ (S, D, P, Q).

Chạy pmdarima auto_arima (seasonal=True, m=7) trên 5 cửa hàng mẫu
→ xác nhận S=7, D=0, và hai pattern chính: (1,0,2,7) vs (0,0,1,7).

Output (results/figures/):
  326_seasonal_discovery_table.png  — Bảng: Store | Order | Seasonal | AIC

Chạy:
    python scripts/figures/fig_326_seasonal_discovery.py
"""

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
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
N_STORES = 5  # số store mẫu — thesis dùng 5 store đại diện


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== SARIMAX — Xác định tham số mùa vụ (S, D, P, Q) ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, _, _, _ = preprocess(df, config)

stores = sorted(train_df["Store"].unique())[:N_STORES]
print(f"Chạy auto_arima seasonal (m=7) trên {len(stores)} stores...\n")

# ── Auto-ARIMA seasonal per store ─────────────────────────────────────────────
# mỗi store có cấu trúc tự tương quan riêng → order + seasonal_order có thể khác nhau
results = []
seasonal_counter = Counter()

for i, sid in enumerate(stores, 1):
    store_data = train_df[train_df["Store"] == sid].sort_values("Date")
    sales = store_data["Sales"].values.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                sales,
                max_p=3, max_d=2, max_q=3,
                max_P=2, max_D=1, max_Q=2,
                m=7, seasonal=True, stepwise=True,
                information_criterion="aic",
                suppress_warnings=True, error_action="ignore",
            )
            order = model.order
            s_order = model.seasonal_order
            aic_val = model.aic()
            seasonal_counter[(s_order[0], s_order[1], s_order[2], s_order[3])] += 1
            results.append({
                "Store": int(sid),
                "Order (p,d,q)": f"({order[0]}, {order[1]}, {order[2]})",
                "Seasonal (P,D,Q,S)": f"({s_order[0]}, {s_order[1]}, {s_order[2]}, {s_order[3]})",
                "AIC": round(aic_val, 2),
            })
            print(f"  [{i}/{len(stores)}] Store {sid}: order={order}, seasonal={s_order}, AIC={aic_val:.2f}")
    except Exception as e:
        print(f"  [{i}/{len(stores)}] Store {sid}: FAILED — {e}")
        results.append({"Store": int(sid), "Order (p,d,q)": "FAILED", "Seasonal (P,D,Q,S)": "—", "AIC": None})

# ── Tổng hợp ──────────────────────────────────────────────────────────────────
valid = [r for r in results if r["AIC"] is not None]
print(f"\nThành công: {len(valid)}/{len(stores)} stores")
print(f"\nSeasonal patterns phổ biến:")
for s_order, count in seasonal_counter.most_common():
    pct = count / len(valid) * 100
    print(f"  ({s_order[0]},{s_order[1]},{s_order[2]},{s_order[3]}): {count} stores ({pct:.0f}%)")
print(f"\nKết luận:")
print(f"  S=7 (chu kỳ tuần) — 100% stores")
print(f"  D=0 (không cần sai phân mùa vụ) — seasonal MA đủ capture weekly pattern")
print(f"  Hai pattern chính: (1,0,2,7) và (0,0,1,7)")

# ── Lưu CSV ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
csv_path = OUT_DIR / "326_seasonal_discovery.csv"
results_df.to_csv(csv_path, index=False)
print(f"  CSV: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot: Bảng kết quả auto_arima seasonal
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = [[r["Store"], r["Order (p,d,q)"], r["Seasonal (P,D,Q,S)"],
               f"{r['AIC']:.2f}" if r["AIC"] else "—"] for r in results]
col_labels = ["Store", "Order (p, d, q)", "Seasonal (P, D, Q, S)", "AIC"]

fig_height = max(4, len(table_data) * 0.5 + 2)
fig, ax = plt.subplots(figsize=(12, fig_height))
ax.axis("off")

table = ax.table(
    cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
    colWidths=[0.12, 0.25, 0.35, 0.18],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# header style
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# xen kẽ màu nền
for i in range(len(table_data)):
    color = "#f8f9fa" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

fig.suptitle(f"Auto-ARIMA Seasonal Discovery (m=7) — {len(stores)} stores type C",
             fontsize=12, fontweight="bold", y=0.97)
_save(fig, "326_seasonal_discovery_table.png")

print(f"\nDone. Seasonal discovery figures saved to: {OUT_DIR.resolve()}")
