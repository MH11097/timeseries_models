"""Hình minh họa Section 3.2.2 — Grid search SARIMAX.

Grid search p, d, q + 2 seasonal orders + trend trên 148 cửa hàng type C.
Đánh giá RMSPE trên tập validation (30 ngày).
Kết quả mong đợi: SARIMAX(4,0,4)(0,0,1,7) trend='n', RMSPE ≈ 0.2008.

Output (results/figures/):
  327_grid_top10_table.png    — Bảng top-10 cấu hình tốt nhất
  327_grid_heatmap_pq.png     — Heatmap RMSPE: p vs q
  327_grid_sensitivity.png    — Sensitivity plots: p, d, q, trend

Chạy:
    python scripts/figures/fig_327_grid_search.py
"""

import copy
import itertools
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all
from src.models.sarimax import SARIMAXModel
from src.tuning.tuning_viz import plot_tuning_heatmap
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# grid từ phân tích Bước 1 (discovery) → mở rộng ±1 quanh top orders
P_GRID = [0, 1, 2, 3, 4]
D_GRID = [0, 1, 2]
Q_GRID = [0, 1, 2, 3, 4]
SEASONAL_GRID = [(1, 0, 2, 7), (0, 0, 1, 7)]  # 2 pattern chính từ auto_arima
TREND_GRID = ["n", "c", "t", "ct"]


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== SARIMAX — Grid Search ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

# eval_df = tập validation (30 ngày cuối)
eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train rows, {len(eval_df)} eval rows")

param_grid = list(itertools.product(P_GRID, D_GRID, Q_GRID, SEASONAL_GRID, TREND_GRID))
print(f"Grid: p={P_GRID}, d={D_GRID}, q={Q_GRID}")
print(f"      seasonal={SEASONAL_GRID}, trend={TREND_GRID}")
print(f"Tổng tổ hợp: {len(param_grid)}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
# mỗi tổ hợp (p,d,q,seasonal,trend) train trên toàn bộ 148 stores → eval RMSPE trên val
results = []
total = len(param_grid)
t0 = time.time()

for i, (p, d, q, seasonal, trend) in enumerate(param_grid, 1):
    cfg = copy.deepcopy(config)
    cfg["model"]["order"] = [p, d, q]
    cfg["model"]["seasonal_order"] = list(seasonal)
    cfg["model"]["trend"] = trend
    cfg["model"]["exog_columns"] = []  # grid search KHÔNG dùng exog — đánh giá thuần structure

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAXModel(cfg)
            start = time.time()
            model.train(train_df)
            elapsed = time.time() - start
            preds = model.predict(eval_df)
            y_true = eval_df["Sales"].values
            metrics = evaluate_all(y_true, preds)
            s_str = f"({seasonal[0]},{seasonal[1]},{seasonal[2]},{seasonal[3]})"
            results.append({
                "order": f"({p}, {d}, {q})", "seasonal": s_str,
                "p": p, "d": d, "q": q,
                "P": seasonal[0], "D": seasonal[1], "Q": seasonal[2],
                "trend": trend, **metrics, "time_seconds": round(elapsed, 2),
            })
    except Exception:
        s_str = f"({seasonal[0]},{seasonal[1]},{seasonal[2]},{seasonal[3]})"
        results.append({
            "order": f"({p}, {d}, {q})", "seasonal": s_str,
            "p": p, "d": d, "q": q,
            "P": seasonal[0], "D": seasonal[1], "Q": seasonal[2],
            "trend": trend, "rmspe": float("inf"), "rmse": float("inf"),
            "mae": float("inf"), "mape": float("inf"), "time_seconds": 0,
        })

    if i % 50 == 0 or i == total:
        elapsed_total = time.time() - t0
        print(f"  [{i:4d}/{total}] {elapsed_total:.0f}s elapsed...")

results_df = pd.DataFrame(results).sort_values("rmspe").reset_index(drop=True)

# ── Lưu CSV + In kết quả ─────────────────────────────────────────────────────
csv_path = OUT_DIR / "327_grid_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path}")

print(f"\nTop 10 cấu hình SARIMAX:")
top10 = results_df.head(10)
print(top10[["order", "seasonal", "trend", "rmspe", "rmse", "time_seconds"]].to_string(index=False))
best = results_df.iloc[0]
print(f"\nBest: SARIMAX{best['order']} x {best['seasonal']} trend='{best['trend']}', RMSPE={best['rmspe']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng top-10
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    table_data.append([
        rank, row["order"], row["seasonal"], row["trend"],
        f"{row['rmspe']:.4f}", f"{row['rmse']:.2f}", f"{row['time_seconds']:.2f}",
    ])
col_labels = ["#", "Order", "Seasonal", "Trend", "RMSPE", "RMSE", "Time (s)"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.05, 0.14, 0.18, 0.08, 0.12, 0.12, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(table_data)):
    color = "#eafaf1" if i == 0 else ("#f8f9fa" if i % 2 == 0 else "white")
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)
fig.suptitle("Top 10 cấu hình SARIMAX — Grid search", fontsize=12, fontweight="bold", y=0.95)
_save(fig, "327_grid_top10_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap p vs q
# ══════════════════════════════════════════════════════════════════════════════
plot_tuning_heatmap(results_df, "p", "q", metric="rmspe",
                    save_path=str(OUT_DIR / "327_grid_heatmap_pq.png"))
print(f"  Saved: {OUT_DIR / '327_grid_heatmap_pq.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Sensitivity 2x3 (p, d, q, P, Q, trend)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, param in zip(axes.flat, ["p", "d", "q", "P", "Q", "trend"]):
    grouped = results_df.groupby(param)["rmspe"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)
    x = range(len(grouped))
    ax.errorbar(x, grouped["mean"], yerr=grouped["std"], marker="o", capsize=4,
                linewidth=2, markersize=6, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in grouped[param]])
    ax.set_xlabel(param, fontsize=11)
    ax.set_ylabel("RMSPE", fontsize=10)
    ax.set_title(f"Sensitivity: {param}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
fig.suptitle("Phân tích sensitivity — RMSPE theo từng tham số SARIMAX", fontsize=13, fontweight="bold")
fig.tight_layout()
_save(fig, "327_grid_sensitivity.png")

print(f"\nDone. SARIMAX grid search figures saved to: {OUT_DIR.resolve()}")
