"""Hình minh họa Section 3.2.1 — Bước 4: Grid search 192 tổ hợp ARIMA.

Grid cố định: p={3,4,5,6}, d={0,1}, q={0,1,3,4,5,6}, trend={n,c,t,ct} → 4×2×6×4 = 192.
Đánh giá RMSPE trên tập validation (30 ngày, 02/07–31/07/2015), 148 cửa hàng type C.
Kết quả mong đợi: ARIMA(5,1,6) trend='t' đạt RMSPE ≈ 0.3143.

Output (results/figures/):
  324_grid_top10_table.png    — Bảng top-10 cấu hình tốt nhất
  324_grid_heatmap_pq.png     — Heatmap RMSPE: p vs q
  324_grid_sensitivity.png    — Sensitivity plots: p, d, q, trend

Chạy:
    python scripts/figures/fig_324_grid_search.py
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
from src.models.arima import ARIMAModel
from src.tuning.tuning_viz import plot_tuning_heatmap
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# grid cố định từ phân tích Bước 2–3: mở rộng ±1 quanh top orders của auto_arima
P_GRID = [0, 1, 2, 3, 4]
D_GRID = [0, 1]
Q_GRID = [0, 1, 3, 4, 5, 6]
TREND_GRID = ["n", "c", "t", "ct"]


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== Bước 4: Grid Search ARIMA (192 tổ hợp) ===\n")

config = load_config("arima")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

# val_df trong code = test period (02/07–31/07) → dùng làm tập đánh giá grid search
eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train rows, {len(eval_df)} eval rows")
print(f"Grid: p={P_GRID}, d={D_GRID}, q={Q_GRID}, trend={TREND_GRID}")

param_grid = list(itertools.product(P_GRID, D_GRID, Q_GRID, TREND_GRID))
print(f"Tổng tổ hợp: {len(param_grid)}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
results = []
total = len(param_grid)
t0 = time.time()

for i, (p, d, q, trend) in enumerate(param_grid, 1):
    cfg = copy.deepcopy(config)
    cfg["model"]["order"] = [p, d, q]
    cfg["model"]["trend"] = trend

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMAModel(cfg)
            start = time.time()
            model.train(train_df)
            elapsed = time.time() - start
            preds = model.predict(eval_df)
            y_true = eval_df["Sales"].values
            metrics = evaluate_all(y_true, preds)
            results.append({
                "order": f"({p}, {d}, {q})", "p": p, "d": d, "q": q,
                "trend": trend, **metrics, "time_seconds": round(elapsed, 2),
            })
    except Exception:
        results.append({
            "order": f"({p}, {d}, {q})", "p": p, "d": d, "q": q,
            "trend": trend, "rmspe": float("inf"), "rmse": float("inf"),
            "mae": float("inf"), "mape": float("inf"), "time_seconds": 0,
        })

    if i % 20 == 0 or i == total:
        elapsed_total = time.time() - t0
        print(f"  [{i:3d}/{total}] {elapsed_total:.0f}s elapsed...")

results_df = pd.DataFrame(results).sort_values("rmspe").reset_index(drop=True)

# ── Lưu CSV + In kết quả ─────────────────────────────────────────────────────
csv_path = OUT_DIR / "324_grid_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path}")

print(f"\nTop 10 cấu hình ARIMA:")
top10 = results_df.head(10)
print(top10[["order", "trend", "rmspe", "rmse", "time_seconds"]].to_string(index=False))
best = results_df.iloc[0]
print(f"\nBest: ARIMA{best['order']} trend='{best['trend']}', RMSPE={best['rmspe']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng top-10
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    table_data.append([
        rank, row["order"], row["trend"],
        f"{row['rmspe']:.4f}", f"{row['rmse']:.2f}", f"{row['time_seconds']:.2f}",
    ])
col_labels = ["#", "Order (p, d, q)", "Trend", "RMSPE", "RMSE", "Time (s)"]

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.06, 0.2, 0.1, 0.15, 0.15, 0.12])
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
fig.suptitle("Top 10 cấu hình ARIMA — Grid search 192 tổ hợp", fontsize=12, fontweight="bold", y=0.95)
_save(fig, "324_grid_top10_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap p vs q (reuse tuning_viz)
# ══════════════════════════════════════════════════════════════════════════════
plot_tuning_heatmap(results_df, "p", "q", metric="rmspe",
                    save_path=str(OUT_DIR / "324_grid_heatmap_pq.png"))
print(f"  Saved: {OUT_DIR / '324_grid_heatmap_pq.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Sensitivity 2x2 (p, d, q, trend)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, param in zip(axes.flat, ["p", "d", "q", "trend"]):
    grouped = results_df.groupby(param)["rmspe"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)
    x = range(len(grouped))
    ax.errorbar(x, grouped["mean"], yerr=grouped["std"], marker="o", capsize=4,
                linewidth=2, markersize=6, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in grouped[param]])
    ax.set_xlabel(param.upper() if param != "trend" else "Trend", fontsize=11)
    ax.set_ylabel("RMSPE", fontsize=10)
    ax.set_title(f"Sensitivity: {param}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
fig.suptitle("Phân tích sensitivity — RMSPE theo từng tham số", fontsize=13, fontweight="bold")
fig.tight_layout()
_save(fig, "324_grid_sensitivity.png")

print(f"\nDone. Grid search figures saved to: {OUT_DIR.resolve()}")
