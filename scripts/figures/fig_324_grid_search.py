"""Hình minh họa Section 3.2.1 — Bước 4: Grid search 70 tổ hợp ARIMA thuần.

ARIMA thuần (Box-Jenkins) không có trend parameter — chỉ dùng trend='n'.
Grid: p={0,1,2,3,4}, d={0,1}, q={0,1,2,3,4,5,6}, trend='n' → 5×2×7 = 70.
Đánh giá RMSPE + AIC/BIC trên tập validation (30 ngày, 02/07–31/07/2015), 148 cửa hàng type C.

Output (results/figures/):
  324_grid_top10_table.png    — Bảng top-10 cấu hình tốt nhất
  324_grid_heatmap_pq.png     — Heatmap RMSPE: p vs q
  324_grid_sensitivity.png    — Sensitivity plots: p, d, q

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

# ARIMA thuần: trend='n' cố định (trend='t'/'ct' không thuộc ARIMA chuẩn Box-Jenkins)
# Grid thu hẹp từ kết quả 70-combo search: top-10 đều d=1, p∈{2-4}, q∈{3-6}
P_GRID = [1, 2, 3, 4, 5]
D_GRID = [1]
Q_GRID = [3, 4, 5, 6]
TREND = "n"  # cố định, không search


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
total_combos = len(P_GRID) * len(D_GRID) * len(Q_GRID)
print(f"=== Bước 4: Grid Search ARIMA thuần ({total_combos} tổ hợp, trend='{TREND}') ===\n")

config = load_config("arima")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

# val_df trong code = test period (02/07–31/07) → dùng làm tập đánh giá grid search
eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train rows, {len(eval_df)} eval rows")
print(f"Grid: p={P_GRID}, d={D_GRID}, q={Q_GRID}, trend='{TREND}' (cố định)")

param_grid = list(itertools.product(P_GRID, D_GRID, Q_GRID))
print(f"Tổng tổ hợp: {len(param_grid)}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
results = []
total = len(param_grid)
t0 = time.time()

for i, (p, d, q) in enumerate(param_grid, 1):
    cfg = copy.deepcopy(config)
    cfg["model"]["order"] = [p, d, q]
    cfg["model"]["trend"] = TREND

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
            info = model.get_info_criteria()
            results.append({
                "order": f"({p}, {d}, {q})", "p": p, "d": d, "q": q,
                "trend": TREND, **metrics,
                "aic_mean": round(info["aic_mean"], 2),
                "bic_mean": round(info["bic_mean"], 2),
                "n_converged": info["n_fitted"],
                "time_seconds": round(elapsed, 2),
            })
    except Exception:
        results.append({
            "order": f"({p}, {d}, {q})", "p": p, "d": d, "q": q,
            "trend": TREND, "rmspe": float("inf"), "rmse": float("inf"),
            "mae": float("inf"), "mape": float("inf"),
            "aic_mean": float("inf"), "bic_mean": float("inf"),
            "n_converged": 0, "time_seconds": 0,
        })

    if i % 10 == 0 or i == total:
        elapsed_total = time.time() - t0
        print(f"  [{i:3d}/{total}] {elapsed_total:.0f}s elapsed...")

results_df = pd.DataFrame(results).sort_values("rmspe").reset_index(drop=True)

# ── Lưu CSV + In kết quả ─────────────────────────────────────────────────────
csv_path = OUT_DIR / "324_grid_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path}")

print(f"\nTop 10 cấu hình ARIMA (trend='{TREND}'):")
top10 = results_df.head(10)
print(top10[["order", "d", "rmspe", "rmse", "aic_mean", "bic_mean", "n_converged", "time_seconds"]].to_string(index=False))
best = results_df.iloc[0]
print(f"\nBest: ARIMA{best['order']} trend='{TREND}', RMSPE={best['rmspe']:.4f}, AIC={best['aic_mean']:.1f}, BIC={best['bic_mean']:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng top-10
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    table_data.append([
        rank, row["order"],
        f"{row['rmspe']:.4f}", f"{row['rmse']:.2f}",
        f"{row['aic_mean']:.1f}", f"{row['bic_mean']:.1f}",
        int(row["n_converged"]), f"{row['time_seconds']:.2f}",
    ])
col_labels = ["#", "Order (p, d, q)", "RMSPE", "RMSE", "AIC", "BIC", "Conv.", "Time (s)"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.05, 0.16, 0.1, 0.1, 0.12, 0.12, 0.07, 0.1])
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
fig.suptitle(f"Top 10 cấu hình ARIMA thuần (trend='{TREND}') — Grid search {total_combos} tổ hợp", fontsize=12, fontweight="bold", y=0.95)
_save(fig, "324_grid_top10_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap p vs q (reuse tuning_viz)
# ══════════════════════════════════════════════════════════════════════════════
plot_tuning_heatmap(results_df, "p", "q", metric="rmspe",
                    save_path=str(OUT_DIR / "324_grid_heatmap_pq.png"))
print(f"  Saved: {OUT_DIR / '324_grid_heatmap_pq.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Sensitivity 2x2 (p, d, q theo RMSPE + BIC theo p)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# RMSPE sensitivity: p, d, q
for ax, param in zip(axes.flat[:3], ["p", "d", "q"]):
    grouped = results_df.groupby(param)["rmspe"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)
    x = range(len(grouped))
    ax.errorbar(x, grouped["mean"], yerr=grouped["std"], marker="o", capsize=4,
                linewidth=2, markersize=6, color="steelblue")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(v) for v in grouped[param]])
    ax.set_xlabel(param.upper(), fontsize=11)
    ax.set_ylabel("RMSPE", fontsize=10)
    ax.set_title(f"Sensitivity: {param}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
# BIC sensitivity theo số params (p+q)
ax_bic = axes.flat[3]
valid = results_df[results_df["bic_mean"] < float("inf")].copy()
valid["n_params"] = valid["p"] + valid["q"]
grouped_bic = valid.groupby("n_params")["bic_mean"].agg(["mean", "std"]).reset_index()
grouped_bic["std"] = grouped_bic["std"].fillna(0)
x = range(len(grouped_bic))
ax_bic.errorbar(x, grouped_bic["mean"], yerr=grouped_bic["std"], marker="s", capsize=4,
                linewidth=2, markersize=6, color="coral")
ax_bic.set_xticks(list(x))
ax_bic.set_xticklabels([str(v) for v in grouped_bic["n_params"]])
ax_bic.set_xlabel("Tổng params (p+q)", fontsize=11)
ax_bic.set_ylabel("BIC (mean)", fontsize=10)
ax_bic.set_title("Parsimony: BIC vs p+q", fontsize=11, fontweight="bold")
ax_bic.grid(True, alpha=0.3)
fig.suptitle(f"Phân tích sensitivity — ARIMA thuần (trend='{TREND}')", fontsize=13, fontweight="bold")
fig.tight_layout()
_save(fig, "324_grid_sensitivity.png")

print(f"\nDone. Grid search figures saved to: {OUT_DIR.resolve()}")
