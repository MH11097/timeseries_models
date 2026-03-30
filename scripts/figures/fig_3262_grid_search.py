"""Hình minh họa Section 3.2.2 — Grid search SARIMAX (S=12).

Grid search trên tổ hợp (p,d,q) × (P,D,Q,S) × trend:
  p ∈ [0..4], d ∈ [0,1], q ∈ [0..4]         → 50 tổ hợp phi mùa vụ
  P ∈ [0,1], D=1 (cố định), Q ∈ [0,1,2]     → 6 tổ hợp mùa vụ
  trend ∈ {n, c, ct}                          → 3 (bỏ 't' đơn lẻ — xung đột khi D=1)
  Tổng: 50 × 6 × 3 = 900 tổ hợp

Tiêu chí: RMSPE trung bình trên 148 stores type C, val set (30 ngày cuối).

Output (results/figures/):
  3262_grid_results.csv         — Toàn bộ kết quả (900 dòng)
  3262_grid_top10_table.png     — Bảng top-10 cấu hình
  3262_grid_heatmap_pq.png      — Heatmap RMSPE: p vs q
  3262_grid_sensitivity.png     — Sensitivity plots: p, d, q, P, Q, trend

Chạy:
    python scripts/figures/fig_3262_grid_search.py
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

# grid phi mùa vụ — từ ACF/PACF analysis (Bước 2 ARIMA, mục 3.2.1)
P_GRID = [0, 1, 2, 3, 4]  # p: AR order
D_GRID = [0, 1]            # d: differencing (ADF/KPSS → d=1 phổ biến)
Q_GRID = [0, 1, 2, 3, 4]  # q: MA order

# grid mùa vụ — từ Bước 2 SARIMAX (seasonal OCSB/CH → D=1, ACF/PACF → P∈[0,1], Q∈[0,2])
S = 12                      # chu kỳ mùa vụ từ Bước 1
SP_GRID = [0, 1]            # P: seasonal AR
SD = 1                      # D: seasonal differencing (cố định, từ OCSB/CH)
SQ_GRID = [0, 1, 2]         # Q: seasonal MA

# trend: bỏ 't' đơn lẻ — khi D=1 đã sai phân mùa vụ, trend tuyến tính tường minh
# xung đột với phép sai phân (vừa loại trend vừa thêm lại → dư thừa, bất ổn số)
TREND_GRID = ["n", "c", "ct"]


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== SARIMAX Grid Search (S=12, D=1) ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

# eval_df = tập validation (30 ngày cuối training period)
eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train rows, {len(eval_df)} eval rows")

# tạo grid tổ hợp
param_grid = list(itertools.product(P_GRID, D_GRID, Q_GRID, SP_GRID, SQ_GRID, TREND_GRID))
print(f"Grid: p={P_GRID}, d={D_GRID}, q={Q_GRID}")
print(f"      P={SP_GRID}, D={SD} (fixed), Q={SQ_GRID}, S={S}")
print(f"      trend={TREND_GRID}")
print(f"Tổng tổ hợp: {len(param_grid)}\n")

# ── Grid search ──────────────────────────────────────────────────────────────
# mỗi tổ hợp: train 148 stores → eval RMSPE trên val set
results = []
total = len(param_grid)
t0 = time.time()

for i, (p, d, q, sp, sq, trend) in enumerate(param_grid, 1):
    cfg = copy.deepcopy(config)
    cfg["model"]["order"] = [p, d, q]
    cfg["model"]["seasonal_order"] = [sp, SD, sq, S]
    cfg["model"]["trend"] = trend
    cfg["model"]["exog_columns"] = []  # grid search thuần structure, không exog
    cfg["model"]["maxiter"] = 300      # S=12 state space lớn → cần nhiều iterations

    s_str = f"({sp},{SD},{sq},{S})"
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
            results.append({
                "order": f"({p},{d},{q})", "seasonal": s_str,
                "p": p, "d": d, "q": q,
                "P": sp, "D": SD, "Q": sq,
                "trend": trend, **metrics, "time_seconds": round(elapsed, 2),
            })
    except Exception:
        results.append({
            "order": f"({p},{d},{q})", "seasonal": s_str,
            "p": p, "d": d, "q": q,
            "P": sp, "D": SD, "Q": sq,
            "trend": trend, "rmspe": float("inf"), "rmse": float("inf"),
            "mae": float("inf"), "mape": float("inf"), "time_seconds": 0,
        })

    if i % 50 == 0 or i == total:
        elapsed_total = time.time() - t0
        eta = elapsed_total / i * (total - i)
        print(f"  [{i:4d}/{total}] {elapsed_total:.0f}s elapsed, ETA ~{eta:.0f}s")

results_df = pd.DataFrame(results).sort_values("rmspe").reset_index(drop=True)

# lọc bỏ các config failed (rmspe=inf)
valid_df = results_df[results_df["rmspe"] < float("inf")]
failed_count = len(results_df) - len(valid_df)

# ── Lưu CSV + In kết quả ────────────────────────────────────────────────────
csv_path = OUT_DIR / "3262_grid_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path}")
print(f"  Thành công: {len(valid_df)}/{total}, Failed: {failed_count}")

print(f"\nTop 10 cấu hình SARIMAX (S={S}):")
top10 = valid_df.head(10)
print(top10[["order", "seasonal", "trend", "rmspe", "rmse", "time_seconds"]].to_string(index=False))

best = valid_df.iloc[0]
print(f"\nBest: SARIMAX{best['order']} × {best['seasonal']} trend='{best['trend']}', RMSPE={best['rmspe']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng top-10
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    table_data.append([
        rank, row["order"], row["seasonal"], row["trend"],
        f"{row['rmspe']:.4f}", f"{row['rmse']:.2f}", f"{row['time_seconds']:.1f}",
    ])
col_labels = ["#", "Order (p,d,q)", "Seasonal (P,D,Q,S)", "Trend", "RMSPE", "RMSE", "Time (s)"]

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.05, 0.14, 0.2, 0.08, 0.12, 0.12, 0.1])
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
        if i == 0:
            table[i + 1, j].set_text_props(fontweight="bold")

fig.suptitle(
    f"Top 10 cấu hình SARIMAX — Grid search (S={S}, D={SD})\n"
    f"900 tổ hợp, eval trên val set (30 ngày), 148 stores type C",
    fontsize=12, fontweight="bold", y=0.97,
)
_save(fig, "3262_grid_top10_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap p vs q (trung bình RMSPE qua tất cả P, Q, d, trend)
# ══════════════════════════════════════════════════════════════════════════════
plot_tuning_heatmap(valid_df, "p", "q", metric="rmspe",
                    save_path=str(OUT_DIR / "3262_grid_heatmap_pq.png"))
print(f"  Saved: {OUT_DIR / '3262_grid_heatmap_pq.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Sensitivity 2×3 (p, d, q, P, Q, trend)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, param in zip(axes.flat, ["p", "d", "q", "P", "Q", "trend"]):
    grouped = valid_df.groupby(param)["rmspe"].agg(["mean", "std"]).reset_index()
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
fig.suptitle(
    f"Phân tích sensitivity — RMSPE theo từng tham số SARIMAX (S={S})",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
_save(fig, "3262_grid_sensitivity.png")

print(f"\nDone. SARIMAX grid search (S={S}) saved to: {OUT_DIR.resolve()}")
