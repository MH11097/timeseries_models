"""Hình minh họa Section 3.3 — Grid search SARIMAX baseline (no exog, trend='n').

SARIMAX baseline (không exog): trend='n', S=12, D=1, d=0. Grid hẹp:
p ∈ {4,5} × q ∈ {3,4} × (P,Q) ∈ {(0,1),(1,0)} = 8 tổ hợp.
Đánh giá RMSPE + AIC/BIC trên tập validation (30 ngày, 02/07–31/07/2015), 148 cửa hàng type C.
Baseline cho ablation study exog (Promo, SchoolHoliday, StateHoliday) ở thực nghiệm sau.

Output (results/figures/):
  327_grid_top10_table.png    — Bảng top-10 cấu hình tốt nhất
  327_grid_heatmap_pq.png     — Heatmap RMSPE: p vs q
  327_grid_sensitivity.png    — Sensitivity plots: p, d, q, P, Q

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

# SARIMAX baseline: trend='n', d=0, D=1, S=12 — không exog
# Grid hẹp rút ra từ 26 combo trước: d=0 áp đảo, Q=2 dư thừa, (P=1,Q=0) & (P=0,Q=1) top
# Mở rộng p=3 để có bảng đầy đủ 3×2×2=12 combos
P_GRID = [3]                    # p=3 (bổ sung); 4,5 đã có trong /tmp/327_grid_results_p4p5.csv
D_GRID = [0]                    # d=0 cố định (pattern áp đảo trong 26 combo)
Q_GRID = [3, 4]                 # q ∈ {3,4}
# Mùa vụ: chỉ 2 cặp thay thế nhau (AR vs MA seasonal)
PQ_PAIRS = [(0, 1), (1, 0)]      # (P, Q): (0,1) MA-seasonal, (1,0) AR-seasonal
SD = 1                           # D — seasonal differencing (cố định)
S = 12                           # S — chu kỳ 2 tuần kinh doanh
TREND = "n"                      # trend cố định — SARIMAX thuần


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
total_combos = len(P_GRID) * len(D_GRID) * len(Q_GRID) * len(PQ_PAIRS)
print(f"=== SARIMAX baseline (no exog) — Grid Search ({total_combos} tổ hợp, trend='{TREND}', S={S}, D={SD}) ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

# eval_df = tập validation (30 ngày cuối)
eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train rows, {len(eval_df)} eval rows")

param_grid = [
    (p, d, q, sP, sQ)
    for p, d, q in itertools.product(P_GRID, D_GRID, Q_GRID)
    for sP, sQ in PQ_PAIRS
]
print(f"Grid phi mùa vụ: p={P_GRID}, d={D_GRID}, q={Q_GRID}")
print(f"Grid mùa vụ:     (P,Q) ∈ {PQ_PAIRS}, D={SD}, S={S}")
print(f"Trend: '{TREND}' (cố định) | Exog: [] (baseline no exog)")
print(f"Tổng tổ hợp: {len(param_grid)}\n")

# ── Grid search ───────────────────────────────────────────────────────────────
# mỗi tổ hợp (p,d,q)×(P,D,Q,S) train trên toàn bộ 148 stores → eval RMSPE trên val
results = []
total = len(param_grid)
t0 = time.time()

for i, (p, d, q, sP, sQ) in enumerate(param_grid, 1):
    seasonal = (sP, SD, sQ, S)
    cfg = copy.deepcopy(config)
    cfg["model"]["order"] = [p, d, q]
    cfg["model"]["seasonal_order"] = list(seasonal)
    cfg["model"]["trend"] = TREND
    # Baseline: KHÔNG exog — ablation study sẽ thêm lần lượt ở thực nghiệm sau
    cfg["model"]["exog_columns"] = []

    s_str = f"({sP},{SD},{sQ},{S})"
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
            # AIC/BIC trung bình từ các store đã hội tụ
            aic_vals = [m.aic for m in model.models.values()]
            bic_vals = [m.bic for m in model.models.values()]
            results.append({
                "order": f"({p}, {d}, {q})", "seasonal": s_str,
                "p": p, "d": d, "q": q, "P": sP, "D": SD, "Q": sQ,
                "trend": TREND, **metrics,
                "aic_mean": round(np.mean(aic_vals), 2),
                "bic_mean": round(np.mean(bic_vals), 2),
                "n_converged": len(model.models),
                "time_seconds": round(elapsed, 2),
            })
    except Exception:
        results.append({
            "order": f"({p}, {d}, {q})", "seasonal": s_str,
            "p": p, "d": d, "q": q, "P": sP, "D": SD, "Q": sQ,
            "trend": TREND, "rmspe": float("inf"), "rmse": float("inf"),
            "mae": float("inf"), "mape": float("inf"),
            "aic_mean": float("inf"), "bic_mean": float("inf"),
            "n_converged": 0, "time_seconds": 0,
        })

    # Log chi tiết từng combo + flush ngay
    elapsed_total = time.time() - t0
    last = results[-1]
    rmspe_str = f"{last['rmspe']:.4f}" if last['rmspe'] != float("inf") else "  FAIL"
    avg_per_combo = elapsed_total / i
    eta_sec = avg_per_combo * (total - i)
    eta_min = eta_sec / 60
    print(
        f"  [{i:3d}/{total}] ({p},{d},{q})x({sP},{SD},{sQ},{S}) "
        f"RMSPE={rmspe_str} conv={last.get('n_converged', 0):3d} "
        f"t={last.get('time_seconds', 0):5.1f}s | "
        f"total={elapsed_total/60:5.1f}m ETA={eta_min:5.1f}m",
        flush=True,
    )

results_df = pd.DataFrame(results)

# ── Merge với kết quả cũ nếu có (để build bảng 12 combos từ 2 lần chạy) ──────
_prev_csv = Path("/tmp/327_grid_results_p4p5.csv")
if _prev_csv.exists():
    prev_df = pd.read_csv(_prev_csv)
    print(f"\n  Merging {len(prev_df)} previous rows from {_prev_csv}")
    results_df = pd.concat([prev_df, results_df], ignore_index=True)

results_df = results_df.sort_values("rmspe").reset_index(drop=True)

# ── Lưu CSV + In kết quả ─────────────────────────────────────────────────────
csv_path = OUT_DIR / "327_grid_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path} ({len(results_df)} rows)")

print(f"\nTop 10 cấu hình SARIMAX (trend='{TREND}', S={S}):")
top10 = results_df.head(10)
print(top10[["order", "seasonal", "rmspe", "rmse", "aic_mean", "bic_mean", "n_converged", "time_seconds"]].to_string(index=False))
best = results_df.iloc[0]
print(f"\nBest: SARIMAX{best['order']} x {best['seasonal']} trend='{TREND}', RMSPE={best['rmspe']:.4f}, AIC={best['aic_mean']:.1f}, BIC={best['bic_mean']:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng top-10
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    table_data.append([
        rank, row["order"], row["seasonal"],
        f"{row['rmspe']:.4f}", f"{row['rmse']:.2f}",
        f"{row['aic_mean']:.1f}", f"{row['bic_mean']:.1f}",
        int(row["n_converged"]), f"{row['time_seconds']:.2f}",
    ])
col_labels = ["#", "Order", "Seasonal", "RMSPE", "RMSE", "AIC", "BIC", "Conv.", "Time (s)"]

fig, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.04, 0.12, 0.14, 0.09, 0.09, 0.11, 0.11, 0.06, 0.08])
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
fig.suptitle(f"Top SARIMAX baseline no-exog (trend='{TREND}', S={S}, d=0) — Grid {total_combos} tổ hợp", fontsize=12, fontweight="bold", y=0.95)
_save(fig, "327_grid_top10_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Heatmap p vs q
# ══════════════════════════════════════════════════════════════════════════════
plot_tuning_heatmap(results_df, "p", "q", metric="rmspe",
                    save_path=str(OUT_DIR / "327_grid_heatmap_pq.png"))
print(f"  Saved: {OUT_DIR / '327_grid_heatmap_pq.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Sensitivity — chỉ các chiều thực sự biến thiên trong grid hẹp
# ══════════════════════════════════════════════════════════════════════════════
# Với grid baseline: d=0 cố định → chỉ plot p, q, P, Q
sensitivity_params = ["p", "q", "P", "Q"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, param in zip(axes.flat, sensitivity_params):
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
fig.suptitle(f"Sensitivity — SARIMAX baseline no-exog (trend='{TREND}', S={S}, d=0)", fontsize=13, fontweight="bold")
fig.tight_layout()
_save(fig, "327_grid_sensitivity.png")

print(f"\nDone. SARIMAX grid search figures saved to: {OUT_DIR.resolve()}")
