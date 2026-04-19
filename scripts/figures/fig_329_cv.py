"""Hình minh họa Section 3.3.x — Cross-validation SARIMAX(3,0,3)(1,1,0,12) trend='n'.

Walk-forward CV 5 folds, expanding window, 30 eval_days, exog=[Promo].
Cấu hình tối ưu từ grid search 12 combos baseline + ablation 8 combos exog.
Validation RMSPE = 0.1455.

Output (results/figures/):
  329_cv_results_table.png  — Bảng per-fold: Fold | RMSPE | RMSE | MAE | MAPE | Time
  329_cv_bar.png            — Bar chart RMSPE theo fold + mean line + std band

Chạy:
    python scripts/figures/fig_329_cv.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.evaluation.cross_validation import walk_forward_cv
from src.models.sarimax import SARIMAXModel
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# cấu hình tối ưu từ grid search 12 combos + ablation 8 combos
BEST_ORDER = [3, 0, 3]
BEST_SEASONAL = [1, 1, 0, 12]
BEST_TREND = "n"
BEST_EXOG = ["Promo"]
N_SPLITS = 5
EVAL_DAYS = 30


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
order_str = f"({BEST_ORDER[0]},{BEST_ORDER[1]},{BEST_ORDER[2]})"
seasonal_str = f"({BEST_SEASONAL[0]},{BEST_SEASONAL[1]},{BEST_SEASONAL[2]},{BEST_SEASONAL[3]})"
print(f"=== SARIMAX CV — order={order_str} seasonal={seasonal_str} trend='{BEST_TREND}' exog={BEST_EXOG} ===\n")

config = load_config("sarimax")
config["model"]["order"] = BEST_ORDER
config["model"]["seasonal_order"] = BEST_SEASONAL
config["model"]["trend"] = BEST_TREND
config["model"]["exog_columns"] = BEST_EXOG
set_seed(config.get("seed", 42))

df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)

n_stores = df["Store"].nunique()
print(f"Stores: {n_stores}, Folds: {N_SPLITS}, Eval days: {EVAL_DAYS}, Expanding window")
print(f"Exog: {BEST_EXOG}")

# ── Walk-forward CV ───────────────────────────────────────────────────────────
print("Đang chạy cross-validation (có thể mất vài phút)...\n")

cv_results = walk_forward_cv(
    model_class=SARIMAXModel,
    config=config,
    df=df,
    n_splits=N_SPLITS,
    expanding=True,
    eval_days=EVAL_DAYS,
)

folds = cv_results["folds"]
agg = cv_results["aggregated"]

# ── In kết quả ────────────────────────────────────────────────────────────────
print(f"{'Fold':<6} {'RMSPE':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'Time (s)':>10}")
print("-" * 55)
for f in folds:
    print(f"  {f['fold']:<4} {f['rmspe']:>8.4f} {f['rmse']:>10.2f} {f['mae']:>10.2f} "
          f"{f['mape']:>8.4f} {f['training_time_seconds']:>10.1f}")
print("-" * 55)
print(f"  Mean {agg['rmspe_mean']:>8.4f} {agg['rmse_mean']:>10.2f} {agg['mae_mean']:>10.2f} "
      f"{agg['mape_mean']:>8.4f}")
print(f"  Std  {agg['rmspe_std']:>8.4f} {agg['rmse_std']:>10.2f} {agg['mae_std']:>10.2f} "
      f"{agg['mape_std']:>8.4f}")
print(f"\nKết luận: RMSPE = {agg['rmspe_mean']:.4f} ± {agg['rmspe_std']:.4f} "
      f"(CV = {agg['rmspe_std']/agg['rmspe_mean']*100:.1f}%)")

# ── Lưu CSV ───────────────────────────────────────────────────────────────────
rows = []
for f in folds:
    rows.append({"fold": f["fold"], "rmspe": f["rmspe"], "rmse": f["rmse"],
                 "mae": f["mae"], "mape": f["mape"], "time_s": f["training_time_seconds"]})
rows.append({"fold": "mean", "rmspe": agg["rmspe_mean"], "rmse": agg["rmse_mean"],
             "mae": agg["mae_mean"], "mape": agg["mape_mean"], "time_s": None})
rows.append({"fold": "std", "rmspe": agg["rmspe_std"], "rmse": agg["rmse_std"],
             "mae": agg["mae_std"], "mape": agg["mape_std"], "time_s": None})
pd.DataFrame(rows).to_csv(OUT_DIR / "329_cv_results.csv", index=False)
print(f"  CSV: {OUT_DIR / '329_cv_results.csv'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng per-fold
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for f in folds:
    table_data.append([
        f"Fold {f['fold']}", f"{f['rmspe']:.4f}", f"{f['rmse']:.2f}",
        f"{f['mae']:.2f}", f"{f['mape']:.4f}", f"{f['training_time_seconds']:.1f}",
    ])
table_data.append([
    "TB ± Std",
    f"{agg['rmspe_mean']:.4f} ± {agg['rmspe_std']:.4f}",
    f"{agg['rmse_mean']:.2f} ± {agg['rmse_std']:.2f}",
    f"{agg['mae_mean']:.2f} ± {agg['mae_std']:.2f}",
    f"{agg['mape_mean']:.4f} ± {agg['mape_std']:.4f}",
    "—",
])
col_labels = ["Fold", "RMSPE", "RMSE", "MAE", "MAPE", "Time (s)"]

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.1, 0.22, 0.18, 0.18, 0.18, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(table_data)):
    color = "#fef9e7" if i == len(table_data) - 1 else ("#f8f9fa" if i % 2 == 0 else "white")
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)
        if i == len(table_data) - 1:
            table[i + 1, j].set_text_props(fontweight="bold")

fig.suptitle(f"5-Fold CV — SARIMAX{order_str}{seasonal_str} trend='{BEST_TREND}' exog={BEST_EXOG}",
             fontsize=11, fontweight="bold", y=0.97)
_save(fig, "329_cv_results_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Bar chart RMSPE per fold
# ══════════════════════════════════════════════════════════════════════════════
fold_names = [f"Fold {f['fold']}" for f in folds]
rmspe_vals = [f["rmspe"] for f in folds]
mean_val = agg["rmspe_mean"]
std_val = agg["rmspe_std"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(fold_names, rmspe_vals, color="#2ecc71", alpha=0.85, edgecolor="white", linewidth=1.5)
ax.axhline(y=mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.4f}")
ax.fill_between(range(-1, len(folds) + 1), mean_val - std_val, mean_val + std_val,
                alpha=0.12, color="#e74c3c", label=f"± Std: {std_val:.4f}")
ax.set_xlim(-0.6, len(folds) - 0.4)

for bar, val in zip(bars, rmspe_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("RMSPE", fontsize=11)
ax.set_title(f"RMSPE per Fold — SARIMAX{order_str}{seasonal_str} trend='{BEST_TREND}'",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
_save(fig, "329_cv_bar.png")

print(f"\nDone. SARIMAX CV figures saved to: {OUT_DIR.resolve()}")
