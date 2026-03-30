"""Hình minh họa Section 3.2.2 — Ablation study biến ngoại sinh SARIMAX.

Cố định SARIMAX(4,0,4)(0,0,1,7) trend='n', thử 8 tổ hợp exog_columns
→ đánh giá đóng góp từng biến ngoại sinh.
Kết quả mong đợi: Promo chiếm −42.2%, Full model RMSPE ≈ 0.1945.

Output (results/figures/):
  328_ablation_table.png  — Bảng 8 thí nghiệm: Experiment | Exog | RMSPE | RMSE | ΔRMSPE
  328_ablation_bar.png    — Bar chart so sánh RMSPE + baseline line

Chạy:
    python scripts/figures/fig_328_ablation.py
"""

import copy
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.data.preprocessor import preprocess
from src.evaluation.metrics import evaluate_all
from src.models.sarimax import SARIMAXModel
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# cấu hình tối ưu từ grid search (KHÔNG có exog)
BEST_ORDER = [4, 0, 4]
BEST_SEASONAL = [0, 0, 1, 7]
BEST_TREND = "n"

# 8 tổ hợp biến ngoại sinh — từ baseline (rỗng) đến full (tất cả)
ABLATION_COMBOS = {
    "Baseline": [],
    "+Promo": ["Promo"],
    "+SchoolHoliday": ["SchoolHoliday"],
    "+StateHoliday": ["StateHoliday"],
    "Promo+School": ["Promo", "SchoolHoliday"],
    "Promo+State": ["Promo", "StateHoliday"],
    "School+State": ["SchoolHoliday", "StateHoliday"],
    "Full": ["Promo", "SchoolHoliday", "StateHoliday"],
}


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print(f"=== SARIMAX Ablation Study — order={BEST_ORDER}, seasonal={BEST_SEASONAL}, trend='{BEST_TREND}' ===\n")

config = load_config("sarimax")
config["model"]["order"] = BEST_ORDER
config["model"]["seasonal_order"] = BEST_SEASONAL
config["model"]["trend"] = BEST_TREND
set_seed(config.get("seed", 42))

df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train, {len(eval_df)} eval rows")
print(f"Thí nghiệm: {len(ABLATION_COMBOS)} tổ hợp\n")

# ── Ablation loop ─────────────────────────────────────────────────────────────
# mỗi thí nghiệm: thay đổi exog_columns, giữ nguyên order/seasonal/trend
results = []
total = len(ABLATION_COMBOS)

for i, (exp_name, exog_cols) in enumerate(ABLATION_COMBOS.items(), 1):
    cfg = copy.deepcopy(config)
    cfg["model"]["exog_columns"] = exog_cols

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
            row = {"experiment": exp_name,
                   "exog": ", ".join(exog_cols) if exog_cols else "(không có)",
                   **metrics, "time_seconds": round(elapsed, 2)}
            results.append(row)
            print(f"  [{i}/{total}] {exp_name}: RMSPE={metrics['rmspe']:.4f} ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  [{i}/{total}] {exp_name}: FAILED — {e}")
        results.append({"experiment": exp_name,
                        "exog": ", ".join(exog_cols) if exog_cols else "(không có)",
                        "rmspe": float("inf"), "rmse": float("inf"),
                        "mae": float("inf"), "mape": float("inf"), "time_seconds": 0})

# ── Tính ΔRMSPE so với baseline ───────────────────────────────────────────────
baseline_rmspe = results[0]["rmspe"]
for r in results:
    if r["rmspe"] != float("inf") and baseline_rmspe > 0:
        r["delta_rmspe"] = f"{(r['rmspe'] - baseline_rmspe) / baseline_rmspe * 100:+.1f}%"
    else:
        r["delta_rmspe"] = "—"

results_df = pd.DataFrame(results)

# ── In kết quả ────────────────────────────────────────────────────────────────
print(f"\n{'Thí nghiệm':<20} {'Exog':<35} {'RMSPE':>8} {'RMSE':>10} {'ΔRMSPE':>10}")
print("-" * 85)
for r in results:
    print(f"  {r['experiment']:<18} {r['exog']:<35} {r['rmspe']:>8.4f} {r['rmse']:>10.2f} {r['delta_rmspe']:>10}")

best = min(results, key=lambda x: x["rmspe"])
print(f"\nBest: {best['experiment']} — RMSPE={best['rmspe']:.4f} ({best['delta_rmspe']})")

# ── Lưu CSV ───────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "328_ablation_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"  CSV: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng ablation
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = [[r["experiment"], r["exog"], f"{r['rmspe']:.4f}", f"{r['rmse']:.2f}", r["delta_rmspe"]]
              for r in results]
col_labels = ["Thí nghiệm", "Biến ngoại sinh", "RMSPE", "RMSE", "Δ RMSPE"]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.16, 0.32, 0.12, 0.14, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# tô màu: baseline xám, best xanh lá, khác xen kẽ
best_idx = min(range(len(results)), key=lambda i: results[i]["rmspe"])
for i in range(len(table_data)):
    if i == 0:
        color = "#fef9e7"  # baseline vàng nhạt
    elif i == best_idx:
        color = "#eafaf1"  # best xanh lá nhạt
    else:
        color = "#f8f9fa" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

fig.suptitle("Ablation Study — Đóng góp biến ngoại sinh (SARIMAX)", fontsize=12, fontweight="bold", y=0.97)
_save(fig, "328_ablation_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Bar chart RMSPE
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
n = len(results_df)
colors = sns.color_palette("Set2", n)
rmspe_vals = [r["rmspe"] for r in results]
bars = ax.bar(range(n), rmspe_vals, color=colors)
ax.set_xticks(range(n))
ax.set_xticklabels([r["experiment"] for r in results], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("RMSPE", fontsize=11)

# baseline line
ax.axhline(y=baseline_rmspe, color="gray", linestyle="--", alpha=0.6,
           label=f"Baseline: {baseline_rmspe:.4f}")

# ghi giá trị trên mỗi cột
for bar, val in zip(bars, rmspe_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_title("Ablation Study — RMSPE theo tổ hợp biến ngoại sinh", fontsize=12, fontweight="bold")
fig.tight_layout()
_save(fig, "328_ablation_bar.png")

print(f"\nDone. Ablation figures saved to: {OUT_DIR.resolve()}")
