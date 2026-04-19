"""Hình minh họa Section 3.3.x — Ablation study exog cho SARIMAX(3,0,3)x(1,1,0,12) trend='n'.

Cố định cấu hình tối ưu từ grid search baseline (RMSPE=0.1477 no-exog).
Quét full factorial 2^3 = 8 tổ hợp exog ⊂ {Promo, SchoolHoliday, StateHoliday}.
Mục tiêu: lượng hóa marginal effect của từng biến exog + interaction effects.

Output (results/figures/):
  328_ablation_results.csv    — Bảng đầy đủ 8 tổ hợp
  328_ablation_table.png      — Bảng sắp xếp theo RMSPE
  328_ablation_bar.png        — Bar chart RMSPE theo n_exog (0/1/2/3)

Chạy:
    python scripts/figures/fig_328_ablation_exog.py
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
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình SARIMAX cố định (best baseline từ grid search 12 combos)
ORDER = (3, 0, 3)
SEASONAL = (1, 1, 0, 12)
TREND = "n"

# Pool exog candidates
EXOG_POOL = ["Promo", "SchoolHoliday", "StateHoliday"]


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _combo_label(combo: tuple) -> str:
    """'(none)' hoặc 'P+S+H' để dùng trong bảng/biểu đồ."""
    if not combo:
        return "(none)"
    short = {"Promo": "P", "SchoolHoliday": "S", "StateHoliday": "H"}
    return "+".join(short[v] for v in combo)


# ── Load dữ liệu (1 lần) ─────────────────────────────────────────────────────
print(f"=== SARIMAX Ablation Exog — {ORDER} x {SEASONAL} trend='{TREND}' ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)
train_df, val_df, test_df, _ = preprocess(df, config)

eval_df = val_df if len(val_df) > 0 else test_df
print(f"Data: {len(train_df)} train rows, {len(eval_df)} eval rows")

# ── Sinh full factorial 2^3 = 8 tổ hợp ───────────────────────────────────────
exog_combos = []
for r in range(len(EXOG_POOL) + 1):
    exog_combos.extend(itertools.combinations(EXOG_POOL, r))
print(f"Tổng số tổ hợp exog: {len(exog_combos)} (full factorial 2^{len(EXOG_POOL)})\n")

# ── Loop ablation ────────────────────────────────────────────────────────────
results = []
t0 = time.time()
total = len(exog_combos)

for i, combo in enumerate(exog_combos, 1):
    cfg = copy.deepcopy(config)
    cfg["model"]["order"] = list(ORDER)
    cfg["model"]["seasonal_order"] = list(SEASONAL)
    cfg["model"]["trend"] = TREND
    cfg["model"]["exog_columns"] = list(combo)

    label = _combo_label(combo)
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
            aic_vals = [m.aic for m in model.models.values()]
            bic_vals = [m.bic for m in model.models.values()]
            row = {
                "combo": label,
                "exog_list": ",".join(combo) if combo else "",
                "n_exog": len(combo),
                **metrics,
                "aic_mean": round(float(np.mean(aic_vals)), 2),
                "bic_mean": round(float(np.mean(bic_vals)), 2),
                "n_converged": len(model.models),
                "time_seconds": round(elapsed, 2),
            }
            results.append(row)
    except Exception as e:
        results.append({
            "combo": label, "exog_list": ",".join(combo) if combo else "",
            "n_exog": len(combo),
            "rmspe": float("inf"), "rmse": float("inf"),
            "mae": float("inf"), "mape": float("inf"),
            "aic_mean": float("inf"), "bic_mean": float("inf"),
            "n_converged": 0, "time_seconds": 0,
        })
        print(f"  [{i:2d}/{total}] {label} FAILED: {e}", flush=True)
        continue

    last = results[-1]
    elapsed_total = time.time() - t0
    avg = elapsed_total / i
    eta = avg * (total - i) / 60
    print(
        f"  [{i:2d}/{total}] exog={label:<10s} RMSPE={last['rmspe']:.4f} "
        f"AIC={last['aic_mean']:.1f} conv={last['n_converged']:3d} "
        f"t={last['time_seconds']:5.1f}s | total={elapsed_total/60:5.1f}m ETA={eta:5.1f}m",
        flush=True,
    )

# ── Lưu CSV ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("rmspe").reset_index(drop=True)
csv_path = OUT_DIR / "328_ablation_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV: {csv_path} ({len(results_df)} rows)")

print(f"\n8 tổ hợp exog (sorted by RMSPE):")
print(results_df[["combo", "n_exog", "rmspe", "rmse", "aic_mean", "bic_mean", "n_converged", "time_seconds"]].to_string(index=False))

best = results_df.iloc[0]
baseline_rmspe = results_df[results_df["n_exog"] == 0]["rmspe"].iloc[0]
improve_pct = (baseline_rmspe - best["rmspe"]) / baseline_rmspe * 100
print(f"\nBest: exog=[{best['exog_list']}] RMSPE={best['rmspe']:.4f}")
print(f"Baseline (no exog): RMSPE={baseline_rmspe:.4f}")
print(f"Cải thiện: {improve_pct:.2f}% RMSPE")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng (sorted by RMSPE)
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

table_data = []
for rank, (_, row) in enumerate(results_df.iterrows(), 1):
    table_data.append([
        rank, row["combo"], int(row["n_exog"]),
        f"{row['rmspe']:.4f}", f"{row['rmse']:.2f}",
        f"{row['mae']:.2f}", f"{row['mape']:.4f}",
        f"{row['aic_mean']:.1f}", f"{row['bic_mean']:.1f}",
        int(row["n_converged"]), f"{row['time_seconds']:.1f}",
    ])
col_labels = ["#", "Exog", "n", "RMSPE", "RMSE", "MAE", "MAPE", "AIC", "BIC", "Conv.", "Time (s)"]

fig, ax = plt.subplots(figsize=(15, 4))
ax.axis("off")
table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.04, 0.10, 0.04, 0.08, 0.09, 0.09, 0.09, 0.10, 0.10, 0.06, 0.08])
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
fig.suptitle(
    f"Ablation Exog — SARIMAX{ORDER}x{SEASONAL} trend='{TREND}' "
    f"({total} tổ hợp, P=Promo, S=SchoolHoliday, H=StateHoliday)",
    fontsize=11, fontweight="bold", y=0.96,
)
_save(fig, "328_ablation_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Bar chart RMSPE theo n_exog (so cải thiện marginal)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# (a) RMSPE per combo (sắp xếp theo thứ tự thêm dần biến)
ordered_combos = sorted(exog_combos, key=lambda c: (len(c), c))
ordered_labels = [_combo_label(c) for c in ordered_combos]
rmspe_by_label = dict(zip(results_df["combo"], results_df["rmspe"]))
rmspe_ordered = [rmspe_by_label[lbl] for lbl in ordered_labels]
n_exog_ordered = [len(c) for c in ordered_combos]

cmap = {0: "#34495e", 1: "#3498db", 2: "#27ae60", 3: "#e67e22"}
colors = [cmap[n] for n in n_exog_ordered]
bars = axes[0].bar(range(len(ordered_labels)), rmspe_ordered, color=colors, edgecolor="black")
# Highlight best
best_idx = ordered_labels.index(best["combo"])
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(2.5)
axes[0].set_xticks(range(len(ordered_labels)))
axes[0].set_xticklabels(ordered_labels, rotation=30, ha="right", fontsize=10)
axes[0].axhline(baseline_rmspe, color="gray", linestyle="--", linewidth=1, label=f"Baseline={baseline_rmspe:.4f}")
axes[0].set_ylabel("RMSPE", fontsize=11)
axes[0].set_title("RMSPE per exog combination", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")
axes[0].legend(loc="upper right")
for i, (bar, v) in enumerate(zip(bars, rmspe_ordered)):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=8)

# (b) Mean RMSPE theo n_exog
group_stats = results_df.groupby("n_exog")["rmspe"].agg(["mean", "min", "max"]).reset_index()
xs = group_stats["n_exog"].values
axes[1].errorbar(xs, group_stats["mean"], yerr=[group_stats["mean"] - group_stats["min"],
                                                  group_stats["max"] - group_stats["mean"]],
                 marker="o", capsize=6, linewidth=2, markersize=10, color="steelblue")
axes[1].set_xticks(xs)
axes[1].set_xlabel("Số biến exog", fontsize=11)
axes[1].set_ylabel("RMSPE", fontsize=11)
axes[1].set_title("Marginal effect: mean RMSPE theo |exog|", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)
for x, m in zip(xs, group_stats["mean"]):
    axes[1].annotate(f"{m:.4f}", (x, m), textcoords="offset points", xytext=(8, 8), fontsize=9)

fig.suptitle(
    f"Ablation Exog — SARIMAX{ORDER}x{SEASONAL} trend='{TREND}'",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
_save(fig, "328_ablation_bar.png")

print(f"\nDone. Ablation figures saved to: {OUT_DIR.resolve()}")
