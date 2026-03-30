"""Hình minh họa Section 3.2.2 — Bước 1: Xác định chu kỳ mùa vụ S.

So sánh 3 giá trị S (chu kỳ mùa vụ) bằng walk-forward CV trên cấu hình SARIMA cơ sở:
  - S=6:  chu kỳ tuần (6 ngày kinh doanh, đã loại Chủ nhật)
  - S=12: chu kỳ 2 tuần
  - S=30: chu kỳ tháng

Cấu hình cơ sở: SARIMA(1,1,1)(1,0,1,S), trend='c', không exog.
CV: 3 folds, expanding window, eval_days=30.

Output (results/figures/):
  3260_seasonal_period_cv.csv          — Bảng kết quả S | RMSPE | RMSE | ...
  3260_seasonal_period_table.png       — Bảng so sánh RMSPE theo S
  3260_seasonal_period_bar.png         — Bar chart RMSPE theo S

Chạy:
    python scripts/figures/fig_3260_seasonal_period_cv.py
"""

import warnings
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

# cấu hình SARIMA cơ sở — chưa tune, chỉ dùng để so sánh S
BASE_ORDER = [1, 1, 1]
BASE_TREND = "c"
S_CANDIDATES = [6, 12, 30]
N_SPLITS = 3
EVAL_DAYS = 30


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== Bước 1: Xác định chu kỳ mùa vụ S — CV so sánh S ∈ {6, 12, 30} ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)

n_stores = df["Store"].nunique()
print(f"Stores type C: {n_stores}")
print(f"Cấu hình cơ sở: SARIMA({BASE_ORDER[0]},{BASE_ORDER[1]},{BASE_ORDER[2]})(1,0,1,S)")
print(f"CV: {N_SPLITS} folds, expanding window, eval_days={EVAL_DAYS}\n")

# ── So sánh từng giá trị S ──────────────────────────────────────────────────
# chạy CV cho mỗi S, ghi nhận RMSPE + số store hội tụ thành công
comparison_rows = []

for s_val in S_CANDIDATES:
    print(f"── S={s_val} ", "─" * 50)

    # cập nhật config cho mỗi S — SARIMA thuần (không exog)
    config["model"]["order"] = BASE_ORDER
    config["model"]["seasonal_order"] = [1, 0, 1, s_val]
    config["model"]["trend"] = BASE_TREND
    config["model"]["exog_columns"] = []  # SARIMA thuần, không dùng biến ngoại sinh
    # S lớn → state space lớn → cần nhiều iterations hơn để hội tụ
    config["model"]["maxiter"] = 300

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = walk_forward_cv(
                model_class=SARIMAXModel,
                config=config,
                df=df,
                n_splits=N_SPLITS,
                expanding=True,
                eval_days=EVAL_DAYS,
            )

        agg = cv_results["aggregated"]
        folds = cv_results["folds"]

        # in kết quả per-fold
        for f in folds:
            print(f"  Fold {f['fold']}: RMSPE={f['rmspe']:.4f}, time={f['training_time_seconds']:.1f}s")

        rmspe_mean = agg["rmspe_mean"]
        rmspe_std = agg["rmspe_std"]
        rmse_mean = agg["rmse_mean"]
        mae_mean = agg["mae_mean"]

        # kiểm tra hội tụ: nếu RMSPE quá cao (>1.0) hoặc NaN → coi là unstable
        note = ""
        if rmspe_mean > 1.0 or np.isnan(rmspe_mean):
            note = "unstable"
        elif rmspe_std / rmspe_mean > 0.5:
            note = "high variance"

        print(f"  → RMSPE = {rmspe_mean:.4f} ± {rmspe_std:.4f} {note}")

        comparison_rows.append({
            "S": s_val,
            "RMSPE_mean": round(rmspe_mean, 4),
            "RMSPE_std": round(rmspe_std, 4),
            "RMSE_mean": round(rmse_mean, 2),
            "MAE_mean": round(mae_mean, 2),
            "note": note,
        })

    except Exception as e:
        print(f"  → FAILED: {e}")
        comparison_rows.append({
            "S": s_val,
            "RMSPE_mean": np.nan,
            "RMSPE_std": np.nan,
            "RMSE_mean": np.nan,
            "MAE_mean": np.nan,
            "note": "failed",
        })

    print()

# ── Tổng hợp kết quả ────────────────────────────────────────────────────────
comp_df = pd.DataFrame(comparison_rows)

# chọn S tối ưu: RMSPE thấp nhất trong các S stable
valid = comp_df[comp_df["note"] != "failed"].copy()
if len(valid) > 0:
    best_idx = valid["RMSPE_mean"].idxmin()
    best_s = int(valid.loc[best_idx, "S"])
    best_rmspe = valid.loc[best_idx, "RMSPE_mean"]
else:
    best_s = S_CANDIDATES[0]
    best_rmspe = np.nan

print("=" * 60)
print("Kết quả so sánh chu kỳ mùa vụ S:")
print(f"{'S':>4} | {'RMSPE':>14} | {'RMSE':>10} | {'MAE':>10} | Ghi chú")
print("-" * 60)
for _, row in comp_df.iterrows():
    rmspe_str = f"{row['RMSPE_mean']:.4f} ± {row['RMSPE_std']:.4f}" if not np.isnan(row["RMSPE_mean"]) else "N/A"
    rmse_str = f"{row['RMSE_mean']:.2f}" if not np.isnan(row["RMSE_mean"]) else "N/A"
    mae_str = f"{row['MAE_mean']:.2f}" if not np.isnan(row["MAE_mean"]) else "N/A"
    note = row["note"] if row["note"] else ""
    print(f"{int(row['S']):>4} | {rmspe_str:>14} | {rmse_str:>10} | {mae_str:>10} | {note}")
print("-" * 60)
print(f"Kết luận: Chọn S={best_s} (RMSPE={best_rmspe:.4f}) — RMSPE thấp nhất, hội tụ ổn định.")

# ── Lưu CSV ──────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "3260_seasonal_period_cv.csv"
comp_df.to_csv(csv_path, index=False)
print(f"  CSV: {csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Bảng so sánh
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

# ghi chú cho từng S — giải thích ý nghĩa nghiệp vụ
s_labels = {6: "Chu kỳ tuần (6 ngày KD)", 12: "Chu kỳ 2 tuần", 30: "Chu kỳ tháng"}

table_data = []
for _, row in comp_df.iterrows():
    s_val = int(row["S"])
    rmspe_str = f"{row['RMSPE_mean']:.4f} ± {row['RMSPE_std']:.4f}" if not np.isnan(row["RMSPE_mean"]) else "N/A"
    rmse_str = f"{row['RMSE_mean']:.2f}" if not np.isnan(row["RMSE_mean"]) else "N/A"
    mae_str = f"{row['MAE_mean']:.2f}" if not np.isnan(row["MAE_mean"]) else "N/A"
    note = row["note"] if row["note"] else ""
    label = s_labels.get(s_val, "")
    table_data.append([s_val, label, rmspe_str, rmse_str, mae_str, note])

col_labels = ["S", "Ghi chú", "RMSPE (mean ± std)", "RMSE", "MAE", "Trạng thái"]

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis("off")

table = ax.table(
    cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center",
    colWidths=[0.06, 0.22, 0.22, 0.12, 0.12, 0.12],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# header style — nền tối, chữ trắng
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# tô màu: dòng có RMSPE thấp nhất → xanh lá, còn lại xen kẽ
for i in range(len(table_data)):
    s_val = table_data[i][0]
    if s_val == best_s:
        color = "#d5f5e3"  # xanh lá nhạt — best
    else:
        color = "#f8f9fa" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)
        if s_val == best_s:
            table[i + 1, j].set_text_props(fontweight="bold")

fig.suptitle(
    f"So sánh chu kỳ mùa vụ S — SARIMA({BASE_ORDER[0]},{BASE_ORDER[1]},{BASE_ORDER[2]})(1,0,1,S)\n"
    f"CV: {N_SPLITS} folds, expanding, eval_days={EVAL_DAYS}",
    fontsize=12, fontweight="bold", y=1.0,
)
_save(fig, "3260_seasonal_period_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Bar chart RMSPE theo S
# ══════════════════════════════════════════════════════════════════════════════
# chỉ plot các S có kết quả hợp lệ
plot_df = comp_df[~comp_df["RMSPE_mean"].isna()].copy()

if len(plot_df) > 0:
    x_labels = [f"S={int(row['S'])}" for _, row in plot_df.iterrows()]
    rmspe_vals = plot_df["RMSPE_mean"].values
    rmspe_stds = plot_df["RMSPE_std"].values

    # màu: best → xanh lá, còn lại → xanh dương nhạt
    colors = ["#2ecc71" if int(row["S"]) == best_s else "#85c1e9" for _, row in plot_df.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        x_labels, rmspe_vals, yerr=rmspe_stds, capsize=6,
        color=colors, alpha=0.9, edgecolor="white", linewidth=2,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "color": "#2c3e50"},
    )

    # ghi giá trị trên mỗi bar
    for bar, val, std in zip(bars, rmspe_vals, rmspe_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("RMSPE", fontsize=12)
    ax.set_title(
        f"RMSPE theo chu kỳ mùa vụ S — SARIMA(1,1,1)(1,0,1,S)\n"
        f"CV: {N_SPLITS} folds, expanding, eval_days={EVAL_DAYS}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "3260_seasonal_period_bar.png")

print(f"\nDone. Seasonal period comparison saved to: {OUT_DIR.resolve()}")
