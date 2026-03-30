"""Hình minh họa Section 3.2.2 — Bước 2: Xác định D, P, Q (tham số mùa vụ).

Với S=12 đã chọn ở Bước 1, tiến hành:
  1. Kiểm định seasonal unit root (OCSB + Canova-Hansen) → xác định D ∈ {0, 1}
  2. Phân tích ACF/PACF tại lag mùa vụ (12, 24, 36) → gợi ý phạm vi P, Q

Output (results/figures/):
  3261_seasonal_d_summary.csv          — Bảng D gợi ý per-store (OCSB + CH)
  3261_seasonal_d_pie.png              — Biểu đồ tròn phân bố D
  3261_seasonal_acf_pacf_grid.png      — ACF/PACF tại seasonal lags cho 6 stores
  3261_seasonal_pq_summary.csv         — Bảng tổng hợp P, Q gợi ý

Chạy:
    python scripts/figures/fig_3261_seasonal_dpq.py
"""

import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima.arima import CHTest, OCSBTest, nsdiffs
from statsmodels.tsa.stattools import acf, pacf

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

S = 12  # chu kỳ mùa vụ từ Bước 1
NLAGS = 48  # cần >= 3*S = 36 để thấy 3 seasonal lags
N_STORES_ACF = 30  # số store phân tích ACF/PACF
N_SHOW = 6  # số store hiển thị trong grid


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print(f"=== Bước 2: Xác định D, P, Q (S={S}) ===\n")

config = load_config("sarimax")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)

# chỉ dùng phần train → không leak val/test vào phân tích
train_end = config["split"]["train_end"]
train_df = df[df["Date"] <= train_end].copy()

stores = sorted(train_df["Store"].unique())
print(f"Stores type C: {len(stores)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: Kiểm định seasonal unit root → xác định D
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Phần 1: Seasonal unit root test (OCSB + Canova-Hansen, m={S}) ──\n")

# chạy 2 test trên mỗi store: OCSB (mặc định pmdarima) và Canova-Hansen
# 2 test bổ sung lẫn nhau: OCSB test H0=no seasonal unit root, CH test H0=seasonal stationarity
d_results = []
d_counter = Counter()

for sid in stores:
    sales = train_df[train_df["Store"] == sid].sort_values("Date")["Sales"].values.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # nsdiffs với OCSB test (mặc định) → D gợi ý
            d_ocsb = nsdiffs(sales, m=S, test="ocsb", max_D=2)
            # nsdiffs với Canova-Hansen test → D gợi ý
            d_ch = nsdiffs(sales, m=S, test="ch", max_D=2)

        # kết hợp 2 test: chọn D cao hơn (conservative — tránh bỏ sót seasonal unit root)
        d_suggested = max(d_ocsb, d_ch)
        d_counter[d_suggested] += 1
        d_results.append({
            "store_id": sid,
            "D_ocsb": d_ocsb,
            "D_ch": d_ch,
            "D_suggested": d_suggested,
        })
    except Exception as e:
        d_results.append({
            "store_id": sid,
            "D_ocsb": None,
            "D_ch": None,
            "D_suggested": None,
        })

d_df = pd.DataFrame(d_results)
valid_d = d_df.dropna(subset=["D_suggested"])

# thống kê phân bố D
print(f"Kết quả (n={len(valid_d)}):")
for d_val in sorted(d_counter.keys()):
    cnt = d_counter[d_val]
    pct = cnt / len(valid_d) * 100
    print(f"  D={d_val}: {cnt} stores ({pct:.1f}%)")

# chọn D phổ biến nhất
best_D = d_counter.most_common(1)[0][0]
print(f"\nKết luận: D={best_D} được chọn (chiếm đa số stores).")

# lưu CSV
csv_d = OUT_DIR / "3261_seasonal_d_summary.csv"
d_df.to_csv(csv_d, index=False)
print(f"  CSV: {csv_d}")

# ── Plot: Biểu đồ tròn phân bố D ────────────────────────────────────────────
print("\nPlotting D distribution...")

colors_map = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}
labels, sizes, colors = [], [], []
for d_val in sorted(d_counter.keys()):
    cnt = d_counter[d_val]
    pct = cnt / len(valid_d) * 100
    labels.append(f"D={d_val}\n{cnt} stores ({pct:.1f}%)")
    sizes.append(cnt)
    colors.append(colors_map.get(d_val, "#95a5a6"))

fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(
    sizes, labels=labels, colors=colors, autopct="", startangle=90,
    textprops={"fontsize": 11}, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
ax.set_title(
    f"Phân bố bậc sai phân mùa vụ D — 148 cửa hàng type C\n"
    f"(Kiểm định kép OCSB + Canova-Hansen, S={S})",
    fontsize=13, fontweight="bold", pad=15,
)
_save(fig, "3261_seasonal_d_pie.png")

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: ACF/PACF tại seasonal lags → gợi ý P, Q
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── Phần 2: ACF/PACF tại seasonal lags (S={S}) ──\n")

# áp dụng regular diff (d=1) trước, rồi phân tích ACF/PACF tại lag S, 2S, 3S
# nếu D=1 ở phần 1 → cũng thử seasonal diff để so sánh pattern
analysis_stores = stores[:N_STORES_ACF]
print(f"Phân tích {len(analysis_stores)} stores (d=1, nlags={NLAGS})")

pq_results = []
# đếm pattern ACF/PACF tại seasonal lags để tổng hợp xu hướng P, Q
acf_seasonal_patterns = []  # "cutoff" hoặc "decay"
pacf_seasonal_patterns = []

for sid in analysis_stores:
    sales = train_df[train_df["Store"] == sid].sort_values("Date")["Sales"].values.astype(float)
    # sai phân bậc 1 — loại trend, giữ seasonal + noise
    diffed = np.diff(sales)
    nlags_use = min(NLAGS, len(diffed) // 2 - 1)

    acf_vals, acf_ci = acf(diffed, nlags=nlags_use, alpha=0.05)
    pacf_vals, pacf_ci = pacf(diffed, nlags=nlags_use, alpha=0.05, method="ywm")

    # kiểm tra ACF/PACF tại lag S, 2S, 3S
    seasonal_lags = [k * S for k in range(1, 4) if k * S <= nlags_use]
    acf_sig_at_seasonal = []
    pacf_sig_at_seasonal = []

    for lag in seasonal_lags:
        # ACF significant tại lag mùa vụ?
        acf_sig = acf_ci[lag, 0] > 0 or acf_ci[lag, 1] < 0
        acf_sig_at_seasonal.append(acf_sig)
        # PACF significant tại lag mùa vụ?
        pacf_sig = pacf_ci[lag, 0] > 0 or pacf_ci[lag, 1] < 0
        pacf_sig_at_seasonal.append(pacf_sig)

    # xác định pattern: "cutoff" nếu chỉ lag S significant, lag 2S+ không
    # "decay" nếu nhiều lag mùa vụ đều significant (tắt dần)
    n_acf_sig = sum(acf_sig_at_seasonal)
    n_pacf_sig = sum(pacf_sig_at_seasonal)

    if n_acf_sig <= 1:
        acf_pattern = "cutoff"
    else:
        acf_pattern = "decay"

    if n_pacf_sig <= 1:
        pacf_pattern = "cutoff"
    else:
        pacf_pattern = "decay"

    acf_seasonal_patterns.append(acf_pattern)
    pacf_seasonal_patterns.append(pacf_pattern)

    pq_results.append({
        "store_id": sid,
        "acf_sig_lags": [lag for lag, sig in zip(seasonal_lags, acf_sig_at_seasonal) if sig],
        "pacf_sig_lags": [lag for lag, sig in zip(seasonal_lags, pacf_sig_at_seasonal) if sig],
        "n_acf_sig": n_acf_sig,
        "n_pacf_sig": n_pacf_sig,
        "acf_pattern": acf_pattern,
        "pacf_pattern": pacf_pattern,
    })

# ── Tổng hợp pattern ────────────────────────────────────────────────────────
acf_cutoff_pct = sum(1 for p in acf_seasonal_patterns if p == "cutoff") / len(acf_seasonal_patterns) * 100
acf_decay_pct = 100 - acf_cutoff_pct
pacf_cutoff_pct = sum(1 for p in pacf_seasonal_patterns if p == "cutoff") / len(pacf_seasonal_patterns) * 100
pacf_decay_pct = 100 - pacf_cutoff_pct

print(f"\nPattern ACF tại seasonal lags (S={S}, 2S={2*S}, 3S={3*S}):")
print(f"  Cutoff (≤1 lag sig): {acf_cutoff_pct:.1f}% stores")
print(f"  Decay  (≥2 lags sig): {acf_decay_pct:.1f}% stores")
print(f"\nPattern PACF tại seasonal lags:")
print(f"  Cutoff (≤1 lag sig): {pacf_cutoff_pct:.1f}% stores")
print(f"  Decay  (≥2 lags sig): {pacf_decay_pct:.1f}% stores")

# gợi ý P, Q dựa trên Box-Jenkins cho seasonal component:
# ACF cutoff + PACF decay → SMA dominant → Q=1, P=0..2
# PACF cutoff + ACF decay → SAR dominant → P=1, Q=0..2
# cả hai decay → SARMA hỗn hợp → P ∈ [0,2], Q ∈ [0,2]
print(f"\nGợi ý phạm vi tham số mùa vụ (Box-Jenkins):")
if acf_cutoff_pct > 60:
    print(f"  ACF cutoff chiếm đa số → SMA dominant → Q ∈ [0, 1], P ∈ [0, 2]")
    suggested_P = [0, 1, 2]
    suggested_Q = [0, 1]
elif pacf_cutoff_pct > 60:
    print(f"  PACF cutoff chiếm đa số → SAR dominant → P ∈ [0, 1], Q ∈ [0, 2]")
    suggested_P = [0, 1]
    suggested_Q = [0, 1, 2]
else:
    print(f"  Cả ACF và PACF đều decay → SARMA hỗn hợp → P ∈ [0, 2], Q ∈ [0, 2]")
    suggested_P = [0, 1, 2]
    suggested_Q = [0, 1, 2]

print(f"  → P ∈ {suggested_P}, D={best_D}, Q ∈ {suggested_Q}, S={S}")

# lưu CSV
pq_df = pd.DataFrame([{
    "store_id": r["store_id"],
    "acf_sig_seasonal": str(r["acf_sig_lags"]),
    "pacf_sig_seasonal": str(r["pacf_sig_lags"]),
    "acf_pattern": r["acf_pattern"],
    "pacf_pattern": r["pacf_pattern"],
} for r in pq_results])
csv_pq = OUT_DIR / "3261_seasonal_pq_summary.csv"
pq_df.to_csv(csv_pq, index=False)
print(f"  CSV: {csv_pq}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot: ACF/PACF grid tại seasonal lags — 6 stores đại diện
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting seasonal ACF/PACF grid...")

show_indices = np.linspace(0, len(analysis_stores) - 1, N_SHOW, dtype=int)
show_stores = [analysis_stores[i] for i in show_indices]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle(
    f"ACF / PACF sau sai phân d=1 — 6 stores đại diện (S={S})\n"
    f"Đường đỏ đứt: seasonal lags ({S}, {2*S}, {3*S})",
    fontsize=14, fontweight="bold",
)

for row in range(3):
    for col_pair in range(2):
        store_idx = row * 2 + col_pair
        sid = show_stores[store_idx]

        sales = train_df[train_df["Store"] == sid].sort_values("Date")["Sales"].values.astype(float)
        diffed = np.diff(sales)
        nlags_use = min(NLAGS, len(diffed) // 2 - 1)

        acf_vals, acf_ci = acf(diffed, nlags=nlags_use, alpha=0.05)
        pacf_vals, pacf_ci = pacf(diffed, nlags=nlags_use, alpha=0.05, method="ywm")

        # ACF subplot
        ax_acf = axes[row, col_pair * 2]
        ax_acf.bar(range(nlags_use + 1), acf_vals, width=0.3, color="steelblue", alpha=0.8)
        lower_a = acf_ci[:, 0] - acf_vals
        upper_a = acf_ci[:, 1] - acf_vals
        ax_acf.fill_between(range(nlags_use + 1), lower_a, upper_a, alpha=0.15, color="blue")
        ax_acf.axhline(y=0, color="black", linewidth=0.5)
        # đánh dấu seasonal lags bằng đường đỏ đứt
        for k in range(1, 4):
            lag = k * S
            if lag <= nlags_use:
                ax_acf.axvline(x=lag, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax_acf.set_title(f"Store {sid} — ACF", fontsize=10)
        ax_acf.set_xlabel("Lag", fontsize=8)
        ax_acf.tick_params(labelsize=7)

        # PACF subplot
        ax_pacf = axes[row, col_pair * 2 + 1]
        ax_pacf.bar(range(nlags_use + 1), pacf_vals, width=0.3, color="darkorange", alpha=0.8)
        lower_p = pacf_ci[:, 0] - pacf_vals
        upper_p = pacf_ci[:, 1] - pacf_vals
        ax_pacf.fill_between(range(nlags_use + 1), lower_p, upper_p, alpha=0.15, color="orange")
        ax_pacf.axhline(y=0, color="black", linewidth=0.5)
        for k in range(1, 4):
            lag = k * S
            if lag <= nlags_use:
                ax_pacf.axvline(x=lag, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax_pacf.set_title(f"Store {sid} — PACF", fontsize=10)
        ax_pacf.set_xlabel("Lag", fontsize=8)
        ax_pacf.tick_params(labelsize=7)

fig.tight_layout()
_save(fig, "3261_seasonal_acf_pacf_grid.png")

# ══════════════════════════════════════════════════════════════════════════════
# Bảng tổng hợp cuối cùng
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"TỔNG HỢP — Tham số mùa vụ SARIMAX (S={S}):")
print(f"  D = {best_D} (OCSB + CH, {d_counter[best_D]}/{len(valid_d)} stores)")
print(f"  P ∈ {suggested_P}")
print(f"  Q ∈ {suggested_Q}")
print(f"  ACF tại seasonal lags: cutoff {acf_cutoff_pct:.1f}% / decay {acf_decay_pct:.1f}%")
print(f"  PACF tại seasonal lags: cutoff {pacf_cutoff_pct:.1f}% / decay {pacf_decay_pct:.1f}%")
print(f"{'=' * 60}")

print(f"\nDone. Seasonal D/P/Q figures saved to: {OUT_DIR.resolve()}")
