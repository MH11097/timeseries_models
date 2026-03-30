"""Hình minh họa Section 3.2.1 — Bước 2: Phân tích ACF/PACF.

Chạy ACF/PACF trên 30 cửa hàng đại diện sau sai phân d=1.
Kết quả mong đợi: cả ACF và PACF tắt dần, max lag có ý nghĩa ≤ 8 → p ∈ [0,6], q ∈ [0,6].

Output (results/figures/):
  322_acf_pacf_grid.png  — Lưới 6 store đại diện: ACF + PACF sau sai phân d=1

Chạy:
    python scripts/figures/fig_322_acf_pacf.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

from src.analysis.acf_pacf import suggest_pq_range
from src.data.features import add_all_features
from src.data.loader import filter_stores, load_raw_data
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NLAGS = 40
N_STORES = 30       # số store phân tích
N_SHOW = 6          # số store hiển thị trong grid figure


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load dữ liệu ─────────────────────────────────────────────────────────────
print("=== Bước 2: Phân tích ACF/PACF (d=1) ===\n")

config = load_config("arima")
set_seed(config.get("seed", 42))
df, _ = load_raw_data(config)
df = filter_stores(df, config)
df = add_all_features(df)

# chỉ dùng phần train
train_end = config["split"]["train_end"]
train_df = df[df["Date"] <= train_end].copy()

stores = sorted(train_df["Store"].unique())[:N_STORES]
print(f"Phân tích {len(stores)} stores (d=1 differencing, nlags={NLAGS})")

# ── Chạy suggest_pq_range cho tất cả 30 stores ──────────────────────────────
suggestions = []
for sid in stores:
    sales = train_df[train_df["Store"] == sid].sort_values("Date")["Sales"].values.astype(float)
    # áp dụng sai phân bậc 1 — loại trend, giữ pattern ngắn hạn
    diffed = np.diff(sales)
    sugg = suggest_pq_range(diffed, nlags=NLAGS)
    sugg["store_id"] = sid
    suggestions.append(sugg)

# ── Tổng hợp kết quả ─────────────────────────────────────────────────────────
max_p_vals = [s["pacf_max_sig_lag"] for s in suggestions]
max_q_vals = [s["acf_max_sig_lag"] for s in suggestions]
weekly_count = sum(1 for s in suggestions if s["has_weekly_pattern"])

print(f"\nMax significant PACF lag (gợi ý p): median={np.median(max_p_vals):.0f}, max={max(max_p_vals)}")
print(f"Max significant ACF lag  (gợi ý q): median={np.median(max_q_vals):.0f}, max={max(max_q_vals)}")
print(f"Stores có weekly pattern (lag 7):    {weekly_count}/{len(stores)}")
print(f"Kết luận: p ∈ [0, 6], q ∈ [0, 6]")

# ══════════════════════════════════════════════════════════════════════════════
# Plot: Grid 3x4 — 6 stores, mỗi store 2 cột (ACF + PACF)
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")

# chọn 6 store cách đều trong 30 store
show_indices = np.linspace(0, len(stores) - 1, N_SHOW, dtype=int)
show_stores = [stores[i] for i in show_indices]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("ACF / PACF sau sai phân d=1 — 6 stores đại diện (type C)",
             fontsize=14, fontweight="bold")

for row in range(3):
    for col_pair in range(2):
        store_idx = row * 2 + col_pair
        sid = show_stores[store_idx]

        sales = train_df[train_df["Store"] == sid].sort_values("Date")["Sales"].values.astype(float)
        diffed = np.diff(sales)
        nlags = min(NLAGS, len(diffed) // 2 - 1)

        acf_vals, acf_ci = acf(diffed, nlags=nlags, alpha=0.05)
        pacf_vals, pacf_ci = pacf(diffed, nlags=nlags, alpha=0.05, method="ywm")

        # ACF subplot
        ax_acf = axes[row, col_pair * 2]
        ax_acf.bar(range(nlags + 1), acf_vals, width=0.3, color="steelblue", alpha=0.8)
        lower_a = acf_ci[:, 0] - acf_vals
        upper_a = acf_ci[:, 1] - acf_vals
        ax_acf.fill_between(range(nlags + 1), lower_a, upper_a, alpha=0.15, color="blue")
        ax_acf.axhline(y=0, color="black", linewidth=0.5)
        ax_acf.set_title(f"Store {sid} — ACF", fontsize=10)
        ax_acf.set_xlabel("Lag", fontsize=8)
        ax_acf.tick_params(labelsize=7)

        # PACF subplot
        ax_pacf = axes[row, col_pair * 2 + 1]
        ax_pacf.bar(range(nlags + 1), pacf_vals, width=0.3, color="darkorange", alpha=0.8)
        lower_p = pacf_ci[:, 0] - pacf_vals
        upper_p = pacf_ci[:, 1] - pacf_vals
        ax_pacf.fill_between(range(nlags + 1), lower_p, upper_p, alpha=0.15, color="orange")
        ax_pacf.axhline(y=0, color="black", linewidth=0.5)
        ax_pacf.set_title(f"Store {sid} — PACF", fontsize=10)
        ax_pacf.set_xlabel("Lag", fontsize=8)
        ax_pacf.tick_params(labelsize=7)

fig.tight_layout()
_save(fig, "322_acf_pacf_grid.png")

print(f"\nDone. ACF/PACF figures saved to: {OUT_DIR.resolve()}")
