"""Hình minh họa Section 2.5 — Quy trình thực nghiệm.

Output (results/figures/):
  25_walk_forward_cv.png         — Sơ đồ walk-forward cross-validation (expanding window)
  25_pipeline_diagram.png        — Sơ đồ pipeline nghiên cứu

Chạy:
    python scripts/figures/fig_25_pipeline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


print("Plotting...\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Walk-forward CV diagram (expanding window)
# ══════════════════════════════════════════════════════════════════════════════
n_splits = 3
total_units = n_splits + 1  # chia data thành n_splits+1 phần

fig, ax = plt.subplots(figsize=(14, 5))

bar_height = 0.5
y_positions = list(range(n_splits - 1, -1, -1))  # fold 1 ở trên

colors_train = "#3498db"
colors_test = "#e74c3c"
colors_unused = "#ecf0f1"

for fold_idx, y in enumerate(y_positions):
    # Expanding: train = 0 → (fold_idx+1)*step, test = tiếp theo 1 step
    train_end_unit = fold_idx + 1
    test_start_unit = train_end_unit
    test_end_unit = train_end_unit + 1

    # Train bar
    ax.barh(y, train_end_unit, left=0, height=bar_height,
            color=colors_train, edgecolor="white", linewidth=1)

    # Test bar
    ax.barh(y, 1, left=test_start_unit, height=bar_height,
            color=colors_test, edgecolor="white", linewidth=1)

    # Unused bar (phần còn lại)
    remaining = total_units - test_end_unit
    if remaining > 0:
        ax.barh(y, remaining, left=test_end_unit, height=bar_height,
                color=colors_unused, edgecolor="white", linewidth=1)

    # Label
    ax.text(-0.15, y, f"Fold {fold_idx + 1}", ha="right", va="center",
            fontsize=11, fontweight="bold")

# Đường thời gian phía dưới
time_labels = ["01/2013", "08/2013", "03/2014", "10/2014", "05/2015", "07/2015"]
# Chia đều total_units điểm
for i, label in enumerate(time_labels[:total_units + 1]):
    x_pos = i * total_units / len(time_labels[:total_units + 1])
    if i < len(time_labels):
        ax.text(i, -0.8, label if i < len(time_labels) else "", ha="center", fontsize=8, color="#666")

ax.set_xlim(-0.5, total_units + 0.3)
ax.set_ylim(-1.2, n_splits)
ax.set_xlabel("Thời gian →", fontsize=11)
ax.set_title("Walk-Forward Cross-Validation (Expanding Window, 3 Folds)",
             fontsize=13, fontweight="bold")

# Legend
legend_handles = [
    mpatches.Patch(color=colors_train, label="Train"),
    mpatches.Patch(color=colors_test, label="Test (eval_days=30)"),
    mpatches.Patch(color=colors_unused, label="Chưa dùng"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
_save(fig, "25_walk_forward_cv.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Pipeline diagram
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

# Các bước pipeline
steps = [
    ("1. Data Loading", "load_raw_data()\nMerge train+store\nLọc Open=1", "#3498db"),
    ("2. Feature Eng.", "32 features:\nTime, Lag, Rolling\nCompetition, Promo", "#2ecc71"),
    ("3. Preprocessing", "Handle missing\nEncode categoricals\nTrain/Test split\nStandardScaler", "#f39c12"),
    ("4. Training", "ARIMA | SARIMAX\nProphet | XGBoost\nRNN | LSTM", "#e74c3c"),
    ("5. Evaluation", "RMSPE (primary)\nRMSE, MAE, MAPE\nWalk-forward CV", "#9b59b6"),
]

n = len(steps)
box_width = 0.15
box_height = 0.6
gap = (1 - n * box_width) / (n + 1)

for i, (title, desc, color) in enumerate(steps):
    x = gap + i * (box_width + gap)
    y = 0.2

    # Box
    rect = mpatches.FancyBboxPatch(
        (x, y), box_width, box_height,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=2, alpha=0.85,
    )
    ax.add_patch(rect)

    # Title
    ax.text(x + box_width / 2, y + box_height - 0.08, title,
            ha="center", va="top", fontsize=10, fontweight="bold", color="white")

    # Description
    ax.text(x + box_width / 2, y + box_height / 2 - 0.05, desc,
            ha="center", va="center", fontsize=8, color="white", linespacing=1.5)

    # Arrow (trừ bước cuối)
    if i < n - 1:
        arrow_x = x + box_width + 0.005
        ax.annotate("", xy=(arrow_x + gap - 0.01, y + box_height / 2),
                    xytext=(arrow_x, y + box_height / 2),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Pipeline nghiên cứu — Từ dữ liệu thô đến đánh giá mô hình",
             fontsize=13, fontweight="bold", y=0.95)
fig.tight_layout()
_save(fig, "25_pipeline_diagram.png")

print(f"\nDone. All pipeline figures saved to: {OUT_DIR.resolve()}")
