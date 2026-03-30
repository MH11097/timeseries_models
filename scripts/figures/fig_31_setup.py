"""Hình minh họa Section 3.1 — Thiết lập thực nghiệm.

Output (results/figures/):
  31_library_versions.png        — Bảng thư viện sử dụng
  31_forecast_strategies.png     — Sơ đồ 3 chiến lược dự báo: Direct, Multioutput, Recursive

Chạy:
    python scripts/figures/fig_31_setup.py
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
# Plot 1: Library versions table
# ══════════════════════════════════════════════════════════════════════════════
lib_data = [
    ["pandas", "≥ 2.0", "Xử lý dữ liệu dạng bảng"],
    ["numpy", "≥ 1.24", "Tính toán số học"],
    ["scikit-learn", "≥ 1.3", "StandardScaler, utilities"],
    ["statsmodels", "≥ 0.14", "ARIMA, SARIMAX"],
    ["pmdarima", "≥ 2.0", "Auto-ARIMA, ADF test"],
    ["prophet", "≥ 1.1", "Mô hình Prophet"],
    ["xgboost", "≥ 2.0", "Gradient boosting"],
    ["torch", "≥ 2.0", "RNN, LSTM (CUDA 11.8)"],
    ["matplotlib", "≥ 3.7", "Biểu đồ"],
    ["seaborn", "≥ 0.12", "Biểu đồ thống kê"],
    ["typer", "≥ 0.9", "CLI interface"],
    ["pyyaml", "≥ 6.0", "Đọc cấu hình YAML"],
]

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

table = ax.table(
    cellText=lib_data,
    colLabels=["Thư viện", "Phiên bản", "Mục đích"],
    cellLoc="left",
    loc="center",
    colWidths=[0.25, 0.2, 0.55],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Header style
for j in range(3):
    cell = table[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Tô màu theo nhóm thư viện
group_colors = {
    # Data processing (0-2)
    0: "#ebf5fb", 1: "#ebf5fb", 2: "#ebf5fb",
    # Statistical models (3-5)
    3: "#eafaf1", 4: "#eafaf1", 5: "#eafaf1",
    # ML/DL (6-7)
    6: "#fef9e7", 7: "#fef9e7",
    # Visualization (8-9)
    8: "#fdedec", 9: "#fdedec",
    # Utilities (10-11)
    10: "#f4ecf7", 11: "#f4ecf7",
}

for i in range(len(lib_data)):
    color = group_colors.get(i, "white")
    for j in range(3):
        table[i + 1, j].set_facecolor(color)

fig.suptitle("Thư viện chính sử dụng trong nghiên cứu", fontsize=13, fontweight="bold", y=0.97)

# Legend cho nhóm màu
legend_items = [
    mpatches.Patch(color="#ebf5fb", label="Data processing"),
    mpatches.Patch(color="#eafaf1", label="Statistical models"),
    mpatches.Patch(color="#fef9e7", label="ML / Deep Learning"),
    mpatches.Patch(color="#fdedec", label="Visualization"),
    mpatches.Patch(color="#f4ecf7", label="Utilities"),
]
ax.legend(handles=legend_items, loc="lower center", ncol=5, fontsize=8,
          bbox_to_anchor=(0.5, -0.05))

fig.tight_layout()
_save(fig, "31_library_versions.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Forecast strategies diagram (Direct, Multioutput, Recursive)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
fig.suptitle("3 chiến lược dự báo multi-step (H=5 minh họa)", fontsize=14, fontweight="bold")

H = 5  # horizon minh họa
seq_len = 6  # input sequence minh họa
colors_input = "#3498db"
colors_pred = "#e74c3c"
colors_arrow = "#2c3e50"
colors_feedback = "#f39c12"

for ax_idx, (strategy, ax) in enumerate(zip(["Direct", "Multioutput", "Recursive"], axes)):
    ax.set_xlim(-1, seq_len + H + 1)
    ax.set_ylim(-0.5, 2)
    ax.axis("off")
    ax.set_title(f"{strategy}", fontsize=12, fontweight="bold",
                 color=["#3498db", "#2ecc71", "#f39c12"][ax_idx])

    # Input sequence (T-5 đến T)
    for i in range(seq_len):
        rect = mpatches.FancyBboxPatch(
            (i - 0.35, 0.3), 0.7, 0.7,
            boxstyle="round,pad=0.05", facecolor=colors_input, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        label = f"T-{seq_len - 1 - i}" if i < seq_len - 1 else "T"
        ax.text(i, 0.65, label, ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Dấu phân cách
    ax.axvline(seq_len - 0.5, color="#95a5a6", linestyle=":", linewidth=1.5, ymin=0.1, ymax=0.9)

    if strategy == "Direct":
        # 1 model → dự đoán tại đúng T+H
        target = seq_len + H - 1
        rect = mpatches.FancyBboxPatch(
            (target - 0.35, 0.3), 0.7, 0.7,
            boxstyle="round,pad=0.05", facecolor=colors_pred, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(target, 0.65, f"T+{H}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        # Arrow từ model → target
        ax.annotate("", xy=(target - 0.4, 0.65), xytext=(seq_len - 0.1, 0.65),
                    arrowprops=dict(arrowstyle="-|>", color=colors_arrow, lw=2))
        ax.text((seq_len + target) / 2, 1.2, "1 model\ndự đoán T+H",
                ha="center", fontsize=9, style="italic", color="#7f8c8d")

        # Chấm mờ cho T+1..T+4 (không dự đoán)
        for j in range(H - 1):
            ax.plot(seq_len + j, 0.65, "o", color="#bdc3c7", markersize=8)
            ax.text(seq_len + j, 0.25, f"T+{j+1}", ha="center", fontsize=7, color="#bdc3c7")

    elif strategy == "Multioutput":
        # 1 model → output H giá trị cùng lúc
        for j in range(H):
            target = seq_len + j
            rect = mpatches.FancyBboxPatch(
                (target - 0.35, 0.3), 0.7, 0.7,
                boxstyle="round,pad=0.05", facecolor=colors_pred, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(target, 0.65, f"T+{j+1}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        # Arrow fan-out
        for j in range(H):
            target = seq_len + j
            ax.annotate("", xy=(target - 0.4, 0.65), xytext=(seq_len - 0.1, 0.65),
                        arrowprops=dict(arrowstyle="-|>", color=colors_arrow, lw=1.2, alpha=0.6))

        ax.text((seq_len + seq_len + H - 1) / 2, 1.3, "1 model\noutput [T+1, ..., T+H] cùng lúc",
                ha="center", fontsize=9, style="italic", color="#7f8c8d")

    elif strategy == "Recursive":
        # 1 model → T+1, feedback, → T+2, feedback, ...
        for j in range(H):
            target = seq_len + j
            rect = mpatches.FancyBboxPatch(
                (target - 0.35, 0.3), 0.7, 0.7,
                boxstyle="round,pad=0.05", facecolor=colors_pred, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(target, 0.65, f"T+{j+1}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

            # Arrow step-by-step
            if j == 0:
                ax.annotate("", xy=(target - 0.4, 0.65), xytext=(seq_len - 0.1, 0.65),
                            arrowprops=dict(arrowstyle="-|>", color=colors_arrow, lw=2))
            else:
                # Feedback arrow (curved)
                ax.annotate("", xy=(target - 0.4, 0.65), xytext=(target - 1 + 0.4, 0.65),
                            arrowprops=dict(arrowstyle="-|>", color=colors_feedback, lw=1.5,
                                          connectionstyle="arc3,rad=-0.3"))

        ax.text((seq_len + seq_len + H - 1) / 2, 1.3,
                "1 model T+1, feedback prediction → input tiếp theo, lặp H lần",
                ha="center", fontsize=9, style="italic", color="#7f8c8d")

fig.tight_layout()
_save(fig, "31_forecast_strategies.png")

print(f"\nDone. All setup figures saved to: {OUT_DIR.resolve()}")
