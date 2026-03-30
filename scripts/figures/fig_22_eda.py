"""Hình minh họa Section 2.2 — Phân tích khám phá dữ liệu (EDA).

Output (results/figures/):
  22_missing_values.png          — Tỷ lệ missing values per column
  22_sales_distribution.png      — Phân phối Sales (raw + log1p)
  22_sales_trend.png             — Xu hướng Sales trung bình theo tháng (2013-2015)
  22_seasonality_dow.png         — Boxplot Sales theo ngày trong tuần
  22_seasonality_month.png       — Boxplot Sales theo tháng
  22_promotion_effect.png        — So sánh Sales Promo=0 vs Promo=1
  22_holiday_effect.png          — So sánh Sales theo loại StateHoliday

Chạy:
    python scripts/figures/fig_22_eda.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.loader import load_raw_data
from src.utils.config import load_config

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
config = load_config()
data_cfg = config["data"]
raw_dir = Path(data_cfg["raw_dir"])

# Đọc raw (chưa lọc Open) để phân tích missing values
train_raw = pd.read_csv(raw_dir / data_cfg["train_file"], parse_dates=["Date"], low_memory=False)
store_df = pd.read_csv(raw_dir / data_cfg["store_file"])
merged = train_raw.merge(store_df, on="Store", how="left")

# Bản filtered (Open=1) cho phân tích Sales
df = merged[merged["Open"] == 1].copy()
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

df["Month"] = df["Date"].dt.month
print(f"  Raw: {len(merged):,} rows | Open=1: {len(df):,} rows | {df['Store'].nunique()} stores")
print("Plotting...\n")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Missing values
# ══════════════════════════════════════════════════════════════════════════════
missing_pct = (merged.isnull().sum() / len(merged) * 100).sort_values(ascending=False)
# Chỉ hiển thị cột có missing > 0
missing_pct = missing_pct[missing_pct > 0]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(missing_pct.index, missing_pct.values, color="#e74c3c", edgecolor="white", linewidth=0.5)
for bar, pct in zip(bars, missing_pct.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Tỷ lệ giá trị thiếu (%)")
ax.set_title("Tỷ lệ Missing Values theo biến (merged dataset)", fontsize=13, fontweight="bold")
ax.invert_yaxis()
fig.tight_layout()
_save(fig, "22_missing_values.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Sales distribution (raw + log1p)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phân phối Sales (Open=1)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(df["Sales"], bins=80, color="#3498db", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(df["Sales"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {df['Sales'].mean():,.0f}")
ax.axvline(df["Sales"].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median = {df['Sales'].median():,.0f}")
ax.set_xlabel("Sales (€)")
ax.set_ylabel("Số bản ghi")
ax.set_title("Raw Sales — lệch phải (right-skewed)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)

ax = axes[1]
log_sales = np.log1p(df["Sales"])
ax.hist(log_sales, bins=80, color="#2ecc71", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(log_sales.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {log_sales.mean():.2f}")
ax.axvline(log_sales.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median = {log_sales.median():.2f}")
ax.set_xlabel("log1p(Sales)")
ax.set_ylabel("Số bản ghi")
ax.set_title("Log-transformed — gần Gaussian hơn")
ax.legend(fontsize=9)

fig.tight_layout()
_save(fig, "22_sales_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Sales trend theo tháng
# ══════════════════════════════════════════════════════════════════════════════
# Sample sớm (sau khi Month đã được tạo) để boxplot nhanh hơn
sample = df.sample(n=min(100_000, len(df)), random_state=42)

df["YearMonth"] = df["Date"].dt.to_period("M")
monthly = df.groupby("YearMonth")["Sales"].agg(["mean", "median", "std"]).reset_index()
monthly["YearMonth"] = monthly["YearMonth"].astype(str)

fig, ax = plt.subplots(figsize=(14, 5))
x = range(len(monthly))
ax.plot(x, monthly["mean"], marker="o", markersize=4, linewidth=2, color="#2c3e50", label="Mean Sales")
ax.fill_between(x,
                monthly["mean"] - monthly["std"],
                monthly["mean"] + monthly["std"],
                alpha=0.15, color="#3498db", label="± 1 Std")
ax.plot(x, monthly["median"], linestyle="--", linewidth=1.5, color="#e67e22", label="Median Sales")

ax.set_xticks(x[::3])
ax.set_xticklabels(monthly["YearMonth"].values[::3], rotation=45, ha="right", fontsize=9)
ax.set_xlabel("Tháng")
ax.set_ylabel("Sales (€)")
ax.set_title("Xu hướng doanh số trung bình theo tháng (2013 – 2015)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
_save(fig, "22_sales_trend.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Seasonality — Day of Week
# ══════════════════════════════════════════════════════════════════════════════
dow_labels = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=sample, x="DayOfWeek", y="Sales", ax=ax, hue="DayOfWeek", palette="Set2", legend=False,
            flierprops=dict(marker=".", markersize=1, alpha=0.3))
ax.set_xticks(range(7))
ax.set_xticklabels(dow_labels)
ax.set_xlabel("Ngày trong tuần")
ax.set_ylabel("Sales (€)")
ax.set_title("Phân phối Sales theo ngày trong tuần", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Thêm mean marker
means = df.groupby("DayOfWeek")["Sales"].mean()
ax.scatter(means.index, means.values, color="red", zorder=5, s=50, marker="D", label="Mean")
ax.legend(fontsize=9)
fig.tight_layout()
_save(fig, "22_seasonality_dow.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Seasonality — Month
# ══════════════════════════════════════════════════════════════════════════════
month_labels = ["Th1", "Th2", "Th3", "Th4", "Th5", "Th6", "Th7", "Th8", "Th9", "Th10", "Th11", "Th12"]

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=sample, x="Month", y="Sales", ax=ax, hue="Month", palette="coolwarm", legend=False,
            flierprops=dict(marker=".", markersize=1, alpha=0.3))
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_xlabel("Tháng")
ax.set_ylabel("Sales (€)")
ax.set_title("Phân phối Sales theo tháng — Tháng 12 (Giáng Sinh) cao nhất", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

means_m = df.groupby("Month")["Sales"].mean()
ax.scatter(range(12), means_m.values, color="red", zorder=5, s=50, marker="D", label="Mean")
ax.legend(fontsize=9)
fig.tight_layout()
_save(fig, "22_seasonality_month.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 6: Promotion effect
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Ảnh hưởng của Khuyến mãi (Promo) lên doanh số", fontsize=13, fontweight="bold")

# Boxplot
ax = axes[0]
sns.boxplot(data=sample, x="Promo", y="Sales", ax=ax, palette=["#95a5a6", "#e74c3c"],
            flierprops=dict(marker=".", markersize=1, alpha=0.3))
ax.set_xticklabels(["Không KM (0)", "Có KM (1)"])
ax.set_xlabel("")
ax.set_ylabel("Sales (€)")
ax.set_title("Boxplot Sales theo Promo")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Bar chart mean
ax = axes[1]
promo_means = df.groupby("Promo")["Sales"].mean()
pct_increase = (promo_means[1] - promo_means[0]) / promo_means[0] * 100
bars = ax.bar(["Không KM (0)", "Có KM (1)"], promo_means.values,
              color=["#95a5a6", "#e74c3c"], edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, promo_means.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,.0f} €", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Sales trung bình (€)")
ax.set_title(f"Mean Sales — KM tăng {pct_increase:.0f}%")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig.tight_layout()
_save(fig, "22_promotion_effect.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 7: Holiday effect
# ══════════════════════════════════════════════════════════════════════════════
# StateHoliday lẫn lộn string/int → chuẩn hóa
df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0": "Không lễ", "a": "Lễ công (a)", "b": "Phục Sinh (b)", "c": "Giáng Sinh (c)"})

holiday_order = ["Không lễ", "Lễ công (a)", "Phục Sinh (b)", "Giáng Sinh (c)"]
colors_h = ["#bdc3c7", "#3498db", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Ảnh hưởng của ngày lễ bang (StateHoliday) lên doanh số", fontsize=13, fontweight="bold")

# Boxplot
ax = axes[0]
plot_df = df[df["StateHoliday"].isin(holiday_order)]
sns.boxplot(data=plot_df, x="StateHoliday", y="Sales", order=holiday_order, ax=ax,
            palette=colors_h, flierprops=dict(marker=".", markersize=1, alpha=0.3))
ax.set_xlabel("")
ax.set_ylabel("Sales (€)")
ax.set_title("Boxplot Sales theo loại ngày lễ")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# Bar chart count + mean
ax = axes[1]
holiday_stats = plot_df.groupby("StateHoliday")["Sales"].agg(["mean", "count"]).reindex(holiday_order)
bars = ax.bar(holiday_order, holiday_stats["mean"].values, color=colors_h, edgecolor="white", linewidth=0.5)
for bar, val, cnt in zip(bars, holiday_stats["mean"].values, holiday_stats["count"].values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,.0f} €\n(n={cnt:,})", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Sales trung bình (€)")
ax.set_title("Mean Sales & số bản ghi theo loại ngày lễ")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

fig.tight_layout()
_save(fig, "22_holiday_effect.png")

print(f"\nDone. All EDA figures saved to: {OUT_DIR.resolve()}")
