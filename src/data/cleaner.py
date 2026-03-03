"""Data cleaning pipeline for Rossmann Store Sales.

Functional style: each function takes DataFrame, returns DataFrame.
Designed for DS readability — like notebook code converted to .py.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_raw(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, store, test CSVs from raw_dir."""
    raw = Path(raw_dir)
    # dữ liệu thô nằm rời 3 file CSV -> đọc riêng rồi trả về tuple để pipeline xử lý tiếp
    train = pd.read_csv(raw / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(raw / "store.csv")
    test = pd.read_csv(raw / "test.csv", parse_dates=["Date"], low_memory=False)
    return train, store, test


def merge_store(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Left join train/test with store on Store column."""
    # train/test chỉ có Store ID -> ghép metadata cửa hàng (loại, khoảng cách đối thủ...) để làm feature
    merged = df.merge(store_df, on="Store", how="left")
    # sắp theo Store+Date để các bước shift/rolling sau đó hoạt động đúng thứ tự thời gian
    return merged.sort_values(["Store", "Date"]).reset_index(drop=True)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: CompetitionDistance→median, others→0."""
    df = df.copy()
    # ~30% store thiếu CompetitionDistance (chưa có đối thủ) -> điền median để không mất dòng
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            df["CompetitionDistance"].median()
        )
    # tháng/năm mở đối thủ bị NaN khi chưa có đối thủ -> điền 0 để tính CompetitionDaysOpen = 0
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    # tuần/năm bắt đầu Promo2 bị NaN khi store không tham gia -> điền 0 nghĩa là không có promo2
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    # PromoInterval NaN = store không tham gia Promo2 -> chuỗi rỗng để so sánh tháng không lỗi
    if "PromoInterval" in df.columns:
        df["PromoInterval"] = df["PromoInterval"].fillna("")
    # test set có vài dòng thiếu Open -> giả định mở cửa (1) vì đa số store đều mở
    if "Open" in df.columns:
        df["Open"] = df["Open"].fillna(1).astype(int)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove closed stores, zero-sales rows, cap Sales at 25000."""
    df = df.copy()
    # cửa hàng đóng cửa có Sales=0, không phải pattern thực -> loại bỏ tránh model học sai
    if "Open" in df.columns:
        df = df[df["Open"] == 1].reset_index(drop=True)
    if "Sales" in df.columns:
        # một số dòng Sales=0 dù Open=1 (lỗi dữ liệu) -> loại bỏ
        df = df[df["Sales"] > 0].reset_index(drop=True)
        # vài dòng Sales cực cao (>25k) là ngoại lai -> giới hạn trên để giảm variance
        df["Sales"] = df["Sales"].clip(upper=25000)
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize StateHoliday, encode StoreType/Assortment, parse dates."""
    df = df.copy()
    # StateHoliday lẫn lộn string "0","a","b","c" và int 0 -> chuẩn hoá thành int để model dùng được
    if "StateHoliday" in df.columns:
        mapping = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3}
        df["StateHoliday"] = df["StateHoliday"].map(mapping).fillna(0).astype(int)
    # StoreType/Assortment là chữ "a","b","c","d" -> mã hoá số thứ tự để XGBoost/NN xử lý được
    store_type_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    if "StoreType" in df.columns:
        df["StoreType"] = df["StoreType"].map(store_type_map).fillna(0).astype(int)
    assortment_map = {"a": 1, "b": 2, "c": 3}
    if "Assortment" in df.columns:
        df["Assortment"] = df["Assortment"].map(assortment_map).fillna(0).astype(int)
    # Date có thể đang là string sau khi merge -> ép datetime để trích xuất Year/Month/DayOfWeek
    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Add Before/After StateHoliday and SchoolHoliday flags."""
    df = df.copy()
    # sales thường tăng trước ngày lễ và giảm sau ngày lễ -> tạo cờ Before/After để model bắt pattern này
    for col, src in [("StateHoliday", "StateHoliday"), ("SchoolHoliday", "SchoolHoliday")]:
        if src not in df.columns:
            continue
        is_holiday = (df[src] > 0) if df[src].dtype in [int, np.int64, float] else (df[src] != "0")
        # shift theo từng store để tránh lẫn dữ liệu giữa các cửa hàng
        shifted_fwd = is_holiday.groupby(df["Store"]).shift(1).fillna(False).astype(bool)
        shifted_bwd = is_holiday.groupby(df["Store"]).shift(-1).fillna(False).astype(bool)
        # ngày hôm qua là lễ + hôm nay không phải lễ = ngày sau lễ
        df[f"After{col}"] = (shifted_fwd & ~is_holiday).astype(int)
        # ngày mai là lễ + hôm nay không phải lễ = ngày trước lễ
        df[f"Before{col}"] = (shifted_bwd & ~is_holiday).astype(int)
    return df


def add_store_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add SalesPerDay, CustomersPerDay per store (computed on full train)."""
    df = df.copy()
    # mỗi cửa hàng có quy mô khác nhau -> tính trung bình sales/customers làm baseline feature
    if "Sales" in df.columns:
        store_means = df.groupby("Store")["Sales"].mean()
        df["SalesPerDay"] = df["Store"].map(store_means)
    if "Customers" in df.columns:
        cust_means = df.groupby("Store")["Customers"].mean()
        df["CustomersPerDay"] = df["Store"].map(cust_means)
    return df


def add_promo_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add IsPromoMonth from Promo2 and PromoInterval."""
    df = df.copy()
    # Rossmann dùng tên tháng viết tắt trong PromoInterval (Jan,Apr,...) -> ánh xạ số→tên
    month_abbr = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    if "PromoInterval" not in df.columns or "Promo2" not in df.columns:
        df["IsPromoMonth"] = 0
        return df
    # kiểm tra tháng hiện tại có nằm trong danh sách tháng khuyến mãi hay không
    # Promo2=1 + tháng nằm trong PromoInterval -> đang trong đợt khuyến mãi liên tục
    month = df["Date"].dt.month.map(month_abbr)
    intervals = df["PromoInterval"].fillna("").str.split(",")
    df["IsPromoMonth"] = [
        int(p2 == 1 and m in [x.strip() for x in iv])
        for p2, m, iv in zip(df["Promo2"], month, intervals)
    ]
    return df


def add_competition_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add CompetitionDaysOpen (daily granularity, capped at 3*365)."""
    df = df.copy()
    has_year = "CompetitionOpenSinceYear" in df.columns
    has_month = "CompetitionOpenSinceMonth" in df.columns
    if not (has_year and has_month):
        df["CompetitionDaysOpen"] = 0
        return df
    # dữ liệu chỉ có năm+tháng, không có ngày chính xác -> giả định ngày 15 giữa tháng
    comp_open = pd.to_datetime(
        df["CompetitionOpenSinceYear"].astype(int).astype(str) + "-"
        + df["CompetitionOpenSinceMonth"].astype(int).astype(str).str.zfill(2) + "-15",
        errors="coerce",
    )
    # tính số ngày từ khi đối thủ mở cửa -> âm nghĩa là chưa mở, clip về 0
    df["CompetitionDaysOpen"] = (df["Date"] - comp_open).dt.days.clip(lower=0).fillna(0).astype(int)
    # sau 3 năm, ảnh hưởng đối thủ đã ổn định -> giới hạn tránh giá trị quá lớn gây lệch scale
    df["CompetitionDaysOpen"] = df["CompetitionDaysOpen"].clip(upper=3 * 365)
    return df


def add_log_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Add LogSales = log1p(Sales)."""
    df = df.copy()
    # Sales lệch phải mạnh (skewed) -> log1p biến đổi cho phân phối gần chuẩn, giảm ảnh hưởng outlier
    if "Sales" in df.columns:
        df["LogSales"] = np.log1p(df["Sales"])
    return df


def validate(df: pd.DataFrame) -> dict:
    """Run quality checks, print summary, return report dict."""
    # kiểm tra chất lượng sau khi clean: đếm null, range Sales, số store -> phát hiện lỗi pipeline sớm
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "null_total": int(df.isnull().sum().sum()),
        "stores": int(df["Store"].nunique()) if "Store" in df.columns else 0,
    }
    if "Sales" in df.columns:
        report["sales_min"] = float(df["Sales"].min())
        report["sales_max"] = float(df["Sales"].max())
        report["sales_mean"] = float(df["Sales"].mean())
    nulls = {k: v for k, v in report["null_counts"].items() if v > 0}
    print(f"Rows: {report['rows']:,} | Cols: {report['columns']} | Stores: {report['stores']}")
    print(f"Nulls total: {report['null_total']}")
    if nulls:
        print(f"Null columns: {nulls}")
    if "Sales" in df.columns:
        print(f"Sales range: [{report['sales_min']:.0f}, {report['sales_max']:.0f}], mean={report['sales_mean']:.0f}")
    return report


def save(df: pd.DataFrame, path: str, fmt: str = "feather") -> None:
    """Save DataFrame to feather or csv."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "feather":
        df.reset_index(drop=True).to_feather(f"{out}.feather")
    else:
        df.to_csv(f"{out}.csv", index=False)
    print(f"Saved {out}.{fmt} ({len(df):,} rows)")
