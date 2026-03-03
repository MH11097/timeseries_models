"""Feature engineering for Rossmann Store Sales."""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from Date column."""
    df = df.copy()
    # Date chỉ là 1 cột -> tách thành nhiều feature thời gian để model bắt pattern theo mùa/tuần/ngày
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfMonth"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    # cuối tuần sales thường khác ngày thường -> tạo cờ binary để model phân biệt
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged Sales features per store."""
    if lags is None:
        lags = [1, 7, 14, 30]
    df = df.copy()
    # sales hôm nay phụ thuộc sales quá khứ -> shift theo từng store để tránh data leakage giữa store
    # lag 1,7 bắt trend ngắn hạn; lag 14,30 bắt trend trung hạn
    for lag in lags:
        df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling mean/std of Sales per store."""
    if windows is None:
        windows = [7, 14, 30]
    df = df.copy()
    # lag feature chỉ là 1 điểm -> rolling mean/std cho xu hướng trung bình và mức biến động gần đây
    # shift(1) trước khi rolling để tránh data leakage (không dùng sales hôm nay để dự đoán hôm nay)
    for w in windows:
        grouped = df.groupby("Store")["Sales"]
        df[f"Sales_rolling_mean_{w}"] = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"Sales_rolling_std_{w}"] = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competition-related features."""
    df = df.copy()
    # đối thủ mở càng lâu thì ảnh hưởng lên sales càng ổn định -> tính số tháng kể từ khi mở
    # âm hoặc NaN = chưa có đối thủ -> clip về 0
    df["CompetitionOpenMonths"] = (
        (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 + (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].clip(lower=0).fillna(0).astype(int)
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Promo2 activity flag."""
    df = df.copy()
    # PromoInterval chứa tên tháng viết tắt (Jan,Apr,...) -> ánh xạ số tháng sang tên để so sánh
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    def _is_promo2_active(row):
        # Promo2=0 hoặc PromoInterval rỗng -> store không tham gia khuyến mãi liên tục
        if row["Promo2"] == 0 or not row["PromoInterval"]:
            return 0
        months = [m.strip() for m in str(row["PromoInterval"]).split(",")]
        current_month = month_map.get(row["Month"], "")
        return 1 if current_month in months else 0

    # kiểm tra từng dòng xem tháng đó có đang trong đợt Promo2 không -> cờ 0/1
    df["Promo2Active"] = df.apply(_is_promo2_active, axis=1)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_competition_features(df)
    df = add_promo2_features(df)
    # lag/rolling tạo NaN ở đầu chuỗi (chưa đủ dữ liệu quá khứ) -> điền 0 để model không bị lỗi
    df = df.fillna(0)
    return df
