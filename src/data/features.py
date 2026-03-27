"""Feature engineering for Rossmann Store Sales."""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from Date column."""
    df = df.copy()
    # Date chỉ là 1 cột -> tách thành nhiều feature thời gian để model bắt pattern theo mùa/tuần/ngày
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfMonth"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    # cuối tuần sales thường khác ngày thường -> tạo cờ binary để model phân biệt
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged Sales features per store."""
    if lags is None:
        # 364 = 52 tuần chẵn -> cùng DayOfWeek năm ngoái, bắt yearly seasonality (Christmas, Easter...)
        lags = [1, 7, 14, 30, 364]
    df = df.copy()
    # sales hôm nay phụ thuộc sales quá khứ -> shift theo từng store để tránh data leakage giữa store
    # lag 1,7 bắt trend ngắn hạn; lag 14,30 bắt trend trung hạn; lag 364 bắt yearly seasonality
    for lag in lags:
        shifted = df.groupby("Store")["Sales"].shift(lag)
        df[f"Sales_lag_{lag}"] = shifted
        # lag_364 bị NaN cho ~48% rows (năm đầu tiên mỗi store chưa có dữ liệu 364 ngày trước)
        # Thêm cột binary để model phân biệt "không có dữ liệu" vs "sales thực sự thấp"
        # Không scale cột này (luôn 0/1) -> xem _get_numeric_feature_cols trong preprocessor.py
        if lag == 364:
            df["lag_364_valid"] = shifted.notna().astype(np.int8)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None, stats: list[str] | None = None) -> pd.DataFrame:
    """Add rolling mean/std/median of Sales per store."""
    if windows is None:
        windows = [7, 14, 30]
    if stats is None:
        stats = ["mean", "std", "median"]
    df = df.copy()
    # lag feature chỉ là 1 điểm -> rolling mean/std/median cho xu hướng và mức biến động gần đây
    # shift(1) trước khi rolling để tránh data leakage (không dùng sales hôm nay để dự đoán hôm nay)
    # median robust hơn mean với outlier (ngày lễ spike) -> ít bị kéo bởi ngày bất thường
    for w in windows:
        grouped = df.groupby("Store")["Sales"]
        if "mean" in stats:
            df[f"Sales_rolling_mean_{w}"]   = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        if "std" in stats:
            df[f"Sales_rolling_std_{w}"]    = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        if "median" in stats:
            df[f"Sales_rolling_median_{w}"] = grouped.transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competition-related features."""
    df = df.copy()
    # đối thủ mở càng lâu thì ảnh hưởng lên sales càng ổn định -> tính số tháng kể từ khi mở
    # âm hoặc NaN = chưa có đối thủ -> clip về 0
    df["CompetitionOpenMonths"] = (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 + (
        df["Month"] - df["CompetitionOpenSinceMonth"]
    )
    df["CompetitionOpenMonths"] = (
        df["CompetitionOpenMonths"].clip(lower=0).fillna(0).astype(int)
    )
    return df


def add_promo_distance_features(df: pd.DataFrame, max_days: int = 999) -> pd.DataFrame:
    """Add days since last Promo and days to next Promo per store.

    DaysSinceLastPromo: tăng dần mỗi ngày sau khi promo kết thúc, reset về 0 khi promo bắt đầu.
    DaysToNextPromo:    giảm dần khi promo sắp đến, reset về 0 khi promo đang chạy.
    Lịch Promo trong Rossmann được công bố trước -> DaysToNextPromo KHÔNG phải data leakage.
    max_days: giá trị fill khi không có promo nào trong lịch sử/tương lai.
    """
    df = df.copy()

    def _since(group):
        promo = group["Promo"].values
        out = np.full(len(promo), float(max_days))
        counter = float(max_days)
        for i in range(len(promo)):
            if promo[i] == 1:
                counter = 0.0
            else:
                counter = min(counter + 1.0, max_days)
            out[i] = counter
        return pd.Series(out, index=group.index)

    def _to_next(group):
        promo = group["Promo"].values
        out = np.full(len(promo), float(max_days))
        counter = float(max_days)
        for i in range(len(promo) - 1, -1, -1):
            if promo[i] == 1:
                counter = 0.0
            else:
                counter = min(counter + 1.0, max_days)
            out[i] = counter
        return pd.Series(out, index=group.index)

    df["DaysSinceLastPromo"] = df.groupby("Store", group_keys=False).apply(_since)
    df["DaysToNextPromo"]    = df.groupby("Store", group_keys=False).apply(_to_next)
    return df


def add_store_dow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add average Sales per (Store, DayOfWeek) as feature.

    Bắt pattern: Store A bán trung bình 5000€ vào thứ Hai, 2000€ vào Chủ Nhật.
    Model học được baseline theo ngày trong tuần của từng cửa hàng.
    Stats tính từ chính df được truyền vào -> gọi với full df trước split.
    """
    df = df.copy()
    store_dow_avg = (
        df.groupby(["Store", "DayOfWeek"])["Sales"]
        .mean()
        .rename("StoreDOWAvg")
        .reset_index()
    )
    df = df.merge(store_dow_avg, on=["Store", "DayOfWeek"], how="left")
    df["StoreDOWAvg"] = df["StoreDOWAvg"].fillna(0)
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Promo2 activity flag."""
    df = df.copy()
    # PromoInterval chứa tên tháng viết tắt (Jan,Apr,...) -> ánh xạ số tháng sang tên để so sánh
    month_map = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sept",
        10: "Oct",
        11: "Nov",
        12: "Dec",
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


def add_all_features(df: pd.DataFrame, feature_cfg: dict | None = None) -> pd.DataFrame:
    """Apply feature engineering steps, controlled by feature_cfg.

    feature_cfg là dict từ config["features"] (configs/features.yaml).
    Nếu None → bật tất cả với default settings (backward-compatible).
    """
    cfg = feature_cfg or {}

    if cfg.get("use_time", True):
        df = add_time_features(df)

    if cfg.get("use_lag", True):
        lags = cfg.get("lag_windows") or None   # None → dùng default trong hàm
        df = add_lag_features(df, lags=lags)

    if cfg.get("use_rolling", True):
        windows = cfg.get("rolling_windows") or None
        stats   = cfg.get("rolling_stats") or ["mean", "std", "median"]
        df = add_rolling_features(df, windows=windows, stats=stats)

    if cfg.get("use_competition", True):
        df = add_competition_features(df)

    if cfg.get("use_promo2", True):
        df = add_promo2_features(df)

    if cfg.get("use_promo_distance", True):
        df = add_promo_distance_features(df)

    if cfg.get("use_store_dow", True):
        df = add_store_dow_features(df)

    # lag/rolling tạo NaN ở đầu chuỗi (chưa đủ dữ liệu quá khứ) -> điền 0 để model không bị lỗi
    df = df.fillna(0)
    return df


# Danh sách tất cả cột dẫn xuất từ Sales cần transform cùng
_SALES_DERIVED_COLS = [
    "Sales",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30", "Sales_lag_364",
    "Sales_rolling_mean_7",   "Sales_rolling_mean_14",   "Sales_rolling_mean_30",
    "Sales_rolling_std_7",    "Sales_rolling_std_14",    "Sales_rolling_std_30",
    "Sales_rolling_median_7", "Sales_rolling_median_14", "Sales_rolling_median_30",
    "StoreDOWAvg",  # trung bình Sales theo (Store, DayOfWeek) cũng ở cùng scale Sales
]


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng log1p lên Sales và toàn bộ feature dẫn xuất từ Sales.

    Lợi ích:
    - Phân phối Sales bị lệch phải -> log1p đưa về gần Gaussian hơn
    - MSELoss trên log(Sales) ≈ tối ưu RMSPE (metric chính của project)
    - Gradient flow ổn định hơn, model không bị bias về store bán nhiều

    Dùng log1p (log(1+x)) thay vì log(x) để xử lý Sales=0 an toàn (log1p(0)=0).
    Inverse transform: expm1(x) = exp(x) - 1.
    """
    df = df.copy()
    for col in _SALES_DERIVED_COLS:
        if col in df.columns:
            # clip về 0 trước để tránh log của số âm (từ rolling_std gần 0 hoặc lỗi dữ liệu)
            df[col] = df[col].clip(lower=0)
            df[col] = df[col].transform("log1p")
    return df
