"""Data preprocessing pipeline for Rossmann Store Sales."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Full preprocessing pipeline: handle missing values, split, scale.

    Args:
        df: Merged train+store DataFrame
        config: Config dict with split dates, model.skip_scaling để bỏ qua StandardScaler

    Returns:
        (train_df, val_df, test_df, scaler)
    """
    df = df.copy()
    df = _handle_missing_values(df)
    df = _encode_categoricals(df)

    # time series không shuffle random -> split theo thời gian để tránh data leakage từ tương lai
    split_cfg = config["split"]
    train_df = df[df["Date"] <= split_cfg["train_end"]].copy()

    # Val: optional — tiểu luận chỉ cần train+test, bỏ val để đơn giản hoá
    if "val_start" in split_cfg and "val_end" in split_cfg:
        val_df = df[(df["Date"] >= split_cfg["val_start"]) & (df["Date"] <= split_cfg["val_end"])].copy()
    else:
        val_df = pd.DataFrame(columns=df.columns)

    # Test: thêm test_end filter để giới hạn đúng 30 ngày cuối
    test_mask = df["Date"] >= split_cfg["test_start"]
    if "test_end" in split_cfg:
        test_mask &= df["Date"] <= split_cfg["test_end"]
    test_df = df[test_mask].copy()

    # Prophet/ARIMA tự xử lý trend+seasonality → skip scaling để giữ giá trị gốc cho regressor (Promo 0/1...)
    # Các model khác (XGBoost, NN) cần chuẩn hoá feature về mean=0, std=1 vì scale khác nhau
    skip_scaling = config.get("model", {}).get("skip_scaling", False)
    numeric_cols = _get_numeric_feature_cols(train_df)
    scaler = StandardScaler()
    if numeric_cols and not skip_scaling:
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        # Guard: val/test có thể rỗng khi config không định nghĩa val split
        if len(val_df) > 0:
            val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
        if len(test_df) > 0:
            test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, val_df, test_df, scaler


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in Rossmann dataset."""
    # NaN = chưa có đối thủ -> điền max (xa nhất) thay vì drop dòng, giữ được dữ liệu
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].max())

    # tháng/năm mở đối thủ NaN = chưa có -> điền 0 để CompetitionOpenMonths tính ra 0
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # store không tham gia Promo2 -> NaN điền 0/rỗng để tránh lỗi khi tính Promo2Active
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    if "PromoInterval" in df.columns:
        df["PromoInterval"] = df["PromoInterval"].fillna("")

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as integers."""
    # StateHoliday lẫn lộn string/int -> chuẩn hoá: 0=không lễ, 1/2/3=loại lễ a/b/c
    if "StateHoliday" in df.columns:
        mapping = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3}
        df["StateHoliday"] = df["StateHoliday"].map(mapping).fillna(0).astype(int)

    # StoreType/Assortment là chữ cái -> label encode thành số để XGBoost/NN xử lý được
    for col in ["StoreType", "Assortment"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map({v: i for i, v in enumerate(sorted(df[col].astype(str).unique()))})

    return df


def _get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric columns suitable for scaling (exclude target, IDs, dates, and ordinal flags)."""
    # Sales/Customers là target, Store là ID, Open là cờ -> không scale, chỉ scale feature thật sự
    # StateHoliday có 99.9% = 0 và chỉ 0.1% = 1/2/3 -> std cực nhỏ (~0.05)
    # StandardScaler biến giá trị 3 (Christmas) thành +63 -> LSTM gate saturation (gradient = 0)
    # Binary flags (lag_364_valid) không cần scale -> già trị đã là 0/1
    _NO_SCALE = {"Sales", "Customers", "Store", "Date", "Open", "StateHoliday", "lag_364_valid"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in _NO_SCALE]
