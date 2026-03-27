"""Data loading utilities for Rossmann Store Sales dataset."""

from pathlib import Path

import pandas as pd


def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge train + store data.

    Args:
        config: Configuration dict with data.raw_dir, data.train_file, data.store_file

    Returns:
        (merged_df, store_df) - merged train+store data and raw store data
    """
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])

    train_df = pd.read_csv(raw_dir / data_cfg["train_file"], parse_dates=["Date"], low_memory=False)
    store_df = pd.read_csv(raw_dir / data_cfg["store_file"])

    # train chỉ có Store ID -> ghép metadata cửa hàng để có thêm feature cho model
    merged_df = train_df.merge(store_df, on="Store", how="left")

    # sắp theo Store+Date để đảm bảo thứ tự thời gian cho lag/rolling features
    merged_df = merged_df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # cửa hàng đóng cửa có Sales=0, không phản ánh nhu cầu thực -> loại bỏ
    merged_df = merged_df[merged_df["Open"] == 1].reset_index(drop=True)

    # StateHoliday/StoreType/Assortment là categorical -> ép kiểu giúp pandas tối ưu bộ nhớ
    cat_cols = ["StateHoliday", "StoreType", "Assortment"]
    for col in cat_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype("category")

    return merged_df, store_df


def load_cleaned_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-cleaned CSV files.

    Args:
        config: Configuration dict with data.cleaned_dir

    Returns:
        (train_df, store_df) - cleaned train and store data
    """
    cleaned_dir = Path(config["data"]["cleaned_dir"])
    # đọc dữ liệu đã qua pipeline clean_data.py -> bỏ qua bước clean, tiết kiệm thời gian
    train_df = pd.read_csv(cleaned_dir / "train_cleaned.csv", low_memory=False)
    store_df = pd.read_csv(cleaned_dir / "store_cleaned.csv", low_memory=False)

    # CSV không giữ kiểu datetime -> ép lại để các bước feature engineering dùng .dt accessor
    if "Date" in train_df.columns:
        train_df["Date"] = pd.to_datetime(train_df["Date"])

    train_df = train_df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return train_df, store_df


def filter_stores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Lọc theo loại cửa hàng nếu store_type được set trong config.

    Dùng cho per-store models (ARIMA, SARIMAX, Prophet) → chỉ giữ stores cùng loại
    để kiểm soát biến khi so sánh model. Ví dụ store_type="c" → 148 stores type C.
    """
    store_type = config.get("store_type")
    if store_type is not None:
        df = df[df["StoreType"] == store_type].reset_index(drop=True)
    return df
