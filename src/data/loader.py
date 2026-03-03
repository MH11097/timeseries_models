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


def sample_stores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Sample subset of stores if max_stores is set in config."""
    # 1115 store train rất lâu (nhất là ARIMA/Prophet fit từng store) -> giới hạn số store để dev/test nhanh
    max_stores = config.get("max_stores")
    if max_stores is not None:
        stores = sorted(df["Store"].unique())[:max_stores]
        df = df[df["Store"].isin(stores)].reset_index(drop=True)
    return df
