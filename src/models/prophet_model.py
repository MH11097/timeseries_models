"""Prophet model - per-store fitting with holidays and regressors."""

import logging
import time
import warnings

import numpy as np
import pandas as pd

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    name = "prophet"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.changepoint_prior_scale = model_cfg.get("changepoint_prior_scale", 0.05)
        self.seasonality_mode = model_cfg.get("seasonality_mode", "multiplicative")
        self.regressors = model_cfg.get("regressors", ["Promo"])
        self.max_stores = model_cfg.get("max_stores", config.get("max_stores"))
        self.models: dict = {}

    def _prepare_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and add regressors for Prophet format."""
        # Prophet yêu cầu cột phải tên "ds" (ngày) và "y" (target) -> đổi tên cho đúng format
        pdf = df[["Date", "Sales"]].copy()
        pdf = pdf.rename(columns={"Date": "ds", "Sales": "y"})
        # thêm các biến ngoại sinh (Promo...) vào df Prophet để dùng làm regressor
        for reg in self.regressors:
            if reg in df.columns:
                pdf[reg] = df[reg].values
        return pdf

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        from prophet import Prophet

        start = time.time()
        stores = sorted(train_df["Store"].unique())
        if self.max_stores:
            stores = stores[: self.max_stores]

        failed = 0
        for store_id in stores:
            store_data = train_df[train_df["Store"] == store_id].sort_values("Date")
            pdf = self._prepare_prophet_df(store_data)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = Prophet(
                        changepoint_prior_scale=self.changepoint_prior_scale,
                        seasonality_mode=self.seasonality_mode,
                    )
                    # Rossmann là chuỗi siêu thị Đức -> thêm ngày lễ Đức để bắt pattern lễ tự động
                    m.add_country_holidays(country_name="DE")
                    # Promo, SchoolHoliday... ảnh hưởng lớn đến sales -> thêm làm external regressor
                    for reg in self.regressors:
                        if reg in pdf.columns:
                            m.add_regressor(reg)
                    m.fit(pdf)
                    self.models[store_id] = m
            except Exception as e:
                logger.warning(f"Prophet failed for store {store_id}: {e}")
                failed += 1

        self._training_time = time.time() - start
        return {"stores_fitted": len(self.models), "stores_failed": failed, "time": self._training_time}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(len(df))
        for store_id, group in df.groupby("Store"):
            idx = group.index
            if store_id in self.models:
                try:
                    future = self._prepare_prophet_df(group)
                    forecast = self.models[store_id].predict(future)
                    predictions[idx] = np.clip(forecast["yhat"].values, 0, None)
                except Exception:
                    predictions[idx] = 0
            else:
                predictions[idx] = 0
        return predictions
