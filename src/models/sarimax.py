"""SARIMAX model - per-store fitting with seasonal component and exogenous vars."""

import logging
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsSARIMAX

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class SARIMAXModel(BaseModel):
    name = "sarimax"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.order = tuple(model_cfg.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(model_cfg.get("seasonal_order", [1, 1, 1, 7]))
        self.exog_columns = model_cfg.get("exog_columns", ["Promo", "SchoolHoliday"])
        self.max_stores = model_cfg.get("max_stores", config.get("max_stores"))
        self.models: dict[int, StatsSARIMAX] = {}

    def _get_exog(self, df: pd.DataFrame) -> np.ndarray | None:
        # SARIMAX hỗ trợ biến ngoại sinh (Promo, SchoolHoliday) -> lọc cột có sẵn trong df
        available = [c for c in self.exog_columns if c in df.columns]
        if not available:
            return None
        return df[available].values

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        stores = sorted(train_df["Store"].unique())
        if self.max_stores:
            stores = stores[: self.max_stores]

        failed = 0
        for store_id in stores:
            store_data = train_df[train_df["Store"] == store_id].sort_values("Date")
            sales = store_data["Sales"].values
            exog = self._get_exog(store_data)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # enforce_stationarity/invertibility=False -> nới lỏng ràng buộc, tránh lỗi trên dữ liệu thực
                    model = StatsSARIMAX(
                        sales,
                        exog=exog,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    # maxiter=50 giới hạn số vòng lặp -> đổi lấy tốc độ, chấp nhận hội tụ gần đúng
                    self.models[store_id] = model.fit(disp=False, maxiter=50)
            except Exception as e:
                logger.warning(f"SARIMAX failed for store {store_id}: {e}")
                failed += 1

        self._training_time = time.time() - start
        return {"stores_fitted": len(self.models), "stores_failed": failed, "time": self._training_time}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(len(df))
        for store_id, group in df.groupby("Store"):
            idx = group.index
            n_steps = len(group)
            if store_id in self.models:
                try:
                    exog = self._get_exog(group)
                    forecast = self.models[store_id].forecast(steps=n_steps, exog=exog)
                    predictions[idx] = np.clip(forecast, 0, None)
                except Exception:
                    predictions[idx] = 0
            else:
                predictions[idx] = 0
        return predictions
