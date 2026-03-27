"""ARIMA model - per-store fitting, univariate."""

import logging
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    name = "arima"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.order = tuple(model_cfg.get("order", [1, 1, 1]))
        # trend kiểm soát thành phần hằng số/xu hướng: 'n'=không, 'c'=hằng số, 't'=tuyến tính, 'ct'=cả hai
        self.trend = model_cfg.get("trend", "c")
        self.models: dict[int, StatsARIMA] = {}
        # lưu số điểm train mỗi store → predict() dùng index tuyệt đối thay vì forecast() để đúng giai đoạn
        self.train_lengths: dict[int, int] = {}

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        stores = sorted(train_df["Store"].unique())
        # ARIMA là univariate → phải fit riêng từng store, mỗi store 1 model riêng biệt
        failed = 0
        for store_id in stores:
            store_data = train_df[train_df["Store"] == store_id].sort_values("Date")
            sales = store_data["Sales"].values.astype(float)
            try:
                # một số store có chuỗi sales không ổn định → suppress warning thay vì crash cả pipeline
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = StatsARIMA(
                        sales,
                        order=self.order,
                        trend=self.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    self.models[store_id] = model.fit()
                    self.train_lengths[store_id] = len(sales)
            except Exception as e:
                # store không hội tụ → bỏ qua, predict sẽ trả 0 cho store này
                logger.warning(f"ARIMA failed for store {store_id}: {e}")
                failed += 1

        self._training_time = time.time() - start
        return {"stores_fitted": len(self.models), "stores_failed": failed, "time": self._training_time}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # mỗi store có model riêng → dự đoán từng nhóm store, ghép lại thành array đầy đủ
        # reset index vì sau split train/val/test, index gốc không liên tục → gán vào array positional sẽ out of bounds
        df = df.reset_index(drop=True)
        predictions = np.zeros(len(df))
        for store_id, group in df.groupby("Store"):
            idx = group.index
            n_steps = len(group)
            if store_id in self.models:
                try:
                    # predict() dùng index tuyệt đối: start=train_len = bước đầu tiên sau training
                    # forecast(steps) luôn bắt đầu từ cuối train → sai nếu có gap giữa train/test
                    train_len = self.train_lengths.get(store_id, 0)
                    forecast = self.models[store_id].predict(
                        start=train_len, end=train_len + n_steps - 1
                    )
                    # sales không thể âm → clip về 0
                    predictions[idx] = np.clip(forecast, 0, None)
                except Exception:
                    predictions[idx] = 0
            else:
                # store không fit được ở train → trả 0 (fallback an toàn)
                predictions[idx] = 0
        return predictions
