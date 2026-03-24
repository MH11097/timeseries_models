"""XGBoost model - global model with Store as feature."""

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel

# danh sách feature đầu vào cho XGBoost - gồm store info, thời gian, lag/rolling, khuyến mãi, đối thủ
# thứ tự không ảnh hưởng kết quả (tree-based), nhưng nhóm theo ý nghĩa để dễ đọc
FEATURE_COLS = [
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "Year",
    "Month",
    "WeekOfYear",
    "DayOfMonth",
    "IsWeekend",
    "Sales_lag_1",
    "Sales_rolling_mean_7",
    "Sales_rolling_mean_14",
    "Sales_rolling_std_7",
    "Sales_rolling_std_30",
    "CompetitionOpenMonths",
]


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.n_estimators = model_cfg.get("n_estimators", 1000)
        self.max_depth = model_cfg.get("max_depth", 7)
        self.learning_rate = model_cfg.get("learning_rate", 0.1)
        self.subsample = model_cfg.get("subsample", 0.8)
        self.colsample_bytree = model_cfg.get("colsample_bytree", 0.8)
        self.early_stopping_rounds = model_cfg.get("early_stopping_rounds", 50)
        self.model: xgb.XGBRegressor | None = None
        self.feature_cols: list[str] = []

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in FEATURE_COLS if c in df.columns]
        self.feature_cols = available
        return df[available].fillna(0)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()

        X_train = self._get_features(train_df)
        y_train = train_df["Sales"].values

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.config.get("seed", 42),
            n_jobs=-1,
        )

        # nếu có validation set -> dùng eval_set để XGBoost theo dõi loss trên val mỗi round
        fit_params = {}
        if val_df is not None and len(val_df) > 0:
            X_val = self._get_features(val_df)
            y_val = val_df["Sales"].values
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self._training_time = time.time() - start
        return {"n_features": len(self.feature_cols), "time": self._training_time}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._get_features(df)
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, None)
