"""Shared PyTorch utilities for RNN and LSTM models."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class RossmannSequenceDataset(Dataset):
    """Sequence windowing dataset for time series models.

    Args:
        forecast_horizon: số ngày dự đoán vào tương lai
        strategy: "direct" | "multioutput" | "recursive"
            - direct/recursive: target là 1 scalar tại T+H
            - multioutput:       target là vector H giá trị [T+1, ..., T+H]
        store_ids: array (n,) chứa Store ID cho từng hàng. Nếu được cung cấp,
            chỉ các window mà đầu và cuối thuộc cùng 1 store mới được dùng,
            tránh window vắt qua biên store-boundary gây nhiễu nghiêm trọng.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 30,
        forecast_horizon: int = 1,
        strategy: str = "direct",
        store_ids: np.ndarray | None = None,    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.strategy = strategy

        n = len(features)
        full_window = seq_len + forecast_horizon  # rows cần cho mỗi sample
        if store_ids is not None and len(store_ids) == n:
            # data sort theo [Store, Date] -> nếu start và end cùng store thì middle cũng cùng store
            valid = [
                i for i in range(n - full_window + 1)
                if store_ids[i] == store_ids[i + full_window - 1]
            ]
            self.valid_indices = np.array(valid, dtype=np.int64)
        else:
            # không có store_ids: hành vi cũ (toàn bộ window)
            self.valid_indices = np.arange(max(0, n - full_window + 1), dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = int(self.valid_indices[idx])
        x = self.features[start : start + self.seq_len]
        if self.strategy == "multioutput":
            # trả về H target liên tiếp: [T+1, T+2, ..., T+H]
            y = self.targets[start + self.seq_len : start + self.seq_len + self.forecast_horizon]
        else:
            # direct hoặc recursive: 1 scalar tại T+H
            y = self.targets[start + self.seq_len + self.forecast_horizon - 1]
        return x, y


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        # val_loss cải thiện -> reset bộ đếm; không cải thiện liên tục patience lần -> dừng train tránh overfit
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _rmspe_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Root Mean Squared Percentage Error.
    eps=1.0 tránh division-by-zero cho các ngày store đóng cửa (Sales=0).
    """
    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.sqrt(torch.mean(((output - target) / denom) ** 2))


def _mape_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((output - target) / denom))


def get_loss_fn(name: str):
    """Factory trả về loss function từ tên chuỗi.

    Args:
        name: "mse" | "mae" | "rmspe" | "mape"

    Returns:
        Callable(output, target) -> scalar tensor
    """
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "rmspe":
        return _rmspe_loss
    elif name == "mape":
        return _mape_loss
    else:
        raise ValueError(f"loss_fn không hợp lệ: '{name}'. Chọn: mse, mae, rmspe, mape")


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        # squeeze(-1): an toàn cho cả direct (output_size=1 → (B,)) và multioutput (output_size=H → (B,H) không đổi)
        output = model(X_batch).squeeze(-1)
        loss = criterion(output, y_batch)
        loss.backward()
        # RNN/LSTM dễ bị exploding gradient -> clip norm về tối đa 1.0 để ổn định training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate one epoch and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


NUMERIC_FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "Year", "Month", "WeekOfYear", "DayOfMonth", "IsWeekend",
    # lag features: short-term trend + same week last year (yearly seasonality)
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30", "Sales_lag_364",
    # lag_364_valid: 1 nếu có dữ liệu thực 364 ngày trước, 0 nếu store chưa đủ 1 năm lịch sử
    # Giúp model phân biệt "Sales_lag_364=0 vì thực sự thấp" vs "=0 vì không có dữ liệu"
    "lag_364_valid",
    # rolling: mean (trend), std (volatility), median (robust to spikes)
    "Sales_rolling_mean_7",   "Sales_rolling_mean_14",   "Sales_rolling_mean_30",
    "Sales_rolling_std_7",    "Sales_rolling_std_14",    "Sales_rolling_std_30",
    "Sales_rolling_median_7", "Sales_rolling_median_14", "Sales_rolling_median_30",
    "CompetitionOpenMonths", "Promo2Active",
    # promo distance: context về vị trí trong chu kỳ khuyến mãi
    "DaysSinceLastPromo", "DaysToNextPromo",
    # store × day-of-week baseline: mỗi store có pattern tuần riêng
    "StoreDOWAvg",
]


def prepare_sequences(df, feature_cols=None, log_target: bool = False):
    """Prepare feature matrix and target from DataFrame.

    Args:
        log_target: nếu True, target (Sales) đã ở log1p space (do apply_log_transform chạy trước).
                    Flag này chỉ để các model biết cần inverse transform khi predict.
                    Hàm này KHÔNG tự apply log nữa vì transform đã xảy ra ở features pipeline.
    """
    # lần đầu (train) tự chọn feature có sẵn; lần sau (val/test) dùng đúng danh sách đã chọn
    if feature_cols is None:
        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    # ép float32 tiết kiệm bộ nhớ GPU, NaN điền 0 để tensor không chứa nan
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    targets = df["Sales"].values.astype(np.float32)
    return features, targets, feature_cols


def save_checkpoint(checkpoint_path: str, model, optimizer, epoch: int, early_stop: EarlyStopping, feature_cols: list[str]):
    """
    Lưu checkpoint để có thể resume training.
    
    Args:
        checkpoint_path: đường dẫn file lưu checkpoint (.pth)
        model: PyTorch model (RNNNet/LSTMNet)
        optimizer: PyTorch optimizer
        epoch: epoch hiện tại
        early_stop: EarlyStopping object
        feature_cols: danh sách feature columns
    """
    import os
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": early_stop.best_loss,
        "patience_counter": early_stop.counter,
        "feature_cols": feature_cols,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Tải checkpoint để resume training.
    
    Args:
        checkpoint_path: đường dẫn file checkpoint (.pth)
        device: thiết bị ("cpu" hoặc "cuda")
    
    Returns:
        dict với các key: model_state_dict, optimizer_state_dict, epoch, best_loss, patience_counter, feature_cols
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint
