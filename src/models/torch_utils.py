"""Shared PyTorch utilities for RNN and LSTM models."""

import numpy as np
import torch
from torch.utils.data import Dataset


class RossmannSequenceDataset(Dataset):
    """Sequence windowing dataset for time series models."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len

    def __len__(self):
        # RNN/LSTM cần seq_len ngày liên tiếp làm input -> số sample = tổng dòng - seq_len
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        # cắt cửa sổ trượt: seq_len ngày liên tiếp làm X, ngày tiếp theo làm target Y
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
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
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


NUMERIC_FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "Year", "Month", "WeekOfYear", "DayOfMonth", "IsWeekend",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30",
    "Sales_rolling_mean_7", "Sales_rolling_mean_14", "Sales_rolling_mean_30",
    "Sales_rolling_std_7", "Sales_rolling_std_14", "Sales_rolling_std_30",
    "CompetitionOpenMonths", "Promo2Active",
]


def prepare_sequences(df, feature_cols=None):
    """Prepare feature matrix and target from DataFrame."""
    # lần đầu (train) tự chọn feature có sẵn; lần sau (val/test) dùng đúng danh sách đã chọn
    if feature_cols is None:
        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    # ép float32 tiết kiệm bộ nhớ GPU, NaN điền 0 để tensor không chứa nan
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    targets = df["Sales"].values.astype(np.float32)
    return features, targets, feature_cols
