"""RNN model - global model with sequence windowing."""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseModel
from src.models.torch_utils import (
    EarlyStopping,
    RossmannSequenceDataset,
    evaluate_epoch,
    prepare_sequences,
    train_epoch,
)


class RNNNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        # chỉ lấy hidden state của bước thời gian cuối cùng -> dự đoán Sales ngày tiếp theo
        return self.fc(out[:, -1, :])


class RNNModel(BaseModel):
    name = "rnn"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.hidden_size = model_cfg.get("hidden_size", 64)
        self.num_layers = model_cfg.get("num_layers", 2)
        self.seq_len = model_cfg.get("seq_len", 30)
        self.batch_size = model_cfg.get("batch_size", 256)
        self.epochs = model_cfg.get("epochs", 50)
        self.lr = model_cfg.get("learning_rate", 0.001)
        self.dropout = model_cfg.get("dropout", 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: RNNNet | None = None
        self.feature_cols: list[str] = []

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        features, targets, self.feature_cols = prepare_sequences(train_df)

        train_ds = RossmannSequenceDataset(features, targets, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_df is not None and len(val_df) > 0:
            val_feat, val_tgt, _ = prepare_sequences(val_df, self.feature_cols)
            val_ds = RossmannSequenceDataset(val_feat, val_tgt, self.seq_len)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self.net = RNNNet(len(self.feature_cols), self.hidden_size, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=10)

        for epoch in range(self.epochs):
            train_epoch(self.net, train_loader, optimizer, criterion, self.device)
            if val_loader:
                val_loss = evaluate_epoch(self.net, val_loader, criterion, self.device)
                if early_stop(val_loss):
                    break

        self._training_time = time.time() - start
        return {"epochs_trained": epoch + 1, "time": self._training_time}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features, _, _ = prepare_sequences(df, self.feature_cols)
        dataset = RossmannSequenceDataset(features, np.zeros(len(features)), self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.net.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                output = self.net(X_batch).squeeze().cpu().numpy()
                preds.append(output)

        if preds:
            raw = np.concatenate(preds)
        else:
            raw = np.array([])

        # seq_len ngày đầu không có đủ cửa sổ để dự đoán -> pad bằng giá trị dự đoán đầu tiên
        full_preds = np.zeros(len(df))
        full_preds[self.seq_len :] = np.clip(raw[: len(df) - self.seq_len], 0, None)
        if len(raw) > 0:
            full_preds[: self.seq_len] = raw[0]
        return full_preds
