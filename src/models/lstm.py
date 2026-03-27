"""LSTM model - global model with sequence windowing."""

import os
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
    get_loss_fn,
    load_checkpoint,
    prepare_sequences,
    save_checkpoint,
    train_epoch,
)


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # output_size=1 cho direct/recursive; output_size=H cho multioutput
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # chỉ lấy output bước cuối -> tổng hợp thông tin cả chuỗi để dự đoán ngày tiếp theo
        return self.fc(out[:, -1, :])


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, config: dict):
        super().__init__(config)
        model_cfg = config.get("model", {})
        self.hidden_size = model_cfg.get("hidden_size", 128)
        self.num_layers = model_cfg.get("num_layers", 2)
        self.seq_len = model_cfg.get("seq_len", 30)
        self.batch_size = model_cfg.get("batch_size", 256)
        self.epochs = model_cfg.get("epochs", 50)
        self.lr = model_cfg.get("learning_rate", 0.001)
        self.dropout = model_cfg.get("dropout", 0.2)
        self.forecast_horizon = model_cfg.get("forecast_horizon", 1)
        # log_target=True: Sales đã ở log1p space -> predict() cần expm1 để trả về Sales gốc
        self.log_target = config.get("use_log_sales", False)
        # chiến lược dự báo: "direct" | "multioutput" | "recursive"
        self.forecast_strategy = config.get("forecast_strategy", "direct")
        # hàm loss: "mse" | "mae" | "rmspe" | "mape"
        self.loss_fn_name = config.get("loss_fn", "mse")
        # early stopping
        self.patience    = model_cfg.get("patience",     10)
        self.min_delta   = model_cfg.get("min_delta",    0.0)
        self.weight_decay = model_cfg.get("weight_decay", 0.0)  # L2 regularization chống overfit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: LSTMNet | None = None
        self.feature_cols: list[str] = []

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        start = time.time()
        features, targets, self.feature_cols = prepare_sequences(train_df)

        # recursive: train dự đoán bước kế tiếp (H=1); multioutput: output vector H giá trị đồng thời
        _train_h = 1 if self.forecast_strategy == "recursive" else self.forecast_horizon
        _out_sz  = self.forecast_horizon if self.forecast_strategy == "multioutput" else 1

        # store_ids: phân biệt store boundary → loại window vắt qua 2 store khi training
        train_sids = np.asarray(train_df["Store"].values) if "Store" in train_df.columns else None
        train_ds = RossmannSequenceDataset(features, targets, self.seq_len, _train_h, self.forecast_strategy, train_sids)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_df is not None and len(val_df) > 0:
            val_feat, val_tgt, _ = prepare_sequences(val_df, self.feature_cols)
            val_sids = np.asarray(val_df["Store"].values) if "Store" in val_df.columns else None
            val_ds = RossmannSequenceDataset(val_feat, val_tgt, self.seq_len, _train_h, self.forecast_strategy, val_sids)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self.net = LSTMNet(len(self.feature_cols), self.hidden_size, self.num_layers, self.dropout, _out_sz).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion  = get_loss_fn(self.loss_fn_name)
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # Hỗ trợ resume từ checkpoint
        start_epoch = 0
        resume_from = self.config.get("resume_from")
        if resume_from and os.path.exists(resume_from):
            print(f"Loading checkpoint from {resume_from}...")
            checkpoint = load_checkpoint(resume_from, device=str(self.device))
            self.net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            early_stop.best_loss = checkpoint["best_loss"]
            early_stop.counter = checkpoint["patience_counter"]
            self.feature_cols = checkpoint["feature_cols"]
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

        # Thư mục lưu checkpoint
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_latest.pth")

        print(f"Training LSTM | Strategy: {self.forecast_strategy} | Device: {self.device} | Epochs: {self.epochs}, LR: {self.lr}, Hidden: {self.hidden_size}, Horizon: +{self.forecast_horizon} days, LogSales: {self.log_target}", flush=True)
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        checkpoint_best_path = os.path.join(checkpoint_dir, f"{self.name}_best.pth")
        for epoch in range(start_epoch, self.epochs):
            train_loss = train_epoch(self.net, train_loader, optimizer, criterion, self.device)
            train_losses.append(train_loss)

            val_loss = None
            if val_loader:
                val_loss = evaluate_epoch(self.net, val_loader, criterion, self.device)
                val_losses.append(val_loss)
                # lưu best model riêng khi val_loss cải thiện vượt min_delta
                # (dùng cùng tiêu chí với EarlyStopping để tránh best_epoch = stop_epoch do micro-improvements)
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    save_checkpoint(checkpoint_best_path, self.net, optimizer, epoch, early_stop, self.feature_cols)
                if early_stop(val_loss):
                    print(f"Early stopping at epoch {epoch + 1} (patience {early_stop.patience} reached).")
                    break

            # In tiến trình mỗi epoch
            best_marker = " *" if val_loss is not None and val_loss == best_val_loss else ""
            if val_loss is not None:
                print(f"Epoch {epoch + 1:3d}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}{best_marker}", flush=True)
            else:
                print(f"Epoch {epoch + 1:3d}/{self.epochs} | Train Loss: {train_loss:.6f}", flush=True)

            # Lưu checkpoint sau mỗi epoch
            save_checkpoint(checkpoint_path, self.net, optimizer, epoch, early_stop, self.feature_cols)

        # khôi phục weights tại epoch tốt nhất trước khi lưu model.pkl
        if val_loader and os.path.exists(checkpoint_best_path):
            best_ckpt = load_checkpoint(checkpoint_best_path, device=str(self.device))
            self.net.load_state_dict(best_ckpt["model_state_dict"])
            print(f"[best] Restored weights at best epoch {best_epoch} (val_loss={best_val_loss:.6f}).")

        self._training_time = time.time() - start
        print(f"Training complete in {self._training_time:.1f}s")
        return {
            "training_time": self._training_time,
            "n_samples": len(train_df),
            "epochs_trained": epoch + 1,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Dự đoán Sales. Trả về array shape (n,).

        - direct:      dự đoán tại đúng bước H. Nhanh nhất.
        - multioutput: lấy output cuối (bước H) của vector H giá trị.
        - recursive:   lăn bánh H bước từ mỗi vị trí. Chậm hơn O(H) lần.

        Lưu ý: predict() luôn chạy per-store để tránh window vắt qua ranh giới store-boundary.
        """
        raw = self._predict_stores(df, all_horizons=False)
        if self.log_target:
            raw = np.expm1(raw)
        return np.clip(raw, 0, None)

    def predict_all_horizons(self, df: pd.DataFrame) -> np.ndarray:
        """Dự đoán toàn bộ bước 1..H. Trả về array shape (n, H).

        Hàng j chứa H dự đoán được thực hiện từ cửa sổ kết thúc tại ngày j.
        Cột h chứa dự đoán tại bước h+1 (0-indexed).
        Chỉ hoạt động với strategy='multioutput' hoặc 'recursive'.
        """
        if self.forecast_strategy == "direct":
            raise ValueError(
                "predict_all_horizons() không dùng được với strategy='direct'. "
                "Train H model riêng (forecast_horizon=1..H) hoặc đổi sang 'multioutput'/'recursive'."
            )
        result = self._predict_stores(df, all_horizons=True)
        if self.log_target:
            result = np.expm1(result)
        return np.clip(result, 0, None)

    # ─── prediction helpers ────────────────────────────────────────────────────────────────
    def _predict_stores(self, df: pd.DataFrame, all_horizons: bool) -> np.ndarray:
        """Chạy prediction per-store để tránh window vắt qua ranh giới store-boundary.

        Mỗi store được xử lý độc lập: features của store N không bị trộn với store N+1.
        """
        n = len(df)
        H = self.forecast_horizon
        result = np.zeros((n, H)) if all_horizons else np.zeros(n)

        try:
            sales_idx: int | None = self.feature_cols.index("Sales")
        except ValueError:
            sales_idx = None

        # reset index để có tọ độ positional 0..n-1 — nếu không có Store col, chạy toàn số trực tiếp
        df_pos = df.reset_index(drop=True)
        if "Store" not in df_pos.columns:
            sub_feats, _, _ = prepare_sequences(df_pos, self.feature_cols)
            if all_horizons:
                result = self._run_multioutput_all(sub_feats, H) if self.forecast_strategy == "multioutput" \
                    else self._run_recursive_all(sub_feats, sales_idx, H)
            elif self.forecast_strategy == "recursive":
                result = self._run_recursive(sub_feats, sales_idx, H)
            elif self.forecast_strategy == "multioutput":
                result = self._run_multioutput_single(sub_feats, H)
            else:
                result = self._run_direct(sub_feats, H)
            return result

        for store_id in df_pos["Store"].unique():
            mask = (df_pos["Store"] == store_id).values  # boolean mask, length n
            pos = np.where(mask)[0]                       # positional indices trong 0..n-1
            store_df = df_pos.iloc[pos]                   # sub-DataFrame của store này

            sub_feats, _, _ = prepare_sequences(store_df, self.feature_cols)

            if all_horizons:
                if self.forecast_strategy == "multioutput":
                    sub_preds = self._run_multioutput_all(sub_feats, H)
                else:
                    sub_preds = self._run_recursive_all(sub_feats, sales_idx, H)
            elif self.forecast_strategy == "recursive":
                sub_preds = self._run_recursive(sub_feats, sales_idx, H)
            elif self.forecast_strategy == "multioutput":
                sub_preds = self._run_multioutput_single(sub_feats, H)
            else:
                sub_preds = self._run_direct(sub_feats, H)

            result[pos] = sub_preds  # đưa kết quả về đúng vị trí trong mảng toàn bộ

        return result
    @staticmethod
    def _pad_preds(raw: np.ndarray, n: int, offset: int) -> np.ndarray:
        """Pad raw prediction array to full length n with given leading offset.

        offset = seq_len + H - 1 cho recursive/multioutput/direct:
          raw[i] predicts day (offset + i), so raw[i] → full[offset + i].
        Khi offset >= n (store quá ít ngày để cho phép even một prediction hợp lệ),
          trả về zeros — không có predictions chính xác cho store này.
        """
        full = np.zeros(n)
        if len(raw) > 0 and offset < n:
            take = n - offset   # số predictions hợp lệ (raw[take:] overshoot khỏi full)
            full[offset:] = np.clip(raw[:take], 0, None)
            full[:offset] = raw[0]
        return full

    def _run_direct(self, features: np.ndarray, H: int) -> np.ndarray:
        ds = RossmannSequenceDataset(features, np.zeros(len(features)), self.seq_len, H, "direct")
        loader = DataLoader(ds, batch_size=self.batch_size)
        self.net.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                preds.append(np.atleast_1d(self.net(X.to(self.device)).squeeze(-1).cpu().numpy()))
        raw = np.concatenate(preds) if preds else np.array([])
        return self._pad_preds(raw, len(features), self.seq_len + H - 1)

    def _run_multioutput_single(self, features: np.ndarray, H: int) -> np.ndarray:
        """Multioutput → lấy bước H (cột cuối) để so sánh với direct."""
        ds = RossmannSequenceDataset(features, np.zeros(len(features)), self.seq_len, H, "multioutput")
        loader = DataLoader(ds, batch_size=self.batch_size)
        self.net.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                preds.append(self.net(X.to(self.device)).cpu().numpy()[:, -1])  # bước H cuối
        raw = np.concatenate(preds) if preds else np.array([])
        return self._pad_preds(raw, len(features), self.seq_len + H - 1)

    def _run_recursive(self, features: np.ndarray, sales_idx: int | None, H: int) -> np.ndarray:
        """Lăn bánh H bước, trả về (n,) tại bước H."""
        n, CHUNK = len(features), max(self.batch_size, 256)
        n_samp = n - self.seq_len
        if n_samp <= 0:
            return np.zeros(n)
        finals: list[np.ndarray] = []
        self.net.eval()
        with torch.no_grad():
            for cs in range(0, n_samp, CHUNK):
                wins = np.stack([features[i : i + self.seq_len] for i in range(cs, min(cs + CHUNK, n_samp))]).copy()
                sp: np.ndarray | None = None
                for step in range(H):
                    sp = np.atleast_1d(self.net(torch.FloatTensor(wins).to(self.device)).squeeze(-1).cpu().numpy())
                    if step < H - 1:
                        nr = wins[:, -1, :].copy()
                        if sales_idx is not None:
                            nr[:, sales_idx] = sp
                        wins = np.concatenate([wins[:, 1:, :], nr[:, np.newaxis, :]], axis=1)
                finals.append(sp)  # type: ignore[arg-type]
        raw = np.concatenate(finals) if finals else np.array([])
        # raw[i] = prediction từ window kết thúc tại (seq_len+i-1), lăn H bước → target là ngày (seq_len+i+H-1)
        # Dùng cùng offset (seq_len+H-1) như multioutput/direct để full[j] = prediction cho ngày j
        return self._pad_preds(raw, n, self.seq_len + H - 1)

    def _run_multioutput_all(self, features: np.ndarray, H: int) -> np.ndarray:
        """Multioutput → trả về (n, H) toàn bộ các bước."""
        n = len(features)
        ds = RossmannSequenceDataset(features, np.zeros(n), self.seq_len, H, "multioutput")
        loader = DataLoader(ds, batch_size=self.batch_size)
        self.net.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for X, _ in loader:
                chunks.append(self.net(X.to(self.device)).cpu().numpy())  # (batch, H)
        raw = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, H))
        off = self.seq_len
        result = np.zeros((n, H))
        result[off : off + len(raw)] = np.clip(raw[: n - off], 0, None)
        if len(raw) > 0:
            result[:off] = raw[0]
        return result

    def _run_recursive_all(self, features: np.ndarray, sales_idx: int | None, H: int) -> np.ndarray:
        """Recursive → trả về (n, H) toàn bộ các bước."""
        n, CHUNK = len(features), max(self.batch_size, 256)
        n_samp = n - self.seq_len
        if n_samp <= 0:
            return np.zeros((n, H))
        chunks: list[np.ndarray] = []
        self.net.eval()
        with torch.no_grad():
            for cs in range(0, n_samp, CHUNK):
                wins = np.stack([features[i : i + self.seq_len] for i in range(cs, min(cs + CHUNK, n_samp))]).copy()
                cp = np.zeros((len(wins), H))
                for step in range(H):
                    sp = np.atleast_1d(self.net(torch.FloatTensor(wins).to(self.device)).squeeze(-1).cpu().numpy())
                    cp[:, step] = sp
                    if step < H - 1:
                        nr = wins[:, -1, :].copy()
                        if sales_idx is not None:
                            nr[:, sales_idx] = sp
                        wins = np.concatenate([wins[:, 1:, :], nr[:, np.newaxis, :]], axis=1)
                chunks.append(cp)
        raw = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, H))
        result = np.zeros((n, H))
        result[self.seq_len : self.seq_len + len(raw)] = np.clip(raw[: n - self.seq_len], 0, None)
        if len(raw) > 0:
            result[: self.seq_len] = raw[0]
        return result
