"""Abstract base class for all forecasting models."""

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_all


class BaseModel(ABC):
    """Base class that all models must implement."""

    name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self._training_time: float = 0.0

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict:
        """Train the model. Returns training info dict."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given DataFrame."""

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model on a DataFrame with Sales column."""
        predictions = self.predict(df)
        y_true = df["Sales"].values
        return evaluate_all(y_true, predictions)

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_result_template(
        self,
        metrics: dict | None = None,
        param_slug: str = "",
        experiment_name: str = "",
    ) -> dict:
        """Generate standardized result JSON."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        run_id = f"{param_slug}__{timestamp}" if param_slug else timestamp
        return {
            "model_name": self.name,
            "run_id": run_id,
            "param_slug": param_slug,
            "experiment_name": experiment_name,
            "timestamp": now.isoformat(),
            "config": self.config,
            "metrics": metrics or {},
            "cv_metrics": {},
            "training_time_seconds": self._training_time,
            "predictions_path": "",
            "metadata": {},
        }

    def save_results(self, results: dict, results_dir: str):
        """Save result JSON to results directory."""
        out_dir = Path(results_dir) / self.name / results["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "result.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return str(out_dir)
