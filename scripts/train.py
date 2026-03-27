"""CLI script for training models.

Hai chế độ chạy:
- CLI mode: `python scripts/train.py --model arima` → dùng cho CI/automation
- Interactive mode: `python scripts/train.py` (không args) → hiện menu chọn model + tham số
"""

import ast
import time
from pathlib import Path

import typer

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_cleaned_data, load_raw_data
from src.data.preprocessor import preprocess
from src.models import MODEL_REGISTRY, get_model_class
from src.utils.config import load_config, make_param_slug
from src.utils.seed import set_seed
from src.analysis.residual_diagnostics import diagnose_residuals
from src.utils.visualization import plot_predictions, plot_residuals

app = typer.Typer(help="Train time series forecasting models.")


def _parse_overrides(set_values: list[str] | None) -> dict:
    """Parse key=value overrides từ CLI. Hỗ trợ list, int, float, null."""
    # cho phép ghi đè config từ CLI, vd: --set store_type=c --set model.order=[2,1,1]
    if not set_values:
        return {}
    overrides = {}
    for item in set_values:
        key, _, val = item.partition("=")
        # thử parse giá trị: "50" → int, "[1,1,1]" → list, "null" → None
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            if val.lower() == "null":
                val = None
        overrides[key] = val
    return overrides


def _evaluate_split(model, split_df, split_name: str, model_name: str, out_dir: str):
    """Đánh giá model trên 1 split (val/test), in metrics + vẽ biểu đồ vào out_dir."""
    if len(split_df) == 0:
        return None
    try:
        metrics = model.evaluate(split_df)
    except Exception as e:
        typer.echo(f"  Warning: evaluate {split_name} failed: {e}", err=True)
        return None
    typer.echo(f"  {split_name} metrics: {metrics}")
    try:
        preds = model.predict(split_df)
        y_true = split_df["Sales"].values
        plot_predictions(y_true, preds, title=f"{model_name} - {split_name}", save_path=str(Path(out_dir) / f"{split_name}_predictions.png"))
        plot_residuals(y_true, preds, title=f"{model_name} - {split_name}", save_path=str(Path(out_dir) / f"{split_name}_residuals.png"))
    except Exception as e:
        typer.echo(f"  Warning: không vẽ được chart {split_name}: {e}", err=True)
    return metrics


def _train_single(model_name: str, overrides: dict, experiment_name: str = ""):
    """Train 1 model + evaluate trên val & test set luôn."""
    config = load_config(model_name, overrides)
    set_seed(config.get("seed", 42))
    param_slug = make_param_slug(config)

    # 2 chế độ: dùng dữ liệu đã clean (nhanh) hoặc clean từ raw (đầy đủ pipeline)
    typer.echo("Loading data...")
    use_cleaned = config.get("data", {}).get("use_cleaned", False)
    if use_cleaned:
        typer.echo("Using pre-cleaned data...")
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)
    # lọc theo loại cửa hàng (vd: type "c") → per-store models chỉ train trên subset cùng loại
    df = filter_stores(df, config)

    # thêm lag, rolling, time features... -> biến dữ liệu thô thành feature matrix cho model
    # typer.echo("Adding/ features...")
    df = add_all_features(df)

    # split train/val/test theo thời gian + scale features -> sẵn sàng cho model
    typer.echo("Preprocessing...")
    train_df, val_df, test_df, scaler = preprocess(df, config)

    typer.echo(f"Training {model_name}... ({len(train_df)} train rows, {len(val_df)} val rows)")
    model_class = get_model_class(model_name)
    model = model_class(config)

    start = time.time()
    train_info = model.train(train_df, val_df)
    elapsed = time.time() - start
    typer.echo(f"Training done in {elapsed:.1f}s. Info: {train_info}")

    # tạo thư mục output trước → evaluate ghi chart trực tiếp vào, không cần gọi 2 lần
    results_dir = config.get("results_dir", "results")
    try:
        val_metrics = model.evaluate(val_df) if len(val_df) > 0 else {}
    except Exception as e:
        typer.echo(f"Warning: evaluate failed: {e}", err=True)
        val_metrics = {}
    results = model.get_result_template(val_metrics, param_slug=param_slug, experiment_name=experiment_name)
    results["metadata"] = {
        "n_stores": df["Store"].nunique(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    out_dir = model.save_results(results, results_dir)

    # evaluate cả val lẫn test luôn → không cần chạy evaluate.py riêng sau khi train
    typer.echo("Evaluating...")
    _evaluate_split(model, val_df, "validation", model_name, out_dir)
    test_metrics = _evaluate_split(model, test_df, "test", model_name, out_dir)
    if test_metrics:
        results["test_metrics"] = test_metrics

    # lưu model pickle -> có thể load lại để predict hoặc evaluate trên tập khác
    model_path = Path(out_dir) / "model.pkl"
    model.save(str(model_path))

    typer.echo(f"Results saved to {out_dir}")
    return model, train_df, val_df, test_df, out_dir


def _run_diagnostics(model, train_df, model_name: str, out_dir: str):
    """Chạy chẩn đoán phần dư sau training — Ljung-Box + ACF phần dư.

    Chỉ chạy trên per-store models (ARIMA/SARIMAX) vì chúng fit riêng từng store.
    Kết quả giúp kiểm tra model đã capture hết signal chưa (white noise residuals).
    """
    import json
    if not hasattr(model, "models"):
        typer.echo("Diagnostics chỉ hỗ trợ per-store models (arima/sarimax).")
        return

    diag_dir = str(Path(out_dir) / "diagnostics")
    Path(diag_dir).mkdir(parents=True, exist_ok=True)
    all_diag = []

    stores = sorted(model.models.keys())[:10]
    for sid in stores:
        fitted = model.models[sid]
        try:
            residuals = fitted.resid
            diag = diagnose_residuals(residuals, store_id=sid, save_dir=diag_dir)
            all_diag.append(diag)
            status = "OK" if diag["ljung_box"].get("overall_adequate") else "CÓ PATTERN"
            typer.echo(f"  Store {sid}: Ljung-Box → {status}")
        except Exception as e:
            typer.echo(f"  Store {sid}: diagnostics failed: {e}", err=True)

    # lưu kết quả tổng hợp → user đọc để đánh giá model adequate hay không
    diag_path = Path(diag_dir) / "diagnostics_summary.json"
    with open(diag_path, "w") as f:
        json.dump(all_diag, f, indent=2, default=str)
    typer.echo(f"Diagnostics saved to {diag_dir}")


@app.command()
def train(
    model: str = typer.Option(None, "--model", "-m", help="Model name, 'all', or omit for interactive"),
    set_values: list[str] = typer.Option(None, "--set", "-s", help="Override config key=value"),
    experiment_name: str = typer.Option("", "--experiment-name", "-e", help="Optional experiment name"),
    diagnostics: bool = typer.Option(False, "--diagnostics", help="Chạy chẩn đoán phần dư sau training"),
):
    """Train a forecasting model. Không truyền --model → vào interactive mode."""
    overrides = _parse_overrides(set_values)

    # không truyền --model → interactive mode: hiện menu chọn model + hỏi tham số
    if model is None:
        from scripts import interactive_train_helpers as ith
        model, interactive_overrides = ith.interactive_select()
        overrides = {**interactive_overrides, **overrides}

    if model == "all":
        for name in MODEL_REGISTRY:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Training: {name}")
            typer.echo(f"{'='*60}")
            try:
                result = _train_single(name, overrides, experiment_name)
                if diagnostics and result:
                    trained_model, train_df, _, _, run_out_dir = result
                    _run_diagnostics(trained_model, train_df, name, run_out_dir)
            except Exception as e:
                typer.echo(f"ERROR training {name}: {e}", err=True)
    else:
        result = _train_single(model, overrides, experiment_name)
        if diagnostics and result:
            trained_model, train_df, _, _, run_out_dir = result
            _run_diagnostics(trained_model, train_df, model, run_out_dir)


if __name__ == "__main__":
    app()
