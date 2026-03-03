"""CLI script for training models."""

import ast
import time
from pathlib import Path

import typer

from src.data.features import add_all_features
from src.data.loader import load_cleaned_data, load_raw_data, sample_stores
from src.data.preprocessor import preprocess
from src.models import MODEL_REGISTRY, get_model_class
from src.utils.config import load_config, make_param_slug
from src.utils.seed import set_seed
from src.utils.visualization import plot_predictions, plot_residuals

app = typer.Typer(help="Train time series forecasting models.")


def _parse_overrides(set_values: list[str] | None) -> dict:
    """Parse key=value overrides from CLI. Supports lists, ints, floats, null."""
    # cho phép ghi đè config từ CLI, vd: --set max_stores=50 --set model.order=[2,1,1]
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


def _train_single(model_name: str, overrides: dict, experiment_name: str = ""):
    """Train a single model."""
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
    # giới hạn số store khi dev/debug -> tránh train hàng giờ chỉ để test code
    df = sample_stores(df, config)

    # thêm lag, rolling, time features... -> biến dữ liệu thô thành feature matrix cho model
    typer.echo("Adding features...")
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

    # đánh giá trên validation set -> metric này dùng để so sánh giữa các lần chạy
    metrics = model.evaluate(val_df)
    typer.echo(f"Validation metrics: {metrics}")

    # lưu config + metrics + metadata vào JSON -> có thể load lại để so sánh bằng compare.py
    results = model.get_result_template(metrics, param_slug=param_slug, experiment_name=experiment_name)
    results["metadata"] = {
        "n_stores": df["Store"].nunique(),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }

    results_dir = config.get("results_dir", "results")
    out_dir = model.save_results(results, results_dir)

    # lưu model pickle -> có thể load lại để predict hoặc evaluate trên tập khác
    model_path = Path(out_dir) / "model.pkl"
    model.save(str(model_path))

    # tự động vẽ biểu đồ actual vs predicted + residuals -> xem nhanh kết quả không cần notebook
    try:
        predictions = model.predict(val_df)
        y_true = val_df["Sales"].values
        plot_predictions(y_true, predictions, title=f"{model_name} - Val Predictions", save_path=str(Path(out_dir) / "val_predictions.png"))
        plot_residuals(y_true, predictions, title=f"{model_name} - Val Residuals", save_path=str(Path(out_dir) / "val_residuals.png"))
    except Exception as e:
        typer.echo(f"Warning: could not generate charts: {e}", err=True)

    typer.echo(f"Results saved to {out_dir}")


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name or 'all'"),
    set_values: list[str] = typer.Option(None, "--set", "-s", help="Override config key=value"),
    experiment_name: str = typer.Option("", "--experiment-name", "-e", help="Optional experiment name"),
):
    """Train a forecasting model."""
    overrides = _parse_overrides(set_values)

    if model == "all":
        for name in MODEL_REGISTRY:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Training: {name}")
            typer.echo(f"{'='*60}")
            try:
                _train_single(name, overrides, experiment_name)
            except Exception as e:
                typer.echo(f"ERROR training {name}: {e}", err=True)
    else:
        _train_single(model, overrides, experiment_name)


if __name__ == "__main__":
    app()
