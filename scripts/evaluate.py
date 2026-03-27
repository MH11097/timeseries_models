"""CLI script for evaluating trained models."""

import json
from pathlib import Path

import typer

from src.data.features import add_all_features
from src.data.loader import filter_stores, load_cleaned_data, load_raw_data
from src.data.preprocessor import preprocess
from src.evaluation.cross_validation import walk_forward_cv
from src.models import get_model_class
from src.models.base import BaseModel
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.visualization import plot_predictions, plot_residuals
from scripts.train import _parse_overrides

app = typer.Typer(help="Evaluate trained forecasting models.")


@app.command()
def evaluate(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    set_values: list[str] = typer.Option(
        None, "--set", "-s", help="Override config key=value"
    ),
    run_dir: str = typer.Option(None, "--run-dir", help="Path to saved run directory"),
    cv: str = typer.Option(None, "--cv", help="CV strategy: 'expanding' or 'sliding'"),
    n_splits: int = typer.Option(5, "--n-splits", help="Number of CV folds"),
    eval_days: int = typer.Option(
        30, "--eval-days", help="Limit evaluation to first N days (default: 30)"
    ),
):
    """Evaluate a model on validation/test data or run cross-validation."""
    overrides = _parse_overrides(set_values)
    config = load_config(model, overrides)
    set_seed(config.get("seed", 42))

    if cv:
        # walk-forward CV mô phỏng thực tế: train trên quá khứ, test trên tương lai gần
        typer.echo(f"Running {cv} walk-forward CV with {n_splits} splits...")
        use_cleaned = config.get("data", {}).get("use_cleaned", False)
        if use_cleaned:
            typer.echo("Using pre-cleaned data...")
            df, _ = load_cleaned_data(config)
        else:
            df, _ = load_raw_data(config)
        df = filter_stores(df, config)
        df = add_all_features(df)

        model_class = get_model_class(model)
        cv_results = walk_forward_cv(
            model_class,
            config,
            df,
            n_splits=n_splits,
            expanding=(cv == "expanding"),
            eval_days=eval_days,
        )
        typer.echo(f"CV Results ({n_splits} folds):")
        for fold in cv_results["folds"]:
            typer.echo(f"  Fold {fold['fold']}: RMSPE={fold['rmspe']:.4f}")
        typer.echo(f"Aggregated: {cv_results['aggregated']}")

        # Save CV results
        results_dir = Path(config.get("results_dir", "results")) / model / "cv"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)
        typer.echo(f"CV results saved to {results_dir}")
        return

    # load model đã train từ trước -> không cần train lại, chỉ predict và đánh giá
    if run_dir:
        model_path = Path(run_dir) / "model.pkl"
    else:
        # không chỉ định run cụ thể -> lấy run mới nhất (sort theo tên thư mục = timestamp)
        results_base = Path(config.get("results_dir", "results")) / model
        if not results_base.exists():
            typer.echo(f"No results found for {model}", err=True)
            raise typer.Exit(1)
        runs = sorted(results_base.iterdir())
        run_dir = str(runs[-1])
        model_path = Path(run_dir) / "model.pkl"

    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading model from {model_path}")
    loaded_model = BaseModel.load(str(model_path))

    # Load and preprocess data
    use_cleaned = config.get("data", {}).get("use_cleaned", False)
    if use_cleaned:
        typer.echo("Using pre-cleaned data...")
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)
    df = filter_stores(df, config)
    df = add_all_features(df)
    _, val_df, test_df, _ = preprocess(df, config)

    # đánh giá trên cả val và test -> val để tune, test để báo cáo kết quả cuối cùng
    for split_name, split_df in [("validation", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            continue

        # giới hạn theo số ngày unique, không phải số dòng
       
        # head(N) chỉ lấy N dòng đầu = 1 store duy nhất do sort (Store, Date)
        if eval_days is not None:
            first_n_dates = sorted(split_df["Date"].unique())[:eval_days]
            # eval_df = split_df.head(eval_days)
            eval_df = split_df[split_df["Date"].isin(first_n_dates)]
            typer.echo(f"Evaluating on first {eval_days} days ({len(eval_df)} samples)")
        else:
            eval_df = split_df
        metrics = loaded_model.evaluate(eval_df)
        typer.echo(f"{split_name.capitalize()} metrics: {metrics}")

        # Save plots
        preds = loaded_model.predict(eval_df)
        plot_predictions(
            eval_df["Sales"].values,
            preds,
            title=f"{model} - {split_name}",
            save_path=f"{run_dir}/{split_name}_predictions.png",
        )
        plot_residuals(
            eval_df["Sales"].values,
            preds,
            title=f"{model} - {split_name}",
            save_path=f"{run_dir}/{split_name}_residuals.png",
        )

    typer.echo(f"Plots saved to {run_dir}")


if __name__ == "__main__":
    app()
