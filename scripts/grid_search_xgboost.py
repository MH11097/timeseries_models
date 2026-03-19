"""Grid search for XGBoost models with specific hyperparameter ranges."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.data.features import add_all_features
from src.data.loader import load_cleaned_data, load_raw_data, sample_stores
from src.data.preprocessor import preprocess
from src.evaluation.metrics import rmspe
from src.utils.config import load_config
from src.utils.seed import set_seed

app = typer.Typer(help="GridSearchCV for XGBoost models with custom parameter ranges.")

# Features for XGBoost model
FEATURE_COLS = [
    "Store",
    "DayOfWeek",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Year",
    "Month",
    "WeekOfYear",
    "DayOfMonth",
    "IsWeekend",
    "Sales_lag_1",
    "Sales_lag_7",
    "Sales_lag_14",
    "Sales_lag_30",
    "Sales_rolling_mean_7",
    "Sales_rolling_mean_14",
    "Sales_rolling_mean_30",
    "Sales_rolling_std_7",
    "Sales_rolling_std_14",
    "Sales_rolling_std_30",
    "CompetitionOpenMonths",
    "Promo2Active",
]


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract available features from dataframe."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].fillna(0)


@app.command()
def grid_search(
    model_name: str = "xgboost",
    use_cleaned: bool = True,
    cv_splits: int = 5,
    n_jobs: int = -1,
    max_stores: int | None = None,
):
    """
    Run GridSearchCV for XGBoost models with custom parameters.

    Parameters:
        - max_depth: [9, 11]
        - learning_rate: [0.01, 0.05]
        - n_estimators: [300, 500]
        - subsample: [1.0] (fixed)
        - colsample_bytree: [1.0] (fixed)

    Args:
        model_name: Model name (default: xgboost)
        use_cleaned: Use pre-cleaned data (default: True)
        cv_splits: Number of cross-validation splits (default: 5)
        n_jobs: Number of parallel jobs (-1 = all cores, default: -1)
        max_stores: Limit number of stores for faster testing (default: None)
    """
    typer.echo("=== GridSearchCV for XGBoost Models ===")

    # Load configuration and data
    config = load_config(model_name)
    seed = config.get("seed", 42)
    set_seed(seed)

    typer.echo("Loading data...")
    if use_cleaned:
        df, _ = load_cleaned_data(config)
    else:
        df, _ = load_raw_data(config)

    if max_stores:
        df = sample_stores(df, {"data": {"max_stores": max_stores}})

    typer.echo("Adding features...")
    df = add_all_features(df)

    typer.echo("Preprocessing...")
    train_df, val_df, test_df, scaler = preprocess(df, config)

    # Combine train and val for GridSearchCV
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    X = get_features(combined_df)
    y = combined_df["Sales"].values

    typer.echo(f"Data shape: X={X.shape}, y={y.shape}")

    # Define parameter grid
    param_grid = {
        "max_depth": [9, 11],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [300, 500],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "reg_alpha": [0, 0.5],
        "reg_lambda": [1.0, 1.5],
    }

    # Calculate total parameter combinations
    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    typer.echo(f"\nParameter Grid:")
    for param, values in param_grid.items():
        typer.echo(f"  {param}: {values}")

    # Create base model
    base_model = xgb.XGBRegressor(random_state=seed, n_jobs=1)

    # Custom scoring function using RMSPE
    # sklearn scorer signature: scorer(estimator, X_test, y_test)
    def rmspe_scorer(estimator, X_test, y_test):
        y_pred = estimator.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative predictions
        return -rmspe(
            y_test, y_pred
        )  # Negative because sklearn expects higher is better

    # Create time series cross-validator (replaces random splits)
    ts_cv = TimeSeriesSplit(n_splits=cv_splits)

    typer.echo("\n" + "=" * 80)
    typer.echo("TIME SERIES GRIDSEARCHCV (Walk-Forward Validation)")
    typer.echo("=" * 80)

    typer.echo(f"Total parameter combinations: {n_combos}")
    typer.echo(f"CV splits (TimeSeriesSplit): {cv_splits}")
    typer.echo(f"Total model evaluations: {n_combos * cv_splits}\n")

    # Run GridSearchCV with time series split
    typer.echo(f"Running GridSearchCV with TimeSeriesSplit...")
    start_time = time.time()

    grid_search_cv = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=ts_cv,  # Time series split instead of random K-fold
        scoring=rmspe_scorer,
        n_jobs=n_jobs,
        verbose=2,  # Verbose output to see progress
    )

    grid_search_cv.fit(X, y)
    elapsed = time.time() - start_time

    # Results
    typer.echo(f"\n" + "=" * 80)
    typer.echo(f"GridSearchCV completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    typer.echo(f"Best score (RMSPE): {-grid_search_cv.best_score_:.4f}")
    typer.echo(f"Best parameters:\n{json.dumps(grid_search_cv.best_params_, indent=2)}")
    typer.echo("=" * 80)

    # Save detailed results
    results_dir = (
        Path(config.get("results_dir", "results")) / "xgboost" / "grid_search_custom"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    best_params_path = results_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(grid_search_cv.best_params_, f, indent=2)
    typer.echo(f"Best parameters saved to {best_params_path}")

    # Save CV results with run indexing
    cv_results_df = pd.DataFrame(grid_search_cv.cv_results_)

    # Add run index for each parameter combination
    cv_results_df.insert(0, "run_index", range(1, len(cv_results_df) + 1))

    cv_results_path = results_dir / "cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    typer.echo(f"CV results saved to {cv_results_path}")

    # Log each run with details
    all_runs_log = []
    for idx, row in cv_results_df.iterrows():
        run_entry = {
            "run_index": int(row["run_index"]),
            "params": {
                k.replace("param_", ""): row[k]
                for k in row.index
                if k.startswith("param_")
            },
            "mean_test_score": float(-row["mean_test_score"]),  # Convert back to RMSPE
            "std_test_score": float(row["std_test_score"]),
            "mean_train_score": float(-row["mean_train_score"]),
            "rank_test_score": int(row["rank_test_score"]),
        }
        all_runs_log.append(run_entry)

    # Save all runs as JSON
    all_runs_path = results_dir / "all_runs.json"
    with open(all_runs_path, "w") as f:
        json.dump(all_runs_log, f, indent=2)
    typer.echo(f"All runs log saved to {all_runs_path}")

    # Display all runs with index in console
    typer.echo(f"\nAll {len(all_runs_log)} runs (indexed):")
    typer.echo("-" * 100)
    for run in all_runs_log:
        typer.echo(
            f"[Run {run['run_index']}] RMSPE: {run['mean_test_score']:.4f} | Params: {run['params']}"
        )
    typer.echo("-" * 100)

    # Top 10 results
    top_10_idx = np.argsort(grid_search_cv.cv_results_["rank_test_score"])[:10]
    top_10 = cv_results_df.iloc[top_10_idx][
        [
            "run_index",
            "param_max_depth",
            "param_learning_rate",
            "param_n_estimators",
            "param_subsample",
            "param_colsample_bytree",
            "mean_test_score",
        ]
    ].copy()
    top_10["mean_test_score"] = -top_10["mean_test_score"]  # Convert back to RMSPE

    typer.echo("\nTop 10 parameter combinations (by RMSPE):")
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        typer.echo(
            f"[Run {int(row['run_index'])}] Rank {idx}: RMSPE={row['mean_test_score']:.4f}"
        )

    top_10_path = results_dir / "top_10_params.csv"
    top_10.to_csv(top_10_path, index=False)
    typer.echo(f"Top 10 saved to {top_10_path}")

    # Summary
    summary = {
        "total_combinations": len(grid_search_cv.cv_results_["params"]),
        "best_rmspe": float(-grid_search_cv.best_score_),
        "best_params": grid_search_cv.best_params_,
        "cv_splits": cv_splits,
        "training_time_seconds": elapsed,
        "data_shape": {"n_samples": X.shape[0], "n_features": X.shape[1]},
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    app()
