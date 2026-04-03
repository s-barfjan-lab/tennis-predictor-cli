from __future__ import annotations

from pathlib import Path

from tennis_cli.models.dataset import (load_baseline_dataframe, get_feature_columns, TARGET_COLUMN, )
from tennis_cli.models.split import (chronological_train_val_test_split, summarize_all_splits, )
from tennis_cli.models.logistic import (fit_logistic_baseline, predict_split, extract_logistic_coefficients, )
from tennis_cli.models.evaluate import (evaluate_predictions, evaluate_multiple_splits, )
from tennis_cli.models.io import (save_model_artifact, save_metrics_json, save_dataframe_csv, save_metadata_json, )


def train_logistic_for_tour(project_root: Path, tour: str) -> dict:
    """
    Train the Phase 3 logistic baseline for one tour and save artifacts.
    """
    tour = tour.lower().strip()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")

    models_dir = project_root / "data" / "models"
    df = load_baseline_dataframe(project_root, tour)

    train_df, val_df, test_df = chronological_train_val_test_split(df)
    split_summary = summarize_all_splits(train_df, val_df, test_df)

    pipeline, _, _, _ = fit_logistic_baseline(train_df)

    val_preds = predict_split(pipeline, val_df)
    test_preds = predict_split(pipeline, test_df)

    val_metrics = evaluate_predictions(val_preds, "validation")
    test_metrics = evaluate_predictions(test_preds, "test")
    all_metrics_df = evaluate_multiple_splits({"validation": val_preds, "test": test_preds, })

    coef_df = extract_logistic_coefficients(pipeline)

    model_path = models_dir / f"{tour}_logit_baseline.joblib"
    metrics_path = models_dir / f"{tour}_logit_baseline_metrics.json"
    metrics_table_path = models_dir / f"{tour}_logit_baseline_metrics.csv"
    coef_path = models_dir / f"{tour}_logit_baseline_coefficients.csv"
    metadata_path = models_dir / f"{tour}_logit_baseline_meta.json"

    save_model_artifact(pipeline, model_path)
    save_metrics_json({"tour": tour, "model_type": "logistic_regression", "validation": val_metrics,
                       "test": test_metrics, }, metrics_path, )
    save_dataframe_csv(all_metrics_df, metrics_table_path)
    save_dataframe_csv(coef_df, coef_path)

    metadata = {
        "tour": tour,
        "phase": 3,
        "model_type": "logistic_regression",
        "source_backbone": "sackmann",
        "feature_columns": get_feature_columns(),
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "validation_date_min": str(val_df["tourney_date"].min().date()),
        "validation_date_max": str(val_df["tourney_date"].max().date()),
        "test_date_min": str(test_df["tourney_date"].min().date()),
        "test_date_max": str(test_df["tourney_date"].max().date()),
        "split_summary": split_summary.to_dict(orient="records"),
    }
    save_metadata_json(metadata, metadata_path)

    return {
        "tour": tour,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics_table_path": str(metrics_table_path),
        "coef_path": str(coef_path),
        "metadata_path": str(metadata_path),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }