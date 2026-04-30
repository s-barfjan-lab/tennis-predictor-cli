from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from tennis_cli.models.dataset import load_baseline_dataframe
from tennis_cli.models.evaluate import evaluate_multiple_splits
from tennis_cli.models.io import (save_dataframe_csv, save_metadata_json, save_metrics_json, save_model_artifact, )
from tennis_cli.models.logistic import (LogisticConfig, extract_logistic_coefficients, fit_logistic_baseline, predict_split, )
from tennis_cli.models.split import (chronological_train_val_test_split, summarize_all_splits, )

console = Console()


def get_project_root() -> Path:
    # src/tennis_cli/models/train_logit.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def train_logistic_for_tour(tour: str) -> None:
    project_root = get_project_root()
    tour = tour.lower().strip()

    console.print(f"[bold]Loading baseline dataframe for[/] {tour.upper()}")
    df = load_baseline_dataframe(project_root=project_root, tour=tour)

    console.print(f"[bold]Chronological split for[/] {tour.upper()}")
    train_df, val_df, test_df = chronological_train_val_test_split(df)
    split_summary_df = summarize_all_splits(train_df, val_df, test_df)

    console.print(split_summary_df)

    console.print(f"[bold]Training logistic baseline for[/] {tour.upper()}")
    pipeline, X_train, y_train, meta_train = fit_logistic_baseline(train_df, config=LogisticConfig(), )

    console.print(f"[bold]Predicting train / validation / test for[/] {tour.upper()}")
    pred_train = predict_split(pipeline, train_df)
    pred_val = predict_split(pipeline, val_df)
    pred_test = predict_split(pipeline, test_df)

    metrics_df = evaluate_multiple_splits({"train": pred_train, "validation": pred_val, "test": pred_test, })
    coef_df = extract_logistic_coefficients(pipeline)

    console.print("[bold green]Metrics:[/]")
    console.print(metrics_df)

    console.print("[bold green]Top coefficients:[/]")
    console.print(coef_df.head(10))

    out_dir = project_root / "data" / "models"
    prefix = f"{tour}_logit_phase3a"

    save_model_artifact(pipeline, out_dir / f"{prefix}.joblib")
    save_dataframe_csv(metrics_df, out_dir / f"{prefix}_metrics.csv")
    save_dataframe_csv(split_summary_df, out_dir / f"{prefix}_split_summary.csv")
    save_dataframe_csv(coef_df, out_dir / f"{prefix}_coefficients.csv")
    save_dataframe_csv(pred_val, out_dir / f"{prefix}_validation_predictions.csv")
    save_dataframe_csv(pred_test, out_dir / f"{prefix}_test_predictions.csv")

    save_metrics_json(
        {
            row["split"]: {
                "rows": row["rows"],
                "accuracy": row["accuracy"],
                "roc_auc": row["roc_auc"],
                "log_loss": row["log_loss"],
                "brier_score": row["brier_score"],
                "positive_rate": row["positive_rate"],
                "avg_pred_prob": row["avg_pred_prob"],
            }
            for _, row in metrics_df.iterrows()
        },
        out_dir / f"{prefix}_metrics.json",
    )

    save_metadata_json(
        {
            "tour": tour,
            "model_type": "logistic_regression",
            "phase": "3A",
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "artifacts_prefix": prefix,
        },
        out_dir / f"{prefix}_metadata.json",
    )

    console.print(f"[bold green]Saved model artifacts in:[/] {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3A logistic baseline.")
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    args = parser.parse_args()

    train_logistic_for_tour(args.tour)


if __name__ == "__main__":
    main()