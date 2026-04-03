from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss, brier_score_loss, )


REQUIRED_PREDICTION_COLUMNS = ["y_true", "prob_player_a_win", "pred_player_a_win", ]


def _validate_prediction_df(pred_df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in pred_df.columns]
    if missing:
        raise ValueError("Prediction dataframe is missing required columns: " + ", ".join(missing))


def evaluate_predictions(pred_df: pd.DataFrame, split_name: str) -> Dict[str, float]:
    """
    Compute core evaluation metrics for one prediction dataframe.
    """
    _validate_prediction_df(pred_df)

    y_true = pred_df["y_true"]
    y_pred = pred_df["pred_player_a_win"]
    y_prob = pred_df["prob_player_a_win"]

    metrics = {
        "split": split_name,
        "rows": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(y_true.mean()),
        "avg_pred_prob": float(y_prob.mean()),
    }

    return metrics


def evaluate_multiple_splits(predictions_by_split: dict[str, pd.DataFrame], ) -> pd.DataFrame:
    """
    Evaluate multiple prediction dataframes and return one summary table.
    """
    rows = []
    for split_name, pred_df in predictions_by_split.items():
        rows.append(evaluate_predictions(pred_df, split_name))

    return pd.DataFrame(rows)