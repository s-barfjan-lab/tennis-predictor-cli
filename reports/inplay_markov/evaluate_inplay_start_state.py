from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_cli.features.markov import markov_match_win_probability
from tennis_cli.models.dataset import TARGET_COLUMN, load_baseline_dataframe
from tennis_cli.models.split import chronological_train_val_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "inplay_markov"
OUT_CSV = OUT_DIR / "inplay_start_state_metrics.csv"
OUT_JSON = OUT_DIR / "inplay_start_state_metrics.json"


def _numeric_or_neutral(series: pd.Series, history_flag: pd.Series | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if history_flag is not None:
        flags = pd.to_numeric(history_flag, errors="coerce").fillna(0).astype(int)
        values = values.where(flags == 1, 0.5)
    return values.fillna(0.5).clip(0.01, 0.99)


def _serve_point_priors(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    required = {
        "service_points_won_pct_30_a",
        "service_points_won_pct_30_b",
        "return_points_won_pct_30_a",
        "return_points_won_pct_30_b",
        "has_serve_history_a",
        "has_serve_history_b",
        "has_return_history_a",
        "has_return_history_b",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Cannot evaluate in-play start state; missing columns: " + ", ".join(missing))

    serve_a = _numeric_or_neutral(df["service_points_won_pct_30_a"], df["has_serve_history_a"])
    serve_b = _numeric_or_neutral(df["service_points_won_pct_30_b"], df["has_serve_history_b"])
    return_a = _numeric_or_neutral(df["return_points_won_pct_30_a"], df["has_return_history_a"])
    return_b = _numeric_or_neutral(df["return_points_won_pct_30_b"], df["has_return_history_b"])

    p_a_serve = ((serve_a + (1.0 - return_b)) / 2.0).clip(0.01, 0.99)
    p_b_serve = ((serve_b + (1.0 - return_a)) / 2.0).clip(0.01, 0.99)
    return p_a_serve, p_b_serve


@lru_cache(maxsize=200_000)
def _start_state_prob(p_a_serve: float, p_b_serve: float, best_of: int) -> float:
    p_a = round(float(p_a_serve), 4)
    p_b = round(float(p_b_serve), 4)
    best = 5 if int(best_of) == 5 else 3
    return markov_match_win_probability(p_a, p_b, best_of=best)


def _predict_start_state(df: pd.DataFrame) -> pd.Series:
    if "markov_match_win_prob_a" in df.columns:
        return pd.to_numeric(df["markov_match_win_prob_a"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    p_a_serve, p_b_serve = _serve_point_priors(df)
    best_of = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)

    probs = [
        _start_state_prob(p_a, p_b, best)
        for p_a, p_b, best in zip(p_a_serve, p_b_serve, best_of)
    ]
    return pd.Series(probs, index=df.index, name="prob_player_a_win")


def _metrics(y_true: pd.Series, pred_prob: pd.Series) -> dict:
    y = y_true.astype(int)
    p = pred_prob.astype(float).clip(1e-6, 1.0 - 1e-6)
    pred_label = (p >= 0.5).astype(int)
    return {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred_label)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "positive_rate": float(y.mean()),
        "avg_pred_prob": float(p.mean()),
    }


def evaluate_branch(tour: str, source: str, surface: str) -> list[dict]:
    df = load_baseline_dataframe(
        project_root=PROJECT_ROOT,
        tour=tour,
        source=source,
        surface=surface,
    )
    _, val_df, test_df = chronological_train_val_test_split(df)

    rows = []
    for split_name, split_df in [("validation", val_df), ("test", test_df)]:
        pred = _predict_start_state(split_df)
        metric_row = _metrics(split_df[TARGET_COLUMN], pred)
        metric_row.update(
            {
                "evaluation": "start_state_no_live_counts",
                "tour": tour,
                "source": source,
                "surface": surface,
                "split": split_name,
                "date_min": str(pd.to_datetime(split_df["tourney_date"]).min().date()),
                "date_max": str(pd.to_datetime(split_df["tourney_date"]).max().date()),
            }
        )
        rows.append(metric_row)
    return rows


def main() -> None:
    branches = [
        ("atp", "sackmann", "Hard"),
        ("atp", "sackmann", "Clay"),
        ("atp", "sackmann", "Grass"),
        ("atp", "tml", "Hard"),
        ("atp", "tml", "Clay"),
        ("atp", "tml", "Grass"),
        ("wta", "sackmann", "Hard"),
        ("wta", "sackmann", "Clay"),
        ("wta", "sackmann", "Grass"),
    ]
    rows: list[dict] = []
    for tour, source, surface in branches:
        rows.extend(evaluate_branch(tour, source, surface))

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[
        [
            "evaluation",
            "tour",
            "source",
            "surface",
            "split",
            "rows",
            "date_min",
            "date_max",
            "log_loss",
            "accuracy",
            "roc_auc",
            "brier_score",
            "positive_rate",
            "avg_pred_prob",
        ]
    ]
    metrics_df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
