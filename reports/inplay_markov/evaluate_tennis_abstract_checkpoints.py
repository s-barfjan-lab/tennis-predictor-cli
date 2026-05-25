from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_cli.features.inplay_markov import (
    DEFAULT_PRIOR_STRENGTH,
    inplay_match_win_probability,
    updated_serve_point_probabilities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "features" / "tennis_abstract_pbp" / "inplay_markov_snapshots.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "inplay_markov"
METRICS_CSV = OUT_DIR / "tennis_abstract_checkpoint_metrics.csv"
METRICS_JSON = OUT_DIR / "tennis_abstract_checkpoint_metrics.json"
PREDICTIONS_SAMPLE_CSV = OUT_DIR / "tennis_abstract_checkpoint_predictions_sample.csv"
SNAPSHOT_SAMPLE_CSV = OUT_DIR / "tennis_abstract_snapshot_sample.csv"
SCHEMA_JSON = OUT_DIR / "tennis_abstract_snapshot_schema.json"


def _split_name(match_date: pd.Series) -> pd.Series:
    dates = pd.to_datetime(match_date, errors="coerce")
    split = pd.Series("train", index=dates.index)
    split[(dates > pd.Timestamp("2022-12-31")) & (dates <= pd.Timestamp("2023-12-31"))] = "validation"
    split[dates > pd.Timestamp("2023-12-31")] = "test"
    return split


def _checkpoint_type(df: pd.DataFrame) -> pd.Series:
    points_00 = df["points_1"].astype(str).eq("0") & df["points_2"].astype(str).eq("0")
    game_total = pd.to_numeric(df["games_1"], errors="coerce").fillna(0).astype(int) + pd.to_numeric(
        df["games_2"], errors="coerce"
    ).fillna(0).astype(int)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[df["point_index"].eq(1)] = "match_start"
    out[points_00 & df["games_1"].eq(0) & df["games_2"].eq(0) & out.isna()] = "set_start"
    out[points_00 & game_total.ge(10) & out.isna()] = "late_game_start_total_ge10"
    return out


def _load_checkpoint_frame() -> pd.DataFrame:
    if not SNAPSHOTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing Tennis Abstract snapshot parquet: {SNAPSHOTS_PATH}. "
            "Run `python -m tennis_cli build-tennis-abstract-pbp` first."
        )

    df = pd.read_parquet(SNAPSHOTS_PATH)
    df["split"] = _split_name(df["match_date"])
    df["checkpoint_type"] = _checkpoint_type(df)
    df = df[df["checkpoint_type"].notna()].copy()
    df = df[df["label_player1_win_match"].notna()].copy()
    df = df[df["split"].isin(["validation", "test"])].copy()
    df = df.sort_values(["match_date", "match_id", "point_index"], kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError("No labeled validation/test checkpoint rows found.")
    return df


@lru_cache(maxsize=250_000)
def _predict_cached(
    sets_1: int,
    sets_2: int,
    games_1: int,
    games_2: int,
    points_1: str,
    points_2: str,
    server: str,
    p1_won_before: int,
    p1_played_before: int,
    p2_won_before: int,
    p2_played_before: int,
    best_of: int,
    prior_strength: float,
) -> float:
    p1_live, p2_live = updated_serve_point_probabilities(
        p_a_serve_point_prior=0.5,
        p_b_serve_point_prior=0.5,
        a_service_points_won=p1_won_before,
        a_service_points_played=p1_played_before,
        b_service_points_won=p2_won_before,
        b_service_points_played=p2_played_before,
        prior_strength=prior_strength,
    )
    return inplay_match_win_probability(
        sets_a=sets_1,
        sets_b=sets_2,
        games_a=games_1,
        games_b=games_2,
        points_a=points_1,
        points_b=points_2,
        server=server,
        p_a_serve_point=p1_live,
        p_b_serve_point=p2_live,
        best_of=best_of,
    )


def _predict_frame(df: pd.DataFrame, prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> pd.Series:
    preds: list[float] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        preds.append(
            _predict_cached(
                sets_1=int(row.sets_1),
                sets_2=int(row.sets_2),
                games_1=int(row.games_1),
                games_2=int(row.games_2),
                points_1=str(row.points_1),
                points_2=str(row.points_2),
                server=str(row.server),
                p1_won_before=int(row.p1_service_points_won_before),
                p1_played_before=int(row.p1_service_points_played_before),
                p2_won_before=int(row.p2_service_points_won_before),
                p2_played_before=int(row.p2_service_points_played_before),
                best_of=int(row.best_of),
                prior_strength=float(prior_strength),
            )
        )
        if idx % 10_000 == 0:
            print(f"Predicted {idx:,} / {len(df):,} checkpoint rows")
    return pd.Series(preds, index=df.index, name="prob_player1_win_match")


def _metric_row(df: pd.DataFrame, group_name: str, group_value: str) -> dict:
    y_true = df["label_player1_win_match"].astype(int)
    pred_prob = df["prob_player1_win_match"].astype(float).clip(1e-6, 1.0 - 1e-6)
    pred_label = (pred_prob >= 0.5).astype(int)
    roc_auc = None
    if y_true.nunique() == 2:
        roc_auc = float(roc_auc_score(y_true, pred_prob))
    return {
        "evaluation": "tennis_abstract_fixed_checkpoints_neutral_prior_live_counts",
        "group": group_name,
        "value": group_value,
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "date_min": str(pd.to_datetime(df["match_date"]).min().date()),
        "date_max": str(pd.to_datetime(df["match_date"]).max().date()),
        "log_loss": float(log_loss(y_true, pred_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "roc_auc": roc_auc,
        "brier_score": float(brier_score_loss(y_true, pred_prob)),
        "positive_rate": float(y_true.mean()),
        "avg_pred_prob": float(pred_prob.mean()),
    }


def _build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for split, split_df in df.groupby("split", sort=True):
        rows.append(_metric_row(split_df, "split", split))
        for checkpoint_type, group_df in split_df.groupby("checkpoint_type", sort=True):
            rows.append(_metric_row(group_df, f"{split}:checkpoint_type", str(checkpoint_type)))
        for (tour, surface), group_df in split_df.groupby(["tour", "surface"], sort=True):
            rows.append(_metric_row(group_df, f"{split}:tour_surface", f"{tour}/{surface}"))
    return pd.DataFrame(rows)


def _write_inspection_files(raw_df: pd.DataFrame, scored_df: pd.DataFrame) -> None:
    sample_cols = [
        "snapshot_id",
        "match_id",
        "tour",
        "match_date",
        "tournament",
        "round",
        "surface",
        "best_of",
        "player1_name",
        "player2_name",
        "point_index",
        "sets_1",
        "sets_2",
        "games_1",
        "games_2",
        "points_1",
        "points_2",
        "server",
        "p1_service_points_won_before",
        "p1_service_points_played_before",
        "p2_service_points_won_before",
        "p2_service_points_played_before",
        "label_player1_win_match",
    ]
    raw_df[sample_cols].head(250).to_csv(SNAPSHOT_SAMPLE_CSV, index=False)

    prediction_cols = sample_cols + ["split", "checkpoint_type", "prob_player1_win_match"]
    scored_df[prediction_cols].head(500).to_csv(PREDICTIONS_SAMPLE_CSV, index=False)

    schema = {
        "parquet_path": str(SNAPSHOTS_PATH),
        "rows": int(len(raw_df)),
        "columns": [{"name": col, "dtype": str(dtype)} for col, dtype in raw_df.dtypes.items()],
        "note": "Use the CSV samples for quick inspection if your editor cannot open the full parquet.",
    }
    SCHEMA_JSON.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_parquet(SNAPSHOTS_PATH)
    checkpoint_df = _load_checkpoint_frame()
    print(f"Loaded {len(checkpoint_df):,} labeled validation/test checkpoint rows")
    checkpoint_df["prob_player1_win_match"] = _predict_frame(checkpoint_df)
    metrics_df = _build_metrics(checkpoint_df)

    metrics_df.to_csv(METRICS_CSV, index=False)
    METRICS_JSON.write_text(json.dumps(metrics_df.to_dict(orient="records"), indent=2), encoding="utf-8")
    _write_inspection_files(raw_df, checkpoint_df)

    print(metrics_df.to_string(index=False))
    print(f"\nWrote {METRICS_CSV}")
    print(f"Wrote {METRICS_JSON}")
    print(f"Wrote {SNAPSHOT_SAMPLE_CSV}")
    print(f"Wrote {PREDICTIONS_SAMPLE_CSV}")
    print(f"Wrote {SCHEMA_JSON}")


if __name__ == "__main__":
    main()
