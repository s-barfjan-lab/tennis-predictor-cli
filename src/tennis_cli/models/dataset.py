from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# -------------------------
# Phase 3 baseline schema
# -------------------------

TARGET_COLUMN = "label_player_a_win"

# I kept this list intentionally simple and numeric for the first logistic baseline.
# We can extend it later, but this is the clean Phase 3 core.
FEATURE_COLUMNS = [
    "delta_rank_adv",
    "delta_rank_points",
    "elo_a",
    "elo_b",
    "delta_elo",
    "delta_age",
    "delta_height",
    "delta_matches_played",
    "delta_win_rate_last10",
    "delta_aces_avg_last10",
    "delta_serve_win_pct_last10",
    "delta_days_since_last_match",
    "delta_matches_last_30_days",
]

# Useful for tracing rows and later prediction/debugging,
# but NOT to be used as model inputs.
METADATA_COLUMNS = [
    "match_id",
    "tour",   
    "tourney_date",
    "surface", # it is categorical and would require encoding to be considered in features. We want to keep this baseline simple.
    "round",   # The same reason as surface. For first logistic baseline, we want to focus on the core player-strength features first.
    "best_of",
    "player_id_a",
    "player_name_a",
    "player_id_b",
    "player_name_b",
    "handedness_combo",
]


def get_baseline_feature_path(project_root: Path, tour: str) -> Path:
    """
    Return the expected baseline parquet path for a given tour.
    """
    tour = tour.lower().strip()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")

    return project_root / "data" / "features" / f"{tour}_baseline_2015_2025.parquet"


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure the dataframe contains the minimum schema required
    for Phase 3 baseline training.
    """
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "tourney_date"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Baseline dataset is missing required columns: "
            + ", ".join(missing)
        )


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure target/date/numeric feature types are usable for modeling.
    """
    out = df.copy()

    out["tourney_date"] = pd.to_datetime(out["tourney_date"], errors="coerce")
    if out["tourney_date"].isna().any():
        bad_count = int(out["tourney_date"].isna().sum())
        raise ValueError(
            f"'tourney_date' contains {bad_count} invalid values after datetime conversion."
        )

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        bad_count = int(out[TARGET_COLUMN].isna().sum())
        raise ValueError(
            f"'{TARGET_COLUMN}' contains {bad_count} invalid values after numeric conversion."
        )

    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def load_baseline_dataframe(project_root: Path, tour: str) -> pd.DataFrame:
    """
    Load the baseline parquet for one tour, validate the schema,
    coerce types, and sort chronologically.
    """
    path = get_baseline_feature_path(project_root, tour)

    if not path.exists():
        raise FileNotFoundError(f"Baseline feature file not found: {path}")

    df = pd.read_parquet(path)
    _validate_required_columns(df)
    df = _coerce_types(df)

    # Chronological order is critical for later time-based splits.
    df = df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)

    return df


def get_feature_columns() -> list[str]:
    """
    Return the ordered list of numeric model features for Phase 3.
    """
    return FEATURE_COLUMNS.copy()


def get_metadata_columns(df: pd.DataFrame) -> list[str]:
    """
    Return metadata columns that actually exist in the given dataframe.
    This makes the loader robust to small schema changes.
    """
    return [col for col in METADATA_COLUMNS if col in df.columns]


def build_training_matrices(df: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Split a validated baseline dataframe into:
    - X: numeric feature matrix
    - y: target vector
    - meta: identifying/context columns

    Missing numeric values are NOT dropped here.
    They will be handled later by the sklearn preprocessing pipeline.
    """
    feature_cols = get_feature_columns()
    meta_cols = get_metadata_columns(df)

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    meta = df[meta_cols].copy()

    return X, y, meta


def load_training_dataset(project_root: Path, tour: str,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Convenience wrapper that returns:
    - full validated dataframe
    - X
    - y
    - meta
    """
    df = load_baseline_dataframe(project_root=project_root, tour=tour)
    X, y, meta = build_training_matrices(df)
    return df, X, y, meta