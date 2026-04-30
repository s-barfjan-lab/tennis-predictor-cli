from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


TARGET_COLUMN = "label_player_a_win"
VALID_SURFACES = {"HARD", "CLAY", "GRASS"}

GLOBAL_FEATURE_COLUMNS = [
    "delta_rank_adv",
    "delta_surface_elo",
    "delta_serve_win_pct_last10",
    "delta_return_win_pct_last10",
    "delta_bp_conversion_last10",
    "delta_serve_win_pct_last10_surface",
    "delta_return_win_pct_last10_surface",
    "delta_bp_conversion_last10_surface",
    "delta_h2h",
    "delta_days_since_last_match",
    "delta_matches_last_30_days",
    "delta_days_since_last_match_surface",
    "delta_matches_last_30_days_surface",
    "delta_age",
    "delta_height",
    "is_clay",
    "is_grass",
    "round_ordinal",
    "best_of",
]

SURFACE_MODEL_FEATURE_COLUMNS = [
    "delta_rank_adv",
    "delta_surface_elo",
    "delta_serve_win_pct_last10",
    "delta_return_win_pct_last10",
    "delta_bp_conversion_last10",
    "delta_serve_win_pct_last10_surface",
    "delta_return_win_pct_last10_surface",
    "delta_bp_conversion_last10_surface",
    "delta_h2h",
    "delta_days_since_last_match",
    "delta_matches_last_30_days",
    "delta_days_since_last_match_surface",
    "delta_matches_last_30_days_surface",
    "delta_age",
    "delta_height",
    "round_ordinal",
    "best_of",
]

METADATA_COLUMNS = [
    "match_id",
    "tour",
    "tourney_date",
    "surface",
    "round",
    "best_of",
    "player_id_a",
    "player_name_a",
    "player_id_b",
    "player_name_b",
    "handedness_combo",
]


def get_baseline_feature_path(project_root: Path, tour: str, source: str = "sackmann") -> Path:
    """
    Return the expected baseline parquet path for a given tour/source.
    """
    tour = tour.lower().strip()
    source = source.lower().strip()

    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")

    if source not in {"sackmann", "tml"}:
        raise ValueError("source must be 'sackmann' or 'tml'")

    if source == "tml":
        if tour != "atp":
            raise ValueError("TML source is currently supported only for ATP.")
        return project_root / "data" / "features" / "atp_baseline_tml_2015_2025.parquet"

    return project_root / "data" / "features" / f"{tour}_baseline_2015_2025.parquet"


def _normalize_surface_value(surface: str | None) -> str | None:
    """
    Normalize a user/model-facing surface value to the dataset convention.
    Expected outputs: Hard, Clay, Grass, or None.
    """
    if surface is None:
        return None

    value = str(surface).strip().title()
    if not value:
        return None

    if value == "Carpet":
        return "Hard"

    if value not in {"Hard", "Clay", "Grass"}:
        raise ValueError("surface must be one of: Hard, Clay, Grass")

    return value


def _drop_invalid_surfaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the core modeling surfaces: HARD, CLAY, GRASS.

    Rows with missing, NONE, CARPET, or any other unexpected surface value
    are discarded from the modeling dataset.
    """
    if "surface" not in df.columns:
        raise ValueError("Baseline dataset does not contain a 'surface' column.")

    out = df.copy()
    surface_norm = out["surface"].astype(str).str.upper().str.strip()
    out = out[surface_norm.isin(VALID_SURFACES)].copy()

    if out.empty:
        raise ValueError("All rows were removed after invalid-surface filtering.")

    return out


def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure the dataframe contains the minimum schema required
    for baseline training.
    """
    required = set(GLOBAL_FEATURE_COLUMNS + [TARGET_COLUMN, "tourney_date", "match_id"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Baseline dataset is missing required columns: " + ", ".join(missing))


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure target/date/numeric feature types are usable for modeling.
    """
    out = df.copy()

    out["tourney_date"] = pd.to_datetime(out["tourney_date"], errors="coerce")
    if out["tourney_date"].isna().any():
        bad_count = int(out["tourney_date"].isna().sum())
        raise ValueError(f"'tourney_date' contains {bad_count} invalid values after datetime conversion.")

    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    if out[TARGET_COLUMN].isna().any():
        bad_count = int(out[TARGET_COLUMN].isna().sum())
        raise ValueError(f"'{TARGET_COLUMN}' contains {bad_count} invalid values after numeric conversion.")

    for col in GLOBAL_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def load_baseline_dataframe(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None, ) -> pd.DataFrame:
    """
    Load the baseline parquet for one tour, validate the schema,
    coerce types, and sort chronologically.
    """

    path = get_baseline_feature_path(project_root, tour, source=source)
    surface = _normalize_surface_value(surface)

    if not path.exists():
        raise FileNotFoundError(f"Baseline feature file not found: {path}")

    df = pd.read_parquet(path)
    _validate_required_columns(df)
    df = _coerce_types(df)
    df = _drop_invalid_surfaces(df)

    if surface is not None:
        if "surface" not in df.columns:
            raise ValueError("Baseline dataset does not contain a 'surface' column required for filtering.")

        df = df[df["surface"].astype(str).str.title() == surface].copy()

        if df.empty:
            raise ValueError(f"No rows found for surface='{surface}' in the selected baseline dataset.")

    df = df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)
    return df


def get_feature_columns(surface_specific: bool = False) -> list[str]:
    """
    Return the ordered list of numeric model features.

    Parameters
    ----------
    surface_specific:
        If True, return the feature set used for a model trained on a single
        surface only. In that case, surface dummy columns are excluded because
        the training data has already been filtered to one surface.
    """
    if surface_specific:
        return SURFACE_MODEL_FEATURE_COLUMNS.copy()

    return GLOBAL_FEATURE_COLUMNS.copy()


def get_metadata_columns(df: pd.DataFrame) -> list[str]:
    """
    Return metadata columns that actually exist in the given dataframe.
    """
    return [col for col in METADATA_COLUMNS if col in df.columns]



def build_training_matrices(df: pd.DataFrame, surface_specific: bool = False, ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Split a validated baseline dataframe into:
    - X: numeric feature matrix
    - y: target vector
    - meta: identifying/context columns

    Missing numeric values are NOT dropped here.
    They are handled by the sklearn preprocessing pipeline.
    """
    feature_cols = get_feature_columns(surface_specific=surface_specific)
    meta_cols = get_metadata_columns(df)

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    meta = df[meta_cols].copy()

    return X, y, meta



def load_training_dataset(project_root: Path, tour: str, source: str = "sackmann",
    surface: str | None = None, surface_specific: bool = False, ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Convenience wrapper that returns:
    - full validated dataframe
    - X
    - y
    - meta
    """
    
    df = load_baseline_dataframe(project_root=project_root, tour=tour, source=source, surface=surface, )
    X, y, meta = build_training_matrices(df, surface_specific=surface_specific)
    return df, X, y, meta