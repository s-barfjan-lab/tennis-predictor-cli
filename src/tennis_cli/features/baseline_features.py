from __future__ import annotations

from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()


ROUND_MAP = {
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}


def _safe_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def build_baseline_match_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert player-centric long view into one row per match with delta features.

    Key design:
    - deterministic player ordering inside each match
    - player A / player B assignment is independent of who won
    - deltas are computed as A minus B
    """
    required_cols = ["match_id", "tour", "tourney_date", "player_id", "player_name", "opponent_id", "opponent_name", "label_win", ]
    missing = [c for c in required_cols if c not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for baseline feature build: {missing}")

    df = long_df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # deterministic ordering inside each match
    df["_player_sort_key"] = (df["player_id"].astype("string").fillna("NA")
        + "__" + df["player_name"].astype("string").fillna("NA"))

    df = df.sort_values(["match_id", "_player_sort_key"]).copy()
    df["side"] = df.groupby("match_id").cumcount()

    bad_matches = df.groupby("match_id")["side"].max()
    bad_matches = bad_matches[bad_matches != 1]
    if not bad_matches.empty:
        raise ValueError("Each match must produce exactly two player rows in long view. "
            f"Problematic match count: {len(bad_matches)}")

    match_meta = df[df["side"] == 0][["match_id", "tour", "tourney_date", "surface", "round", "best_of"]].copy()

    cols_to_carry = [
        "match_id",
        "player_id",
        "player_name",
        "player_hand",
        "player_ht",
        "player_age",
        "player_rank",
        "player_rank_points",
        "player_elo_pre",
        "player_surface_elo_pre",
        "matches_played",
        "win_rate_last10",
        "aces_avg_last10",
        "serve_win_pct_last10",
        "return_win_pct_last10",
        "bp_conversion_last10",
        "days_since_last_match",
        "matches_last_30_days",
        "serve_win_pct_last10_surface",
        "return_win_pct_last10_surface",
        "bp_conversion_last10_surface",
        "days_since_last_match_surface",
        "matches_last_30_days_surface",
        "h2h_win_ratio",
        "label_win",
    ]

    available_cols = [c for c in cols_to_carry if c in df.columns]

    side_a = df[df["side"] == 0][available_cols].copy()
    side_b = df[df["side"] == 1][available_cols].copy()

    rename_a = {col: f"{col}_a" for col in available_cols if col != "match_id"}
    rename_b = {col: f"{col}_b" for col in available_cols if col != "match_id"}

    side_a = side_a.rename(columns=rename_a)
    side_b = side_b.rename(columns=rename_b)

    wide = match_meta.merge(side_a, on="match_id", how="inner").merge(side_b, on="match_id", how="inner")

    # numeric cleanup
    numeric_cols = [
        "best_of",
        "player_ht_a", "player_ht_b",
        "player_age_a", "player_age_b",
        "player_rank_a", "player_rank_b",
        "player_rank_points_a", "player_rank_points_b",
        "player_elo_pre_a", "player_elo_pre_b",
        "player_surface_elo_pre_a", "player_surface_elo_pre_b",
        "matches_played_a", "matches_played_b",
        "win_rate_last10_a", "win_rate_last10_b",
        "aces_avg_last10_a", "aces_avg_last10_b",
        "serve_win_pct_last10_a", "serve_win_pct_last10_b",
        "return_win_pct_last10_a", "return_win_pct_last10_b",
        "bp_conversion_last10_a", "bp_conversion_last10_b",
        "days_since_last_match_a", "days_since_last_match_b",
        "matches_last_30_days_a", "matches_last_30_days_b",
        "serve_win_pct_last10_surface_a", "serve_win_pct_last10_surface_b",
        "return_win_pct_last10_surface_a", "return_win_pct_last10_surface_b",
        "bp_conversion_last10_surface_a", "bp_conversion_last10_surface_b",
        "days_since_last_match_surface_a", "days_since_last_match_surface_b",
        "matches_last_30_days_surface_a", "matches_last_30_days_surface_b",
        "h2h_win_ratio_a", "h2h_win_ratio_b",
        "label_win_a", "label_win_b",
    ]
    for col in numeric_cols:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    # target
    wide["label_player_a_win"] = wide["label_win_a"]

    # helpful metadata
    wide["handedness_combo"] = (_safe_series(wide, "player_hand_a", "U").fillna("U").astype(str)
        + "_vs_" + _safe_series(wide, "player_hand_b", "U").fillna("U").astype(str))

    # context encoding
    surface_clean = _safe_series(wide, "surface", "").fillna("").astype(str).str.upper()
    wide["is_clay"] = (surface_clean == "CLAY").astype(int)
    wide["is_grass"] = (surface_clean == "GRASS").astype(int)

    round_clean = _safe_series(wide, "round", "").fillna("").astype(str).str.upper()
    wide["round_ordinal"] = round_clean.map(ROUND_MAP).fillna(3)

    # delta features (A minus B)
    # rank_adv: positive means A has better ranking (lower numeric rank)
    wide["delta_rank_adv"] = wide["player_rank_b"] - wide["player_rank_a"]

    # keep rank_points delta, but later we can compare with/without it
    wide["delta_rank_points"] = wide["player_rank_points_a"] - wide["player_rank_points_b"]

    # multicollinearity cleanup target:
    # we will use delta_elo / delta_surface_elo, not raw elo_a / elo_b in training
    wide["delta_elo"] = wide["player_elo_pre_a"] - wide["player_elo_pre_b"]
    wide["delta_surface_elo"] = wide["player_surface_elo_pre_a"] - wide["player_surface_elo_pre_b"]

    wide["delta_age"] = wide["player_age_a"] - wide["player_age_b"]
    wide["delta_height"] = wide["player_ht_a"] - wide["player_ht_b"]
    wide["delta_matches_played"] = wide["matches_played_a"] - wide["matches_played_b"]
    wide["delta_win_rate_last10"] = wide["win_rate_last10_a"] - wide["win_rate_last10_b"]
    wide["delta_aces_avg_last10"] = wide["aces_avg_last10_a"] - wide["aces_avg_last10_b"]
    wide["delta_serve_win_pct_last10"] = wide["serve_win_pct_last10_a"] - wide["serve_win_pct_last10_b"]
    wide["delta_return_win_pct_last10"] = wide["return_win_pct_last10_a"] - wide["return_win_pct_last10_b"]
    wide["delta_bp_conversion_last10"] = wide["bp_conversion_last10_a"] - wide["bp_conversion_last10_b"]
    wide["delta_days_since_last_match"] = wide["days_since_last_match_a"] - wide["days_since_last_match_b"]
    wide["delta_matches_last_30_days"] = wide["matches_last_30_days_a"] - wide["matches_last_30_days_b"]
    wide["delta_serve_win_pct_last10_surface"] = (wide["serve_win_pct_last10_surface_a"] - wide["serve_win_pct_last10_surface_b"])
    wide["delta_return_win_pct_last10_surface"] = (wide["return_win_pct_last10_surface_a"] - wide["return_win_pct_last10_surface_b"])
    wide["delta_bp_conversion_last10_surface"] = (wide["bp_conversion_last10_surface_a"] - wide["bp_conversion_last10_surface_b"])
    wide["delta_days_since_last_match_surface"] = (wide["days_since_last_match_surface_a"] - wide["days_since_last_match_surface_b"])
    wide["delta_matches_last_30_days_surface"] = (wide["matches_last_30_days_surface_a"] - wide["matches_last_30_days_surface_b"])
    wide["delta_h2h"] = wide["h2h_win_ratio_a"] - wide["h2h_win_ratio_b"]

    preferred_order = [
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
        "label_player_a_win",
        "delta_rank_adv",
        "delta_rank_points",
        "delta_elo",
        "delta_surface_elo",
        "delta_age",
        "delta_height",
        "delta_matches_played",
        "delta_win_rate_last10",
        "delta_aces_avg_last10",
        "delta_serve_win_pct_last10",
        "delta_return_win_pct_last10",
        "delta_bp_conversion_last10",
        "delta_days_since_last_match",
        "delta_matches_last_30_days",
        "delta_serve_win_pct_last10_surface",
        "delta_return_win_pct_last10_surface",
        "delta_bp_conversion_last10_surface",
        "delta_days_since_last_match_surface",
        "delta_matches_last_30_days_surface",
        "delta_h2h",
        "is_clay",
        "is_grass",
        "round_ordinal",
    ]

    existing_order = [c for c in preferred_order if c in wide.columns]
    remaining_cols = [c for c in wide.columns if c not in existing_order]
    wide = wide[existing_order + remaining_cols]

    return wide


def save_baseline_match_table(baseline_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]Writing baseline match table:[/] {output_path}")
    baseline_df.to_parquet(output_path, index=False)