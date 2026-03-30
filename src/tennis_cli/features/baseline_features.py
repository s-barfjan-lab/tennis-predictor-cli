from __future__ import annotations

import pandas as pd
from rich.console import Console

console = Console()


def _flatten_unstacked_columns(columns) -> list[str]:
    """
    Flatten MultiIndex columns produced by unstack().

    Example:
    ('player_rank', 0) -> player_rank_a
    ('player_rank', 1) -> player_rank_b
    """
    flat = []
    for col_name, side in columns:
        suffix = "a" if side == 0 else "b"
        flat.append(f"{col_name}_{suffix}")
    return flat


def build_baseline_match_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the player-centric long view into a match-row baseline table
    with delta features.

    Output:
    - one row per match
    - symmetric A/B ordering
    - delta features = A - B (or B - A where needed to keep interpretation clean)
    - label = did player A win?
    """
    df = long_df.copy()

    # Keep only matches that have exactly two player rows
    counts = df.groupby("match_id").size()
    valid_match_ids = counts[counts == 2].index
    df = df[df["match_id"].isin(valid_match_ids)].copy()

    # Deterministic player ordering independent of the outcome
    # This avoids bias such as "winner is always player_a"
    df["_player_sort_key"] = (
        df["player_id"].astype("string").fillna("NA")
        + "__"
        + df["player_name"].astype("string").fillna("NA")
    )

    df = df.sort_values(["match_id", "_player_sort_key"]).copy()
    df["side"] = df.groupby("match_id").cumcount()

    # Columns we want to carry from the long view into the match-row table
    keep_cols = [
        "match_id",
        "tour",
        "tourney_date",
        "surface",
        "round",
        "best_of",

        "player_id",
        "player_name",
        "player_hand",
        "player_ht",
        "player_age",
        "player_rank",
        "player_rank_points",

        "label_win",

        "matches_played",
        "win_rate_last10",
        "aces_avg_last10",
        "serve_win_pct_last10",
        "days_since_last_match",
        "matches_last_30_days",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    wide = (
        df[keep_cols + ["side"]]
        .set_index(["match_id", "side"])
        .unstack("side")
    )

    # Flatten columns like ("player_rank", 0) -> player_rank_a
    wide.columns = _flatten_unstacked_columns(wide.columns)
    wide = wide.reset_index()

    # -------------------------
    # Label
    # -------------------------
    wide["label_player_a_win"] = wide["label_win_a"]

    # -------------------------
    # Human-readable context
    # -------------------------
    wide["handedness_combo"] = (
        wide["player_hand_a"].astype("string").fillna("U")
        + "_vs_"
        + wide["player_hand_b"].astype("string").fillna("U")
    )

    # -------------------------
    # Delta features
    # -------------------------
    # For rank, lower is better, so we define:
    # delta_rank_adv = rank_B - rank_A
    # positive => player A has a better ranking position
    wide["delta_rank_adv"] = wide["player_rank_b"] - wide["player_rank_a"]

    # Rank points: higher is better, so A - B is already intuitive
    wide["delta_rank_points"] = wide["player_rank_points_a"] - wide["player_rank_points_b"]

    # Physical / demographic differences
    wide["delta_age"] = wide["player_age_a"] - wide["player_age_b"]
    wide["delta_height"] = wide["player_ht_a"] - wide["player_ht_b"]

    # Experience / form / fatigue
    wide["delta_matches_played"] = wide["matches_played_a"] - wide["matches_played_b"]
    wide["delta_win_rate_last10"] = wide["win_rate_last10_a"] - wide["win_rate_last10_b"]
    wide["delta_aces_avg_last10"] = wide["aces_avg_last10_a"] - wide["aces_avg_last10_b"]
    wide["delta_serve_win_pct_last10"] = (
        wide["serve_win_pct_last10_a"] - wide["serve_win_pct_last10_b"]
    )
    wide["delta_days_since_last_match"] = (
        wide["days_since_last_match_a"] - wide["days_since_last_match_b"]
    )
    wide["delta_matches_last_30_days"] = (
        wide["matches_last_30_days_a"] - wide["matches_last_30_days_b"]
    )

    # -------------------------
    # Final output selection
    # -------------------------
    output_cols = [
        "match_id",
        "tour_a",
        "tourney_date_a",
        "surface_a",
        "round_a",
        "best_of_a",

        "player_id_a",
        "player_name_a",
        "player_id_b",
        "player_name_b",
        "handedness_combo",

        "label_player_a_win",

        "delta_rank_adv",
        "delta_rank_points",
        "delta_age",
        "delta_height",
        "delta_matches_played",
        "delta_win_rate_last10",
        "delta_aces_avg_last10",
        "delta_serve_win_pct_last10",
        "delta_days_since_last_match",
        "delta_matches_last_30_days",
    ]

    output_cols = [c for c in output_cols if c in wide.columns]
    out = wide[output_cols].copy()

    # Rename common context columns to clean names
    rename_map = {
        "tour_a": "tour",
        "tourney_date_a": "tourney_date",
        "surface_a": "surface",
        "round_a": "round",
        "best_of_a": "best_of",
    }
    out = out.rename(columns=rename_map)

    return out


def save_baseline_match_table(df: pd.DataFrame, output_path) -> None:
    """
    Save the baseline match-row feature table.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]Writing baseline feature table:[/] {output_path}")
    df.to_parquet(output_path, index=False)