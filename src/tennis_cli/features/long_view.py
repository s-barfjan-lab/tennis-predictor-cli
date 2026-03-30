from __future__ import annotations

from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()


def _safe_col(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    """
    Return a column if it exists, otherwise return a default-filled Series.

    This makes the code robust to schema differences across datasets/years.
    """
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _build_match_id(df: pd.DataFrame) -> pd.Series:
    """
    Build a deterministic match identifier from available columns.

    We do this because the processed match tables may not come with a single
    official unique match_id column.
    """
    tourney_date = _safe_col(df, "tourney_date", "").astype(str)
    tourney_name = _safe_col(df, "tourney_name", "").astype(str)
    round_col = _safe_col(df, "round", "").astype(str)
    winner_id = _safe_col(df, "winner_id", "").astype(str)
    loser_id = _safe_col(df, "loser_id", "").astype(str)

    match_id = (
        tourney_date
        + "__"
        + tourney_name
        + "__"
        + round_col
        + "__"
        + winner_id
        + "__"
        + loser_id
    )
    return match_id


def build_long_view(matches_df: pd.DataFrame, tour: str) -> pd.DataFrame:
    """
    Convert a match-centric dataframe into a player-centric long view.

    Input: one row per match
    Output: two rows per match (player perspective)
    """
    df = matches_df.copy()

    # Ensure deterministic ordering
    if "tourney_date" in df.columns:
        df = df.sort_values("tourney_date").reset_index(drop=True)

    df["match_id"] = _build_match_id(df)
    df["tour"] = tour.lower().strip()

    # -------------------------
    # Winner-perspective rows
    # -------------------------
    winners = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "tour": df["tour"],
            "tourney_date": _safe_col(df, "tourney_date"),
            "tourney_name": _safe_col(df, "tourney_name"),
            "surface": _safe_col(df, "surface"),
            "round": _safe_col(df, "round"),
            "best_of": _safe_col(df, "best_of"),
            "minutes": _safe_col(df, "minutes"),

            "player_id": _safe_col(df, "winner_id"),
            "player_name": _safe_col(df, "winner_name"),
            "player_hand": _safe_col(df, "winner_hand"),
            "player_ht": _safe_col(df, "winner_ht"),
            "player_age": _safe_col(df, "winner_age"),
            "player_rank": _safe_col(df, "winner_rank"),
            "player_rank_points": _safe_col(df, "winner_rank_points"),
            "player_seed": _safe_col(df, "winner_seed"),

            "opponent_id": _safe_col(df, "loser_id"),
            "opponent_name": _safe_col(df, "loser_name"),
            "opponent_hand": _safe_col(df, "loser_hand"),
            "opponent_ht": _safe_col(df, "loser_ht"),
            "opponent_age": _safe_col(df, "loser_age"),
            "opponent_rank": _safe_col(df, "loser_rank"),
            "opponent_rank_points": _safe_col(df, "loser_rank_points"),
            "opponent_seed": _safe_col(df, "loser_seed"),

            "is_winner": 1,
            "label_win": 1,

            "aces": _safe_col(df, "w_ace"),
            "double_faults": _safe_col(df, "w_df"),
            "serve_points": _safe_col(df, "w_svpt"),
            "first_in": _safe_col(df, "w_1stIn"),
            "first_won": _safe_col(df, "w_1stWon"),
            "second_won": _safe_col(df, "w_2ndWon"),
            "service_games": _safe_col(df, "w_SvGms"),
            "break_points_saved": _safe_col(df, "w_bpSaved"),
            "break_points_faced": _safe_col(df, "w_bpFaced"),
        }
    )

    # -------------------------
    # Loser-perspective rows
    # -------------------------
    losers = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "tour": df["tour"],
            "tourney_date": _safe_col(df, "tourney_date"),
            "tourney_name": _safe_col(df, "tourney_name"),
            "surface": _safe_col(df, "surface"),
            "round": _safe_col(df, "round"),
            "best_of": _safe_col(df, "best_of"),
            "minutes": _safe_col(df, "minutes"),

            "player_id": _safe_col(df, "loser_id"),
            "player_name": _safe_col(df, "loser_name"),
            "player_hand": _safe_col(df, "loser_hand"),
            "player_ht": _safe_col(df, "loser_ht"),
            "player_age": _safe_col(df, "loser_age"),
            "player_rank": _safe_col(df, "loser_rank"),
            "player_rank_points": _safe_col(df, "loser_rank_points"),
            "player_seed": _safe_col(df, "loser_seed"),

            "opponent_id": _safe_col(df, "winner_id"),
            "opponent_name": _safe_col(df, "winner_name"),
            "opponent_hand": _safe_col(df, "winner_hand"),
            "opponent_ht": _safe_col(df, "winner_ht"),
            "opponent_age": _safe_col(df, "winner_age"),
            "opponent_rank": _safe_col(df, "winner_rank"),
            "opponent_rank_points": _safe_col(df, "winner_rank_points"),
            "opponent_seed": _safe_col(df, "winner_seed"),

            "is_winner": 0,
            "label_win": 0,

            "aces": _safe_col(df, "l_ace"),
            "double_faults": _safe_col(df, "l_df"),
            "serve_points": _safe_col(df, "l_svpt"),
            "first_in": _safe_col(df, "l_1stIn"),
            "first_won": _safe_col(df, "l_1stWon"),
            "second_won": _safe_col(df, "l_2ndWon"),
            "service_games": _safe_col(df, "l_SvGms"),
            "break_points_saved": _safe_col(df, "l_bpSaved"),
            "break_points_faced": _safe_col(df, "l_bpFaced"),
        }
    )

    long_df = pd.concat([winners, losers], ignore_index=True)

    # Numeric cleanup where possible
    numeric_cols = [
        "player_ht",
        "player_age",
        "player_rank",
        "player_rank_points",
        "opponent_ht",
        "opponent_age",
        "opponent_rank",
        "opponent_rank_points",
        "minutes",
        "aces",
        "double_faults",
        "serve_points",
        "first_in",
        "first_won",
        "second_won",
        "service_games",
        "break_points_saved",
        "break_points_faced",
    ]

    for col in numeric_cols:
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    # Derived player-centric per-match stats
    long_df["first_serve_in_pct"] = long_df["first_in"] / long_df["serve_points"]
    long_df["first_serve_won_pct"] = long_df["first_won"] / long_df["first_in"]
    long_df["second_serve_attempts"] = long_df["serve_points"] - long_df["first_in"]
    long_df["second_serve_won_pct"] = long_df["second_won"] / long_df["second_serve_attempts"]
    long_df["aces_per_service_point"] = long_df["aces"] / long_df["serve_points"]
    long_df["df_per_service_point"] = long_df["double_faults"] / long_df["serve_points"]
    long_df["bp_saved_pct"] = long_df["break_points_saved"] / long_df["break_points_faced"]

    # Service points won
    long_df["service_points_won"] = long_df["first_won"] + long_df["second_won"]
    long_df["service_points_won_pct"] = long_df["service_points_won"] / long_df["serve_points"]

    # Clean infinities from divisions
    long_df = long_df.replace([float("inf"), float("-inf")], pd.NA)

    # Final order for readability
    preferred_order = [
        "match_id",
        "tour",
        "tourney_date",
        "tourney_name",
        "surface",
        "round",
        "best_of",
        "player_id",
        "player_name",
        "opponent_id",
        "opponent_name",
        "label_win",
        "player_rank",
        "opponent_rank",
        "aces",
        "double_faults",
        "serve_points",
        "first_serve_in_pct",
        "first_serve_won_pct",
        "second_serve_won_pct",
        "service_points_won_pct",
        "bp_saved_pct",
    ]

    existing_order = [c for c in preferred_order if c in long_df.columns]
    remaining_cols = [c for c in long_df.columns if c not in existing_order]
    long_df = long_df[existing_order + remaining_cols]

    return long_df


def save_long_view(long_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the long-view dataset to parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold green]Writing long-view dataset:[/] {output_path}")
    long_df.to_parquet(output_path, index=False)