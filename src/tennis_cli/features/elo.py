from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


DEFAULT_ELO = 1500.0
DEFAULT_K = 32.0


@dataclass
class EloConfig:
    initial_rating: float = DEFAULT_ELO
    k_factor: float = DEFAULT_K
    date_col: str = "tourney_date"


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Standard Elo expected score for player A against player B.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_win_probability(rating_a: float, rating_b: float) -> float:
    """
    Alias for expected_score, useful for thesis readability.
    """
    return expected_score(rating_a, rating_b)


def _validate_match_columns(df: pd.DataFrame, date_col: str) -> None:
    required = [
        date_col,
        "winner_id",
        "loser_id",
        "winner_name",
        "loser_name",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Elo computation: {missing}")


def _coerce_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        bad_rows = out[out[date_col].isna()]
        raise ValueError(
            f"Some rows in '{date_col}' could not be converted to datetime. "
            f"Bad row count: {len(bad_rows)}"
        )
    return out


def _stable_match_sort(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensures deterministic chronological processing.
    If multiple matches share the same date, we still want stable order.
    """
    sort_cols = [date_col]

    fallback_cols = [
        col for col in [
            "tourney_id",
            "match_num",
            "round",
            "winner_id",
            "loser_id",
            "winner_name",
            "loser_name",
        ]
        if col in df.columns
    ]

    return df.sort_values(sort_cols + fallback_cols).reset_index(drop=True)


def compute_elo_features(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Compute pre-match Elo ratings and post-match updates chronologically.

    Input assumptions:
    - one row per match
    - columns include:
      date_col, winner_id, loser_id, winner_name, loser_name
    - winner/loser reflect actual match result

    Output:
    Original match dataframe plus:
    - winner_elo_pre
    - loser_elo_pre
    - winner_elo_post
    - loser_elo_post
    - elo_diff_pre
    - elo_prob_winner_pre
    """
    if config is None:
        config = EloConfig()

    _validate_match_columns(matches, config.date_col)

    df = _coerce_date(matches, config.date_col)
    df = _stable_match_sort(df, config.date_col)

    ratings: Dict[str, float] = {}

    winner_elo_pre = []
    loser_elo_pre = []
    winner_elo_post = []
    loser_elo_post = []
    elo_diff_pre = []
    elo_prob_winner_pre = []

    for row in df.itertuples(index=False):
        winner_key = str(row.winner_id)
        loser_key = str(row.loser_id)

        r_winner = ratings.get(winner_key, config.initial_rating)
        r_loser = ratings.get(loser_key, config.initial_rating)

        p_winner = expected_score(r_winner, r_loser)
        p_loser = 1.0 - p_winner

        new_r_winner = r_winner + config.k_factor * (1.0 - p_winner)
        new_r_loser = r_loser + config.k_factor * (0.0 - p_loser)

        winner_elo_pre.append(r_winner)
        loser_elo_pre.append(r_loser)
        winner_elo_post.append(new_r_winner)
        loser_elo_post.append(new_r_loser)
        elo_diff_pre.append(r_winner - r_loser)
        elo_prob_winner_pre.append(p_winner)

        ratings[winner_key] = new_r_winner
        ratings[loser_key] = new_r_loser

    out = df.copy()
    out["winner_elo_pre"] = winner_elo_pre
    out["loser_elo_pre"] = loser_elo_pre
    out["winner_elo_post"] = winner_elo_post
    out["loser_elo_post"] = loser_elo_post
    out["elo_diff_pre"] = elo_diff_pre
    out["elo_prob_winner_pre"] = elo_prob_winner_pre

    return out


def build_latest_player_elo_snapshot(
    elo_matches: pd.DataFrame,
    date_col: str = "tourney_date",
) -> pd.DataFrame:
    """
    Builds a player-level snapshot table from an Elo-enriched match table.

    Returns one row per player per match appearance with:
    - player_id
    - player_name
    - date
    - elo_pre
    - elo_post
    """
    required = [
        date_col,
        "winner_id",
        "loser_id",
        "winner_name",
        "loser_name",
        "winner_elo_pre",
        "loser_elo_pre",
        "winner_elo_post",
        "loser_elo_post",
    ]
    missing = [col for col in required if col not in elo_matches.columns]
    if missing:
        raise ValueError(f"Missing required Elo columns for snapshot build: {missing}")

    winners = elo_matches[
        [date_col, "winner_id", "winner_name", "winner_elo_pre", "winner_elo_post"]
    ].copy()
    winners.columns = ["date", "player_id", "player_name", "elo_pre", "elo_post"]

    losers = elo_matches[
        [date_col, "loser_id", "loser_name", "loser_elo_pre", "loser_elo_post"]
    ].copy()
    losers.columns = ["date", "player_id", "player_name", "elo_pre", "elo_post"]

    out = pd.concat([winners, losers], ignore_index=True)
    out = out.sort_values(["date", "player_id", "player_name"]).reset_index(drop=True)
    return out