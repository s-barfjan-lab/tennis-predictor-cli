from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


DEFAULT_ELO = 1500.0
DEFAULT_K = 24.0
BAD_SCORE_TOKENS = ("RET", "W/O", "WO", "DEF", "ABN", "UNP", "CANC")


@dataclass
class EloConfig:
    initial_rating: float = DEFAULT_ELO
    k_factor: float = DEFAULT_K
    date_col: str = "tourney_date"

    # Dynamic K controls
    use_tourney_k: bool = True
    use_margin_k: bool = True

    # Sackmann-style tourney_level multipliers
    # G = Grand Slam
    # M = Masters / top-tier event
    # A = mid tier
    # B = lower tier
    # D/F = Davis/Fed/etc. or miscellaneous team events
    tourney_k_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "G": 1.25,
            "M": 1.10,
            "A": 1.00,
            "B": 0.90,
            "D": 0.85,
            "F": 0.85,
        }
    )

    # Surface handling
    allowed_surfaces: tuple[str, ...] = ("Hard", "Clay", "Grass")
    fallback_surface: str = "Hard"


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
    required = [date_col, "winner_id", "loser_id", "winner_name", "loser_name", ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Elo computation: {missing}")


def _coerce_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        bad_rows = out[out[date_col].isna()]
        raise ValueError(f"Some rows in '{date_col}' could not be converted to datetime. "
            f"Bad row count: {len(bad_rows)}")
    return out


def _stable_match_sort(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensures deterministic chronological processing.
    If multiple matches share the same date, we still want stable order.
    """
    sort_cols = [date_col]

    fallback_cols = [
        col for col in ["tourney_id", "match_num", "round", "winner_id", "loser_id", "winner_name", "loser_name", ]
        if col in df.columns]

    return df.sort_values(sort_cols + fallback_cols).reset_index(drop=True)


def _normalize_surface(surface: object, config: EloConfig) -> str:
    """
    Normalize surfaces to the 3 main buckets used for surface Elo.
    Carpet and unknown surfaces fall back to config.fallback_surface.
    """
    if pd.isna(surface):
        return config.fallback_surface

    value = str(surface).strip().title()

    if value in config.allowed_surfaces:
        return value

    if value == "Carpet":
        return config.fallback_surface

    return config.fallback_surface


def _extract_set_score_parts(set_token: str) -> tuple[int, int] | None:
    """
    Parse a single set token like:
    - 6-4
    - 7-6(5)
    - 6(3)-7
    Returns (games_a, games_b) or None if unusable.
    """
    token = str(set_token).strip()

    if not token or "-" not in token:
        return None

    left, right = token.split("-", 1)

    left_digits = "".join(ch for ch in left if ch.isdigit())
    right_digits = ""
    for ch in right:
        if ch.isdigit():
            right_digits += ch
        else:
            break

    if not left_digits or not right_digits:
        return None

    return int(left_digits), int(right_digits)


def _count_sets_won(score: object) -> tuple[int, int]:
    """
    Count sets won by winner and loser from the score column.

    Examples:
    - '6-4 7-6(5)' -> (2, 0)
    - '6-7(4) 6-3 6-1' -> (2, 1)

    Returns (winner_sets, loser_sets).
    Falls back to (1, 0) if score is missing/unusable.
    """
    if pd.isna(score):
        return 1, 0

    score_text = str(score).strip()
    if not score_text:
        return 1, 0

    upper = score_text.upper()
    if any(tok in upper for tok in BAD_SCORE_TOKENS):
        return 1, 0

    winner_sets = 0
    loser_sets = 0

    for token in score_text.split():
        parsed = _extract_set_score_parts(token)
        if parsed is None:
            continue

        games_w, games_l = parsed
        if games_w > games_l:
            winner_sets += 1
        elif games_l > games_w:
            loser_sets += 1

    if winner_sets == 0 and loser_sets == 0:
        return 1, 0

    return winner_sets, loser_sets


def _should_update_elo(score: object) -> bool:
    if pd.isna(score):
        return True

    score_text = str(score).strip()
    if not score_text:
        return True

    upper = score_text.upper()
    return not any(tok in upper for tok in BAD_SCORE_TOKENS)


def _margin_multiplier(score: object) -> float:
    """
    Simple tennis-specific K adjustment based on decisiveness.
    """
    winner_sets, loser_sets = _count_sets_won(score)
    margin = winner_sets - loser_sets

    if margin >= 3:
        return 1.30
    if margin == 2:
        return 1.20
    if margin == 1:
        return 1.00
    return 1.00


def _tourney_multiplier(tourney_level: object, config: EloConfig) -> float:
    """
    Map tourney_level to a K-factor multiplier.
    Unknown levels default to 1.0.
    """
    if pd.isna(tourney_level):
        return 1.0

    key = str(tourney_level).strip().upper()
    return config.tourney_k_multipliers.get(key, 1.0)


def _effective_k(row, config: EloConfig) -> float:
    """
    Compute match-specific effective K-factor.
    """
    k = config.k_factor

    if config.use_tourney_k:
        tourney_level = getattr(row, "tourney_level", None)
        k *= _tourney_multiplier(tourney_level, config)

    if config.use_margin_k:
        score = getattr(row, "score", None)
        k *= _margin_multiplier(score)

    return k


def _prepare_matches_for_elo(matches: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """
    Shared preprocessing for all Elo pipelines:
    - validate required columns
    - coerce date
    - sort chronologically
    - normalize surface once if available
    """
    _validate_match_columns(matches, config.date_col)

    df = _coerce_date(matches, config.date_col)
    df = _stable_match_sort(df, config.date_col).copy()

    if "surface" in df.columns:
        df["surface_normalized"] = df["surface"].apply(lambda x: _normalize_surface(x, config))

    return df


def _compute_overall_elo_on_prepared(df: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
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

        if _should_update_elo(getattr(row, "score", None)):
            k = _effective_k(row, config)
            new_r_winner = r_winner + k * (1.0 - p_winner)
            new_r_loser = r_loser + k * (0.0 - p_loser)
            ratings[winner_key] = new_r_winner
            ratings[loser_key] = new_r_loser
        else:
            new_r_winner = r_winner
            new_r_loser = r_loser

        winner_elo_pre.append(r_winner)
        loser_elo_pre.append(r_loser)
        winner_elo_post.append(new_r_winner)
        loser_elo_post.append(new_r_loser)
        elo_diff_pre.append(r_winner - r_loser)
        elo_prob_winner_pre.append(p_winner)

    out = df.copy()
    out["winner_elo_pre"] = winner_elo_pre
    out["loser_elo_pre"] = loser_elo_pre
    out["winner_elo_post"] = winner_elo_post
    out["loser_elo_post"] = loser_elo_post
    out["elo_diff_pre"] = elo_diff_pre
    out["elo_prob_winner_pre"] = elo_prob_winner_pre

    return out


def _compute_surface_elo_on_prepared(df: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    if "surface_normalized" not in df.columns:
        raise ValueError("Missing normalized surface column for surface Elo.")

    surface_ratings: Dict[str, Dict[str, float]] = {
        surface: {} for surface in config.allowed_surfaces
    }

    winner_surface_elo_pre = []
    loser_surface_elo_pre = []
    winner_surface_elo_post = []
    loser_surface_elo_post = []
    surface_elo_diff_pre = []
    surface_elo_prob_winner_pre = []

    def initial_surface_rating(global_rating: object) -> float:
        if pd.isna(global_rating):
            return config.initial_rating
        return 0.70 * float(global_rating) + 0.30 * config.initial_rating

    for row in df.itertuples(index=False):
        surface = row.surface_normalized
        ratings = surface_ratings[surface]

        winner_key = str(row.winner_id)
        loser_key = str(row.loser_id)

        r_winner = ratings.get(
            winner_key,
            initial_surface_rating(getattr(row, "winner_elo_pre", pd.NA)),
        )
        r_loser = ratings.get(
            loser_key,
            initial_surface_rating(getattr(row, "loser_elo_pre", pd.NA)),
        )

        p_winner = expected_score(r_winner, r_loser)
        p_loser = 1.0 - p_winner

        if _should_update_elo(getattr(row, "score", None)):
            k = _effective_k(row, config)
            new_r_winner = r_winner + k * (1.0 - p_winner)
            new_r_loser = r_loser + k * (0.0 - p_loser)
            ratings[winner_key] = new_r_winner
            ratings[loser_key] = new_r_loser
        else:
            new_r_winner = r_winner
            new_r_loser = r_loser

        winner_surface_elo_pre.append(r_winner)
        loser_surface_elo_pre.append(r_loser)
        winner_surface_elo_post.append(new_r_winner)
        loser_surface_elo_post.append(new_r_loser)
        surface_elo_diff_pre.append(r_winner - r_loser)
        surface_elo_prob_winner_pre.append(p_winner)

    out = df.copy()
    out["winner_surface_elo_pre"] = winner_surface_elo_pre
    out["loser_surface_elo_pre"] = loser_surface_elo_pre
    out["winner_surface_elo_post"] = winner_surface_elo_post
    out["loser_surface_elo_post"] = loser_surface_elo_post
    out["surface_elo_diff_pre"] = surface_elo_diff_pre
    out["surface_elo_prob_winner_pre"] = surface_elo_prob_winner_pre

    return out


def compute_elo_features(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Compute pre-match overall Elo ratings and post-match updates chronologically.
    """
    if config is None:
        config = EloConfig()

    prepared = _prepare_matches_for_elo(matches, config)
    return _compute_overall_elo_on_prepared(prepared, config)


def compute_surface_elo_features(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Compute surface-specific Elo ratings chronologically.
    """
    if config is None:
        config = EloConfig()

    prepared = _prepare_matches_for_elo(matches, config)

    if "surface" not in prepared.columns:
        raise ValueError("Missing required column for surface Elo: ['surface']")

    return _compute_surface_elo_on_prepared(prepared, config)


def compute_all_elo_features(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """
    Compute both overall Elo and surface-specific Elo in one dataframe,
    using shared preprocessing only once.
    """
    if config is None:
        config = EloConfig()

    prepared = _prepare_matches_for_elo(matches, config)

    if "surface" not in prepared.columns:
        raise ValueError("Missing required column for surface Elo: ['surface']")

    overall = _compute_overall_elo_on_prepared(prepared, config)
    surface = _compute_surface_elo_on_prepared(overall, config)

    cols_to_add = [
        "surface_normalized",
        "winner_surface_elo_pre",
        "loser_surface_elo_pre",
        "winner_surface_elo_post",
        "loser_surface_elo_post",
        "surface_elo_diff_pre",
        "surface_elo_prob_winner_pre",
    ]

    out = overall.copy()
    for col in cols_to_add:
        out[col] = surface[col].values

    return out


def build_latest_player_elo_snapshot(elo_matches: pd.DataFrame, date_col: str = "tourney_date", ) -> pd.DataFrame:
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

    winners = elo_matches[[date_col, "winner_id", "winner_name", "winner_elo_pre", "winner_elo_post"]].copy()
    winners.columns = ["date", "player_id", "player_name", "elo_pre", "elo_post"]

    losers = elo_matches[[date_col, "loser_id", "loser_name", "loser_elo_pre", "loser_elo_post"]].copy()
    losers.columns = ["date", "player_id", "player_name", "elo_pre", "elo_post"]

    out = pd.concat([winners, losers], ignore_index=True)
    out = out.sort_values(["date", "player_id", "player_name"]).reset_index(drop=True)
    return out
