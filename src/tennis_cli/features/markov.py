from __future__ import annotations

from functools import lru_cache
from math import comb

import numpy as np
import pandas as pd


NEUTRAL_POINT_PROB = 0.5
PROB_EPSILON = 0.01


def _clip_probability(p: object) -> float:
    p_num = pd.to_numeric(pd.Series([p]), errors="coerce").iloc[0]
    if pd.isna(p_num):
        return NEUTRAL_POINT_PROB
    return float(np.clip(float(p_num), PROB_EPSILON, 1.0 - PROB_EPSILON))


def game_win_prob_from_point_prob(p: float) -> float:
    """
    Return the probability that the server holds from point-win probability p.

    Sipko (2015), following the tennis Markov-chain literature, decomposes
    match probability recursively from point to game, set, and match states.
    This function uses O'Malley's closed-form service-game probability, which
    is algebraically equivalent to the recursive game model:

        p^4 * (15 - 34p + 28p^2 - 8p^3) / (1 - 2p + 2p^2)

    The receiver's break probability is the complement of the opponent's hold
    probability; there is no separate break formula.

    References:
    - Sipko, M. (2015). Machine Learning for the Prediction of Professional
      Tennis Matches.
    - O'Malley, A. J. (2008). Probability Formulas and Statistical Analysis in
      Tennis.
    """
    p = _clip_probability(p)
    numerator = (p ** 4) * (15.0 - (34.0 * p) + (28.0 * (p ** 2)) - (8.0 * (p ** 3)))
    denominator = 1.0 - (2.0 * p) + (2.0 * (p ** 2))
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _tiebreak_server(point_index: int, first_server: str) -> str:
    if point_index == 0:
        return first_server

    first_server = first_server.upper()
    other_server = "B" if first_server == "A" else "A"
    return other_server if ((point_index - 1) // 2) % 2 == 0 else first_server


def tiebreak_win_prob(p_a_serve_point: float, p_b_serve_point: float,
    first_server: str = "A", max_points: int = 60, ) -> float:
    """
    Approximate A's tiebreak win probability with recursive point DP.

    The serve order follows the standard tiebreak pattern: one point by the
    first server, then two-point service blocks alternating. The recursion is
    truncated after `max_points`; unresolved long tiebreaks are assigned 0.5.
    That tail is tiny for the clipped probabilities used here, and this keeps
    the implementation transparent until a full O'Malley tiebreak formula is
    needed.
    """
    p_a_serve_point = _clip_probability(p_a_serve_point)
    p_b_serve_point = _clip_probability(p_b_serve_point)
    first_server = first_server.upper()
    if first_server not in {"A", "B"}:
        raise ValueError("first_server must be 'A' or 'B'")

    @lru_cache(maxsize=None)
    def rec(a_points: int, b_points: int) -> float:
        if a_points >= 7 and a_points - b_points >= 2:
            return 1.0
        if b_points >= 7 and b_points - a_points >= 2:
            return 0.0
        if a_points + b_points >= max_points:
            return NEUTRAL_POINT_PROB

        server = _tiebreak_server(a_points + b_points, first_server)
        p_a_wins_point = p_a_serve_point if server == "A" else 1.0 - p_b_serve_point
        return (p_a_wins_point * rec(a_points + 1, b_points)) + (
            (1.0 - p_a_wins_point) * rec(a_points, b_points + 1))

    return float(np.clip(rec(0, 0), 0.0, 1.0))


def set_win_prob_from_hold_probs(hold_a: float, hold_b: float, p_a_serve_point: float,
    p_b_serve_point: float, first_server: str = "A", ) -> float:
    """
    Return A's set win probability from hold probabilities.

    Games alternate server. At 6-6 this uses `tiebreak_win_prob`; because 12
    games have been played, the tiebreak first server is the same player who
    served first in the set.
    """
    hold_a = _clip_probability(hold_a)
    hold_b = _clip_probability(hold_b)
    first_server = first_server.upper()
    if first_server not in {"A", "B"}:
        raise ValueError("first_server must be 'A' or 'B'")

    @lru_cache(maxsize=None)
    def rec(a_games: int, b_games: int) -> float:
        if a_games >= 6 and a_games - b_games >= 2:
            return 1.0
        if b_games >= 6 and b_games - a_games >= 2:
            return 0.0
        if a_games == 6 and b_games == 6:
            return tiebreak_win_prob(p_a_serve_point, p_b_serve_point, first_server=first_server)

        game_index = a_games + b_games
        server = first_server if game_index % 2 == 0 else ("B" if first_server == "A" else "A")
        p_a_wins_game = hold_a if server == "A" else 1.0 - hold_b
        return (p_a_wins_game * rec(a_games + 1, b_games)) + (
            (1.0 - p_a_wins_game) * rec(a_games, b_games + 1))

    return float(np.clip(rec(0, 0), 0.0, 1.0))


def match_win_prob_from_set_prob(set_prob: float, best_of: object = 3) -> float:
    set_prob = _clip_probability(set_prob)
    best_of_num = pd.to_numeric(pd.Series([best_of]), errors="coerce").iloc[0]
    best_of_int = 5 if not pd.isna(best_of_num) and int(best_of_num) == 5 else 3
    sets_to_win = (best_of_int // 2) + 1

    prob = 0.0
    for losses in range(sets_to_win):
        sets_played_before_final = (sets_to_win - 1) + losses
        prob += (
            comb(sets_played_before_final, losses)
            * (set_prob ** sets_to_win)
            * ((1.0 - set_prob) ** losses))

    return float(np.clip(prob, 0.0, 1.0))


def markov_match_win_probability(p_a_serve_point: float, p_b_serve_point: float,
    best_of: object = 3, ) -> float:
    """
    Estimate A's pre-match win probability from serve-point probabilities.

    The first server is unknown in this pre-match feature, so this averages the
    match probability from the two first-server scenarios. Set probabilities
    are treated as identically distributed within a match; this is a deliberate
    pre-match approximation and a domain feature for the ML model, not a full
    replacement for the model.
    """
    p_a_serve_point = _clip_probability(p_a_serve_point)
    p_b_serve_point = _clip_probability(p_b_serve_point)
    hold_a = game_win_prob_from_point_prob(p_a_serve_point)
    hold_b = game_win_prob_from_point_prob(p_b_serve_point)

    set_prob_a_first = set_win_prob_from_hold_probs(hold_a, hold_b,
        p_a_serve_point, p_b_serve_point, first_server="A", )
    set_prob_b_first = set_win_prob_from_hold_probs(hold_a, hold_b,
        p_a_serve_point, p_b_serve_point, first_server="B", )

    match_prob_a_first = match_win_prob_from_set_prob(set_prob_a_first, best_of=best_of)
    match_prob_b_first = match_win_prob_from_set_prob(set_prob_b_first, best_of=best_of)
    return float(np.clip((match_prob_a_first + match_prob_b_first) / 2.0, 0.0, 1.0))


def _history_or_neutral(history: pd.Series, has_history: pd.Series) -> pd.Series:
    values = pd.to_numeric(history, errors="coerce")
    flags = pd.to_numeric(has_history, errors="coerce").fillna(0).astype(int)
    return values.where(flags == 1, NEUTRAL_POINT_PROB).fillna(NEUTRAL_POINT_PROB)


def add_markov_match_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Markov-derived match probability features to a baseline match table.

    Expected inputs are the deterministic A/B columns produced by
    `build_baseline_match_table`: `service_points_won_pct_30_*`,
    `return_points_won_pct_30_*`, `has_serve_history_*`,
    `has_return_history_*`, and `best_of`.
    """
    df = match_df.copy()
    n_rows = len(df)

    defaults = {
        "service_points_won_pct_30_a": NEUTRAL_POINT_PROB,
        "service_points_won_pct_30_b": NEUTRAL_POINT_PROB,
        "return_points_won_pct_30_a": NEUTRAL_POINT_PROB,
        "return_points_won_pct_30_b": NEUTRAL_POINT_PROB,
        "has_serve_history_a": 0,
        "has_serve_history_b": 0,
        "has_return_history_a": 0,
        "has_return_history_b": 0,
        "best_of": 3,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = pd.Series([default] * n_rows, index=df.index)

    serve_a = _history_or_neutral(df["service_points_won_pct_30_a"], df["has_serve_history_a"])
    serve_b = _history_or_neutral(df["service_points_won_pct_30_b"], df["has_serve_history_b"])
    return_a = _history_or_neutral(df["return_points_won_pct_30_a"], df["has_return_history_a"])
    return_b = _history_or_neutral(df["return_points_won_pct_30_b"], df["has_return_history_b"])

    p_a_serve_point = (serve_a + (1.0 - return_b)) / 2.0
    p_b_serve_point = (serve_b + (1.0 - return_a)) / 2.0

    probs = [
        markov_match_win_probability(p_a, p_b, best_of=best_of)
        for p_a, p_b, best_of in zip(p_a_serve_point, p_b_serve_point, df["best_of"])
    ]

    df["has_serve_history"] = (
        (pd.to_numeric(df["has_serve_history_a"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df["has_serve_history_b"], errors="coerce").fillna(0).astype(int) == 1)
    ).astype(int)
    df["has_return_history"] = (
        (pd.to_numeric(df["has_return_history_a"], errors="coerce").fillna(0).astype(int) == 1)
        & (pd.to_numeric(df["has_return_history_b"], errors="coerce").fillna(0).astype(int) == 1)
    ).astype(int)
    df["markov_match_win_prob_a"] = probs
    df["markov_match_win_prob_b"] = 1.0 - df["markov_match_win_prob_a"]
    df["delta_markov_match_win"] = df["markov_match_win_prob_a"] - df["markov_match_win_prob_b"]

    return df
