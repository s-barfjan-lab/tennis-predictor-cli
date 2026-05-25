from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


NEUTRAL_POINT_PROB = 0.5
PROB_EPSILON = 0.01
DEFAULT_PRIOR_STRENGTH = 48.0
MAX_TIEBREAK_POINTS = 60

SERVER_A = "A"
SERVER_B = "B"
SERVERS = {SERVER_A, SERVER_B}


def _clip_probability(p: object) -> float:
    p_num = pd.to_numeric(pd.Series([p]), errors="coerce").iloc[0]
    if pd.isna(p_num):
        return NEUTRAL_POINT_PROB
    return float(np.clip(float(p_num), PROB_EPSILON, 1.0 - PROB_EPSILON))


def _other_server(server: str) -> str:
    server = _normalize_server(server)
    return SERVER_B if server == SERVER_A else SERVER_A


def _normalize_server(server: str) -> str:
    normalized = str(server).strip().upper()
    if normalized not in SERVERS:
        raise ValueError("server must be 'A' or 'B'")
    return normalized


def _sets_to_win(best_of: object) -> int:
    best_of_num = pd.to_numeric(pd.Series([best_of]), errors="coerce").iloc[0]
    best_of_int = 5 if not pd.isna(best_of_num) and int(best_of_num) == 5 else 3
    return (best_of_int // 2) + 1


def parse_tennis_point_score(value: Any) -> int:
    """
    Convert a tennis point label to point-count form.

    Normal-game labels map as `0 -> 0`, `15 -> 1`, `30 -> 2`,
    `40 -> 3`, and `AD/A -> 4`. Numeric values other than 15/30/40 are
    accepted directly so the same parser can be used for tiebreak points.
    """
    if value is None:
        return 0

    if isinstance(value, str):
        text = value.strip().upper()
        labels = {
            "": 0,
            "0": 0,
            "LOVE": 0,
            "15": 1,
            "30": 2,
            "40": 3,
            "A": 4,
            "AD": 4,
            "ADV": 4,
            "ADVANTAGE": 4,
        }
        if text in labels:
            return labels[text]
        try:
            parsed = int(text)
        except ValueError as exc:
            raise ValueError(f"unsupported point score: {value!r}") from exc
        return parsed

    parsed = int(value)
    if parsed == 15:
        return 1
    if parsed == 30:
        return 2
    if parsed == 40:
        return 3
    return parsed


def _normalize_normal_game_points(points_a: int, points_b: int) -> tuple[int, int]:
    if points_a < 0 or points_b < 0:
        raise ValueError("point scores cannot be negative")

    if points_a >= 3 and points_b >= 3:
        diff = points_a - points_b
        if diff == 0:
            return 3, 3
        if diff == 1:
            return 4, 3
        if diff == -1:
            return 3, 4

    return points_a, points_b


def _normal_game_winner(points_a: int, points_b: int) -> str | None:
    if points_a >= 4 and points_a - points_b >= 2:
        return SERVER_A
    if points_b >= 4 and points_b - points_a >= 2:
        return SERVER_B
    return None


def normal_game_win_probability_from_score(
    points_a: Any,
    points_b: Any,
    server: str,
    p_a_serve_point: float,
    p_b_serve_point: float,
) -> float:
    """
    Return A's probability of winning the current non-tiebreak game.

    The game is solved from the current point score with the deuce/advantage
    loop collapsed analytically, which avoids recursion cycles at deuce.
    """
    server = _normalize_server(server)
    p_a_serve = _clip_probability(p_a_serve_point)
    p_b_serve = _clip_probability(p_b_serve_point)
    p_a_point = p_a_serve if server == SERVER_A else 1.0 - p_b_serve
    p_b_point = 1.0 - p_a_point
    start_a, start_b = _normalize_normal_game_points(
        parse_tennis_point_score(points_a),
        parse_tennis_point_score(points_b),
    )
    deuce_prob = (p_a_point**2) / ((p_a_point**2) + (p_b_point**2))

    @lru_cache(maxsize=None)
    def rec(a_points: int, b_points: int) -> float:
        normalized_a, normalized_b = _normalize_normal_game_points(a_points, b_points)
        if (normalized_a, normalized_b) != (a_points, b_points):
            return rec(normalized_a, normalized_b)

        winner = _normal_game_winner(a_points, b_points)
        if winner == SERVER_A:
            return 1.0
        if winner == SERVER_B:
            return 0.0

        if a_points == 3 and b_points == 3:
            return deuce_prob
        if a_points == 4 and b_points == 3:
            return p_a_point + (p_b_point * deuce_prob)
        if a_points == 3 and b_points == 4:
            return p_a_point * deuce_prob

        return (p_a_point * rec(a_points + 1, b_points)) + (
            p_b_point * rec(a_points, b_points + 1)
        )

    return float(np.clip(rec(start_a, start_b), 0.0, 1.0))


def _tiebreak_winner(points_a: int, points_b: int) -> str | None:
    if points_a >= 7 and points_a - points_b >= 2:
        return SERVER_A
    if points_b >= 7 and points_b - points_a >= 2:
        return SERVER_B
    return None


def _tiebreak_server(point_index: int, first_server: str) -> str:
    first_server = _normalize_server(first_server)
    if point_index == 0:
        return first_server

    other_server = _other_server(first_server)
    return other_server if ((point_index - 1) // 2) % 2 == 0 else first_server


def _infer_tiebreak_first_server(point_index: int, current_server: str) -> str:
    current_server = _normalize_server(current_server)
    for candidate in (SERVER_A, SERVER_B):
        if _tiebreak_server(point_index, candidate) == current_server:
            return candidate
    raise ValueError("could not infer tiebreak first server from state")


def bayesian_update_serve_point_probability(
    prior_probability: float,
    service_points_won: int = 0,
    service_points_played: int = 0,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> float:
    """
    Update a pre-match serve-point prior with in-match service points observed so far.

    This is intentionally leakage-controlled: only aggregate points that have
    already happened in the current match should be supplied. Future match
    totals, full-match box-score percentages, or post-match stats must not be
    used as inputs.
    """
    prior = _clip_probability(prior_probability)
    played = int(service_points_played or 0)
    won = int(service_points_won or 0)
    strength = float(prior_strength)

    if played < 0 or won < 0:
        raise ValueError("service point counts cannot be negative")
    if won > played:
        raise ValueError("service_points_won cannot exceed service_points_played")
    if strength < 0:
        raise ValueError("prior_strength cannot be negative")
    if played == 0 and strength == 0:
        return prior

    return _clip_probability(((strength * prior) + won) / (strength + played))


def updated_serve_point_probabilities(
    p_a_serve_point_prior: float,
    p_b_serve_point_prior: float,
    a_service_points_won: int = 0,
    a_service_points_played: int = 0,
    b_service_points_won: int = 0,
    b_service_points_played: int = 0,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> tuple[float, float]:
    """Return live-updated serve-point probabilities for player A and player B."""
    return (
        bayesian_update_serve_point_probability(
            p_a_serve_point_prior,
            service_points_won=a_service_points_won,
            service_points_played=a_service_points_played,
            prior_strength=prior_strength,
        ),
        bayesian_update_serve_point_probability(
            p_b_serve_point_prior,
            service_points_won=b_service_points_won,
            service_points_played=b_service_points_played,
            prior_strength=prior_strength,
        ),
    )


def inplay_match_win_probability(
    sets_a: int,
    sets_b: int,
    games_a: int,
    games_b: int,
    points_a: Any,
    points_b: Any,
    server: str,
    p_a_serve_point: float,
    p_b_serve_point: float,
    best_of: int = 3,
    tiebreak_at_six_all: bool = True,
) -> float:
    """
    Return player A's in-play match-win probability from the current score.

    The model is a point-level Markov recursion over the live score state.
    It is intentionally separate from the pre-match feature pipeline: the only
    performance inputs are pre-match serve-point priors and, optionally, live
    service-point aggregates that have already occurred.

    Leakage guardrails:
    - Do not pass full-match box-score percentages into `p_*_serve_point`.
    - Do not pass final match service totals into the Bayesian updater.
    - The current score should be the score at prediction time only.

    This implementation assumes a standard tiebreak at 6-6, including the
    final set. Advantage sets are deliberately not modelled yet.
    """
    if not tiebreak_at_six_all:
        raise NotImplementedError("advantage sets are not implemented")

    sets_to_win = _sets_to_win(best_of)
    sets_a = int(sets_a)
    sets_b = int(sets_b)
    games_a = int(games_a)
    games_b = int(games_b)
    points_a_int = parse_tennis_point_score(points_a)
    points_b_int = parse_tennis_point_score(points_b)
    server = _normalize_server(server)
    p_a_serve = _clip_probability(p_a_serve_point)
    p_b_serve = _clip_probability(p_b_serve_point)

    if min(sets_a, sets_b, games_a, games_b, points_a_int, points_b_int) < 0:
        raise ValueError("score values cannot be negative")
    if sets_a > sets_to_win or sets_b > sets_to_win:
        raise ValueError("sets won cannot exceed the match target")
    if games_a > 7 or games_b > 7:
        raise ValueError("games in a set cannot exceed 7 in this model")

    in_tiebreak = tiebreak_at_six_all and games_a == 6 and games_b == 6
    if not in_tiebreak:
        points_a_int, points_b_int = _normalize_normal_game_points(points_a_int, points_b_int)

    tb_first_server = (
        _infer_tiebreak_first_server(points_a_int + points_b_int, server)
        if in_tiebreak
        else ""
    )

    def p_a_wins_point(point_server: str) -> float:
        return p_a_serve if point_server == SERVER_A else 1.0 - p_b_serve

    def advance_after_game(
        state_sets_a: int,
        state_sets_b: int,
        state_games_a: int,
        state_games_b: int,
        game_winner: str,
        next_server: str,
    ) -> tuple[int, int, int, int, str]:
        next_games_a = state_games_a + (1 if game_winner == SERVER_A else 0)
        next_games_b = state_games_b + (1 if game_winner == SERVER_B else 0)
        set_winner: str | None = None

        if next_games_a >= 6 and next_games_a - next_games_b >= 2:
            set_winner = SERVER_A
        elif next_games_b >= 6 and next_games_b - next_games_a >= 2:
            set_winner = SERVER_B
        elif next_games_a == 7 and next_games_b == 6:
            set_winner = SERVER_A
        elif next_games_b == 7 and next_games_a == 6:
            set_winner = SERVER_B

        if set_winner is None:
            return state_sets_a, state_sets_b, next_games_a, next_games_b, next_server

        return (
            state_sets_a + (1 if set_winner == SERVER_A else 0),
            state_sets_b + (1 if set_winner == SERVER_B else 0),
            0,
            0,
            next_server,
        )

    @lru_cache(maxsize=None)
    def rec(
        state_sets_a: int,
        state_sets_b: int,
        state_games_a: int,
        state_games_b: int,
        state_points_a: int,
        state_points_b: int,
        state_server: str,
        state_tb_first_server: str,
    ) -> float:
        if state_sets_a >= sets_to_win:
            return 1.0
        if state_sets_b >= sets_to_win:
            return 0.0

        def continue_after_game(next_state: tuple[int, int, int, int, str]) -> float:
            (
                next_sets_a,
                next_sets_b,
                next_games_a,
                next_games_b,
                next_server,
            ) = next_state
            return rec(
                next_sets_a,
                next_sets_b,
                next_games_a,
                next_games_b,
                0,
                0,
                next_server,
                "",
            )

        state_in_tiebreak = (
            tiebreak_at_six_all and state_games_a == 6 and state_games_b == 6
        )

        if state_in_tiebreak:
            first_server = state_tb_first_server or _infer_tiebreak_first_server(
                state_points_a + state_points_b,
                state_server,
            )
            tb_winner = _tiebreak_winner(state_points_a, state_points_b)
            if tb_winner is not None:
                next_state = advance_after_game(
                    state_sets_a,
                    state_sets_b,
                    state_games_a,
                    state_games_b,
                    tb_winner,
                    _other_server(first_server),
                )
                return continue_after_game(next_state)
            if state_points_a + state_points_b >= MAX_TIEBREAK_POINTS:
                a_next_state = advance_after_game(
                    state_sets_a,
                    state_sets_b,
                    state_games_a,
                    state_games_b,
                    SERVER_A,
                    _other_server(first_server),
                )
                b_next_state = advance_after_game(
                    state_sets_a,
                    state_sets_b,
                    state_games_a,
                    state_games_b,
                    SERVER_B,
                    _other_server(first_server),
                )
                return (0.5 * continue_after_game(a_next_state)) + (
                    0.5 * continue_after_game(b_next_state)
                )

            point_server = _tiebreak_server(state_points_a + state_points_b, first_server)
            p_point = p_a_wins_point(point_server)
            next_server_if_a = _tiebreak_server(
                state_points_a + state_points_b + 1,
                first_server,
            )
            return (
                p_point
                * rec(
                    state_sets_a,
                    state_sets_b,
                    state_games_a,
                    state_games_b,
                    state_points_a + 1,
                    state_points_b,
                    next_server_if_a,
                    first_server,
                )
                + (1.0 - p_point)
                * rec(
                    state_sets_a,
                    state_sets_b,
                    state_games_a,
                    state_games_b,
                    state_points_a,
                    state_points_b + 1,
                    next_server_if_a,
                    first_server,
                )
            )

        normalized_points_a, normalized_points_b = _normalize_normal_game_points(
            state_points_a,
            state_points_b,
        )
        if (normalized_points_a, normalized_points_b) != (state_points_a, state_points_b):
            return rec(
                state_sets_a,
                state_sets_b,
                state_games_a,
                state_games_b,
                normalized_points_a,
                normalized_points_b,
                state_server,
                "",
            )

        p_game_a = normal_game_win_probability_from_score(
            normalized_points_a,
            normalized_points_b,
            state_server,
            p_a_serve,
            p_b_serve,
        )
        a_next_state = advance_after_game(
            state_sets_a,
            state_sets_b,
            state_games_a,
            state_games_b,
            SERVER_A,
            _other_server(state_server),
        )
        b_next_state = advance_after_game(
            state_sets_a,
            state_sets_b,
            state_games_a,
            state_games_b,
            SERVER_B,
            _other_server(state_server),
        )
        return (
            p_game_a * continue_after_game(a_next_state)
            + (1.0 - p_game_a) * continue_after_game(b_next_state)
        )

    return float(
        np.clip(
            rec(
                sets_a,
                sets_b,
                games_a,
                games_b,
                points_a_int,
                points_b_int,
                server,
                tb_first_server,
            ),
            0.0,
            1.0,
        )
    )
