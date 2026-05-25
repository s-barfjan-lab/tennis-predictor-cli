from __future__ import annotations

from tennis_cli.features.inplay_markov import (
    DEFAULT_PRIOR_STRENGTH,
    inplay_match_win_probability,
    updated_serve_point_probabilities,
)


def predict_inplay_for_match(
    sets_a: int,
    sets_b: int,
    games_a: int,
    games_b: int,
    points_a: str,
    points_b: str,
    server: str,
    p_a_serve_point_prior: float = 0.5,
    p_b_serve_point_prior: float = 0.5,
    best_of: int = 3,
    a_service_points_won: int = 0,
    a_service_points_played: int = 0,
    b_service_points_won: int = 0,
    b_service_points_played: int = 0,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict:
    """
    Predict live match probability from score state and observed service points.

    The live service counts must be counts available at prediction time, not
    full-match post-hoc box-score totals.
    """
    p_a_live, p_b_live = updated_serve_point_probabilities(
        p_a_serve_point_prior=p_a_serve_point_prior,
        p_b_serve_point_prior=p_b_serve_point_prior,
        a_service_points_won=a_service_points_won,
        a_service_points_played=a_service_points_played,
        b_service_points_won=b_service_points_won,
        b_service_points_played=b_service_points_played,
        prior_strength=prior_strength,
    )

    prob_a = inplay_match_win_probability(
        sets_a=sets_a,
        sets_b=sets_b,
        games_a=games_a,
        games_b=games_b,
        points_a=points_a,
        points_b=points_b,
        server=server,
        p_a_serve_point=p_a_live,
        p_b_serve_point=p_b_live,
        best_of=best_of,
    )

    return {
        "sets_a": int(sets_a),
        "sets_b": int(sets_b),
        "games_a": int(games_a),
        "games_b": int(games_b),
        "points_a": str(points_a),
        "points_b": str(points_b),
        "server": str(server).strip().upper(),
        "best_of": int(best_of),
        "p_a_serve_point_prior": float(p_a_serve_point_prior),
        "p_b_serve_point_prior": float(p_b_serve_point_prior),
        "p_a_serve_point_live": p_a_live,
        "p_b_serve_point_live": p_b_live,
        "a_service_points_won": int(a_service_points_won),
        "a_service_points_played": int(a_service_points_played),
        "b_service_points_won": int(b_service_points_won),
        "b_service_points_played": int(b_service_points_played),
        "prior_strength": float(prior_strength),
        "tiebreak_at_six_all": True,
        "prob_player_a_win": prob_a,
        "prob_player_b_win": 1.0 - prob_a,
    }
