import pandas as pd
import pytest

from tennis_cli.features.markov import ( add_markov_match_features, 
    game_win_prob_from_point_prob, markov_match_win_probability, )


def test_game_win_prob_from_point_prob_uses_correct_omalley_formula():
    assert game_win_prob_from_point_prob(0.5) == pytest.approx(0.5)
    assert game_win_prob_from_point_prob(0.6) == pytest.approx(0.7357, abs=1e-4)
    assert game_win_prob_from_point_prob(0.7) == pytest.approx(0.9008, abs=1e-4)


def test_markov_match_probability_is_bounded():
    for p_a, p_b in [(0.5, 0.5), (0.6, 0.55), (0.7, 0.6), (0.4, 0.7)]:
        prob = markov_match_win_probability(p_a, p_b, best_of=3)
        assert 0.0 <= prob <= 1.0


def test_add_markov_match_features_outputs_probability_identities():
    df = pd.DataFrame({
        "best_of": [3, 5],
        "service_points_won_pct_30_a": [0.62, pd.NA],
        "return_points_won_pct_30_a": [0.39, 0.50],
        "service_points_won_pct_30_b": [0.58, 0.50],
        "return_points_won_pct_30_b": [0.36, 0.50],
        "has_serve_history_a": [1, 0],
        "has_return_history_a": [1, 1],
        "has_serve_history_b": [1, 1],
        "has_return_history_b": [1, 1],
    })

    out = add_markov_match_features(df)

    assert out["markov_match_win_prob_a"].between(0.0, 1.0).all()
    assert out["markov_match_win_prob_b"].between(0.0, 1.0).all()
    assert out["markov_match_win_prob_b"].tolist() == pytest.approx(
        (1.0 - out["markov_match_win_prob_a"]).tolist()
    )
    assert out["delta_markov_match_win"].tolist() == pytest.approx(
        (out["markov_match_win_prob_a"] - out["markov_match_win_prob_b"]).tolist()
    )
    assert out["has_serve_history"].tolist() == [1, 0]
    assert out["has_return_history"].tolist() == [1, 1]
