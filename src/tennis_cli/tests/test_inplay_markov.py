import pytest

from tennis_cli.features.inplay_markov import (
    bayesian_update_serve_point_probability,
    inplay_match_win_probability,
    parse_tennis_point_score,
    updated_serve_point_probabilities,
)
from tennis_cli.pipelines.predict_inplay import predict_inplay_for_match


def test_parse_tennis_point_score_supports_normal_and_tiebreak_scores():
    assert parse_tennis_point_score("0") == 0
    assert parse_tennis_point_score("15") == 1
    assert parse_tennis_point_score("30") == 2
    assert parse_tennis_point_score("40") == 3
    assert parse_tennis_point_score("AD") == 4
    assert parse_tennis_point_score("8") == 8


def test_terminal_match_states_are_exact():
    assert inplay_match_win_probability(
        sets_a=2,
        sets_b=0,
        games_a=0,
        games_b=0,
        points_a=0,
        points_b=0,
        server="A",
        p_a_serve_point=0.5,
        p_b_serve_point=0.5,
        best_of=3,
    ) == pytest.approx(1.0)
    assert inplay_match_win_probability(
        sets_a=0,
        sets_b=2,
        games_a=0,
        games_b=0,
        points_a=0,
        points_b=0,
        server="A",
        p_a_serve_point=0.5,
        p_b_serve_point=0.5,
        best_of=3,
    ) == pytest.approx(0.0)


def test_neutral_starting_state_is_fifty_fifty():
    prob = inplay_match_win_probability(
        sets_a=0,
        sets_b=0,
        games_a=0,
        games_b=0,
        points_a=0,
        points_b=0,
        server="A",
        p_a_serve_point=0.5,
        p_b_serve_point=0.5,
        best_of=3,
    )

    assert prob == pytest.approx(0.5)


def test_score_state_moves_probability_in_expected_direction():
    ahead = inplay_match_win_probability(
        sets_a=0,
        sets_b=0,
        games_a=3,
        games_b=2,
        points_a="40",
        points_b="0",
        server="A",
        p_a_serve_point=0.62,
        p_b_serve_point=0.58,
        best_of=3,
    )
    behind = inplay_match_win_probability(
        sets_a=0,
        sets_b=0,
        games_a=3,
        games_b=2,
        points_a="0",
        points_b="40",
        server="A",
        p_a_serve_point=0.62,
        p_b_serve_point=0.58,
        best_of=3,
    )

    assert 0.0 <= behind <= 1.0
    assert 0.0 <= ahead <= 1.0
    assert ahead > behind


def test_tiebreak_state_is_bounded():
    prob = inplay_match_win_probability(
        sets_a=0,
        sets_b=0,
        games_a=6,
        games_b=6,
        points_a=4,
        points_b=3,
        server="B",
        p_a_serve_point=0.60,
        p_b_serve_point=0.57,
        best_of=3,
    )

    assert 0.0 <= prob <= 1.0


def test_bayesian_live_update_uses_only_supplied_observed_points():
    updated = bayesian_update_serve_point_probability(
        prior_probability=0.60,
        service_points_won=7,
        service_points_played=10,
        prior_strength=40,
    )

    assert updated == pytest.approx(((40 * 0.60) + 7) / 50)


def test_updated_serve_point_probabilities_validate_counts():
    with pytest.raises(ValueError):
        updated_serve_point_probabilities(
            p_a_serve_point_prior=0.60,
            p_b_serve_point_prior=0.58,
            a_service_points_won=11,
            a_service_points_played=10,
        )


def test_pipeline_outputs_complementary_probabilities():
    result = predict_inplay_for_match(
        sets_a=1,
        sets_b=1,
        games_a=4,
        games_b=4,
        points_a="30",
        points_b="15",
        server="B",
        p_a_serve_point_prior=0.61,
        p_b_serve_point_prior=0.59,
        a_service_points_won=32,
        a_service_points_played=50,
        b_service_points_won=29,
        b_service_points_played=48,
        prior_strength=48,
    )

    assert 0.0 <= result["prob_player_a_win"] <= 1.0
    assert result["prob_player_b_win"] == pytest.approx(
        1.0 - result["prob_player_a_win"]
    )
