import pandas as pd
import pytest

from tennis_cli.features.elo import compute_elo_features


def test_compute_elo_features_uses_pre_match_ratings_chronologically():
    df = pd.DataFrame({
        "tourney_date": ["2025-01-01", "2025-01-05", "2025-01-10"],
        "winner_id": [1, 1, 2],
        "loser_id": [2, 3, 1],
        "winner_name": ["Player A", "Player A", "Player B"],
        "loser_name": ["Player B", "Player C", "Player A"],
    })

    out = compute_elo_features(df)

    assert list(out["winner_elo_pre"]) == pytest.approx([1500.0, 1512.0, 1488.0])
    assert list(out["loser_elo_pre"]) == pytest.approx([1500.0, 1500.0, 1523.585699])
    assert out["winner_elo_post"].iloc[0] > out["winner_elo_pre"].iloc[0]
    assert out["loser_elo_post"].iloc[0] < out["loser_elo_pre"].iloc[0]
    assert out["tourney_date"].is_monotonic_increasing
