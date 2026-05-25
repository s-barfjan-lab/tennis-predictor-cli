import pandas as pd
import pytest

from tennis_cli.features.rolling import add_rolling_features


def _make_slam_path():
    """One player who advances through all rounds of a 128-draw Slam.
    All matches share the same tourney_date (Sackmann's Monday convention).
    match_num values for round positions: R128=5, R64=68, R32=99, R16=114, QF=121, SF=125, F=127.
    """
    rows = []
    schedule = [
        (5, "R128", 1),
        (68, "R64", 1),
        (99, "R32", 1),
        (114, "R16", 1),
        (121, "QF", 1),
        (125, "SF", 1),
        (127, "F", 0),
    ]
    for mn, rd, lw in schedule:
        rows.append(
            {
                "match_id": f"2024-580__{mn}",
                "tourney_id": "2024-580",
                "match_num": mn,
                "tour": "atp",
                "tourney_date": "2024-01-15",
                "player_id": "A",
                "opponent_id": f"X{mn}",
                "opponent_hand": "R",
                "surface": "HARD",
                "round": rd,
                "label_win": lw,
            }
        )
    df = pd.DataFrame(rows)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    stat_cols = [
        "aces",
        "aces_per_service_point",
        "df_per_service_point",
        "first_serve_in_pct",
        "ace_vs_df",
        "second_serve_won_per_service_game",
        "service_points_won_pct",
        "return_points_won_pct",
        "bp_conversion_pct",
        "bp_saved_pct",
    ]
    for col in stat_cols:
        df[col] = 0.5
    return df


def test_rolling_features_are_chronologically_ordered_in_a_slam_path():
    df = _make_slam_path()
    out = add_rolling_features(df)
    out = out.sort_values(["player_id", "tourney_date", "match_num"]).reset_index(drop=True)

    assert list(out["round"]) == ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    assert list(out["matches_played"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(out["current_win_streak"]) == [0, 1, 2, 3, 4, 5, 6]
    assert pd.isna(out["previous_match_win"].iloc[0])
    assert list(out["previous_match_win"].iloc[1:]) == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert list(out["has_surface_history"]) == [0, 0, 0, 0, 0, 1, 1]
    assert out["serve_win_pct_last10_surface"].iloc[:5].isna().all()
    assert list(out["serve_win_pct_last10_surface"].iloc[5:]) == pytest.approx([0.5, 0.5])


def test_markov_rolling_histories_use_prior_matches_only():
    rows = []
    for i in range(11):
        rows.append(
            {
                "match_id": f"m{i}",
                "tourney_id": "t",
                "match_num": i,
                "tour": "atp",
                "tourney_date": f"2024-01-{i + 1:02d}",
                "player_id": "A",
                "opponent_id": f"X{i}",
                "opponent_hand": "R",
                "surface": "HARD",
                "round": "R32",
                "label_win": 1,
                "service_points_won_pct": 0.50 + (i * 0.01),
                "return_points_won_pct": 0.40 + (i * 0.01),
            }
        )
    df = pd.DataFrame(rows)
    for col in [
        "aces",
        "aces_per_service_point",
        "df_per_service_point",
        "first_serve_in_pct",
        "ace_vs_df",
        "second_serve_won_per_service_game",
        "bp_conversion_pct",
        "bp_saved_pct",
    ]:
        df[col] = 0.5

    out = add_rolling_features(df).sort_values(["player_id", "tourney_date"]).reset_index(drop=True)

    assert out["service_points_won_pct_30"].iloc[:10].isna().all()
    assert out["return_points_won_pct_30"].iloc[:10].isna().all()
    assert list(out["has_serve_history"].iloc[:10]) == [0] * 10
    assert list(out["has_return_history"].iloc[:10]) == [0] * 10
    assert out["has_serve_history"].iloc[10] == 1
    assert out["has_return_history"].iloc[10] == 1
    assert out["service_points_won_pct_30"].iloc[10] == pytest.approx(
        df["service_points_won_pct"].iloc[:10].mean()
    )
    assert out["return_points_won_pct_30"].iloc[10] == pytest.approx(
        df["return_points_won_pct"].iloc[:10].mean()
    )
