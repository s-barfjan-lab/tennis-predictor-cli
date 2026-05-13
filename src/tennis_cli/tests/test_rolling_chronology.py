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
