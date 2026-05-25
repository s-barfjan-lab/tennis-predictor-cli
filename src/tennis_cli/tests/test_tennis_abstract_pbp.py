import pandas as pd

from tennis_cli.pipelines.build_tennis_abstract_pbp import (
    build_tennis_abstract_snapshots_from_frames,
    split_point_score,
    split_point_score_player_order,
)


def test_split_point_score_handles_normal_and_advantage_scores():
    assert split_point_score("0-0") == ("0", "0")
    assert split_point_score("40-AD") == ("40", "AD")
    assert split_point_score("6-5") == ("6", "5")
    assert split_point_score(None) == ("0", "0")
    assert split_point_score_player_order("AD-40", 2) == ("40", "AD")


def test_tennis_abstract_snapshots_shift_live_service_counts():
    matches = pd.DataFrame(
        {
            "match_id": ["m1"],
            "Player 1": ["Player One"],
            "Player 2": ["Player Two"],
            "Date": [20240101],
            "Tournament": ["Example Open"],
            "Round": ["F"],
            "Surface": ["Hard"],
            "Best of": [3],
            "Final TB?": ["1"],
        }
    )
    points = pd.DataFrame(
        {
            "match_id": ["m1"] * 4,
            "Pt": [1, 2, 3, 4],
            "Set1": [0, 0, 0, 0],
            "Set2": [0, 0, 0, 0],
            "Gm1": [0, 0, 0, 0],
            "Gm2": [0, 0, 0, 0],
            "Pts": ["0-0", "15-0", "15-15", "30-15"],
            "TB?": [0, 0, 0, 0],
            "Svr": [1, 1, 1, 1],
            "PtWinner": [1, 2, 1, 1],
            "isSvrWinner": [1, 0, 1, 1],
        }
    )

    snapshots = build_tennis_abstract_snapshots_from_frames(
        points_df=points,
        matches_df=matches,
        tour="atp",
    )

    assert snapshots["p1_service_points_played_before"].tolist() == [0, 1, 2, 3]
    assert snapshots["p1_service_points_won_before"].tolist() == [0, 1, 1, 2]
    assert snapshots["p2_service_points_played_before"].tolist() == [0, 0, 0, 0]
    assert snapshots["p2_service_points_won_before"].tolist() == [0, 0, 0, 0]

    first = snapshots.iloc[0]
    assert first["point_winner_player"] == 1
    assert first["p1_service_points_played_before"] == 0
    assert first["p1_service_points_won_before"] == 0


def test_tennis_abstract_snapshots_keep_data_lane_fields_separate():
    matches = pd.DataFrame(
        {
            "match_id": ["m2"],
            "Player 1": ["Server First"],
            "Player 2": ["Returner First"],
            "Date": [20240601],
            "Tournament": ["Separate Lane Cup"],
            "Round": ["R32"],
            "Surface": ["Clay"],
            "Best of": [3],
            "Final TB?": ["1"],
        }
    )
    points = pd.DataFrame(
        {
            "match_id": ["m2"] * 4,
            "Pt": [1, 2, 3, 4],
            "Set1": [1, 1, 1, 1],
            "Set2": [0, 0, 0, 0],
            "Gm1": [5, 5, 5, 5],
            "Gm2": [0, 0, 0, 0],
            "Pts": ["0-0", "15-0", "30-0", "40-0"],
            "TB?": [0, 0, 0, 0],
            "Svr": [1, 1, 1, 1],
            "PtWinner": [1, 1, 1, 1],
            "isSvrWinner": [1, 1, 1, 1],
        }
    )

    snapshots = build_tennis_abstract_snapshots_from_frames(
        points_df=points,
        matches_df=matches,
        tour="atp",
    )

    assert snapshots["tour"].unique().tolist() == ["atp"]
    assert snapshots["match_date"].dt.year.unique().tolist() == [2024]
    assert snapshots["server"].tolist() == ["A", "A", "A", "A"]
    assert snapshots["label_player1_win_match"].dropna().unique().tolist() == [1]
    assert "delta_elo" not in snapshots.columns
    assert "label_player_a_win" not in snapshots.columns
