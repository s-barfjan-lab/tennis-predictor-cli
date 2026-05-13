from __future__ import annotations

import pandas as pd


def compute_h2h_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add head-to-head features from the player perspective.

    Output columns:
    - h2h_wins
    - h2h_losses
    - h2h_win_ratio

    Important:
    current match must NOT leak into its own H2H values.
    So we read the prior H2H for both rows of a match first,
    then update the tracker only after that.
    """
    required_cols = ["match_id", "tourney_date", "player_id", "opponent_id", "label_win", ]
    missing = [c for c in required_cols if c not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for H2H features: {missing}")

    df = long_df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    sort_keys = ["tourney_date"]
    if "tourney_id" in df.columns:
        sort_keys.append("tourney_id")
    if "match_num" in df.columns:
        df = df.copy()
        df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce")
        sort_keys.append("match_num")
    else:
        sort_keys.append("match_id")
    sort_keys.append("player_id")
    df = df.sort_values(sort_keys).reset_index(drop=True)

    h2h_tracker: dict[tuple[str, str], int] = {}

    h2h_wins_before = [0] * len(df)
    h2h_losses_before = [0] * len(df)

    for _, match_idx in df.groupby("match_id", sort=False).groups.items():
        idx_list = list(match_idx)

        # 1) read prior H2H state for both rows first
        for i in idx_list:
            player_key = str(df.at[i, "player_id"])
            opponent_key = str(df.at[i, "opponent_id"])

            wins_before = h2h_tracker.get((player_key, opponent_key), 0)
            losses_before = h2h_tracker.get((opponent_key, player_key), 0)

            h2h_wins_before[i] = wins_before
            h2h_losses_before[i] = losses_before

        # 2) only after reading prior state, update tracker using the winner row
        for i in idx_list:
            if int(df.at[i, "label_win"]) == 1:
                winner_key = str(df.at[i, "player_id"])
                loser_key = str(df.at[i, "opponent_id"])
                h2h_tracker[(winner_key, loser_key)] = (h2h_tracker.get((winner_key, loser_key), 0) + 1)
                break

    df["h2h_wins"] = h2h_wins_before
    df["h2h_wins"] = pd.to_numeric(df["h2h_wins"], errors="coerce").fillna(0).astype(int)
    df["h2h_losses"] = h2h_losses_before
    df["h2h_losses"] = pd.to_numeric(df["h2h_losses"], errors="coerce").fillna(0).astype(int)

    total = df["h2h_wins"] + df["h2h_losses"]
    df["h2h_win_ratio"] = df["h2h_wins"] / total.replace(0, pd.NA)
    df["h2h_win_ratio"] = pd.to_numeric(df["h2h_win_ratio"], errors="coerce").fillna(0.5)

    return df
