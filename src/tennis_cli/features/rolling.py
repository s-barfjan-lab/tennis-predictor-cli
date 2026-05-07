from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_mean_past(series: pd.Series, window: int) -> pd.Series:
    """
    Mean over the previous `window` matches only.
    The current match is excluded via shift(1).
    """
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _expanding_mean_past(series: pd.Series) -> pd.Series:
    """
    Mean over all previous rows only.
    The current match is excluded via shift(1).
    """
    return series.shift(1).expanding(min_periods=1).mean()


def _days_since_last_match(dates: pd.Series) -> pd.Series:
    """
    Days since the previous match for each player.
    """
    dates = pd.to_datetime(dates, errors="coerce")
    return dates.diff().dt.days


def _matches_in_last_n_days_from_dates(dates: pd.Series, days: int = 30) -> pd.Series:
    """
    Count how many PRIOR matches this player played in the last `days` days.

    For each row i:
    count matches j < i such that
    dates[i] - days <= dates[j] < dates[i]
    """
    dt = pd.to_datetime(dates, errors="coerce")
    out = pd.Series(index=dates.index, dtype="float64")

    valid_mask = dt.notna()
    if not valid_mask.any():
        out.loc[:] = pd.NA
        return out

    valid_dates = dt[valid_mask]
    date_values = valid_dates.to_numpy(dtype="datetime64[ns]")

    counts = np.zeros(len(valid_dates), dtype=float)

    for i in range(len(date_values)):
        current_date = date_values[i]
        left_bound = current_date - np.timedelta64(days, "D")

        left_idx = np.searchsorted(date_values, left_bound, side="left")
        right_idx = np.searchsorted(date_values, current_date, side="left")

        counts[i] = max(0, right_idx - left_idx)

    out.loc[valid_dates.index] = counts
    out.loc[~valid_mask] = pd.NA
    return out


def _win_pct_in_last_n_days(labels: pd.Series, dates: pd.Series, days: int = 365) -> pd.Series:
    """
    Win percentage over prior matches in the last `days` days.
    """
    dt = pd.to_datetime(dates, errors="coerce")
    label_num = pd.to_numeric(labels, errors="coerce")
    out = pd.Series(index=labels.index, dtype="float64")

    valid_mask = dt.notna()
    if not valid_mask.any():
        out.loc[:] = pd.NA
        return out

    valid_dates = dt[valid_mask]
    valid_labels = label_num[valid_mask]
    date_values = valid_dates.to_numpy(dtype="datetime64[ns]")

    values = []
    for i in range(len(valid_dates)):
        current_date = date_values[i]
        left_bound = current_date - np.timedelta64(days, "D")
        prior_mask = (date_values >= left_bound) & (date_values < current_date)
        prior_labels = valid_labels.iloc[prior_mask].dropna()
        values.append(float(prior_labels.mean()) if not prior_labels.empty else np.nan)

    out.loc[valid_dates.index] = values
    out.loc[~valid_mask] = pd.NA
    return out


def _current_win_streak_past(labels: pd.Series) -> pd.Series:
    """
    Consecutive wins entering each row, excluding the current match.
    """
    out = []
    streak = 0
    for label in pd.to_numeric(labels, errors="coerce"):
        out.append(streak)
        if pd.isna(label):
            streak = 0
        elif int(label) == 1:
            streak += 1
        else:
            streak = 0
    return pd.Series(out, index=labels.index, dtype="float64")


def add_rolling_features(long_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Add player-history rolling features to the long-view table.

    Features added:
    - matches_played
    - win_rate_last10
    - aces_avg_last10
    - serve_win_pct_last10
    - return_win_pct_last10
    - bp_conversion_last10
    - days_since_last_match
    - matches_last_30_days
    """
    df = long_df.copy()

    required_cols = ["player_id", "tourney_date", "match_id", "surface", "round", "opponent_hand"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for rolling features: {missing}")

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.sort_values(["player_id", "tourney_date", "match_id"]).reset_index(drop=True)

    grouped = df.groupby("player_id", group_keys=False)

    # Same-surface grouping for surface-specific recent-form features
    df["_surface_group"] = df["surface"].astype(str).str.upper().str.strip()
    grouped_surface = df.groupby(["player_id", "_surface_group"], group_keys=False)

    # Same-opponent-hand grouping for handedness matchup history.
    df["_opponent_hand_group"] = df["opponent_hand"].fillna("U").astype(str).str.upper().str.strip()
    grouped_opponent_hand = df.groupby(["player_id", "_opponent_hand_group"], group_keys=False)

    df["_round_group"] = df["round"].fillna("UNK").astype(str).str.upper().str.strip()
    grouped_round = df.groupby(["player_id", "_round_group"], group_keys=False)

    # Cumulative match count before current match
    df["matches_played"] = grouped.cumcount()

    # Existing rolling features
    df["win_rate_last10"] = grouped["label_win"].transform(lambda s: _rolling_mean_past(s, window))
    df["win_pct_last_365_days"] = pd.Series(index=df.index, dtype="float64")
    for _, group_idx in grouped.groups.items():
        idx = list(group_idx)
        df.loc[idx, "win_pct_last_365_days"] = _win_pct_in_last_n_days(
            df.loc[idx, "label_win"],
            df.loc[idx, "tourney_date"],
            days=365,
        )
    df["previous_match_win"] = grouped["label_win"].shift(1)
    df["current_win_streak"] = grouped["label_win"].transform(_current_win_streak_past)
    df["aces_avg_last10"] = grouped["aces"].transform(lambda s: _rolling_mean_past(s, window))
    df["ace_pct_last10"] = grouped["aces_per_service_point"].transform(lambda s: _rolling_mean_past(s, window))
    df["df_pct_last10"] = grouped["df_per_service_point"].transform(lambda s: _rolling_mean_past(s, window))
    df["first_serve_in_pct_last10"] = grouped["first_serve_in_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["ace_vs_df_last10"] = grouped["ace_vs_df"].transform(lambda s: _rolling_mean_past(s, window))
    df["second_serve_won_per_service_game_last10"] = grouped["second_serve_won_per_service_game"].transform(
        lambda s: _rolling_mean_past(s, window)
    )
    df["serve_win_pct_last10"] = grouped["service_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))

    # New rolling return-side features
    df["return_win_pct_last10"] = grouped["return_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["bp_conversion_last10"] = grouped["bp_conversion_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["bp_saved_pct_last10"] = grouped["bp_saved_pct"].transform(lambda s: _rolling_mean_past(s, window))

    # Same-surface rolling features (same feature family, but only on the current surface)
    df["surface_win_pct_last10"] = grouped_surface["label_win"].transform(lambda s: _rolling_mean_past(s, window))
    df["serve_win_pct_last10_surface"] = grouped_surface["service_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["return_win_pct_last10_surface"] = grouped_surface["return_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["bp_conversion_last10_surface"] = grouped_surface["bp_conversion_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["bp_saved_pct_last10_surface"] = grouped_surface["bp_saved_pct"].transform(lambda s: _rolling_mean_past(s, window))

    # Win percentage against the same opponent handedness as this row's opponent.
    df["hand_win_pct_last10"] = grouped_opponent_hand["label_win"].transform(lambda s: _rolling_mean_past(s, window))

    # Historical win percentage in the same round type.
    df["round_win_pct"] = grouped_round["label_win"].transform(_expanding_mean_past)

    # Time/fatigue features
    df["days_since_last_match"] = grouped["tourney_date"].transform(_days_since_last_match)
    df["matches_last_7_days"] = grouped["tourney_date"].transform(lambda s: _matches_in_last_n_days_from_dates(s, days=7))
    df["matches_last_30_days"] = grouped["tourney_date"].transform(lambda s: _matches_in_last_n_days_from_dates(s, days=30))
    df["matches_last_365_days"] = grouped["tourney_date"].transform(lambda s: _matches_in_last_n_days_from_dates(s, days=365))

    df["days_since_last_match_surface"] = grouped_surface["tourney_date"].transform(_days_since_last_match)
    df["matches_last_30_days_surface"] = grouped_surface["tourney_date"].transform(lambda s: _matches_in_last_n_days_from_dates(s, days=30))

    return df.drop(columns=["_surface_group", "_opponent_hand_group", "_round_group"])
