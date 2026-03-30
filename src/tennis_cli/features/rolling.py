from __future__ import annotations

import pandas as pd
from rich.console import Console

console = Console()


def _rolling_mean_past(series: pd.Series, window: int) -> pd.Series:
    """
    For one player's timeline:
    - shift by 1 so the current match is excluded
    - compute rolling mean over the previous `window` matches
    """
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def _matches_in_last_n_days(group: pd.DataFrame, days: int = 30) -> pd.Series:
    """
    Count how many previous matches this player played in the last `days` days.
    This uses only past matches (not the current one).
    """
    dates = pd.to_datetime(group["tourney_date"], errors="coerce")
    out = []

    for i, current_date in enumerate(dates):
        if pd.isna(current_date):
            out.append(pd.NA)
            continue

        past_dates = dates.iloc[:i]
        count = ((current_date - past_dates).dt.days <= days).sum()
        out.append(count)

    return pd.Series(out, index=group.index, dtype="float")


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling player features using ONLY past matches.
    """
    df = df.copy()

    # Ensure date is datetime and rows are ordered correctly
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.sort_values(["player_id", "tourney_date", "match_id"]).reset_index(drop=True)

    # -------------------------
    # BASIC FORM FEATURES
    # -------------------------
    df["matches_played"] = df.groupby("player_id").cumcount()

    df["win_rate_last10"] = (
        df.groupby("player_id", group_keys=False)["label_win"]
        .apply(lambda s: _rolling_mean_past(s, window))
    )

    # -------------------------
    # SERVE FEATURES
    # -------------------------
    df["aces_avg_last10"] = (
        df.groupby("player_id", group_keys=False)["aces"]
        .apply(lambda s: _rolling_mean_past(s, window))
    )

    df["serve_win_pct_last10"] = (
        df.groupby("player_id", group_keys=False)["service_points_won_pct"]
        .apply(lambda s: _rolling_mean_past(s, window))
    )

    # -------------------------
    # FATIGUE FEATURES
    # -------------------------
    df["days_since_last_match"] = (
        df.groupby("player_id")["tourney_date"]
        .diff()
        .dt.days
    )

    df["matches_last_30_days"] = (
        df.groupby("player_id", group_keys=False)
        .apply(lambda g: _matches_in_last_n_days(g, days=30))
        .reset_index(level=0, drop=True)
    )

    return df