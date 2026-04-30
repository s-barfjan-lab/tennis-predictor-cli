from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_mean_past(series: pd.Series, window: int) -> pd.Series:
    """
    Mean over the previous `window` matches only.
    The current match is excluded via shift(1).
    """
    return series.shift(1).rolling(window=window, min_periods=1).mean()


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

    required_cols = ["player_id", "tourney_date", "match_id", "surface"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for rolling features: {missing}")

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.sort_values(["player_id", "tourney_date", "match_id"]).reset_index(drop=True)

    grouped = df.groupby("player_id", group_keys=False)

    # Same-surface grouping for surface-specific recent-form features
    df["_surface_group"] = df["surface"].astype(str).str.upper().str.strip()
    grouped_surface = df.groupby(["player_id", "_surface_group"], group_keys=False)

    # Cumulative match count before current match
    df["matches_played"] = grouped.cumcount()

    # Existing rolling features
    df["win_rate_last10"] = grouped["label_win"].transform(lambda s: _rolling_mean_past(s, window))
    df["aces_avg_last10"] = grouped["aces"].transform(lambda s: _rolling_mean_past(s, window))
    df["serve_win_pct_last10"] = grouped["service_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))

    # New rolling return-side features
    df["return_win_pct_last10"] = grouped["return_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["bp_conversion_last10"] = grouped["bp_conversion_pct"].transform(lambda s: _rolling_mean_past(s, window))

    # Same-surface rolling features (same feature family, but only on the current surface)
    df["serve_win_pct_last10_surface"] = grouped_surface["service_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["return_win_pct_last10_surface"] = grouped_surface["return_points_won_pct"].transform(lambda s: _rolling_mean_past(s, window))
    df["bp_conversion_last10_surface"] = grouped_surface["bp_conversion_pct"].transform(lambda s: _rolling_mean_past(s, window))

    # Time/fatigue features
    df["days_since_last_match"] = grouped["tourney_date"].transform(_days_since_last_match)
    df["matches_last_30_days"] = grouped["tourney_date"].transform(lambda s: _matches_in_last_n_days_from_dates(s, days=30))

    df["days_since_last_match_surface"] = grouped_surface["tourney_date"].transform(_days_since_last_match)
    df["matches_last_30_days_surface"] = grouped_surface["tourney_date"].transform(lambda s: _matches_in_last_n_days_from_dates(s, days=30))

    return df.drop(columns=["_surface_group"])