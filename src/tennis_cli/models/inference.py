from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
import json

from tennis_cli.features.markov import add_markov_match_features
from tennis_cli.models.dataset import get_feature_columns
from tennis_cli.models.io import load_model_artifact
from tennis_cli.models.xgboost_model import (apply_calibration, predict_proba_from_xgb_artifact, )


DEFAULT_ELO = 1500.0
PEAK_TENNIS_AGE = 30.0
VALID_MODEL_SURFACES = {"HARD", "CLAY", "GRASS"}

ROUND_MAP = {
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}


@dataclass
class ResolvedPlayer:
    player_id: object
    player_name: str


def _normalize_name(name: str) -> str:
    return str(name).strip().casefold()


def _player_sort_key(player_id: object, player_name: str) -> str:
    pid = "NA" if pd.isna(player_id) else str(player_id)
    pname = "NA" if pd.isna(player_name) else str(player_name)
    return f"{pid}__{pname}"



def _normalize_surface(surface: str | None) -> str | None:
    if surface is None:
        return None

    value = str(surface).strip().title()
    if not value:
        return None

    if value not in {"Hard", "Clay", "Grass"}:
        raise ValueError("surface must be one of: Hard, Clay, Grass")

    return value


#this was added later to clead inference dataframe from CARPET and NONE
def _drop_invalid_surfaces_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the core modeling surfaces in long-view data:
    HARD, CLAY, GRASS.
    """
    if "surface" not in df.columns:
        raise ValueError("Long-view dataset does not contain a 'surface' column.")

    out = df.copy()
    surface_norm = out["surface"].astype(str).str.upper().str.strip()
    out = out[surface_norm.isin(VALID_MODEL_SURFACES)].copy()

    if out.empty:
        raise ValueError("All long-view rows were removed after invalid-surface filtering.")

    return out




def _build_artifact_suffix(source: str, surface: str | None) -> str:
    """
    Build the same source/surface suffix used during training.

    Examples
    --------
    sackmann + None   -> ""
    tml + None        -> "_tml"
    sackmann + Clay   -> "_clay"
    tml + Grass       -> "_tml_grass"
    """
    parts = []

    if source != "sackmann":
        parts.append(source)

    if surface is not None:
        parts.append(surface.lower())

    if not parts:
        return ""

    return "_" + "_".join(parts)



def _round_to_ordinal(round_name: str | None):
    if round_name is None:
        return pd.NA
    value = str(round_name).strip().upper()
    if not value:
        return pd.NA
    return ROUND_MAP.get(value, pd.NA)


def _is_round_robin(round_name: str | None):
    if round_name is None:
        return pd.NA
    return int(str(round_name).strip().upper() == "RR")


def _normalize_round_name(round_name: str | None) -> str:
    if round_name is None:
        return "UNK"
    value = str(round_name).strip().upper()
    return value if value else "UNK"


def _normalize_tourney_level(tourney_level: str | None) -> str:
    if tourney_level is None:
        return "U"
    value = str(tourney_level).strip().upper()
    return value if value else "U"


def _age_peak_closeness(age):
    age_num = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    if pd.isna(age_num):
        return pd.NA
    return -abs(float(age_num) - PEAK_TENNIS_AGE)


def _age_peak_distance_squared(age):
    age_num = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    if pd.isna(age_num):
        return pd.NA
    return (float(age_num) - PEAK_TENNIS_AGE) ** 2


def _normalize_hand(hand) -> str:
    if pd.isna(hand):
        return "U"
    value = str(hand).strip().upper()
    return value if value else "U"


def _load_long_view(project_root: Path, tour: str, source: str = "sackmann") -> pd.DataFrame:
    suffix = "" if source == "sackmann" else f"_{source}"
    path = project_root / "data" / "features" / f"{tour}_long{suffix}_2015_2025.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Long-view feature file not found: {path}")

    df = pd.read_parquet(path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    if df["tourney_date"].isna().any():
        raise ValueError("Long-view dataset contains invalid 'tourney_date' values")

    df = _drop_invalid_surfaces_long(df)

    df = df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)
    return df


def _resolve_player(long_df: pd.DataFrame, player_name: str, as_of_date: pd.Timestamp) -> ResolvedPlayer:
    normalized = _normalize_name(player_name)

    exact = long_df[long_df["player_name"].astype(str).str.casefold() == normalized].copy()

    if exact.empty:
        unique_names = sorted(long_df["player_name"].dropna().astype(str).unique())
        suggestions = get_close_matches(player_name, unique_names, n=5, cutoff=0.6)
        suggestion_text = ", ".join(suggestions) if suggestions else "no close matches found"
        raise ValueError(
            f"Player '{player_name}' not found in {long_df['tour'].iloc[0].upper()} long-view data. "
            f"Suggestions: {suggestion_text}"
        )

    prior = exact[exact["tourney_date"] < as_of_date].copy()
    choice_pool = prior if not prior.empty else exact

    latest = choice_pool.sort_values(["tourney_date", "match_id"]).iloc[-1]
    return ResolvedPlayer(
        player_id=latest["player_id"],
        player_name=str(latest["player_name"]),
    )


def _get_latest_surface_elo_before_date(
    long_df: pd.DataFrame,
    player_id: object,
    as_of_date: pd.Timestamp,
    surface: str | None,
):
    if surface is None:
        return pd.NA

    hist = long_df[
        (long_df["player_id"] == player_id) &
        (long_df["tourney_date"] < as_of_date)
    ].copy()

    if hist.empty:
        return pd.NA

    if "surface_normalized" in hist.columns:
        hist_surface = hist[hist["surface_normalized"].astype(str) == surface].copy()
    else:
        hist_surface = hist[hist["surface"].astype(str).str.title() == surface].copy()

    if hist_surface.empty:
        return pd.NA

    hist_surface = hist_surface.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)
    latest = hist_surface.iloc[-1]

    current_surface_elo = latest.get("player_surface_elo_post", pd.NA)
    if pd.isna(current_surface_elo):
        current_surface_elo = latest.get("player_surface_elo_pre", pd.NA)
    if pd.isna(current_surface_elo):
        current_surface_elo = DEFAULT_ELO

    return current_surface_elo


def _compute_h2h_summary_before_date(
    long_df: pd.DataFrame,
    player_a_id: object,
    player_b_id: object,
    as_of_date: pd.Timestamp,
) -> dict:
    hist_a = long_df[
        (long_df["player_id"] == player_a_id) &
        (long_df["opponent_id"] == player_b_id) &
        (long_df["tourney_date"] < as_of_date)
    ].copy()

    hist_b = long_df[
        (long_df["player_id"] == player_b_id) &
        (long_df["opponent_id"] == player_a_id) &
        (long_df["tourney_date"] < as_of_date)
    ].copy()

    wins_a = int(hist_a["label_win"].sum()) if not hist_a.empty else 0
    wins_b = int(hist_b["label_win"].sum()) if not hist_b.empty else 0
    total = wins_a + wins_b

    if total == 0:
        ratio_a = 0.5
        ratio_b = 0.5
    else:
        ratio_a = wins_a / total
        ratio_b = wins_b / total

    return {
        "delta_h2h": ratio_a - ratio_b,
        "delta_h2h_wins": wins_a - wins_b,
        "delta_h2h_losses": wins_b - wins_a,
        "h2h_matches_total": total,
        "first_meeting": int(total == 0),
    }


def _compute_common_opponent_summary_before_date(
    long_df: pd.DataFrame,
    player_a_id: object,
    player_b_id: object,
    as_of_date: pd.Timestamp,
) -> dict:
    hist = long_df[long_df["tourney_date"] < as_of_date].copy()
    if hist.empty or "opponent_id" not in hist.columns:
        return {
            "common_opp_count": 0.0,
            "delta_common_opp_win_pct": 0.0,
            "delta_common_opp_matches": 0.0,
        }

    hist_a = hist[hist["player_id"] == player_a_id].copy()
    hist_b = hist[hist["player_id"] == player_b_id].copy()

    if hist_a.empty or hist_b.empty:
        return {
            "common_opp_count": 0.0,
            "delta_common_opp_win_pct": 0.0,
            "delta_common_opp_matches": 0.0,
        }

    opp_a = set(hist_a["opponent_id"].dropna().astype(str))
    opp_b = set(hist_b["opponent_id"].dropna().astype(str))
    common_opponents = opp_a.intersection(opp_b)

    if not common_opponents:
        return {
            "common_opp_count": 0.0,
            "delta_common_opp_win_pct": 0.0,
            "delta_common_opp_matches": 0.0,
        }

    common_a = hist_a[hist_a["opponent_id"].astype(str).isin(common_opponents)].copy()
    common_b = hist_b[hist_b["opponent_id"].astype(str).isin(common_opponents)].copy()
    win_pct_a = pd.to_numeric(common_a["label_win"], errors="coerce").mean()
    win_pct_b = pd.to_numeric(common_b["label_win"], errors="coerce").mean()

    return {
        "common_opp_count": float(len(common_opponents)),
        "delta_common_opp_win_pct": float(win_pct_a - win_pct_b),
        "delta_common_opp_matches": float(len(common_a) - len(common_b)),
    }


def _compute_hand_win_pct_before_date(
    long_df: pd.DataFrame,
    player_id: object,
    opponent_hand,
    as_of_date: pd.Timestamp,
    window: int = 10,
):
    opponent_hand_norm = _normalize_hand(opponent_hand)
    hist = long_df[
        (long_df["player_id"] == player_id) &
        (long_df["tourney_date"] < as_of_date)
    ].copy()

    if hist.empty or "opponent_hand" not in hist.columns:
        return pd.NA

    hist["_opponent_hand_group"] = hist["opponent_hand"].apply(_normalize_hand)
    hist = hist[hist["_opponent_hand_group"] == opponent_hand_norm].copy()
    if hist.empty:
        return pd.NA

    hist = hist.sort_values(["tourney_date", "match_id"]).tail(window)
    return float(hist["label_win"].mean())


def _current_win_streak_from_history(hist: pd.DataFrame) -> int:
    streak = 0
    if hist.empty:
        return streak

    for value in pd.to_numeric(hist.sort_values(["tourney_date", "match_id"])["label_win"], errors="coerce").iloc[::-1]:
        if pd.isna(value) or int(value) != 1:
            break
        streak += 1
    return streak


def _prior_mean_with_history_indicator(hist: pd.DataFrame, col: str, window: int = 30, min_periods: int = 10) -> tuple:
    if col not in hist.columns:
        return pd.NA, 0

    values = pd.to_numeric(hist[col].tail(window), errors="coerce").dropna()
    if len(values) < min_periods:
        return pd.NA, 0

    return float(values.mean()), 1


def _build_player_state(
    long_df: pd.DataFrame,
    resolved: ResolvedPlayer,
    as_of_date: pd.Timestamp,
    match_surface: str | None,
    round_name: str | None,
) -> dict:
    hist = long_df[
        (long_df["player_id"] == resolved.player_id) &
        (long_df["tourney_date"] < as_of_date)
    ].copy()

    hist = hist.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)

    if hist.empty:
        raise ValueError(
            f"No historical matches found before {as_of_date.date()} for player '{resolved.player_name}'."
        )

    latest = hist.iloc[-1]
    recent10 = hist.tail(10)
    last_365_start = as_of_date - pd.Timedelta(days=365)
    recent365 = hist[hist["tourney_date"] >= last_365_start]
    surface_hist = (hist[hist["surface"].astype(str).str.title() == match_surface].copy()
        if match_surface is not None
        else hist.iloc[0:0].copy())
    recent10_surface = surface_hist.tail(10)
    has_surface_history = int(len(surface_hist) >= 5)
    last_7_start = as_of_date - pd.Timedelta(days=7)
    last_30_start = as_of_date - pd.Timedelta(days=30)
    recent7 = hist[hist["tourney_date"] >= last_7_start]
    recent30 = hist[hist["tourney_date"] >= last_30_start]
    recent30_surface = surface_hist[surface_hist["tourney_date"] >= last_30_start]

    last_match_date = latest["tourney_date"]
    days_since_last_match = int((as_of_date - last_match_date).days)

    current_elo = latest.get("player_elo_post", pd.NA)
    if pd.isna(current_elo):
        current_elo = latest.get("player_elo_pre", pd.NA)
    if pd.isna(current_elo):
        current_elo = DEFAULT_ELO

    current_surface_elo = _get_latest_surface_elo_before_date(
        long_df=long_df,
        player_id=resolved.player_id,
        as_of_date=as_of_date,
        surface=match_surface,)

    # use raw recent-match stats when available; otherwise fall back to latest rolling values
    serve_win_pct_last10 = (recent10["service_points_won_pct"].mean()
        if "service_points_won_pct" in recent10.columns and len(recent10) > 0
        else latest.get("serve_win_pct_last10", pd.NA))

    return_win_pct_last10 = (recent10["return_points_won_pct"].mean()
        if "return_points_won_pct" in recent10.columns and len(recent10) > 0
        else latest.get("return_win_pct_last10", pd.NA))

    service_points_won_pct_30, has_serve_history = _prior_mean_with_history_indicator(
        hist,
        "service_points_won_pct",
        window=30,
        min_periods=10,
    )
    return_points_won_pct_30, has_return_history = _prior_mean_with_history_indicator(
        hist,
        "return_points_won_pct",
        window=30,
        min_periods=10,
    )

    bp_conversion_last10 = (recent10["bp_conversion_pct"].mean()
        if "bp_conversion_pct" in recent10.columns and len(recent10) > 0
        else latest.get("bp_conversion_last10", pd.NA))

    bp_saved_pct_last10 = (recent10["bp_saved_pct"].mean()
        if "bp_saved_pct" in recent10.columns and len(recent10) > 0
        else latest.get("bp_saved_pct_last10", pd.NA))

    ace_pct_last10 = (recent10["aces_per_service_point"].mean()
        if "aces_per_service_point" in recent10.columns and len(recent10) > 0
        else latest.get("ace_pct_last10", pd.NA))

    df_pct_last10 = (recent10["df_per_service_point"].mean()
        if "df_per_service_point" in recent10.columns and len(recent10) > 0
        else latest.get("df_pct_last10", pd.NA))

    first_serve_in_pct_last10 = (recent10["first_serve_in_pct"].mean()
        if "first_serve_in_pct" in recent10.columns and len(recent10) > 0
        else latest.get("first_serve_in_pct_last10", pd.NA))

    if "ace_vs_df" in recent10.columns and len(recent10) > 0:
        ace_vs_df_last10 = recent10["ace_vs_df"].mean()
    elif {"aces", "double_faults"}.issubset(recent10.columns) and len(recent10) > 0:
        ace_vs_df_last10 = (recent10["aces"] / recent10["double_faults"].replace(0, pd.NA)).mean()
    else:
        ace_vs_df_last10 = latest.get("ace_vs_df_last10", pd.NA)

    if "second_serve_won_per_service_game" in recent10.columns and len(recent10) > 0:
        second_serve_won_per_service_game_last10 = recent10["second_serve_won_per_service_game"].mean()
    elif {"second_won", "service_games"}.issubset(recent10.columns) and len(recent10) > 0:
        second_serve_won_per_service_game_last10 = (recent10["second_won"] / recent10["service_games"]).mean()
    else:
        second_serve_won_per_service_game_last10 = latest.get("second_serve_won_per_service_game_last10", pd.NA)
    
    serve_win_pct_last10_surface = (recent10_surface["service_points_won_pct"].mean()
        if "service_points_won_pct" in recent10_surface.columns and has_surface_history
        else latest.get("serve_win_pct_last10_surface", pd.NA))

    return_win_pct_last10_surface = (recent10_surface["return_points_won_pct"].mean()
        if "return_points_won_pct" in recent10_surface.columns and has_surface_history
        else latest.get("return_win_pct_last10_surface", pd.NA))

    bp_conversion_last10_surface = (recent10_surface["bp_conversion_pct"].mean()
        if "bp_conversion_pct" in recent10_surface.columns and has_surface_history
        else latest.get("bp_conversion_last10_surface", pd.NA))

    bp_saved_pct_last10_surface = (recent10_surface["bp_saved_pct"].mean()
        if "bp_saved_pct" in recent10_surface.columns and has_surface_history
        else latest.get("bp_saved_pct_last10_surface", pd.NA))

    surface_win_pct_last10 = (recent10_surface["label_win"].mean()
        if has_surface_history
        else latest.get("surface_win_pct_last10", pd.NA))

    if not has_surface_history:
        serve_win_pct_last10_surface = pd.NA
        return_win_pct_last10_surface = pd.NA
        bp_conversion_last10_surface = pd.NA
        bp_saved_pct_last10_surface = pd.NA
        surface_win_pct_last10 = pd.NA

    if len(surface_hist) > 0:
        last_surface_match_date = surface_hist.iloc[-1]["tourney_date"]
        days_since_last_match_surface = int((as_of_date - last_surface_match_date).days)
    else:
        days_since_last_match_surface = pd.NA

    aces_avg_last10 = (recent10["aces"].mean()
        if "aces" in recent10.columns and len(recent10) > 0
        else latest.get("aces_avg_last10", pd.NA))

    win_rate_last10 = (recent10["label_win"].mean()
        if len(recent10) > 0
        else latest.get("win_rate_last10", pd.NA))

    win_pct_last_365_days = (recent365["label_win"].mean()
        if len(recent365) > 0
        else latest.get("win_pct_last_365_days", pd.NA))

    previous_match_win = latest.get("label_win", latest.get("previous_match_win", pd.NA))
    current_win_streak = _current_win_streak_from_history(hist)

    round_key = _normalize_round_name(round_name)
    round_hist = hist[hist["round"].fillna("UNK").astype(str).str.upper().str.strip() == round_key].copy()
    round_win_pct = (round_hist["label_win"].mean()
        if len(round_hist) > 0
        else latest.get("round_win_pct", pd.NA))

    return {
        "player_id": resolved.player_id,
        "player_name": resolved.player_name,
        "player_hand": latest.get("player_hand", pd.NA),
        "player_ht": latest.get("player_ht", pd.NA),
        "player_age": latest.get("player_age", pd.NA),
        "player_rank": latest.get("player_rank", pd.NA),
        "player_rank_points": latest.get("player_rank_points", pd.NA),
        "player_elo_pre": current_elo,
        "player_surface_elo_pre": current_surface_elo,
        "matches_played": int(len(hist)),
        "win_rate_last10": win_rate_last10,
        "win_pct_last_365_days": win_pct_last_365_days,
        "previous_match_win": previous_match_win,
        "round_win_pct": round_win_pct,
        "current_win_streak": current_win_streak,
        "aces_avg_last10": aces_avg_last10,
        "ace_pct_last10": ace_pct_last10,
        "df_pct_last10": df_pct_last10,
        "first_serve_in_pct_last10": first_serve_in_pct_last10,
        "ace_vs_df_last10": ace_vs_df_last10,
        "second_serve_won_per_service_game_last10": second_serve_won_per_service_game_last10,
        "serve_win_pct_last10": serve_win_pct_last10,
        "return_win_pct_last10": return_win_pct_last10,
        "service_points_won_pct_30": service_points_won_pct_30,
        "return_points_won_pct_30": return_points_won_pct_30,
        "has_serve_history": has_serve_history,
        "has_return_history": has_return_history,
        "bp_conversion_last10": bp_conversion_last10,
        "bp_saved_pct_last10": bp_saved_pct_last10,
        "days_since_last_match": days_since_last_match,
        "matches_last_7_days": int(len(recent7)),
        "matches_last_30_days": int(len(recent30)),
        "matches_last_365_days": int(len(recent365)),
        "has_surface_history": has_surface_history,
        "surface_win_pct_last10": surface_win_pct_last10,
        "serve_win_pct_last10_surface": serve_win_pct_last10_surface,
        "return_win_pct_last10_surface": return_win_pct_last10_surface,
        "bp_conversion_last10_surface": bp_conversion_last10_surface,
        "bp_saved_pct_last10_surface": bp_saved_pct_last10_surface,
        "days_since_last_match_surface": days_since_last_match_surface,
        "matches_last_30_days_surface": int(len(recent30_surface)),
    }


def _build_baseline_row(state_a: dict, state_b: dict, tour: str, as_of_date: pd.Timestamp, surface: str | None,
    round_name: str | None, best_of: int | None, tourney_level: str | None, h2h_summary: dict,
    common_opponent_summary: dict, ) -> pd.DataFrame:
    round_ordinal = _round_to_ordinal(round_name)
    tourney_level = _normalize_tourney_level(tourney_level)
    hand_a = _normalize_hand(state_a["player_hand"])
    hand_b = _normalize_hand(state_b["player_hand"])

    row = {
        "match_id": f"predict__{as_of_date.date()}__{state_a['player_id']}__{state_b['player_id']}",
        "tour": tour,
        "tourney_date": as_of_date,
        "surface": surface if surface is not None else pd.NA,
        "round": round_name if round_name is not None else pd.NA,
        "best_of": best_of if best_of is not None else pd.NA,
        "tourney_level": tourney_level,

        "player_id_a": state_a["player_id"],
        "player_name_a": state_a["player_name"],
        "player_id_b": state_b["player_id"],
        "player_name_b": state_b["player_name"],
        "handedness_combo": f"{hand_a}_vs_{hand_b}",

        "label_player_a_win": pd.NA,

        "delta_rank_adv": state_b["player_rank"] - state_a["player_rank"],
        "delta_rank_points": state_a["player_rank_points"] - state_b["player_rank_points"],

        "elo_a": state_a["player_elo_pre"],
        "elo_b": state_b["player_elo_pre"],
        "delta_elo": state_a["player_elo_pre"] - state_b["player_elo_pre"],

        "delta_surface_elo": state_a["player_surface_elo_pre"] - state_b["player_surface_elo_pre"],
        "delta_surface_advantage": (
            (state_a["player_surface_elo_pre"] - state_a["player_elo_pre"])
            - (state_b["player_surface_elo_pre"] - state_b["player_elo_pre"])
        ),

        "delta_age": state_a["player_age"] - state_b["player_age"],
        "delta_age_30": _age_peak_closeness(state_a["player_age"]) - _age_peak_closeness(state_b["player_age"]),
        "delta_age_int": (
            _age_peak_distance_squared(state_a["player_age"])
            - _age_peak_distance_squared(state_b["player_age"])
        ),
        "delta_height": state_a["player_ht"] - state_b["player_ht"],

        "delta_matches_played": state_a["matches_played"] - state_b["matches_played"],
        "delta_win_rate_last10": state_a["win_rate_last10"] - state_b["win_rate_last10"],
        "delta_win_pct_last_365_days": state_a["win_pct_last_365_days"] - state_b["win_pct_last_365_days"],
        "delta_previous_match_win": state_a["previous_match_win"] - state_b["previous_match_win"],
        "delta_round_win_pct": state_a["round_win_pct"] - state_b["round_win_pct"],
        "delta_current_win_streak": state_a["current_win_streak"] - state_b["current_win_streak"],
        "delta_aces_avg_last10": state_a["aces_avg_last10"] - state_b["aces_avg_last10"],
        "delta_ace_pct_last10": state_a["ace_pct_last10"] - state_b["ace_pct_last10"],
        "delta_df_pct_last10": state_a["df_pct_last10"] - state_b["df_pct_last10"],
        "delta_first_serve_in_pct_last10": state_a["first_serve_in_pct_last10"] - state_b["first_serve_in_pct_last10"],
        "delta_ace_vs_df_last10": state_a["ace_vs_df_last10"] - state_b["ace_vs_df_last10"],
        "delta_second_serve_won_per_service_game_last10": (
            state_a["second_serve_won_per_service_game_last10"]
            - state_b["second_serve_won_per_service_game_last10"]
        ),
        "delta_serve_win_pct_last10": state_a["serve_win_pct_last10"] - state_b["serve_win_pct_last10"],
        "delta_return_win_pct_last10": state_a["return_win_pct_last10"] - state_b["return_win_pct_last10"],
        "service_points_won_pct_30_a": state_a["service_points_won_pct_30"],
        "service_points_won_pct_30_b": state_b["service_points_won_pct_30"],
        "return_points_won_pct_30_a": state_a["return_points_won_pct_30"],
        "return_points_won_pct_30_b": state_b["return_points_won_pct_30"],
        "has_serve_history_a": state_a["has_serve_history"],
        "has_serve_history_b": state_b["has_serve_history"],
        "has_return_history_a": state_a["has_return_history"],
        "has_return_history_b": state_b["has_return_history"],
        "delta_bp_conversion_last10": state_a["bp_conversion_last10"] - state_b["bp_conversion_last10"],
        "delta_bp_saved_pct_last10": state_a["bp_saved_pct_last10"] - state_b["bp_saved_pct_last10"],
        "delta_days_since_last_match": state_a["days_since_last_match"] - state_b["days_since_last_match"],
        "delta_matches_last_7_days": state_a["matches_last_7_days"] - state_b["matches_last_7_days"],
        "delta_matches_last_30_days": state_a["matches_last_30_days"] - state_b["matches_last_30_days"],
        "delta_matches_last_365_days": state_a["matches_last_365_days"] - state_b["matches_last_365_days"],
        "has_surface_history": int(state_a["has_surface_history"] == 1 and state_b["has_surface_history"] == 1),
        "delta_surface_win_pct_last10": state_a["surface_win_pct_last10"] - state_b["surface_win_pct_last10"],
        "delta_hand_win_pct_last10": state_a["hand_win_pct_last10"] - state_b["hand_win_pct_last10"],
        "delta_serve_win_pct_last10_surface": (state_a["serve_win_pct_last10_surface"] - state_b["serve_win_pct_last10_surface"]),
        "delta_return_win_pct_last10_surface": (state_a["return_win_pct_last10_surface"] - state_b["return_win_pct_last10_surface"]),
        "delta_bp_conversion_last10_surface": (state_a["bp_conversion_last10_surface"] - state_b["bp_conversion_last10_surface"]),
        "delta_bp_saved_pct_last10_surface": (state_a["bp_saved_pct_last10_surface"] - state_b["bp_saved_pct_last10_surface"]),
        "delta_days_since_last_match_surface": (state_a["days_since_last_match_surface"] - state_b["days_since_last_match_surface"]),
        "delta_matches_last_30_days_surface": (state_a["matches_last_30_days_surface"] - state_b["matches_last_30_days_surface"]),
        "delta_h2h": h2h_summary["delta_h2h"],
        "delta_h2h_wins": h2h_summary["delta_h2h_wins"],
        "delta_h2h_losses": h2h_summary["delta_h2h_losses"],
        "h2h_matches_total": h2h_summary["h2h_matches_total"],
        "first_meeting": h2h_summary["first_meeting"],
        "common_opp_count": common_opponent_summary["common_opp_count"],
        "delta_common_opp_win_pct": common_opponent_summary["delta_common_opp_win_pct"],
        "delta_common_opp_matches": common_opponent_summary["delta_common_opp_matches"],

        "is_clay": int(surface == "Clay") if surface is not None else pd.NA,
        "is_grass": int(surface == "Grass") if surface is not None else pd.NA,
        "same_hand": int(hand_a == hand_b and hand_a != "U"),
        "round_rr": _is_round_robin(round_name),
        "round_ordinal": round_ordinal,
    }

    return add_markov_match_features(pd.DataFrame([row]))



def predict_match_probability(project_root: Path, tour: str, requested_player_a: str, requested_player_b: str,
    match_date: str, surface: str | None = None, round_name: str | None = None, best_of: int | None = None,
    tourney_level: str | None = None, source: str = "sackmann", model: str = "logit",
    model_variant: str = "baseline", ) -> dict:

    tour = tour.lower().strip()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")
    
    source = source.lower().strip()
    if source not in {"sackmann", "tml"}:
        raise ValueError("source must be either 'sackmann' or 'tml'")

    if source == "tml" and tour != "atp":
        raise ValueError("TML source is currently supported only for ATP.")
    
    model = model.lower().strip()
    if model not in {"logit", "xgb"}:
        raise ValueError("model must be either 'logit' or 'xgb'")

    model_variant = model_variant.lower().strip()
    if model_variant not in {"baseline", "tuned"}:
        raise ValueError("model_variant must be either 'baseline' or 'tuned'")

    as_of_date = pd.Timestamp(match_date)
    surface = _normalize_surface(surface)

    long_df = _load_long_view(project_root, tour, source=source)

    resolved_a = _resolve_player(long_df, requested_player_a, as_of_date)
    resolved_b = _resolve_player(long_df, requested_player_b, as_of_date)

    state_req_a = _build_player_state(long_df, resolved_a, as_of_date, surface, round_name)
    state_req_b = _build_player_state(long_df, resolved_b, as_of_date, surface, round_name)
    state_req_a["hand_win_pct_last10"] = _compute_hand_win_pct_before_date(
        long_df=long_df,
        player_id=state_req_a["player_id"],
        opponent_hand=state_req_b["player_hand"],
        as_of_date=as_of_date,
    )
    state_req_b["hand_win_pct_last10"] = _compute_hand_win_pct_before_date(
        long_df=long_df,
        player_id=state_req_b["player_id"],
        opponent_hand=state_req_a["player_hand"],
        as_of_date=as_of_date,
    )

    key_a = _player_sort_key(state_req_a["player_id"], state_req_a["player_name"])
    key_b = _player_sort_key(state_req_b["player_id"], state_req_b["player_name"])

    if key_a <= key_b:
        state_internal_a = state_req_a
        state_internal_b = state_req_b
        requested_a_is_internal_a = True
    else:
        state_internal_a = state_req_b
        state_internal_b = state_req_a
        requested_a_is_internal_a = False

    h2h_summary = _compute_h2h_summary_before_date(
        long_df=long_df,
        player_a_id=state_internal_a["player_id"],
        player_b_id=state_internal_b["player_id"],
        as_of_date=as_of_date,
    )
    common_opponent_summary = _compute_common_opponent_summary_before_date(
        long_df=long_df,
        player_a_id=state_internal_a["player_id"],
        player_b_id=state_internal_b["player_id"],
        as_of_date=as_of_date,
    )

    baseline_row = _build_baseline_row(
        state_internal_a,
        state_internal_b,
        tour=tour,
        as_of_date=as_of_date,
        surface=surface,
        round_name=round_name,
        best_of=best_of,
        tourney_level=tourney_level,
        h2h_summary=h2h_summary,
        common_opponent_summary=common_opponent_summary,
    )

    surface_specific = surface is not None
    feature_cols = get_feature_columns(surface_specific=surface_specific)
    X = baseline_row[feature_cols].copy()

    raw_prob_internal_a = None
    chosen_calibration_method = None

        

    if model == "logit":
        suffix = _build_artifact_suffix(source=source, surface=surface)
        model_path = project_root / "data" / "models" / f"{tour}_logit_baseline{suffix}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model artifact not found: {model_path}. Train the model first.")

        pipeline = load_model_artifact(model_path)
        prob_internal_a = float(pipeline.predict_proba(X)[:, 1][0])


    elif model == "xgb":
        xgb_assets = load_xgb_prediction_artifacts(project_root, tour, source, surface, model_variant)

        model_artifact = xgb_assets["model_artifact"]
        chosen_calibration_method = xgb_assets["chosen_calibration_method"]
        calibrator_artifact = xgb_assets["calibrator_artifact"]

        raw_pred_prob = predict_proba_from_xgb_artifact(model_artifact, X)
        calibrated_pred_prob = apply_calibration(
            calibration_method=chosen_calibration_method,
            calibrator_artifact=calibrator_artifact,
            pred_prob=raw_pred_prob,
        )

        raw_prob_internal_a = float(raw_pred_prob.iloc[0] if hasattr(raw_pred_prob, "iloc") else raw_pred_prob[0])
        prob_internal_a = float(
            calibrated_pred_prob.iloc[0] if hasattr(calibrated_pred_prob, "iloc") else calibrated_pred_prob[0]
        )

    if requested_a_is_internal_a:
        prob_requested_a = prob_internal_a
    else:
        prob_requested_a = 1.0 - prob_internal_a

    prob_requested_b = 1.0 - prob_requested_a

    return {
        "tour": tour,
        "source": source,
        "model": model,
        "model_variant": model_variant,
        "chosen_calibration_method": chosen_calibration_method,
        "match_date": str(as_of_date.date()),
        "surface": surface,
        "round": round_name,
        "best_of": best_of,
        "tourney_level": _normalize_tourney_level(tourney_level),
        "requested_player_a": requested_player_a,
        "requested_player_b": requested_player_b,
        "canonical_player_a": resolved_a.player_name,
        "canonical_player_b": resolved_b.player_name,
        "prob_requested_player_a_win": prob_requested_a,
        "prob_requested_player_b_win": prob_requested_b,
        "internal_player_a": state_internal_a["player_name"],
        "internal_player_b": state_internal_b["player_name"],
        "internal_prob_player_a_win": prob_internal_a,
        "raw_internal_prob_player_a_win": raw_prob_internal_a,
        "feature_snapshot": baseline_row.iloc[0].to_dict(),
    }

      

def _get_xgb_artifact_paths(
    project_root: Path,
    tour: str,
    source: str,
    surface: str | None,
    model_variant: str = "baseline",
) -> dict[str, Path]:
    model_variant = model_variant.lower().strip()
    if model_variant not in {"baseline", "tuned"}:
        raise ValueError("model_variant must be either 'baseline' or 'tuned'")

    suffix = _build_artifact_suffix(source=source, surface=surface)
    models_dir = project_root / "data" / "models"
    artifact_stem = f"{tour}_xgb_{model_variant}{suffix}"

    return {
        "model": models_dir / f"{artifact_stem}.joblib",
        "metrics": models_dir / f"{artifact_stem}_metrics.json",
        "metadata": models_dir / f"{artifact_stem}_meta.json",
        "isotonic_calibrator": models_dir / f"{artifact_stem}_isotonic_calibrator.joblib",
        "sigmoid_calibrator": models_dir / f"{artifact_stem}_sigmoid_calibrator.joblib",
    }


def load_xgb_prediction_artifacts(
    project_root: Path,
    tour: str,
    source: str,
    surface: str | None,
    model_variant: str = "baseline",
) -> dict:

    paths = _get_xgb_artifact_paths(project_root, tour, source, surface, model_variant)
    model_artifact = load_model_artifact(paths["model"])

    metrics_payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    chosen_calibration_method = metrics_payload.get("chosen_calibration_method")

    calibrator_artifact = None
    if chosen_calibration_method == "isotonic":
        calibrator_artifact = load_model_artifact(paths["isotonic_calibrator"])
    elif chosen_calibration_method == "sigmoid":
        calibrator_artifact = load_model_artifact(paths["sigmoid_calibrator"])

    return {
        "model_artifact": model_artifact,
        "chosen_calibration_method": chosen_calibration_method,
        "calibrator_artifact": calibrator_artifact,
        "paths": paths,
    }
