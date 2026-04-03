from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

from tennis_cli.models.dataset import get_feature_columns
from tennis_cli.models.io import load_model_artifact


DEFAULT_ELO = 1500.0


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


def _load_long_view(project_root: Path, tour: str) -> pd.DataFrame:
    path = project_root / "data" / "features" / f"{tour}_long_2015_2025.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Long-view feature file not found: {path}")

    df = pd.read_parquet(path)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    if df["tourney_date"].isna().any():
        raise ValueError("Long-view dataset contains invalid 'tourney_date' values")

    df = df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)
    return df


def _resolve_player(long_df: pd.DataFrame, player_name: str, as_of_date: pd.Timestamp) -> ResolvedPlayer:
    """
    Resolve a user-provided player name to a canonical player_id/player_name pair.

    Matching rule:
    - case-insensitive exact match on player_name
    - if multiple player_ids share the same name, pick the one with the latest
      appearance before the requested date
    """
    normalized = _normalize_name(player_name)

    exact = long_df[long_df["player_name"].astype(str).str.casefold() == normalized].copy()

    if exact.empty:
        unique_names = sorted(long_df["player_name"].dropna().astype(str).unique())
        suggestions = get_close_matches(player_name, unique_names, n=5, cutoff=0.6)
        suggestion_text = ", ".join(suggestions) if suggestions else "no close matches found"
        raise ValueError(f"Player '{player_name}' not found in {long_df['tour'].iloc[0].upper()} long-view data. "
            f"Suggestions: {suggestion_text}")

    prior = exact[exact["tourney_date"] < as_of_date].copy()
    choice_pool = prior if not prior.empty else exact

    latest = choice_pool.sort_values(["tourney_date", "match_id"]).iloc[-1]
    return ResolvedPlayer(player_id=latest["player_id"], player_name=str(latest["player_name"]), )


def _build_player_state(long_df: pd.DataFrame, resolved: ResolvedPlayer, as_of_date: pd.Timestamp, ) -> dict:
    """
    Build a pre-match player state using only history strictly before as_of_date.
    """
    hist = long_df[(long_df["player_id"] == resolved.player_id) & (long_df["tourney_date"] < as_of_date)].copy()

    hist = hist.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)

    if hist.empty:
        raise ValueError(
            f"No historical matches found before {as_of_date.date()} for player '{resolved.player_name}'.")

    latest = hist.iloc[-1]
    recent10 = hist.tail(10)
    last_30_start = as_of_date - pd.Timedelta(days=30)
    recent30 = hist[hist["tourney_date"] >= last_30_start]

    last_match_date = latest["tourney_date"]
    days_since_last_match = int((as_of_date - last_match_date).days)

    # For an upcoming match, the best Elo state is the POST value from the player's
    # most recent completed historical match.
    current_elo = latest["player_elo_post"]
    if pd.isna(current_elo):
        current_elo = latest["player_elo_pre"]
    if pd.isna(current_elo):
        current_elo = DEFAULT_ELO

    return {
        "player_id": resolved.player_id,
        "player_name": resolved.player_name,
        "player_hand": latest.get("player_hand", pd.NA),
        "player_ht": latest.get("player_ht", pd.NA),
        "player_age": latest.get("player_age", pd.NA),
        "player_rank": latest.get("player_rank", pd.NA),
        "player_rank_points": latest.get("player_rank_points", pd.NA),
        "player_elo_pre": current_elo,
        "matches_played": int(len(hist)),
        "win_rate_last10": recent10["label_win"].mean() if len(recent10) > 0 else pd.NA,
        "aces_avg_last10": recent10["aces"].mean() if len(recent10) > 0 else pd.NA,
        "serve_win_pct_last10": (recent10["service_points_won_pct"].mean() if len(recent10) > 0 else pd.NA),
        "days_since_last_match": days_since_last_match,
        "matches_last_30_days": int(len(recent30)),
    }


def _build_baseline_row(state_a: dict, state_b: dict, tour: str, as_of_date: pd.Timestamp, ) -> pd.DataFrame:
    """
    Build one baseline-style match row using the same feature formulas as training.
    """
    row = {
        "match_id": f"predict__{as_of_date.date()}__{state_a['player_id']}__{state_b['player_id']}",
        "tour": tour,
        "tourney_date": as_of_date,
        "surface": pd.NA,
        "round": pd.NA,
        "best_of": pd.NA,

        "player_id_a": state_a["player_id"],
        "player_name_a": state_a["player_name"],
        "player_id_b": state_b["player_id"],
        "player_name_b": state_b["player_name"],
        "handedness_combo": (f"{str(state_a['player_hand']) if pd.notna(state_a['player_hand']) else 'U'}"
            f"_vs_"f"{str(state_b['player_hand']) if pd.notna(state_b['player_hand']) else 'U'}"),

        # Placeholder target not used in inference, but helpful for consistent shape
        "label_player_a_win": pd.NA,

        "delta_rank_adv": state_b["player_rank"] - state_a["player_rank"],
        "delta_rank_points": state_a["player_rank_points"] - state_b["player_rank_points"],

        "elo_a": state_a["player_elo_pre"],
        "elo_b": state_b["player_elo_pre"],
        "delta_elo": state_a["player_elo_pre"] - state_b["player_elo_pre"],

        "delta_age": state_a["player_age"] - state_b["player_age"],
        "delta_height": state_a["player_ht"] - state_b["player_ht"],

        "delta_matches_played": state_a["matches_played"] - state_b["matches_played"],
        "delta_win_rate_last10": state_a["win_rate_last10"] - state_b["win_rate_last10"],
        "delta_aces_avg_last10": state_a["aces_avg_last10"] - state_b["aces_avg_last10"],
        "delta_serve_win_pct_last10": (state_a["serve_win_pct_last10"] - state_b["serve_win_pct_last10"]),
        "delta_days_since_last_match": (state_a["days_since_last_match"] - state_b["days_since_last_match"]),
        "delta_matches_last_30_days": (state_a["matches_last_30_days"] - state_b["matches_last_30_days"]),
    }

    return pd.DataFrame([row])


def predict_match_probability(project_root: Path, tour: str, requested_player_a: str, 
                              requested_player_b: str, match_date: str, ) -> dict:
    """
    Predict the probability of a match outcome for two players on a given date.

    The returned probability is aligned to the REQUESTED player order,
    even though the internal baseline row uses deterministic A/B ordering.
    """
    tour = tour.lower().strip()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")

    as_of_date = pd.Timestamp(match_date)
    long_df = _load_long_view(project_root, tour)

    resolved_a = _resolve_player(long_df, requested_player_a, as_of_date)
    resolved_b = _resolve_player(long_df, requested_player_b, as_of_date)

    state_req_a = _build_player_state(long_df, resolved_a, as_of_date)
    state_req_b = _build_player_state(long_df, resolved_b, as_of_date)

    # Internal deterministic ordering must match training logic
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

    baseline_row = _build_baseline_row(state_internal_a, state_internal_b, tour=tour, as_of_date=as_of_date, )

    model_path = project_root / "data" / "models" / f"{tour}_logit_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model artifact not found: {model_path}. Train the model first.")

    pipeline = load_model_artifact(model_path)

    feature_cols = get_feature_columns()
    X = baseline_row[feature_cols].copy()

    prob_internal_a = float(pipeline.predict_proba(X)[:, 1][0])

    if requested_a_is_internal_a:
        prob_requested_a = prob_internal_a
    else:
        prob_requested_a = 1.0 - prob_internal_a

    prob_requested_b = 1.0 - prob_requested_a

    return {
        "tour": tour,
        "match_date": str(as_of_date.date()),
        "requested_player_a": requested_player_a,
        "requested_player_b": requested_player_b,
        "canonical_player_a": resolved_a.player_name,
        "canonical_player_b": resolved_b.player_name,
        "prob_requested_player_a_win": prob_requested_a,
        "prob_requested_player_b_win": prob_requested_b,
        "internal_player_a": state_internal_a["player_name"],
        "internal_player_b": state_internal_b["player_name"],
        "internal_prob_player_a_win": prob_internal_a,
        "feature_snapshot": baseline_row.iloc[0].to_dict(),
    }