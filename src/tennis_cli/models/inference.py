from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
import json

from tennis_cli.models.dataset import get_feature_columns
from tennis_cli.models.io import load_model_artifact
from tennis_cli.models.xgboost_model import (apply_calibration, predict_proba_from_xgb_artifact, )


DEFAULT_ELO = 1500.0
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


def _compute_h2h_delta_before_date(
    long_df: pd.DataFrame,
    player_a_id: object,
    player_b_id: object,
    as_of_date: pd.Timestamp,
) -> float:
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

    if hist_a.empty and hist_b.empty:
        ratio_a = 0.5
        ratio_b = 0.5
    else:
        ratio_a = float(hist_a["label_win"].mean()) if not hist_a.empty else 0.5
        ratio_b = float(hist_b["label_win"].mean()) if not hist_b.empty else 0.5

    return ratio_a - ratio_b


def _build_player_state(
    long_df: pd.DataFrame,
    resolved: ResolvedPlayer,
    as_of_date: pd.Timestamp,
    match_surface: str | None,
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
    surface_hist = (hist[hist["surface"].astype(str).str.title() == match_surface].copy()
        if match_surface is not None
        else hist.iloc[0:0].copy())
    recent10_surface = surface_hist.tail(10)
    last_30_start = as_of_date - pd.Timedelta(days=30)
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

    bp_conversion_last10 = (recent10["bp_conversion_pct"].mean()
        if "bp_conversion_pct" in recent10.columns and len(recent10) > 0
        else latest.get("bp_conversion_last10", pd.NA))
    
    serve_win_pct_last10_surface = (recent10_surface["service_points_won_pct"].mean()
        if "service_points_won_pct" in recent10_surface.columns and len(recent10_surface) > 0
        else latest.get("serve_win_pct_last10_surface", pd.NA))

    return_win_pct_last10_surface = (recent10_surface["return_points_won_pct"].mean()
        if "return_points_won_pct" in recent10_surface.columns and len(recent10_surface) > 0
        else latest.get("return_win_pct_last10_surface", pd.NA))

    bp_conversion_last10_surface = (recent10_surface["bp_conversion_pct"].mean()
        if "bp_conversion_pct" in recent10_surface.columns and len(recent10_surface) > 0
        else latest.get("bp_conversion_last10_surface", pd.NA))

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
        "aces_avg_last10": aces_avg_last10,
        "serve_win_pct_last10": serve_win_pct_last10,
        "return_win_pct_last10": return_win_pct_last10,
        "bp_conversion_last10": bp_conversion_last10,
        "days_since_last_match": days_since_last_match,
        "matches_last_30_days": int(len(recent30)),
        "serve_win_pct_last10_surface": serve_win_pct_last10_surface,
        "return_win_pct_last10_surface": return_win_pct_last10_surface,
        "bp_conversion_last10_surface": bp_conversion_last10_surface,
        "days_since_last_match_surface": days_since_last_match_surface,
        "matches_last_30_days_surface": int(len(recent30_surface)),
    }


def _build_baseline_row(state_a: dict, state_b: dict, tour: str, as_of_date: pd.Timestamp, surface: str | None,
    round_name: str | None, best_of: int | None, delta_h2h: float, ) -> pd.DataFrame:
    round_ordinal = _round_to_ordinal(round_name)

    row = {
        "match_id": f"predict__{as_of_date.date()}__{state_a['player_id']}__{state_b['player_id']}",
        "tour": tour,
        "tourney_date": as_of_date,
        "surface": surface if surface is not None else pd.NA,
        "round": round_name if round_name is not None else pd.NA,
        "best_of": best_of if best_of is not None else pd.NA,

        "player_id_a": state_a["player_id"],
        "player_name_a": state_a["player_name"],
        "player_id_b": state_b["player_id"],
        "player_name_b": state_b["player_name"],
        "handedness_combo": (
            f"{str(state_a['player_hand']) if pd.notna(state_a['player_hand']) else 'U'}"
            f"_vs_"
            f"{str(state_b['player_hand']) if pd.notna(state_b['player_hand']) else 'U'}"
        ),

        "label_player_a_win": pd.NA,

        "delta_rank_adv": state_b["player_rank"] - state_a["player_rank"],
        "delta_rank_points": state_a["player_rank_points"] - state_b["player_rank_points"],

        "elo_a": state_a["player_elo_pre"],
        "elo_b": state_b["player_elo_pre"],
        "delta_elo": state_a["player_elo_pre"] - state_b["player_elo_pre"],

        "delta_surface_elo": state_a["player_surface_elo_pre"] - state_b["player_surface_elo_pre"],

        "delta_age": state_a["player_age"] - state_b["player_age"],
        "delta_height": state_a["player_ht"] - state_b["player_ht"],

        "delta_matches_played": state_a["matches_played"] - state_b["matches_played"],
        "delta_win_rate_last10": state_a["win_rate_last10"] - state_b["win_rate_last10"],
        "delta_aces_avg_last10": state_a["aces_avg_last10"] - state_b["aces_avg_last10"],
        "delta_serve_win_pct_last10": state_a["serve_win_pct_last10"] - state_b["serve_win_pct_last10"],
        "delta_return_win_pct_last10": state_a["return_win_pct_last10"] - state_b["return_win_pct_last10"],
        "delta_bp_conversion_last10": state_a["bp_conversion_last10"] - state_b["bp_conversion_last10"],
        "delta_days_since_last_match": state_a["days_since_last_match"] - state_b["days_since_last_match"],
        "delta_matches_last_30_days": state_a["matches_last_30_days"] - state_b["matches_last_30_days"],
        "delta_serve_win_pct_last10_surface": (state_a["serve_win_pct_last10_surface"] - state_b["serve_win_pct_last10_surface"]),
        "delta_return_win_pct_last10_surface": (state_a["return_win_pct_last10_surface"] - state_b["return_win_pct_last10_surface"]),
        "delta_bp_conversion_last10_surface": (state_a["bp_conversion_last10_surface"] - state_b["bp_conversion_last10_surface"]),
        "delta_days_since_last_match_surface": (state_a["days_since_last_match_surface"] - state_b["days_since_last_match_surface"]),
        "delta_matches_last_30_days_surface": (state_a["matches_last_30_days_surface"] - state_b["matches_last_30_days_surface"]),
        "delta_h2h": delta_h2h,

        "is_clay": int(surface == "Clay") if surface is not None else pd.NA,
        "is_grass": int(surface == "Grass") if surface is not None else pd.NA,
        "round_ordinal": round_ordinal,
    }

    return pd.DataFrame([row])



def predict_match_probability(project_root: Path, tour: str, requested_player_a: str, requested_player_b: str,
    match_date: str, surface: str | None = None, round_name: str | None = None, best_of: int | None = None,
    source: str = "sackmann", model: str = "logit", ) -> dict:

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

    as_of_date = pd.Timestamp(match_date)
    surface = _normalize_surface(surface)

    long_df = _load_long_view(project_root, tour, source=source)

    resolved_a = _resolve_player(long_df, requested_player_a, as_of_date)
    resolved_b = _resolve_player(long_df, requested_player_b, as_of_date)

    state_req_a = _build_player_state(long_df, resolved_a, as_of_date, surface)
    state_req_b = _build_player_state(long_df, resolved_b, as_of_date, surface)

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

    delta_h2h = _compute_h2h_delta_before_date(
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
        delta_h2h=delta_h2h,
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
        xgb_assets = load_xgb_prediction_artifacts(project_root, tour, source, surface)

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
        "chosen_calibration_method": chosen_calibration_method,
        "match_date": str(as_of_date.date()),
        "surface": surface,
        "round": round_name,
        "best_of": best_of,
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

      

def _get_xgb_artifact_paths(project_root: Path, tour: str, source: str, surface: str | None) -> dict[str, Path]:
    suffix = _build_artifact_suffix(source=source, surface=surface)
    models_dir = project_root / "data" / "models"

    return {
        "model": models_dir / f"{tour}_xgb_baseline{suffix}.joblib",
        "metrics": models_dir / f"{tour}_xgb_baseline{suffix}_metrics.json",
        "metadata": models_dir / f"{tour}_xgb_baseline{suffix}_meta.json",
        "isotonic_calibrator": models_dir / f"{tour}_xgb_baseline{suffix}_isotonic_calibrator.joblib",
        "sigmoid_calibrator": models_dir / f"{tour}_xgb_baseline{suffix}_sigmoid_calibrator.joblib",
    }


def load_xgb_prediction_artifacts(project_root: Path, tour: str, source: str, surface: str | None) -> dict:

    paths = _get_xgb_artifact_paths(project_root, tour, source, surface)
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