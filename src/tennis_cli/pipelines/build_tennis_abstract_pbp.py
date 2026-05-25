from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from tennis_cli.config import Paths
from tennis_cli.pipelines.update_tennis_abstract_pbp import (
    tennis_abstract_match_charting_repo_dir,
)


TENNIS_ABSTRACT_PBP_FOLDER = "tennis_abstract_pbp"
POINTS_OUTPUT_FILENAME = "match_charting_points.parquet"
SNAPSHOTS_OUTPUT_FILENAME = "inplay_markov_snapshots.parquet"

POINT_SCORE_MAP = {
    "0": 0,
    "LOVE": 0,
    "15": 1,
    "30": 2,
    "40": 3,
    "A": 4,
    "AD": 4,
    "ADV": 4,
}
NORMAL_POINT_LABELS = {"0", "15", "30", "40", "A", "AD", "ADV"}


@dataclass(frozen=True)
class TennisAbstractPbpArtifacts:
    processed_points_path: Path
    feature_snapshots_path: Path
    rows: int
    matches: int


def tennis_abstract_processed_dir(paths: Paths) -> Path:
    return paths.processed_dir / TENNIS_ABSTRACT_PBP_FOLDER


def tennis_abstract_features_dir(paths: Paths) -> Path:
    return paths.features_dir / TENNIS_ABSTRACT_PBP_FOLDER


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _read_point_files(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = _read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No Tennis Abstract point files were found.")
    return pd.concat(frames, ignore_index=True)


def _parse_date(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def _normalize_match_metadata(matches_df: pd.DataFrame, tour: str) -> pd.DataFrame:
    required = ["match_id", "Player 1", "Player 2", "Date", "Tournament", "Round", "Surface", "Best of", "Final TB?"]
    missing = [col for col in required if col not in matches_df.columns]
    if missing:
        raise ValueError(f"Match metadata missing required columns: {missing}")

    out = matches_df[required].copy()
    out = out.rename(
        columns={
            "Player 1": "player1_name",
            "Player 2": "player2_name",
            "Date": "match_date",
            "Tournament": "tournament",
            "Round": "round",
            "Surface": "surface",
            "Best of": "best_of",
            "Final TB?": "final_tiebreak_rule",
        }
    )
    out["tour"] = tour
    out["match_date"] = _parse_date(out["match_date"])
    out["best_of"] = pd.to_numeric(out["best_of"], errors="coerce").fillna(3).astype(int)
    out["surface"] = out["surface"].astype(str).str.strip().str.title()
    out.loc[~out["surface"].isin({"Hard", "Clay", "Grass", "Carpet"}), "surface"] = "Unknown"
    return out


def _normalize_point_score_token(token: object) -> str:
    if pd.isna(token):
        return "0"
    text = str(token).strip().upper()
    return text if text else "0"


def split_point_score(score: object) -> tuple[str, str]:
    """
    Split Tennis Abstract's `Pts` field into first/second point labels.

    In Match Charting point rows, the score is recorded in tennis score order
    for the current point. Downstream code converts it into Player1/Player2
    order using the current server.
    """
    if pd.isna(score):
        return "0", "0"
    text = str(score).strip().upper()
    if not text:
        return "0", "0"
    parts = text.split("-")
    if len(parts) != 2:
        return "0", "0"
    return _normalize_point_score_token(parts[0]), _normalize_point_score_token(parts[1])


def split_point_score_player_order(score: object, server_player: object) -> tuple[str, str]:
    first, second = split_point_score(score)
    server = pd.to_numeric(pd.Series([server_player]), errors="coerce").iloc[0]
    if pd.isna(server) or int(server) == 1:
        return first, second
    return second, first


def _point_count(token: object, tiebreak: bool) -> int:
    text = _normalize_point_score_token(token)
    if not tiebreak and text in POINT_SCORE_MAP:
        return POINT_SCORE_MAP[text]
    if text in POINT_SCORE_MAP and text not in {"A", "AD", "ADV"}:
        return POINT_SCORE_MAP[text]
    return int(pd.to_numeric(pd.Series([text]), errors="coerce").fillna(0).iloc[0])


def _truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().upper()
    return text in {"1", "TRUE", "T", "YES", "Y"}


def _point_winner(value: object) -> int | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    winner = int(parsed)
    return winner if winner in {1, 2} else None


def _wins_normal_game(points_1: int, points_2: int, winner: int) -> bool:
    if winner == 1:
        points_1 += 1
    else:
        points_2 += 1
    return (points_1 >= 4 or points_2 >= 4) and abs(points_1 - points_2) >= 2


def _wins_tiebreak(points_1: int, points_2: int, winner: int) -> bool:
    if winner == 1:
        points_1 += 1
    else:
        points_2 += 1
    return (points_1 >= 7 or points_2 >= 7) and abs(points_1 - points_2) >= 2


def _set_winner_after_game(games_1: int, games_2: int, game_winner: int, tiebreak: bool) -> int | None:
    if game_winner == 1:
        games_1 += 1
    else:
        games_2 += 1

    if tiebreak:
        if games_1 == 7 and games_2 == 6:
            return 1
        if games_2 == 7 and games_1 == 6:
            return 2

    if games_1 >= 6 and games_1 - games_2 >= 2:
        return 1
    if games_2 >= 6 and games_2 - games_1 >= 2:
        return 2
    return None


def _match_winner_from_final_point(row: pd.Series) -> int | None:
    winner = _point_winner(row.get("PtWinner"))
    if winner is None:
        return None

    tiebreak = bool(int(pd.to_numeric(pd.Series([row.get("TB?")]), errors="coerce").fillna(0).iloc[0]))
    point_1, point_2 = split_point_score_player_order(row.get("Pts"), row.get("Svr"))
    points_1 = _point_count(point_1, tiebreak=tiebreak)
    points_2 = _point_count(point_2, tiebreak=tiebreak)

    game_won = _wins_tiebreak(points_1, points_2, winner) if tiebreak else _wins_normal_game(points_1, points_2, winner)
    if not game_won:
        return None

    games_1 = int(pd.to_numeric(pd.Series([row.get("Gm1")]), errors="coerce").fillna(0).iloc[0])
    games_2 = int(pd.to_numeric(pd.Series([row.get("Gm2")]), errors="coerce").fillna(0).iloc[0])
    set_winner = _set_winner_after_game(games_1, games_2, winner, tiebreak=tiebreak)
    if set_winner is None:
        return None

    sets_1 = int(pd.to_numeric(pd.Series([row.get("Set1")]), errors="coerce").fillna(0).iloc[0])
    sets_2 = int(pd.to_numeric(pd.Series([row.get("Set2")]), errors="coerce").fillna(0).iloc[0])
    if set_winner == 1:
        sets_1 += 1
    else:
        sets_2 += 1

    best_of = int(pd.to_numeric(pd.Series([row.get("best_of")]), errors="coerce").fillna(3).iloc[0])
    sets_to_win = (best_of // 2) + 1
    if sets_1 >= sets_to_win:
        return 1
    if sets_2 >= sets_to_win:
        return 2
    return None


def _match_winner_labels(points_df: pd.DataFrame) -> pd.Series:
    labels: dict[object, int | None] = {}
    for match_id, grp in points_df.groupby("match_id", sort=False):
        last_row = grp.sort_values("point_index").iloc[-1]
        labels[match_id] = _match_winner_from_final_point(last_row)
    return points_df["match_id"].map(labels)


def _prior_cumsum_by_match(df: pd.DataFrame, current_col: str) -> pd.Series:
    current = df[current_col].astype(int)
    cumsum = current.groupby(df["match_id"], sort=False).cumsum()
    return cumsum.groupby(df["match_id"], sort=False).shift(fill_value=0).astype(int)


def build_tennis_abstract_snapshots_from_frames(
    points_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    tour: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Build leakage-safe in-play snapshots from Match Charting point data.

    Snapshot inputs use the score state before each point. Live service counts
    are cumulative counts shifted by one point within each match, so the current
    point outcome is never included in the features for that point.
    """
    point_required = ["match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Svr", "PtWinner"]
    missing = [col for col in point_required if col not in points_df.columns]
    if missing:
        raise ValueError(f"Point data missing required columns: {missing}")

    matches = _normalize_match_metadata(matches_df, tour=tour)
    optional_cols = [col for col in ["TB?", "TbSet", "isSvrWinner", "source_file"] if col in points_df.columns]
    points = points_df[point_required + optional_cols].copy()
    points["point_index"] = pd.to_numeric(points["Pt"], errors="coerce")
    points = points.dropna(subset=["match_id", "point_index"]).copy()
    points["point_index"] = points["point_index"].astype(int)

    out = points.merge(matches, on="match_id", how="inner")
    if start_year is not None:
        out = out[out["match_date"].dt.year >= int(start_year)].copy()
    if end_year is not None:
        out = out[out["match_date"].dt.year <= int(end_year)].copy()

    if out.empty:
        raise ValueError("No Tennis Abstract point rows remain after filtering.")

    out = out.sort_values(["match_date", "match_id", "point_index"], kind="stable").reset_index(drop=True)

    out["sets_1"] = pd.to_numeric(out["Set1"], errors="coerce").fillna(0).astype(int)
    out["sets_2"] = pd.to_numeric(out["Set2"], errors="coerce").fillna(0).astype(int)
    out["games_1"] = pd.to_numeric(out["Gm1"], errors="coerce").fillna(0).astype(int)
    out["games_2"] = pd.to_numeric(out["Gm2"], errors="coerce").fillna(0).astype(int)
    out["server_player"] = pd.to_numeric(out["Svr"], errors="coerce").astype("Int64")
    out["server"] = out["server_player"].map({1: "A", 2: "B"})
    point_scores = [
        split_point_score_player_order(score, server)
        for score, server in zip(out["Pts"], out["server_player"])
    ]
    out["points_1"] = [score[0] for score in point_scores]
    out["points_2"] = [score[1] for score in point_scores]
    out["point_winner_player"] = pd.to_numeric(out["PtWinner"], errors="coerce").astype("Int64")
    out["label_player1_wins_point"] = (out["point_winner_player"] == 1).astype(int)

    if "TB?" in out.columns:
        out["is_tiebreak_point"] = pd.to_numeric(out["TB?"], errors="coerce").fillna(0).astype(int)
    else:
        tb_set = out["TbSet"].apply(_truthy) if "TbSet" in out.columns else pd.Series(False, index=out.index)
        normal_score = out["points_1"].isin(NORMAL_POINT_LABELS) & out["points_2"].isin(NORMAL_POINT_LABELS)
        tied_late_games = (out["games_1"] == out["games_2"]) & (out["games_1"] >= 6)
        numeric_tiebreak_like = ~normal_score | tied_late_games
        out["is_tiebreak_point"] = (tb_set & numeric_tiebreak_like).astype(int)
    out["TB?"] = out["is_tiebreak_point"]

    if "isSvrWinner" in out.columns:
        out["is_server_winner"] = pd.to_numeric(out["isSvrWinner"], errors="coerce").fillna(0).astype(int)
    else:
        out["is_server_winner"] = (out["server_player"] == out["point_winner_player"]).astype(int)

    out["_p1_service_point_current"] = (out["server_player"] == 1).astype(int)
    out["_p2_service_point_current"] = (out["server_player"] == 2).astype(int)
    out["_p1_service_point_won_current"] = (
        (out["server_player"] == 1) & (out["is_server_winner"] == 1)
    ).astype(int)
    out["_p2_service_point_won_current"] = (
        (out["server_player"] == 2) & (out["is_server_winner"] == 1)
    ).astype(int)

    out["p1_service_points_played_before"] = _prior_cumsum_by_match(out, "_p1_service_point_current")
    out["p1_service_points_won_before"] = _prior_cumsum_by_match(out, "_p1_service_point_won_current")
    out["p2_service_points_played_before"] = _prior_cumsum_by_match(out, "_p2_service_point_current")
    out["p2_service_points_won_before"] = _prior_cumsum_by_match(out, "_p2_service_point_won_current")

    winner = _match_winner_labels(out)
    out["match_winner_player"] = pd.to_numeric(winner, errors="coerce").astype("Int64")
    out["label_player1_win_match"] = (out["match_winner_player"] == 1).astype("Int64")
    out["snapshot_id"] = out["match_id"].astype(str) + "::" + out["point_index"].astype(str)

    keep_cols = [
        "snapshot_id",
        "match_id",
        "tour",
        "match_date",
        "tournament",
        "round",
        "surface",
        "best_of",
        "final_tiebreak_rule",
        "player1_name",
        "player2_name",
        "point_index",
        "source_file",
        "sets_1",
        "sets_2",
        "games_1",
        "games_2",
        "points_1",
        "points_2",
        "is_tiebreak_point",
        "server_player",
        "server",
        "p1_service_points_won_before",
        "p1_service_points_played_before",
        "p2_service_points_won_before",
        "p2_service_points_played_before",
        "point_winner_player",
        "label_player1_wins_point",
        "match_winner_player",
        "label_player1_win_match",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    out = out[keep_cols].copy()
    out = out.dropna(subset=["server"]).reset_index(drop=True)
    return out


def _iter_tour_files(repo_dir: Path, tours: Iterable[str]) -> Iterable[tuple[str, list[Path], Path]]:
    for tour in tours:
        tour_norm = tour.lower().strip()
        if tour_norm in {"m", "men", "atp"}:
            yield "atp", sorted(repo_dir.glob("charting-m-points*.csv")), repo_dir / "charting-m-matches.csv"
        elif tour_norm in {"w", "women", "wta"}:
            yield "wta", sorted(repo_dir.glob("charting-w-points*.csv")), repo_dir / "charting-w-matches.csv"
        else:
            raise ValueError("tours must contain only atp/wta or m/w")


def build_tennis_abstract_pbp_artifacts(
    paths: Paths,
    tours: Iterable[str] = ("atp", "wta"),
    start_year: int | None = 2015,
    end_year: int | None = 2025,
) -> TennisAbstractPbpArtifacts:
    """
    Build isolated Tennis Abstract point/snapshot artifacts.

    Outputs:
    - `data/processed/tennis_abstract_pbp/match_charting_points.parquet`
    - `data/features/tennis_abstract_pbp/inplay_markov_snapshots.parquet`
    """
    repo_dir = tennis_abstract_match_charting_repo_dir(paths)
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Tennis Abstract Match Charting repo not found: {repo_dir}. "
            "Run update-tennis-abstract-pbp first."
        )

    frames: list[pd.DataFrame] = []
    for tour, points_paths, matches_path in _iter_tour_files(repo_dir, tours):
        if not points_paths or not matches_path.exists():
            raise FileNotFoundError(f"Missing Tennis Abstract files for {tour}: {points_paths}, {matches_path}")
        frames.append(
            build_tennis_abstract_snapshots_from_frames(
                points_df=_read_point_files(points_paths),
                matches_df=_read_csv(matches_path),
                tour=tour,
                start_year=start_year,
                end_year=end_year,
            )
        )

    snapshots = pd.concat(frames, ignore_index=True)
    snapshots = snapshots.sort_values(["match_date", "match_id", "point_index"], kind="stable").reset_index(drop=True)

    processed_dir = tennis_abstract_processed_dir(paths)
    features_dir = tennis_abstract_features_dir(paths)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    processed_points_path = processed_dir / POINTS_OUTPUT_FILENAME
    feature_snapshots_path = features_dir / SNAPSHOTS_OUTPUT_FILENAME

    snapshots.to_parquet(processed_points_path, index=False)
    snapshots.to_parquet(feature_snapshots_path, index=False)

    return TennisAbstractPbpArtifacts(
        processed_points_path=processed_points_path,
        feature_snapshots_path=feature_snapshots_path,
        rows=int(len(snapshots)),
        matches=int(snapshots["match_id"].nunique()),
    )
