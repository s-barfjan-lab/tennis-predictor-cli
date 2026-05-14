from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from tennis_cli.config import Paths, get_settings


# Keep this separate from Sackmann.
# Do NOT overwrite your original processed ATP parquet.
OUTPUT_FILENAME = "atp_matches_tml_2015_2025.parquet"


# These are the columns we want in the normalized output.
# They are based on the Sackmann-style match schema my project already uses.
TARGET_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]


# If TML uses slightly different names, we try to map them.
# First match found wins.
COLUMN_CANDIDATES = {
    "tourney_id": ["tourney_id", "tournament_id"],
    "tourney_name": ["tourney_name", "tournament", "tournament_name"],
    "surface": ["surface"],
    "draw_size": ["draw_size", "draw"],
    "tourney_level": ["tourney_level", "level"],
    "tourney_date": ["tourney_date", "date"],
    "match_num": ["match_num", "match_id"],
    "winner_id": ["winner_id", "winner_player_id"],
    "winner_seed": ["winner_seed"],
    "winner_entry": ["winner_entry"],
    "winner_name": ["winner_name", "winner"],
    "winner_hand": ["winner_hand"],
    "winner_ht": ["winner_ht", "winner_height"],
    "winner_ioc": ["winner_ioc", "winner_country"],
    "winner_age": ["winner_age"],
    "loser_id": ["loser_id", "loser_player_id"],
    "loser_seed": ["loser_seed"],
    "loser_entry": ["loser_entry"],
    "loser_name": ["loser_name", "loser"],
    "loser_hand": ["loser_hand"],
    "loser_ht": ["loser_ht", "loser_height"],
    "loser_ioc": ["loser_ioc", "loser_country"],
    "loser_age": ["loser_age"],
    "score": ["score"],
    "best_of": ["best_of"],
    "round": ["round"],
    "minutes": ["minutes", "mins"],
    "w_ace": ["w_ace"],
    "w_df": ["w_df"],
    "w_svpt": ["w_svpt"],
    "w_1stIn": ["w_1stIn", "w_1stin"],
    "w_1stWon": ["w_1stWon", "w_1stwon"],
    "w_2ndWon": ["w_2ndWon", "w_2ndwon"],
    "w_SvGms": ["w_SvGms", "w_svgms"],
    "w_bpSaved": ["w_bpSaved", "w_bpsaved"],
    "w_bpFaced": ["w_bpFaced", "w_bpfaced"],
    "l_ace": ["l_ace"],
    "l_df": ["l_df"],
    "l_svpt": ["l_svpt"],
    "l_1stIn": ["l_1stIn", "l_1stin"],
    "l_1stWon": ["l_1stWon", "l_1stwon"],
    "l_2ndWon": ["l_2ndWon", "l_2ndwon"],
    "l_SvGms": ["l_SvGms", "l_svgms"],
    "l_bpSaved": ["l_bpSaved", "l_bpsaved"],
    "l_bpFaced": ["l_bpFaced", "l_bpfaced"],
    "winner_rank": ["winner_rank"],
    "winner_rank_points": ["winner_rank_points"],
    "loser_rank": ["loser_rank"],
    "loser_rank_points": ["loser_rank_points"],
}


CORE_REQUIRED_COLUMNS = [
    "tourney_date",
    "surface",
    "winner_name",
    "loser_name",
    "score",
    "best_of",
    "round",
]


def _read_csv_flexible(path: Path, nrows: int | None = None) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "latin1", "cp1252"]

    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not read {path} with tested encodings. Last error: {last_error}")


def _find_source_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _normalize_one_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = pd.DataFrame()

    for target_col in TARGET_COLUMNS:
        source_col = _find_source_column(df, COLUMN_CANDIDATES.get(target_col, [target_col]))
        if source_col is not None:
            out[target_col] = df[source_col]
        else:
            out[target_col] = pd.NA

    # Keep only completed matches with winner and loser names
    out = out.dropna(subset=["winner_name", "loser_name"])

    # Add source tracking
    out["data_source"] = "tml"
    out["season"] = year

    return out


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "draw_size",
        "match_num",
        "winner_seed",
        "winner_ht",
        "winner_age",
        "loser_seed",
        "loser_ht",
        "loser_age",
        "best_of",
        "minutes",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
        "winner_rank",
        "winner_rank_points",
        "loser_rank",
        "loser_rank_points",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tourney_date" in df.columns:
        # Keep same spirit as previous pipeline: numeric-like YYYYMMDD
        df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce").astype("Int64")

    return df


def _filter_matches(df: pd.DataFrame, drop_walkovers: bool, drop_retirements: bool) -> pd.DataFrame:
    out = df.copy()

    if drop_walkovers and "score" in out.columns:
        out = out[~out["score"].astype(str).str.contains("W/O", case=False, na=False)]

    if drop_retirements and "score" in out.columns:
        out = out[~out["score"].astype(str).str.contains("RET", case=False, na=False)]

    return out


def _parse_tourney_dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    cleaned = values.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def _filter_to_reference_date_range(df: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    if not reference_path.exists():
        return df

    reference_dates = _parse_tourney_dates(pd.read_parquet(reference_path, columns=["tourney_date"])["tourney_date"])
    reference_min = reference_dates.min()
    reference_max = reference_dates.max()

    if pd.isna(reference_min) or pd.isna(reference_max):
        raise ValueError(f"Reference dataset has invalid tourney_date values: {reference_path}")

    out_dates = _parse_tourney_dates(df["tourney_date"])
    if out_dates.isna().any():
        bad_count = int(out_dates.isna().sum())
        raise ValueError(f"TML dataset has {bad_count} invalid tourney_date values after normalization.")

    return df[(out_dates >= reference_min) & (out_dates <= reference_max)].copy()



"This was added later due to bug"
def _fill_missing_match_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "match_num" not in df.columns:
        df["match_num"] = pd.NA

    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce")

    missing_mask = df["match_num"].isna()

    if missing_mask.any():
        # We create a fallback sequential number inside each tournament.
        # +100000 avoids collision with normal match numbers.
        df.loc[missing_mask, "match_num"] = (df.loc[missing_mask]
            .groupby(["season", "tourney_id"], sort=False).cumcount().add(100000))

    df["match_num"] = df["match_num"].astype("Int64")

    return df




def build_tml_dataset(paths: Paths, start_year: int = 2015, end_year: int = 2025) -> Path:
    settings = get_settings()
    repo_dir = paths.tml_dir / "TML-Database"

    if not repo_dir.exists():
        raise FileNotFoundError(f"TML repo not found: {repo_dir}")

    yearly_frames: list[pd.DataFrame] = []

    print("\n" + "=" * 80)
    print("BUILDING TML NORMALIZED ATP DATASET")
    print("=" * 80)
    print(f"Source repo: {repo_dir}")
    print(f"Years: {start_year} -> {end_year}")

    for year in range(start_year, end_year + 1):
        year_file = repo_dir / f"{year}.csv"

        if not year_file.exists():
            print(f"[WARNING] Missing file: {year_file.name}")
            continue

        raw_df = _read_csv_flexible(year_file)
        normalized_df = _normalize_one_year(raw_df, year=year)

        missing_core = [c for c in CORE_REQUIRED_COLUMNS if c not in normalized_df.columns]
        if missing_core:
            raise ValueError(f"{year_file.name} missing required normalized columns: {missing_core}")

        print(
            f"{year_file.name} -> raw_rows={len(raw_df):,}, "
            f"normalized_rows={len(normalized_df):,}, "
            f"raw_cols={len(raw_df.columns)}"
        )

        yearly_frames.append(normalized_df)

    if not yearly_frames:
        raise ValueError("No yearly TML files were processed successfully.")

    final_df = pd.concat(yearly_frames, ignore_index=True)
    final_df = _coerce_types(final_df)
    final_df = _filter_matches(
        final_df,
        drop_walkovers=settings.drop_walkovers,
        drop_retirements=settings.drop_retirements,
    )
    reference_path = paths.processed_dir / f"atp_matches_{settings.year_min}_{settings.year_max}.parquet"
    final_df = _filter_to_reference_date_range(final_df, reference_path)
    final_df = _fill_missing_match_num(final_df)

    # Optional light cleanup: sort by date when possible
    if "tourney_date" in final_df.columns:
        final_df = final_df.sort_values("tourney_date", kind="stable").reset_index(drop=True)

    output_path = paths.processed_dir / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    print("\n" + "=" * 80)
    print("TML DATASET BUILT SUCCESSFULLY")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Final rows: {len(final_df):,}")
    print(f"Final columns: {len(final_df.columns)}")
    print(f"Unique seasons: {sorted(final_df['season'].dropna().unique().tolist())}")

    return output_path
