from __future__ import annotations

from pathlib import Path
import pandas as pd
from rich.console import Console

from ..config import get_paths, get_settings
from ..features.elo import compute_elo_features, compute_all_elo_features
from ..features.long_view import build_long_view, save_long_view
from ..features.rolling import add_rolling_features
from ..features.h2h import compute_h2h_features
from ..features.baseline_features import (build_baseline_match_table, save_baseline_match_table, )

console = Console()


"This was added later to fix the date problem"
def _ensure_real_tourney_date(df: pd.DataFrame) -> pd.DataFrame:
    if "tourney_date" not in df.columns:
        raise ValueError("Missing required column: tourney_date")

    if pd.api.types.is_datetime64_any_dtype(df["tourney_date"]):
        return df

    # Handle values like 20150104 or 20150104.0
    cleaned = (df["tourney_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip())

    df["tourney_date"] = pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce", )

    return df


def _get_processed_input_path(paths, settings, tour: str, source: str) -> Path:
    tour = tour.lower().strip()
    source = source.lower().strip()

    if source == "sackmann":
        return paths.processed_dir / f"{tour}_matches_{settings.year_min}_{settings.year_max}.parquet"

    if source == "tml":
        if tour != "atp":
            raise ValueError("TML source is currently supported only for ATP.")
        return paths.processed_dir / f"atp_matches_tml_{settings.year_min}_{settings.year_max}.parquet"

    raise ValueError(f"Unsupported source: {source}")


def _get_feature_output_path(paths, settings, tour: str, track: str, source: str) -> Path:
    tour = tour.lower().strip()
    track = track.lower().strip()
    source = source.lower().strip()

    if track not in {"player", "baseline"}:
        raise ValueError(f"Unsupported track: {track}")

    if source == "sackmann":
        if track == "player":
            return paths.features_dir / f"{tour}_long_{settings.year_min}_{settings.year_max}.parquet"
        return paths.features_dir / f"{tour}_baseline_{settings.year_min}_{settings.year_max}.parquet"

    if source == "tml":
        if tour != "atp":
            raise ValueError("TML source is currently supported only for ATP.")
        if track == "player":
            return paths.features_dir / f"atp_long_tml_{settings.year_min}_{settings.year_max}.parquet"
        return paths.features_dir / f"atp_baseline_tml_{settings.year_min}_{settings.year_max}.parquet"

    raise ValueError(f"Unsupported source: {source}")



def build_long_view_artifact(tour: str, source: str = "sackmann") -> Path:
    """
    Build and save the player-centric long-view dataset for a given tour.
    """
    paths = get_paths()
    settings = get_settings()

    tour = tour.lower().strip()
    source = source.lower().strip()

    processed_path = _get_processed_input_path(paths, settings, tour, source)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}. "
            f"Run the proper dataset build step first.")

    console.print(f"[bold]Loading processed dataset:[/] {processed_path}")
    console.print(f"[bold]Source:[/] {source}")
    matches_df = pd.read_parquet(processed_path)
    matches_df = _ensure_real_tourney_date(matches_df)

    console.print(f"[bold]Computing Elo features for[/] {tour.upper()}")
    matches_df = compute_all_elo_features(matches_df)

    console.print(f"[bold]Building long-view dataset for[/] {tour.upper()}")
    long_df = build_long_view(matches_df, tour=tour)

    console.print("[bold]Adding rolling features...[/]")
    long_df = add_rolling_features(long_df)

    console.print("[bold]Adding H2H features...[/]")
    long_df = compute_h2h_features(long_df)

    output_path = _get_feature_output_path(paths, settings, tour, "player", source)
    save_long_view(long_df, output_path)

    console.print(f"[green]Done.[/] Long-view rows: {len(long_df):,} "
        f"(from {len(matches_df):,} Elo-enriched match rows).")
    console.print(f"[bold]Saved to:[/] {output_path}")

    return output_path



def build_baseline_feature_artifact(tour: str, source: str = "sackmann") -> Path:
    """
    Build and save the baseline match-row delta-feature dataset for a given tour.
    """
    paths = get_paths()
    settings = get_settings()

    tour = tour.lower().strip()
    source = source.lower().strip()

    long_path = _get_feature_output_path(paths, settings, tour, "player", source)

    if not long_path.exists():
        raise FileNotFoundError(f"Long-view dataset not found: {long_path}. "
            f"Run build-features --track player first for this source.")

    console.print(f"[bold]Loading long-view dataset:[/] {long_path}")
    console.print(f"[bold]Source:[/] {source}")
    long_df = pd.read_parquet(long_path)

    console.print(f"[bold]Building baseline delta table for[/] {tour.upper()}")
    baseline_df = build_baseline_match_table(long_df)

    output_path = _get_feature_output_path(paths, settings, tour, "baseline", source)
    save_baseline_match_table(baseline_df, output_path)

    console.print(f"[green]Done.[/] Baseline rows: {len(baseline_df):,} "
        f"(from {len(long_df):,} long-view rows).")
    console.print(f"[bold]Saved to:[/] {output_path}")

    return output_path