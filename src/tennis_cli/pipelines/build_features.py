from __future__ import annotations

from pathlib import Path
import pandas as pd
from rich.console import Console

from ..config import get_paths, get_settings
from ..features.long_view import build_long_view, save_long_view
from ..features.rolling import add_rolling_features
from ..features.baseline_features import (build_baseline_match_table,save_baseline_match_table,)

console = Console()


def build_long_view_artifact(tour: str) -> Path:
    """
    Build and save the player-centric long-view dataset for a given tour.
    """
    paths = get_paths()
    settings = get_settings()

    tour = tour.lower().strip()
    processed_path = paths.processed_dir / f"{tour}_matches_{settings.year_min}_{settings.year_max}.parquet"

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_path}. "
            f"Run build-datasets first."
        )

    console.print(f"[bold]Loading processed dataset:[/] {processed_path}")
    matches_df = pd.read_parquet(processed_path)

    console.print(f"[bold]Building long-view dataset for[/] {tour.upper()}")
    long_df = build_long_view(matches_df, tour=tour)

    console.print("[bold]Adding rolling features...[/]")
    long_df = add_rolling_features(long_df)

    output_path = paths.features_dir / f"{tour}_long_{settings.year_min}_{settings.year_max}.parquet"
    save_long_view(long_df, output_path)

    console.print(
        f"[green]Done.[/] Long-view rows: {len(long_df):,} "
        f"(from {len(matches_df):,} match rows)."
    )

    return output_path



def build_baseline_feature_artifact(tour: str) -> Path:
    """
    Build and save the baseline match-row delta-feature dataset for a given tour.
    """
    paths = get_paths()
    settings = get_settings()

    tour = tour.lower().strip()
    long_path = paths.features_dir / f"{tour}_long_{settings.year_min}_{settings.year_max}.parquet"

    if not long_path.exists():
        raise FileNotFoundError(
            f"Long-view dataset not found: {long_path}. "
            f"Run build-features --track player first."
        )

    console.print(f"[bold]Loading long-view dataset:[/] {long_path}")
    long_df = pd.read_parquet(long_path)

    console.print(f"[bold]Building baseline delta table for[/] {tour.upper()}")
    baseline_df = build_baseline_match_table(long_df)

    output_path = paths.features_dir / f"{tour}_baseline_{settings.year_min}_{settings.year_max}.parquet"
    save_baseline_match_table(baseline_df, output_path)

    console.print(
        f"[green]Done.[/] Baseline rows: {len(baseline_df):,} "
        f"(from {len(long_df):,} long-view rows)."
    )

    return output_path