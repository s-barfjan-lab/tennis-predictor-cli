from __future__ import annotations

import pandas as pd

from tennis_cli.config import Paths
from tennis_cli.features.elo import EloConfig, compute_elo_features


def build_elo_for_tour(tour: str, paths: Paths) -> pd.DataFrame:
    """
    Load processed match data for one tour, compute Elo features,
    and save the Elo-enriched match table back to data/processed.
    """
    tour = tour.lower()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")

    input_path = paths.processed_dir / f"{tour}_matches_2015_2025.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Processed file not found: {input_path}")

    print(f"Loading processed matches from: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"Loaded {len(df):,} rows for {tour.upper()}")

    elo_df = compute_elo_features(df, config=EloConfig(initial_rating=1500.0,
            k_factor=32.0,date_col="tourney_date",),)

    print("Elo columns added:")
    print([
        "winner_elo_pre",
        "loser_elo_pre",
        "winner_elo_post",
        "loser_elo_post",
        "elo_diff_pre",
        "elo_prob_winner_pre",
    ])

    elo_df.to_parquet(input_path, index=False)
    print(f"Saved Elo-enriched dataset to: {input_path}")

    return elo_df