from __future__ import annotations

from pathlib import Path

from tennis_cli.models.inference import predict_match_probability


def predict_match_for_tour(project_root: Path, tour: str, player_a: str, player_b: str, match_date: str, ) -> dict:
    """
    Pipeline wrapper for baseline match prediction.
    """
    return predict_match_probability(
        project_root=project_root,
        tour=tour,
        requested_player_a=player_a,
        requested_player_b=player_b,
        match_date=match_date,
    )