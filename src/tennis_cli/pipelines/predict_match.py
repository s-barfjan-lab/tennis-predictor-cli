from __future__ import annotations

from pathlib import Path


from tennis_cli.models.inference import predict_match_probability



def predict_match_for_tour(project_root: Path, tour: str, player_a: str, player_b: str, match_date: str, surface: str | None = None,
    round_name: str | None = None, best_of: int | None = None, tourney_level: str | None = None,
    source: str = "sackmann", model: str = "logit", ) -> dict:
   
    return predict_match_probability(project_root=project_root, tour=tour, model=model, requested_player_a=player_a, requested_player_b=player_b,
        match_date=match_date, surface=surface, round_name=round_name, best_of=best_of, tourney_level=tourney_level,
        source=source, )

