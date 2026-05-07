from pathlib import Path
from tennis_cli.pipelines.train_model import run_seed_noise_floor_experiment

project_root = Path(r"C:\Users\LENOVO\Projects\tennis-predictor-cli")
seeds = [7, 17, 29, 42, 88]
half_life_days = 730

runs = [
    {"model": "xgb_tuned", "tour": "atp", "source": "sackmann", "surface": "Hard",  "search_profile": "base"},
    {"model": "logit",     "tour": "atp", "source": "sackmann", "surface": "Clay"},
    {"model": "xgb",       "tour": "atp", "source": "sackmann", "surface": "Grass"},
    {"model": "xgb",       "tour": "atp", "source": "tml",      "surface": "Hard"},
    {"model": "xgb",       "tour": "atp", "source": "tml",      "surface": "Clay"},
    {"model": "xgb",       "tour": "atp", "source": "tml",      "surface": "Grass"},
    {"model": "logit",     "tour": "wta", "source": "sackmann", "surface": "Hard"},
    {"model": "xgb",       "tour": "wta", "source": "sackmann", "surface": "Clay"},
    {"model": "logit",     "tour": "wta", "source": "sackmann", "surface": "Grass"},
]

for cfg in runs:
    kwargs = dict(
        project_root=project_root,
        model=cfg["model"],
        tour=cfg["tour"],
        source=cfg["source"],
        surface=cfg["surface"],
        seeds=seeds,
        half_life_days=half_life_days,
    )
    if "search_profile" in cfg:
        kwargs["search_profile"] = cfg["search_profile"]

    result = run_seed_noise_floor_experiment(**kwargs)
    print(result["summary_path"])
