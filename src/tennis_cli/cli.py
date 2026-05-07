# interface layer
import typer
from rich.console import Console
from . import __app_name__, __version__
from pathlib import Path
from .config import get_paths, get_settings
from .pipelines.update_data import update_sackmann_data
from .pipelines.build_datasets import build_tour_dataset, explore_dataset
from .pipelines.build_features import (build_long_view_artifact, build_baseline_feature_artifact,)

from .pipelines.train_model import (train_logistic_for_tour, train_tuned_logistic_for_tour, train_xgb_for_tour, )
from .pipelines.train_model import (train_logistic_for_tour, train_tuned_logistic_for_tour, train_xgb_for_tour, train_tuned_xgb_for_tour, )
from .pipelines.predict_match import predict_match_for_tour

from tennis_cli.pipelines.update_tml import update_tml_repo
from tennis_cli.pipelines.inspect_tml import inspect_tml_repo
from tennis_cli.pipelines.build_tml_dataset import build_tml_dataset


# Create a Typer app
app = typer.Typer(help="Tennis predictor CLI (Phase 0 skeleton)")

# Create a Rich console for pretty output
console = Console()


# This function runs before any command, defines a --version/-v, flag Default value is False
@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(None, "--version", "-v", 
        help="Show the tennis-cli version and exit.", callback=None,)):
    """
    Main entry point for the tennis CLI.
    """
    if version:
        console.print(f"[bold green]{__app_name__}[/] version [cyan]{__version__}[/]")
        raise typer.Exit() # this prevents Typer from expecting a command
    

# just for testing that the CLI works
@app.command("hello")
def hello(name: str = "world"):
    """
    Test command: say hello to someone.
    """
    console.print(f"🎾 Hello, [bold]{name}[/]! Welcome to the tennis CLI (Phase 0).")





@app.command("train")
def train_cmd(tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
    model: str = typer.Option("logit", "--model", help="Model type: logit, logit_tuned, xgb, or xgb_tuned"),
    source: str = typer.Option("sackmann", "--source", help="Source: sackmann or tml"),
    surface: str = typer.Option(None, "--surface", help="Optional surface filter: Hard, Clay, or Grass"), ):
    """
    Train a baseline model and save artifacts to data/models/.
    """
    tour = tour.lower().strip()
    model = model.lower().strip()
    source = source.lower().strip()
    surface = None if surface is None else surface.strip().title()

    if tour not in {"atp", "wta"}:
        raise typer.BadParameter("tour must be 'atp' or 'wta'")

    if model not in {"logit", "logit_tuned", "xgb", "xgb_tuned"}:
        raise typer.BadParameter("model must be one of: 'logit', 'logit_tuned', 'xgb', 'xgb_tuned'")

    if source not in {"sackmann", "tml"}:
        raise typer.BadParameter("source must be 'sackmann' or 'tml'")

    if source == "tml" and tour != "atp":
        raise typer.BadParameter("TML source is currently supported only for ATP")
    
    if surface is not None and surface not in {"Hard", "Clay", "Grass"}:
        raise typer.BadParameter("surface must be one of: Hard, Clay, Grass")   


    paths = get_paths()


    if model == "logit":
        result = train_logistic_for_tour(paths.project_root, tour, source=source, surface=surface, )
    elif model == "logit_tuned":
        result = train_tuned_logistic_for_tour(paths.project_root, tour, source=source, surface=surface, )
    elif model == "xgb":
        result = train_xgb_for_tour(paths.project_root, tour, source=source, surface=surface, )
    else:
        result = train_tuned_xgb_for_tour(paths.project_root, tour, source=source, surface=surface, )

    console.print(f"[bold green]Training completed for[/] {tour.upper()}")
    console.print(f"[bold]Source:[/] {source}")
    console.print(f"[bold]Surface filter:[/] {result['surface_filter']}")
    console.print(f"[bold]Model type:[/] {model}")
    if model in {"logit_tuned", "xgb_tuned"}:
        console.print(f"[bold]Best params:[/] {result['best_params']}")
    if model in {"xgb", "xgb_tuned"}:
        console.print(f"[bold]Chosen calibration:[/] {result['chosen_calibration_method']}")
    console.print(f"[bold green]Model:[/] {result['model_path']}")
    console.print(f"[bold green]Metrics JSON:[/] {result['metrics_path']}")
    console.print(f"[bold green]Metrics CSV:[/] {result['metrics_table_path']}")
    console.print(f"[bold green]Metadata:[/] {result['metadata_path']}")


    if model in {"logit_tuned", "xgb_tuned"}:
        console.print(f"[bold green]CV results:[/] {result['cv_results_path']}")

    if model in {"logit", "logit_tuned"}:
        console.print(f"[bold green]Coefficients:[/] {result['coef_path']}")
    else:
        console.print(f"[bold green]Feature importance:[/] {result['importance_path']}")
    

    console.print("\n[bold cyan]Validation metrics:[/]")
    console.print(result["validation_metrics"])

    console.print("\n[bold cyan]Test metrics:[/]")
    console.print(result["test_metrics"])



@app.command("predict-match")
def predict_match_cmd(
    tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
    player_a: str = typer.Option(..., "--player-a", help="First player name"),
    player_b: str = typer.Option(..., "--player-b", help="Second player name"),
    date: str = typer.Option(..., "--date", help="Match date in YYYY-MM-DD format"),
    surface: str = typer.Option(None, "--surface", help="Surface: Hard, Clay, Grass"),
    round_name: str = typer.Option(None, "--round", help="Round: RR, R128, R64, R32, R16, QF, SF, F"),
    best_of: int = typer.Option(None, "--best-of", help="Best of sets, e.g. 3 or 5"),
    tourney_level: str = typer.Option(None, "--tourney-level", help="Tournament level, e.g. G, M, A, C"),
    source: str = typer.Option("sackmann", "--source", help="Source: sackmann or tml"),
    model: str = typer.Option("logit", "--model", help="Model type: logit or xgb"),
):
    """
    Predict match winner probability for two players on a given date.
    """
    tour = tour.lower().strip()
    source = source.lower().strip()
    model = model.lower().strip()

    if tour not in {"atp", "wta"}:
        raise typer.BadParameter("tour must be 'atp' or 'wta'")

    if source not in {"sackmann", "tml"}:
        raise typer.BadParameter("source must be either 'sackmann' or 'tml'")

    if model not in {"logit", "xgb"}:
        raise typer.BadParameter("model must be either 'logit' or 'xgb'")

    if source == "tml" and tour != "atp":
        raise typer.BadParameter("TML source is currently supported only for ATP")

    paths = get_paths()
    result = predict_match_for_tour(
        project_root=paths.project_root,
        tour=tour,
        player_a=player_a,
        player_b=player_b,
        match_date=date,
        surface=surface,
        round_name=round_name,
        best_of=best_of,
        tourney_level=tourney_level,
        source=source,
        model=model,
    )

    console.print(f"[bold green]Prediction completed for[/] {tour.upper()}")
    console.print(f"[bold]Source:[/] {result['source']}")
    console.print(f"[bold]Model:[/] {result['model']}")
    if result["model"] == "xgb":
        console.print(f"[bold]Chosen calibration:[/] {result['chosen_calibration_method']}")
    console.print(f"[bold]Date:[/] {result['match_date']}")
    console.print(f"[bold]Surface:[/] {result['surface']}")
    console.print(f"[bold]Round:[/] {result['round']}")
    console.print(f"[bold]Best of:[/] {result['best_of']}")
    console.print(f"[bold]Tournament level:[/] {result['tourney_level']}")
    console.print(f"[bold]Requested matchup:[/] {result['requested_player_a']} vs {result['requested_player_b']}")
    console.print(f"[bold]Canonical matchup:[/] {result['canonical_player_a']} vs {result['canonical_player_b']}")

    console.print("")
    console.print(
        f"[bold cyan]{result['requested_player_a']} win probability:[/] "
        f"{result['prob_requested_player_a_win']:.4f}"
    )
    console.print(
        f"[bold cyan]{result['requested_player_b']} win probability:[/] "
        f"{result['prob_requested_player_b_win']:.4f}"
    )

    if result["model"] == "xgb" and result["raw_internal_prob_player_a_win"] is not None:
        console.print("")
        console.print(
            f"[bold yellow]Raw internal probability ({result['internal_player_a']}):[/] "
            f"{result['raw_internal_prob_player_a_win']:.4f}"
        )
        console.print(
            f"[bold yellow]Calibrated internal probability ({result['internal_player_a']}):[/] "
            f"{result['internal_prob_player_a_win']:.4f}"
        )

    snapshot = result["feature_snapshot"]
    console.print("")
    console.print("[bold magenta]Key feature snapshot used:[/]")
    console.print(f"delta_rank_adv: {snapshot.get('delta_rank_adv')}")
    console.print(f"delta_elo: {snapshot.get('delta_elo')}")
    console.print(f"delta_surface_elo: {snapshot.get('delta_surface_elo')}")
    console.print(f"delta_h2h: {snapshot.get('delta_h2h')}")
    console.print(f"round_ordinal: {snapshot.get('round_ordinal')}")



@app.command("update-data")
def update_data():
    """
    Download/update local datasets (local-first).
    Currently: Jeff Sackmann ATP/WTA repos.
    """
    paths = get_paths()
    # calls the central path resolver
    update_sackmann_data(paths.sackmann_dir)  
    # calls the pipeline logic

@app.command("build-datasets")
def build_datasets(
    # enforces ATP/WTA separation at the CLI level -> Requires a --tour option (... means mandatory)
    tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
):
    """
    Build clean processed datasets (ATP/WTA separately).
    Outputs parquet files to data/processed/.
    """
    paths = get_paths()
    settings = get_settings()

    artifacts = build_tour_dataset(
        tour=tour,
        sackmann_root=paths.sackmann_dir,
        processed_dir=paths.processed_dir,
        year_min=settings.year_min,
        year_max=settings.year_max,
        drop_walkovers=settings.drop_walkovers,
        drop_retirements=settings.drop_retirements,
    )
    # CLI passes: what the user asked for (tour), where data is, what rules to apply
    console.print(f"[bold green]Saved:[/] {artifacts.matches_parquet}") 
    # This confirms -> success and exact output location

@app.command("explore-data")
def explore_data_cmd(
    tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
):
    """
    Print dataset sanity checks (counts, surfaces, missingness).
    """
    paths = get_paths()
    settings = get_settings()
    parquet_path = paths.processed_dir / f"{tour}_matches_{settings.year_min}_{settings.year_max}.parquet"
    explore_dataset(parquet_path) 



@app.command("build-features")
def build_features_cmd(tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
    track: str = typer.Option("player", "--track", help="Feature track: player or baseline"),
    source: str = typer.Option("sackmann", "--source", help="Source: sackmann or tml"), ):
    """
    Build feature artifacts.
    """
    tour = tour.lower().strip()
    track = track.lower().strip()
    source = source.lower().strip()

    if source not in {"sackmann", "tml"}:
        raise ValueError("source must be either 'sackmann' or 'tml'")

    if source == "tml" and tour != "atp":
        raise ValueError("TML source is currently supported only for ATP.")

    if track == "player":
        output_path = build_long_view_artifact(tour, source=source)
    elif track == "baseline":
        output_path = build_baseline_feature_artifact(tour, source=source)
    else:
        raise ValueError("track must be either 'player' or 'baseline'")

    console.print(f"[bold green]Saved:[/] {output_path}")



# This function allows multiple ways to launch the app (python -m tennis_cli)
def run():
    """
    Entry point used by `python -m tennis_cli`.
    """
    app()

# this allows the file to be run directly
if __name__ == "__main__":
    run()   



@app.command("update-tml")
def update_tml() -> None:
    paths = get_paths()
    repo_dir = update_tml_repo(paths)
    typer.echo(f"TML repo ready: {repo_dir}")



@app.command("inspect-tml")
def inspect_tml() -> None:
    paths = get_paths()
    inspect_tml_repo(paths)


@app.command("build-tml-dataset")
def build_tml_dataset_command() -> None:
    paths = get_paths()
    build_tml_dataset(paths)
