# interface layer
import typer
from rich.console import Console
from . import __app_name__, __version__
from pathlib import Path
from .config import get_paths, get_settings
from .pipelines.update_data import update_sackmann_data
from .pipelines.build_datasets import build_tour_dataset, explore_dataset
from .pipelines.build_features import (build_long_view_artifact, build_baseline_feature_artifact,)

from .pipelines.train_model import train_logistic_for_tour
from .pipelines.predict_match import predict_match_for_tour

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
def train_cmd(
    tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
    model: str = typer.Option("logit", "--model", help="Model type: currently only logit"),
):
    """
    Train a Phase 3 baseline model and save artifacts to data/models/.
    """
    tour = tour.lower().strip()
    model = model.lower().strip()

    if tour not in {"atp", "wta"}:
        raise typer.BadParameter("tour must be 'atp' or 'wta'")

    if model != "logit":
        raise typer.BadParameter("Only 'logit' is currently implemented")

    paths = get_paths()
    result = train_logistic_for_tour(paths.project_root, tour)

    console.print(f"[bold green]Training completed for[/] {tour.upper()}")
    console.print(f"[bold green]Model:[/] {result['model_path']}")
    console.print(f"[bold green]Metrics JSON:[/] {result['metrics_path']}")
    console.print(f"[bold green]Metrics CSV:[/] {result['metrics_table_path']}")
    console.print(f"[bold green]Coefficients:[/] {result['coef_path']}")
    console.print(f"[bold green]Metadata:[/] {result['metadata_path']}")

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
):
    """
    Predict match winner probability for two players on a given date.
    """
    tour = tour.lower().strip()

    if tour not in {"atp", "wta"}:
        raise typer.BadParameter("tour must be 'atp' or 'wta'")

    paths = get_paths()
    result = predict_match_for_tour(project_root=paths.project_root, tour=tour,
        player_a=player_a, player_b=player_b, match_date=date, )

    console.print(f"[bold green]Prediction completed for[/] {tour.upper()}")
    console.print(f"[bold]Date:[/] {result['match_date']}")
    console.print(f"[bold]Requested matchup:[/] {result['requested_player_a']} vs {result['requested_player_b']}")
    console.print(f"[bold]Canonical matchup:[/] {result['canonical_player_a']} vs {result['canonical_player_b']}")

    console.print("")
    console.print(f"[bold cyan]{result['requested_player_a']} win probability:[/] "
        f"{result['prob_requested_player_a_win']:.4f}")
    console.print(f"[bold cyan]{result['requested_player_b']} win probability:[/] "
        f"{result['prob_requested_player_b_win']:.4f}")

    snapshot = result["feature_snapshot"]
    console.print("")
    console.print("[bold magenta]Key feature snapshot used:[/]")
    console.print(f"delta_elo: {snapshot.get('delta_elo')}")
    console.print(f"delta_rank_adv: {snapshot.get('delta_rank_adv')}")
    console.print(f"delta_win_rate_last10: {snapshot.get('delta_win_rate_last10')}")
    console.print(f"delta_serve_win_pct_last10: {snapshot.get('delta_serve_win_pct_last10')}")





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
def build_features_cmd(
    tour: str = typer.Option(..., "--tour", help="Tour: atp or wta"),
    track: str = typer.Option("player", "--track", help="Feature track: player or baseline"),
):
    """
    Build feature artifacts.
    """
    track = track.lower().strip()

    if track == "player":
        output_path = build_long_view_artifact(tour)
    elif track == "baseline":
        output_path = build_baseline_feature_artifact(tour)
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
