# interface layer
import typer
from rich.console import Console
from . import __app_name__, __version__
from pathlib import Path
from .config import get_paths, get_settings
from .pipelines.update_data import update_sackmann_data
from .pipelines.build_datasets import build_tour_dataset, explore_dataset


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


# This function should be completed in phase 2 -> now just documents future intent
@app.command("train-baseline")
def train_baseline():
    """
    Placeholder for training a baseline model.
    """
    console.print("[magenta]Baseline training is not implemented yet.[/]")
    console.print("In a later phase, this will train a logistic regression model.") 


# This function allows multiple ways to launch the app (python -m tennis_cli)
def run():
    """
    Entry point used by `python -m tennis_cli`.
    """
    app()

# this allows the file to be run directly
if __name__ == "__main__":
    run()   


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