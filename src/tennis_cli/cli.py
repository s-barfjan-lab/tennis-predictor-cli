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
from .pipelines.predict_inplay import predict_inplay_for_match
from .pipelines.update_tennis_abstract_pbp import update_tennis_abstract_pbp_repo
from .pipelines.build_tennis_abstract_pbp import build_tennis_abstract_pbp_artifacts

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
    model_variant: str = typer.Option("baseline", "--model-variant", help="Model variant: baseline or tuned"),
):
    """
    Predict match winner probability for two players on a given date.
    """
    tour = tour.lower().strip()
    source = source.lower().strip()
    model = model.lower().strip()
    model_variant = model_variant.lower().strip()

    if tour not in {"atp", "wta"}:
        raise typer.BadParameter("tour must be 'atp' or 'wta'")

    if source not in {"sackmann", "tml"}:
        raise typer.BadParameter("source must be either 'sackmann' or 'tml'")

    if model not in {"logit", "xgb"}:
        raise typer.BadParameter("model must be either 'logit' or 'xgb'")

    if model_variant not in {"baseline", "tuned"}:
        raise typer.BadParameter("model-variant must be either 'baseline' or 'tuned'")

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
        model_variant=model_variant,
    )

    console.print(f"[bold green]Prediction completed for[/] {tour.upper()}")
    console.print(f"[bold]Source:[/] {result['source']}")
    console.print(f"[bold]Model:[/] {result['model']}")
    console.print(f"[bold]Model variant:[/] {result['model_variant']}")
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


@app.command("predict-inplay")
def predict_inplay_cmd(
    sets_a: int = typer.Option(..., "--sets-a", help="Sets won by player A at prediction time"),
    sets_b: int = typer.Option(..., "--sets-b", help="Sets won by player B at prediction time"),
    games_a: int = typer.Option(..., "--games-a", help="Games won by player A in the current set"),
    games_b: int = typer.Option(..., "--games-b", help="Games won by player B in the current set"),
    points_a: str = typer.Option("0", "--points-a", help="Current point score for A: 0, 15, 30, 40, AD, or tiebreak points"),
    points_b: str = typer.Option("0", "--points-b", help="Current point score for B: 0, 15, 30, 40, AD, or tiebreak points"),
    server: str = typer.Option(..., "--server", help="Current point server: A or B"),
    p_a_serve_point: float = typer.Option(0.5, "--p-a-serve-point", help="Pre-match prior probability that A wins a point on serve"),
    p_b_serve_point: float = typer.Option(0.5, "--p-b-serve-point", help="Pre-match prior probability that B wins a point on serve"),
    best_of: int = typer.Option(3, "--best-of", help="Best-of format: 3 or 5"),
    a_service_points_won: int = typer.Option(0, "--a-service-points-won", help="A service points won so far in this match"),
    a_service_points_played: int = typer.Option(0, "--a-service-points-played", help="A service points played so far in this match"),
    b_service_points_won: int = typer.Option(0, "--b-service-points-won", help="B service points won so far in this match"),
    b_service_points_played: int = typer.Option(0, "--b-service-points-played", help="B service points played so far in this match"),
    prior_strength: float = typer.Option(48.0, "--prior-strength", help="Pseudo-service-points used to shrink live point rates toward the prior"),
):
    """
    Predict live match probability from current score state.
    """
    server = server.upper().strip()
    if server not in {"A", "B"}:
        raise typer.BadParameter("server must be either 'A' or 'B'")
    if best_of not in {3, 5}:
        raise typer.BadParameter("best-of must be either 3 or 5")

    result = predict_inplay_for_match(
        sets_a=sets_a,
        sets_b=sets_b,
        games_a=games_a,
        games_b=games_b,
        points_a=points_a,
        points_b=points_b,
        server=server,
        p_a_serve_point_prior=p_a_serve_point,
        p_b_serve_point_prior=p_b_serve_point,
        best_of=best_of,
        a_service_points_won=a_service_points_won,
        a_service_points_played=a_service_points_played,
        b_service_points_won=b_service_points_won,
        b_service_points_played=b_service_points_played,
        prior_strength=prior_strength,
    )

    console.print("[bold green]In-play Markov prediction completed[/]")
    console.print(f"[bold]Score:[/] sets {sets_a}-{sets_b}, games {games_a}-{games_b}, points {points_a}-{points_b}")
    console.print(f"[bold]Server:[/] {server}")
    console.print(f"[bold]Best of:[/] {best_of}")
    console.print(f"[bold]A serve-point prior/live:[/] {p_a_serve_point:.4f} -> {result['p_a_serve_point_live']:.4f}")
    console.print(f"[bold]B serve-point prior/live:[/] {p_b_serve_point:.4f} -> {result['p_b_serve_point_live']:.4f}")
    console.print("")
    console.print(f"[bold cyan]Player A win probability:[/] {result['prob_player_a_win']:.4f}")
    console.print(f"[bold cyan]Player B win probability:[/] {result['prob_player_b_win']:.4f}")



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


@app.command("update-tennis-abstract-pbp")
def update_tennis_abstract_pbp_command() -> None:
    """
    Clone/update Tennis Abstract Match Charting point-by-point data separately.
    """
    paths = get_paths()
    repo_dir = update_tennis_abstract_pbp_repo(paths)
    typer.echo(f"Tennis Abstract PBP repo ready: {repo_dir}")


@app.command("build-tennis-abstract-pbp")
def build_tennis_abstract_pbp_command(
    tour: str = typer.Option("both", "--tour", help="Tour to build: atp, wta, or both"),
    start_year: int = typer.Option(2015, "--start-year", help="First match year to keep"),
    end_year: int = typer.Option(2025, "--end-year", help="Last match year to keep"),
) -> None:
    """
    Build isolated Tennis Abstract point-by-point in-play snapshots.
    """
    tour = tour.lower().strip()
    if tour == "both":
        tours = ("atp", "wta")
    elif tour in {"atp", "wta"}:
        tours = (tour,)
    else:
        raise typer.BadParameter("tour must be one of: atp, wta, both")

    if start_year > end_year:
        raise typer.BadParameter("start-year must be <= end-year")

    paths = get_paths()
    artifacts = build_tennis_abstract_pbp_artifacts(
        paths=paths,
        tours=tours,
        start_year=start_year,
        end_year=end_year,
    )
    console.print("[bold green]Tennis Abstract PBP artifacts built separately[/]")
    console.print(f"[bold]Rows:[/] {artifacts.rows:,}")
    console.print(f"[bold]Matches:[/] {artifacts.matches:,}")
    console.print(f"[bold green]Processed points:[/] {artifacts.processed_points_path}")
    console.print(f"[bold green]Feature snapshots:[/] {artifacts.feature_snapshots_path}")
