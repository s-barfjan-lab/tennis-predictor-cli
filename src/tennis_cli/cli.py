import typer
from rich.console import Console
from . import __app_name__, __version__

# Create a Typer app
app = typer.Typer(help="Tennis predictor CLI (Phase 0 skeleton)")

# Create a Rich console for pretty output
console = Console()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the tennis-cli version and exit.",
        callback=None,
    )
):
    """
    Main entry point for the tennis CLI.
    """
    if version:
        console.print(f"[bold green]{__app_name__}[/] version [cyan]{__version__}[/]")
        raise typer.Exit()
    

@app.command("hello")
def hello(name: str = "world"):
    """
    Test command: say hello to someone.
    """
    console.print(f"🎾 Hello, [bold]{name}[/]! Welcome to the tennis CLI (Phase 0).")


@app.command("explore-data")
def explore_data():
    """
    Placeholder for data exploration.

    For now it just prints a message.
    Later it will call your real data exploration code (pandas, etc.).
    """
    console.print("[yellow]Data exploration is not implemented yet.[/]")
    console.print("In Phase 1, we'll connect this command to your real code.")


@app.command("train-baseline")
def train_baseline():
    """
    Placeholder for training a baseline model.
    """
    console.print("[magenta]Baseline training is not implemented yet.[/]")
    console.print("In a later phase, this will train a logistic regression model.") 


def run():
    """
    Entry point used by `python -m tennis_cli`.
    """
    app()

if __name__ == "__main__":
    run()       