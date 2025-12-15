import subprocess
from pathlib import Path
from rich.console import Console

console = Console()


# Runs a terminal command (like git clone, git pull)
def _run(cmd: list[str], cwd: Path | None = None) -> None: 
# cmd -> a list of words (the command), means “run this command inside this folder”
# It returns nothing (-> None)
    console.print(f"[dim]Running:[/] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    # check=True -> if the command fails -> crash immediately

# Git controller-> decides whether to download a repository or update it -> decision maker
# Given -> a GitHub repository URL & a destination folder
# Decides -> clone if not present OR pull if already present
def _clone_or_pull(repo_url: str, dest_dir: Path) -> None:
    if dest_dir.exists():
        console.print(f"[green]Updating[/] {dest_dir.name} (git pull)")
        _run(["git", "pull"], cwd=dest_dir)
    else:
        console.print(f"[green]Downloading[/] {dest_dir.name} (git clone)")
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        # parents=True -> create all missing folders, 
        # exist_ok=True -> don’t crash if it already exists
        _run(["git", "clone", repo_url, str(dest_dir)])


# The entry point that updates all Sackmann data (ATP + WTA)
def update_sackmann_data(sackmann_root: Path) -> None:
    atp_dir = sackmann_root / "tennis_atp"
    wta_dir = sackmann_root / "tennis_wta"

    _clone_or_pull("https://github.com/JeffSackmann/tennis_atp.git", atp_dir)
    _clone_or_pull("https://github.com/JeffSackmann/tennis_wta.git", wta_dir)

    console.print("[bold green]Done.[/] Sackmann ATP/WTA repos are up to date.")