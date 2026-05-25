from __future__ import annotations

import subprocess
from pathlib import Path

from tennis_cli.config import Paths


TENNIS_ABSTRACT_PBP_REPO_URL = "https://github.com/JeffSackmann/tennis_MatchChartingProject.git"
TENNIS_ABSTRACT_PBP_FOLDER = "tennis_abstract_pbp"
MATCH_CHARTING_REPO_FOLDER = "tennis_MatchChartingProject"


def tennis_abstract_pbp_raw_dir(paths: Paths) -> Path:
    return paths.raw_dir / TENNIS_ABSTRACT_PBP_FOLDER


def tennis_abstract_match_charting_repo_dir(paths: Paths) -> Path:
    return tennis_abstract_pbp_raw_dir(paths) / MATCH_CHARTING_REPO_FOLDER


def _run_git_command(args: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Git command failed.\n"
            f"Command: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _validate_match_charting_repo(repo_dir: Path) -> None:
    required_files = [
        "charting-m-matches.csv",
        "charting-w-matches.csv",
    ]
    missing = [filename for filename in required_files if not (repo_dir / filename).exists()]
    if not list(repo_dir.glob("charting-m-points*.csv")):
        missing.append("charting-m-points*.csv")
    if not list(repo_dir.glob("charting-w-points*.csv")):
        missing.append("charting-w-points*.csv")
    if missing:
        raise FileNotFoundError(
            "Tennis Abstract Match Charting repo validation failed. Missing files: "
            + ", ".join(missing)
        )


def update_tennis_abstract_pbp_repo(paths: Paths) -> Path:
    """
    Clone or refresh Jeff Sackmann's Tennis Abstract Match Charting Project.

    This writes only under `data/raw/tennis_abstract_pbp/` and deliberately
    avoids the existing Sackmann/TML raw folders and all model artifacts.
    """
    raw_dir = tennis_abstract_pbp_raw_dir(paths)
    raw_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = tennis_abstract_match_charting_repo_dir(paths)
    if not repo_dir.exists():
        _run_git_command(["git", "clone", TENNIS_ABSTRACT_PBP_REPO_URL, str(repo_dir)], cwd=raw_dir)
    else:
        _run_git_command(["git", "pull"], cwd=repo_dir)

    _validate_match_charting_repo(repo_dir)
    return repo_dir
