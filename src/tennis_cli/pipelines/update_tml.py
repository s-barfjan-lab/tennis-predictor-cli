from __future__ import annotations

import subprocess
from pathlib import Path

from tennis_cli.config import Paths

TML_REPO_URL = "https://github.com/Tennismylife/TML-Database.git"


def _run_git_command(args: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(args, cwd=str(cwd) if cwd is not None else None,
        check=False, capture_output=True, text=True, )
    if completed.returncode != 0:
        raise RuntimeError("Git command failed.\n"
            f"Command: {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}")


def _validate_tml_repo(repo_dir: Path) -> None:
    required_files = [
        repo_dir / "ATP_Database.csv",
        repo_dir / "2015.csv",
        repo_dir / "2016.csv",
        repo_dir / "2017.csv",
        repo_dir / "2018.csv",
        repo_dir / "2019.csv",
        repo_dir / "2020.csv",
        repo_dir / "2021.csv",
        repo_dir / "2022.csv",
        repo_dir / "2023.csv",
        repo_dir / "2024.csv",
        repo_dir / "2025.csv",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("TML repository validation failed. Missing expected files:\n" + "\n".join(missing))


def update_tml_repo(paths: Paths) -> Path:
    """
    Clone or refresh the TML repository into its own raw folder.
    Does not modify processed/features/models artifacts.
    """
    paths.tml_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = paths.tml_dir / "TML-Database"

    if not repo_dir.exists():
        _run_git_command(["git", "clone", TML_REPO_URL, str(repo_dir)], cwd=paths.tml_dir, )
    else:
        _run_git_command(["git", "pull"], cwd=repo_dir)

    _validate_tml_repo(repo_dir)
    return repo_dir