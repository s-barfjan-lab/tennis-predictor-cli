from dataclasses import dataclass
from pathlib import Path

# =========================
# PATH CONFIGURATION
# To understand where files live on disk.
# Instead of every file guessing paths, they all ask this function.
# =========================

@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    sackmann_dir: Path
# Each one is a Path, not a string
# frozen=True -> once created, these paths cannot be changed by mistake (prevents bugs later)

def get_paths() -> Paths:
    """
    Centralized path resolver.
    Assumes this file lives in: src/tennis_cli/config.py
    """
    project_root = Path(__file__).resolve().parents[2] 
    # automatically finding the project root

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    sackmann_dir = raw_dir / "sackmann"

    return Paths(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        sackmann_dir=sackmann_dir,
    )


# =========================
# BUILD / FILTER SETTINGS
# This defines the rules the project uses when building datasets, in one single place.
# Instead of spreading decisions across many files, I put them here.
# =========================

@dataclass(frozen=True)
class BuildSettings:
    """
    Global dataset build settings.
    Change here to affect all pipelines.
    """
    drop_walkovers: bool = True
    drop_retirements: bool = True
# Walkover = a match that was never played (injury before first ball)
# Retirement = match started, but one player retired mid-match

    year_min: int = 2015
    year_max: int = 2025
# Why this matters: -older tennis (pre-2010) has different dynamics
#                   -surfaces, equipment, play style evolved
#                   -Elo and form features work better on recent data
# Changing these two numbers immediately changes every pipeline.

def get_settings() -> BuildSettings:
    return BuildSettings()