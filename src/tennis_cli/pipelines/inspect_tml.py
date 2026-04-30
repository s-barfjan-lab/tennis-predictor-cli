from __future__ import annotations

from pathlib import Path

import pandas as pd

from tennis_cli.config import Paths


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


def inspect_tml_repo(paths: Paths) -> None:
    """
    Inspect the raw TML repository without modifying any project artifacts.
    """
    repo_dir = paths.tml_dir / "TML-Database"

    if not repo_dir.exists():
        raise FileNotFoundError(f"TML repo not found at: {repo_dir}\n"
            "Run: python -m tennis_cli update-tml")

    _print_header("TML REPOSITORY LOCATION")
    print(repo_dir)

    csv_files = sorted(repo_dir.glob("*.csv"))

    _print_header("CSV FILES FOUND")
    if not csv_files:
        print("No CSV files found.")
        return

    for path in csv_files:
        print(path.name)

    main_atp_file = repo_dir / "ATP_Database.csv"

    if not main_atp_file.exists():
        raise FileNotFoundError(f"Main ATP file not found: {main_atp_file}")

    _print_header("MAIN ATP FILE: BASIC INFO")
    df_head = _safe_read_csv(main_atp_file, nrows=1000)

    print(f"File: {main_atp_file.name}")
    print(f"Sample rows loaded: {len(df_head)}")
    print(f"Columns count: {len(df_head.columns)}")

    _print_header("MAIN ATP FILE: COLUMN NAMES")
    for col in df_head.columns:
        print(col)

    _print_header("MAIN ATP FILE: FIRST 3 ROWS")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_head.head(3).to_string(index=False))

    yearly_files = []
    for year in range(2015, 2026):
        year_file = repo_dir / f"{year}.csv"
        if year_file.exists():
            yearly_files.append(year_file)

    _print_header("YEARLY FILE CHECK")
    if not yearly_files:
        print("No yearly files found for 2015-2025.")
    else:
        for path in yearly_files:
            sample = _safe_read_csv(path, nrows=5)
            print(f"{path.name} -> sample_rows={len(sample)}, "
                f"columns={len(sample.columns)}")

    _print_header("INSPECTION COMPLETE")
    print("TML inspection finished successfully.")
    print("No files were modified.")