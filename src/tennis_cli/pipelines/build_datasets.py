from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()

TOUR_ATP = "atp"
TOUR_WTA = "wta"

@dataclass(frozen=True)
class DatasetArtifacts:
    matches_parquet: Path

# takes messy surface names from raw data and converts them into a clean, standardized format
def _standardize_surface(s: str | float) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    # str(s) -> makes sure the value is a string (even if it wasn’t)
    #.strip() -> removes extra spaces
    #.lower() -> makes everything lowercase

    mapping = {
        "hard": "HARD",
        "clay": "CLAY",
        "grass": "GRASS",
        "carpet": "CARPET",
    }
    return mapping.get(s, s.upper())
    # If s exists in mapping -> return the clean version
    # If s does NOT exist -> return s.upper() -> Instead of crashing or silently dropping 
    # them, I take the uppercase form of whatever exists -> I can later decide how to handle it


# This function finds all yearly ATP/WTA match CSV files in the Sackmann repo, loads them, 
# and merges them into one big table
def _load_sackmann_matches(repo_dir: Path, year_min: int, year_max: int) -> pd.DataFrame:
    # Sackmann naming: ATP: atp_matches_YYYY.csv
    #                  WTA: wta_matches_YYYY.csv
    csvs = []  # paths to all matching CSV files found on disk will be here
    for y in range(year_min, year_max + 1):
        # try both patterns (we’ll pass correct repo_dir for each tour)
        for pattern in [f"*matches_{y}.csv"]: 
        # For each year, this creates patterns like: *matches_2019.csv or *matches_2020.csv
        # The * means -> Anything before matches_YYYY.csv -> atp or wta
            for fp in repo_dir.glob(pattern):
                csvs.append(fp)
            # glob() -> searches the folder for files matching the pattern
            # returns file paths (Path objects)

    if not csvs:
        raise FileNotFoundError(
            f"No match CSVs found in {repo_dir} for years {year_min}-{year_max}")

    console.print(f"[cyan]Found {len(csvs)} match CSV files[/] in {repo_dir}")

    dfs = []  # This list will store data tables, not file paths
    for fp in sorted(csvs):
        df = pd.read_csv(fp, low_memory=False)
        # low_memory=False -> prevents Pandas from guessing column types incorrectly
        df["source_file"] = fp.name # So later I can trace a row back to its source file
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) # stacks all yearly DataFrames row-wise


# This function cleans and standardizes the most important match columns so they behave 
# consistently everywhere in the project.
# Minimal normalized schema (we’ll expand later)
def _normalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy() # In order not to mutate raw input

    # Date
    if "tourney_date" in out.columns:
        out["tourney_date"] = pd.to_datetime(out["tourney_date"].astype(str), format="%Y%m%d",
                                              errors="coerce")
        # errors="coerce" -> if something is broken -> don’t crash but replace with NaT
    
    # Surface standardized
    if "surface" in out.columns:
        out["surface"] = out["surface"].apply(_standardize_surface)

    # Winner/Loser names normalize (string)
    for col in ["winner_name", "loser_name", "tourney_name", "round", "score"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    # Normalize seed columns (mixed numeric / string -> string) -> I encountered a bug while 
    # testing, Parquet could not recognize different data (string, number, etc.), so for these
    # 2 columns in phase 1 I made everything a string.
    for col in ["winner_seed", "loser_seed"]:
        if col in out.columns:
            out[col] = out[col].astype(str)

    return out


# This function removes matches that did not represent “normal tennis performance”, based 
# on the chosen rules coming from BuildSettings.
def _filter_matches(df: pd.DataFrame, drop_walkovers: bool, 
                    drop_retirements: bool) -> pd.DataFrame:
    out = df.copy()

    # Walkovers are often encoded in score as "W/O" or similar
    if drop_walkovers and "score" in out.columns:
        out = out[~out["score"].str.contains("W/O", case=False, na=False)]
        # checks if the string "W/O" appears in the score
        # case=False -> "w/o", "W/o", "W/O" all match
        # na=False -> missing values are treated as “does not contain”
        
    # Retirements often have "RET" or "DEF" etc in score; Sackmann commonly uses "RET"
    if drop_retirements and "score" in out.columns:
        out = out[~out["score"].str.contains("RET", case=False, na=False)]

    return out


# This function takes raw Sackmann data and turns it into one clean, filtered, saved dataset 
# for either ATP or WTA -> core function of this part
def build_tour_dataset(
    tour: str,
    sackmann_root: Path,
    processed_dir: Path,
    year_min: int,
    year_max: int,
    drop_walkovers: bool,
    drop_retirements: bool,
) -> DatasetArtifacts:
    
    tour = tour.lower().strip()

    if tour not in {TOUR_ATP, TOUR_WTA}:
        raise ValueError("tour must be 'atp' or 'wta'")

    repo_dir = sackmann_root / ("tennis_atp" if tour == TOUR_ATP else "tennis_wta")

    console.print(f"[bold]Loading[/] {tour.upper()} matches from {repo_dir}")
    raw = _load_sackmann_matches(repo_dir, year_min, year_max)

    console.print("[bold]Normalizing[/] columns")
    norm = _normalize_matches(raw)

    console.print("[bold]Filtering[/] walkovers/retirements (if enabled)")
    clean = _filter_matches(norm, drop_walkovers, drop_retirements)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{tour}_matches_{year_min}_{year_max}.parquet"

    console.print(f"[bold green]Writing[/] {out_path}")
    clean.to_parquet(out_path, index=False)

    console.print(
        f"[green]Done.[/] {tour.upper()} matches: {len(clean):,} rows "
        f"(from {len(raw):,} raw rows)."
    )

    return DatasetArtifacts(matches_parquet=out_path)


# This function prints a compact health report of the dataset so I can verify it before modeling
def explore_dataset(parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)

    console.print(f"\n[bold]Dataset:[/] {parquet_path.name}")
    console.print(f"Rows: [cyan]{len(df):,}[/]  Columns: [cyan]{len(df.columns)}[/]")

    if "tourney_date" in df.columns:
        console.print(f"Date range: {df['tourney_date'].min()} → {df['tourney_date'].max()}")

    if "surface" in df.columns:
        console.print("\n[bold]Surface counts[/]")
        console.print(df["surface"].value_counts(dropna=False).head(10).to_string())
    # how many matches per surface, whether surfaces are standardized correctly,whether missing values exist

    if "round" in df.columns:
        console.print("\n[bold]Round counts[/]")
        console.print(df["round"].value_counts(dropna=False).head(10).to_string())
    # pressure increases with round, finals behave differently,models may perform differently 
    # by round -> event context

    if "score" in df.columns:
        contains_ret = df["score"].str.contains("RET", case=False, na=False).mean()
        contains_wo = df["score"].str.contains("W/O", case=False, na=False).mean()
        console.print(f"\nRET fraction: {contains_ret:.4f} | W/O fraction: {contains_wo:.4f}")
    # this verifies my filtering: confirms assumptions were enforced, prevents silent 
    # contamination of training data, If these numbers are not near zero -> something is wrong

    console.print("\n[bold]Missing values (top 15)[/]")
    na = df.isna().mean().sort_values(ascending=False).head(15)
    console.print(na.to_string())
    # This computes, for every column: fraction of missing values,then shows the worst
    # tells which features are unreliable