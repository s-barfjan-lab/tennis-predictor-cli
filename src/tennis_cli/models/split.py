from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tennis_cli.models.dataset import TARGET_COLUMN


@dataclass
class TimeSplitConfig:
    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"
    date_col: str = "tourney_date"


def _validate_split_input(df: pd.DataFrame, date_col: str) -> None:
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    if df[date_col].isna().any():
        bad_count = int(df[date_col].isna().sum())
        raise ValueError(
            f"'{date_col}' contains {bad_count} missing values. Split requires valid dates."
        )


def chronological_train_val_test_split(df: pd.DataFrame, config: TimeSplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split a baseline dataframe into train / validation / test.

    Default behavior:
    - train: <= 2022-12-31
    - val:   2023-01-01 .. 2023-12-31
    - test:  >= 2024-01-01
    """
    if config is None:
        config = TimeSplitConfig()

    _validate_split_input(df, config.date_col)

    out = df.copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="raise")
    out = out.sort_values([config.date_col, "match_id"]).reset_index(drop=True)

    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    if train_end >= val_end:
        raise ValueError("train_end must be strictly earlier than val_end")

    train_df = out[out[config.date_col] <= train_end].copy()
    val_df = out[(out[config.date_col] > train_end) & (out[config.date_col] <= val_end)].copy()
    test_df = out[out[config.date_col] > val_end].copy()

    if train_df.empty:
        raise ValueError("Train split is empty")
    if val_df.empty:
        raise ValueError("Validation split is empty")
    if test_df.empty:
        raise ValueError("Test split is empty")

    return train_df, val_df, test_df


def summarize_split(df: pd.DataFrame, split_name: str) -> dict:
    """
    Return a compact summary for one split.
    """
    summary = {
        "split": split_name,
        "rows": int(len(df)),
        "positive_labels": int(df[TARGET_COLUMN].sum()),
        "negative_labels": int((1 - df[TARGET_COLUMN]).sum()),
        "positive_rate": float(df[TARGET_COLUMN].mean()),
        "date_min": str(df["tourney_date"].min().date()),
        "date_max": str(df["tourney_date"].max().date()),
    }
    return summary


def summarize_all_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,) -> pd.DataFrame:
    """
    Build a small dataframe summarizing train / val / test.
    """
    rows = [
        summarize_split(train_df, "train"),
        summarize_split(val_df, "validation"),
        summarize_split(test_df, "test"),
    ]
    return pd.DataFrame(rows)