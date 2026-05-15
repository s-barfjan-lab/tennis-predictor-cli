from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tennis_cli.models.dataset import TARGET_COLUMN
from sklearn.model_selection import TimeSeriesSplit


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


def split_train_into_train_and_calibration(
    train_df: pd.DataFrame,
    date_col: str = "tourney_date",
    calibration_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve the last `calibration_days` of the training partition into a separate
    calibration set, used for early stopping and post-hoc calibrator fitting.
    The remaining earlier portion is the new inner-train set.
    """
    train_df = train_df.copy()
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    cutoff = train_df[date_col].max() - pd.Timedelta(days=calibration_days)
    train_inner = train_df[train_df[date_col] <= cutoff].copy()
    calib = train_df[train_df[date_col] > cutoff].copy()
    if len(train_inner) == 0 or len(calib) == 0:
        raise ValueError("Empty split when carving calibration set; lower calibration_days.")
    return train_inner, calib


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

def summarize_surface_balance(df: pd.DataFrame, config: TimeSplitConfig | None = None, ) -> pd.DataFrame:
    """
    Summarize class balance by split and surface.

    Output columns:
    - split
    - surface
    - rows
    - positive_labels
    - negative_labels
    - positive_rate
    - date_min
    - date_max
    """
    if config is None:
        config = TimeSplitConfig()

    _validate_split_input(df, config.date_col)

    out = df.copy()
    out[config.date_col] = pd.to_datetime(out[config.date_col], errors="raise")
    out = out.sort_values([config.date_col, "match_id"]).reset_index(drop=True)

    train_end = pd.Timestamp(config.train_end)
    val_end = pd.Timestamp(config.val_end)

    def _label_split(ts: pd.Timestamp) -> str:
        if ts <= train_end:
            return "train"
        if ts <= val_end:
            return "validation"
        return "test"

    out["split"] = out[config.date_col].apply(_label_split)
    out["surface_norm"] = out["surface"].astype(str).str.upper()

    rows = []
    for (split_name, surface_name), grp in out.groupby(["split", "surface_norm"], dropna=False):
        rows.append(
            {
                "split": split_name,
                "surface": surface_name,
                "rows": int(len(grp)),
                "positive_labels": int(grp[TARGET_COLUMN].sum()),
                "negative_labels": int((1 - grp[TARGET_COLUMN]).sum()),
                "positive_rate": float(grp[TARGET_COLUMN].mean()),
                "date_min": str(grp[config.date_col].min().date()),
                "date_max": str(grp[config.date_col].max().date()),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["split", "surface"],
        ignore_index=True,
    )
    return summary_df


def build_inner_time_series_cv(
    n_splits: int = 5,
    n_samples: int | None = None,
    validation_fraction: float = 0.10,
    gap: int = 0,
) -> TimeSeriesSplit:
    """
    Build the inner cross-validation splitter used for hyperparameter tuning
    on the training portion only.

    We use fixed-size validation blocks so each fold is closer to a single
    recent operational slice instead of sklearn's default expanding test size.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    test_size = None
    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        test_size = max(1, int(validation_fraction * n_samples))
        required_rows = (n_splits * test_size) + gap + 1
        if required_rows > n_samples:
            raise ValueError(
                "Not enough rows for requested inner CV: "
                f"n_samples={n_samples}, n_splits={n_splits}, "
                f"test_size={test_size}, gap={gap}"
            )

    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
