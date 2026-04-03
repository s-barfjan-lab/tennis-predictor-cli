from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_model_artifact(model: Any, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    joblib.dump(model, output_path)


def load_model_artifact(input_path: Path) -> Any:
    return joblib.load(input_path)


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)


def save_metadata_json(metadata: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)