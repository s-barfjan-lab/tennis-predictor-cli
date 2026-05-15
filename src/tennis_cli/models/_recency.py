from __future__ import annotations

import numpy as np
import pandas as pd


def compute_recency_weights_from_dates(dates: pd.Series, half_life_days: int) -> np.ndarray:
    """
    Compute recency weights anchored to the newest date in the provided fold.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    parsed_dates = pd.to_datetime(dates, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Cannot compute recency weights: invalid dates detected.")

    age_days = (parsed_dates.max() - parsed_dates).dt.days.astype(float)
    return (0.5 ** (age_days / float(half_life_days))).to_numpy()
