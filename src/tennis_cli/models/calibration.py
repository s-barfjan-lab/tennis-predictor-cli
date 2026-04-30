from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _to_1d_float_array(values: Any) -> np.ndarray:
    """
    Convert input values to a clean 1D float numpy array.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr


def fit_isotonic_calibrator(y_true: Any, raw_probabilities: Any, ) -> IsotonicRegression:
    """
    Fit an isotonic calibrator on validation targets and raw model probabilities.

    Parameters
    ----------
    y_true:
        True binary labels for the validation set.
    raw_probabilities:
        Raw predicted probabilities from the already-fitted model.

    Returns
    -------
    IsotonicRegression
        Fitted isotonic calibrator.
    """
    y = _to_1d_float_array(y_true)
    p = _to_1d_float_array(raw_probabilities)

    if len(y) != len(p):
        raise ValueError("y_true and raw_probabilities must have the same length.")

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", )
    calibrator.fit(p, y)
    return calibrator


def apply_isotonic_calibration(calibrator: IsotonicRegression, raw_probabilities: Any, ) -> np.ndarray:
    """
    Apply a fitted isotonic calibrator to raw probabilities.
    """
    p = _to_1d_float_array(raw_probabilities)
    calibrated = calibrator.predict(p)
    calibrated = np.asarray(calibrated, dtype=float)
    calibrated = np.clip(calibrated, 0.0, 1.0)
    return calibrated