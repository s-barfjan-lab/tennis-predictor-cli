from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from tennis_cli.models.dataset import (
    build_training_matrices,
    get_categorical_feature_columns,
    get_numeric_feature_columns,
)

from tennis_cli.models.split import build_inner_time_series_cv



@dataclass
class LogisticConfig:
    max_iter: int = 2000
    solver: str = "lbfgs"
    C: float = 1.0
    class_weight: str | None = None
    random_state: int = 42


@dataclass
class LogisticCVSearchResult:
    """
    Small GridSearchCV-compatible result object for logistic tuning.

    GridSearchCV cannot recompute sample weights separately for each fold, so
    the manual search below preserves the attributes used by the training
    pipeline while allowing fold-anchored recency weights.
    """

    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    best_score_: float
    cv_results_: dict[str, Any]


def build_logistic_pipeline(config: LogisticConfig | None = None, surface_specific: bool = False) -> Pipeline:
    """
    Build the Phase 3 logistic regression pipeline.

    Steps:
    1. Median imputation for numeric missing values
    2. Standard scaling
    3. Logistic regression
    """
    if config is None:
        config = LogisticConfig()

    numeric_features = get_numeric_feature_columns(surface_specific=surface_specific)
    categorical_features = get_categorical_feature_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
            ("model", LogisticRegression(
        max_iter=config.max_iter,
        solver=config.solver,
        C=config.C,
        class_weight=config.class_weight,
        random_state=config.random_state,),),])
    
    return pipeline




def get_logistic_param_grid() -> list[dict]:
    """
    Compact, stable tuning grid for logistic regression.

    We tune only solver, regularization strength, and class weighting.
    """
    return [{"model__solver": ["lbfgs", "liblinear"],
            "model__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],}]


def _compute_recency_weights_from_dates(dates: pd.Series, half_life_days: int) -> np.ndarray:
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


def _rank_scores_descending(scores: list[float]) -> list[int]:
    """
    Rank scores where larger is better, matching GridSearchCV's rank semantics.
    """
    order = np.argsort([-score for score in scores])
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = rank
    return ranks




def fit_logistic_baseline(train_df: pd.DataFrame, config: LogisticConfig | None = None, surface_specific: bool = False,
    sample_weight: pd.Series | None = None, ) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Fit the logistic baseline on a training split dataframe.

    Returns:
    - fitted sklearn pipeline
    - X_train
    - y_train
    - meta_train
    """
    
    X_train, y_train, meta_train = build_training_matrices(train_df, surface_specific=surface_specific, )

    pipeline = build_logistic_pipeline(config=config, surface_specific=surface_specific)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight.to_numpy()

    pipeline.fit(X_train, y_train, **fit_kwargs)

    return pipeline, X_train, y_train, meta_train



def tune_logistic_baseline(train_df: pd.DataFrame, config: LogisticConfig | None = None, surface_specific: bool = False,
    n_splits: int = 5, refit_metric: str = "neg_log_loss", sample_weight: pd.Series | None = None,
    half_life_days: int = 730, date_col: str = "tourney_date",
    ) -> tuple[LogisticCVSearchResult, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Tune logistic regression on the training split only, using an inner
    time-series cross-validation loop.

    Recency weights are recomputed per fold and anchored to that fold's
    training max date. The final refit still uses the full training-split
    weights passed via ``sample_weight``.

    Returns:
    - fitted GridSearchCV-compatible result object
    - X_train
    - y_train
    - meta_train
    """
    X_train, y_train, meta_train = build_training_matrices(train_df, surface_specific=surface_specific, )
    if date_col not in train_df.columns:
        raise ValueError(f"Missing required date column for CV recency weights: {date_col}")

    pipeline = build_logistic_pipeline(config=config, surface_specific=surface_specific)
    cv = build_inner_time_series_cv(n_splits=n_splits, n_samples=len(X_train))
    param_candidates = list(ParameterGrid(get_logistic_param_grid()))
    fold_splits = list(cv.split(X_train, y_train))

    if refit_metric != "neg_log_loss":
        raise ValueError("Manual logistic CV currently supports refit_metric='neg_log_loss' only")

    results: list[dict[str, Any]] = []
    for params in param_candidates:
        split_neg_log_loss: list[float] = []
        split_roc_auc: list[float] = []
        split_accuracy: list[float] = []
        fit_times: list[float] = []
        score_times: list[float] = []

        for fold_train_idx, fold_val_idx in fold_splits:
            fold_pipeline = clone(pipeline).set_params(**params)
            fold_X_train = X_train.iloc[fold_train_idx]
            fold_y_train = y_train.iloc[fold_train_idx]
            fold_X_val = X_train.iloc[fold_val_idx]
            fold_y_val = y_train.iloc[fold_val_idx]
            fold_weight = _compute_recency_weights_from_dates(
                train_df.iloc[fold_train_idx][date_col],
                half_life_days=half_life_days,
            )

            fit_start = perf_counter()
            fold_pipeline.fit(fold_X_train, fold_y_train, model__sample_weight=fold_weight)
            fit_times.append(perf_counter() - fit_start)

            score_start = perf_counter()
            val_prob = fold_pipeline.predict_proba(fold_X_val)[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            split_neg_log_loss.append(-log_loss(fold_y_val, val_prob, labels=[0, 1]))
            split_accuracy.append(accuracy_score(fold_y_val, val_pred))
            try:
                split_roc_auc.append(roc_auc_score(fold_y_val, val_prob))
            except ValueError:
                split_roc_auc.append(np.nan)
            score_times.append(perf_counter() - score_start)

        results.append({
            "params": params,
            "mean_fit_time": float(np.mean(fit_times)),
            "std_fit_time": float(np.std(fit_times)),
            "mean_score_time": float(np.mean(score_times)),
            "std_score_time": float(np.std(score_times)),
            "mean_test_neg_log_loss": float(np.mean(split_neg_log_loss)),
            "std_test_neg_log_loss": float(np.std(split_neg_log_loss)),
            "mean_test_accuracy": float(np.mean(split_accuracy)),
            "std_test_accuracy": float(np.std(split_accuracy)),
            "mean_test_roc_auc": float(np.nanmean(split_roc_auc)),
            "std_test_roc_auc": float(np.nanstd(split_roc_auc)),
            "split_test_neg_log_loss": split_neg_log_loss,
            "split_test_accuracy": split_accuracy,
            "split_test_roc_auc": split_roc_auc,
        })

    best_idx = int(np.argmax([row["mean_test_neg_log_loss"] for row in results]))
    best_params = results[best_idx]["params"]
    best_pipeline = clone(pipeline).set_params(**best_params)

    final_fit_kwargs = {}
    if sample_weight is not None:
        final_fit_kwargs["model__sample_weight"] = sample_weight.to_numpy()

    best_pipeline.fit(X_train, y_train, **final_fit_kwargs)

    cv_results: dict[str, Any] = {
        "params": [row["params"] for row in results],
        "mean_fit_time": [row["mean_fit_time"] for row in results],
        "std_fit_time": [row["std_fit_time"] for row in results],
        "mean_score_time": [row["mean_score_time"] for row in results],
        "std_score_time": [row["std_score_time"] for row in results],
        "mean_test_neg_log_loss": [row["mean_test_neg_log_loss"] for row in results],
        "std_test_neg_log_loss": [row["std_test_neg_log_loss"] for row in results],
        "rank_test_neg_log_loss": _rank_scores_descending([row["mean_test_neg_log_loss"] for row in results]),
        "mean_test_accuracy": [row["mean_test_accuracy"] for row in results],
        "std_test_accuracy": [row["std_test_accuracy"] for row in results],
        "mean_test_roc_auc": [row["mean_test_roc_auc"] for row in results],
        "std_test_roc_auc": [row["std_test_roc_auc"] for row in results],
    }
    for param_name in sorted(best_params):
        cv_results[f"param_{param_name}"] = [row["params"][param_name] for row in results]

    for split_idx in range(n_splits):
        cv_results[f"split{split_idx}_test_neg_log_loss"] = [
            row["split_test_neg_log_loss"][split_idx] for row in results
        ]
        cv_results[f"split{split_idx}_test_accuracy"] = [
            row["split_test_accuracy"][split_idx] for row in results
        ]
        cv_results[f"split{split_idx}_test_roc_auc"] = [
            row["split_test_roc_auc"][split_idx] for row in results
        ]

    search = LogisticCVSearchResult(
        best_estimator_=best_pipeline,
        best_params_=best_params,
        best_score_=float(results[best_idx]["mean_test_neg_log_loss"]),
        cv_results_=cv_results,
    )

    return search, X_train, y_train, meta_train



def predict_split(pipeline: Pipeline, split_df: pd.DataFrame, surface_specific: bool = False, ) -> pd.DataFrame:
    """
    Run prediction for one split dataframe and return a tidy result table.

    Output columns:
    - y_true
    - prob_player_a_win
    - pred_player_a_win
    plus metadata columns
    """
    X, y, meta = build_training_matrices(split_df, surface_specific=surface_specific, )

    prob = pipeline.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    out = meta.copy()
    out["y_true"] = y.values
    out["prob_player_a_win"] = prob
    out["pred_player_a_win"] = pred

    return out



def extract_logistic_coefficients(pipeline) -> pd.DataFrame:
    """
    Extract fitted logistic regression coefficients using the
    actual feature names that survived preprocessing.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps.get("preprocessor", pipeline[:-1])

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"feature_{i}" for i in range(len(model.coef_[0]))]

    coefs = model.coef_[0]

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs, })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df
