from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tennis_cli.models._recency import compute_recency_weights_from_dates
from tennis_cli.models.dataset import get_categorical_feature_columns, get_numeric_feature_columns
from tennis_cli.models.split import build_inner_time_series_cv


@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"
    tree_method: str = "hist"


@dataclass
class XGBCVSearchResult:
    """
    Small GridSearchCV-compatible result object for XGBoost tuning.

    GridSearchCV cannot recompute recency sample weights separately for each
    fold, so the manual search below keeps the attributes used by the training
    pipeline while anchoring fold weights to each fold's training max date.
    """

    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    best_score_: float
    cv_results_: dict[str, Any]



def get_xgb_param_grid(search_profile: str = "base") -> list[dict]:
    """
    XGBoost tuning grids.

    base:
        current compact grid for broad coverage
    richer:
        second-stage grid for strongest branches only
    """
    search_profile = str(search_profile).strip().lower()

    if search_profile == "base":
        return [{
                "model__n_estimators": [300, 600, 1000, 1500],
                "model__learning_rate": [0.03, 0.05],
                "model__max_depth": [3, 4],
                "model__min_child_weight": [1, 5],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__reg_lambda": [1.0, 5.0],
                "model__reg_alpha": [0.0],
                "model__gamma": [0.0], }]

    if search_profile == "richer":
        return [{
                "model__n_estimators": [300, 600, 1000, 1500],
                "model__learning_rate": [0.02, 0.03],
                "model__max_depth": [3, 4],
                "model__min_child_weight": [1, 3],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__reg_lambda": [1.0, 5.0],
                "model__reg_alpha": [0.0, 0.1],
                "model__gamma": [0.0, 0.1], }]

    raise ValueError("search_profile must be 'base' or 'richer'")



def build_xgb_preprocessor(surface_specific: bool = False) -> ColumnTransformer:
    """
    Preprocess raw numeric and categorical model columns for XGBoost.
    """
    numeric_features = get_numeric_feature_columns(surface_specific=surface_specific)
    categorical_features = get_categorical_feature_columns()

    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
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


def _dense_values(values):
    if hasattr(values, "toarray"):
        return values.toarray()
    return values


def _as_feature_frame(X: pd.DataFrame, preprocessor: ColumnTransformer, fit: bool, ) -> pd.DataFrame:

    if fit:
        values = preprocessor.fit_transform(X)
    else:
        values = preprocessor.transform(X)

    columns = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(_dense_values(values), columns=columns, index=X.index)


def build_xgb_search_pipeline(random_state: int = 42, surface_specific: bool = False) -> Pipeline:
    """
    Pipeline used only for XGB hyperparameter tuning.

    We do not use early stopping inside GridSearchCV.
    """
    return Pipeline(steps=[("preprocessor", build_xgb_preprocessor(surface_specific=surface_specific)),
            ("model", xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    gamma=0.0,
                    random_state=random_state, ),),])



def fit_xgb_classifier(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_columns: list[str],
    config: XGBConfig | None = None, sample_weight: pd.Series | None = None, surface_specific: bool = False,
    ) -> dict[str, Any]:

    if config is None:
        config = XGBConfig()

    preprocessor = build_xgb_preprocessor(surface_specific=surface_specific)

    X_train_imp = _as_feature_frame(X=X_train[feature_columns], preprocessor=preprocessor, fit=True, )
    X_val_imp = _as_feature_frame(X=X_val[feature_columns], preprocessor=preprocessor, fit=False, )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        gamma=config.gamma,
        random_state=config.random_state,
        eval_metric=config.eval_metric,
        early_stopping_rounds=config.early_stopping_rounds,
        tree_method=config.tree_method,
    )

    fit_kwargs = {"eval_set": [(X_val_imp, y_val)],
        "verbose": False, }

    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.to_numpy()

    model.fit(X_train_imp, y_train, **fit_kwargs)


    return {"model_type": "xgb",
        "feature_columns": feature_columns,
        "encoded_feature_columns": list(X_train_imp.columns),
        "preprocessor": preprocessor,
        "model": model,
        "config": asdict(config),
        "best_iteration": getattr(model, "best_iteration", None), }



def _rank_scores_descending(scores: list[float]) -> list[int]:
    """
    Rank scores where larger is better, matching GridSearchCV's rank semantics.
    """
    order = np.argsort([-score for score in scores])
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = rank
    return ranks


def tune_xgb_classifier(X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series | None = None, n_splits: int = 3,
    refit_metric: str = "neg_log_loss", random_state: int = 42, search_profile: str = "base",
    surface_specific: bool = False, train_dates: pd.Series | None = None, half_life_days: int = 730, ) -> XGBCVSearchResult:
    """
    Tune XGBoost using inner time-series CV on the training split only.

    Recency weights are recomputed per fold and anchored to that fold's
    training max date. No early stopping is used inside CV; each candidate uses
    its fixed ``model__n_estimators`` value from the parameter grid.
    """
    if refit_metric != "neg_log_loss":
        raise ValueError("Manual XGBoost CV currently supports refit_metric='neg_log_loss' only")

    if train_dates is None:
        raise ValueError("train_dates is required for fold-anchored recency weights.")

    if len(train_dates) != len(X_train):
        raise ValueError("train_dates must have the same length as X_train.")

    pipeline = build_xgb_search_pipeline(random_state=random_state, surface_specific=surface_specific)
    cv = build_inner_time_series_cv(n_splits=n_splits, n_samples=len(X_train))
    param_candidates = list(ParameterGrid(get_xgb_param_grid(search_profile=search_profile)))
    fold_splits = list(cv.split(X_train, y_train))

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
            fold_weight = compute_recency_weights_from_dates(
                train_dates.iloc[fold_train_idx],
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

    search = XGBCVSearchResult(
        best_estimator_=best_pipeline,
        best_params_=best_params,
        best_score_=float(results[best_idx]["mean_test_neg_log_loss"]),
        cv_results_=cv_results,
    )
    return search



def xgb_config_from_gridsearch_best_params(best_params: dict, random_state: int = 42) -> XGBConfig:
    """
    Convert GridSearchCV best_params_ into our XGBConfig dataclass.
    """
    return XGBConfig(n_estimators=best_params["model__n_estimators"],
        learning_rate=best_params["model__learning_rate"],
        max_depth=best_params["model__max_depth"],
        min_child_weight=best_params["model__min_child_weight"],
        subsample=best_params["model__subsample"],
        colsample_bytree=best_params["model__colsample_bytree"],
        reg_lambda=best_params["model__reg_lambda"],
        reg_alpha=best_params["model__reg_alpha"],
        gamma=best_params.get("model__gamma", 0.0),
        random_state=random_state, )


def predict_proba_from_xgb_artifact(artifact: dict[str, Any], X: pd.DataFrame, ) -> pd.Series:
    feature_columns = artifact["feature_columns"]
    model = artifact["model"]

    if "preprocessor" in artifact:
        preprocessor = artifact["preprocessor"]
        values = preprocessor.transform(X[feature_columns])
        encoded_columns = artifact.get("encoded_feature_columns", list(preprocessor.get_feature_names_out()))
        X_imp = pd.DataFrame(_dense_values(values), columns=encoded_columns, index=X.index, )
    else:
        imputer = artifact["imputer"]
        X_imp = pd.DataFrame(imputer.transform(X[feature_columns]), columns=feature_columns, index=X.index, )

    probs = model.predict_proba(X_imp)[:, 1]
    return pd.Series(probs, index=X.index, name="pred_prob")


def feature_importance_from_xgb_artifact(artifact: dict[str, Any], importance_type: str = "gain", ) -> pd.DataFrame:
    model = artifact["model"]
    booster = model.get_booster()
    raw_scores = booster.get_score(importance_type=importance_type)

    rows = []
    feature_columns = artifact.get("encoded_feature_columns", artifact["feature_columns"])
    for feature_name in feature_columns:
        rows.append({"feature": feature_name, "importance": float(raw_scores.get(feature_name, 0.0)), "importance_type": importance_type, })

    out = pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True, )
    return out


def fit_isotonic_calibrator(pred_prob: pd.Series, y_true: pd.Series) -> dict[str, Any]:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, )
    calibrator.fit(pred_prob.astype(float), y_true.astype(int))

    return {"calibration_method": "isotonic", "calibrator": calibrator, }


def apply_isotonic_calibration(calibrator_artifact: dict[str, Any], pred_prob: pd.Series, ) -> pd.Series:
    calibrator = calibrator_artifact["calibrator"]
    calibrated = calibrator.predict(pred_prob.astype(float))

    return pd.Series(calibrated, index=pred_prob.index, name="pred_prob", )



def fit_sigmoid_calibrator(pred_prob: pd.Series, y_true: pd.Series) -> dict[str, Any]:
    calibrator = LogisticRegression()
    X = pred_prob.astype(float).to_numpy().reshape(-1, 1)
    y = y_true.astype(int).to_numpy()
    calibrator.fit(X, y)

    return {"calibration_method": "sigmoid", "calibrator": calibrator, }


def apply_sigmoid_calibration(calibrator_artifact: dict[str, Any], pred_prob: pd.Series, ) -> pd.Series:
    calibrator = calibrator_artifact["calibrator"]
    X = pred_prob.astype(float).to_numpy().reshape(-1, 1)
    calibrated = calibrator.predict_proba(X)[:, 1]

    return pd.Series(calibrated, index=pred_prob.index, name="pred_prob", )


def apply_calibration(calibration_method: str | None, calibrator_artifact: dict[str, Any] | None, pred_prob: pd.Series, ) -> pd.Series:
    
    if calibration_method is None or calibrator_artifact is None:
        return pd.Series(pred_prob, index=pred_prob.index, name="pred_prob")

    calibration_method = calibration_method.lower().strip()

    if calibration_method == "isotonic":
        return apply_isotonic_calibration(calibrator_artifact, pred_prob)

    if calibration_method == "sigmoid":
        return apply_sigmoid_calibration(calibrator_artifact, pred_prob)

    raise ValueError(f"Unsupported calibration_method: {calibration_method}")
