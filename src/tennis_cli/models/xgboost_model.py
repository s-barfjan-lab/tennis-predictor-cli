from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
                "model__n_estimators": [300, 600],
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
                "model__n_estimators": [300, 600, 1000],
                "model__learning_rate": [0.02, 0.03],
                "model__max_depth": [3, 4],
                "model__min_child_weight": [1, 3],
                "model__subsample": [0.8],
                "model__colsample_bytree": [0.8],
                "model__reg_lambda": [1.0, 5.0],
                "model__reg_alpha": [0.0, 0.1],
                "model__gamma": [0.0, 0.1], }]

    raise ValueError("search_profile must be 'base' or 'richer'")



def _as_feature_frame(X: pd.DataFrame, columns: list[str], imputer: SimpleImputer, fit: bool, ) -> pd.DataFrame:

    if fit:
        values = imputer.fit_transform(X[columns])
    else:
        values = imputer.transform(X[columns])

    return pd.DataFrame(values, columns=columns, index=X.index)


def build_xgb_search_pipeline(random_state: int = 42) -> Pipeline:
    """
    Pipeline used only for XGB hyperparameter tuning.

    We do not use early stopping inside GridSearchCV.
    """
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    gamma=0.0,
                    random_state=random_state, ),),])



def fit_xgb_classifier(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, feature_columns: list[str],
    config: XGBConfig | None = None, sample_weight: pd.Series | None = None, ) -> dict[str, Any]:

    if config is None:
        config = XGBConfig()

    imputer = SimpleImputer(strategy="median")

    X_train_imp = _as_feature_frame(X=X_train, columns=feature_columns, imputer=imputer, fit=True, )
    X_val_imp = _as_feature_frame(X=X_val, columns=feature_columns, imputer=imputer, fit=False, )

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
        "imputer": imputer,
        "model": model,
        "config": asdict(config),
        "best_iteration": getattr(model, "best_iteration", None), }



def tune_xgb_classifier(X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series | None = None, n_splits: int = 3,
    refit_metric: str = "neg_log_loss", random_state: int = 42, search_profile: str = "base", ) -> GridSearchCV:
    """
    Tune XGBoost using inner time-series CV on the training split only.
    """
    pipeline = build_xgb_search_pipeline(random_state=random_state)
    cv = build_inner_time_series_cv(n_splits=n_splits)

    search = GridSearchCV(estimator=pipeline, param_grid=get_xgb_param_grid(search_profile=search_profile),
        scoring={"neg_log_loss": "neg_log_loss", "roc_auc": "roc_auc", "accuracy": "accuracy", }, refit=refit_metric,
        cv=cv, n_jobs=-1, verbose=1, )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight.to_numpy()

    search.fit(X_train, y_train, **fit_kwargs)
    return search


def xgb_config_from_gridsearch_best_params(best_params: dict) -> XGBConfig:
    """
    Convert GridSearchCV best_params_ into our XGBConfig dataclass.
    """
    return XGBConfig(
        n_estimators=best_params["model__n_estimators"],
        learning_rate=best_params["model__learning_rate"],
        max_depth=best_params["model__max_depth"],
        min_child_weight=best_params["model__min_child_weight"],
        subsample=best_params["model__subsample"],
        colsample_bytree=best_params["model__colsample_bytree"],
        reg_lambda=best_params["model__reg_lambda"],
        reg_alpha=best_params["model__reg_alpha"],
        gamma=best_params.get("model__gamma", 0.0),
    )



def predict_proba_from_xgb_artifact(artifact: dict[str, Any], X: pd.DataFrame, ) -> pd.Series:
    feature_columns = artifact["feature_columns"]
    imputer = artifact["imputer"]
    model = artifact["model"]

    X_imp = pd.DataFrame(imputer.transform(X[feature_columns]), columns=feature_columns, index=X.index, )

    probs = model.predict_proba(X_imp)[:, 1]
    return pd.Series(probs, index=X.index, name="pred_prob")


def feature_importance_from_xgb_artifact(artifact: dict[str, Any], importance_type: str = "gain", ) -> pd.DataFrame:
    model = artifact["model"]
    booster = model.get_booster()
    raw_scores = booster.get_score(importance_type=importance_type)

    rows = []
    for feature_name in artifact["feature_columns"]:
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