from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_cli.models.dataset import (TARGET_COLUMN, get_feature_columns, load_baseline_dataframe, )
from tennis_cli.models.evaluate import (evaluate_multiple_splits, evaluate_predictions, )
from tennis_cli.models.io import (save_dataframe_csv, save_metadata_json, save_metrics_json, save_model_artifact, )
from tennis_cli.models.logistic import (extract_logistic_coefficients, fit_logistic_baseline, predict_split, )
from tennis_cli.models.logistic import (extract_logistic_coefficients, fit_logistic_baseline, predict_split, tune_logistic_baseline, )
from tennis_cli.models.split import (chronological_train_val_test_split, summarize_all_splits, )
from tennis_cli.models.xgboost_model import (apply_isotonic_calibration, apply_sigmoid_calibration, feature_importance_from_xgb_artifact,
    fit_isotonic_calibrator, fit_sigmoid_calibrator, fit_xgb_classifier, predict_proba_from_xgb_artifact,
    tune_xgb_classifier, xgb_config_from_gridsearch_best_params, )



def _validate_tour_and_source(tour: str, source: str) -> tuple[str, str]:

    tour = tour.lower().strip()
    if tour not in {"atp", "wta"}:
        raise ValueError("tour must be 'atp' or 'wta'")

    source = source.lower().strip()
    if source not in {"sackmann", "tml"}:
        raise ValueError("source must be 'sackmann' or 'tml'")

    if source == "tml" and tour != "atp":
        raise ValueError("TML source is currently supported only for ATP.")

    return tour, source


def _normalize_surface(surface: str | None) -> str | None:
    if surface is None:
        return None

    value = str(surface).strip().title()
    if not value:
        return None

    if value == "Carpet":
        return "Hard"

    if value not in {"Hard", "Clay", "Grass"}:
        raise ValueError("surface must be one of: Hard, Clay, Grass")

    return value


def _build_artifact_suffix(source: str, surface: str | None) -> str:
    """
    Build a file suffix that keeps source and surface variants separate.

    Examples
    --------
    sackmann + None   -> ""
    tml + None        -> "_tml"
    sackmann + Clay   -> "_clay"
    tml + Grass       -> "_tml_grass"
    """
    parts = []

    if source != "sackmann":
        parts.append(source)

    if surface is not None:
        parts.append(surface.lower())

    if not parts:
        return ""

    return "_" + "_".join(parts)


def _compute_recency_sample_weights(df: pd.DataFrame, half_life_days: int = 365, date_col: str = "tourney_date", ) -> pd.Series:
    """
    Exponential recency weighting:
    newest row gets weight 1.0,
    older rows decay by half every `half_life_days`.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column for recency weights: {date_col}")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("Cannot compute recency weights: invalid dates detected.")

    max_date = dates.max()
    age_days = (max_date - dates).dt.days.astype(float)

    weights = 0.5 ** (age_days / float(half_life_days))
    return pd.Series(weights, index=df.index, name="sample_weight")


def _summarize_sample_weights(sample_weight: pd.Series) -> dict:
    return {
        "min": float(sample_weight.min()),
        "max": float(sample_weight.max()),
        "mean": float(sample_weight.mean()),
        "median": float(sample_weight.median()),
    }


def _compute_binary_metrics(y_true: pd.Series, pred_prob: pd.Series, split_name: str) -> dict:

    y_true = y_true.astype(int)
    pred_prob = pred_prob.astype(float)
    pred_label = (pred_prob >= 0.5).astype(int)

    try:
        auc_value = float(roc_auc_score(y_true, pred_prob))
    except ValueError:
        auc_value = None

    return {
        "split": split_name,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "roc_auc": auc_value,
        "log_loss": float(log_loss(y_true, pred_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, pred_prob)),
        "positive_rate": float(y_true.mean()),
        "avg_pred_prob": float(pred_prob.mean()), }


def _metrics_table_from_dicts(metrics_list: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(metrics_list)


def _prepare_gridsearch_results_df(cv_results: dict, sort_by: str = "rank_test_neg_log_loss") -> pd.DataFrame:
    """
    Convert sklearn GridSearchCV cv_results_ into a tidy dataframe.
    """
    df = pd.DataFrame(cv_results).copy()

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)

    return df


def get_logit_tuning_targets() -> list[dict]:
    """
    Central experiment matrix for tuned logistic runs.

    Each row is one tuning target we want to complete.
    """
    return [{"tour": "atp", "source": "sackmann", "surface": "Hard"},
        {"tour": "atp", "source": "sackmann", "surface": "Clay"},
        {"tour": "atp", "source": "sackmann", "surface": "Grass"},
        {"tour": "atp", "source": "tml", "surface": "Hard"},
        {"tour": "atp", "source": "tml", "surface": "Clay"},
        {"tour": "atp", "source": "tml", "surface": "Grass"},
        {"tour": "wta", "source": "sackmann", "surface": "Hard"},
        {"tour": "wta", "source": "sackmann", "surface": "Clay"},
        {"tour": "wta", "source": "sackmann", "surface": "Grass"},]


def get_xgb_tuning_targets() -> list[dict]:
    """
    Central experiment matrix for tuned XGBoost runs.

    Each row is one tuning target we want to complete.
    """
    return [{"tour": "atp", "source": "sackmann", "surface": "Hard"},
        {"tour": "atp", "source": "sackmann", "surface": "Clay"},
        {"tour": "atp", "source": "sackmann", "surface": "Grass"},
        {"tour": "atp", "source": "tml", "surface": "Hard"},
        {"tour": "atp", "source": "tml", "surface": "Clay"},
        {"tour": "atp", "source": "tml", "surface": "Grass"},
        {"tour": "wta", "source": "sackmann", "surface": "Hard"},
        {"tour": "wta", "source": "sackmann", "surface": "Clay"},
        {"tour": "wta", "source": "sackmann", "surface": "Grass"},]




def train_logistic_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None, 
                            half_life_days: int = 730,) -> dict:
    """
    Train the Phase 3 logistic baseline for one tour and save artifacts.
    """
    tour, source = _validate_tour_and_source(tour, source)
    surface = _normalize_surface(surface)
    surface_specific = surface is not None

    models_dir = project_root / "data" / "models"
    df = load_baseline_dataframe(project_root, tour, source=source, surface=surface)

    train_df, val_df, test_df = chronological_train_val_test_split(df)
    split_summary = summarize_all_splits(train_df, val_df, test_df)
    train_sample_weight = _compute_recency_sample_weights(train_df, half_life_days=half_life_days,)

    pipeline, _, _, _ = fit_logistic_baseline(train_df, surface_specific=surface_specific, sample_weight=train_sample_weight, )

    val_preds = predict_split(pipeline, val_df, surface_specific=surface_specific, )
    test_preds = predict_split(pipeline, test_df, surface_specific=surface_specific, )

    val_metrics = evaluate_predictions(val_preds, "validation")
    test_metrics = evaluate_predictions(test_preds, "test")
    all_metrics_df = evaluate_multiple_splits({"validation": val_preds, "test": test_preds, })

    coef_df = extract_logistic_coefficients(pipeline)

    suffix = _build_artifact_suffix(source=source, surface=surface)

    model_path = models_dir / f"{tour}_logit_baseline{suffix}.joblib"
    metrics_path = models_dir / f"{tour}_logit_baseline{suffix}_metrics.json"
    metrics_table_path = models_dir / f"{tour}_logit_baseline{suffix}_metrics.csv"
    coef_path = models_dir / f"{tour}_logit_baseline{suffix}_coefficients.csv"
    metadata_path = models_dir / f"{tour}_logit_baseline{suffix}_meta.json"

    save_model_artifact(pipeline, model_path)
    save_metrics_json(
        {
            "tour": tour,
            "model_type": "logistic_regression",
            "validation": val_metrics,
            "test": test_metrics,
        },
        metrics_path,
    )
    save_dataframe_csv(all_metrics_df, metrics_table_path)
    save_dataframe_csv(coef_df, coef_path)

    metadata = {
        "tour": tour,
        "phase": 3,
        "model_type": "logistic_regression",
        "source_backbone": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "feature_columns": get_feature_columns(surface_specific=surface_specific),
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "validation_date_min": str(val_df["tourney_date"].min().date()),
        "validation_date_max": str(val_df["tourney_date"].max().date()),
        "test_date_min": str(test_df["tourney_date"].min().date()),
        "test_date_max": str(test_df["tourney_date"].max().date()),
        "split_summary": split_summary.to_dict(orient="records"),
        "recency_weighting": {"enabled": True, "half_life_days": half_life_days,
             "train_weight_summary": _summarize_sample_weights(train_sample_weight), }, }
    
    save_metadata_json(metadata, metadata_path)

    return {
        "tour": tour,
        "source": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "model_type": "logit",
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics_table_path": str(metrics_table_path),
        "coef_path": str(coef_path),
        "metadata_path": str(metadata_path),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics, }



def train_tuned_logistic_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None,
                                   half_life_days: int = 730, ) -> dict:
    """
    Tune logistic regression using inner time-series CV on the training split only,
    then evaluate the best estimator on validation and test.
    """
    tour, source = _validate_tour_and_source(tour, source)
    surface = _normalize_surface(surface)
    surface_specific = surface is not None

    models_dir = project_root / "data" / "models"
    df = load_baseline_dataframe(project_root, tour, source=source, surface=surface)

    train_df, val_df, test_df = chronological_train_val_test_split(df)
    split_summary = summarize_all_splits(train_df, val_df, test_df)
    train_sample_weight = _compute_recency_sample_weights(train_df, half_life_days=half_life_days,)

    search, _, _, _ = tune_logistic_baseline(train_df, surface_specific=surface_specific, n_splits=3,
        refit_metric="neg_log_loss", sample_weight=train_sample_weight, )

    best_pipeline = search.best_estimator_

    val_preds = predict_split(best_pipeline, val_df, surface_specific=surface_specific, )
    test_preds = predict_split(best_pipeline, test_df, surface_specific=surface_specific, )

    val_metrics = evaluate_predictions(val_preds, "validation")
    test_metrics = evaluate_predictions(test_preds, "test")
    all_metrics_df = evaluate_multiple_splits({"validation": val_preds, "test": test_preds})

    coef_df = extract_logistic_coefficients(best_pipeline)
    cv_results_df = _prepare_gridsearch_results_df(search.cv_results_)

    suffix = _build_artifact_suffix(source=source, surface=surface)

    model_path = models_dir / f"{tour}_logit_tuned{suffix}.joblib"
    metrics_path = models_dir / f"{tour}_logit_tuned{suffix}_metrics.json"
    metrics_table_path = models_dir / f"{tour}_logit_tuned{suffix}_metrics.csv"
    coef_path = models_dir / f"{tour}_logit_tuned{suffix}_coefficients.csv"
    cv_results_path = models_dir / f"{tour}_logit_tuned{suffix}_cv_results.csv"
    metadata_path = models_dir / f"{tour}_logit_tuned{suffix}_meta.json"

    save_model_artifact(best_pipeline, model_path)
    save_metrics_json(
        {
            "tour": tour,
            "model_type": "logistic_regression_tuned",
            "best_params": search.best_params_,
            "best_cv_score_neg_log_loss": float(search.best_score_),
            "validation": val_metrics,
            "test": test_metrics,
        },
        metrics_path,
    )
    save_dataframe_csv(all_metrics_df, metrics_table_path)
    save_dataframe_csv(coef_df, coef_path)
    save_dataframe_csv(cv_results_df, cv_results_path)

    metadata = {
        "tour": tour,
        "phase": "4B",
        "model_type": "logistic_regression_tuned",
        "source_backbone": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "feature_columns": get_feature_columns(surface_specific=surface_specific),
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "validation_date_min": str(val_df["tourney_date"].min().date()),
        "validation_date_max": str(val_df["tourney_date"].max().date()),
        "test_date_min": str(test_df["tourney_date"].min().date()),
        "test_date_max": str(test_df["tourney_date"].max().date()),
        "split_summary": split_summary.to_dict(orient="records"),
        "best_params": search.best_params_,
        "best_cv_score_neg_log_loss": float(search.best_score_),
        "cv_refit_metric": "neg_log_loss",
        "cv_n_splits": 3,
        "recency_weighting": {"enabled": True, "half_life_days": half_life_days,
            "train_weight_summary": _summarize_sample_weights(train_sample_weight), }, }
    save_metadata_json(metadata, metadata_path)

    return {
        "tour": tour,
        "source": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "model_type": "logit_tuned",
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics_table_path": str(metrics_table_path),
        "coef_path": str(coef_path),
        "cv_results_path": str(cv_results_path),
        "metadata_path": str(metadata_path),
        "best_params": search.best_params_,
        "best_cv_score_neg_log_loss": float(search.best_score_),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }



def train_tuned_xgb_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None,
                             half_life_days: int = 730, search_profile: str = "base", ) -> dict:
    """
    Tune XGBoost using inner time-series CV on the training split only,
    then fit the final tuned model with early stopping on validation,
    and finally evaluate raw + calibrated predictions on validation/test.
    """
    tour, source = _validate_tour_and_source(tour, source)
    surface = _normalize_surface(surface)
    surface_specific = surface is not None

    models_dir = project_root / "data" / "models"
    df = load_baseline_dataframe(project_root, tour, source=source, surface=surface)

    train_df, val_df, test_df = chronological_train_val_test_split(df)
    split_summary = summarize_all_splits(train_df, val_df, test_df)
    train_sample_weight = _compute_recency_sample_weights(train_df, half_life_days=half_life_days,)

    feature_columns = get_feature_columns(surface_specific=surface_specific)

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_val = val_df[feature_columns].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    search = tune_xgb_classifier(X_train=X_train, y_train=y_train, sample_weight=train_sample_weight, n_splits=3,
        refit_metric="neg_log_loss", random_state=42, search_profile=search_profile, )


    tuned_config = xgb_config_from_gridsearch_best_params(search.best_params_)


    artifact = fit_xgb_classifier(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, feature_columns=feature_columns,
        config=tuned_config, sample_weight=train_sample_weight, )

    raw_val_pred_prob = predict_proba_from_xgb_artifact(artifact, X_val)
    raw_test_pred_prob = predict_proba_from_xgb_artifact(artifact, X_test)

    raw_val_metrics = _compute_binary_metrics(y_val, raw_val_pred_prob, "validation_raw")
    raw_test_metrics = _compute_binary_metrics(y_test, raw_test_pred_prob, "test_raw")

    isotonic_artifact = fit_isotonic_calibrator(pred_prob=raw_val_pred_prob, y_true=y_val, )
    isotonic_val_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_val_pred_prob)
    isotonic_test_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_test_pred_prob)

    isotonic_val_metrics = _compute_binary_metrics(y_val, isotonic_val_pred_prob, "validation_isotonic", )
    isotonic_test_metrics = _compute_binary_metrics(y_test, isotonic_test_pred_prob, "test_isotonic", )

    sigmoid_artifact = fit_sigmoid_calibrator(pred_prob=raw_val_pred_prob, y_true=y_val, )
    sigmoid_val_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_val_pred_prob)
    sigmoid_test_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_test_pred_prob)

    sigmoid_val_metrics = _compute_binary_metrics(y_val, sigmoid_val_pred_prob, "validation_sigmoid", )
    sigmoid_test_metrics = _compute_binary_metrics(y_test, sigmoid_test_pred_prob, "test_sigmoid", )

    if isotonic_val_metrics["log_loss"] <= sigmoid_val_metrics["log_loss"]:
        calibrator_artifact = isotonic_artifact
        val_metrics = isotonic_val_metrics
        test_metrics = isotonic_test_metrics
        chosen_calibration_method = "isotonic"
    else:
        calibrator_artifact = sigmoid_artifact
        val_metrics = sigmoid_val_metrics
        test_metrics = sigmoid_test_metrics
        chosen_calibration_method = "sigmoid"

    all_metrics_df = _metrics_table_from_dicts([
        raw_val_metrics,
        raw_test_metrics,
        isotonic_val_metrics,
        isotonic_test_metrics,
        sigmoid_val_metrics,
        sigmoid_test_metrics,
        val_metrics,
        test_metrics, ])

    importance_df = feature_importance_from_xgb_artifact(artifact, importance_type="gain", )
    cv_results_df = _prepare_gridsearch_results_df(search.cv_results_)

    suffix = _build_artifact_suffix(source=source, surface=surface)

    model_path = models_dir / f"{tour}_xgb_tuned{suffix}.joblib"
    isotonic_calibrator_path = models_dir / f"{tour}_xgb_tuned{suffix}_isotonic_calibrator.joblib"
    sigmoid_calibrator_path = models_dir / f"{tour}_xgb_tuned{suffix}_sigmoid_calibrator.joblib"
    metrics_path = models_dir / f"{tour}_xgb_tuned{suffix}_metrics.json"
    metrics_table_path = models_dir / f"{tour}_xgb_tuned{suffix}_metrics.csv"
    importance_path = models_dir / f"{tour}_xgb_tuned{suffix}_feature_importance.csv"
    cv_results_path = models_dir / f"{tour}_xgb_tuned{suffix}_cv_results.csv"
    metadata_path = models_dir / f"{tour}_xgb_tuned{suffix}_meta.json"

    save_model_artifact(artifact, model_path)
    save_model_artifact(isotonic_artifact, isotonic_calibrator_path)
    save_model_artifact(sigmoid_artifact, sigmoid_calibrator_path)

    save_metrics_json({
            "tour": tour,
            "model_type": "xgboost_tuned",
            "best_params": search.best_params_,
            "best_cv_score_neg_log_loss": float(search.best_score_),
            "raw_validation": raw_val_metrics,
            "raw_test": raw_test_metrics,
            "isotonic_validation": isotonic_val_metrics,
            "isotonic_test": isotonic_test_metrics,
            "sigmoid_validation": sigmoid_val_metrics,
            "sigmoid_test": sigmoid_test_metrics,
            "chosen_calibration_method": chosen_calibration_method,
            "chosen_validation": val_metrics,
            "chosen_test": test_metrics,
        }, metrics_path, )

    save_dataframe_csv(all_metrics_df, metrics_table_path)
    save_dataframe_csv(importance_df, importance_path)
    save_dataframe_csv(cv_results_df, cv_results_path)

    metadata = {
        "tour": tour,
        "phase": "4B",
        "model_type": "xgboost_tuned",
        "source_backbone": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "validation_date_min": str(val_df["tourney_date"].min().date()),
        "validation_date_max": str(val_df["tourney_date"].max().date()),
        "test_date_min": str(test_df["tourney_date"].min().date()),
        "test_date_max": str(test_df["tourney_date"].max().date()),
        "split_summary": split_summary.to_dict(orient="records"),
        "best_params": search.best_params_,
        "best_cv_score_neg_log_loss": float(search.best_score_),
        "cv_refit_metric": "neg_log_loss",
        "cv_n_splits": 3,
        "xgb_search_profile": search_profile,
        "recency_weighting": {"enabled": True, "half_life_days": half_life_days,
            "train_weight_summary": _summarize_sample_weights(train_sample_weight), },
        "best_iteration": artifact.get("best_iteration"),
        "xgb_config": artifact.get("config"),
        "available_calibrations": ["isotonic", "sigmoid"],
        "chosen_calibration_method": chosen_calibration_method,
        "calibration_fit_split": "validation",
    }
    save_metadata_json(metadata, metadata_path)

    return {
        "tour": tour,
        "source": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "model_type": "xgb_tuned",
        "model_path": str(model_path),
        "isotonic_calibrator_path": str(isotonic_calibrator_path),
        "sigmoid_calibrator_path": str(sigmoid_calibrator_path),
        "chosen_calibration_method": chosen_calibration_method,
        "metrics_path": str(metrics_path),
        "metrics_table_path": str(metrics_table_path),
        "importance_path": str(importance_path),
        "cv_results_path": str(cv_results_path),
        "metadata_path": str(metadata_path),
        "best_params": search.best_params_,
        "best_cv_score_neg_log_loss": float(search.best_score_),
        "xgb_search_profile": search_profile,
        "raw_validation_metrics": raw_val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "isotonic_validation_metrics": isotonic_val_metrics,
        "isotonic_test_metrics": isotonic_test_metrics,
        "sigmoid_validation_metrics": sigmoid_val_metrics,
        "sigmoid_test_metrics": sigmoid_test_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }



def train_xgb_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None, 
                       half_life_days: int = 730, ) -> dict:
    """
    Train the Phase 4A XGBoost baseline for one tour and save artifacts.
    """
    tour, source = _validate_tour_and_source(tour, source)
    surface = _normalize_surface(surface)
    surface_specific = surface is not None

    models_dir = project_root / "data" / "models"
    df = load_baseline_dataframe(project_root, tour, source=source, surface=surface)

    train_df, val_df, test_df = chronological_train_val_test_split(df)
    split_summary = summarize_all_splits(train_df, val_df, test_df)
    train_sample_weight = _compute_recency_sample_weights(train_df, half_life_days=half_life_days,)

    feature_columns = get_feature_columns(surface_specific=surface_specific)

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_val = val_df[feature_columns].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    artifact = fit_xgb_classifier(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, feature_columns=feature_columns,
        sample_weight=train_sample_weight, )

    raw_val_pred_prob = predict_proba_from_xgb_artifact(artifact, X_val)
    raw_test_pred_prob = predict_proba_from_xgb_artifact(artifact, X_test)

    raw_val_metrics = _compute_binary_metrics(y_val, raw_val_pred_prob, "validation_raw")
    raw_test_metrics = _compute_binary_metrics(y_test, raw_test_pred_prob, "test_raw")

    isotonic_artifact = fit_isotonic_calibrator(pred_prob=raw_val_pred_prob, y_true=y_val, )
    isotonic_val_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_val_pred_prob, )
    isotonic_test_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_test_pred_prob, )

    isotonic_val_metrics = _compute_binary_metrics(y_val, isotonic_val_pred_prob, "validation_isotonic", )
    isotonic_test_metrics = _compute_binary_metrics(y_test, isotonic_test_pred_prob, "test_isotonic",)

    sigmoid_artifact = fit_sigmoid_calibrator(pred_prob=raw_val_pred_prob, y_true=y_val, )
    sigmoid_val_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_val_pred_prob, )
    sigmoid_test_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_test_pred_prob, )

    sigmoid_val_metrics = _compute_binary_metrics(y_val, sigmoid_val_pred_prob, "validation_sigmoid", )
    sigmoid_test_metrics = _compute_binary_metrics(y_test, sigmoid_test_pred_prob, "test_sigmoid", )

    if isotonic_val_metrics["log_loss"] <= sigmoid_val_metrics["log_loss"]:
        calibrator_artifact = isotonic_artifact
        calibrated_val_pred_prob = isotonic_val_pred_prob
        calibrated_test_pred_prob = isotonic_test_pred_prob
        val_metrics = isotonic_val_metrics
        test_metrics = isotonic_test_metrics
        chosen_calibration_method = "isotonic"
    else:
        calibrator_artifact = sigmoid_artifact
        calibrated_val_pred_prob = sigmoid_val_pred_prob
        calibrated_test_pred_prob = sigmoid_test_pred_prob
        val_metrics = sigmoid_val_metrics
        test_metrics = sigmoid_test_metrics
        chosen_calibration_method = "sigmoid"

    all_metrics_df = _metrics_table_from_dicts([
            raw_val_metrics,
            raw_test_metrics,
            isotonic_val_metrics,
            isotonic_test_metrics,
            sigmoid_val_metrics,
            sigmoid_test_metrics,
            val_metrics,
            test_metrics, ])

    importance_df = feature_importance_from_xgb_artifact(artifact, importance_type="gain", )


    suffix = _build_artifact_suffix(source=source, surface=surface)

    model_path = models_dir / f"{tour}_xgb_baseline{suffix}.joblib"
    isotonic_calibrator_path = models_dir / f"{tour}_xgb_baseline{suffix}_isotonic_calibrator.joblib"
    sigmoid_calibrator_path = models_dir / f"{tour}_xgb_baseline{suffix}_sigmoid_calibrator.joblib"
    metrics_path = models_dir / f"{tour}_xgb_baseline{suffix}_metrics.json"
    metrics_table_path = models_dir / f"{tour}_xgb_baseline{suffix}_metrics.csv"
    importance_path = models_dir / f"{tour}_xgb_baseline{suffix}_feature_importance.csv"
    metadata_path = models_dir / f"{tour}_xgb_baseline{suffix}_meta.json"

    save_model_artifact(artifact, model_path)
    save_model_artifact(isotonic_artifact, isotonic_calibrator_path)
    save_model_artifact(sigmoid_artifact, sigmoid_calibrator_path)

    save_metrics_json(
        {
            "tour": tour,
            "model_type": "xgboost",
            "raw_validation": raw_val_metrics,
            "raw_test": raw_test_metrics,
            "isotonic_validation": isotonic_val_metrics,
            "isotonic_test": isotonic_test_metrics,
            "sigmoid_validation": sigmoid_val_metrics,
            "sigmoid_test": sigmoid_test_metrics,
            "chosen_calibration_method": chosen_calibration_method,
            "chosen_validation": val_metrics,
            "chosen_test": test_metrics,
        },
        metrics_path,
    )

    save_dataframe_csv(all_metrics_df, metrics_table_path)
    save_dataframe_csv(importance_df, importance_path)

    metadata = {
        "tour": tour,
        "phase": "4A",
        "model_type": "xgboost",
        "source_backbone": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "validation_date_min": str(val_df["tourney_date"].min().date()),
        "validation_date_max": str(val_df["tourney_date"].max().date()),
        "test_date_min": str(test_df["tourney_date"].min().date()),
        "test_date_max": str(test_df["tourney_date"].max().date()),
        "split_summary": split_summary.to_dict(orient="records"),
        "recency_weighting": {"enabled": True, "half_life_days": half_life_days,
            "train_weight_summary": _summarize_sample_weights(train_sample_weight), },
        "best_iteration": artifact.get("best_iteration"),
        "xgb_config": artifact.get("config"),
        "available_calibrations": ["isotonic", "sigmoid"],
        "chosen_calibration_method": chosen_calibration_method,
        "calibration_fit_split": "validation",
    }
    save_metadata_json(metadata, metadata_path)

    return {
        "tour": tour,
        "source": source,
        "surface_filter": surface,
        "surface_specific_model": surface_specific,
        "model_type": "xgb",
        "model_path": str(model_path),
        "isotonic_calibrator_path": str(isotonic_calibrator_path),
        "sigmoid_calibrator_path": str(sigmoid_calibrator_path),
        "chosen_calibration_method": chosen_calibration_method,
        "metrics_path": str(metrics_path),
        "metrics_table_path": str(metrics_table_path),
        "importance_path": str(importance_path),
        "metadata_path": str(metadata_path),
        "raw_validation_metrics": raw_val_metrics,
        "raw_test_metrics": raw_test_metrics,
        "isotonic_validation_metrics": isotonic_val_metrics,
        "isotonic_test_metrics": isotonic_test_metrics,
        "sigmoid_validation_metrics": sigmoid_val_metrics,
        "sigmoid_test_metrics": sigmoid_test_metrics,
        "chosen_calibration_method": chosen_calibration_method,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }