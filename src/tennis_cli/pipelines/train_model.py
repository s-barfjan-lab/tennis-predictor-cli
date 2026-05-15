from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from tennis_cli.models.dataset import (TARGET_COLUMN, get_feature_columns, load_baseline_dataframe, )
from tennis_cli.models.evaluate import (evaluate_multiple_splits, evaluate_predictions, )
from tennis_cli.models.io import (save_dataframe_csv, save_metadata_json, save_metrics_json, save_model_artifact, )
from tennis_cli.models.logistic import (LogisticConfig, extract_logistic_coefficients, fit_logistic_baseline, predict_split,
    tune_logistic_baseline, )
from tennis_cli.models.split import (
    chronological_train_val_test_split,
    split_train_into_train_and_calibration,
    summarize_all_splits,
)
from tennis_cli.models.xgboost_model import (XGBConfig, apply_isotonic_calibration, apply_sigmoid_calibration,
    feature_importance_from_xgb_artifact, fit_isotonic_calibrator, fit_sigmoid_calibrator, fit_xgb_classifier,
    predict_proba_from_xgb_artifact, tune_xgb_classifier, xgb_config_from_gridsearch_best_params, )


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


    
def _build_artifact_suffix(source: str, surface: str | None, artifact_tag: str | None = None) -> str:
    """
    Build a file suffix that keeps source / surface / experiment tag separate.

    Examples
    --------
    sackmann + None + None      -> ""
    tml + None + None           -> "_tml"
    sackmann + Clay + None      -> "_clay"
    tml + Grass + None          -> "_tml_grass"
    sackmann + Clay + seed7     -> "_clay_seed7"
    """
    parts = []

    if source != "sackmann":
        parts.append(source)

    if surface is not None:
        parts.append(surface.lower())

    if artifact_tag is not None:
        tag = str(artifact_tag).strip().lower().replace(" ", "_")
        if tag:
            parts.append(tag)

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


def _choose_xgb_calibration(
    raw_val_metrics: dict,
    raw_test_metrics: dict,
    sigmoid_val_metrics: dict,
    sigmoid_test_metrics: dict,
    isotonic_val_metrics: dict,
    isotonic_test_metrics: dict,
) -> tuple[str | None, dict, dict, dict]:
    """
    Pick the calibration method using validation log-loss.

    Raw probabilities are a valid candidate because XGBoost's logistic output
    is often already better calibrated than a noisy post-hoc calibrator.
    """
    candidates = [
        (None, raw_val_metrics, raw_test_metrics),
        ("isotonic", isotonic_val_metrics, isotonic_test_metrics),
        ("sigmoid", sigmoid_val_metrics, sigmoid_test_metrics),
    ]

    chosen_method, chosen_val, chosen_test = min(
        candidates,
        key=lambda item: item[1]["log_loss"],
    )

    policy = {
        "selection_metric": "validation_log_loss",
        "raw_eligible": True,
        "sigmoid_eligible": True,
        "isotonic_eligible": True,
    }
    return chosen_method, chosen_val, chosen_test, policy


def _prepare_gridsearch_results_df(cv_results: dict, sort_by: str = "rank_test_neg_log_loss") -> pd.DataFrame:
    """
    Convert sklearn GridSearchCV cv_results_ into a tidy dataframe.
    """
    df = pd.DataFrame(cv_results).copy()

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)

    return df



def _build_noise_floor_row(result: dict, seed: int, model: str, tour: str, source: str, surface: str | None, ) -> dict:
    row = {"seed": int(seed),
        "model": model,
        "tour": tour,
        "source": source,
        "surface": surface,
        "validation_accuracy": float(result["validation_metrics"]["accuracy"]),
        "validation_roc_auc": float(result["validation_metrics"]["roc_auc"]) if result["validation_metrics"]["roc_auc"] is not None else None,
        "validation_log_loss": float(result["validation_metrics"]["log_loss"]),
        "validation_brier_score": float(result["validation_metrics"]["brier_score"]),
        "test_accuracy": float(result["test_metrics"]["accuracy"]),
        "test_roc_auc": float(result["test_metrics"]["roc_auc"]) if result["test_metrics"]["roc_auc"] is not None else None,
        "test_log_loss": float(result["test_metrics"]["log_loss"]),
        "test_brier_score": float(result["test_metrics"]["brier_score"]), }

    if "chosen_calibration_method" in result:
        row["chosen_calibration_method"] = result["chosen_calibration_method"]

    if "raw_validation_metrics" in result:
        row["raw_validation_accuracy"] = float(result["raw_validation_metrics"]["accuracy"])
        row["raw_validation_roc_auc"] = float(result["raw_validation_metrics"]["roc_auc"]) if result["raw_validation_metrics"]["roc_auc"] is not None else None
        row["raw_validation_log_loss"] = float(result["raw_validation_metrics"]["log_loss"])
        row["raw_validation_brier_score"] = float(result["raw_validation_metrics"]["brier_score"])

    if "raw_test_metrics" in result:
        row["raw_test_accuracy"] = float(result["raw_test_metrics"]["accuracy"])
        row["raw_test_roc_auc"] = float(result["raw_test_metrics"]["roc_auc"]) if result["raw_test_metrics"]["roc_auc"] is not None else None
        row["raw_test_log_loss"] = float(result["raw_test_metrics"]["log_loss"])
        row["raw_test_brier_score"] = float(result["raw_test_metrics"]["brier_score"])

    return row


def _summarize_noise_floor(detail_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in detail_df.columns if pd.api.types.is_numeric_dtype(detail_df[c]) and c != "seed"]

    rows = []
    for col in numeric_cols:
        values = detail_df[col].dropna()
        if values.empty:
            continue

        rows.append({"metric": col,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()), })

    return pd.DataFrame(rows)



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
                            half_life_days: int = 730, random_state: int = 42, artifact_tag: str | None = None,) -> dict:
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

    
    config = LogisticConfig(random_state=random_state)

    pipeline, _, _, _ = fit_logistic_baseline(train_df, config=config, surface_specific=surface_specific, sample_weight=train_sample_weight, )
    

    val_preds = predict_split(pipeline, val_df, surface_specific=surface_specific, )
    test_preds = predict_split(pipeline, test_df, surface_specific=surface_specific, )

    val_metrics = evaluate_predictions(val_preds, "validation")
    test_metrics = evaluate_predictions(test_preds, "test")
    all_metrics_df = evaluate_multiple_splits({"validation": val_preds, "test": test_preds, })

    coef_df = extract_logistic_coefficients(pipeline)

    suffix = _build_artifact_suffix(source=source, surface=surface, artifact_tag=artifact_tag)

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
             "train_weight_summary": _summarize_sample_weights(train_sample_weight), }, 
        "random_state": random_state,
        "artifact_tag": artifact_tag, }
    
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
        "test_metrics": test_metrics, 
        "random_state": random_state,
        "artifact_tag": artifact_tag, }



def train_tuned_logistic_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None,
        half_life_days: int = 730, random_state: int = 42, artifact_tag: str | None = None, ) -> dict:
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


    config = LogisticConfig(random_state=random_state)
    search, _, _, _ = tune_logistic_baseline(train_df, config=config, surface_specific=surface_specific, n_splits=5,
        refit_metric="neg_log_loss", sample_weight=train_sample_weight, half_life_days=half_life_days, )

    best_pipeline = search.best_estimator_

    val_preds = predict_split(best_pipeline, val_df, surface_specific=surface_specific, )
    test_preds = predict_split(best_pipeline, test_df, surface_specific=surface_specific, )

    val_metrics = evaluate_predictions(val_preds, "validation")
    test_metrics = evaluate_predictions(test_preds, "test")
    all_metrics_df = evaluate_multiple_splits({"validation": val_preds, "test": test_preds})

    coef_df = extract_logistic_coefficients(best_pipeline)
    cv_results_df = _prepare_gridsearch_results_df(search.cv_results_)


    suffix = _build_artifact_suffix(source=source, surface=surface, artifact_tag=artifact_tag)
    
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
        "cv_n_splits": 5,
        "cv_validation_fraction": 0.10,
        "cv_recency_weight_anchor": "fold_train_max_date",
        "recency_weighting": {"enabled": True, "half_life_days": half_life_days,
            "train_weight_summary": _summarize_sample_weights(train_sample_weight), }, 
        "random_state": random_state,
        "artifact_tag": artifact_tag, }
    
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
        "random_state": random_state,
        "artifact_tag": artifact_tag, }



def train_tuned_xgb_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None,
        half_life_days: int = 730, search_profile: str = "base", random_state: int = 42, artifact_tag: str | None = None,) -> dict:
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
    train_inner_df, calibration_df = split_train_into_train_and_calibration(train_df, calibration_days=90)
    train_sample_weight = _compute_recency_sample_weights(train_inner_df, half_life_days=half_life_days,)

    feature_columns = get_feature_columns(surface_specific=surface_specific)

    X_train = train_inner_df[feature_columns].copy()
    y_train = train_inner_df[TARGET_COLUMN].copy()

    X_calib = calibration_df[feature_columns].copy()
    y_calib = calibration_df[TARGET_COLUMN].copy()

    X_val = val_df[feature_columns].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    search = tune_xgb_classifier(X_train=X_train, y_train=y_train, sample_weight=train_sample_weight, n_splits=5,
        refit_metric="neg_log_loss", random_state=random_state, search_profile=search_profile,
        surface_specific=surface_specific, train_dates=train_inner_df["tourney_date"], half_life_days=half_life_days, )
    

    tuned_config = xgb_config_from_gridsearch_best_params(search.best_params_, random_state=random_state, )

    artifact = fit_xgb_classifier(X_train=X_train, y_train=y_train, X_val=X_calib, y_val=y_calib, feature_columns=feature_columns,
        config=tuned_config, sample_weight=train_sample_weight, surface_specific=surface_specific, )

    raw_calib_pred_prob = predict_proba_from_xgb_artifact(artifact, X_calib)
    raw_val_pred_prob = predict_proba_from_xgb_artifact(artifact, X_val)
    raw_test_pred_prob = predict_proba_from_xgb_artifact(artifact, X_test)

    raw_val_metrics = _compute_binary_metrics(y_val, raw_val_pred_prob, "validation_raw")
    raw_test_metrics = _compute_binary_metrics(y_test, raw_test_pred_prob, "test_raw")

    isotonic_artifact = fit_isotonic_calibrator(pred_prob=raw_calib_pred_prob, y_true=y_calib, )
    isotonic_val_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_val_pred_prob)
    isotonic_test_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_test_pred_prob)

    isotonic_val_metrics = _compute_binary_metrics(y_val, isotonic_val_pred_prob, "validation_isotonic", )
    isotonic_test_metrics = _compute_binary_metrics(y_test, isotonic_test_pred_prob, "test_isotonic", )

    sigmoid_artifact = fit_sigmoid_calibrator(pred_prob=raw_calib_pred_prob, y_true=y_calib, )
    sigmoid_val_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_val_pred_prob)
    sigmoid_test_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_test_pred_prob)

    sigmoid_val_metrics = _compute_binary_metrics(y_val, sigmoid_val_pred_prob, "validation_sigmoid", )
    sigmoid_test_metrics = _compute_binary_metrics(y_test, sigmoid_test_pred_prob, "test_sigmoid", )

    chosen_calibration_method, val_metrics, test_metrics, calibration_selection_policy = _choose_xgb_calibration(
        raw_val_metrics=raw_val_metrics,
        raw_test_metrics=raw_test_metrics,
        sigmoid_val_metrics=sigmoid_val_metrics,
        sigmoid_test_metrics=sigmoid_test_metrics,
        isotonic_val_metrics=isotonic_val_metrics,
        isotonic_test_metrics=isotonic_test_metrics,
    )

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

    suffix = _build_artifact_suffix(source=source, surface=surface, artifact_tag=artifact_tag)

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
            "calibration_selection_policy": calibration_selection_policy,
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
        "train_inner_rows": int(len(train_inner_df)),
        "calibration_rows": int(len(calibration_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "train_inner_date_min": str(train_inner_df["tourney_date"].min().date()),
        "train_inner_date_max": str(train_inner_df["tourney_date"].max().date()),
        "calibration_date_min": str(calibration_df["tourney_date"].min().date()),
        "calibration_date_max": str(calibration_df["tourney_date"].max().date()),
        "validation_date_min": str(val_df["tourney_date"].min().date()),
        "validation_date_max": str(val_df["tourney_date"].max().date()),
        "test_date_min": str(test_df["tourney_date"].min().date()),
        "test_date_max": str(test_df["tourney_date"].max().date()),
        "split_summary": split_summary.to_dict(orient="records"),
        "best_params": search.best_params_,
        "best_cv_score_neg_log_loss": float(search.best_score_),
        "cv_refit_metric": "neg_log_loss",
        "cv_n_splits": 5,
        "cv_validation_fraction": 0.10,
        "xgb_search_profile": search_profile,
        "recency_weighting": {"enabled": True, "half_life_days": half_life_days,
            "train_weight_summary": _summarize_sample_weights(train_sample_weight), },
        "best_iteration": artifact.get("best_iteration"),
        "xgb_config": artifact.get("config"),
        "available_calibrations": ["isotonic", "sigmoid"],
        "chosen_calibration_method": chosen_calibration_method,
        "calibration_selection_policy": calibration_selection_policy,
        "calibration_fit_split": "training_tail_90_days",
        "random_state": random_state,
        "artifact_tag": artifact_tag, }
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
        "random_state": random_state,
        "artifact_tag": artifact_tag, }



def train_xgb_for_tour(project_root: Path, tour: str, source: str = "sackmann", surface: str | None = None, 
         half_life_days: int = 730, random_state: int = 42, artifact_tag: str | None = None, ) -> dict:
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
    train_inner_df, calibration_df = split_train_into_train_and_calibration(train_df, calibration_days=90)
    train_sample_weight = _compute_recency_sample_weights(train_inner_df, half_life_days=half_life_days,)

    feature_columns = get_feature_columns(surface_specific=surface_specific)

    X_train = train_inner_df[feature_columns].copy()
    y_train = train_inner_df[TARGET_COLUMN].copy()

    X_calib = calibration_df[feature_columns].copy()
    y_calib = calibration_df[TARGET_COLUMN].copy()

    X_val = val_df[feature_columns].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()


    config = XGBConfig(random_state=random_state)
    artifact = fit_xgb_classifier(X_train=X_train, y_train=y_train, X_val=X_calib, y_val=y_calib, feature_columns=feature_columns,
         config=config, sample_weight=train_sample_weight, surface_specific=surface_specific, )


    raw_calib_pred_prob = predict_proba_from_xgb_artifact(artifact, X_calib)
    raw_val_pred_prob = predict_proba_from_xgb_artifact(artifact, X_val)
    raw_test_pred_prob = predict_proba_from_xgb_artifact(artifact, X_test)

    raw_val_metrics = _compute_binary_metrics(y_val, raw_val_pred_prob, "validation_raw")
    raw_test_metrics = _compute_binary_metrics(y_test, raw_test_pred_prob, "test_raw")

    isotonic_artifact = fit_isotonic_calibrator(pred_prob=raw_calib_pred_prob, y_true=y_calib, )
    isotonic_val_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_val_pred_prob, )
    isotonic_test_pred_prob = apply_isotonic_calibration(isotonic_artifact, raw_test_pred_prob, )

    isotonic_val_metrics = _compute_binary_metrics(y_val, isotonic_val_pred_prob, "validation_isotonic", )
    isotonic_test_metrics = _compute_binary_metrics(y_test, isotonic_test_pred_prob, "test_isotonic",)

    sigmoid_artifact = fit_sigmoid_calibrator(pred_prob=raw_calib_pred_prob, y_true=y_calib, )
    sigmoid_val_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_val_pred_prob, )
    sigmoid_test_pred_prob = apply_sigmoid_calibration(sigmoid_artifact, raw_test_pred_prob, )

    sigmoid_val_metrics = _compute_binary_metrics(y_val, sigmoid_val_pred_prob, "validation_sigmoid", )
    sigmoid_test_metrics = _compute_binary_metrics(y_test, sigmoid_test_pred_prob, "test_sigmoid", )

    chosen_calibration_method, val_metrics, test_metrics, calibration_selection_policy = _choose_xgb_calibration(
        raw_val_metrics=raw_val_metrics,
        raw_test_metrics=raw_test_metrics,
        sigmoid_val_metrics=sigmoid_val_metrics,
        sigmoid_test_metrics=sigmoid_test_metrics,
        isotonic_val_metrics=isotonic_val_metrics,
        isotonic_test_metrics=isotonic_test_metrics,
    )

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


    suffix = _build_artifact_suffix(source=source, surface=surface, artifact_tag=artifact_tag)

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
            "calibration_selection_policy": calibration_selection_policy,
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
        "train_inner_rows": int(len(train_inner_df)),
        "calibration_rows": int(len(calibration_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["tourney_date"].min().date()),
        "train_date_max": str(train_df["tourney_date"].max().date()),
        "train_inner_date_min": str(train_inner_df["tourney_date"].min().date()),
        "train_inner_date_max": str(train_inner_df["tourney_date"].max().date()),
        "calibration_date_min": str(calibration_df["tourney_date"].min().date()),
        "calibration_date_max": str(calibration_df["tourney_date"].max().date()),
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
        "calibration_selection_policy": calibration_selection_policy,
        "calibration_fit_split": "training_tail_90_days",
        "random_state": random_state,
        "artifact_tag": artifact_tag, }
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
        "random_state": random_state,
        "artifact_tag": artifact_tag, }


def run_seed_noise_floor_experiment(project_root: Path, model: str, tour: str, source: str = "sackmann", surface: str | None = None,
    seeds: list[int] | None = None, half_life_days: int = 730, search_profile: str = "base", ) -> dict:
    """
    Re-run one fixed branch configuration across 5 random seeds and compute
    the standard deviation of test log-loss as the branch noise floor.

    This is meant to be run on the CURRENT BEST CONFIGURATION for that branch
    (for example: xgb_tuned on ATP/TML/Hard, or logit_tuned on a branch where
    logistic is currently best).

    Per-seed artifacts are saved with suffixes like _seed7, _seed17, etc.,
    so main artifacts are not overwritten.
    """
    tour, source = _validate_tour_and_source(tour, source)
    surface = _normalize_surface(surface)

    model = str(model).strip().lower()
    if model not in {"logit", "logit_tuned", "xgb", "xgb_tuned"}:
        raise ValueError("model must be one of: logit, logit_tuned, xgb, xgb_tuned")

    if seeds is None:
        seeds = [7, 17, 29, 42, 88]

    models_dir = project_root / "data" / "models"
    detail_rows = []

    for seed in seeds:
        common_kwargs = {"project_root": project_root,
            "tour": tour,
            "source": source,
            "surface": surface,
            "half_life_days": half_life_days,
            "random_state": int(seed),
            "artifact_tag": f"seed{int(seed)}", }

        if model == "logit":
            result = train_logistic_for_tour(**common_kwargs)
        elif model == "logit_tuned":
            result = train_tuned_logistic_for_tour(**common_kwargs)
        elif model == "xgb":
            result = train_xgb_for_tour(**common_kwargs)
        else:
            result = train_tuned_xgb_for_tour(**common_kwargs, search_profile=search_profile)

        detail_rows.append(_build_noise_floor_row(result=result, seed=int(seed), model=model, tour=tour, source=source, surface=surface, ))

    detail_df = pd.DataFrame(detail_rows)
    summary_df = _summarize_noise_floor(detail_df)

    suffix = _build_artifact_suffix(source=source, surface=surface)
    detail_path = models_dir / f"{tour}_{model}_noise_floor{suffix}_detail.csv"
    summary_path = models_dir / f"{tour}_{model}_noise_floor{suffix}_summary.csv"
    metadata_path = models_dir / f"{tour}_{model}_noise_floor{suffix}_meta.json"

    save_dataframe_csv(detail_df, detail_path)
    save_dataframe_csv(summary_df, summary_path)

    test_ll_row = summary_df[summary_df["metric"] == "test_log_loss"]
    test_acc_row = summary_df[summary_df["metric"] == "test_accuracy"]

    metadata = {
        "model": model,
        "tour": tour,
        "source": source,
        "surface": surface,
        "seeds": [int(s) for s in seeds],
        "half_life_days": half_life_days,
        "search_profile": search_profile if "xgb" in model else None,
        "noise_floor_test_log_loss_std": float(test_ll_row.iloc[0]["std"]) if not test_ll_row.empty else None,
        "noise_floor_test_accuracy_std": float(test_acc_row.iloc[0]["std"]) if not test_acc_row.empty else None,
    }

    save_metadata_json(metadata, metadata_path)

    return {
        "model": model,
        "tour": tour,
        "source": source,
        "surface": surface,
        "seeds": [int(s) for s in seeds],
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "metadata_path": str(metadata_path),
        "noise_floor_test_log_loss_std": metadata["noise_floor_test_log_loss_std"],
        "noise_floor_test_accuracy_std": metadata["noise_floor_test_accuracy_std"],
        "detail_df": detail_df,
        "summary_df": summary_df,
    }
