from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_cli.models.dataset import build_training_matrices, get_feature_columns

from sklearn.model_selection import GridSearchCV
from tennis_cli.models.split import build_inner_time_series_cv



@dataclass
class LogisticConfig:
    max_iter: int = 2000
    solver: str = "lbfgs"
    C: float = 1.0
    class_weight: str | None = None
    random_state: int = 42
    


def build_logistic_pipeline(config: LogisticConfig | None = None) -> Pipeline:
    """
    Build the Phase 3 logistic regression pipeline.

    Steps:
    1. Median imputation for numeric missing values
    2. Standard scaling
    3. Logistic regression
    """
    if config is None:
        config = LogisticConfig()

    pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()),
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
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],}]




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

    pipeline = build_logistic_pipeline(config=config)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight.to_numpy()

    pipeline.fit(X_train, y_train, **fit_kwargs)

    return pipeline, X_train, y_train, meta_train



def tune_logistic_baseline(train_df: pd.DataFrame, config: LogisticConfig | None = None, surface_specific: bool = False,
    n_splits: int = 3, refit_metric: str = "neg_log_loss", sample_weight: pd.Series | None = None,
    ) -> tuple[GridSearchCV, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Tune logistic regression on the training split only, using an inner
    time-series cross-validation loop.

    Returns:
    - fitted GridSearchCV object
    - X_train
    - y_train
    - meta_train
    """
    X_train, y_train, meta_train = build_training_matrices(train_df, surface_specific=surface_specific, )

    pipeline = build_logistic_pipeline(config=config)
    cv = build_inner_time_series_cv(n_splits=n_splits)

    search = GridSearchCV(estimator=pipeline,
        param_grid=get_logistic_param_grid(),
        scoring={"neg_log_loss": "neg_log_loss", "roc_auc": "roc_auc", "accuracy": "accuracy",},
        refit=refit_metric,
        cv=cv,
        n_jobs=-1,
        verbose=1, )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight.to_numpy()

    search.fit(X_train, y_train, **fit_kwargs)

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
    preprocessor = pipeline[:-1]

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"feature_{i}" for i in range(len(model.coef_[0]))]

    coefs = model.coef_[0]

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs, })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df