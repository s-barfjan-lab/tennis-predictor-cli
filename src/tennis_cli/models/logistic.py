from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tennis_cli.models.dataset import build_training_matrices, get_feature_columns


@dataclass
class LogisticConfig:
    max_iter: int = 2000
    solver: str = "lbfgs"
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
            ("model",LogisticRegression(max_iter=config.max_iter,solver=config.solver,random_state=config.random_state,),),])
    
    return pipeline


def fit_logistic_baseline(train_df: pd.DataFrame,config: LogisticConfig | None = None,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Fit the logistic baseline on a training split dataframe.

    Returns:
    - fitted sklearn pipeline
    - X_train
    - y_train
    - meta_train
    """
    X_train, y_train, meta_train = build_training_matrices(train_df)

    pipeline = build_logistic_pipeline(config=config)
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, y_train, meta_train



def predict_split(pipeline: Pipeline, split_df: pd.DataFrame,) -> pd.DataFrame:
    """
    Run prediction for one split dataframe and return a tidy result table.

    Output columns:
    - y_true
    - prob_player_a_win
    - pred_player_a_win
    plus metadata columns
    """
    X, y, meta = build_training_matrices(split_df)

    prob = pipeline.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    out = meta.copy()
    out["y_true"] = y.values
    out["prob_player_a_win"] = prob
    out["pred_player_a_win"] = pred

    return out


def extract_logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract the fitted logistic regression coefficients in feature order.
    """
    feature_names = get_feature_columns()
    model = pipeline.named_steps["model"]

    coefs = model.coef_[0]

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs, "abs_coefficient": pd.Series(coefs).abs(), }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df