# Day 6 Remediation: Fold-Anchored Recency for XGBoost CV

Day 6 ports the logistic fold-anchored recency protocol to XGBoost tuning. XGBoost no longer uses `GridSearchCV` with one global sample-weight vector. It now runs a manual `ParameterGrid` loop over inner `TimeSeriesSplit` folds, recomputes recency weights for each fold using that fold's training max date, scores each fold by validation log loss, picks the minimum mean fold log loss, and refits on the full training partition with the full training weights.

## `tune_xgb_classifier` Policy Diff

```diff
-from sklearn.model_selection import GridSearchCV
+from sklearn.model_selection import ParameterGrid
+from sklearn.base import clone
+from tennis_cli.models._recency import compute_recency_weights_from_dates
```

```diff
-def tune_xgb_classifier(..., surface_specific: bool = False) -> GridSearchCV:
-    pipeline = build_xgb_search_pipeline(...)
-    cv = build_inner_time_series_cv(...)
-    search = GridSearchCV(
-        estimator=pipeline,
-        param_grid=get_xgb_param_grid(...),
-        scoring={"neg_log_loss": "neg_log_loss", "roc_auc": "roc_auc", "accuracy": "accuracy"},
-        refit=refit_metric,
-        cv=cv,
-        n_jobs=-1,
-        verbose=1,
-    )
-    search.fit(X_train, y_train, model__sample_weight=sample_weight.to_numpy())
-    return search
+def tune_xgb_classifier(..., train_dates: pd.Series | None = None, half_life_days: int = 730) -> XGBCVSearchResult:
+    if train_dates is None:
+        raise ValueError("train_dates is required for fold-anchored recency weights.")
+    pipeline = build_xgb_search_pipeline(...)
+    cv = build_inner_time_series_cv(...)
+    param_candidates = list(ParameterGrid(get_xgb_param_grid(...)))
+    fold_splits = list(cv.split(X_train, y_train))
+    for params in param_candidates:
+        for fold_train_idx, fold_val_idx in fold_splits:
+            fold_pipeline = clone(pipeline).set_params(**params)
+            fold_weight = compute_recency_weights_from_dates(
+                train_dates.iloc[fold_train_idx],
+                half_life_days=half_life_days,
+            )
+            fold_pipeline.fit(fold_X_train, fold_y_train, model__sample_weight=fold_weight)
+            val_prob = fold_pipeline.predict_proba(fold_X_val)[:, 1]
+            split_neg_log_loss.append(-log_loss(fold_y_val, val_prob, labels=[0, 1]))
+    best_pipeline = clone(pipeline).set_params(**best_params)
+    best_pipeline.fit(X_train, y_train, model__sample_weight=sample_weight.to_numpy())
+    return XGBCVSearchResult(...)
```

The shared fold-recency helper now lives in `src/tennis_cli/models/_recency.py`, and logistic imports the same helper instead of carrying a private copy.

## Best `n_estimators` Under Fold-Anchored CV

| tour | source | surface | best_n_estimators | best_learning_rate | best_max_depth | best_min_child_weight | best_reg_lambda | best_cv_neg_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atp | sackmann | Hard | 300 | 0.030000 | 3 | 5 | 1.000000 | -0.601677 |
| atp | sackmann | Clay | 300 | 0.030000 | 3 | 1 | 5.000000 | -0.618665 |
| atp | sackmann | Grass | 300 | 0.030000 | 3 | 1 | 5.000000 | -0.610715 |
| atp | tml | Hard | 300 | 0.030000 | 3 | 5 | 1.000000 | -0.604153 |
| atp | tml | Clay | 300 | 0.030000 | 3 | 1 | 5.000000 | -0.623971 |
| atp | tml | Grass | 300 | 0.030000 | 3 | 1 | 5.000000 | -0.618723 |
| wta | sackmann | Hard | 300 | 0.030000 | 3 | 5 | 5.000000 | -0.605613 |
| wta | sackmann | Clay | 300 | 0.030000 | 3 | 5 | 1.000000 | -0.616177 |
| wta | sackmann | Grass | 300 | 0.030000 | 3 | 5 | 5.000000 | -0.632519 |

The protocol change did not move `n_estimators` away from 300 on any branch. This is a finding: once fold-anchored recency weights are made symmetric with logistic CV, the current XGBoost grid still prefers shallow 300-tree models everywhere. On ATP/TML/Hard specifically, the best 300-tree candidate scored `-0.604153` mean CV neg log loss, while the best 600/1000/1500-tree candidates were worse at `-0.608580`, `-0.614816`, and `-0.621509`.
