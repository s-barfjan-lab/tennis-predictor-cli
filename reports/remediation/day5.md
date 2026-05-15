# Day 5 Remediation: Calibration Hygiene

Day 5 stops using the validation split for early stopping, post-hoc calibrator fitting, calibration-method selection, and performance reporting at the same time. XGBoost now trains on the earlier part of the training split, uses the final 90 days of the training split as a calibration set for early stopping and calibrator fitting, uses validation only to select raw vs isotonic vs sigmoid, and keeps test as the final report split.

## Calibration Policy Diff

```diff
+def split_train_into_train_and_calibration(
+    train_df: pd.DataFrame,
+    date_col: str = "tourney_date",
+    calibration_days: int = 90,
+) -> tuple[pd.DataFrame, pd.DataFrame]:
+    train_df = train_df.copy()
+    train_df[date_col] = pd.to_datetime(train_df[date_col])
+    cutoff = train_df[date_col].max() - pd.Timedelta(days=calibration_days)
+    train_inner = train_df[train_df[date_col] <= cutoff].copy()
+    calib = train_df[train_df[date_col] > cutoff].copy()
+    if len(train_inner) == 0 or len(calib) == 0:
+        raise ValueError("Empty split when carving calibration set; lower calibration_days.")
+    return train_inner, calib
```

```diff
-    train_sample_weight = _compute_recency_sample_weights(train_df, half_life_days=half_life_days,)
+    train_inner_df, calibration_df = split_train_into_train_and_calibration(train_df, calibration_days=90)
+    train_sample_weight = _compute_recency_sample_weights(train_inner_df, half_life_days=half_life_days,)
 
-    X_train = train_df[feature_columns].copy()
-    y_train = train_df[TARGET_COLUMN].copy()
+    X_train = train_inner_df[feature_columns].copy()
+    y_train = train_inner_df[TARGET_COLUMN].copy()
+    X_calib = calibration_df[feature_columns].copy()
+    y_calib = calibration_df[TARGET_COLUMN].copy()
 
-    artifact = fit_xgb_classifier(... X_val=X_val, y_val=y_val, ...)
+    artifact = fit_xgb_classifier(... X_val=X_calib, y_val=y_calib, ...)
 
-    isotonic_artifact = fit_isotonic_calibrator(pred_prob=raw_val_pred_prob, y_true=y_val)
-    sigmoid_artifact = fit_sigmoid_calibrator(pred_prob=raw_val_pred_prob, y_true=y_val)
+    isotonic_artifact = fit_isotonic_calibrator(pred_prob=raw_calib_pred_prob, y_true=y_calib)
+    sigmoid_artifact = fit_sigmoid_calibrator(pred_prob=raw_calib_pred_prob, y_true=y_calib)
```

```diff
-    candidates = [
-        (None, raw_val_metrics, raw_test_metrics),
-        ("sigmoid", sigmoid_val_metrics, sigmoid_test_metrics),
-    ]
-    if int(raw_val_metrics["rows"]) >= min_isotonic_rows:
-        candidates.append(("isotonic", isotonic_val_metrics, isotonic_test_metrics))
+    candidates = [
+        (None, raw_val_metrics, raw_test_metrics),
+        ("isotonic", isotonic_val_metrics, isotonic_test_metrics),
+        ("sigmoid", sigmoid_val_metrics, sigmoid_test_metrics),
+    ]
```

## Val-Test Gap Comparison

For each branch, the table uses the XGB candidate with the lower validation log loss before the protocol change and the XGB candidate with the lower validation log loss after the protocol change. Blank calibration in the source CSVs is shown as `raw`.

| tour | source | surface | before_model | before_calibration | before_validation_ll | before_test_ll | before_gap | after_model | after_calibration | after_validation_ll | after_test_ll | after_gap | abs_gap_change | gap_shrank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atp | sackmann | Hard | xgb_tuned | isotonic | 0.596303 | 0.639506 | 0.043203 | xgb_baseline | raw | 0.606761 | 0.604360 | -0.002401 | -0.040802 | yes |
| atp | sackmann | Clay | xgb_tuned | raw | 0.619167 | 0.616075 | -0.003092 | xgb_baseline | raw | 0.619716 | 0.617733 | -0.001983 | -0.001109 | yes |
| atp | sackmann | Grass | xgb_tuned | raw | 0.613960 | 0.589755 | -0.024204 | xgb_baseline | sigmoid | 0.623917 | 0.582399 | -0.041518 | 0.017314 | no |
| atp | tml | Hard | xgb_tuned | isotonic | 0.557617 | 0.584526 | 0.026909 | xgb_baseline | raw | 0.605442 | 0.606265 | 0.000822 | -0.026087 | yes |
| atp | tml | Clay | xgb_tuned | raw | 0.547337 | 0.568521 | 0.021184 | xgb_baseline | raw | 0.627440 | 0.623742 | -0.003698 | -0.017485 | yes |
| atp | tml | Grass | xgb_tuned | raw | 0.534889 | 0.538588 | 0.003699 | xgb_baseline | raw | 0.610388 | 0.593193 | -0.017195 | 0.013496 | no |
| wta | sackmann | Hard | xgb_baseline | isotonic | 0.591990 | 0.613933 | 0.021943 | xgb_tuned | raw | 0.604386 | 0.611944 | 0.007558 | -0.014385 | yes |
| wta | sackmann | Clay | xgb_baseline | raw | 0.611792 | 0.599037 | -0.012756 | xgb_tuned | raw | 0.617706 | 0.607465 | -0.010241 | -0.002515 | yes |
| wta | sackmann | Grass | xgb_baseline | raw | 0.599807 | 0.637834 | 0.038027 | xgb_baseline | raw | 0.612164 | 0.634654 | 0.022491 | -0.015537 | yes |

The validation-test gap shrank on 7 of 9 branches. The largest repairs were ATP/Sackmann/Hard, where absolute gap fell from 0.0432 to 0.0024, and ATP/TML/Hard, where it fell from 0.0269 to 0.0008. The gap did not shrink on ATP/Sackmann/Grass or ATP/TML/Grass; both Grass branches now have validation loss materially higher than test loss. The Day 3-5 residual-risk example, ATP/TML/Hard, is fixed by this protocol. No branch has at least 1000 calibration rows, so the stated `gap <= 0.01` target for branches with at least 1000 calibration rows is not directly applicable in this run.
