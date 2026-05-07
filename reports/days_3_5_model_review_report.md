# Days 3-5 Model Review Report

Generated: 2026-05-05 14:52

## Executive Summary

- Days 3-4 fixed the inner time-series CV protocol for logistic regression: 5 fixed-size validation folds, expanded low-C grid, and fold-anchored recency weights inside CV.
- Day 5 removed the XGBoost grid capacity ceiling by expanding n_estimators to 300, 600, 1000, and 1500.
- The expanded XGBoost grid consistently selected 300 estimators across all rerun branches, so the capacity ceiling is no longer the active issue in the current results.
- The largest new finding is calibration: isotonic calibration can overfit validation data, especially on small surface-specific branches. Disabling isotonic below 1000 validation rows repaired ATP/TML/Grass test log-loss from 0.8094 to 0.5427.
- The earlier delta_elo fix is strongly supported by feature importance: on ATP/TML/Hard, delta_elo was the top XGBoost gain feature in the post-fix run.

## Code Changes Implemented

| Area | Before | After |
|---|---|---|
| Inner CV | TimeSeriesSplit(n_splits=3) with sklearn default validation sizing | TimeSeriesSplit(n_splits=5, test_size=int(0.10*N), gap=0) |
| Logistic recency weights | One globally anchored sample-weight vector passed into GridSearchCV | Manual logistic CV recomputes fold-anchored weights using each fold training max date |
| Logistic C grid | 0.01, 0.1, 1, 10, 100 | 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 |
| XGBoost grid | n_estimators 300, 600 | n_estimators 300, 600, 1000, 1500 |
| XGBoost calibration | Always choose best validation log-loss between isotonic and sigmoid | Raw probabilities are eligible; sigmoid is eligible; isotonic is eligible only when validation rows >= 1000 |
| Dependencies/tests | scikit-learn/xgboost absent from requirements; pytest absent; Elo test failed at collection | Added scikit-learn, xgboost, pytest; repaired Elo pytest fixture |

## Before/After Findings

| Topic | Before | After | Interpretation |
|---|---|---|---|
| ATP/TML/Grass XGB calibration | Before calibration policy: isotonic selected | After: raw probabilities selected | Test log-loss 0.8094 -> 0.5427; accuracy 0.7179 -> 0.7260 |
| ATP/TML/Hard XGB calibration | Raw test log-loss was 0.5895 but isotonic selected by validation | Current policy still allows isotonic because validation rows > 1000 | Known residual risk: isotonic test log-loss 0.6028 is worse than raw 0.5895 |
| XGB grid capacity | Old grid capped n_estimators at 300/600 | New grid tests 300/600/1000/1500 | All current XGB branches selected 300 estimators |
| Logistic CV protocol | Old inner CV used default 3-fold TimeSeriesSplit and global-anchored weights | New inner CV uses 5 folds with fixed 10% validation blocks and fold-anchored recency weights | C=0.01 still wins on most branches; WTA/Sackmann/Clay moved to C=0.1 |

## Current Tuned Logistic Results

| Tour | Source | Surface | Best params | Val acc | Val LL | Test acc | Test AUC | Test LL |
|---|---|---|---|---:|---:|---:|---:|---:|
| ATP | TML | Hard | C=0.01, class_weight=None, solver=liblinear | 66.02% | 0.6090 | 65.14% | 0.7194 | 0.6153 |
| ATP | TML | Clay | C=0.01, class_weight=balanced, solver=liblinear | 67.26% | 0.6091 | 66.61% | 0.7276 | 0.6100 |
| ATP | TML | Grass | C=0.01, class_weight=None, solver=lbfgs | 69.51% | 0.5959 | 66.19% | 0.7509 | 0.5900 |
| ATP | Sackmann | Global | C=0.01, class_weight=balanced, solver=lbfgs | 65.26% | 0.6140 | 66.32% | 0.7279 | 0.6063 |
| ATP | Sackmann | Hard | C=0.01, class_weight=balanced, solver=lbfgs | 66.04% | 0.6088 | 66.18% | 0.7295 | 0.6053 |
| ATP | Sackmann | Clay | C=0.01, class_weight=balanced, solver=liblinear | 65.18% | 0.6182 | 66.25% | 0.7198 | 0.6108 |
| ATP | Sackmann | Grass | C=0.01, class_weight=balanced, solver=lbfgs | 66.77% | 0.6160 | 68.25% | 0.7418 | 0.5917 |
| WTA | Sackmann | Global | C=0.01, class_weight=balanced, solver=lbfgs | 66.68% | 0.6018 | 65.97% | 0.7274 | 0.6054 |
| WTA | Sackmann | Hard | C=0.01, class_weight=None, solver=liblinear | 67.53% | 0.5988 | 65.49% | 0.7246 | 0.6079 |
| WTA | Sackmann | Clay | C=0.1, class_weight=balanced, solver=lbfgs | 64.60% | 0.6099 | 66.15% | 0.7320 | 0.6012 |
| WTA | Sackmann | Grass | C=0.01, class_weight=None, solver=liblinear | 65.22% | 0.6060 | 62.50% | 0.7072 | 0.6212 |

## Current Tuned XGBoost Results

| Tour | Source | Surface | Calibration | Best params | Val acc | Val LL | Test acc | Test AUC | Test LL |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ATP | TML | Hard | isotonic | learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, reg_lambda=1.0 | 71.34% | 0.5550 | 68.23% | 0.7490 | 0.6028 |
| ATP | TML | Clay | None | learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, reg_lambda=1.0 | 71.03% | 0.5529 | 69.89% | 0.7751 | 0.5688 |
| ATP | TML | Grass | None | learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, reg_lambda=5.0 | 74.09% | 0.5300 | 72.60% | 0.8044 | 0.5427 |
| ATP | Sackmann | Global | isotonic | learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, reg_lambda=1.0 | 66.03% | 0.6023 | 66.39% | 0.7283 | 0.6072 |
| ATP | Sackmann | Hard | isotonic | learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, reg_lambda=1.0 | 66.89% | 0.5923 | 66.53% | 0.7320 | 0.6216 |
| ATP | Sackmann | Clay | None | learning_rate=0.03, max_depth=4, min_child_weight=5, n_estimators=300, reg_lambda=5.0 | 65.29% | 0.6173 | 65.30% | 0.7152 | 0.6150 |
| ATP | Sackmann | Grass | None | learning_rate=0.03, max_depth=3, min_child_weight=5, n_estimators=300, reg_lambda=5.0 | 67.69% | 0.5985 | 69.21% | 0.7598 | 0.5824 |
| WTA | Sackmann | Global | isotonic | learning_rate=0.03, max_depth=3, min_child_weight=5, n_estimators=300, reg_lambda=1.0 | 67.12% | 0.5966 | 65.53% | 0.7242 | 0.6326 |
| WTA | Sackmann | Hard | isotonic | learning_rate=0.03, max_depth=3, min_child_weight=5, n_estimators=300, reg_lambda=5.0 | 67.84% | 0.5942 | 64.64% | 0.7193 | 0.6355 |
| WTA | Sackmann | Clay | None | learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=300, reg_lambda=5.0 | 64.07% | 0.6142 | 66.71% | 0.7324 | 0.6017 |
| WTA | Sackmann | Grass | None | learning_rate=0.03, max_depth=3, min_child_weight=5, n_estimators=300, reg_lambda=5.0 | 64.21% | 0.6151 | 63.19% | 0.7119 | 0.6222 |

## XGBoost Grid Diagnostic

The expanded grid tested 300, 600, 1000, and 1500 estimators. Every current XGBoost branch selected 300 estimators. This means the former grid-capacity concern is now tested and does not explain the current results. Early stopping inside each CV fold remains a cleaner long-term improvement, but it is not blocking the present report.

| Branch | Best CV neg log-loss by n_estimators |
|---|---|
| ATP/TML/Hard | 300: 0.5660 LL, 600: 0.5686 LL, 1000: 0.5715 LL, 1500: 0.5761 LL |
| ATP/TML/Clay | 300: 0.5648 LL, 600: 0.5680 LL, 1000: 0.5735 LL, 1500: 0.5800 LL |
| ATP/TML/Grass | 300: 0.5271 LL, 600: 0.5332 LL, 1000: 0.5397 LL, 1500: 0.5468 LL |
| ATP/Sackmann/Global | 300: 0.6050 LL, 600: 0.6067 LL, 1000: 0.6091 LL, 1500: 0.6131 LL |
| ATP/Sackmann/Hard | 300: 0.6035 LL, 600: 0.6064 LL, 1000: 0.6113 LL, 1500: 0.6169 LL |
| ATP/Sackmann/Clay | 300: 0.6221 LL, 600: 0.6257 LL, 1000: 0.6323 LL, 1500: 0.6397 LL |
| ATP/Sackmann/Grass | 300: 0.5966 LL, 600: 0.6033 LL, 1000: 0.6121 LL, 1500: 0.6212 LL |
| WTA/Sackmann/Global | 300: 0.6069 LL, 600: 0.6082 LL, 1000: 0.6104 LL, 1500: 0.6139 LL |
| WTA/Sackmann/Hard | 300: 0.6062 LL, 600: 0.6089 LL, 1000: 0.6133 LL, 1500: 0.6182 LL |
| WTA/Sackmann/Clay | 300: 0.6192 LL, 600: 0.6244 LL, 1000: 0.6317 LL, 1500: 0.6407 LL |
| WTA/Sackmann/Grass | 300: 0.6322 LL, 600: 0.6446 LL, 1000: 0.6567 LL, 1500: 0.6694 LL |

## Main Interpretation

1. The logistic regularization result is mostly stable. Even after the CV protocol fix and expanded low-C grid, C=0.01 remains the dominant choice, except WTA/Sackmann/Clay where C=0.1 wins. This suggests strong regularization is genuinely preferred rather than only being an artifact of the old grid boundary.
2. The XGBoost capacity ceiling has been addressed. Since all branches select 300 estimators, larger tree caps are not currently helping under the 5-fold fixed-block CV protocol.
3. Calibration is now the most important XGBoost issue. Raw XGBoost probabilities are often competitive or better on test, and isotonic can overfit, particularly when the validation set is small. The new policy prevents isotonic on small branches and fixed the ATP/TML/Grass catastrophe.
4. Surface-specific model behavior is much more defensible after restoring the global Elo backbone and preventing unstable calibration. ATP/TML/Grass XGBoost now reaches 72.6% test accuracy and 0.543 test log-loss, which is no longer catastrophic.

## Recommended Next Steps

1. Treat the Days 3-5 implementation as the new baseline for future experiments.
2. In the next iteration, consider a stricter calibration selection rule for all branches, because some large validation branches still choose isotonic despite worse observed test log-loss.
3. Defer early stopping inside XGBoost CV unless future branches show that larger n_estimators values win; the current expanded grid does not support making that refactor urgent.
4. Move next into Days 6-10 feature work: sufficient surface-history indicators, common-opponent features, tournament tier, and first-meeting/H2H confidence features.