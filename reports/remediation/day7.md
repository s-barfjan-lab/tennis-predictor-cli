# Day 7 Remediation

Commit message:

```text
chore(model): drop redundant feature; bootstrap surface Elo; reconcile Elo config; fix inference paths
```

Day 7 completed the six cleanup changes, rebuilt the feature artifacts, and reran the 18-row baseline matrix against the Day 4 noise floor. The matrix did not remain within the Day 4 threshold: only 3 of 18 rows were within 1x Day 4 test-log-loss std, so this deliverable records the result as an audit failure rather than a clean hygiene-only pass.

## Six Changes And Diffs

- Dropped redundant H2H model input.
  Diff: removed `delta_h2h_losses` from `GLOBAL_FEATURE_COLUMNS` and `SURFACE_MODEL_FEATURE_COLUMNS` in `src/tennis_cli/models/dataset.py`; left the feature-generation column in `src/tennis_cli/features/baseline_features.py`.

- Added surface-history gating for surface rolling features.
  Diff: changed `_rolling_mean_past(..., min_periods=1)` in `src/tennis_cli/features/rolling.py`; surface rolling calls now pass `min_periods=5`; added `has_surface_history`; added the new input to `dataset.py`, baseline construction, inference snapshot construction, and rolling chronology tests.

- Reconciled Elo configuration.
  Diff: changed `src/tennis_cli/pipelines/build_elo.py` from its own `compute_elo_features(... EloConfig(k_factor=32.0) ...)` call to the shared `compute_all_elo_features(df)` path; removed the stale `compute_elo_features` import from `src/tennis_cli/pipelines/build_features.py`.

- Fixed tuned XGB inference loading.
  Diff: added `model_variant="baseline"` through `_get_xgb_artifact_paths`, `load_xgb_prediction_artifacts`, `predict_match_probability`, `predict_match_for_tour`, and the CLI `--model-variant baseline|tuned` option; tuned XGB now resolves paths such as `atp_xgb_tuned_tml_grass.joblib`.

- Bootstrapped first surface Elo from global Elo.
  Diff: in `src/tennis_cli/features/elo.py`, first-time surface ratings now use `0.70 * global_elo_pre + 0.30 * initial_rating` when global Elo exists; `compute_all_elo_features()` now passes the overall Elo-enriched dataframe into surface Elo; added an Elo regression test.

- Prevented walkovers/retirements from updating Elo.
  Diff: added `BAD_SCORE_TOKENS` and `_should_update_elo(score)` in `src/tennis_cli/features/elo.py`; overall and surface Elo loops now leave pre/post ratings unchanged and skip rating-dict updates for explicit `RET`, `W/O`, `WO`, `DEF`, `ABN`, `UNP`, and `CANC` scores; added an Elo regression test.

## 18-Row Before/After Table

CSV copy: `reports/remediation/day7_before_after.csv`.

| tour | source | surface | model | Day 4 LL mean | Day 4 LL std | Day 7 LL | delta | within 1x std |
|---|---|---|---|---:|---:|---:|---:|---|
| atp | sackmann | Hard | logit_baseline | 0.604483 | 0.000000 | 0.602986 | -0.001497 | no |
| atp | sackmann | Hard | xgb_baseline | 0.641440 | 0.019183 | 0.603671 | -0.037769 | no |
| atp | sackmann | Clay | logit_baseline | 0.614261 | 0.000000 | 0.615780 | 0.001519 | no |
| atp | sackmann | Clay | xgb_baseline | 0.618015 | 0.001308 | 0.621534 | 0.003520 | no |
| atp | sackmann | Grass | logit_baseline | 0.596310 | 0.000000 | 0.585441 | -0.010870 | no |
| atp | sackmann | Grass | xgb_baseline | 0.585582 | 0.003931 | 0.586422 | 0.000840 | yes |
| atp | tml | Hard | logit_baseline | 0.603321 | 0.000000 | 0.602367 | -0.000955 | no |
| atp | tml | Hard | xgb_baseline | 0.625964 | 0.015478 | 0.605591 | -0.020373 | no |
| atp | tml | Clay | logit_baseline | 0.613107 | 0.000000 | 0.616621 | 0.003514 | no |
| atp | tml | Clay | xgb_baseline | 0.618336 | 0.003921 | 0.620729 | 0.002393 | yes |
| atp | tml | Grass | logit_baseline | 0.588492 | 0.000000 | 0.578688 | -0.009804 | no |
| atp | tml | Grass | xgb_baseline | 0.592467 | 0.001498 | 0.588230 | -0.004237 | no |
| wta | sackmann | Hard | logit_baseline | 0.610653 | 0.000000 | 0.610515 | -0.000138 | no |
| wta | sackmann | Hard | xgb_baseline | 0.620784 | 0.011952 | 0.612234 | -0.008550 | yes |
| wta | sackmann | Clay | logit_baseline | 0.593403 | 0.000000 | 0.589239 | -0.004164 | no |
| wta | sackmann | Clay | xgb_baseline | 0.599580 | 0.001888 | 0.601972 | 0.002392 | no |
| wta | sackmann | Grass | logit_baseline | 0.647533 | 0.000000 | 0.653408 | 0.005875 | no |
| wta | sackmann | Grass | xgb_baseline | 0.641407 | 0.003613 | 0.635252 | -0.006155 | no |

## Noise Confirmation Per Change

- Dropping `delta_h2h_losses` is not confirmed within noise because the combined Day 7 rerun exceeded the Day 4 threshold in 15 of 18 rows.
- Surface rolling `min_periods=5` and `has_surface_history` are not confirmed within noise because multiple surface branches moved beyond the Day 4 threshold.
- Elo configuration reconciliation is not confirmed within noise because the Day 7 matrix was rerun after all six changes together and exceeded threshold.
- Tuned XGB inference loading is code-path hygiene and does not affect training metrics directly, but the full Day 7 rerun is not confirmed within noise.
- Surface Elo bootstrapping is not confirmed within noise because it intentionally changes feature values and likely contributes to the observed branch movement.
- Walkover/retirement Elo skipping is not confirmed within noise because it intentionally changes feature values and likely contributes to the observed branch movement.

## Verification

Feature artifacts rebuilt:

- `data/features/atp_long_2015_2025.parquet`
- `data/features/atp_baseline_2015_2025.parquet`
- `data/features/wta_long_2015_2025.parquet`
- `data/features/wta_baseline_2015_2025.parquet`
- `data/features/atp_long_tml_2015_2025.parquet`
- `data/features/atp_baseline_tml_2015_2025.parquet`

Test command:

```powershell
.\.venv\Scripts\python.exe -m pytest src/tennis_cli/tests/test_elo.py
```

Result: `3 passed`; pytest emitted the existing cache warning.
