# In-Play Markov Extension Report

## Purpose

This is a separate, removable extension for live tennis match probabilities. It is not part of the Day 8-9 pre-match Markov feature experiment, and it does not add any active model inputs to `src/tennis_cli/models/dataset.py`.

The extension estimates the probability that player A wins the match from the current score state:

```text
sets_a, sets_b
games_a, games_b
points_a, points_b
current server
best_of
serve-point probabilities for A and B
optional live service-point counts observed so far
```

## Files Added or Changed

- Added `src/tennis_cli/features/inplay_markov.py`
- Added `src/tennis_cli/pipelines/predict_inplay.py`
- Added `src/tennis_cli/pipelines/update_tennis_abstract_pbp.py`
- Added `src/tennis_cli/pipelines/build_tennis_abstract_pbp.py`
- Added `src/tennis_cli/tests/test_inplay_markov.py`
- Added `src/tennis_cli/tests/test_tennis_abstract_pbp.py`
- Added `predict-inplay` command in `src/tennis_cli/cli.py`
- Added `update-tennis-abstract-pbp` command in `src/tennis_cli/cli.py`
- Added `build-tennis-abstract-pbp` command in `src/tennis_cli/cli.py`
- Added `reports/inplay_markov/evaluate_inplay_start_state.py`
- Added `reports/inplay_markov/inplay_start_state_metrics.csv`
- Added `reports/inplay_markov/inplay_start_state_metrics.json`
- Added this report under `reports/inplay_markov/`

No training feature list was changed. No pre-match model artifact is required by this extension.

## Tennis Abstract Point-by-Point Lane

Tennis Abstract point-by-point support was added as a separate data lane using Jeff Sackmann's Match Charting Project:

- Source: `https://github.com/JeffSackmann/tennis_MatchChartingProject`
- Source data dictionary: `https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/data_dictionary.txt`
- License: CC BY-NC-SA 4.0, as stated by the source repository.

The implementation keeps this data out of the existing remediation/pre-match pipeline.

Isolated paths:

```text
data/raw/tennis_abstract_pbp/tennis_MatchChartingProject/
data/processed/tennis_abstract_pbp/match_charting_points.parquet
data/features/tennis_abstract_pbp/inplay_markov_snapshots.parquet
```

Commands:

```bash
python -m tennis_cli update-tennis-abstract-pbp
python -m tennis_cli build-tennis-abstract-pbp --tour both --start-year 2015 --end-year 2025
```

The builder reads `charting-m-matches.csv`, `charting-w-matches.csv`, and all matching `charting-*-points*.csv` decade files. It creates one row per prediction-time point state with:

- sets/games/points before the point,
- current server,
- tiebreak flag,
- player names and match metadata,
- service points won/played before the current point for each player,
- current point winner and final match winner labels where available.

In the current Match Charting repository, point rows are split by decade, for example `charting-m-points-2010s.csv`, `charting-m-points-2020s.csv`, and `charting-m-points-to-2009.csv`. The builder concatenates all matching decade files for ATP and WTA. The source point score `Pts` is interpreted in current server/returner order and then converted into Player1/Player2 order for the saved snapshot columns `points_1` and `points_2`.

The key leakage guardrail is that live service counts are cumulative counts shifted by one point within each match. Therefore point `t` can use only points `< t`; it cannot use the outcome of point `t` or any later point.

Build result from the isolated lane:

```text
Rows: 1,189,661
Matches: 7,774
ATP rows: 757,766
WTA rows: 431,895
Rows with final match labels: 1,173,313
Rows with missing final match labels: 16,348
```

The small set of missing final-match labels is retained rather than silently coerced. These rows can still be useful for live prediction snapshots, but they should be excluded from supervised evaluation until their match winner can be verified.

## Method

The model is a point-level Markov score-state engine. For normal games, the probability of winning the current game is solved from the current point score, with the deuce/advantage loop collapsed analytically to avoid recursive cycles. The match recursion then advances through games, sets, tiebreaks, and match termination.

At 6-6, the implementation assumes a standard tiebreak, including in the final set. Advantage sets are deliberately not exposed as supported behavior yet. Tiebreak service order follows the standard pattern: one point by the first server, then alternating two-point service blocks. Very long unresolved tiebreak tails are truncated at 60 total tiebreak points and treated as a 50/50 tiebreak continuation approximation.

The live serve-point probabilities are updated with simple Bayesian shrinkage:

```text
live_p = (prior_strength * prior_p + service_points_won_so_far)
         / (prior_strength + service_points_played_so_far)
```

The default `prior_strength` is 48 pseudo-service-points. This keeps the estimate close to the pre-match prior early in the match and lets observed in-match service performance matter more as the match progresses.

## Leakage Controls

This extension is intentionally written to avoid the most common in-play leakage traps:

- It accepts only the current score at prediction time.
- It accepts only live service-point counts that have already occurred.
- It does not read post-match box-score totals.
- It does not use full-match service percentages from the current match.
- It does not change the pre-match training feature list.
- The CLI names the live count inputs as `*_so far` in the help text/report semantics.

For thesis use, any evaluation dataset must snapshot each prediction at a specific in-match time. A final match row with full-match totals is not valid input for this extension.

## CLI Usage

Example:

```bash
python -m tennis_cli predict-inplay \
  --sets-a 0 \
  --sets-b 0 \
  --games-a 3 \
  --games-b 2 \
  --points-a 40 \
  --points-b 15 \
  --server A \
  --p-a-serve-point 0.62 \
  --p-b-serve-point 0.58 \
  --best-of 3 \
  --a-service-points-won 20 \
  --a-service-points-played 32 \
  --b-service-points-won 18 \
  --b-service-points-played 31
```

Smoke-test output from the implementation:

```text
Player A win probability: 0.8192
Player B win probability: 0.1808
```

## Validation

Focused tests were added in `src/tennis_cli/tests/test_inplay_markov.py`.

Verified:

- Tennis point-score parsing supports normal scores and tiebreak point counts.
- Terminal match states return exact probabilities: `1.0` for A already won and `0.0` for B already won.
- A neutral 0-0 match with equal serve-point probabilities returns approximately `0.5`.
- Favorable point states increase player A's probability relative to unfavorable point states.
- Tiebreak states return bounded probabilities.
- Bayesian live updating validates impossible service counts.
- Pipeline outputs complementary probabilities.

Command run:

```bash
PYTHONPATH=src python -m pytest src/tennis_cli/tests/test_inplay_markov.py
```

Result:

```text
8 passed
```

There was one existing pytest cache warning on Windows about `.pytest_cache`; it did not affect the test result.

Additional Tennis Abstract lane tests:

```bash
PYTHONPATH=src python -m pytest src/tennis_cli/tests/test_tennis_abstract_pbp.py src/tennis_cli/tests/test_inplay_markov.py
```

Result:

```text
11 passed
```

The Tennis Abstract tests verify score parsing, isolated schema fields, and leakage-safe shifting of live service counts. In the leakage test, point 1 has zero prior service points even when the server wins point 1, confirming the current point is not included in its own input features.

## Evaluation Metrics

Before the Tennis Abstract point-by-point lane was added, the repo did not contain leakage-safe point-by-point or timestamped in-play snapshots. The table below is kept as the earlier start-state sanity check. It should not be treated as the main in-play backtest now that the Tennis Abstract fixed-checkpoint evaluation is available.

The table below is therefore a separate **start-state / no-live-counts** evaluation. It evaluates the Markov probability at `sets 0-0, games 0-0, points 0-0`, with first server unknown and averaged across the two first-server scenarios. Inputs are prior serve/return histories only, so this does not alter or replace the previous pre-match remediation results. The fourth metric is **Brier score**.

Artifacts:

- `reports/inplay_markov/evaluate_inplay_start_state.py`
- `reports/inplay_markov/inplay_start_state_metrics.csv`
- `reports/inplay_markov/inplay_start_state_metrics.json`

Command run:

```bash
PYTHONPATH=src python reports/inplay_markov/evaluate_inplay_start_state.py
```

| tour | source | surface | split | rows | log loss | accuracy | ROC AUC | Brier |
|---|---|---|---|---:|---:|---:|---:|---:|
| atp | sackmann | Hard | validation | 1640 | 0.6436 | 0.6220 | 0.6714 | 0.2267 |
| atp | sackmann | Hard | test | 1709 | 0.6452 | 0.6080 | 0.6696 | 0.2271 |
| atp | sackmann | Clay | validation | 873 | 0.6589 | 0.5704 | 0.6357 | 0.2337 |
| atp | sackmann | Clay | test | 950 | 0.6451 | 0.5979 | 0.6598 | 0.2278 |
| atp | sackmann | Grass | validation | 325 | 0.6407 | 0.6308 | 0.6842 | 0.2255 |
| atp | sackmann | Grass | test | 315 | 0.6149 | 0.6317 | 0.7057 | 0.2144 |
| atp | tml | Hard | validation | 1649 | 0.6426 | 0.6186 | 0.6732 | 0.2262 |
| atp | tml | Hard | test | 1695 | 0.6440 | 0.6088 | 0.6717 | 0.2266 |
| atp | tml | Clay | validation | 873 | 0.6582 | 0.5704 | 0.6363 | 0.2334 |
| atp | tml | Clay | test | 951 | 0.6447 | 0.5952 | 0.6602 | 0.2276 |
| atp | tml | Grass | validation | 325 | 0.6422 | 0.6308 | 0.6774 | 0.2262 |
| atp | tml | Grass | test | 315 | 0.6143 | 0.6317 | 0.7102 | 0.2141 |
| wta | sackmann | Hard | validation | 1654 | 0.6490 | 0.6179 | 0.6616 | 0.2291 |
| wta | sackmann | Hard | test | 1533 | 0.6379 | 0.6360 | 0.6823 | 0.2240 |
| wta | sackmann | Clay | validation | 757 | 0.6605 | 0.5945 | 0.6387 | 0.2342 |
| wta | sackmann | Clay | test | 708 | 0.6497 | 0.6059 | 0.6659 | 0.2291 |
| wta | sackmann | Grass | validation | 299 | 0.6051 | 0.6789 | 0.7464 | 0.2083 |
| wta | sackmann | Grass | test | 288 | 0.6374 | 0.6424 | 0.6885 | 0.2233 |

Interpretation: these numbers are useful as a no-leakage sanity check for the score-state engine's starting probability, but they should not be presented as live in-play performance. A proper in-play evaluation needs snapshots taken before each prediction point, with only score and service-point counts observed up to that instant.

## Tennis Abstract Fixed-Checkpoint Evaluation

After adding Tennis Abstract point-by-point snapshots, a separate fixed-checkpoint in-play evaluation was run. This is the first real point-by-point evaluation of the in-play Markov engine in this project.

Runtime note: evaluating every point in the 1.18M-row snapshot parquet is too slow for the current recursive engine. The fixed checkpoint schedule keeps runtime reasonable while staying deterministic and leakage-safe:

- `match_start`: first point of each labeled match.
- `set_start`: first point of later sets.
- `late_game_start_total_ge10`: game starts when at least ten games have already been played in the current set.

The first attempted wider checkpoint set had about 98k rows and timed out. The final fixed schedule has 13,012 validation/test rows across 3,218 matches. It uses a neutral `0.5` serve-point prior for both players and Bayesian live service-point updates from prior points only.

Artifacts:

- `reports/inplay_markov/evaluate_tennis_abstract_checkpoints.py`
- `reports/inplay_markov/tennis_abstract_checkpoint_metrics.csv`
- `reports/inplay_markov/tennis_abstract_checkpoint_metrics.json`
- `reports/inplay_markov/tennis_abstract_checkpoint_predictions_sample.csv`
- `reports/inplay_markov/tennis_abstract_snapshot_sample.csv`
- `reports/inplay_markov/tennis_abstract_snapshot_schema.json`

Command run:

```bash
PYTHONPATH=src python reports/inplay_markov/evaluate_tennis_abstract_checkpoints.py
```

Overall fixed-checkpoint results:

| group | value | rows | matches | log loss | accuracy | ROC AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| split | test | 8897 | 2277 | 0.5882 | 0.6718 | 0.7490 | 0.2002 |
| split | validation | 4115 | 941 | 0.5985 | 0.6615 | 0.7469 | 0.2031 |

By checkpoint type:

| group | value | rows | matches | log loss | accuracy | ROC AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| test:checkpoint_type | late_game_start_total_ge10 | 3258 | 951 | 0.6141 | 0.6562 | 0.7268 | 0.2116 |
| test:checkpoint_type | match_start | 2277 | 2277 | 0.6931 | 0.5358 | 0.5000 | 0.2500 |
| test:checkpoint_type | set_start | 3362 | 2277 | 0.4921 | 0.7790 | 0.8508 | 0.1554 |
| validation:checkpoint_type | late_game_start_total_ge10 | 1687 | 463 | 0.6172 | 0.6627 | 0.7261 | 0.2134 |
| validation:checkpoint_type | match_start | 942 | 941 | 0.6930 | 0.4936 | 0.5011 | 0.2499 |
| validation:checkpoint_type | set_start | 1486 | 941 | 0.5174 | 0.7665 | 0.8444 | 0.1618 |

By tour and surface, excluding the tiny `Unknown` metadata group:

| group | value | rows | matches | log loss | accuracy | ROC AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| test:tour_surface | atp/Clay | 1550 | 380 | 0.6156 | 0.6535 | 0.7241 | 0.2097 |
| test:tour_surface | atp/Grass | 671 | 118 | 0.5904 | 0.6930 | 0.7653 | 0.1973 |
| test:tour_surface | atp/Hard | 3545 | 795 | 0.5845 | 0.6705 | 0.7532 | 0.1995 |
| test:tour_surface | wta/Clay | 796 | 250 | 0.6068 | 0.6583 | 0.7296 | 0.2069 |
| test:tour_surface | wta/Grass | 319 | 99 | 0.5483 | 0.7116 | 0.7800 | 0.1839 |
| test:tour_surface | wta/Hard | 2010 | 634 | 0.5712 | 0.6801 | 0.7576 | 0.1946 |
| validation:tour_surface | atp/Clay | 690 | 155 | 0.6518 | 0.6159 | 0.6989 | 0.2274 |
| validation:tour_surface | atp/Grass | 401 | 74 | 0.5941 | 0.6883 | 0.7646 | 0.2011 |
| validation:tour_surface | atp/Hard | 1841 | 378 | 0.5945 | 0.6659 | 0.7564 | 0.1995 |
| validation:tour_surface | wta/Clay | 193 | 53 | 0.5947 | 0.6425 | 0.7398 | 0.2069 |
| validation:tour_surface | wta/Grass | 188 | 42 | 0.6856 | 0.5585 | 0.6267 | 0.2440 |
| validation:tour_surface | wta/Hard | 802 | 239 | 0.5445 | 0.7057 | 0.7917 | 0.1811 |

Interpretation: the neutral-prior model is intentionally weak at match start (`0.5` for everyone), so match-start rows behave like a coin flip. The lift appears once score state enters the model, especially at set starts after one player has already won a set. This is expected for an in-play score-state model and is separate from the pre-match remediation baseline.

## Parquet Inspection Note

The full Tennis Abstract snapshot parquet is large and may not open smoothly in some editors:

```text
data/features/tennis_abstract_pbp/inplay_markov_snapshots.parquet
```

For inspection without a parquet viewer, use:

- `reports/inplay_markov/tennis_abstract_snapshot_sample.csv`
- `reports/inplay_markov/tennis_abstract_checkpoint_predictions_sample.csv`
- `reports/inplay_markov/tennis_abstract_snapshot_schema.json`

## Removability

If this extension is not approved, remove these files and the CLI import/command block:

- `src/tennis_cli/features/inplay_markov.py`
- `src/tennis_cli/pipelines/predict_inplay.py`
- `src/tennis_cli/pipelines/update_tennis_abstract_pbp.py`
- `src/tennis_cli/pipelines/build_tennis_abstract_pbp.py`
- `src/tennis_cli/tests/test_inplay_markov.py`
- `src/tennis_cli/tests/test_tennis_abstract_pbp.py`
- `reports/inplay_markov/`
- the `predict_inplay_for_match`, `update_tennis_abstract_pbp_repo`, and `build_tennis_abstract_pbp_artifacts` imports in `src/tennis_cli/cli.py`
- the `predict-inplay`, `update-tennis-abstract-pbp`, and `build-tennis-abstract-pbp` command blocks in `src/tennis_cli/cli.py`

Because the extension does not modify `dataset.py`, the active pre-match model inputs do not need to be reverted.
