# Days 8-9 Remediation: Markov Match-Probability Feature

Commit message:

```text
chore(experiment): record negative result for Markov match-probability feature
```

## Formula Implementation

This experiment adds a pre-match domain-knowledge feature: each player receives prior-only rolling serve and return point estimates, then tennis scoring structure maps those point estimates into an implied match-win probability. The feature is not an in-play model and does not use live score state or current-match data.

Rolling inputs are computed in `src/tennis_cli/features/rolling.py` with `shift(1)`, `window=30`, and `min_periods=10`:

- `service_points_won_pct_30`
- `return_points_won_pct_30`
- `has_serve_history`
- `has_return_history`

For each deterministic A/B match row:

```text
p_a_serve_point = 0.5 * (service_points_won_pct_30_a + (1 - return_points_won_pct_30_b))
p_b_serve_point = 0.5 * (service_points_won_pct_30_b + (1 - return_points_won_pct_30_a))
```

If history is insufficient, the point estimate falls back to the neutral value `0.5`, and the history indicator flags tell the model that the estimate is less reliable.

The service-game hold probability uses the corrected O'Malley closed form:

```text
P_game(p) = p^4 * (15 - 34p + 28p^2 - 8p^3) / (1 - 2p + 2p^2)
```

The originally proposed formula with `p^4 * (15 - 4p - 10p^2) / (...)` was not used because it can produce invalid probabilities greater than 1. O'Malley's formula gives the server's hold probability; the receiver's break probability is the complement of the opponent's hold probability.

Sipko (2015) is cited for the recursive hierarchical decomposition from point probabilities to game, set, and match probabilities. O'Malley, "Probability Formulas and Statistical Analysis in Tennis," is cited for the closed-form game/hold probability formula and related tennis probability formulas. The implementation lives in `src/tennis_cli/features/markov.py`; set probability is computed by dynamic programming with alternating service games, and match probability uses the row's `best_of` value. Because first server is unknown pre-match, the implementation averages the A-serves-first and B-serves-first scenarios. Tiebreaks use a documented recursive approximation with the standard tiebreak service pattern.

Sources:

- Sipko, M. (2015), "Machine Learning for the Prediction of Professional Tennis Matches": https://www.doc.ic.ac.uk/teaching/distinguished-projects/2015/m.sipko.pdf
- O'Malley, A. J. (2008), "Probability Formulas and Statistical Analysis in Tennis": https://ideas.repec.org/a/bpj/jqsprt/v4y2008i2n15.html

## Experiment Setup

The "without Markov" column is the Day 7 post-remediation baseline from `reports/remediation/day7_before_after.csv`. The "with Markov" column comes from a temporary active-feature run where `delta_markov_match_win`, `has_serve_history`, and `has_return_history` were added to the official model feature lists and all 18 model/surface branches were retrained.

After the result failed the pass criterion, the Markov columns were removed from the active model feature lists in `src/tennis_cli/models/dataset.py`. The generated columns remain in the feature parquet/report path so the negative experiment is reproducible.

## 18-Row Comparison

CSV copy: `reports/remediation/day8_9_markov_comparison.csv`.

| branch | model | without Markov LL | with Markov LL | noise std | improvement | row result |
|---|---|---:|---:|---:|---:|---|
| atp/sackmann/Hard | logit_baseline | 0.602986 | 0.603396 | 0.000000 | -0.000410 | regress>noise |
| atp/sackmann/Hard | xgb_baseline | 0.603671 | 0.600638 | 0.019183 | 0.003033 | within_noise |
| atp/sackmann/Clay | logit_baseline | 0.615780 | 0.615225 | 0.000000 | 0.000554 | improve>=noise |
| atp/sackmann/Clay | xgb_baseline | 0.621534 | 0.615121 | 0.001308 | 0.006413 | improve>=noise |
| atp/sackmann/Grass | logit_baseline | 0.585441 | 0.584588 | 0.000000 | 0.000853 | improve>=noise |
| atp/sackmann/Grass | xgb_baseline | 0.586422 | 0.586227 | 0.003931 | 0.000195 | within_noise |
| atp/tml/Hard | logit_baseline | 0.602367 | 0.602997 | 0.000000 | -0.000630 | regress>noise |
| atp/tml/Hard | xgb_baseline | 0.605591 | 0.603389 | 0.015478 | 0.002202 | within_noise |
| atp/tml/Clay | logit_baseline | 0.616621 | 0.616277 | 0.000000 | 0.000344 | improve>=noise |
| atp/tml/Clay | xgb_baseline | 0.620729 | 0.620029 | 0.003921 | 0.000700 | within_noise |
| atp/tml/Grass | logit_baseline | 0.578688 | 0.577424 | 0.000000 | 0.001264 | improve>=noise |
| atp/tml/Grass | xgb_baseline | 0.588230 | 0.584644 | 0.001498 | 0.003586 | improve>=noise |
| wta/sackmann/Hard | logit_baseline | 0.610515 | 0.611492 | 0.000000 | -0.000977 | regress>noise |
| wta/sackmann/Hard | xgb_baseline | 0.612234 | 0.614426 | 0.011952 | -0.002192 | within_noise |
| wta/sackmann/Clay | logit_baseline | 0.589239 | 0.588761 | 0.000000 | 0.000478 | improve>=noise |
| wta/sackmann/Clay | xgb_baseline | 0.601972 | 0.603582 | 0.001888 | -0.001610 | within_noise |
| wta/sackmann/Grass | logit_baseline | 0.653408 | 0.657093 | 0.000000 | -0.003685 | regress>noise |
| wta/sackmann/Grass | xgb_baseline | 0.635252 | 0.632844 | 0.003613 | 0.002408 | within_noise |

## Criterion Result

Pass criterion:

- Test log loss must improve by at least 1x the per-branch noise std on at least 4 of 9 branches simultaneously.
- No branch may regress by more than 1x std.

Observed:

- 5 of 9 branches had at least one model row improve by at least the noise threshold.
- 4 of 9 branches had at least one model row regress beyond the noise threshold: ATP/Sackmann/Hard, ATP/TML/Hard, WTA/Sackmann/Hard, and WTA/Sackmann/Grass.

Final decision: **fail**. The Markov feature is rejected as an active model input for now. The negative result is retained because the feature produced some real improvements, especially ATP/Sackmann/Clay and ATP/TML/Grass, but it violated the no-regression requirement.

## Verification

Focused validation:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest src/tennis_cli/tests/test_markov.py src/tennis_cli/tests/test_rolling_chronology.py
```

Result: `5 passed`; pytest emitted the existing cache warning.
