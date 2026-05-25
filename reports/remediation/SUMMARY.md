# Remediation Summary

## Thesis Position

The project is in **State B**: the Markov feature did not pass, but the corrected baseline is materially different from the original numbers. The defensible thesis claim is therefore not that a new feature wins across the board. The contribution is a rigorous correction and audit of a pre-match tennis prediction pipeline: chronological feature ordering was fixed, leakage-affected claims were withdrawn, calibrated probability estimates were re-evaluated, and a literature-derived Markov serve/return feature was tested under a strict noise-floor criterion and rejected.

## What Was Found

The major finding was chronology leakage in the earlier feature pipeline. Day 2 showed that the bug was concentrated in ATP/TML XGBoost branches. The most affected examples were:

- ATP/TML/Grass XGB moved from `0.5357` to `0.6108` test log loss after correction, with accuracy dropping from `72.6%` to `65.7%`.
- ATP/TML/Clay XGB moved from `0.5718` to `0.6252` test log loss.
- ATP/TML/Hard XGB moved from `0.5861` to `0.6341` test log loss.

The Sackmann branches were mostly stable under the chronology correction, which helped isolate the problem rather than invalidating the entire project. The lesson is sharper: the attractive ATP/TML XGB headline was not an honest model improvement.

Day 4 re-established a corrected random-seed noise floor. Logistic baselines were deterministic across seeds in this setup (`test_ll_std = 0.0`), while XGBoost had branch-dependent noise. The largest XGB test-log-loss std values were ATP/Sackmann/Hard (`0.019183`), ATP/TML/Hard (`0.015478`), and WTA/Sackmann/Hard (`0.011952`). This became the standard for judging later changes.

Day 8-9 tested a pre-match Markov match-probability decomposition based on prior-30 serve and return point performance. The feature improved at least one model by the noise threshold on 5 of 9 branches, but it regressed 4 branches beyond the allowed threshold. Because the criterion required no branch to regress by more than 1x noise, the feature was rejected as an active input. The negative result is still useful: literature-grade scoring structure alone was not robust enough to improve this corrected tabular pipeline.

## What Was Fixed

- Chronological rolling features now exclude the current match and sort same-tournament matches by numeric `match_num` when available.
- Surface rolling means now require at least five prior same-surface matches and expose `has_surface_history`.
- Redundant `delta_h2h_losses` was removed from active model inputs.
- Elo configuration was reconciled through the shared `compute_all_elo_features()` path.
- Surface Elo now bootstraps from current global Elo for a player's first match on a surface.
- Walkovers, retirements, defaults, abandoned, unplayed, and cancelled matches no longer update Elo.
- XGB inference now supports `--model-variant baseline|tuned` instead of always loading baseline artifacts.
- The Markov feature implementation and tests remain in the codebase, but the feature is not active in `dataset.py` after failing the Day 8-9 criterion.

## New Headline Numbers

The canonical post-remediation baseline is the no-Markov Day 7 active feature set. Best active model per branch:

| branch | selected model | test log loss | test accuracy |
|---|---|---:|---:|
| ATP/Sackmann/Hard | logit_baseline | 0.6030 | 65.77% |
| ATP/Sackmann/Clay | logit_baseline | 0.6158 | 65.68% |
| ATP/Sackmann/Grass | logit_baseline | 0.5854 | 69.21% |
| ATP/TML/Hard | logit_baseline | 0.6024 | 65.55% |
| ATP/TML/Clay | logit_baseline | 0.6166 | 64.46% |
| ATP/TML/Grass | logit_baseline | 0.5787 | 71.11% |
| WTA/Sackmann/Hard | logit_baseline | 0.6105 | 66.01% |
| WTA/Sackmann/Clay | logit_baseline | 0.5892 | 68.50% |
| WTA/Sackmann/Grass | xgb_baseline | 0.6353 | 61.11% |

Mean across these 9 branch winners: `0.6041` test log loss and `66.38%` accuracy. These are the defensible headline numbers until a later tag supersedes `v0-postleak`.

## Final Position

The project should be described as a corrected, reproducible pre-match tennis prediction benchmark with calibrated probability emphasis, not as an unchecked accuracy race. Reports and claims prior to `v0-postleak` should be treated as non-canonical because they may include chronology leakage or pre-remediation calibration assumptions.

The recommended commit message for Day 10 is:

```text
docs(remediation): final summary; pin v0-postleak as the canonical baseline
```
