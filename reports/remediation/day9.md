# Day 9 Remediation: Markov Feature Decision

Day 9 ran the Markov feature acceptance experiment across all 9 branches for `logit_baseline` and `xgb_baseline`. The full table and formula documentation are in `reports/remediation/day8_9.md`.

Decision: **fail**.

The feature improved at least one model by the Day 4 noise threshold on 5 of 9 branches, but it also produced regressions beyond the allowed threshold on 4 branches. That violates the pass criterion: at least 4 improving branches and no branch regressing by more than 1x noise.

Resulting action:

- The Markov columns remain generated in the feature pipeline for reproducibility and reporting.
- The Markov inputs were removed from the active model feature lists in `src/tennis_cli/models/dataset.py`.
- Saved baseline artifacts were retrained on the no-Markov active feature set.

Commit message for the experiment:

```text
chore(experiment): record negative result for Markov match-probability feature
```
