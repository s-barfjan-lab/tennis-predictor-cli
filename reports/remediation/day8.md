# Day 8 Remediation: Markov Feature Implementation

Day 8 implemented the pre-match Markov match-probability feature and its tests. The detailed combined implementation and experiment write-up is in `reports/remediation/day8_9.md`.

Key artifacts:

- `src/tennis_cli/features/markov.py`
- `src/tennis_cli/tests/test_markov.py`
- Rolling prior-30 serve/return history in `src/tennis_cli/features/rolling.py`
- Baseline match-row Markov feature construction in `src/tennis_cli/features/baseline_features.py`

The feature was intentionally evaluated as a candidate model input only after Day 8 implementation. Its final accept/reject result is recorded in `day8_9.md` and summarized in `day9.md`.
