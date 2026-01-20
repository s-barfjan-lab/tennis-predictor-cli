# Tennis Match Outcome Prediction (Pre-Match & In-Play)

A **modular, CLI-first Python system** for predicting **tennis match outcomes** (pre-match & live), **set/game probabilities**, and **long-term player trajectories** by combining **structured performance statistics** with **lightweight LLM-extracted context** (injury, fatigue, coaching changes).

Built with a strong focus on **probability calibration, reproducibility, and interpretability**, grounded in peer-reviewed sports analytics literature.

---

## What This Project Does

- **Pre-match prediction**: win probabilities using Elo, rankings, serve/return micro-stats, form, and event context  
- **In-play prediction**: live match/set/game probabilities via a **point-based Markov model** with Bayesian updates  
- **Context awareness**: short-term signals (injury, travel fatigue, coach changes) extracted using **tiny LLMs**  
- **Long-term modeling**: surface-aware Elo trajectories and 6–12 month forecasts  

---

## Data & Features

**Data sources**
- Tennis Abstract (Elo, player stats)
- Jeff Sackmann datasets (ATP/WTA results, rankings, point-by-point)
- TML-Database (live-updated ATP match stats)

**Key features**
- Elo & ranking differences (surface-aware)
- Serve/return micro-metrics (hold, break, 1st/2nd serve stats)
- Recent form & fatigue indicators
- Head-to-head and matchup attributes
- Event context (surface, tournament level, round)
- LLM-derived text signals (recency-weighted)

---

## Models

- **Baselines**: Logistic regression, Elo-implied probabilities  
- **Primary**: XGBoost on tabular features  
- **Neural option**: Siamese sequence encoder over recent match histories  
- **Live engine**: Markov / point-based model for game, set, and match probabilities  
- **Ensembles & calibration**: stacking, Platt / isotonic calibration  

Accuracy targets are aligned with the literature (~70%), with emphasis on **well-calibrated probabilities** rather than raw accuracy.

---

## CLI (Planned / In Progress)

```bash
update-data        # ingest & validate datasets
build-features     # snapshot features by date/surface
train              # train & calibrate models
predict-match      # pre-match win probabilities
predict-inplay     # live probabilities from score state
project-player     # long-term player projections
explain            # feature importance & model explanations
