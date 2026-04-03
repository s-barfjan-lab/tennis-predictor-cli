import pandas as pd
from tennis_cli.features.elo import compute_elo_features

df = pd.DataFrame({
    "date": ["2025-01-01", "2025-01-05", "2025-01-10"],
    "winner_id": [1, 1, 2],
    "loser_id": [2, 3, 1],
    "winner_name": ["Player A", "Player A", "Player B"],
    "loser_name": ["Player B", "Player C", "Player A"],
})

out = compute_elo_features(df)

print(out[[
    "date",
    "winner_name",
    "loser_name",
    "winner_elo_pre",
    "loser_elo_pre",
    "winner_elo_post",
    "loser_elo_post",
    "elo_diff_pre",
    "elo_prob_winner_pre",
]])