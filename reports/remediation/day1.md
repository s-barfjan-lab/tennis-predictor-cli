# Day 1 Remediation Report: Chronology Fix

## Code Diff

```diff
diff --git a/src/tennis_cli/features/h2h.py b/src/tennis_cli/features/h2h.py
index dcdb37f..b7aa1d1 100644
--- a/src/tennis_cli/features/h2h.py
+++ b/src/tennis_cli/features/h2h.py
@@ -24,7 +24,17 @@ def compute_h2h_features(long_df: pd.DataFrame) -> pd.DataFrame:
 
     df = long_df.copy()
     df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
-    df = df.sort_values(["tourney_date", "match_id", "player_id"]).reset_index(drop=True)
+    sort_keys = ["tourney_date"]
+    if "tourney_id" in df.columns:
+        sort_keys.append("tourney_id")
+    if "match_num" in df.columns:
+        df = df.copy()
+        df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce")
+        sort_keys.append("match_num")
+    else:
+        sort_keys.append("match_id")
+    sort_keys.append("player_id")
+    df = df.sort_values(sort_keys).reset_index(drop=True)
 
     h2h_tracker: dict[tuple[str, str], int] = {}
 
@@ -62,4 +72,4 @@ def compute_h2h_features(long_df: pd.DataFrame) -> pd.DataFrame:
     df["h2h_win_ratio"] = df["h2h_wins"] / total.replace(0, pd.NA)
     df["h2h_win_ratio"] = pd.to_numeric(df["h2h_win_ratio"], errors="coerce").fillna(0.5)
 
-    return df
\ No newline at end of file
+    return df
diff --git a/src/tennis_cli/features/long_view.py b/src/tennis_cli/features/long_view.py
index 94dec52..02e93c7 100644
--- a/src/tennis_cli/features/long_view.py
+++ b/src/tennis_cli/features/long_view.py
@@ -76,6 +76,8 @@ def build_long_view(matches_df: pd.DataFrame, tour: str) -> pd.DataFrame:
         {
             "match_id": df["match_id"],
             "tour": df["tour"],
+            "tourney_id": _safe_col(df, "tourney_id"),
+            "match_num": _safe_col(df, "match_num"),
             "tourney_date": _safe_col(df, "tourney_date"),
             "tourney_name": _safe_col(df, "tourney_name"),
             "tourney_level": _safe_col(df, "tourney_level", "U"),
@@ -141,6 +143,8 @@ def build_long_view(matches_df: pd.DataFrame, tour: str) -> pd.DataFrame:
         {
             "match_id": df["match_id"],
             "tour": df["tour"],
+            "tourney_id": _safe_col(df, "tourney_id"),
+            "match_num": _safe_col(df, "match_num"),
             "tourney_date": _safe_col(df, "tourney_date"),
             "tourney_name": _safe_col(df, "tourney_name"),
             "tourney_level": _safe_col(df, "tourney_level", "U"),
@@ -204,6 +208,7 @@ def build_long_view(matches_df: pd.DataFrame, tour: str) -> pd.DataFrame:
     # Numeric cleanup where possible
     numeric_cols = [
         "player_ht",
+        "match_num",
         "player_age",
         "player_rank",
         "player_rank_points",
diff --git a/src/tennis_cli/features/rolling.py b/src/tennis_cli/features/rolling.py
index 68fa381..399f8ba 100644
--- a/src/tennis_cli/features/rolling.py
+++ b/src/tennis_cli/features/rolling.py
@@ -9,6 +9,7 @@ def _rolling_mean_past(series: pd.Series, window: int) -> pd.Series:
     Mean over the previous `window` matches only.
     The current match is excluded via shift(1).
     """
+    series = pd.to_numeric(series, errors="coerce")
     return series.shift(1).rolling(window=window, min_periods=1).mean()
 
 
@@ -17,6 +18,7 @@ def _expanding_mean_past(series: pd.Series) -> pd.Series:
     Mean over all previous rows only.
     The current match is excluded via shift(1).
     """
+    series = pd.to_numeric(series, errors="coerce")
     return series.shift(1).expanding(min_periods=1).mean()
 
 
@@ -132,7 +134,19 @@ def add_rolling_features(long_df: pd.DataFrame, window: int = 10) -> pd.DataFram
         raise ValueError(f"Missing required columns for rolling features: {missing}")
 
     df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
-    df = df.sort_values(["player_id", "tourney_date", "match_id"]).reset_index(drop=True)
+    # Within a tournament, every match shares tourney_date (Sackmann's Monday-of-week
+    # convention). Chronology inside a tournament is encoded in numeric match_num.
+    # Sorting by string match_id is lexicographic and puts "100" before "5".
+    sort_keys = ["player_id", "tourney_date"]
+    if "tourney_id" in df.columns:
+        sort_keys.append("tourney_id")
+    if "match_num" in df.columns:
+        df = df.copy()
+        df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce")
+        sort_keys.append("match_num")
+    else:
+        sort_keys.append("match_id")
+    df = df.sort_values(sort_keys).reset_index(drop=True)
 
     grouped = df.groupby("player_id", group_keys=False)
```

## Regression Test Output

```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\LENOVO\Projects\tennis-predictor-cli\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\LENOVO\Projects\tennis-predictor-cli
collecting ... collected 1 item

src/tennis_cli/tests/test_rolling_chronology.py::test_rolling_features_are_chronologically_ordered_in_a_slam_path PASSED [100%]

============================== warnings summary ===============================
.venv\Lib\site-packages\_pytest\cacheprovider.py:475
  C:\Users\LENOVO\Projects\tennis-predictor-cli\.venv\Lib\site-packages\_pytest\cacheprovider.py:475: PytestCacheWarning: could not create cache path C:\Users\LENOVO\Projects\tennis-predictor-cli\.pytest_cache\v\cache\nodeids: [WinError 183] Cannot create a file when that file already exists: 'C:\\Users\\LENOVO\\Projects\\tennis-predictor-cli\\.pytest_cache\\v\\cache'
    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 1 passed, 1 warning in 2.14s =========================
```

## Sanity-Check Model

ATP/TML/Hard `xgb_baseline`, `random_state=42`.

| Run | Test log loss | Test accuracy |
|---|---:|---:|
| Before chronology fix | 0.5861 | 68.56% |
| After chronology fix | 0.6341 | 65.39% |

The test direction matches expectation: log loss increased and accuracy decreased after removing within-tournament chronology leakage.

## Rebuilt Feature Artifacts

| Artifact | Rows |
|---|---:|
| ATP Sackmann long-view | 53,548 |
| ATP Sackmann baseline | 26,774 |
| ATP TML long-view | 61,434 |
| ATP TML baseline | 30,717 |
| WTA Sackmann long-view | 49,916 |
| WTA Sackmann baseline | 24,958 |

## Gating Answers

### 1. ATP/TML/Hard xgb_baseline sanity check

| Metric | Before | After |
|---|---:|---:|
| Test log loss | 0.5861 | 0.6341 |
| Test accuracy | 68.56% | 65.39% |

### 2. Test row counts for 9 source x surface cells

| Cell | Pre-fix test rows | Post-fix test rows |
|---|---:|---:|
| ATP/Sackmann/Hard | 1,709 | 1,709 |
| ATP/Sackmann/Clay | 951 | 951 |
| ATP/Sackmann/Grass | 315 | 315 |
| ATP/TML/Hard | 3,629 | 3,629 |
| ATP/TML/Clay | 1,767 | 1,767 |
| ATP/TML/Grass | 624 | 624 |
| WTA/Sackmann/Hard | 1,533 | 1,533 |
| WTA/Sackmann/Clay | 709 | 709 |
| WTA/Sackmann/Grass | 288 | 288 |

### 3. Seven-row player-A regression-test printout

```text
player_id round  match_num  matches_played  current_win_streak
        A  R128          5               0                 0.0
        A   R64         68               1                 1.0
        A   R32         99               2                 2.0
        A   R16        114               3                 3.0
        A    QF        121               4                 4.0
        A    SF        125               5                 5.0
        A     F        127               6                 6.0
```

## Commit Message

```text
fix(features): sort by numeric match_num to preserve chronology within tournament
```
