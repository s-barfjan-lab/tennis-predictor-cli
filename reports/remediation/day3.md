# Day 3 Remediation: TML/Sackmann Symmetry

Day 3 removed the remaining dataset-composition confounds between ATP Sackmann and ATP TML. TML now applies the same walkover/retirement score filtering as Sackmann, is capped to the ATP Sackmann processed date range, and the model feature pipeline drops non-model bronze / third-place rounds instead of silently mapping unknown rounds to `R32`.

In the row-count table below, `raw_rows` means rows in the processed match parquet on the modeled surface. `filtered_rows` means rows eligible for modeling after the then-current model filters. Pre-fix TML is reconstructed from the raw TML CSVs without the new walkover/retirement filter, date cap, or explicit round filter.

| source | surface | pre_raw_rows | pre_filtered_rows | post_raw_rows | post_filtered_rows |
| --- | --- | ---: | ---: | ---: | ---: |
| sackmann | Hard | 15764 | 15764 | 15764 | 15761 |
| sackmann | Clay | 8101 | 8101 | 8101 | 8100 |
| sackmann | Grass | 2839 | 2839 | 2839 | 2839 |
| tml | Hard | 18199 | 18199 | 15765 | 15762 |
| tml | Clay | 9139 | 9139 | 8111 | 8110 |
| tml | Grass | 3229 | 3229 | 2840 | 2840 |

The final chronological ATP test-row gate passes: Hard is 1709 Sackmann vs 1695 TML, Clay is 950 vs 951, and Grass is 315 vs 315. All three are within 10%.

## Groupby Audit: Before

### ATP Sackmann processed parquet before

The Sackmann processed parquet was not rebuilt for Day 3, so its raw `tourney_level` / `round` audit is unchanged before vs after.

```text
tourney_level round  rows
            A    BR     2
            A     F   499
            A    QF  1904
            A   R16  3826
            A   R32  6325
            A   R64   571
            A    RR   284
            A    SF   978
            D    RR  2105
            F    BR     1
            F     F    14
            F    RR   166
            F    SF    27
            G     F    39
            G    QF   146
            G  R128  2402
            G   R16   300
            G   R32   600
            G   R64  1200
            G    SF    74
            M     F    82
            M    QF   310
            M  R128   744
            M   R16   631
            M   R32  1270
            M   R64  2053
            M    SF   158
            O    BR     1
            O     F     1
            O    QF     4
            O   R16     8
            O   R32    15
            O   R64    32
            O    SF     2
```

### ATP TML processed parquet before

```text
tourney_level   round  rows
          250       F   405
          250      QF  1620
          250     R16  3240
          250     R32  5015
          250     R64   240
          250      SF   810
          500       F   131
          500      QF   524
          500     R16  1048
          500     R32  2089
          500     R64   318
          500      SF   262
            A 3rd/4th     1
            A       F    17
            A      QF    12
            A    R128    72
            A     R16    16
            A     R32    24
            A      RR   318
            A      SF    32
            D      RR  2492
            F 3rd/4th     1
            F       F    12
            F      RR   144
            F      SF    24
            G       F    43
            G      QF   172
            G    R128  2752
            G     R16   344
            G     R32   688
            G     R64  1376
            G      SF    86
            M       F    91
            M      QF   364
            M    R128   992
            M     R16   728
            M     R32  1456
            M     R64  2384
            M      SF   182
            O      BR     3
            O       F     3
            O      QF    12
            O     R16    24
            O     R32    48
            O     R64    96
            O      SF     6
```

## Groupby Audit: After

### ATP Sackmann processed parquet after

```text
tourney_level round  rows
            A    BR     2
            A     F   499
            A    QF  1904
            A   R16  3826
            A   R32  6325
            A   R64   571
            A    RR   284
            A    SF   978
            D    RR  2105
            F    BR     1
            F     F    14
            F    RR   166
            F    SF    27
            G     F    39
            G    QF   146
            G  R128  2402
            G   R16   300
            G   R32   600
            G   R64  1200
            G    SF    74
            M     F    82
            M    QF   310
            M  R128   744
            M   R16   631
            M   R32  1270
            M   R64  2053
            M    SF   158
            O    BR     1
            O     F     1
            O    QF     4
            O   R16     8
            O   R32    15
            O   R64    32
            O    SF     2
```

### ATP TML processed parquet after

```text
tourney_level   round  rows
          250       F   369
          250      QF  1444
          250     R16  2907
          250     R32  4504
          250     R64   217
          250      SF   728
          500       F   115
          500      QF   432
          500     R16   887
          500     R32  1769
          500     R64   292
          500      SF   218
            A       F    13
            A      QF    11
            A    R128    69
            A     R16    16
            A     R32    24
            A      RR   259
            A      SF    29
            D      RR  2187
            F 3rd/4th     1
            F       F    11
            F      RR   130
            F      SF    22
            G       F    39
            G      QF   146
            G    R128  2405
            G     R16   300
            G     R32   601
            G     R64  1201
            G      SF    74
            M       F    82
            M      QF   310
            M    R128   744
            M     R16   631
            M     R32  1270
            M     R64  2054
            M      SF   158
            O      BR     3
            O       F     3
            O      QF    12
            O     R16    24
            O     R32    46
            O     R64    94
            O      SF     6
```

## Round Decisions

The non-standard rounds found were `BR` in Sackmann ATP, `BR` in TML ATP, `3rd/4th` in TML ATP, and `BR` in WTA Sackmann. `BR` and `3rd/4th` are bronze / third-place matches, not normal main-draw progression rounds and not qualifying rounds, so both are dropped from model feature builds and from model loading. No qualifying rounds (`Q1`, `Q2`, `Q3`) and no `ER` rows were found in the current ATP processed parquets. The silent `round_ordinal = fillna(3)` fallback was removed; any unknown round that remains after explicitly dropping bronze / third-place rows now raises instead of being treated as `R32`.
