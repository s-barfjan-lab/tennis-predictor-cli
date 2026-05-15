# Day 4 Remediation: Corrected-Pipeline Noise Floor

Day 4 re-establishes the random-seed noise floor on the corrected pipeline. The logistic baseline is deterministic across the five seeds in every branch (`test_ll_std = 0.0`). XGBoost carries the relevant seed noise: the least stable branches are the Hard branches, especially ATP/Sackmann/Hard (`0.019183`), ATP/TML/Hard (`0.015478`), and WTA/Sackmann/Hard (`0.011952`). The most stable XGBoost branches are ATP/Sackmann/Clay (`0.001308`), ATP/TML/Grass (`0.001498`), and WTA/Sackmann/Clay (`0.001888`). Going forward, a branch-level test log-loss improvement is not reportable unless it exceeds that branch's threshold below.

| tour | source | surface | model | test_ll_mean | test_ll_std | test_ll_min | test_ll_max | test_acc_mean | test_acc_std | test_acc_min | test_acc_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atp | sackmann | Hard | logit_baseline | 0.604483 | 0.000000 | 0.604483 | 0.604483 | 0.652428 | 0.000000 | 0.652428 | 0.652428 |
| atp | sackmann | Hard | xgb_baseline | 0.641440 | 0.019183 | 0.620937 | 0.660924 | 0.664131 | 0.003096 | 0.660035 | 0.668227 |
| atp | sackmann | Clay | logit_baseline | 0.614261 | 0.000000 | 0.614261 | 0.614261 | 0.655789 | 0.000000 | 0.655789 | 0.655789 |
| atp | sackmann | Clay | xgb_baseline | 0.618015 | 0.001308 | 0.616407 | 0.619585 | 0.649474 | 0.006093 | 0.643158 | 0.657895 |
| atp | sackmann | Grass | logit_baseline | 0.596310 | 0.000000 | 0.596310 | 0.596310 | 0.669841 | 0.000000 | 0.669841 | 0.669841 |
| atp | sackmann | Grass | xgb_baseline | 0.585582 | 0.003931 | 0.582681 | 0.591959 | 0.693968 | 0.007308 | 0.682540 | 0.701587 |
| atp | tml | Hard | logit_baseline | 0.603321 | 0.000000 | 0.603321 | 0.603321 | 0.650147 | 0.000000 | 0.650147 | 0.650147 |
| atp | tml | Hard | xgb_baseline | 0.625964 | 0.015478 | 0.603348 | 0.640612 | 0.660059 | 0.004138 | 0.653097 | 0.663717 |
| atp | tml | Clay | logit_baseline | 0.613107 | 0.000000 | 0.613107 | 0.613107 | 0.639327 | 0.000000 | 0.639327 | 0.639327 |
| atp | tml | Clay | xgb_baseline | 0.618336 | 0.003921 | 0.615466 | 0.625250 | 0.634490 | 0.010452 | 0.617245 | 0.644585 |
| atp | tml | Grass | logit_baseline | 0.588492 | 0.000000 | 0.588492 | 0.588492 | 0.695238 | 0.000000 | 0.695238 | 0.695238 |
| atp | tml | Grass | xgb_baseline | 0.592467 | 0.001498 | 0.591308 | 0.594856 | 0.684444 | 0.004815 | 0.679365 | 0.692063 |
| wta | sackmann | Hard | logit_baseline | 0.610653 | 0.000000 | 0.610653 | 0.610653 | 0.660796 | 0.000000 | 0.660796 | 0.660796 |
| wta | sackmann | Hard | xgb_baseline | 0.620784 | 0.011952 | 0.611102 | 0.634735 | 0.652838 | 0.002895 | 0.648402 | 0.655577 |
| wta | sackmann | Clay | logit_baseline | 0.593403 | 0.000000 | 0.593403 | 0.593403 | 0.682203 | 0.000000 | 0.682203 | 0.682203 |
| wta | sackmann | Clay | xgb_baseline | 0.599580 | 0.001888 | 0.597674 | 0.602615 | 0.662994 | 0.003816 | 0.658192 | 0.666667 |
| wta | sackmann | Grass | logit_baseline | 0.647533 | 0.000000 | 0.647533 | 0.647533 | 0.628472 | 0.000000 | 0.628472 | 0.628472 |
| wta | sackmann | Grass | xgb_baseline | 0.641407 | 0.003613 | 0.637834 | 0.646916 | 0.619444 | 0.007994 | 0.611111 | 0.628472 |

## Reportable Thresholds

- ATP/Sackmann/Hard: `0.019183`
- ATP/Sackmann/Clay: `0.001308`
- ATP/Sackmann/Grass: `0.003931`
- ATP/TML/Hard: `0.015478`
- ATP/TML/Clay: `0.003921`
- ATP/TML/Grass: `0.001498`
- WTA/Sackmann/Hard: `0.011952`
- WTA/Sackmann/Clay: `0.001888`
- WTA/Sackmann/Grass: `0.003613`
