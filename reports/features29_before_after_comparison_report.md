# 29-Feature Model Comparison Report

Comparison: `data/models_before_29_features` vs current `data/models`.

## ATP Sackmann - Hard
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 1709 | 0.6606 | 0.6524 | -0.0082 worse | 0.7295 | 0.7317 | +0.0022 good | 0.6065 | 0.6044 | -0.0021 good | 0.2100 | 0.2092 | -0.0007 good |  |  |
| logit_tuned | 1709 | 0.6618 | 0.6612 | -0.0006 worse | 0.7295 | 0.7322 | +0.0027 good | 0.6053 | 0.6030 | -0.0023 good | 0.2097 | 0.2088 | -0.0009 good |  |  |
| xgb | 1709 | 0.6659 | 0.6600 | -0.0059 worse | 0.7301 | 0.7292 | -0.0009 worse | 0.6605 | 0.6617 | +0.0012 worse | 0.2088 | 0.2092 | +0.0004 worse | isotonic | isotonic |
| xgb_tuned | 1709 | 0.6653 | 0.6665 | +0.0012 good | 0.7320 | 0.7326 | +0.0006 good | 0.6216 | 0.6395 | +0.0179 worse | 0.2091 | 0.2078 | -0.0013 good | isotonic | isotonic |

## ATP Sackmann - Clay
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 951 | 0.6540 | 0.6562 | +0.0021 good | 0.7192 | 0.7177 | -0.0015 worse | 0.6105 | 0.6141 | +0.0036 worse | 0.2124 | 0.2137 | +0.0013 worse |  |  |
| logit_tuned | 951 | 0.6625 | 0.6519 | -0.0105 worse | 0.7198 | 0.7160 | -0.0038 worse | 0.6108 | 0.6133 | +0.0025 worse | 0.2123 | 0.2135 | +0.0012 worse |  |  |
| xgb | 951 | 0.6583 | 0.6467 | -0.0116 worse | 0.7143 | 0.7099 | -0.0044 worse | 0.6555 | 0.6181 | -0.0374 good | 0.2159 | 0.2155 | -0.0004 good | isotonic | None |
| xgb_tuned | 951 | 0.6530 | 0.6456 | -0.0074 worse | 0.7152 | 0.7118 | -0.0035 worse | 0.6150 | 0.6161 | +0.0011 worse | 0.2141 | 0.2147 | +0.0006 worse | None | None |

## ATP Sackmann - Grass
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 315 | 0.6889 | 0.6698 | -0.0190 worse | 0.7463 | 0.7423 | -0.0040 worse | 0.5907 | 0.5973 | +0.0066 worse | 0.2041 | 0.2066 | +0.0025 worse |  |  |
| logit_tuned | 315 | 0.6825 | 0.6921 | +0.0095 good | 0.7418 | 0.7478 | +0.0059 good | 0.5917 | 0.5892 | -0.0025 good | 0.2045 | 0.2033 | -0.0011 good |  |  |
| xgb | 315 | 0.6921 | 0.6952 | +0.0032 good | 0.7467 | 0.7685 | +0.0217 good | 0.8055 | 0.5828 | -0.2227 good | 0.2053 | 0.1984 | -0.0069 good | isotonic | None |
| xgb_tuned | 315 | 0.6921 | 0.6857 | -0.0063 worse | 0.7598 | 0.7575 | -0.0023 worse | 0.5824 | 0.5898 | +0.0074 worse | 0.1991 | 0.2016 | +0.0024 worse | None | None |

## ATP TML - Hard
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 3629 | 0.6514 | 0.6503 | -0.0011 worse | 0.7191 | 0.7209 | +0.0018 good | 0.6174 | 0.6178 | +0.0003 worse | 0.2139 | 0.2137 | -0.0002 good |  |  |
| logit_tuned | 3629 | 0.6514 | 0.6575 | +0.0061 good | 0.7194 | 0.7204 | +0.0010 good | 0.6153 | 0.6152 | -0.0001 good | 0.2134 | 0.2133 | -0.0001 good |  |  |
| xgb | 3629 | 0.6870 | 0.6856 | -0.0014 worse | 0.7540 | 0.7545 | +0.0006 good | 0.5964 | 0.5861 | -0.0104 good | 0.2015 | 0.2007 | -0.0007 good | isotonic | isotonic |
| xgb_tuned | 3629 | 0.6823 | 0.6900 | +0.0077 good | 0.7490 | 0.7569 | +0.0078 good | 0.6028 | 0.5845 | -0.0182 good | 0.2040 | 0.2000 | -0.0039 good | isotonic | isotonic |

## ATP TML - Clay
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 1767 | 0.6684 | 0.6508 | -0.0175 worse | 0.7281 | 0.7275 | -0.0006 worse | 0.6120 | 0.6161 | +0.0041 worse | 0.2106 | 0.2113 | +0.0007 worse |  |  |
| logit_tuned | 1767 | 0.6661 | 0.6593 | -0.0068 worse | 0.7276 | 0.7259 | -0.0017 worse | 0.6100 | 0.6117 | +0.0017 worse | 0.2105 | 0.2109 | +0.0004 worse |  |  |
| xgb | 1767 | 0.6904 | 0.7057 | +0.0153 good | 0.7748 | 0.7754 | +0.0006 good | 0.5893 | 0.5718 | -0.0175 good | 0.1943 | 0.1937 | -0.0006 good | isotonic | None |
| xgb_tuned | 1767 | 0.6989 | 0.6967 | -0.0023 worse | 0.7751 | 0.7766 | +0.0015 good | 0.5688 | 0.5685 | -0.0003 good | 0.1933 | 0.1928 | -0.0005 good | None | None |

## ATP TML - Grass
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 624 | 0.6747 | 0.6955 | +0.0208 good | 0.7517 | 0.7622 | +0.0105 good | 0.5896 | 0.5802 | -0.0094 good | 0.2033 | 0.1994 | -0.0039 good |  |  |
| logit_tuned | 624 | 0.6619 | 0.6811 | +0.0192 good | 0.7509 | 0.7597 | +0.0088 good | 0.5900 | 0.5842 | -0.0058 good | 0.2036 | 0.2007 | -0.0029 good |  |  |
| xgb | 624 | 0.7212 | 0.7260 | +0.0048 good | 0.7967 | 0.8073 | +0.0106 good | 0.7086 | 0.5357 | -0.1728 good | 0.1853 | 0.1804 | -0.0049 good | isotonic | None |
| xgb_tuned | 624 | 0.7260 | 0.7019 | -0.0240 worse | 0.8044 | 0.8049 | +0.0005 good | 0.5427 | 0.5386 | -0.0041 good | 0.1823 | 0.1820 | -0.0003 good | None | None |

## WTA Sackmann - Hard
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 1533 | 0.6497 | 0.6588 | +0.0091 good | 0.7232 | 0.7216 | -0.0016 worse | 0.6091 | 0.6106 | +0.0015 worse | 0.2114 | 0.2122 | +0.0008 worse |  |  |
| logit_tuned | 1533 | 0.6549 | 0.6654 | +0.0104 good | 0.7246 | 0.7232 | -0.0014 worse | 0.6079 | 0.6088 | +0.0009 worse | 0.2108 | 0.2114 | +0.0006 worse |  |  |
| xgb | 1533 | 0.6523 | 0.6523 | 0.0000 | 0.7184 | 0.7203 | +0.0018 good | 0.6532 | 0.6138 | -0.0395 good | 0.2127 | 0.2132 | +0.0005 worse | isotonic | isotonic |
| xgb_tuned | 1533 | 0.6464 | 0.6484 | +0.0020 good | 0.7193 | 0.7211 | +0.0018 good | 0.6355 | 0.6109 | -0.0246 good | 0.2140 | 0.2125 | -0.0015 good | isotonic | isotonic |

## WTA Sackmann - Clay
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 709 | 0.6573 | 0.6812 | +0.0240 good | 0.7311 | 0.7492 | +0.0181 good | 0.6020 | 0.5925 | -0.0095 good | 0.2091 | 0.2040 | -0.0051 good |  |  |
| logit_tuned | 709 | 0.6615 | 0.6657 | +0.0042 good | 0.7320 | 0.7455 | +0.0135 good | 0.6012 | 0.5937 | -0.0075 good | 0.2090 | 0.2052 | -0.0037 good |  |  |
| xgb | 709 | 0.6573 | 0.6671 | +0.0099 good | 0.7357 | 0.7368 | +0.0011 good | 0.5968 | 0.6000 | +0.0032 worse | 0.2073 | 0.2073 | +0.0000 worse | isotonic | None |
| xgb_tuned | 709 | 0.6671 | 0.6502 | -0.0169 worse | 0.7324 | 0.7334 | +0.0010 good | 0.6017 | 0.6015 | -0.0003 good | 0.2086 | 0.2082 | -0.0003 good | None | None |

## WTA Sackmann - Grass
| Model | Rows | Accuracy Before | Accuracy After | Accuracy Delta | ROC AUC Before | ROC AUC After | ROC AUC Delta | Log Loss Before | Log Loss After | Log Loss Delta | Brier Before | Brier After | Brier Delta | Calibration Before | Calibration After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | 288 | 0.6424 | 0.6181 | -0.0243 worse | 0.7095 | 0.6847 | -0.0247 worse | 0.6203 | 0.6470 | +0.0267 worse | 0.2164 | 0.2280 | +0.0115 worse |  |  |
| logit_tuned | 288 | 0.6250 | 0.6285 | +0.0035 good | 0.7072 | 0.6907 | -0.0165 worse | 0.6212 | 0.6337 | +0.0125 worse | 0.2165 | 0.2222 | +0.0056 worse |  |  |
| xgb | 288 | 0.6632 | 0.6215 | -0.0417 worse | 0.6962 | 0.6897 | -0.0065 worse | 0.6391 | 0.6360 | -0.0031 good | 0.2228 | 0.2235 | +0.0007 worse | isotonic | None |
| xgb_tuned | 288 | 0.6319 | 0.6215 | -0.0104 worse | 0.7119 | 0.6856 | -0.0264 worse | 0.6222 | 0.6371 | +0.0150 worse | 0.2165 | 0.2241 | +0.0076 worse | None | None |
