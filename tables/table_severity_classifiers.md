| Type   | Split   | Test Macro-F1   | ROC-AUC   | 5-fold CV   |   Train rows | Notes                          |
|:-------|:--------|:----------------|:----------|:------------|-------------:|:-------------------------------|
| EQ     | random  | 0.878           | 0.935     | 0.920±0.019 |          670 | rapidpopdescription null 36%   |
| TC     | time    | 0.815           | 0.926     | 0.634±0.062 |         1438 | val only 4 orange rows         |
| WF     | random  | 0.798           | 0.96      | 0.779±0.098 |          544 | orange_or_red globally only 44 |
| DR     | time    | ~0.51 (weak)    | —         | 0.508±0.063 |          258 | Known weak; test set too small |