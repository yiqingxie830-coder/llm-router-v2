# R1 regularization choice (dev set 5-fold, eval votes, unweighted log loss / AUC)

| Version | C | Log loss | AUC |
|---|---|---|---|
| gap | 1.0 | 0.6796 | 0.5844 |
| gap+id | 0.01 | 0.6794 | 0.5851 |
| gap+id | 0.03 | 0.6795 | 0.5850 |
| gap+id | 0.1 | 0.6795 | 0.5850 |
| gap+id | 1.0 | 0.6795 | 0.5850 |
| gap+id+e | 0.01 | 0.6767 | 0.5940 |
| gap+id+e | 0.03 | 0.6754 | 0.5977 |
| gap+id+e | 0.1 | 0.6747 | 0.5995 |
| gap+id+e | 1.0 | 0.6762 | 0.5966 |

Chosen: gap → C=1.0, gap+id → C=0.01, gap+id+e → C=0.1
