# PC1 language confound check: quintile table redone on English votes only

Strong-vs-weak votes (dev set). Lower t1 = the strong tier has more of an edge (open-ended advice end); higher t1 = closer to the code and definite-answer end.

## Out-of-fold t1 (main table)

English share of votes 55.3% (langdetect), 56.1% (dataset language column).

|                                                  |   1 (lowest) |      2 |      3 |      4 |   5 (highest) |
|:-------------------------------------------------|-------------:|-------:|-------:|-------:|--------------:|
| all languages · votes                            |       1667   | 1666   | 1667   | 1666   |        1667   |
| all languages · strong win %                     |         75.5 |   68.1 |   60.6 |   55.5 |          50.9 |
| all languages · tie %                            |         12.8 |   16.9 |   21.9 |   23.8 |          27.8 |
| all languages · weak win %                       |         11.7 |   15.1 |   17.5 |   20.6 |          21.4 |
| English (langdetect) · votes                     |        766   |  822   |  906   |  983   |        1128   |
| English (langdetect) · strong win %              |         76.1 |   66.3 |   58.6 |   56   |          50.1 |
| English (langdetect) · tie %                     |         11   |   16.9 |   22.6 |   22.9 |          28.7 |
| English (langdetect) · weak win %                |         12.9 |   16.8 |   18.8 |   21.2 |          21.2 |
| English (dataset language column) · votes        |        776   |  841   |  927   | 1004   |        1130   |
| English (dataset language column) · strong win % |         75.8 |   66   |   59   |   55.9 |          51.2 |
| English (dataset language column) · tie %        |         11.3 |   16.9 |   22.7 |   22.7 |          28.3 |
| English (dataset language column) · weak win %   |         12.9 |   17.1 |   18.3 |   21.4 |          20.4 |
| English, quintiles re-cut · votes                |        921   |  921   |  921   |  921   |         921   |
| English, quintiles re-cut · strong win %         |         74.5 |   65.8 |   55.6 |   56.7 |          48.6 |
| English, quintiles re-cut · tie %                |         11.7 |   17.8 |   25.1 |   21.8 |          29.6 |
| English, quintiles re-cut · weak win %           |         13.8 |   16.4 |   19.3 |   21.5 |          21.7 |

## In-sample t1

English share of votes 55.3% (langdetect), 56.1% (dataset language column).

|                                                  |   1 (lowest) |      2 |      3 |      4 |   5 (highest) |
|:-------------------------------------------------|-------------:|-------:|-------:|-------:|--------------:|
| all languages · votes                            |       1667   | 1666   | 1667   | 1666   |        1667   |
| all languages · strong win %                     |         77.3 |   68.8 |   62.5 |   53.1 |          48.9 |
| all languages · tie %                            |         12.3 |   16.6 |   21.1 |   25.2 |          28.1 |
| all languages · weak win %                       |         10.4 |   14.6 |   16.4 |   21.8 |          23   |
| English (langdetect) · votes                     |        799   |  826   |  907   |  943   |        1130   |
| English (langdetect) · strong win %              |         78.1 |   66.7 |   60.4 |   53.3 |          48.5 |
| English (langdetect) · tie %                     |         10.4 |   17.3 |   22.3 |   23.6 |          28.8 |
| English (langdetect) · weak win %                |         11.5 |   16   |   17.3 |   23   |          22.7 |
| English (dataset language column) · votes        |        810   |  841   |  932   |  960   |        1135   |
| English (dataset language column) · strong win % |         77.9 |   66.5 |   60.5 |   53.5 |          49.5 |
| English (dataset language column) · tie %        |         10.6 |   17.4 |   22.5 |   23.2 |          28.5 |
| English (dataset language column) · weak win %   |         11.5 |   16.2 |   17   |   23.2 |          22   |
| English, quintiles re-cut · votes                |        921   |  921   |  921   |  921   |         921   |
| English, quintiles re-cut · strong win %         |         75.7 |   66.9 |   57.4 |   54.2 |          47   |
| English, quintiles re-cut · tie %                |         11.1 |   18.5 |   23.3 |   24   |          29.2 |
| English, quintiles re-cut · weak win %           |         13.2 |   14.7 |   19.2 |   21.8 |          23.8 |
