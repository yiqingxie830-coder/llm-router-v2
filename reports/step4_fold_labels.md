# Step 4 Bradley-Terry fitted per fold

## Votes and strong-vs-weak view size per context

|                    |   votes |   strong_vs_weak |   higher_rated_clear_win |
|:-------------------|--------:|-----------------:|-------------------------:|
| ('cv0', 'eval')    |   18205 |             1629 |                    0.468 |
| ('cv0', 'train')   |   72718 |             6704 |                    0.467 |
| ('cv1', 'eval')    |   18121 |             1709 |                    0.464 |
| ('cv1', 'train')   |   72802 |             6624 |                    0.468 |
| ('cv2', 'eval')    |   18143 |             1721 |                    0.467 |
| ('cv2', 'train')   |   72780 |             6612 |                    0.467 |
| ('cv3', 'eval')    |   18206 |             1628 |                    0.468 |
| ('cv3', 'train')   |   72717 |             6705 |                    0.468 |
| ('cv4', 'eval')    |   18248 |             1646 |                    0.469 |
| ('cv4', 'train')   |   72675 |             6687 |                    0.467 |
| ('final', 'eval')  |   10135 |              999 |                    0.465 |
| ('final', 'train') |   90923 |             8333 |                    0.468 |

## Flips of the higher-rated side

- Of 101058 votes, the higher-rated side flips between contexts on **3961 votes (3.92%)**
- Model pairs involved: 63
- Flipped votes by full-data rating gap (flips happen only where the two models were already very close):

| rating_gap           |   votes |
|:---------------------|--------:|
| (-0.001, 5.0]        |    2922 |
| (5.0, 10.0]          |     849 |
| (10.0, 20.0]         |     190 |
| (20.0, 50.0]         |       0 |
| (50.0, 1000000000.0] |       0 |

## Strong-vs-weak view membership changes: 0 votes are in the view in some contexts but not others (fixed lists, should be 0)

Lowest strong-tier rating vs highest weak-tier rating per context: cv0: 1063 > 935, cv1: 1068 > 931, cv2: 1065 > 936, cv3: 1065 > 931, cv4: 1065 > 936, final: 1065 > 932

## Largest rating difference across contexts (same model, max - min)

|       |   points |
|:------|---------:|
| count |     53   |
| mean  |      8.1 |
| std   |      4.9 |
| min   |      0.9 |
| 25%   |      4.4 |
| 50%   |      7.7 |
| 75%   |      9.2 |
| max   |     25.4 |