# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 20.00%           |       97.989 |         2965.8 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 30.00%           |      831.53  |        22060.4 | N/A               | Yes              | +10.00 pp                   | +733.5413 J                   | +19094.5 ms                    |
| Zero-shot CoT    | 20.00%           |     1136.36  |        28109.5 | N/A               | No               | +0.00 pp                    | +1038.3700 J                  | +25143.7 ms                    |
| Short CoT        | 10.00%           |      306.753 |         7611.4 | N/A               | No               | -10.00 pp                   | +208.7637 J                   | +4645.5 ms                     |
| Long CoT         | 20.00%           |     2023.73  |        49091.7 | N/A               | No               | +0.00 pp                    | +1925.7458 J                  | +46125.9 ms                    |
