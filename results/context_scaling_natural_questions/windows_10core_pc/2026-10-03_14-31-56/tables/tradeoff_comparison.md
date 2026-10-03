# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 0.00%            |      806.282 |        20307.5 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 10.00%           |     1228.65  |        30301.4 | N/A               | Yes              | +10.00 pp                   | +422.3713 J                   | +9993.9 ms                     |
| Context 1024 tokens | 0.00%            |     1608.71  |        38040.2 | N/A               | No               | +0.00 pp                    | +802.4312 J                   | +17732.7 ms                    |
| Context 2048 tokens | 10.00%           |     2158.68  |        51650.1 | N/A               | No               | +10.00 pp                   | +1352.3977 J                  | +31342.6 ms                    |
| Context 4096 tokens | 0.00%            |     1538.02  |        36978.1 | N/A               | No               | +0.00 pp                    | +731.7363 J                   | +16670.6 ms                    |
