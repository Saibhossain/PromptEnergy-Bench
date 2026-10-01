# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      542.06  |        15706.3 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      678.333 |        17748.8 |               256 | No               | +0.00 pp                    | +136.2735 J                   | +2042.5 ms                     |
| Zero-shot CoT    | 0.00%            |     1207.78  |        29506.8 |               512 | No               | +0.00 pp                    | +665.7163 J                   | +13800.6 ms                    |
| Short CoT        | 0.00%            |     1244.08  |        29593.8 |               512 | No               | +0.00 pp                    | +702.0200 J                   | +13887.5 ms                    |
| Long CoT         | 0.00%            |     2471.27  |        57850.2 |              1024 | No               | +0.00 pp                    | +1929.2054 J                  | +42143.9 ms                    |
