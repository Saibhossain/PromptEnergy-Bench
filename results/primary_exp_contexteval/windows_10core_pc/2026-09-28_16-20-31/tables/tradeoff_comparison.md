# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      1065.74 |        25190.8 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      1275.29 |        29624.2 |               256 | No               | +0.00 pp                    | +209.5510 J                   | +4433.4 ms                     |
| Zero-shot CoT    | 0.00%            |      2057.75 |        48211.5 |               512 | No               | +0.00 pp                    | +992.0100 J                   | +23020.8 ms                    |
| Short CoT        | 0.00%            |      2035.71 |        48007.2 |               512 | No               | +0.00 pp                    | +969.9694 J                   | +22816.4 ms                    |
| Long CoT         | 0.00%            |      3993    |        94070.8 |              1024 | No               | +0.00 pp                    | +2927.2568 J                  | +68880.0 ms                    |
