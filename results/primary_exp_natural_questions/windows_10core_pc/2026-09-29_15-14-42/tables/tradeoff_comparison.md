# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      574.741 |        14484.6 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      636.868 |        15589   |               256 | No               | +0.00 pp                    | +62.1272 J                    | +1104.5 ms                     |
| Zero-shot CoT    | 0.00%            |     1203.17  |        28715.4 |               512 | No               | +0.00 pp                    | +628.4331 J                   | +14230.8 ms                    |
| Short CoT        | 0.00%            |     1224.92  |        28840   |               512 | No               | +0.00 pp                    | +650.1812 J                   | +14355.4 ms                    |
| Long CoT         | 0.00%            |     2427.09  |        56842.8 |              1024 | No               | +0.00 pp                    | +1852.3469 J                  | +42358.2 ms                    |
