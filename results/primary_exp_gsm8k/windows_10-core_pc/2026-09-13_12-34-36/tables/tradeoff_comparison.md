# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      198.262 |         6679.2 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      637.777 |        13929.3 | N/A               | No               | +0.00 pp                    | +439.5149 J                   | +7250.1 ms                     |
| Zero-shot CoT    | 100.00%          |     1471.96  |        30775.4 | N/A               | No               | +100.00 pp                  | +1273.6974 J                  | +24096.2 ms                    |
| Short CoT        | 100.00%          |      617.841 |        14352.2 | N/A               | Yes              | +100.00 pp                  | +419.5788 J                   | +7673.0 ms                     |
| Long CoT         | 100.00%          |     1073.86  |        29416.5 | N/A               | No               | +100.00 pp                  | +875.5978 J                   | +22737.3 ms                    |
