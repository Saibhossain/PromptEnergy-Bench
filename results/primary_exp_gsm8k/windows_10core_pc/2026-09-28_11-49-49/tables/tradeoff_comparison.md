# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      1071.2  |        27770   |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      1499.78 |        32623.9 |               256 | No               | +0.00 pp                    | +428.5833 J                   | +4853.9 ms                     |
| Zero-shot CoT    | 0.00%            |      2324.56 |        51056.8 |               512 | No               | +0.00 pp                    | +1253.3574 J                  | +23286.8 ms                    |
| Short CoT        | 0.00%            |      2338.12 |        51447.8 |               512 | No               | +0.00 pp                    | +1266.9179 J                  | +23677.8 ms                    |
| Long CoT         | 0.00%            |      4493.35 |        98915.5 |              1024 | No               | +0.00 pp                    | +3422.1516 J                  | +71145.5 ms                    |
