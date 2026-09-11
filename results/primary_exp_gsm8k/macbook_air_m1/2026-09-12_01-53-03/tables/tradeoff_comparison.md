# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      34.0086 |         6210.8 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      32.3606 |         6964.4 |               256 | Yes              | +0.00 pp                    | -1.6480 J                     | +753.7 ms                      |
| Zero-shot CoT    | 0.00%            |      59.939  |        12817   |               512 | No               | +0.00 pp                    | +25.9304 J                    | +6606.2 ms                     |
| Short CoT        | 0.00%            |      60.3226 |        14363.3 |               512 | No               | +0.00 pp                    | +26.3140 J                    | +8152.5 ms                     |
| Long CoT         | 0.00%            |     122.854  |        25077.2 |              1024 | No               | +0.00 pp                    | +88.8451 J                    | +18866.5 ms                    |
