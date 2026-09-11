# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Direct   | Energy Increase vs Direct   | Latency Increase vs Direct   |
|:-----------------|:-----------|-------------:|---------------:|------------------:|:-----------------|:--------------------------|:----------------------------|:-----------------------------|
| Zero-shot Direct | 4.00%      |     115.338  |        14410.4 |               256 | No               | Baseline                  | Baseline                    | Baseline                     |
| Few-shot (3)     | 4.00%      |      96.8315 |        12094.6 |               256 | Yes              | +0.00 pp                  | -18.5069 J                  | -2315.7 ms                   |
| Zero-shot CoT    | 6.00%      |     183.545  |        22949.8 |               512 | Yes              | +2.00 pp                  | +68.2062 J                  | +8539.5 ms                   |
| Short CoT        | 4.00%      |     202.097  |        25269.5 |               512 | No               | +0.00 pp                  | +86.7584 J                  | +10859.1 ms                  |
| Long CoT         | 14.00%     |     370.093  |        46323   |              1024 | Yes              | +10.00 pp                 | +254.7550 J                 | +31912.6 ms                  |
