# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Direct   | Energy Increase vs Direct   | Latency Increase vs Direct   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:--------------------------|:----------------------------|:-----------------------------|
| Zero-shot Direct | 0.00%            |      16.0187 |         7486.8 |               256 | Yes              | Baseline                  | Baseline                    | Baseline                     |
| Few-shot (3)     | 0.00%            |      15.6242 |         7684.9 |               256 | Yes              | +0.00 pp                  | -0.3945 J                   | +198.2 ms                    |
| Zero-shot CoT    | 0.00%            |      38.7642 |        14255.1 |               512 | No               | +0.00 pp                  | +22.7455 J                  | +6768.3 ms                   |
| Short CoT        | 0.00%            |      40.8043 |        14769.2 |               512 | No               | +0.00 pp                  | +24.7856 J                  | +7282.4 ms                   |
| Long CoT         | 0.00%            |      82.4473 |        28367.5 |              1024 | No               | +0.00 pp                  | +66.4286 J                  | +20880.7 ms                  |
