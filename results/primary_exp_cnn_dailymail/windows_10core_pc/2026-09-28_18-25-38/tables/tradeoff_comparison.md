# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      1488.61 |        36151.6 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      1605.17 |        37877.1 |               256 | No               | +0.00 pp                    | +116.5529 J                   | +1725.5 ms                     |
| Zero-shot CoT    | 0.00%            |      2592.82 |        60624.8 |               512 | No               | +0.00 pp                    | +1104.2019 J                  | +24473.2 ms                    |
| Short CoT        | 0.00%            |      2659.99 |        60521.5 |               512 | No               | +0.00 pp                    | +1171.3807 J                  | +24369.9 ms                    |
| Long CoT         | 0.00%            |      4748.59 |       107596   |              1024 | No               | +0.00 pp                    | +3259.9807 J                  | +71444.8 ms                    |
