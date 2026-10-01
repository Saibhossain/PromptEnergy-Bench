# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      168.096 |         4373.6 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      191.507 |         5012.6 | N/A               | No               | +0.00 pp                    | +23.4112 J                    | +639.0 ms                      |
| Zero-shot CoT    | 0.00%            |     1190.14  |        30281.5 | N/A               | No               | +0.00 pp                    | +1022.0403 J                  | +25908.0 ms                    |
| Short CoT        | 0.00%            |      254.426 |         6368.6 | N/A               | No               | +0.00 pp                    | +86.3296 J                    | +1995.0 ms                     |
| Long CoT         | 0.00%            |     4492.55  |       110538   | N/A               | No               | +0.00 pp                    | +4324.4555 J                  | +106164.7 ms                   |
