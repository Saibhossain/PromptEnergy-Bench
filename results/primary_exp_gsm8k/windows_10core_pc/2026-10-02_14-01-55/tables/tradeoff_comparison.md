# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 10.00%           |      494.863 |        15372.5 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 40.00%           |     1528.25  |        38485.8 | N/A               | No               | +30.00 pp                   | +1033.3886 J                  | +23113.2 ms                    |
| Zero-shot CoT    | 50.00%           |     2293.27  |        54494.7 | N/A               | No               | +40.00 pp                   | +1798.4099 J                  | +39122.2 ms                    |
| Short CoT        | 50.00%           |      653.956 |        16023.2 | N/A               | Yes              | +40.00 pp                   | +159.0921 J                   | +650.7 ms                      |
| Long CoT         | 70.00%           |     4718.89  |       101637   | N/A               | Yes              | +60.00 pp                   | +4224.0255 J                  | +86264.1 ms                    |
