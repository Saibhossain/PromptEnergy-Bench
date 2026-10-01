# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 60.00%           |      971.969 |        24550.1 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 70.00%           |     1298.72  |        31973.9 | N/A               | Yes              | +10.00 pp                   | +326.7497 J                   | +7423.7 ms                     |
| Zero-shot CoT    | 0.00%            |     2968.36  |        72664.1 | N/A               | No               | -60.00 pp                   | +1996.3957 J                  | +48113.9 ms                    |
| Short CoT        | 20.00%           |     1317.87  |        32133.2 | N/A               | No               | -40.00 pp                   | +345.9031 J                   | +7583.0 ms                     |
| Long CoT         | 0.00%            |     5712.74  |       140384   | N/A               | No               | -60.00 pp                   | +4740.7725 J                  | +115834.1 ms                   |
