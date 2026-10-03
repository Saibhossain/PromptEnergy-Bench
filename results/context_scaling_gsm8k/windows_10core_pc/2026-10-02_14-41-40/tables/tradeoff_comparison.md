# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 50.00%           |      2642.89 |        57148.6 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 70.00%           |      3973.09 |        76524.2 | N/A               | Yes              | +20.00 pp                   | +1330.2031 J                  | +19375.6 ms                    |
| Context 1024 tokens | 50.00%           |      4062.8  |        77118.3 | N/A               | No               | +0.00 pp                    | +1419.9119 J                  | +19969.7 ms                    |
| Context 2048 tokens | 50.00%           |      4720.57 |        93591   | N/A               | No               | +0.00 pp                    | +2077.6822 J                  | +36442.4 ms                    |
| Context 4096 tokens | 50.00%           |      4162.66 |        91245   | N/A               | No               | +0.00 pp                    | +1519.7712 J                  | +34096.4 ms                    |
