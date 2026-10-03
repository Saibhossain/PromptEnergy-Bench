# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 0.00%            |      1888.58 |        45629.3 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 0.00%            |      2098.92 |        49316   | N/A               | No               | +0.00 pp                    | +210.3432 J                   | +3686.6 ms                     |
| Context 1024 tokens | 10.00%           |      3279.26 |        76525.1 | N/A               | Yes              | +10.00 pp                   | +1390.6868 J                  | +30895.7 ms                    |
| Context 2048 tokens | 10.00%           |      4484.51 |       100493   | N/A               | No               | +10.00 pp                   | +2595.9340 J                  | +54863.8 ms                    |
| Context 4096 tokens | 0.00%            |      2706.46 |        65475.2 | N/A               | No               | +0.00 pp                    | +817.8816 J                   | +19845.8 ms                    |
