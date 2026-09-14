# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      1406.12 |        28388   |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      1908.08 |        27635.3 |               256 | Yes              | +0.00 pp                    | +501.9613 J                   | -752.6 ms                      |
| Zero-shot CoT    | 0.00%            |      4178.73 |        46008.8 |               512 | No               | +0.00 pp                    | +2772.6167 J                  | +17620.8 ms                    |
| Short CoT        | 0.00%            |      4339.17 |        45453.9 |               512 | No               | +0.00 pp                    | +2933.0483 J                  | +17066.0 ms                    |
| Long CoT         | 0.00%            |      8946.03 |        89677.2 |              1024 | No               | +0.00 pp                    | +7539.9087 J                  | +61289.3 ms                    |
