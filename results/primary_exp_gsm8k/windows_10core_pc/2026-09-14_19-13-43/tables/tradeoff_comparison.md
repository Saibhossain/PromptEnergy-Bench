# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      1069.49 |        25539   |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      2109.55 |        27791.6 |               256 | No               | +0.00 pp                    | +1040.0562 J                  | +2252.7 ms                     |
| Zero-shot CoT    | 0.00%            |      4097.79 |        45899   |               512 | No               | +0.00 pp                    | +3028.3022 J                  | +20360.0 ms                    |
| Short CoT        | 0.00%            |      4885.05 |        50693.1 |               512 | No               | +0.00 pp                    | +3815.5615 J                  | +25154.2 ms                    |
| Long CoT         | 0.00%            |      9123.96 |        90820   |              1024 | No               | +0.00 pp                    | +8054.4686 J                  | +65281.0 ms                    |
