# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      837.424 |        20573.4 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      888.041 |        21279.1 |               256 | No               | +0.00 pp                    | +50.6166 J                    | +705.7 ms                      |
| Zero-shot CoT    | 0.00%            |     1463.11  |        34597.5 |               512 | No               | +0.00 pp                    | +625.6859 J                   | +14024.1 ms                    |
| Short CoT        | 0.00%            |     1457.44  |        34499.3 |               512 | No               | +0.00 pp                    | +620.0134 J                   | +13926.0 ms                    |
| Long CoT         | 0.00%            |     2704.05  |        63352.7 |              1024 | No               | +0.00 pp                    | +1866.6292 J                  | +42779.4 ms                    |
