# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 0.00%            |      2310.33 |        51030.9 |               512 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 10.00%           |      2646.37 |        60225.7 |               512 | Yes              | +10.00 pp                   | +336.0379 J                   | +9194.8 ms                     |
| Context 1024 tokens | 0.00%            |      3309.68 |        74859.7 |               512 | No               | +0.00 pp                    | +999.3502 J                   | +23828.7 ms                    |
| Context 2048 tokens | 0.00%            |      4132.53 |        91660.7 |               512 | No               | +0.00 pp                    | +1822.2060 J                  | +40629.8 ms                    |
| Context 4096 tokens | 0.00%            |      4285.41 |        97293.5 |               512 | No               | +0.00 pp                    | +1975.0860 J                  | +46262.6 ms                    |
