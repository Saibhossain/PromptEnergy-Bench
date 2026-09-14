# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 80.00%           |      7015.08 |        71416.2 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 74.00%           |      9388.64 |        92549.7 | N/A               | No               | -6.00 pp                    | +2373.5594 J                  | +21133.5 ms                    |
| Context 1024 tokens | 82.00%           |     11413.9  |       112578   | N/A               | Yes              | +2.00 pp                    | +4398.8053 J                  | +41162.1 ms                    |
| Context 2048 tokens | 76.00%           |     18199.8  |       178591   | N/A               | No               | -4.00 pp                    | +11184.7195 J                 | +107174.8 ms                   |
| Context 4096 tokens | 76.00%           |     17046.6  |       166693   | N/A               | No               | -4.00 pp                    | +10031.5616 J                 | +95277.1 ms                    |
