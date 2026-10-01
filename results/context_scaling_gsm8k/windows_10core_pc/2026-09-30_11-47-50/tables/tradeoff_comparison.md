# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 80.00%           |      1338.96 |        33908.9 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 60.00%           |      2012.01 |        49368.5 | N/A               | No               | -20.00 pp                   | +673.0505 J                   | +15459.5 ms                    |
| Context 1024 tokens | 70.00%           |      2354    |        56141.3 | N/A               | No               | -10.00 pp                   | +1015.0483 J                  | +22232.4 ms                    |
| Context 2048 tokens | 80.00%           |      3432.42 |        81990.5 | N/A               | No               | +0.00 pp                    | +2093.4673 J                  | +48081.6 ms                    |
| Context 4096 tokens | 60.00%           |      4055.88 |        98206.4 | N/A               | No               | -20.00 pp                   | +2716.9273 J                  | +64297.5 ms                    |
