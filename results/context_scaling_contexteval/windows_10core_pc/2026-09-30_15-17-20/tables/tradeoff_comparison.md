# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 0.00%            |      1743.47 |        44265.7 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 0.00%            |      2719.13 |        67846.8 | N/A               | No               | +0.00 pp                    | +975.6606 J                   | +23581.1 ms                    |
| Context 1024 tokens | 10.00%           |      3425.41 |        85153.5 | N/A               | Yes              | +10.00 pp                   | +1681.9400 J                  | +40887.8 ms                    |
| Context 2048 tokens | 0.00%            |      4469.84 |       110676   | N/A               | No               | +0.00 pp                    | +2726.3663 J                  | +66410.5 ms                    |
| Context 4096 tokens | 0.00%            |      3964.35 |        98185.7 | N/A               | No               | +0.00 pp                    | +2220.8849 J                  | +53920.0 ms                    |
