# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy            | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:--------------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Context 0 tokens    | 20.00%           |      1103.58 |        27333.7 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Context 512 tokens  | 20.00%           |      1370.49 |        33540.9 | N/A               | No               | +0.00 pp                    | +266.9115 J                   | +6207.3 ms                     |
| Context 1024 tokens | 30.00%           |      1599.82 |        38431.9 | N/A               | Yes              | +10.00 pp                   | +496.2444 J                   | +11098.2 ms                    |
| Context 2048 tokens | 40.00%           |      1830.59 |        44174.3 | N/A               | Yes              | +20.00 pp                   | +727.0108 J                   | +16840.7 ms                    |
| Context 4096 tokens | 30.00%           |      1828.91 |        44444.8 | N/A               | No               | +10.00 pp                   | +725.3300 J                   | +17111.1 ms                    |
