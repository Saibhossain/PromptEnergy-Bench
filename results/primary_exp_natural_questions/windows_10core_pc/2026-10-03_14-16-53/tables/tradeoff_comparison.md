# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |     107.009  |         2982.4 | N/A               | No               | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 10.00%           |      77.6799 |         2136.2 | N/A               | Yes              | +10.00 pp                   | -29.3287 J                    | -846.2 ms                      |
| Zero-shot CoT    | 0.00%            |     806.522  |        20493.8 | N/A               | No               | +0.00 pp                    | +699.5136 J                   | +17511.3 ms                    |
| Short CoT        | 10.00%           |     171.265  |         4249.7 | N/A               | No               | +10.00 pp                   | +64.2567 J                    | +1267.3 ms                     |
| Long CoT         | 0.00%            |    2114.57   |        49673.6 | N/A               | No               | +0.00 pp                    | +2007.5565 J                  | +46691.1 ms                    |
