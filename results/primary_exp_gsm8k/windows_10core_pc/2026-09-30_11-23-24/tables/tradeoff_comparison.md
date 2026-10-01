# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 20.00%           |      276.3   |         9378.6 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 30.00%           |      428.774 |        12876.4 | N/A               | Yes              | +10.00 pp                   | +152.4747 J                   | +3497.7 ms                     |
| Zero-shot CoT    | 80.00%           |     1388.21  |        37533.8 | N/A               | Yes              | +60.00 pp                   | +1111.9083 J                  | +28155.2 ms                    |
| Short CoT        | 70.00%           |      612.904 |        15810   | N/A               | Yes              | +50.00 pp                   | +336.6044 J                   | +6431.3 ms                     |
| Long CoT         | 70.00%           |     2298.75  |        57949.1 | N/A               | No               | +50.00 pp                   | +2022.4525 J                  | +48570.5 ms                    |
