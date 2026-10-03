# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | N/A              |      644.586 |        15636.6 | N/A               | No               | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      784.84  |        18276.6 | N/A               | Yes              | N/A                         | +140.2543 J                   | +2639.9 ms                     |
| Zero-shot CoT    | 0.00%            |     3441.55  |        78949.7 | N/A               | No               | N/A                         | +2796.9609 J                  | +63313.0 ms                    |
| Short CoT        | N/A              |      827.048 |        19289.4 | N/A               | No               | N/A                         | +182.4621 J                   | +3652.8 ms                     |
| Long CoT         | 0.00%            |     5853.54  |       131482   | N/A               | No               | N/A                         | +5208.9552 J                  | +115845.5 ms                   |
