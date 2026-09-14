# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 8.00%            |      316.769 |         6954.6 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 18.00%           |      807.084 |        13946.8 | N/A               | Yes              | +10.00 pp                   | +490.3154 J                   | +6992.3 ms                     |
| Zero-shot CoT    | 80.00%           |     2869.6   |        48414.5 | N/A               | Yes              | +72.00 pp                   | +2552.8326 J                  | +41459.9 ms                    |
| Short CoT        | 76.00%           |     1394.11  |        24587.6 | N/A               | Yes              | +68.00 pp                   | +1077.3432 J                  | +17633.1 ms                    |
| Long CoT         | 86.00%           |     3919.54  |        64954.9 | N/A               | Yes              | +78.00 pp                   | +3602.7721 J                  | +58000.4 ms                    |
