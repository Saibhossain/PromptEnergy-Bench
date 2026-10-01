# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      295.981 |         7861.2 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      490.601 |        12807   | N/A               | No               | +0.00 pp                    | +194.6194 J                   | +4945.7 ms                     |
| Zero-shot CoT    | 0.00%            |     1762.27  |        44615.4 | N/A               | No               | +0.00 pp                    | +1466.2934 J                  | +36754.2 ms                    |
| Short CoT        | 0.00%            |      367.48  |         9262.2 | N/A               | No               | +0.00 pp                    | +71.4987 J                    | +1401.0 ms                     |
| Long CoT         | 10.00%           |     4349.72  |       108808   | N/A               | Yes              | +10.00 pp                   | +4053.7434 J                  | +100946.7 ms                   |
