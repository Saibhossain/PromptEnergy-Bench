# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      911.422 |        28014   |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |     1173.48  |        32887.6 |               256 | No               | +0.00 pp                    | +262.0595 J                   | +4873.6 ms                     |
| Zero-shot CoT    | 0.00%            |     3502.22  |        59749.2 |               512 | No               | +0.00 pp                    | +2590.7936 J                  | +31735.1 ms                    |
| Short CoT        | 0.00%            |     3879.98  |        59676.5 |               512 | No               | +0.00 pp                    | +2968.5580 J                  | +31662.5 ms                    |
| Long CoT         | 0.00%            |    10852     |       130798   |              1024 | No               | +0.00 pp                    | +9940.5899 J                  | +102784.4 ms                   |
