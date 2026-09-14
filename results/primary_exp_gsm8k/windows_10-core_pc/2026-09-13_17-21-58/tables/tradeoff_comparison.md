# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      2387.39 |        27122   |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |      3283.49 |        32649.6 |               256 | No               | +0.00 pp                    | +896.0975 J                   | +5527.6 ms                     |
| Zero-shot CoT    | 0.00%            |      5503.87 |        54069.7 |               512 | No               | +0.00 pp                    | +3116.4821 J                  | +26947.7 ms                    |
| Short CoT        | 0.00%            |      4936.65 |        49889.1 |               512 | No               | +0.00 pp                    | +2549.2620 J                  | +22767.1 ms                    |
| Long CoT         | 0.00%            |      3066.57 |        60484.3 |              1024 | No               | +0.00 pp                    | +679.1764 J                   | +33362.3 ms                    |
