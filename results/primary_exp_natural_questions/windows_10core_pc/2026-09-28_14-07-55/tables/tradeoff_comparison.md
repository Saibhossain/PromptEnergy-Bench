# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      995.618 |        24630.1 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| Few-shot (3)     | 0.00%            |     1096.26  |        26472.5 |               256 | No               | +0.00 pp                    | +100.6394 J                   | +1842.5 ms                     |
| Zero-shot CoT    | 0.00%            |     2016.72  |        47898.3 |               512 | No               | +0.00 pp                    | +1021.1073 J                  | +23268.2 ms                    |
| Short CoT        | 0.00%            |     2064.2   |        48370.6 |               512 | No               | +0.00 pp                    | +1068.5856 J                  | +23740.5 ms                    |
| Long CoT         | 0.00%            |     4005.22  |        94401.2 |              1024 | No               | +0.00 pp                    | +3009.6062 J                  | +69771.1 ms                    |
