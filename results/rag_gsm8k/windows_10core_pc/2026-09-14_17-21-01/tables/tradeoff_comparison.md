# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      2031.29 |        21441.7 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |      2835.07 |        29360.7 |               256 | No               | +0.00 pp                    | +803.7792 J                   | +7919.0 ms                     |
| BM25 RAG (top-3) | 0.00%            |      3709.9  |        38124.9 |               256 | No               | +0.00 pp                    | +1678.6084 J                  | +16683.2 ms                    |
| BM25 RAG (top-5) | 0.00%            |      4125.6  |        42205.5 |               256 | No               | +0.00 pp                    | +2094.3120 J                  | +20763.8 ms                    |
