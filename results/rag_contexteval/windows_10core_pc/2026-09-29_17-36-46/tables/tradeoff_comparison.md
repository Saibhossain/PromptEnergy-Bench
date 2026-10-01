# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      579.13  |        14263.1 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |      619.168 |        15026.5 |               256 | No               | +0.00 pp                    | +40.0381 J                    | +763.3 ms                      |
| BM25 RAG (top-3) | 0.00%            |      663.369 |        16068.1 |               256 | No               | +0.00 pp                    | +84.2390 J                    | +1805.0 ms                     |
| BM25 RAG (top-5) | 0.00%            |      698.254 |        16887.5 |               256 | No               | +0.00 pp                    | +119.1244 J                   | +2624.4 ms                     |
