# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      25.3553 |         9927.4 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |      28.5133 |         9386.8 |               256 | Yes              | +0.00 pp                    | +3.1580 J                     | -540.6 ms                      |
| BM25 RAG (top-3) | 0.00%            |      31.2543 |        10205   |               256 | No               | +0.00 pp                    | +5.8990 J                     | +277.6 ms                      |
| BM25 RAG (top-5) | 0.00%            |      35.7687 |        10969.3 |               256 | No               | +0.00 pp                    | +10.4134 J                    | +1041.9 ms                     |
