# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      474.786 |        14071.8 |             256   | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |     1050.72  |        36558   |             256   | No               | +0.00 pp                    | +575.9375 J                   | +22486.2 ms                    |
| BM25 RAG (top-3) | 0.00%            |     1398.69  |        48738.1 |             235.6 | No               | +0.00 pp                    | +923.9092 J                   | +34666.4 ms                    |
| BM25 RAG (top-5) | 0.00%            |     1131.7   |        39573.2 |             256   | No               | +0.00 pp                    | +656.9179 J                   | +25501.4 ms                    |
