# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      977.325 |        23886.9 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |     1076.44  |        25871.7 |               256 | No               | +0.00 pp                    | +99.1117 J                    | +1984.8 ms                     |
| BM25 RAG (top-3) | 0.00%            |     1153.02  |        27658.4 |               256 | No               | +0.00 pp                    | +175.6910 J                   | +3771.5 ms                     |
| BM25 RAG (top-5) | 0.00%            |     1230.23  |        29714.8 |               256 | No               | +0.00 pp                    | +252.9061 J                   | +5827.9 ms                     |
