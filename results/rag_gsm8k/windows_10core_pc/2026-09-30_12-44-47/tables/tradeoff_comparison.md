# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 20.00%           |      333.953 |         9325.9 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 20.00%           |      549.78  |        14566.9 | N/A               | No               | +0.00 pp                    | +215.8262 J                   | +5241.1 ms                     |
| BM25 RAG (top-3) | 50.00%           |     1458.7   |        37124.3 | N/A               | No               | +30.00 pp                   | +1124.7448 J                  | +27798.4 ms                    |
| BM25 RAG (top-5) | 50.00%           |     1393.73  |        34952.2 | N/A               | Yes              | +30.00 pp                   | +1059.7729 J                  | +25626.4 ms                    |
