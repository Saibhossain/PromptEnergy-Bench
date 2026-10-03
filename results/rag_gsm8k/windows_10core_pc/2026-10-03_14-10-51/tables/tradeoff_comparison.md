# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 20.00%           |       64.484 |         1726.8 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |      155.971 |         4277.8 | N/A               | No               | -20.00 pp                   | +91.4867 J                    | +2551.1 ms                     |
| BM25 RAG (top-3) | 10.00%           |      312.64  |         8578.1 | N/A               | No               | -10.00 pp                   | +248.1556 J                   | +6851.3 ms                     |
| BM25 RAG (top-5) | 10.00%           |      361.42  |         9742.6 | N/A               | No               | -10.00 pp                   | +296.9358 J                   | +8015.8 ms                     |
