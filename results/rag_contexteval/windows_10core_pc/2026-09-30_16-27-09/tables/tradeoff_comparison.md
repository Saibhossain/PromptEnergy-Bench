# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      326.43  |         8373.4 | N/A               | No               | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |      318.789 |         7913.4 | N/A               | Yes              | +0.00 pp                    | -7.6407 J                     | -460.0 ms                      |
| BM25 RAG (top-3) | 0.00%            |      412.743 |        10433   | N/A               | No               | +0.00 pp                    | +86.3130 J                    | +2059.7 ms                     |
| BM25 RAG (top-5) | 0.00%            |      482.242 |        12091.8 | N/A               | No               | +0.00 pp                    | +155.8125 J                   | +3718.4 ms                     |
