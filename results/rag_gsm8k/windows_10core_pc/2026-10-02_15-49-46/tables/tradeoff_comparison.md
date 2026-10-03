# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 10.00%           |      697.87  |        15640.1 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 50.00%           |      771.034 |        18886.6 | N/A               | Yes              | +40.00 pp                   | +73.1644 J                    | +3246.5 ms                     |
| BM25 RAG (top-3) | 20.00%           |     1340.64  |        32702   | N/A               | No               | +10.00 pp                   | +642.7654 J                   | +17061.8 ms                    |
| BM25 RAG (top-5) | 30.00%           |     1298.87  |        32227.7 | N/A               | No               | +20.00 pp                   | +601.0041 J                   | +16587.6 ms                    |
