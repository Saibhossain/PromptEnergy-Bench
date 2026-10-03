# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      202.492 |         5486.2 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 10.00%           |      290.396 |         7993.3 | N/A               | Yes              | +10.00 pp                   | +87.9040 J                    | +2507.1 ms                     |
| BM25 RAG (top-3) | 40.00%           |      549.312 |        15166.5 | N/A               | Yes              | +40.00 pp                   | +346.8202 J                   | +9680.2 ms                     |
| BM25 RAG (top-5) | 40.00%           |      811.326 |        20955.6 | N/A               | No               | +40.00 pp                   | +608.8335 J                   | +15469.4 ms                    |
