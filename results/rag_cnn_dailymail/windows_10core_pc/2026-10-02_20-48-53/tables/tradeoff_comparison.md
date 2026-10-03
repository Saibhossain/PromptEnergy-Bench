# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) | Thinking Tokens   | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|:------------------|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 40.00%           |      320.577 |         9418.6 | N/A               | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 30.00%           |     1021.06  |        34280.4 | N/A               | No               | -10.00 pp                   | +700.4873 J                   | +24861.9 ms                    |
| BM25 RAG (top-3) | 30.00%           |     1428.32  |        46012.1 | N/A               | No               | -10.00 pp                   | +1107.7411 J                  | +36593.5 ms                    |
| BM25 RAG (top-5) | 30.00%           |     1622.94  |        52149.6 | N/A               | No               | -10.00 pp                   | +1302.3612 J                  | +42731.1 ms                    |
