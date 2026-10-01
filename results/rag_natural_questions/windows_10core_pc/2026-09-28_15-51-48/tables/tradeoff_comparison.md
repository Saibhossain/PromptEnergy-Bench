# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      997.179 |        24601.2 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |     1339.01  |        32523.1 |               256 | No               | +0.00 pp                    | +341.8327 J                   | +7921.9 ms                     |
| BM25 RAG (top-3) | 0.00%            |     1889.82  |        46630.1 |               256 | No               | +0.00 pp                    | +892.6422 J                   | +22028.9 ms                    |
| BM25 RAG (top-5) | 0.00%            |     2047.17  |        48759.4 |               256 | No               | +0.00 pp                    | +1049.9941 J                  | +24158.2 ms                    |
