# Energy-Accuracy-Latency Trade-off and Pareto Frontier Analysis

| Strategy         | Total Accuracy   |   Energy (J) |   Latency (ms) |   Thinking Tokens | Pareto Optimal   | Accuracy Gain vs Baseline   | Energy Increase vs Baseline   | Latency Increase vs Baseline   |
|:-----------------|:-----------------|-------------:|---------------:|------------------:|:-----------------|:----------------------------|:------------------------------|:-------------------------------|
| Zero-shot Direct | 0.00%            |      565.821 |        14440.1 |               256 | Yes              | Baseline                    | Baseline                      | Baseline                       |
| BM25 RAG (top-1) | 0.00%            |      717.758 |        18133   |               256 | No               | +0.00 pp                    | +151.9368 J                   | +3692.9 ms                     |
| BM25 RAG (top-3) | 0.00%            |     1004.81  |        25537.8 |               256 | No               | +0.00 pp                    | +438.9933 J                   | +11097.7 ms                    |
| BM25 RAG (top-5) | 0.00%            |     1054.09  |        26790.5 |               256 | No               | +0.00 pp                    | +488.2709 J                   | +12350.4 ms                    |
