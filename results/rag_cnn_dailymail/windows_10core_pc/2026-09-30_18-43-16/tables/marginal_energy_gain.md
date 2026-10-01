# Marginal Energy Gain (MEG) Relative to Baseline

| Baseline Strategy   | Comparison Strategy   | Accuracy Change   | Energy Change   | Marginal Energy Gain (% / J)   | Latency Change   |
|:--------------------|:----------------------|:------------------|:----------------|:-------------------------------|:-----------------|
| Zero-shot Direct    | BM25 RAG (top-1)      | +0.00 pp          | +1074.0570 J    | 0.0000 %/J                     | +45657.2 ms      |
| Zero-shot Direct    | BM25 RAG (top-3)      | -30.00 pp         | +2172.8830 J    | -0.0138 %/J                    | +85258.9 ms      |
| Zero-shot Direct    | BM25 RAG (top-5)      | -30.00 pp         | +1681.3593 J    | -0.0178 %/J                    | +68229.1 ms      |
