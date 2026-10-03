# Marginal Energy Gain (MEG) Relative to Baseline

| Baseline Strategy   | Comparison Strategy   | Accuracy Change   | Energy Change   | Marginal Energy Gain (% / J)   | Latency Change   |
|:--------------------|:----------------------|:------------------|:----------------|:-------------------------------|:-----------------|
| Zero-shot Direct    | BM25 RAG (top-1)      | +40.00 pp         | +73.1644 J      | 0.5467 %/J                     | +3246.5 ms       |
| Zero-shot Direct    | BM25 RAG (top-3)      | +10.00 pp         | +642.7654 J     | 0.0156 %/J                     | +17061.8 ms      |
| Zero-shot Direct    | BM25 RAG (top-5)      | +20.00 pp         | +601.0041 J     | 0.0333 %/J                     | +16587.6 ms      |
