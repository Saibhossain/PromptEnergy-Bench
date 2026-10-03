# Statistical Summary Across Prompting Strategies

| Strategy         | Metric                  | Mean     | Median   | Standard Deviation   | 95% Confidence Interval   |
|:-----------------|:------------------------|:---------|:---------|:---------------------|:--------------------------|
| Zero-shot Direct | Total Accuracy          | 0.10     | 0.00     | 0.32                 | N/A (n=1)                 |
| Zero-shot Direct | Energy (J)              | 697.87   | 494.49   | 663.33               | N/A (n=1)                 |
| Zero-shot Direct | Total Latency (ms)      | 15640.11 | 11412.83 | 14686.33             | N/A (n=1)                 |
| Zero-shot Direct | Generation Latency (ms) | 12601.34 | 9135.22  | 15723.63             | N/A (n=1)                 |
| Zero-shot Direct | TTFT (ms)               | 3038.77  | 3605.72  | 2014.83              | N/A (n=1)                 |
| Zero-shot Direct | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| Zero-shot Direct | Output Tokens           | 121.70   | 86.00    | 157.62               | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Accuracy          | 0.50     | 0.50     | 0.53                 | N/A (n=1)                 |
| BM25 RAG (top-1) | Energy (J)              | 771.03   | 811.53   | 370.71               | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Latency (ms)      | 18886.64 | 20243.41 | 9256.62              | N/A (n=1)                 |
| BM25 RAG (top-1) | Generation Latency (ms) | 10695.08 | 10806.92 | 8776.36              | N/A (n=1)                 |
| BM25 RAG (top-1) | TTFT (ms)               | 8191.56  | 7736.92  | 1731.69              | N/A (n=1)                 |
| BM25 RAG (top-1) | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-1) | Output Tokens           | 111.10   | 112.00   | 90.50                | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Accuracy          | 0.20     | 0.00     | 0.42                 | N/A (n=1)                 |
| BM25 RAG (top-3) | Energy (J)              | 1340.64  | 1330.41  | 646.39               | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Latency (ms)      | 32701.95 | 32525.09 | 15615.06             | N/A (n=1)                 |
| BM25 RAG (top-3) | Generation Latency (ms) | 15041.86 | 13615.19 | 15744.15             | N/A (n=1)                 |
| BM25 RAG (top-3) | TTFT (ms)               | 17660.09 | 17520.31 | 2318.40              | N/A (n=1)                 |
| BM25 RAG (top-3) | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-3) | Output Tokens           | 153.90   | 139.00   | 161.31               | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Accuracy          | 0.30     | 0.00     | 0.48                 | N/A (n=1)                 |
| BM25 RAG (top-5) | Energy (J)              | 1298.87  | 1288.48  | 508.57               | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Latency (ms)      | 32227.72 | 32089.37 | 12370.36             | N/A (n=1)                 |
| BM25 RAG (top-5) | Generation Latency (ms) | 11536.69 | 11742.03 | 11235.80             | N/A (n=1)                 |
| BM25 RAG (top-5) | TTFT (ms)               | 20691.03 | 20476.97 | 2539.26              | N/A (n=1)                 |
| BM25 RAG (top-5) | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-5) | Output Tokens           | 118.20   | 118.00   | 114.00               | N/A (n=1)                 |
