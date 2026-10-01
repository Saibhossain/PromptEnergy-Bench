# Statistical Summary Across Prompting Strategies

| Strategy         | Metric                  | Mean     | Median   | Standard Deviation   | 95% Confidence Interval   |
|:-----------------|:------------------------|:---------|:---------|:---------------------|:--------------------------|
| Zero-shot Direct | Total Accuracy          | 0.20     | 0.00     | 0.42                 | N/A (n=1)                 |
| Zero-shot Direct | Energy (J)              | 333.95   | 210.77   | 313.08               | N/A (n=1)                 |
| Zero-shot Direct | Total Latency (ms)      | 9325.85  | 5849.25  | 8787.10              | N/A (n=1)                 |
| Zero-shot Direct | Generation Latency (ms) | 4781.14  | 854.91   | 8789.65              | N/A (n=1)                 |
| Zero-shot Direct | TTFT (ms)               | 4544.72  | 4437.45  | 749.78               | N/A (n=1)                 |
| Zero-shot Direct | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| Zero-shot Direct | Output Tokens           | 44.00    | 9.00     | 78.16                | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Accuracy          | 0.20     | 0.00     | 0.42                 | N/A (n=1)                 |
| BM25 RAG (top-1) | Energy (J)              | 549.78   | 535.55   | 129.54               | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Latency (ms)      | 14566.92 | 14123.06 | 3297.91              | N/A (n=1)                 |
| BM25 RAG (top-1) | Generation Latency (ms) | 2730.20  | 844.04   | 3050.40              | N/A (n=1)                 |
| BM25 RAG (top-1) | TTFT (ms)               | 11836.72 | 10971.50 | 2233.66              | N/A (n=1)                 |
| BM25 RAG (top-1) | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-1) | Output Tokens           | 24.10    | 8.00     | 24.58                | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Accuracy          | 0.50     | 0.50     | 0.53                 | N/A (n=1)                 |
| BM25 RAG (top-3) | Energy (J)              | 1458.70  | 1347.02  | 593.56               | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Latency (ms)      | 37124.28 | 34463.19 | 14999.91             | N/A (n=1)                 |
| BM25 RAG (top-3) | Generation Latency (ms) | 11797.18 | 6188.85  | 12383.70             | N/A (n=1)                 |
| BM25 RAG (top-3) | TTFT (ms)               | 25327.10 | 24661.81 | 4023.41              | N/A (n=1)                 |
| BM25 RAG (top-3) | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-3) | Output Tokens           | 98.60    | 50.00    | 102.72               | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Accuracy          | 0.50     | 0.50     | 0.53                 | N/A (n=1)                 |
| BM25 RAG (top-5) | Energy (J)              | 1393.73  | 1342.65  | 228.58               | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Latency (ms)      | 34952.25 | 33521.88 | 5737.52              | N/A (n=1)                 |
| BM25 RAG (top-5) | Generation Latency (ms) | 4537.80  | 1648.30  | 5019.96              | N/A (n=1)                 |
| BM25 RAG (top-5) | TTFT (ms)               | 30414.45 | 30383.46 | 2274.38              | N/A (n=1)                 |
| BM25 RAG (top-5) | Thinking Tokens         | N/A      | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-5) | Output Tokens           | 37.40    | 14.00    | 40.82                | N/A (n=1)                 |
