# Statistical Summary Across Prompting Strategies

| Strategy         | Metric                  | Mean    | Median   | Standard Deviation   | 95% Confidence Interval   |
|:-----------------|:------------------------|:--------|:---------|:---------------------|:--------------------------|
| Zero-shot Direct | Total Accuracy          | 0.20    | 0.00     | 0.42                 | N/A (n=1)                 |
| Zero-shot Direct | Energy (J)              | 64.48   | 52.67    | 55.12                | N/A (n=1)                 |
| Zero-shot Direct | Total Latency (ms)      | 1726.77 | 1378.42  | 1418.04              | N/A (n=1)                 |
| Zero-shot Direct | Generation Latency (ms) | 843.03  | 327.19   | 1072.87              | N/A (n=1)                 |
| Zero-shot Direct | TTFT (ms)               | 883.74  | 682.71   | 774.99               | N/A (n=1)                 |
| Zero-shot Direct | Thinking Tokens         | N/A     | N/A      | N/A                  | N/A                       |
| Zero-shot Direct | Output Tokens           | 16.20   | 6.00     | 19.42                | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Accuracy          | 0.00    | 0.00     | 0.00                 | N/A (n=1)                 |
| BM25 RAG (top-1) | Energy (J)              | 155.97  | 142.75   | 34.86                | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Latency (ms)      | 4277.84 | 3945.22  | 1018.53              | N/A (n=1)                 |
| BM25 RAG (top-1) | Generation Latency (ms) | 294.12  | 280.84   | 110.35               | N/A (n=1)                 |
| BM25 RAG (top-1) | TTFT (ms)               | 3983.72 | 3679.14  | 981.26               | N/A (n=1)                 |
| BM25 RAG (top-1) | Thinking Tokens         | N/A     | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-1) | Output Tokens           | 6.70    | 6.50     | 2.26                 | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Accuracy          | 0.10    | 0.00     | 0.32                 | N/A (n=1)                 |
| BM25 RAG (top-3) | Energy (J)              | 312.64  | 298.80   | 67.78                | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Latency (ms)      | 8578.11 | 8132.82  | 1928.16              | N/A (n=1)                 |
| BM25 RAG (top-3) | Generation Latency (ms) | 262.95  | 271.60   | 78.52                | N/A (n=1)                 |
| BM25 RAG (top-3) | TTFT (ms)               | 8315.16 | 7880.18  | 1906.57              | N/A (n=1)                 |
| BM25 RAG (top-3) | Thinking Tokens         | N/A     | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-3) | Output Tokens           | 5.90    | 6.00     | 1.60                 | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Accuracy          | 0.10    | 0.00     | 0.32                 | N/A (n=1)                 |
| BM25 RAG (top-5) | Energy (J)              | 361.42  | 365.23   | 38.32                | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Latency (ms)      | 9742.60 | 9917.62  | 989.19               | N/A (n=1)                 |
| BM25 RAG (top-5) | Generation Latency (ms) | 286.05  | 250.00   | 135.58               | N/A (n=1)                 |
| BM25 RAG (top-5) | TTFT (ms)               | 9456.55 | 9547.19  | 958.31               | N/A (n=1)                 |
| BM25 RAG (top-5) | Thinking Tokens         | N/A     | N/A      | N/A                  | N/A                       |
| BM25 RAG (top-5) | Output Tokens           | 6.20    | 5.50     | 2.70                 | N/A (n=1)                 |
