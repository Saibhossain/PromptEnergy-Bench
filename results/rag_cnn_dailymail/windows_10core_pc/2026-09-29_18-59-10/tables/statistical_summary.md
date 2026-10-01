# Statistical Summary Across Prompting Strategies

| Strategy         | Metric                  |     Mean |   Median |   Standard Deviation | 95% Confidence Interval   |
|:-----------------|:------------------------|---------:|---------:|---------------------:|:--------------------------|
| Zero-shot Direct | Total Accuracy          |     0    |     0    |                 0    | N/A (n=1)                 |
| Zero-shot Direct | Energy (J)              |   474.79 |   482.04 |                23.71 | N/A (n=1)                 |
| Zero-shot Direct | Total Latency (ms)      | 14071.8  | 14076.5  |               215.38 | N/A (n=1)                 |
| Zero-shot Direct | Generation Latency (ms) | 13761.1  | 13763.8  |               230.55 | N/A (n=1)                 |
| Zero-shot Direct | TTFT (ms)               |   310.68 |   273.21 |                93.57 | N/A (n=1)                 |
| Zero-shot Direct | Thinking Tokens         |   256    |   256    |                 0    | N/A (n=1)                 |
| Zero-shot Direct | Output Tokens           |   256    |   256    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Accuracy          |     0    |     0    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-1) | Energy (J)              |  1050.72 |   998.14 |               220    | N/A (n=1)                 |
| BM25 RAG (top-1) | Total Latency (ms)      | 36558    | 34591.1  |              7694.12 | N/A (n=1)                 |
| BM25 RAG (top-1) | Generation Latency (ms) | 14633.3  | 14122.4  |              1544.99 | N/A (n=1)                 |
| BM25 RAG (top-1) | TTFT (ms)               | 21924.7  | 20914.7  |              6771.02 | N/A (n=1)                 |
| BM25 RAG (top-1) | Thinking Tokens         |   256    |   256    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-1) | Output Tokens           |   256    |   256    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Accuracy          |     0    |     0    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-3) | Energy (J)              |  1398.69 |  1457.38 |               240.16 | N/A (n=1)                 |
| BM25 RAG (top-3) | Total Latency (ms)      | 48738.1  | 50862.7  |              8398.62 | N/A (n=1)                 |
| BM25 RAG (top-3) | Generation Latency (ms) | 14453.8  | 15476.8  |              3902.79 | N/A (n=1)                 |
| BM25 RAG (top-3) | TTFT (ms)               | 34284.4  | 34736.6  |              9342.75 | N/A (n=1)                 |
| BM25 RAG (top-3) | Thinking Tokens         |   235.6  |   256    |                64.51 | N/A (n=1)                 |
| BM25 RAG (top-3) | Output Tokens           |   235.6  |   256    |                64.51 | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Accuracy          |     0    |     0    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-5) | Energy (J)              |  1131.7  |  1104.67 |                66.12 | N/A (n=1)                 |
| BM25 RAG (top-5) | Total Latency (ms)      | 39573.2  | 38643.6  |              2314.07 | N/A (n=1)                 |
| BM25 RAG (top-5) | Generation Latency (ms) | 14408    | 14294    |               468.28 | N/A (n=1)                 |
| BM25 RAG (top-5) | TTFT (ms)               | 25165.3  | 24306.7  |              1979.1  | N/A (n=1)                 |
| BM25 RAG (top-5) | Thinking Tokens         |   256    |   256    |                 0    | N/A (n=1)                 |
| BM25 RAG (top-5) | Output Tokens           |   256    |   256    |                 0    | N/A (n=1)                 |
