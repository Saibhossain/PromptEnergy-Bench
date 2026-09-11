# Statistical Summary Across Prompting Strategies

| Strategy         | Metric          |     Mean |   Median |   Standard Deviation | 95% Confidence Interval   |
|:-----------------|:----------------|---------:|---------:|---------------------:|:--------------------------|
| Zero-shot Direct | Accuracy        |     0.04 |     0    |                 0.2  | N/A (n=1)                 |
| Zero-shot Direct | Energy (J)      |   115.34 |    85.64 |                80.43 | N/A (n=1)                 |
| Zero-shot Direct | Latency (ms)    | 14410.4  | 10723.1  |             10027.5  | N/A (n=1)                 |
| Zero-shot Direct | TTFT (ms)       |  3698.64 |   614.67 |              8306.31 | N/A (n=1)                 |
| Zero-shot Direct | Thinking Tokens |   256    |   256    |                 0    | N/A (n=1)                 |
| Zero-shot Direct | Output Tokens   |   256    |   256    |                 0    | N/A (n=1)                 |
| Few-shot (3)     | Accuracy        |     0.04 |     0    |                 0.2  | N/A (n=1)                 |
| Few-shot (3)     | Energy (J)      |    96.83 |    90.3  |                53.65 | N/A (n=1)                 |
| Few-shot (3)     | Latency (ms)    | 12094.6  | 11299.9  |              6691.95 | N/A (n=1)                 |
| Few-shot (3)     | TTFT (ms)       |  1467.98 |   597.92 |              4911.83 | N/A (n=1)                 |
| Few-shot (3)     | Thinking Tokens |   256    |   256    |                 0    | N/A (n=1)                 |
| Few-shot (3)     | Output Tokens   |   256    |   256    |                 0    | N/A (n=1)                 |
| Zero-shot CoT    | Accuracy        |     0.06 |     0    |                 0.24 | N/A (n=1)                 |
| Zero-shot CoT    | Energy (J)      |   183.54 |   168.86 |                70.98 | N/A (n=1)                 |
| Zero-shot CoT    | Latency (ms)    | 22949.8  | 21118.8  |              8869.66 | N/A (n=1)                 |
| Zero-shot CoT    | TTFT (ms)       |   723.82 |   482.77 |               603.84 | N/A (n=1)                 |
| Zero-shot CoT    | Thinking Tokens |   512    |   512    |                 0    | N/A (n=1)                 |
| Zero-shot CoT    | Output Tokens   |   512    |   512    |                 0    | N/A (n=1)                 |
| Short CoT        | Accuracy        |     0.04 |     0    |                 0.2  | N/A (n=1)                 |
| Short CoT        | Energy (J)      |   202.1  |   194.5  |               108.96 | N/A (n=1)                 |
| Short CoT        | Latency (ms)    | 25269.5  | 24462    |             13595.6  | N/A (n=1)                 |
| Short CoT        | TTFT (ms)       |  3152.87 |   575.77 |              8055.39 | N/A (n=1)                 |
| Short CoT        | Thinking Tokens |   512    |   512    |                 0    | N/A (n=1)                 |
| Short CoT        | Output Tokens   |   512    |   512    |                 0    | N/A (n=1)                 |
| Long CoT         | Accuracy        |     0.14 |     0    |                 0.35 | N/A (n=1)                 |
| Long CoT         | Energy (J)      |   370.09 |   385.96 |               135.03 | N/A (n=1)                 |
| Long CoT         | Latency (ms)    | 46323    | 48359.3  |             16891.2  | N/A (n=1)                 |
| Long CoT         | TTFT (ms)       |  1405.77 |   551.41 |              5315.29 | N/A (n=1)                 |
| Long CoT         | Thinking Tokens |  1024    |  1024    |                 0    | N/A (n=1)                 |
| Long CoT         | Output Tokens   |  1024    |  1024    |                 0    | N/A (n=1)                 |
