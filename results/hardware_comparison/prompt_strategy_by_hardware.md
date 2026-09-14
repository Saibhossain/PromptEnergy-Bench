# Prompt Strategy Performance by Hardware

| Prompt Strategy | Hardware | Accuracy (%) | Mean Energy (J) | Mean TTFT (ms) | Mean Latency (ms) | Runs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| few_shot_3 | MacBook Air M1 | 1.4 | nan | 1618.9 | 10472.9 | 146 |
| long_cot | MacBook Air M1 | 1.4 | 182.60 | 1537.7 | 26706.2 | 142 |
| short_cot | MacBook Air M1 | 1.4 | 177.85 | 2935.8 | 16603.5 | 143 |
| zero_shot_cot | MacBook Air M1 | 1.4 | nan | 1039.1 | 15604.8 | 145 |
| zero_shot_direct | MacBook Air M1 | 2.9 | nan | 7368.7 | 17292.5 | 171 |
| ctx_0 | Windows 10-Core PC | 77.6 | 7062.85 | 7705.6 | 71672.5 | 49 |
| ctx_1024 | Windows 10-Core PC | 78.8 | 11269.05 | 59978.5 | 111302.6 | 52 |
| ctx_2048 | Windows 10-Core PC | 76.0 | 18199.80 | 127782.3 | 178591.0 | 50 |
| ctx_4096 | Windows 10-Core PC | 76.0 | 17046.64 | 113368.7 | 166693.3 | 50 |
| ctx_512 | Windows 10-Core PC | 71.2 | 9244.47 | 28896.7 | 91557.1 | 52 |
| few_shot_3 | Windows 10-Core PC | 8.7 | 2008.02 | 9877.3 | 23143.7 | 103 |
| long_cot | Windows 10-Core PC | 42.7 | 3464.61 | 4151.7 | 62404.3 | 103 |
| rag_top_1 | Windows 10-Core PC | 0.0 | 1182.21 | 7542.1 | 34191.5 | 1 |
| short_cot | Windows 10-Core PC | 37.9 | 3110.39 | 4510.2 | 36914.3 | 103 |
| zero_shot_cot | Windows 10-Core PC | 39.8 | 4120.48 | 4507.6 | 50742.3 | 103 |
| zero_shot_direct | Windows 10-Core PC | 3.9 | 1324.60 | 4207.5 | 16916.0 | 103 |
