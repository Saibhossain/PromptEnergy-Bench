# Prompt Strategy Performance by Hardware

| Prompt Strategy | Hardware | Accuracy (%) | Mean Energy (J) | Mean TTFT (ms) | Mean Latency (ms) | Runs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ctx_0 | MacBook Air M1 | 0.0 | 33.36 | 373.8 | 22096.3 | 40 |
| ctx_1024 | MacBook Air M1 | 0.0 | 98.84 | 6725.0 | 60613.6 | 40 |
| ctx_2048 | MacBook Air M1 | 0.0 | 88.92 | 10022.5 | 49544.0 | 40 |
| ctx_4096 | MacBook Air M1 | 0.0 | 92.87 | 14902.1 | 41470.7 | 40 |
| ctx_512 | MacBook Air M1 | 0.0 | 45.12 | 2534.7 | 27989.6 | 40 |
| ctx_8192 | MacBook Air M1 | 0.0 | 152.87 | 30544.4 | 58059.3 | 40 |
| few_shot_3 | MacBook Air M1 | 1.1 | nan | 1566.1 | 11820.4 | 186 |
| long_cot | MacBook Air M1 | 1.1 | 159.45 | 1321.7 | 31678.3 | 182 |
| rag_top_1 | MacBook Air M1 | 0.0 | 28.51 | 1066.0 | 9386.8 | 40 |
| rag_top_3 | MacBook Air M1 | 0.0 | 31.25 | 2098.9 | 10205.0 | 40 |
| rag_top_5 | MacBook Air M1 | 0.0 | 35.77 | 2821.9 | 10969.3 | 40 |
| short_cot | MacBook Air M1 | 1.1 | 147.36 | 2454.9 | 19144.8 | 183 |
| zero_shot_cot | MacBook Air M1 | 1.1 | nan | 1056.0 | 19906.2 | 185 |
| zero_shot_direct | MacBook Air M1 | 2.0 | nan | 5145.1 | 14400.3 | 251 |
| ctx_0 | Windows 10-Core PC | 63.3 | 6595.98 | 6714.1 | 67151.4 | 60 |
| ctx_1024 | Windows 10-Core PC | 65.1 | 10507.02 | 53635.6 | 103958.3 | 63 |
| ctx_2048 | Windows 10-Core PC | 62.3 | 16435.11 | 111397.6 | 161542.7 | 61 |
| ctx_4096 | Windows 10-Core PC | 62.3 | 15606.19 | 100764.8 | 152885.2 | 61 |
| ctx_512 | Windows 10-Core PC | 58.7 | 8620.76 | 25897.2 | 85599.5 | 63 |
| ctx_8192 | Windows 10-Core PC | 0.0 | 8958.46 | 44011.7 | 88498.3 | 10 |
| few_shot_3 | Windows 10-Core PC | 3.3 | 2262.75 | 5862.4 | 24756.2 | 275 |
| long_cot | Windows 10-Core PC | 26.7 | 5690.18 | 3788.3 | 74119.6 | 165 |
| rag_top_1 | Windows 10-Core PC | 0.0 | 2684.81 | 8210.3 | 29799.8 | 11 |
| rag_top_3 | Windows 10-Core PC | 0.0 | 3709.90 | 17031.9 | 38124.9 | 10 |
| rag_top_5 | Windows 10-Core PC | 0.0 | 4125.60 | 20214.8 | 42205.5 | 10 |
| short_cot | Windows 10-Core PC | 14.2 | 4299.37 | 3506.1 | 45200.2 | 275 |
| zero_shot_cot | Windows 10-Core PC | 14.9 | 4518.38 | 3424.6 | 49145.8 | 275 |
| zero_shot_direct | Windows 10-Core PC | 1.4 | 1743.56 | 2363.1 | 20974.9 | 290 |
