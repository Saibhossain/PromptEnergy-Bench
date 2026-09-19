# Cross-Hardware Latency and Phase Decomposition

| Hardware | Prompt Strategy | Mean TTFT (ms) | Mean Decode Latency (ms) | Mean Total Latency (ms) |
| ---: | ---: | ---: | ---: | ---: |
| MacBook Air M1 | ctx_0 | 373.8 | 21722.4 | 22096.3 |
| MacBook Air M1 | ctx_1024 | 6725.0 | 53888.6 | 60613.6 |
| MacBook Air M1 | ctx_2048 | 10022.5 | 39521.6 | 49544.0 |
| MacBook Air M1 | ctx_4096 | 14902.1 | 26568.7 | 41470.7 |
| MacBook Air M1 | ctx_512 | 2534.7 | 25454.9 | 27989.6 |
| MacBook Air M1 | ctx_8192 | 30544.4 | 27514.9 | 58059.3 |
| MacBook Air M1 | few_shot_3 | 1566.1 | 10254.3 | 11820.4 |
| MacBook Air M1 | long_cot | 1321.7 | 30356.7 | 31678.3 |
| MacBook Air M1 | rag_top_1 | 1066.0 | 8320.8 | 9386.8 |
| MacBook Air M1 | rag_top_3 | 2098.9 | 8106.1 | 10205.0 |
| MacBook Air M1 | rag_top_5 | 2821.9 | 8147.3 | 10969.3 |
| MacBook Air M1 | short_cot | 2454.9 | 16689.9 | 19144.8 |
| MacBook Air M1 | zero_shot_cot | 1056.0 | 18850.2 | 19906.2 |
| MacBook Air M1 | zero_shot_direct | 5145.1 | 9255.2 | 14400.3 |
| Windows 10-Core PC | ctx_0 | 6714.1 | 60437.3 | 67151.4 |
| Windows 10-Core PC | ctx_1024 | 53635.6 | 50322.8 | 103958.3 |
| Windows 10-Core PC | ctx_2048 | 111397.6 | 50145.1 | 161542.7 |
| Windows 10-Core PC | ctx_4096 | 100764.8 | 52120.4 | 152885.2 |
| Windows 10-Core PC | ctx_512 | 25897.2 | 59702.3 | 85599.5 |
| Windows 10-Core PC | ctx_8192 | 44011.7 | 44486.5 | 88498.3 |
| Windows 10-Core PC | few_shot_3 | 5862.4 | 18893.8 | 24756.2 |
| Windows 10-Core PC | long_cot | 3788.3 | 70331.2 | 74119.6 |
| Windows 10-Core PC | rag_top_1 | 8210.3 | 21589.5 | 29799.8 |
| Windows 10-Core PC | rag_top_3 | 17031.9 | 21093.0 | 38124.9 |
| Windows 10-Core PC | rag_top_5 | 20214.8 | 21990.7 | 42205.5 |
| Windows 10-Core PC | short_cot | 3506.1 | 41694.0 | 45200.2 |
| Windows 10-Core PC | zero_shot_cot | 3424.6 | 45721.2 | 49145.8 |
| Windows 10-Core PC | zero_shot_direct | 2363.1 | 18611.8 | 20974.9 |
