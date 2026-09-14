# Cross-Hardware Experimental Summary

| Hardware | Model | Prompt Strategy | Accuracy (%) | Mean Energy (J) | Median Energy (J) | Energy SD | Mean TTFT (ms) | Mean Latency (ms) | Input Tokens | Output Tokens | Measurement Method | Runs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MacBook Air M1 | qwen3.5:0.8b-mlx | few_shot_3 | 1.4 | nan | nan | nan | 1618.9 | 10472.9 | 306 | 225 | apple_estimated | 146 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | long_cot | 1.4 | 182.60 | 111.23 | 165.33 | 1537.7 | 26706.2 | 128 | 696 | apple_estimated | 142 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | short_cot | 1.4 | 177.85 | 70.29 | 819.90 | 2935.8 | 16603.5 | 133 | 376 | apple_estimated | 143 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_cot | 1.4 | nan | nan | nan | 1039.1 | 15604.8 | 126 | 385 | apple_estimated | 145 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_direct | 2.9 | nan | nan | nan | 7368.7 | 17292.5 | 376 | 267 | apple_estimated | 171 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_0 | 77.6 | 7062.85 | 6809.93 | 2700.71 | 7705.6 | 71672.5 | 142 | 316 | codecarbon_estimated | 49 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_1024 | 78.8 | 11269.05 | 10852.90 | 2771.03 | 59978.5 | 111302.6 | 1177 | 226 | codecarbon_estimated | 52 |
| Windows 10-Core PC | gemma3:4b | ctx_2048 | 76.0 | 18199.80 | 17360.73 | 3022.90 | 127782.3 | 178591.0 | 2363 | 218 | codecarbon_estimated | 50 |
| Windows 10-Core PC | gemma3:4b | ctx_4096 | 76.0 | 17046.64 | 16393.11 | 3395.32 | 113368.7 | 166693.3 | 2051 | 221 | codecarbon_estimated | 50 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_512 | 71.2 | 9244.47 | 8569.65 | 3317.96 | 28896.7 | 91557.1 | 606 | 292 | codecarbon_estimated | 52 |
| Windows 10-Core PC | qwen3.5:0.8b | few_shot_3 | 8.7 | 2008.02 | 1027.05 | 1244.59 | 9877.3 | 23143.7 | 324 | 132 | codecarbon_estimated | 103 |
| Windows 10-Core PC | qwen3.5:0.8b | long_cot | 42.7 | 3464.61 | 3038.72 | 1697.15 | 4151.7 | 62404.3 | 146 | 718 | codecarbon_estimated | 103 |
| Windows 10-Core PC | qwen3.5:0.8b | rag_top_1 | 0.0 | 1182.21 | 1182.21 | 0.00 | 7542.1 | 34191.5 | 356 | 256 | codecarbon_estimated | 1 |
| Windows 10-Core PC | qwen3.5:0.8b | short_cot | 37.9 | 3110.39 | 2091.10 | 1993.02 | 4510.2 | 36914.3 | 145 | 326 | codecarbon_estimated | 103 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_cot | 39.8 | 4120.48 | 4621.27 | 1550.84 | 4507.6 | 50742.3 | 142 | 409 | codecarbon_estimated | 103 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_direct | 3.9 | 1324.60 | 584.08 | 1206.67 | 4207.5 | 16916.0 | 133 | 135 | codecarbon_estimated | 103 |
