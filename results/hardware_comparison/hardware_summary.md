# Cross-Hardware Experimental Summary

| Hardware | Model | Prompt Strategy | Accuracy (%) | Mean Energy (J) | Median Energy (J) | Energy SD | Mean TTFT (ms) | Mean Latency (ms) | Input Tokens | Output Tokens | Measurement Method | Runs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_0 | 0.0 | 33.36 | 32.82 | 2.95 | 373.8 | 22096.3 | 143 | 512 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_1024 | 0.0 | 98.84 | 81.39 | 40.47 | 6725.0 | 60613.6 | 1216 | 512 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_2048 | 0.0 | 88.92 | 84.95 | 24.13 | 10022.5 | 49544.0 | 2382 | 512 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_4096 | 0.0 | 92.87 | 86.04 | 21.29 | 14902.1 | 41470.7 | 4644 | 512 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_512 | 0.0 | 45.12 | 45.75 | 7.10 | 2534.7 | 27989.6 | 588 | 512 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_8192 | 0.0 | 152.87 | 143.88 | 39.48 | 30544.4 | 58059.3 | 9405 | 512 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | few_shot_3 | 1.1 | nan | nan | nan | 1566.1 | 11820.4 | 311 | 232 | apple_estimated | 186 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | long_cot | 1.1 | 159.45 | 83.01 | 152.44 | 1321.7 | 31678.3 | 132 | 768 | apple_estimated | 182 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | rag_top_1 | 0.0 | 28.51 | 27.96 | 2.66 | 1066.0 | 9386.8 | 386 | 256 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | rag_top_3 | 0.0 | 31.25 | 31.04 | 1.57 | 2098.9 | 10205.0 | 869 | 256 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | rag_top_5 | 0.0 | 35.77 | 35.43 | 2.48 | 2821.9 | 10969.3 | 1330 | 256 | apple_powermetrics | 40 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | short_cot | 1.1 | 147.36 | 54.15 | 727.07 | 2454.9 | 19144.8 | 136 | 406 | apple_estimated | 183 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_cot | 1.1 | nan | nan | nan | 1056.0 | 19906.2 | 129 | 412 | apple_estimated | 185 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_direct | 2.0 | nan | nan | nan | 5145.1 | 14400.3 | 299 | 263 | apple_estimated | 251 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_0 | 63.3 | 6595.98 | 5887.26 | 2639.23 | 6714.1 | 67151.4 | 142 | 352 | codecarbon_estimated | 60 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_1024 | 65.1 | 10507.02 | 10220.35 | 3017.43 | 53635.6 | 103958.3 | 1191 | 276 | codecarbon_estimated | 63 |
| Windows 10-Core PC | gemma3:4b | ctx_2048 | 62.3 | 16435.11 | 16914.10 | 4656.72 | 111397.6 | 161542.7 | 2379 | 271 | codecarbon_estimated | 61 |
| Windows 10-Core PC | gemma3:4b | ctx_4096 | 62.3 | 15606.19 | 15131.25 | 4346.08 | 100764.8 | 152885.2 | 2051 | 273 | codecarbon_estimated | 61 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_512 | 58.7 | 8620.76 | 8325.87 | 3309.87 | 25897.2 | 85599.5 | 601 | 330 | codecarbon_estimated | 63 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_8192 | 0.0 | 8958.46 | 8998.55 | 69.54 | 44011.7 | 88498.3 | 2050 | 512 | codecarbon_estimated | 10 |
| Windows 10-Core PC | qwen3.5:0.8b | few_shot_3 | 3.3 | 2262.75 | 2250.17 | 849.12 | 5862.4 | 24756.2 | 325 | 210 | codecarbon_estimated | 275 |
| Windows 10-Core PC | qwen3.5:0.8b | long_cot | 26.7 | 5690.18 | 3362.03 | 3181.50 | 3788.3 | 74119.6 | 146 | 833 | codecarbon_estimated | 165 |
| Windows 10-Core PC | qwen3.5:0.8b | rag_top_1 | 0.0 | 2684.81 | 2807.66 | 487.03 | 8210.3 | 29799.8 | 392 | 256 | codecarbon_estimated | 11 |
| Windows 10-Core PC | qwen3.5:0.8b | rag_top_3 | 0.0 | 3709.90 | 3671.85 | 268.06 | 17031.9 | 38124.9 | 814 | 256 | codecarbon_estimated | 10 |
| Windows 10-Core PC | qwen3.5:0.8b | rag_top_5 | 0.0 | 4125.60 | 4198.22 | 157.06 | 20214.8 | 42205.5 | 1241 | 256 | codecarbon_estimated | 10 |
| Windows 10-Core PC | qwen3.5:0.8b | short_cot | 14.2 | 4299.37 | 4819.40 | 1593.86 | 3506.1 | 45200.2 | 145 | 442 | codecarbon_estimated | 275 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_cot | 14.9 | 4518.38 | 4784.94 | 1037.95 | 3424.6 | 49145.8 | 142 | 473 | codecarbon_estimated | 275 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_direct | 1.4 | 1743.56 | 2080.17 | 861.84 | 2363.1 | 20974.9 | 133 | 213 | codecarbon_estimated | 290 |
