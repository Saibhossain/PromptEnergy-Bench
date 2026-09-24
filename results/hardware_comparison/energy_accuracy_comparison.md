# Cross-Hardware Energy-Accuracy Trade-off Matrix

| Hardware | Model | Prompt Strategy | Accuracy (%) | Mean Energy (J) | Energy / Correct (J) | Accuracy / Joule |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_0 | 0.0 | 33.36 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_1024 | 0.0 | 98.84 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_2048 | 0.0 | 88.92 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_4096 | 0.0 | 92.87 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_512 | 0.0 | 45.12 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | ctx_8192 | 0.0 | 152.87 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | few_shot_3 | 1.1 | nan | nan | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | long_cot | 1.1 | 159.45 | 14510.20 | 0.000069 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | rag_top_1 | 0.0 | 28.51 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | rag_top_3 | 0.0 | 31.25 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | rag_top_5 | 0.0 | 35.77 | N/A | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | short_cot | 1.1 | 147.36 | 13483.75 | 0.000074 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_cot | 1.1 | nan | nan | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_direct | 2.0 | nan | nan | 0.000000 |
| Windows 10-Core PC | gemma3:4b | ctx_0 | 76.0 | 7037.87 | 9260.36 | 0.000108 |
| Windows 10-Core PC | gemma3:4b | ctx_1024 | 78.8 | 11269.05 | 14292.46 | 0.000070 |
| Windows 10-Core PC | gemma3:4b | ctx_2048 | 76.0 | 18199.80 | 23947.10 | 0.000042 |
| Windows 10-Core PC | gemma3:4b | ctx_4096 | 76.0 | 17046.64 | 22429.79 | 0.000045 |
| Windows 10-Core PC | gemma3:4b | ctx_512 | 71.2 | 9244.47 | 12992.23 | 0.000077 |
| Windows 10-Core PC | qwen3.5:0.8b | few_shot_3 | 8.6 | 1992.12 | 23241.42 | 0.000043 |
| Windows 10-Core PC | qwen3.5:0.8b | long_cot | 41.9 | 3605.32 | 8603.61 | 0.000116 |
| Windows 10-Core PC | qwen3.5:0.8b | rag_top_1 | 0.0 | 1182.21 | N/A | 0.000000 |
| Windows 10-Core PC | qwen3.5:0.8b | short_cot | 37.1 | 3125.05 | 8413.59 | 0.000119 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_cot | 39.0 | 4108.71 | 10522.29 | 0.000095 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_direct | 3.8 | 1316.73 | 34564.08 | 0.000029 |
