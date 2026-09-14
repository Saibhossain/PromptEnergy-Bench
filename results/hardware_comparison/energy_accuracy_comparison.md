# Cross-Hardware Energy-Accuracy Trade-off Matrix

| Hardware | Model | Prompt Strategy | Accuracy (%) | Mean Energy (J) | Energy / Correct (J) | Accuracy / Joule |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MacBook Air M1 | qwen3.5:0.8b-mlx | few_shot_3 | 1.4 | nan | nan | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | long_cot | 1.4 | 182.60 | 12964.48 | 0.000077 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | short_cot | 1.4 | 177.85 | 12716.24 | 0.000079 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_cot | 1.4 | nan | nan | 0.000000 |
| MacBook Air M1 | qwen3.5:0.8b-mlx | zero_shot_direct | 2.9 | nan | nan | 0.000000 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_0 | 77.6 | 7062.85 | 9107.36 | 0.000110 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_1024 | 78.8 | 11269.05 | 14292.46 | 0.000070 |
| Windows 10-Core PC | gemma3:4b | ctx_2048 | 76.0 | 18199.80 | 23947.10 | 0.000042 |
| Windows 10-Core PC | gemma3:4b | ctx_4096 | 76.0 | 17046.64 | 22429.79 | 0.000045 |
| Windows 10-Core PC | qwen3.5:0.8b | ctx_512 | 71.2 | 9244.47 | 12992.23 | 0.000077 |
| Windows 10-Core PC | qwen3.5:0.8b | few_shot_3 | 8.7 | 2008.02 | 22980.65 | 0.000044 |
| Windows 10-Core PC | qwen3.5:0.8b | long_cot | 42.7 | 3464.61 | 8110.34 | 0.000123 |
| Windows 10-Core PC | qwen3.5:0.8b | rag_top_1 | 0.0 | 1182.21 | N/A | 0.000000 |
| Windows 10-Core PC | qwen3.5:0.8b | short_cot | 37.9 | 3110.39 | 8214.62 | 0.000122 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_cot | 39.8 | 4120.48 | 10351.45 | 0.000097 |
| Windows 10-Core PC | qwen3.5:0.8b | zero_shot_direct | 3.9 | 1324.60 | 34108.36 | 0.000029 |
