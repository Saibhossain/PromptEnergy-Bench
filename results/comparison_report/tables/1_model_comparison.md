# Cross-Model Comparative Performance and Energy Decomposition Summary

| Model | Hardware | Samples | Accuracy (%) | Total Energy (J) | Prefill Energy (J) | Decode Energy (J) | Prefill (mJ/tok) | Decode (mJ/tok) | Decode TPS (tok/s) | Mean TTFT (ms) | Mean Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5:0.8b-mlx | macbook_air_m1, unknown | 6854 | 0.95% | 73.407 | 3.744 | 69.664 | 15.35 | 140.46 | 33.42 | 844.1 | 16794.3 |
| qwen3.5:0.8b | unknown, windows_10core_pc | 700 | 0.00% | 3801.296 | 363.095 | 3438.202 | 1453.02 | 7967.23 | 11.64 | 3763.9 | 40667.6 |
| gemma3:4b | unknown | 497 | 65.59% | 7256.037 | 3688.993 | 3567.044 | 5075.38 | 16911.87 | 6.06 | 37881.3 | 78172.5 |
