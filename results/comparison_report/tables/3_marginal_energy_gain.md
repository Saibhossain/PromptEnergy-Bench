# Marginal Energy Gain (MEG) across Prompting Strategies

| Model | Baseline | Strategy | Delta Acc (% pts) | Delta Energy (J) | MEG (% / J) |
| --- | --- | --- | --- | --- | --- |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_0 | -0.36% | -1.227 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_512 | -0.36% | +3.983 J | -0.091 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_1024 | -0.36% | +57.698 J | -0.006 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_2048 | -0.36% | +47.777 J | -0.008 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_4096 | -0.36% | +51.734 J | -0.007 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_8192 | -0.36% | +111.735 J | -0.003 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | few_shot_3 | -0.21% | -0.781 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | zero_shot_cot | -0.21% | +35.787 J | -0.006 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | short_cot | -0.21% | +39.415 J | -0.005 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | long_cot | +3.73% | +88.637 J | 0.042 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_1 | -0.36% | -12.626 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_3 | -0.36% | -9.885 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_5 | -0.36% | -5.370 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | ctx_0 | +0.00% | +1720.845 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | ctx_512 | +0.00% | +3495.561 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | ctx_1024 | +0.00% | +4950.836 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | ctx_2048 | +0.00% | +6185.020 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | ctx_4096 | +0.00% | +6904.819 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | ctx_8192 | +0.00% | +6762.985 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | few_shot_3 | +0.00% | +415.519 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | zero_shot_cot | +0.00% | +2806.238 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | short_cot | +0.00% | +2855.251 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | long_cot | +0.00% | +871.124 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | rag_top_1 | +0.00% | +329.123 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | rag_top_3 | +0.00% | +1514.423 J | 0.000 %/J |
| qwen3.5:0.8b | zero_shot_direct | rag_top_5 | +0.00% | +1930.127 J | 0.000 %/J |
| gemma3:4b | zero_shot_direct | ctx_0 | +72.85% | +6914.528 J | 0.011 %/J |
| gemma3:4b | zero_shot_direct | ctx_512 | +66.00% | +9072.138 J | 0.007 %/J |
| gemma3:4b | zero_shot_direct | ctx_1024 | +74.00% | +11097.384 J | 0.007 %/J |
| gemma3:4b | zero_shot_direct | ctx_2048 | +68.00% | +17883.298 J | 0.004 %/J |
| gemma3:4b | zero_shot_direct | ctx_4096 | +68.00% | +16730.140 J | 0.004 %/J |
| gemma3:4b | zero_shot_direct | few_shot_3 | +10.00% | +491.714 J | 0.020 %/J |
| gemma3:4b | zero_shot_direct | zero_shot_cot | +72.00% | +2545.398 J | 0.028 %/J |
| gemma3:4b | zero_shot_direct | short_cot | +68.00% | +1073.465 J | 0.063 %/J |
| gemma3:4b | zero_shot_direct | long_cot | +78.00% | +3585.816 J | 0.022 %/J |
