# Marginal Energy Gain (MEG) across Prompting Strategies

| Model | Baseline | Strategy | Delta Acc (% pts) | Delta Energy (J) | MEG (% / J) |
| --- | --- | --- | --- | --- | --- |
| qwen3.5:0.8b-mlx | zero_shot_direct | few_shot_3 | +0.00% | +21.021 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | zero_shot_cot | +0.00% | +55.242 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | short_cot | +0.00% | +52.242 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | long_cot | +0.00% | +110.259 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_0 | +0.00% | +44.762 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_512 | +0.00% | +129.861 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_1024 | +0.00% | +38.307 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_2048 | +0.00% | +70.212 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_4096 | +0.00% | +82.505 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_1 | +0.00% | -6.351 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_3 | +0.00% | -3.139 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_5 | +0.00% | -1.944 J | 0.000 %/J |
