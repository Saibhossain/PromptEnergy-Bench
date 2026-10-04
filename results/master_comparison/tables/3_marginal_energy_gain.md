# Marginal Energy Gain (MEG) across Prompting Strategies

| Model | Baseline | Strategy | Delta Acc (% pts) | Delta Energy (J) | MEG (% / J) |
| --- | --- | --- | --- | --- | --- |
| qwen3.5:2b-mlx | zero_shot_direct | few_shot_3 | +3.00% | +35.272 J | 0.085 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | zero_shot_cot | +5.00% | +129.009 J | 0.039 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | short_cot | +1.00% | -1.744 J | 0.000 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | long_cot | +17.00% | +219.268 J | 0.078 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | ctx_0 | +11.00% | +122.141 J | 0.090 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | ctx_512 | +3.00% | +139.173 J | 0.022 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | ctx_1024 | +11.00% | +167.965 J | 0.065 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | ctx_2048 | +7.00% | +181.417 J | 0.039 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | ctx_4096 | +5.00% | +228.039 J | 0.022 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | rag_top_1 | -15.00% | +3.120 J | -4.808 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | rag_top_3 | -7.00% | +22.515 J | -0.311 %/J |
| qwen3.5:2b-mlx | zero_shot_direct | rag_top_5 | -11.00% | +31.322 J | -0.351 %/J |
