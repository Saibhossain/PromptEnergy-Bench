# Marginal Energy Gain (MEG) across Prompting Strategies

| Model | Baseline | Strategy | Delta Acc (% pts) | Delta Energy (J) | MEG (% / J) |
| --- | --- | --- | --- | --- | --- |
| qwen3.5:0.8b-mlx | zero_shot_direct | few_shot_3 | +30.00% | +28.837 J | 1.040 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | zero_shot_cot | +34.00% | +48.836 J | 0.696 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | short_cot | +32.00% | +11.127 J | 2.876 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | long_cot | +38.00% | +110.771 J | 0.343 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_0 | +38.00% | +65.454 J | 0.581 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_512 | +38.00% | +44.886 J | 0.847 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_1024 | +34.00% | +47.603 J | 0.714 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_2048 | +22.00% | +100.548 J | 0.219 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | ctx_4096 | +30.00% | +93.322 J | 0.321 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_1 | +0.00% | -2.466 J | 0.000 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_3 | +2.00% | +1.386 J | 1.443 %/J |
| qwen3.5:0.8b-mlx | zero_shot_direct | rag_top_5 | +2.00% | +2.919 J | 0.685 %/J |
