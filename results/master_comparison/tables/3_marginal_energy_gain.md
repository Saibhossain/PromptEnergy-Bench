# Marginal Energy Gain (MEG) across Prompting Strategies

| Model | Baseline | Strategy | Delta Acc (% pts) | Delta Energy (J) | MEG (% / J) |
| --- | --- | --- | --- | --- | --- |
| qwen3.5:9b | zero_shot_direct | few_shot_3 | -16.00% | +1743.581 J | -0.009 %/J |
| qwen3.5:9b | zero_shot_direct | zero_shot_cot | -2.00% | +2811.916 J | -0.001 %/J |
| qwen3.5:9b | zero_shot_direct | short_cot | -2.00% | -502.174 J | 0.000 %/J |
| qwen3.5:9b | zero_shot_direct | long_cot | +0.00% | +5089.700 J | 0.000 %/J |
| qwen3.5:9b | zero_shot_direct | ctx_0 | -2.00% | +2970.205 J | -0.001 %/J |
| qwen3.5:9b | zero_shot_direct | ctx_512 | +2.00% | +4024.979 J | 0.000 %/J |
| qwen3.5:9b | zero_shot_direct | ctx_1024 | +2.00% | +5229.922 J | 0.000 %/J |
| qwen3.5:9b | zero_shot_direct | ctx_2048 | +2.00% | +8588.873 J | 0.000 %/J |
| qwen3.5:9b | zero_shot_direct | ctx_4096 | -4.00% | +15210.416 J | -0.000 %/J |
| qwen3.5:9b | zero_shot_direct | rag_top_1 | -4.00% | +612.921 J | -0.007 %/J |
| qwen3.5:9b | zero_shot_direct | rag_top_3 | -4.00% | +1791.510 J | -0.002 %/J |
| qwen3.5:9b | zero_shot_direct | rag_top_5 | -6.00% | +3034.590 J | -0.002 %/J |
