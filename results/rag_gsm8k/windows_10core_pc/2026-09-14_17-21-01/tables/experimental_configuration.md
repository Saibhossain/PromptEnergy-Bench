# Experimental Configuration and Reproducibility Parameters

| Parameter                 | Value                                                                                            |
|:--------------------------|:-------------------------------------------------------------------------------------------------|
| Experiment                | rag_gsm8k                                                                                        |
| Dataset                   | GSM8K                                                                                            |
| Evaluation Split          | TEST                                                                                             |
| Model                     | qwen3.5:0.8b                                                                                     |
| Backend Operator          | ollama                                                                                           |
| Model Format              | gguf                                                                                             |
| Device                    | windows_10core_pc                                                                                |
| Operating System          | Windows                                                                                          |
| CPU                       | Intel64 Family 6 Model 85 Stepping 4, GenuineIntel                                               |
| GPU                       | None                                                                                             |
| System RAM                | 32 GB                                                                                            |
| Evaluation Examples       | 5                                                                                                |
| Prompt Strategies         | 4                                                                                                |
| Warmup Runs               | 1                                                                                                |
| Repetitions               | 1                                                                                                |
| Max Generation Tokens     | Zero-shot Direct: 256 / Few-shot (3): 256 / Zero-shot CoT: 512 / Short CoT: 512 / Long CoT: 1024 |
| Sampling Temperature      | 0.0                                                                                              |
| Sampling Seed             | 42                                                                                               |
| Energy Measurement Method | codecarbon_estimated                                                                             |
| Energy Quality / Level    | estimated (estimated_system)                                                                     |
