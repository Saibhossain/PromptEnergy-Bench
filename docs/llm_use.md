# PromptEnergy-Bench: Large Language Model (LLM) Specifications & Model Registry

This document provides a comprehensive technical reference for all **Large Language Models (LLMs)** and **Small Language Models (SLMs)** evaluated within **PromptEnergy-Bench**. It details release dates, parameter architectures, attention mechanisms, vocabulary sizes, quantization formats, and hardware deployment mappings for the research paper.

---

## 1. Executive Summary & Model Taxonomy

To systematically investigate how **model architecture, parameter scale, attention mechanisms, and reasoning paradigms** interact with **hardware energy efficiency ($E_{\text{total}}$, $E_{\text{prefill}}$, $E_{\text{decode}}$, $\text{J/token}$)**, PromptEnergy-Bench selects a representative, heterogeneous spectrum of open-weight causal decoder models spanning **0.5B to 12B parameters**, along with cloud API baselines.

```
                                  ┌─────────────────────────────────────────────────────────────┐
                                  │            PromptEnergy-Bench Model Families                │
                                  └──────────────────────────────┬──────────────────────────────┘
                                                                 │
         ┌───────────────────────────┬───────────────────────────┼───────────────────────────┬───────────────────────────┐
         ▼                           ▼                           ▼                           ▼                           ▼
  [Qwen 3.5 / 2.5]            [Llama 3.x / 3.2]          [Gemma 3 / 4]                [Mistral / R1]              [Cloud API]
Alibaba (0.5B-9B)             Meta (1B-8B)               Google (1B-12B)              DeepSeek / Mistral          OpenAI (Mini/Nano)
Dense / Thinking CoT         GQA / 128k Long-Context     GeGLU / Edge-Optimized       SWA / Distilled Reasoning   Closed API Baseline
```

---

## 2. Master Model Specifications Table (For Research Paper)

The following table provides the formal reference metadata for all benchmarked models in the experimental matrix:

| Model Identifier | Developer / Family | Release Date | Total Parameters | Active Parameters | Architecture Type | Attention Mechanism | Vocabulary Size | Native Context Window | Quantization / Formats | Native Thinking Mode |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **`qwen2.5:0.5b`** | Alibaba Cloud | Sep 2024 | 0.49 B | 0.49 B | Dense Causal Decoder | GQA (14 Q / 2 KV heads) | 151,936 | 32,768 | GGUF (Q4_K_M), FP16 | No |
| **`qwen3.5:0.8b`** / `0.8b-mlx` | Alibaba Cloud | Jan 2025 | 0.82 B | 0.82 B | Dense Causal Decoder | GQA (16 Q / 2 KV heads) | 151,936 | 8,192 (4k MLX) | GGUF (Q4_K_M), MLX 4-bit | Yes |
| **`gemma3:1b`** | Google DeepMind | Feb 2025 | 1.15 B | 1.15 B | Dense Causal Decoder | Multi-Query / GQA | 256,000 | 8,192 | GGUF (Q4_K_M), FP16 | No |
| **`llama3.2:1b`** | Meta AI | Sep 2024 | 1.23 B | 1.23 B | Dense Causal Decoder | GQA (32 Q / 8 KV heads) | 128,256 | 131,072 | GGUF (Q4_K_M), MLX 4-bit | No |
| **`qwen3.5:2b`** / `2b-mlx` | Alibaba Cloud | Jan 2025 | 2.14 B | 2.14 B | Dense Causal Decoder | GQA (24 Q / 4 KV heads) | 151,936 | 8,192 (4k MLX) | GGUF (Q4_K_M), MLX 4-bit | Yes |
| **`gemma4:e2b`** | Google DeepMind | May 2025 | 2.30 B | 2.30 B | Edge Causal Decoder | GQA (GeGLU) | 256,000 | 8,192 | GGUF (Q4_K_M), MLX 4-bit | No |
| **`llama3.2:3b`** | Meta AI | Sep 2024 | 3.21 B | 3.21 B | Dense Causal Decoder | GQA (24 Q / 8 KV heads) | 128,256 | 131,072 | GGUF (Q4_K_M), FP16 | No |
| **`gemma3:4b`** | Google DeepMind | Feb 2025 | 4.32 B | 4.32 B | Dense Causal Decoder | GQA (GeGLU activations) | 256,000 | 8,192 | GGUF (Q4_K_M), FP16 | No |
| **`gemma4:e4b-mlx`** | Google DeepMind | May 2025 | 4.40 B | 4.40 B | Edge Causal Decoder | GQA (GeGLU activations) | 256,000 | 8,192 | MLX 4-bit, GGUF | No |
| **`mistral:7b`** | Mistral AI | May 2024 | 7.24 B | 7.24 B | Dense Causal Decoder | Sliding Window + GQA | 32,768 | 32,768 | GGUF (Q4_K_M), FP16 | No |
| **`deepseek-r1:7b`** | DeepSeek AI | Jan 2025 | 7.61 B | 7.61 B | Distilled Reasoning | GQA (Qwen-derived) | 152,064 | 65,536 | GGUF (Q4_K_M), FP16 | Yes (`<think>`) |
| **`qwen2.5:7b`** | Alibaba Cloud | Sep 2024 | 7.61 B | 7.61 B | Dense Causal Decoder | GQA (28 Q / 4 KV heads) | 152,064 | 131,072 | GGUF (Q4_K_M), MLX 4-bit | No |
| **`llama3:8b`** | Meta AI | Apr 2024 | 8.03 B | 8.03 B | Dense Causal Decoder | GQA (32 Q / 8 KV heads) | 128,256 | 8,192 | GGUF (Q4_K_M), FP16 | No |
| **`llama3.1:8b`** | Meta AI | Jul 2024 | 8.03 B | 8.03 B | Dense Causal Decoder | GQA (32 Q / 8 KV heads) | 128,256 | 131,072 | GGUF (Q4_K_M), MLX 4-bit | No |
| **`qwen3.5:9b`** / `9b-mlx` | Alibaba Cloud | Jan 2025 | 9.12 B | 9.12 B | Dense Causal Decoder | GQA (32 Q / 4 KV heads) | 151,936 | 8,192 | GGUF (Q4_K_M), MLX 4-bit | Yes |
| **`gemma4:12b`** | Google DeepMind | May 2025 | 12.10 B | 12.10 B | Dense Causal Decoder | GQA (GeGLU activations) | 256,000 | 8,192 | GGUF (Q4_K_M), FP16 | No |

---

## 3. Detailed Architectural Family Profiles

### 3.1 Qwen 3.5 & Qwen 2.5 Series (Alibaba Cloud)
* **Design Philosophy**: High token efficiency with comprehensive multilingual pre-training and native step-by-step reasoning modes.
* **Architectural Highlights**:
  * **Rotary Position Embedding (RoPE)** with base frequency scaled up to $10^6$ for robust long-context handling.
  * **Grouped-Query Attention (GQA)** across all parameter scales (significantly reduces KV cache memory footprint during prefill and generation).
  * **Dual Reasoning Support**: Qwen 3.5 supports native `<think>...</think>` internal token emission for mathematical reasoning, or can operate with explicit prompt-driven CoT.
  * **Vocab Expansion**: Large 151.9k vocabulary minimizes byte-fallback tokens on mathematical equations and code syntax.

### 3.2 Llama 3, 3.1 & 3.2 Series (Meta AI)
* **Design Philosophy**: Standard-setting open-weights dense causal decoders trained on $>15$ trillion tokens.
* **Architectural Highlights**:
  * **Extended Context Window**: Llama 3.1 and 3.2 feature native 128k context support utilizing scaled RoPE frequencies.
  * **Optimized Grouped-Query Attention**: 8 KV heads paired with 24–32 Query heads, accelerating memory bandwidth throughput on edge devices.
  * **Tokenization**: 128,256 token vocabulary built on byte-level BPE, reducing average prompt sequence length by $\sim 15\%$ compared to Llama 2.
  * **Edge Specialization**: Llama 3.2 (1B and 3B) employs aggressive weight pruning and knowledge distillation for sub-4GB RAM client devices.

### 3.3 Gemma 3 & Gemma 4 Series (Google DeepMind)
* **Design Philosophy**: Lightweight, high-throughput models engineered from the same research foundation as Gemini.
* **Architectural Highlights**:
  * **GeGLU Activations**: Gated Gaussian Error Linear Units providing non-linear representation capacity in compact parameter counts.
  * **RMSNorm with Unit Offset**: Normalization applied both at input and post-attention layers for training stability at high learning rates.
  * **Massive Vocabulary (256,000 tokens)**: Direct subword compression optimized for multilingual reasoning, scientific terms, and structured markdown.

### 3.4 DeepSeek-R1 Distill Series (DeepSeek AI)
* **Design Philosophy**: Reinforcement Learning (RL) distilled reasoning architecture providing frontier-grade mathematical and logical reasoning within edge parameter budgets.
* **Architectural Highlights**:
  * **Reasoning Token Expansion**: Emits dynamic internal exploration tokens (`<think>`) before producing the final synthesized answer.
  * **Base Architecture**: Built upon Qwen 2.5/3.5 architectures with specialized reasoning policy weights distilled from DeepSeek-R1 671B.

### 3.5 Mistral 7B (Mistral AI)
* **Design Philosophy**: High-performance 7B workhorse leveraging sliding-window attention for linear compute scaling.
* **Architectural Highlights**:
  * **Sliding Window Attention (SWA)**: Theoretical attention span of 4,096 tokens per layer with receptive field expanding linearly across depth.
  * **Byte-fallback BPE**: 32,768 vocabulary ensuring compact token table allocation.

---

## 4. Hardware Deployment & Quantization Mapping

To preserve measurement validity, models are mapped to the most efficient native runtime operators per hardware platform:

| Hardware Platform | Architecture | Compute Engine | Model Formats | Active Models in Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **macOS (MacBook Air M1)** | ARM64 Apple Silicon | MLX / Ollama Metal | Apple MLX 4-bit, GGUF | `qwen3.5:0.8b-mlx`, `qwen3.5:2b-mlx`, `qwen3.5:9b-mlx`, `gemma4:e4b-mlx` |
| **Windows PC (10-Core CPU)** | x86_64 Intel/AMD | Ollama / llama.cpp | GGUF (`Q4_K_M`, `Q8_0`) | `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:9b`, `llama3.2:1b`, `llama3:8b`, `gemma3:4b`, `mistral:7b` |
| **Kaggle (NVIDIA T4 16GB)** | Turing Architecture | Ollama / PyTorch CUDA | GGUF (`Q4_K_M`), FP16 | Full suite (1B to 12B parameters + DeepSeek-R1) |
| **Google Colab (NVIDIA T4)** | Turing Architecture | Ollama / PyTorch CUDA | GGUF (`Q4_K_M`), FP16 | Full suite (1B to 12B parameters + DeepSeek-R1) |

---

## 5. Mathematical Complexity & KV-Cache Footprint

During autoregressive generation, memory bandwidth saturation is the primary driver of decoding energy consumption:

$$\text{KV Cache Size (Bytes)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times b_{\text{precision}}$$

Where:
* $n_{\text{layers}}$: Total transformer decoder layers.
* $n_{\text{kv\_heads}}$: Number of Key/Value attention heads (reduced via GQA).
* $d_{\text{head}}$: Head dimension ($d_{\text{model}} / n_{\text{query\_heads}}$).
* $L$: Active sequence context length (prompt tokens + generated tokens).
* $b_{\text{precision}}$: Bytes per element (e.g., 2 bytes for FP16, 0.5 bytes for 4-bit quantized cache).

### Key Empirical Implications for PromptEnergy-Bench:
1. **GQA Advantage**: Models using $n_{\text{kv\_heads}} \ll n_{\text{query\_heads}}$ (e.g., Qwen 3.5, Llama 3.1) exhibit significantly lower energy scaling slopes ($\Delta \text{Joules} / \Delta \text{Tokens}$) across Experiment 2 (Context Scaling).
2. **Reasoning Trade-off**: Thinking models (`deepseek-r1:7b`, `qwen3.5:9b`) generate $3\times\text{ to }8\times$ more output tokens, trading increased decode energy ($E_{\text{decode}}$) for higher task accuracy on GSM8K.
