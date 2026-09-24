# PromptEnergy-Bench: Dataset Architecture, Folder Analysis & Research Task Mapping

This document provides a comprehensive technical analysis of the `datasets/` repository folder, detailing the data assets, schemas, sample distributions, and the research mapping to each experimental task in **PromptEnergy-Bench** as formulated in [`updated_paper_idea.md`](file:///Users/mdsaibhossain/code/Research_paper/PromptEnergy-Bench/updated_paper_idea.md) and [`research_paper.md`](file:///Users/mdsaibhossain/code/Research_paper/PromptEnergy-Bench/research_paper.md).

---

## 1. Executive Summary & Experimental Scope

PromptEnergy-Bench systematically evaluates the **energy-accuracy-latency trade-offs** of prompting and inference strategies for Large Language Models across heterogeneous edge and client hardware (Apple Silicon M-Series SoC, x86-64 Intel/AMD CPUs, and NVIDIA GPUs).

To isolate and decompose computational costs—specifically **prefill phase energy ($E_{\text{prefill}}$)**, **autoregressive decoding energy ($E_{\text{decode}}$)**, **retrieval overhead ($E_{\text{retrieval}}$)**, and **reasoning token expansion ($E_{\text{reasoning}}$)**—the benchmark spans **four core task families**:

```
                              ┌───────────────────────────────────────────────────────────┐
                              │            PromptEnergy-Bench Task Families               │
                              └─────────────────────────────┬─────────────────────────────┘
                                                            │
         ┌──────────────────────────┬───────────────────────┴───────────────────────┬──────────────────────────┐
         ▼                          ▼                                               ▼                          ▼
  [Task 1: Math]             [Task 2: RAG QA]                              [Task 3: Long Context]        [Task 4: Summarization]
      GSM8K                 Natural Questions                                    ContextEval                 CNN/DailyMail
(Reasoning Scaling &      (Knowledge Retrieval &                              (Context Scaling &          (Generation-Heavy &
  CoT Token Growth)       Energy Decomposition)                                Prefill Energy)            Autoregressive Decode)
```

---

## 2. `datasets/` Folder Analysis Report

### 2.1 File System Structure & Disk Footprint

The `datasets/` directory contains both **HuggingFace Arrow format caches** (for fast zero-copy memory mapping) and **line-delimited JSONL files** (for independent, streaming, and reproducible evaluation).

```
datasets/
├── gsm8k/
│   ├── test.jsonl                       (0.71 MB  | 1,319 examples)
│   └── train.jsonl                      (3.96 MB  | 7,473 examples)
├── natural_questions/
│   ├── hf_arrow/                        (Arrow cache)
│   └── train.jsonl                      (66.34 MB | 100,231 examples)
├── contexteval/
│   ├── hf_arrow/                        (Arrow cache)
│   └── test.jsonl                       (0.65 MB  | 3,580 examples)
└── cnn_dailymail/
    ├── hf_arrow/                        (Arrow cache)
    ├── test.jsonl                       (48.28 MB | 11,490 examples)
    ├── validation.jsonl                 (55.84 MB | 13,368 examples)
    └── train.jsonl                      (1,220.18 MB | 287,113 examples)
```

### 2.2 Dataset Inventory & Structural Breakdown

| Dataset Key | HuggingFace Hub Path | Split | Rows / Examples | Disk Size | Features / Columns | Primary Task Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gsm8k`** | `openai/gsm8k` (`main`) | `test` | 1,319 | 0.71 MB | `question`, `answer` | Task 1: Primary CoT Reasoning Benchmark |
| | | `train` | 7,473 | 3.96 MB | `question`, `answer` | Few-shot exemplars & BM25 Context Corpus |
| **`natural_questions`** | `sentence-transformers/natural-questions` (`pair`) | `train` | 100,231 | 66.34 MB | `query`, `answer` | Task 2: Open-Domain QA & RAG Decomposition |
| **`contexteval`** | `allenai/ContextEval` | `test` | 3,580 | 0.65 MB | `query`, `source`, `example_id` | Task 3: Real-World Long-Context Scaling |
| **`cnn_dailymail`** | `abisee/cnn_dailymail` (`3.0.0`) | `test` | 11,490 | 48.28 MB | `article`, `highlights`, `id` | Task 4: Long-form Generation & Output Scaling |
| | | `validation` | 13,368 | 55.84 MB | `article`, `highlights`, `id` | Hyperparameter & prompt template tuning |
| | | `train` | 287,113 | 1,220.18 MB | `article`, `highlights`, `id` | Extended context corpus & few-shot demos |

---

## 3. Task-by-Task Research Mapping

---

### Task 1: Mathematical Reasoning & CoT Budget Scaling

* **Canonical Dataset**: **GSM8K (Grade School Math 8K)**
* **HuggingFace Path**: `openai/gsm8k` (config: `main`)
* **Local Location**: `datasets/gsm8k/`

#### 1. Research Purpose & Hypotheses
Mathematical problem solving requires multi-step deductive reasoning. It serves as the primary testbed for **Hypothesis H1 (Reasoning vs. Energy Trade-off)**:
- Adding Chain-of-Thought (CoT) tokens improves task accuracy $A$, but increases generation latency $T_{\text{decode}}$ and total energy consumption $E_{\text{total}}$ proportionally to the number of reasoning tokens generated.
- Allows measuring the **Marginal Energy Gain (MEG)**:
  $$\text{MEG} = \frac{A(r_2) - A(r_1)}{E(r_2) - E(r_1)} \quad \left[\frac{\Delta \text{ Accuracy}}{\text{Joule}}\right]$$

#### 2. Experimental Prompting Strategies Applied
1. **Zero-Shot Direct**: Answers directly with single numeric value; minimal output tokens.
2. **Few-Shot (3-shot)**: Standard 3 in-context demonstration examples.
3. **Zero-Shot CoT**: Prompted with *"Let's think step by step"*.
4. **Short CoT**: Constrained reasoning with concise derivations ($\le 256$ tokens).
5. **Long CoT / Extended Thinking**: Deep reasoning with full derivations ($\le 1024$ tokens).

#### 3. Evaluation Metrics & Parsing
* **Strict Numeric Extraction**: Extracts value following `####` or last numeric token.
* **Accuracy Metrics**: Total Accuracy, Exact Match, Valid-Answer Accuracy.
* **Truncation Handling**: Any generation truncated by `max_tokens` (`generation_truncated=True`) is strictly penalized as `answer_correct=False` and `answer_parse_success=False`.

---

### Task 2: Knowledge-Intensive QA & RAG Energy Decomposition

* **Canonical Dataset**: **Natural Questions (NQ)**
* **HuggingFace Path**: `sentence-transformers/natural-questions` (config: `pair`)
* **Local Location**: `datasets/natural_questions/`

#### 1. Research Purpose & Hypotheses
Tests **Hypothesis H3 (RAG vs. In-Context Parametric Trade-off)**:
- Retrieval-Augmented Generation offloads parametric memory requirements by injecting relevant document chunks into the prompt context.
- Decomposes the end-to-end RAG pipeline energy ($E_{\text{RAG}}$) into discrete sub-components:
  $$E_{\text{RAG}} = E_{\text{embedding}} + E_{\text{retrieval}} + E_{\text{prefill}} + E_{\text{decode}}$$
- Evaluates whether RAG achieves higher accuracy per Joule than unaugmented extended reasoning.

#### 2. Experimental Configurations
1. **Zero-Shot Direct (No Context)**: Baseline parametric generation.
2. **BM25 Retrieval ($k=1, 3, 5$)**: Sparse keyword retrieval over the background corpus.
3. **Dense Embedding Retrieval ($k=1, 3, 5$)**: Dense vector retrieval using bi-encoder representations.
4. **Grounding & Faithfulness**: Verifies whether model output is factual and grounded in retrieved passages.

#### 3. Evaluation Metrics & Parsing
* **Token-Level F1 & Exact Match**: SQuAD-style normalized string matching (article removal, whitespace normalization, lowercase).
* **Retrieval Metrics**: Recall@k, Precision@k, Context Utilization Overlap Score.
* **Energy Accounting**: Separated retrieval overhead ($E_{\text{retrieval}}$) vs generation energy ($E_{\text{generation}}$).

---

### Task 3: Long-Context QA & Prefill Scaling ($O(N)$ / $O(N^2)$ Effects)

* **Canonical Datasets**: **AllenAI ContextEval** (`allenai/ContextEval`) + **Controlled GSM8K/NQ Corpus Builder**
* **Local Location**: `datasets/contexteval/` & `src/data/context_builder.py`

#### 1. Research Purpose & Hypotheses
Tests **Hypothesis H2 (Context Scaling & Attention Overhead)**:
- As input context length $N$ scales ($0 \to 512 \to 1024 \to 2048 \to 4096 \to 8192$ tokens), Time-To-First-Token ($\text{TTFT}$) and prefill energy $E_{\text{prefill}}$ increase with super-linear scaling on long sequences due to quadratic attention computations ($O(N^2)$).
- Measures the impact on hardware memory bandwidth saturation and DRAM/Unified Memory power draw.

#### 2. Experimental Configurations
- **Evaluated Context Lengths**: $[0, 512, 1024, 2048, 4096, 8192]$ tokens.
- **Context Builder Modes**:
  - `relevant`: Synthetic and real domain-relevant background passages.
  - `distractor`: Semantic distractors to test context resilience and retrieval degradation.
  - `contexteval_queries`: Real queries with heterogeneous contextual requirements.

#### 3. Evaluation Metrics & Hardware Diagnostics
* **Time-To-First-Token (TTFT)**: High-resolution timestamp of initial token output ($\text{ms}$).
* **Prefill Energy ($E_{\text{prefill}}$)**: Energy consumed during prompt processing prior to first output token.
* **Prefill Throughput**: $\text{Input Tokens} / \text{TTFT}$ (tokens/sec).

---

### Task 4: Abstractive Summarization & Autoregressive Decoding Scaling

* **Canonical Dataset**: **CNN / DailyMail**
* **HuggingFace Path**: `abisee/cnn_dailymail` (config: `3.0.0`)
* **Local Location**: `datasets/cnn_dailymail/`

#### 1. Research Purpose & Hypotheses
Tests **Hypothesis H4 (Decode-Heavy vs. Prefill-Heavy Energy Regimes)**:
- Summarization exhibits long input prompts (news articles: $\approx 500\text{--}1500$ tokens) combined with substantial output generation (highlights: $\approx 50\text{--}150$ tokens).
- Isolates memory-bandwidth-bound autoregressive decoding from compute-bound matrix multiplication prefill.
- Evaluates energy efficiency per output token across quantized formats (MLX 4-bit, GGUF Q4_K_M, FP16).

#### 2. Experimental Configurations
1. **Zero-Shot Direct Summarization**: Standard headline/bullet-point generation prompt.
2. **Zero-Shot CoT Summarization**: Extract salient entities/events before synthesizing highlights.
3. **Budget Scaling**: Evaluating output length constraints ($\text{max\_tokens} \in [64, 128, 256, 512]$).

#### 3. Evaluation Metrics
* **N-gram Overlap**: ROUGE-1, ROUGE-2, and ROUGE-L (Longest Common Subsequence).
* **Generation Diversity & Repetition**: Distinct-1, Distinct-2 n-gram ratios.
* **Decode Energy Efficiency**: Joules per generated token ($E_{\text{decode}} / \text{Tokens}_{\text{output}}$).

---

## 4. Summary Cross-Task Matrix

| Feature | Task 1: Math (GSM8K) | Task 2: QA & RAG (NQ) | Task 3: Context Scaling (ContextEval) | Task 4: Summarization (CNN/DM) |
| :--- | :--- | :--- | :--- | :--- |
| **Dominant Energy Phase** | $E_{\text{decode}}$ (Reasoning tokens) | $E_{\text{retrieval}} + E_{\text{prefill}}$ | $E_{\text{prefill}}$ (Context processing) | $E_{\text{prefill}} + E_{\text{decode}}$ (Balanced) |
| **Typical Input Tokens** | 50 – 150 tokens | 200 – 1,000 tokens | 512 – 8,192 tokens | 500 – 1,500 tokens |
| **Typical Output Tokens** | 20 – 500 tokens | 10 – 50 tokens | 20 – 100 tokens | 50 – 200 tokens |
| **Core Research Metric** | Accuracy, MEG ($\Delta A / \Delta \text{J}$) | F1, $E_{\text{retrieval}} / E_{\text{total}}$ | TTFT ($\text{ms}$), $E_{\text{prefill}}$ (J) | ROUGE-1/2/L, J / Output Token |
| **Hardware Bottleneck** | Memory Bandwidth (Decode) | Vector Indexing + Prefill | Compute ($O(N^2)$ Attention) | Memory Bandwidth (KV-Cache) |

---

## 5. Dataset Ingestion & Experiment Execution

### 5.1 Dataset Download Command
To ensure all datasets are present and verified in `datasets/`:
```bash
./.venv/bin/python scripts/download_datasets.py --datasets all
```

### 5.2 Running Experiments Across Tasks
- **Master Suite (Runs all experiments)**:
  ```bash
  ./.venv/bin/python scripts/run_all_experiments.py \
      --hardware "MacBook Air M1" \
      --operator ollama \
      --models "qwen3.5:0.8b-mlx" \
      --eval-size 20 \
      --experiment all
  ```
- **Experiment 1 (Prompting Strategies on GSM8K)**:
  ```bash
  ./.venv/bin/python experiments/primary_exp_gsm8k.py --eval-size 20 --non-interactive
  ```
- **Experiment 2 (Context Scaling 0 to 8K)**:
  ```bash
  ./.venv/bin/python experiments/context_scaling_gsm8k.py --context-lengths 0 512 1024 2048 4096 --eval-size 20 --non-interactive
  ```
- **Experiment 3 (BM25 RAG Pipeline)**:
  ```bash
  ./.venv/bin/python experiments/rag_gsm8k.py --top-k-list 1 3 5 --eval-size 20 --non-interactive
  ```
