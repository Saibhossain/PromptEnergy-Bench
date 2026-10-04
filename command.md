# PromptEnergy-Bench: Master Experiment Command Guide

Comprehensive execution reference for running all benchmark experiments across **Windows**, **macOS (Apple Silicon)**, **Kaggle**, and **Google Colab** across all supported **models** and **datasets**.

---

## Table of Contents

1. [Hardware Presets & Supported Models](#1-hardware-presets--supported-models)
2. [Dataset Overview](#2-dataset-overview)
3. [Windows 10-Core CPU Commands (PowerShell & CMD)](#3-windows-10-core-cpu-commands)
4. [MacBook Air / Pro (Apple Silicon / MLX) Commands](#4-macbook-apple-silicon--mlx-commands)
5. [Kaggle GPU (T4 / Dual T4) Commands](#5-kaggle-gpu-commands)
6. [Google Colab GPU (T4 / A100) Commands](#6-google-colab-gpu-commands)
7. [Targeted Experiment Commands (Prompting, Context, RAG)](#7-targeted-experiment-commands)
8. [Quick Verification Commands (10-Sample Fast Check)](#8-quick-verification-commands)
9. [Automated Cross-Model Comparison Report](#9-automated-cross-model-comparison-report)

---

## 1. Hardware Presets & Supported Models

| Environment | `--hardware` Flag | Inference Engine | Supported Model Formats | Recommended Models |
| :--- | :--- | :--- | :--- | :--- |
| **Windows PC** | `windows_10core_pc` | Ollama / llama.cpp | GGUF | `qwen3.5:0.8b`, `qwen3.5:2b`, `llama3.2:1b`, `llama3.2:3b`, `gemma3:4b`, `llama3:8b`, `qwen3.5:9b` |
| **MacBook (M1/M2/M3)** | `macbook_air_m1` | MLX / Ollama | MLX / GGUF | `qwen3.5:0.8b-mlx`, `qwen3.5:2b-mlx`, `qwen3.5:9b-mlx`, `gemma4:e4b-mlx`, `llama3.2:1b` |
| **Kaggle GPU** | `kaggle_t4_gpu` | Ollama / Transformers / vLLM | GGUF / FP16 | `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen2.5:7b`, `qwen3.5:9b`, `gemma3:4b`, `llama3:8b` |
| **Google Colab** | `colab_t4_gpu` | Ollama / Transformers | GGUF / FP16 | `qwen3.5:0.8b`, `qwen3.5:2b`, `llama3.2:3b`, `gemma3:4b`, `llama3:8b` |

---

## 2. Dataset Overview

| Dataset Key | Task Family | Default Metric | Evaluation Split Size |
| :--- | :--- | :--- | :--- |
| `gsm8k` | Multi-step Mathematical Reasoning | Exact Match Accuracy & Numerical Error | 1,000 samples (from 1,319) |
| `natural_questions` | Knowledge-Intensive Open-Domain QA | Exact Match & Token F1 | 1,000 samples (from 100,231) |
| `contexteval` | Long-Context Document Reasoning | Exact Match & Token F1 | 1,000 samples (from 3,580) |
| `cnn_dailymail` | Abstractive Multi-Sentence Summarization | ROUGE-1, ROUGE-2, ROUGE-L | 1,000 samples (from 11,490) |
| `all` | Full Benchmark Suite | Evaluates all 4 datasets sequentially | 4,000 total evaluations |

---

## 3. Windows 10-Core CPU Commands

> **Note for Windows PowerShell**: Use the backtick ( **`` ` ``** ) for multi-line commands, or run the single-line version.

### A. GSM8K (Math Reasoning) — Full 1,000 Samples, All 3 Experiments

#### 1. Qwen 3.5 (2B)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:2b" `
    --dataset gsm8k `
    --eval-size 1k `
    --experiment all `
    --runs-per-condition 1
```

#### 2. Qwen 3.5 (0.8B)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:0.8b" `
    --dataset gsm8k `
    --eval-size 1k `
    --experiment all `
    --runs-per-condition 1
```

#### 3. LLaMA 3.2 (1B)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "llama3.2:1b" `
    --dataset gsm8k `
    --eval-size 1k `
    --experiment all `
    --runs-per-condition 1
```

#### 4. Gemma 3 (4B)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "gemma3:4b" `
    --dataset gsm8k `
    --eval-size 1k `
    --experiment all `
    --runs-per-condition 1
```

#### 5. LLaMA 3 (8B)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "llama3:8b" `
    --dataset gsm8k `
    --eval-size 1k `
    --experiment all `
    --runs-per-condition 1
```

#### 6. Qwen 3.5 (9B)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:9b" `
    --dataset gsm8k `
    --eval-size 1k `
    --experiment all `
    --runs-per-condition 1
```

---

### B. Other Datasets on Windows (Single-Line Commands)

#### Natural Questions (Factoid QA)
```powershell
python scripts/run_all_experiments.py --hardware windows_10core_pc --models "qwen3.5:2b" --dataset natural_questions --eval-size 1k --experiment all --runs-per-condition 1
```

#### ContextEval (Long-Context QA)
```powershell
python scripts/run_all_experiments.py --hardware windows_10core_pc --models "qwen3.5:2b" --dataset contexteval --eval-size 1k --experiment all --runs-per-condition 1
```

#### CNN / DailyMail (Summarization)
```powershell
python scripts/run_all_experiments.py --hardware windows_10core_pc --models "qwen3.5:2b" --dataset cnn_dailymail --eval-size 1k --experiment all --runs-per-condition 1
```

#### Full Matrix: All 4 Datasets Sequentially
```powershell
python scripts/run_all_experiments.py --hardware windows_10core_pc --models "qwen3.5:2b" --dataset all --eval-size 1k --experiment all --runs-per-condition 1
```

---

## 4. MacBook (Apple Silicon / MLX) Commands

> **Note for macOS (zsh/bash)**: Use the backslash ( **`\`** ) for multi-line commands.

### A. GSM8K (Math Reasoning) — Full 1,000 Samples, All 3 Experiments

#### 1. Both MLX Models Sequentially (Recommended)
```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:0.8b-mlx,qwen3.5:2b-mlx" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 2. Qwen 3.5 (0.8B MLX)
```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:0.8b-mlx" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 3. Qwen 3.5 (2B MLX)
```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:2b-mlx" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 4. Gemma 4 (4B MLX)
```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "gemma4:e4b-mlx" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 5. Qwen 3.5 (9B MLX)
```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:9b-mlx" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

---

### B. All Datasets on macOS

```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:2b-mlx" \
    --dataset all \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

---

## 5. Kaggle GPU Commands

> **Environment Setup on Kaggle**: Ensure GPU acceleration (T4 $\times 2$ or P100) is enabled in the notebook settings.

### A. Sequential Model Execution (Multi-Model Suite)
```bash
python scripts/run_all_experiments.py \
    --hardware kaggle_t4_gpu \
    --models "qwen3.5:0.8b,qwen3.5:2b,gemma3:4b,qwen2.5:7b,llama3:8b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

### B. Individual Model Runs on Kaggle

#### 1. Qwen 3.5 (2B)
```bash
python scripts/run_all_experiments.py \
    --hardware kaggle_t4_gpu \
    --models "qwen3.5:2b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 2. Qwen 2.5 / 3.5 (7B / 9B)
```bash
python scripts/run_all_experiments.py \
    --hardware kaggle_t4_gpu \
    --models "qwen2.5:7b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 3. LLaMA 3 (8B)
```bash
python scripts/run_all_experiments.py \
    --hardware kaggle_t4_gpu \
    --models "llama3:8b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

#### 4. Full Benchmark Matrix on Kaggle (All 4 Datasets)
```bash
python scripts/run_all_experiments.py \
    --hardware kaggle_t4_gpu \
    --models "qwen3.5:2b" \
    --dataset all \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

---

## 6. Google Colab GPU Commands

> **Colab Setup**: Add `!` at the beginning if running inside a notebook cell.

### A. GSM8K Evaluation on Colab T4 GPU
```bash
!python scripts/run_all_experiments.py \
    --hardware colab_t4_gpu \
    --models "qwen3.5:2b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

### B. Multi-Model Evaluation on Colab
```bash
!python scripts/run_all_experiments.py \
    --hardware colab_t4_gpu \
    --models "qwen3.5:0.8b,qwen3.5:2b,gemma3:4b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment all \
    --runs-per-condition 1
```

---

## 7. Targeted Experiment Commands

If you only want to execute one specific experiment type instead of all three:

### A. Experiment 1 Only: Prompting Strategy Comparison
*(Evaluates `zero_shot_direct`, `few_shot_3`, `zero_shot_cot`, `short_cot`, `long_cot`)*
```bash
python scripts/run_all_experiments.py \
    --hardware windows_10core_pc \
    --models "qwen3.5:2b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment primary \
    --runs-per-condition 1
```

### B. Experiment 2 Only: Context-Length Scaling
*(Evaluates scaling across $0$, $512$, $1024$, $2048$, $4096$ tokens)*
```bash
python scripts/run_all_experiments.py \
    --hardware windows_10core_pc \
    --models "qwen3.5:2b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment context \
    --context-lengths 0 512 1024 2048 4096 \
    --runs-per-condition 1
```

### C. Experiment 3 Only: BM25 RAG Energy Decomposition
*(Evaluates retrieval vs. generation across baseline, top-1, top-3, top-5)*
```bash
python scripts/run_all_experiments.py \
    --hardware windows_10core_pc \
    --models "qwen3.5:2b" \
    --dataset gsm8k \
    --eval-size 1k \
    --experiment rag \
    --top-k-list 1 3 5 \
    --runs-per-condition 1
```

---

## 8. Quick Verification Commands (10-Sample Fast Check)

Use `--eval-size 10` to quickly test the entire pipeline in **under 1–2 minutes** before launching long runs:

### Windows PowerShell:
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:2b" `
    --dataset gsm8k `
    --eval-size 10 `
    --experiment all `
    --runs-per-condition 1
```

### macOS / Linux:
```bash
python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:2b-mlx" \
    --dataset gsm8k \
    --eval-size 10 \
    --experiment all \
    --runs-per-condition 1
```

---

## 9. Automated Cross-Model Comparison Report

When all model experiments finish, you can generate a master cross-hardware and cross-model comparison report:

```bash
python scripts/compare_hardware_results.py \
    --results-dir results \
    --output-dir results/master_comparison
```

This compiles:
- Cross-model Pareto frontier plots
- Hardware energy-per-token efficiency rankings
- Comprehensive multi-model Markdown and LaTeX tables
