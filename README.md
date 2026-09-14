# PromptEnergy-Bench: Energy–Accuracy–Latency Benchmark for LLM Prompting

A reproducible, device-portable benchmarking framework designed to empirically measure and optimize the Energy–Accuracy–Latency trade-offs induced by prompting strategies, controlled context scaling, and retrieval-augmented generation (RAG) across heterogeneous hardware platforms.

---

## 1. Project Purpose & Research Scope

While prompt engineering techniques such as Few-Shot prompting and Chain-of-Thought (CoT) reasoning often yield improvements in task accuracy, they systematically expand token generation budgets, inflight KV-cache occupancy, prefill compute, and decode latency. 

**PromptEnergy-Bench** quantifies the marginal energy cost of reasoning:
* **Marginal Energy Gain (MEG)**:
  $$\text{MEG}(S_1, S_2) = \frac{\text{Accuracy}(S_2) - \text{Accuracy}(S_1)}{\text{Energy}(S_2) - \text{Energy}(S_1)}$$
* **Accuracy per Joule ($APJ$)**: Normalized task accuracy achieved per unit of electrical energy consumed.
* **Energy per Correct Answer**: Total energy consumed divided by the number of correct responses.
* **Pareto Frontier Efficiency**: Identifying configurations that optimize the tripartite trade-off between Accuracy, Energy (Joules), and Latency (ms).

---

## 2. Dataset Structure & Leakage Protection

The benchmark evaluates on **GSM8K** (Grade School Math 8K):

```text
datasets/gsm8k/
├── test.jsonl     <-- STRICTLY EVALUATION BENCHMARK (1,319 samples)
├── train.jsonl    <-- SOLE SOURCE FOR FEW-SHOT EXAMPLES, CONTEXT SCALING & RAG CORPUS (7,473 samples)
└── dataset_dict.json
```

### Strict Train/Test Separation Rules
* `datasets/gsm8k/test.jsonl` is the sole evaluation dataset.
* `datasets/gsm8k/train.jsonl` provides few-shot examples, controlled context scaling blocks, and BM25 retrieval documents.
* A test question is never evaluated against its own answer or used as retrieved context (`exclude_id` enforcement).
* Selecting `--eval-size full` evaluates the entire GSM8K **TEST** split (1,319 problems). It never switches evaluation to the train split.

---

## 3. Supported Backends & Configured Models

The framework supports both local and cloud inference backends:

| Model Identifier | Backend | Format | Context Limit | Reasoning / `<think>` Support |
| :--- | :--- | :--- | :--- | :--- |
| `qwen3.5:0.8b` | Ollama | GGUF | 8,192 | Yes (deep-think `<think>` tags) |
| `qwen3.5:0.8b-mlx` | Ollama / MLX | MLX | 4,096 | Yes |
| `qwen3.5:2b-mlx` | Ollama / MLX | MLX | 4,096 | Yes |
| `qwen3.5:9b` | Ollama | GGUF | 8,192 | Yes |
| `gemma3:1b` | Ollama | GGUF | 8,192 | No |
| `gemma3:4b` | Ollama | GGUF | 8,192 | No |
| `mistral:7b` | Ollama | GGUF | 8,192 | No |
| `llama3.2:3b` | Ollama | GGUF | 8,192 | No |
| `gpt-4o-mini` | OpenAI API | Cloud API | 128,000 | No |
| `gpt-4o` | OpenAI API | Cloud API | 128,000 | No |

---

## 4. Hardware Profiles & Energy Telemetry

### Measurement Adapters
1. **Apple Silicon Hardware (`apple_powermetrics`)**: Direct SoC hardware counter telemetry via `/usr/bin/powermetrics` (requires passwordless sudo). Quality: `hardware_reported`, Level: `soc_package`.
2. **NVIDIA GPU NVML (`nvidia_nvml`)**: High-frequency GPU power polling via NVML (`nvidia-ml-py`) integrated over time ($E = \int P(t) dt$). Quality: `hardware_reported`, Level: `gpu_only`.
3. **Linux RAPL (`rapl`)**: Intel/AMD Running Average Power Limit CPU energy counters. Quality: `hardware_reported`, Level: `cpu_package`.
4. **CodeCarbon Software Estimator (`codecarbon_estimated`)**: Platform software emission and energy tracking. Quality: `software_estimate`, Level: `estimated_system`.
5. **Continuous Resource Monitor (`BackgroundResourceMonitor`)**: Samples host CPU utilization (%) and RAM (GB) at 50–100ms intervals throughout prompt prefill, decode, and retrieval.
6. **Cloud Environmental Estimator (`cloud_lca_estimate`)**: Software estimation for cloud APIs. **Cloud and physical energy measurements are never conflated.**

---

## 5. Directory Structure & Saved Artifacts

Every run produces an isolated, timestamped directory:

```text
results/
├── {experiment_name}/
│   └── {normalized_device_name}/
│       └── {experiment_timestamp}/
│           ├── metadata.json           # System, git, hardware, and package provenance
│           ├── config.json             # Exact experiment configuration
│           ├── results.jsonl           # Per-sample atomic inference records
│           ├── raw_results.jsonl       # Mirror of atomic records
│           ├── summary.json            # Aggregate metrics across conditions
│           ├── summary.csv             # Root condition summary
│           ├── tables/                 # Publication-ready tables (CSV, Markdown, LaTeX)
│           │   ├── strategy_comparison.csv/.md/.tex
│           │   ├── tradeoff_comparison.csv/.md/.tex
│           │   ├── marginal_energy_gain.csv/.md/.tex
│           │   └── statistical_summary.csv/.md/.tex
│           ├── plots/                  # Run-specific visual figures (.png, .pdf, .svg)
│           └── logs/
│               └── experiment.log      # Full execution transcript
├── analysis/                           # 8 Unrounded Research Summary CSVs (Section 11)
├── figures/                            # 12 Publication Figures (PNG 300 DPI & PDF)
└── hardware_comparison/                # 7 Cross-Hardware Tables & 10 Comparison Figures
```

---

## 6. Execution Instructions

### A. Run All Three Experiments via Master Runner

```powershell
# Small validation run (default 5 samples, 1 repetition)
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "gemma3:4b" `
    --eval-size 5 `
    --runs-per-condition 1

# Run across all local Ollama models
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:0.8b,gemma3:4b,mistral:7b,llama3.2:3b" `
    --eval-size 50 `
    --runs-per-condition 3

# Run on MacBook Air M1
python scripts/run_all_experiments.py `
    --hardware macbook_air_m1 `
    --models "qwen3.5:0.8b-mlx" `
    --eval-size 50 `
    --runs-per-condition 3
```

---

### B. Run Individual Experiments Independently

#### Experiment 1: Prompting Strategy Comparison
```powershell
python experiments/primary_exp_gsm8k.py `
    --device-name "windows_10core_pc" `
    --model "gemma3:4b" `
    --operator "ollama" `
    --eval-size 50 `
    --warmups 1 `
    --repetitions 3
```

#### Experiment 2: Controlled Context Scaling ($0 \dots 8192$ tokens)
```powershell
python experiments/context_scaling_gsm8k.py `
    --device-name "windows_10core_pc" `
    --model "gemma3:4b" `
    --operator "ollama" `
    --context-lengths 0 512 1024 2048 4096 8192 `
    --context-type "relevant" `
    --eval-size 50 `
    --warmups 1 `
    --repetitions 1
```

#### Experiment 3: Retrieval-Augmented Generation (BM25 RAG)
```powershell
python experiments/rag_gsm8k.py `
    --device-name "windows_10core_pc" `
    --model "gemma3:4b" `
    --operator "ollama" `
    --top-k-list 1 3 5 `
    --include-baseline `
    --eval-size 50 `
    --warmups 1 `
    --repetitions 1
```

---

### C. Generate Publication Figures (PNG 300 DPI & PDF)

```powershell
python scripts/generate_publication_figures.py `
    --input-dir results/ `
    --output-dir results/figures/
```

Generates all 12 publication-grade figures in `results/figures/`:
1. `energy_accuracy`
2. `energy_prompt_strategy`
3. `accuracy_prompt_strategy`
4. `pareto_frontier`
5. `context_energy`
6. `context_ttft`
7. `prefill_energy`
8. `decode_energy`
9. `latency_comparison`
10. `model_comparison`
11. `rag_energy_decomposition`
12. `energy_per_correct_answer`

---

### D. Compare Hardware Results & Generate Cross-Hardware Deliverables

```powershell
python scripts/compare_hardware_results.py `
    --input-dirs results/macbook_air_m1 results/windows_10core_pc `
    --output-dir results/hardware_comparison/
```

Generates 7 comparative tables (CSV, Markdown, LaTeX) and 10 cross-hardware figures in `results/hardware_comparison/`.

---

## 7. Running Tests & Quality Verification

Execute the complete automated unit test suite (77 tests):

```powershell
python -m unittest discover -s test
```

---

## 8. Requirements & Setup

1. **Python Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Ollama Setup**:
   Ensure Ollama is running locally:
   ```bash
   ollama pull qwen3.5:0.8b
   ollama pull gemma3:4b
   ollama pull mistral:7b
   ollama pull llama3.2:3b
   ```
3. **OpenAI Cloud API (Optional)**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
4. **Apple Silicon Hardware Telemetry (macOS only)**:
   For direct powermetrics access without password prompts:
   ```bash
   sudo visudo
   # Add: <username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
   ```

---

## 9. Known Limitations

* **Cloud Energy Comparability**: Cloud API energy values are software/LCA estimates and must not be compared directly against hardware-level wattmeters without explicit labeling.
* **CPU vs GPU Power Isolation**: On CPU-only systems (e.g. standard Windows PCs), CodeCarbon measures system-level estimation; on NVIDIA servers, NVML measures GPU-only package power.
