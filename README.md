# PromptEnergy-Bench: Energy–Accuracy–Latency Benchmark for LLM Prompting

A reproducible, device-portable benchmarking framework designed to empirically measure and optimize the Energy–Accuracy–Latency trade-offs induced by prompting strategies, controlled context scaling, and retrieval-augmented generation (RAG) across heterogeneous hardware platforms.

---

## 1. Project Purpose & Research Scope

While prompt engineering techniques such as Few-Shot prompting and Chain-of-Thought (CoT) reasoning often yield improvements in task accuracy, they systematically expand token generation budgets, inflight KV-cache occupancy, prefill compute, and decode latency. 

**PromptEnergy-Bench** quantifies the marginal energy cost of reasoning:
* **Marginal Energy Gain (MEG)**:
  $$\text{MEG}(S_1, S_2) = \frac{\text{Energy}(S_2) - \text{Energy}(S_1)}{\text{Accuracy}(S_2) - \text{Accuracy}(S_1)} \quad \left( \frac{\text{Joules}}{\Delta\text{Accuracy \%}} \right)$$
* **Accuracy per Joule ($APJ$)**: Normalized task accuracy achieved per unit of electrical energy consumed.
* **Energy per Correct Answer**: Total energy consumed divided by the number of correct responses.
* **Pareto Frontier Efficiency**: Identifying configurations that optimize the tripartite trade-off between Accuracy, Energy (Joules), and Latency (ms).

---

## 2. Dataset Coverage & Leakage Protection

The benchmark provides a universal dataset loader supporting 4 core benchmark datasets:

```text
datasets/
├── gsm8k/               <-- Grade School Math Reasoning (Test: 1,319 | Train: 7,473)
├── natural_questions/   <-- Factoid Open QA (Train/Test: 100k+ questions & Wikipedia corpus)
├── contexteval/         <-- Long-Context Grounded QA (Test: 3,580 questions)
└── cnn_dailymail/       <-- Multi-Document Summarization (Test: 11,490 | Validation: 13,368)
```

### Strict Train/Test Separation Rules
* Evaluation is strictly performed on the **TEST** split of each dataset.
* Training splits provide in-context exemplars, context-scaling distractors, and BM25 retrieval corpora.
* A test question is never evaluated against its own answer or used as retrieved context (`exclude_id` enforcement).
* Selecting `--eval-size full` evaluates the entire test split without downsampling.

---

## 3. Supported Hardware Profiles & Telemetry Adapters

| Platform / Environment | Telemetry Adapter | Measurement Level | Telemetry Resolution | Measurement Interface |
| :--- | :--- | :--- | :--- | :--- |
| **macOS (Apple Silicon M1/M2/M3)** | `apple_powermetrics` | Hardware Reported (Full SoC) | 100 ms | `/usr/bin/powermetrics` register counters |
| **Windows with NVIDIA GPU** | `nvidia_nvml` / `codecarbon` | Hardware Reported (GPU Package) | 10 ms | NVML (`nvidia-ml-py`) power integration |
| **Windows (CPU-Only / Intel / AMD)** | `rapl` / `codecarbon_estimated` | Software/RAPL Estimated (CPU Package) | 100 ms | Intel/AMD MSR power estimation + `psutil` |
| **Linux (x86 Server + NVIDIA GPU)** | `rapl` + `nvidia_nvml` | Hardware Reported (CPU + GPU) | Microsecond / 10 ms | `/sys/class/powercap` + NVML |

---

## 4. Cross-Device Execution Guide

### A. macOS (Apple Silicon M1/M2/M3 — 8GB / 16GB / 24GB RAM)

Apple Silicon leverages hardware register counters via `powermetrics` and optimized Apple Metal MLX inference.

1. **Setup Passwordless Telemetry** (allows `powermetrics` to read energy counters without prompting for password):
   ```bash
   sudo visudo
   # Add the following line at the bottom:
   <your_mac_username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
   ```

2. **Run Full Benchmark Suite (Master Runner)**:
   ```bash
   ./.venv/bin/python scripts/run_all_experiments.py \
       --hardware macbook_air_m1 \
       --models "qwen3.5:0.8b-mlx,qwen3.5:2b-mlx" \
       --dataset all \
       --eval-size 50 \
       --runs-per-condition 3 \
       --skip-existing
   ```

---

### B. Windows with NVIDIA GPU (e.g., RTX 3060, 3080, 4070, 4090)

Uses NVML high-frequency GPU telemetry ($10\text{ms}$ sampling) integrated over inference execution time.

1. **Prerequisites**: Ensure CUDA drivers and Ollama are running.
   ```powershell
   ollama pull qwen3.5:0.8b
   ollama pull gemma3:4b
   ollama pull mistral:7b
   ollama pull llama3.2:3b
   ```

2. **Run Full Benchmark Suite (PowerShell)**:
   ```powershell
   python scripts/run_all_experiments.py `
       --hardware windows_cuda_pc `
       --models "qwen3.5:0.8b,gemma3:4b,llama3.2:3b" `
       --dataset all `
       --eval-size 50 `
       --runs-per-condition 3 `
       --skip-existing
   ```

---

### C. Windows without Dedicated GPU (CPU-Only / Intel & AMD)

Profiles CPU power draw and memory bandwidth via RAPL estimation and continuous `psutil` sampling.

1. **Prerequisites**: Pull lightweight models optimized for CPU inference:
   ```powershell
   ollama pull qwen3.5:0.8b
   ollama pull qwen2.5:0.5b
   ollama pull gemma3:1b
   ```

2. **Run Full Benchmark Suite (PowerShell)**:
   ```powershell
   python scripts/run_all_experiments.py `
       --hardware windows_10core_pc `
       --models "qwen3.5:0.8b,gemma3:1b" `
       --dataset gsm8k `
       --eval-size 50 `
       --runs-per-condition 1 `
       --skip-existing
   ```

---

### D. Linux Server (Ubuntu/Debian + NVIDIA GPU Cluster)

Profiles dual CPU package (RAPL) and GPU (NVML) power simultaneously.

1. **Run Full Benchmark Suite (Bash)**:
   ```bash
   python scripts/run_all_experiments.py \
       --hardware generic_cuda_server \
       --models "qwen3.5:0.8b,gemma3:4b,mistral:7b" \
       --dataset all \
       --eval-size 100 \
       --runs-per-condition 3 \
       --skip-existing
   ```

---

## 5. Comparing Results Across All Devices

All benchmark runs produce standardized `results.jsonl` files stored under `results/{experiment_name}_{dataset}/{hardware_name}/{timestamp}/`.

To compare across devices (e.g., MacBook Air M1 vs. Windows GPU vs. Windows CPU vs. Linux Server):

```text
results/
├── primary_exp_gsm8k/
│   ├── macbook_air_m1/       <-- MacBook Air M1 runs
│   ├── windows_cuda_pc/      <-- Windows NVIDIA GPU runs
│   └── windows_10core_pc/    <-- Windows CPU-only runs
└── context_scaling_gsm8k/
    ├── macbook_air_m1/
    └── windows_cuda_pc/
```

### Option 1: Interactive Comparison Mode (Terminal Selection Menu)
Run the comparison tool without arguments to automatically discover all runs across all devices and select which runs to compare via interactive checkboxes:

```bash
python scripts/compare_results.py
```

### Option 2: Command-Line Multi-Device Comparison
Pass all device folders or specific run paths to generate a unified cross-device report:

```bash
python scripts/compare_results.py \
    --inputs results/*/*/* \
    --output-dir results/cross_device_comparison
```

### Option 3: Compare Specific Hardware Platforms Directly
```bash
python scripts/compare_hardware_results.py \
    --input-dirs results/primary_exp_gsm8k/macbook_air_m1 results/primary_exp_gsm8k/windows_cuda_pc \
    --output-dir results/hardware_comparison/
```

---

## 6. Generated Comparison Deliverables

Every comparison execution outputs camera-ready publication artifacts:

### Publication Tables (`.csv`, `.md`, `.tex` Booktabs):
1. **`table_model_comparison.*`**: Accuracy, Total Energy (J), Prefill Energy (J), Decode Energy (J), TTFT (ms), Throughput (tok/s) across devices and models.
2. **`table_strategy_comparison.*`**: Fine-grained metrics for Zero-Shot, Few-Shot-3/5, Chain-of-Thought (CoT), Role-Play, and System Prompts.
3. **`table_marginal_energy_gain.*`**: Marginal Energy Gain ($\text{MEG}$) quantifying the Joules required per 1% accuracy improvement.
4. **`cross_hardware_energy_ratios.*`**: Energy consumption ratios ($E_{\text{device\_A}} / E_{\text{device\_B}}$) and throughput speedups.

### High-Resolution Figures (PNG @ 300 DPI, Vector PDF, SVG):
1. **`fig_energy_accuracy_pareto.*`**: Multi-objective Energy vs. Accuracy Pareto Frontiers by device and model.
2. **`fig_strategy_energy_breakdown.*`**: Stacked Prefill vs. Decode energy charts.
3. **`fig_latency_breakdown.*`**: Time-to-First-Token (TTFT) and decode latency distributions.
4. **`fig_context_scaling_curve.*`**: Energy growth vs Context Length ($0 \dots 8192$ tokens) scaling curves.

---

## 7. Running Unit Tests & Quality Verification

Execute the complete automated test suite (77 tests across data loading, evaluators, telemetry, and analysis):

```bash
PYTHONPATH=. python -m unittest discover -s test -p "test_*.py"
```

---

## 8. Requirements & Setup

1. **Python Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
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

3. **OpenAI Cloud API (Optional for API Baselines)**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

---

## 9. Methodological & Rigor Standards

* **Zero-Coercion Policy**: Failed or unparseable queries preserve strict `null` values (rendered as `"N/A"` in tables/plots) and are never coerced to `0.0` or synthetic defaults.
* **Denominator Integrity**: Explicitly reports $N_{\text{requested}}$, $N_{\text{completed}}$, $N_{\text{valid}}$, and $N_{\text{correct}}$ to eliminate reporting bias.
* **Idle Power Calibration**: All hardware runs calibrate a 10-second idle baseline ($P_{\text{idle}}$) to compute net dynamic energy: $E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \Delta t)$.
* **Warmup & JIT Normalization**: Execution of unmeasured warmup cycles ($W \ge 1$) prior to benchmark logging ensures memory paging and cache initialization do not contaminate energy measurements.