# PromptEnergy-Bench: Energy–Accuracy–Latency Benchmark for LLM Prompting

A reproducible, device-portable benchmarking framework designed to empirically measure and optimize the Energy–Accuracy–Latency trade-offs induced by prompting strategies, controlled context scaling, and retrieval-augmented generation (RAG) across heterogeneous hardware platforms.

---

## 1. Project Purpose & Research Scope

While prompt engineering techniques such as Few-Shot prompting and Chain-of-Thought (CoT) reasoning often yield improvements in task accuracy, they systematically expand token generation budgets, inflight KV-cache occupancy, prefill compute, and decode latency. 

**PromptEnergy-Bench** quantifies the marginal energy cost of reasoning:
* **Marginal Energy Gain (MEG)**:
  $$\text{MEG}(S_1, S_2) = \frac{\text{Accuracy}(S_2) - \text{Accuracy}(S_1)}{\text{Energy}(S_2) - \text{Energy}(S_1)} \quad \left( \frac{\Delta\text{Accuracy \%}}{\text{Joules}} \right)$$
* **Prompt Processing (Prefill) Energy ($E_{\text{prefill}}$)**: Energy consumed during prompt ingestion and initial KV-cache construction:
  $$E_{\text{prefill}} = E_{\text{total}} \times \left( \frac{\text{TTFT}}{\text{Total Latency}} \right)$$
* **Autoregressive Generation (Decode) Energy ($E_{\text{decode}}$)**: Energy consumed sequentially generating output tokens:
  $$E_{\text{decode}} = E_{\text{total}} \times \left( \frac{\text{Generation Latency}}{\text{Total Latency}} \right)$$
* **Unit Token Energy ($m\text{J/token}$)**: Input prompt processing cost vs. output generation cost.
* **Accuracy per Joule ($APJ$)**: Task accuracy achieved per unit of electrical energy consumed.
* **Pareto Frontier Efficiency**: Identifying configurations that optimize the tripartite trade-off between Accuracy, Energy (Joules), and Latency (ms).

---

## 2. Dataset Coverage & Leakage Protection

The benchmark provides a universal dataset loader supporting 4 core benchmark datasets:

```text
datasets/
├── gsm8k/               <-- Grade School Math Reasoning (Test: 1,319 | Train: 7,473)
├── natural_questions/   <-- Factoid Open QA & RAG (Test: 10,000+ | Wikipedia corpus)
├── contexteval/         <-- Long-Context Grounded QA (Test: 3,580 questions)
└── cnn_dailymail/       <-- Multi-Sentence Summarization (Test: 11,490 | Train: 287,113)
```

### Strict Train/Test Separation Rules
* Evaluation is strictly performed on the **TEST** split of each dataset.
* Training splits provide in-context exemplars, context-scaling distractors, and BM25 retrieval corpora.
* A test question is never evaluated against its own answer or used as retrieved context (`exclude_id` enforcement).
* Selecting `--eval-size full` evaluates the entire test split without downsampling.

For verbatim prompt templates and strategy definitions across all 4 datasets, see [prompts.md](file:///Users/mdsaibhossain/code/Research_paper/PromptEnergy-Bench/prompts.md).

---

## 3. Supported Hardware Profiles & Telemetry Adapters

| Platform / Environment | Telemetry Adapter | Measurement Level | Telemetry Resolution | Measurement Interface | Hardware Flag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Windows (CPU-Only / Intel / AMD)** | `rapl` / `codecarbon_estimated` | Software/RAPL Estimated (CPU Package) | 100 ms | Intel/AMD MSR power estimation + `psutil` | `--hardware windows_10core_pc` |
| **Windows with NVIDIA GPU** | `nvidia_nvml` / `codecarbon` | Hardware Reported (GPU Package) | 10 ms | NVML (`nvidia-ml-py`) power integration | `--hardware windows_cuda_pc` |
| **Google Colab (NVIDIA T4 / A100 GPU)** | `nvidia_nvml` | Hardware Reported (GPU Package) | 10 ms | NVML (`nvidia-ml-py`) power integration | `--hardware colab_t4_gpu` |
| **macOS (Apple Silicon M1/M2/M3)** | `apple_powermetrics` | Hardware Reported (Full SoC) | 100 ms | `/usr/bin/powermetrics` register counters | `--hardware macbook_air_m1` |
| **Linux (x86 Server + NVIDIA GPU)** | `rapl` + `nvidia_nvml` | Hardware Reported (CPU + GPU) | Microsecond / 10 ms | `/sys/class/powercap` + NVML | `--hardware generic_cuda_server` |

---

## 4. How to Run the Exact Same Experiment on Any Device

The master execution entry point is `scripts/run_all_experiments.py`, which orchestrates all 3 benchmark experiment modules:
1. **Experiment 1: Prompting Strategies** (`zero_shot_direct`, `few_shot_3`, `zero_shot_cot`, `short_cot`, `long_cot`)
2. **Experiment 2: Context Scaling** (`ctx_0`, `ctx_512`, `ctx_1024`, `ctx_2048`, `ctx_4096`, `ctx_8192`)
3. **Experiment 3: Retrieval-Augmented Generation** (`rag_top_1`, `rag_top_3`, `rag_top_5`)

The standard experiment evaluates **3 core models** (`qwen3.5:2b`, `qwen3.5:0.8b`, `gemma3:4b`) across **all 4 benchmark datasets** (`gsm8k`, `natural_questions`, `contexteval`, `cnn_dailymail`).

---

### Step 1: Environment Setup & Model Pulling

Ensure Ollama is running, then pull the benchmark models:

```bash
# Pull the 3 standard benchmark models
ollama pull qwen3.5:2b
ollama pull qwen3.5:0.8b
ollama pull gemma3:4b

# Verify active models
ollama list
```

---

### Step 2: Multi-Platform Execution Guide

#### A. Windows PC (CPU-Only / Standard x86 PC)
Run from PowerShell inside your activated virtual environment:

```powershell
# Quick Validation Test (10 samples per dataset)
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:2b,qwen3.5:0.8b,gemma3:4b" `
    --dataset all `
    --experiment all `
    --eval-size 10 `
    --runs-per-condition 1 `
    --skip-existing

# Full Benchmark Run (Full evaluation splits)
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:2b,qwen3.5:0.8b,gemma3:4b" `
    --dataset all `
    --experiment all `
    --eval-size full `
    --runs-per-condition 1 `
    --skip-existing
```

> **Convenience Script**: You can also run the PowerShell wrapper directly:
> ```powershell
> .\scripts\test_pipeline_run.ps1
> ```

---

#### B. Windows PC with Dedicated NVIDIA GPU (CUDA)
Run from PowerShell with NVML hardware power polling:

```powershell
python scripts/run_all_experiments.py `
    --hardware windows_cuda_pc `
    --models "qwen3.5:2b,qwen3.5:0.8b,gemma3:4b" `
    --dataset all `
    --experiment all `
    --eval-size 10 `
    --runs-per-condition 1 `
    --skip-existing
```

---

#### C. Google Colab (Free NVIDIA T4 / A100 GPU)

In Google Colab, set your runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`), then run the following cells:

##### **Cell 1: Install & Start Ollama Daemon**
```python
!curl -fsSL https://ollama.com/install.sh | sh

import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
print("Ollama daemon is running!")
```

##### **Cell 2: Pull the 3 Models**
```bash
!ollama pull qwen3.5:2b
!ollama pull qwen3.5:0.8b
!ollama pull gemma3:4b
!ollama list
```

##### **Cell 3: Clone Repository & Install Dependencies**
```bash
!git clone https://github.com/Saibhossain/PromptEnergy-Bench.git
%cd PromptEnergy-Bench
!pip install -r requirements.txt
```

##### **Cell 4: Execute the Benchmark on Colab**
```bash
!python scripts/run_all_experiments.py \
    --hardware colab_t4_gpu \
    --models "qwen3.5:2b,qwen3.5:0.8b,gemma3:4b" \
    --dataset all \
    --experiment all \
    --eval-size 10 \
    --runs-per-condition 1 \
    --skip-existing
```

##### **Cell 5: Download Colab Results Package**
```python
!zip -r colab_results.zip results/
from google.colab import files
files.download('colab_results.zip')
```

---

#### D. Kaggle (Dual NVIDIA T4 GPUs — 2x Faster Parallel Execution)

In Kaggle Notebooks, select **Accelerator -> GPU T4 x 2** in the right-hand panel.

##### **Cell 1: Install & Launch Ollama Daemon**
```bash
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
```

##### **Cell 2: Pull Benchmark Models**
```bash
!ollama pull qwen3.5:2b
!ollama pull qwen3.5:0.8b
!ollama pull gemma3:4b
```

##### **Cell 3: Clone Repository & Install Dependencies**
```bash
!git clone https://github.com/Saibhossain/PromptEnergy-Bench.git
%cd PromptEnergy-Bench
!pip install -r requirements.txt
```

##### **Cell 4: Run Dual-GPU Parallel Execution**
Split model workloads across GPU 0 and GPU 1 to complete the benchmark in half the time:
```python
import subprocess

# GPU 0 executes qwen3.5:2b
p0 = subprocess.Popen([
    "python", "scripts/run_all_experiments.py",
    "--hardware", "kaggle_t4_gpu",
    "--gpu-id", "0",
    "--models", "qwen3.5:2b",
    "--dataset", "all",
    "--eval-size", "10",
    "--skip-existing"
])

# GPU 1 executes qwen3.5:0.8b & gemma3:4b
p1 = subprocess.Popen([
    "python", "scripts/run_all_experiments.py",
    "--hardware", "kaggle_t4_gpu",
    "--gpu-id", "1",
    "--models", "qwen3.5:0.8b,gemma3:4b",
    "--dataset", "all",
    "--eval-size", "10",
    "--skip-existing"
])

p0.wait()
p1.wait()
print("All Kaggle Dual-GPU experiments completed!")
```

##### **Cell 5: Download Kaggle Results Package**
```python
!zip -r kaggle_results.zip results/
```

---

#### E. macOS (Apple Silicon M1/M2/M3)

1. **Configure Passwordless Telemetry** (once):
   ```bash
   sudo visudo
   # Add at the bottom: <your_username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
   ```

2. **Execute Benchmark**:
   ```bash
   ./.venv/bin/python scripts/run_all_experiments.py \
       --hardware macbook_air_m1 \
       --models "qwen3.5:2b-mlx,qwen3.5:0.8b-mlx,gemma3:4b" \
       --dataset all \
       --experiment all \
       --eval-size 10 \
       --runs-per-condition 1 \
       --skip-existing
   ```

---

#### F. Linux Server (Ubuntu/Debian + NVIDIA GPU Cluster)

```bash
python scripts/run_all_experiments.py \
    --hardware generic_cuda_server \
    --models "qwen3.5:2b,qwen3.5:0.8b,gemma3:4b" \
    --dataset all \
    --experiment all \
    --eval-size 10 \
    --runs-per-condition 1 \
    --skip-existing
```

---

### Step 3: Merging Results Across Heterogeneous Devices

Because PromptEnergy-Bench isolates results by normalized hardware slug, results from different devices never collide:

```text
results/
├── primary_exp_gsm8k/
│   ├── windows_10core_pc/     <-- Windows CPU run
│   ├── colab_t4_gpu/          <-- Google Colab T4 GPU run
│   └── macbook_air_m1/        <-- Apple Silicon M1 run
├── context_scaling_gsm8k/
│   ├── windows_10core_pc/
│   └── colab_t4_gpu/
└── rag_gsm8k/
    ├── windows_10core_pc/
    └── colab_t4_gpu/
```

**To merge Google Colab results into your main workspace:**
1. Extract the downloaded `colab_results.zip` directly into your repository's root `results/` folder.
2. The `colab_t4_gpu/` subdirectories will sit seamlessly alongside your `windows_10core_pc/` folders.
3. (Optional) Commit and push the merged results to GitHub:
   ```bash
   git add results/
   git commit -m "Merge Colab T4 GPU benchmark results"
   git push
   ```

---

### Step 4: Generating Cross-Hardware Comparison Deliverables

Once multiple hardware folders exist in `results/`, generate cross-hardware comparative tables (CSV, Markdown, LaTeX) and high-resolution figures (PNG 300 DPI, PDF):

```powershell
python scripts/compare_hardware_results.py `
    --input-dirs results/ `
    --output-dir results/hardware_comparison
```

#### Generated Comparison Deliverables in `results/hardware_comparison/`:

| Deliverable Type | Files Generated | Description | Formats |
| :--- | :--- | :--- | :--- |
| **7 Comparison Tables** | `hardware_summary` | Overall system specs, idle power, active power, and throughput | `.csv`, `.md`, `.tex` |
| | `prompt_strategy_by_hardware` | Energy, latency, and accuracy per strategy across devices | `.csv`, `.md`, `.tex` |
| | `model_by_hardware` | Cross-model energy consumption and inference speedups | `.csv`, `.md`, `.tex` |
| | `energy_accuracy_comparison` | Energy efficiency per percentage point of accuracy | `.csv`, `.md`, `.tex` |
| | `latency_comparison` | TTFT and decode latency across devices | `.csv`, `.md`, `.tex` |
| | `measurement_method_comparison` | NVML vs. CodeCarbon vs. RAPL telemetry audit | `.csv`, `.md`, `.tex` |
| | `cross_hardware_energy_ratios` | Relative energy consumption ($E_{\text{device\_A}} / E_{\text{device\_B}}$) | `.csv`, `.md`, `.tex` |
| **10 Comparison Figures** | `energy_by_hardware_prompt` | Grouped bar chart of energy per prompt strategy by device | `.png`, `.pdf` |
| | `accuracy_by_hardware_prompt` | Accuracy consistency across hardware platforms | `.png`, `.pdf` |
| | `ttft_by_hardware_prompt` | Time-to-First-Token prefill latency by hardware | `.png`, `.pdf` |
| | `latency_by_hardware_prompt` | Total query latency comparison | `.png`, `.pdf` |
| | `energy_accuracy_by_hardware` | Multi-device Pareto frontiers | `.png`, `.pdf` |
| | `model_energy_across_hardware` | Energy consumption across model scales and hardware | `.png`, `.pdf` |
| | `context_scaling_across_hardware`| Context length scaling slopes across devices | `.png`, `.pdf` |
| | `rag_energy_across_hardware` | Retrieval vs. Generation energy across devices | `.png`, `.pdf` |
| | `hardware_energy_ratio_plot` | Relative hardware efficiency ratios | `.png`, `.pdf` |
| | `summary_heatmap` | Normalized multi-dimensional performance heatmap | `.png`, `.pdf` |

---

### Step 5: Run Targeted Experiments or Datasets Individually

You can customize the runner using CLI flags:

#### Run Only a Specific Experiment
```bash
# Run only Experiment 1 (Prompting Strategies)
./.venv/bin/python scripts/run_all_experiments.py --experiment primary --dataset gsm8k --eval-size 50

# Run only Experiment 2 (Context Scaling)
./.venv/bin/python scripts/run_all_experiments.py --experiment context --dataset contexteval --eval-size 50

# Run only Experiment 3 (BM25 RAG)
./.venv/bin/python scripts/run_all_experiments.py --experiment rag --dataset natural_questions --eval-size 50
```

#### Run a Single Dataset
```bash
# GSM8K (Math)
./.venv/bin/python scripts/run_all_experiments.py --dataset gsm8k --eval-size full --skip-existing

# Natural Questions (Open QA)
./.venv/bin/python scripts/run_all_experiments.py --dataset natural_questions --eval-size full --skip-existing

# ContextEval (Long-Context QA)
./.venv/bin/python scripts/run_all_experiments.py --dataset contexteval --eval-size full --skip-existing

# CNN/DailyMail (Summarization)
./.venv/bin/python scripts/run_all_experiments.py --dataset cnn_dailymail --eval-size full --skip-existing
```

---

### CLI Reference Flags (`scripts/run_all_experiments.py`)

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--hardware` | Hardware preset name (`macbook_air_m1`, `windows_cuda_pc`, `windows_10core_pc`, `generic_cuda_server`) | Auto-detected |
| `--models` | Comma-separated list of model names (`"qwen3.5:0.8b-mlx,qwen3.5:2b-mlx"`) | `"qwen3.5:0.8b-mlx"` |
| `--dataset` | Target dataset(s): `all`, `gsm8k`, `natural_questions`, `contexteval`, `cnn_dailymail` | `all` |
| `--experiment` | Target experiment(s): `all`, `primary`, `context`, `rag` | `all` |
| `--eval-size` | Number of test samples to evaluate (`10`, `50`, `100`, or `full`) | `10` |
| `--runs-per-condition` | Number of repeated runs per experimental condition for statistical power | `1` |
| `--warmups` | Number of unmeasured warmup queries before evaluation | `1` |
| `--skip-existing` | Skip previously completed runs to enable fast resumability | `False` |
| `--no-compare` | Disable automatic post-run comparison generation | `False` |

---

## 5. Result Comparison & Publication Deliverables

All runs produce standardized raw telemetry in `results/{experiment}_{dataset}/{hardware}/{timestamp}/results.jsonl`.

### Option 1: Interactive Comparison Menu
Run without arguments to select runs across devices and models interactively:
```bash
./.venv/bin/python scripts/compare_results.py
```

### Option 2: Automated Multi-Run Comparison Report
```bash
./.venv/bin/python scripts/compare_results.py \
    --inputs results/*/*/* \
    --output-dir results/master_comparison
```

### Option 3: Generate Global Publication Figures
Generates 12 camera-ready publication figures (PNG @ 300 DPI, vector PDF, and SVG):
```bash
./.venv/bin/python scripts/generate_publication_figures.py \
    --input-dir results/ \
    --output-dir results/figures/
```

### Generated Deliverables

#### Publication Tables (`.csv`, `.md`, `.tex` Booktabs):
1. **`1_model_comparison.*`**: Model parameter size, total energy (J), prefill energy (J), decode energy (J), prefill/decode cost ($m\text{J/token}$), throughput (tok/s), TTFT (ms), and total latency (ms).
2. **`2_strategy_comparison.*`**: Prompt strategy breakdown across Zero-Shot Direct, Few-Shot (3), Zero-Shot CoT, Short CoT, Long CoT, and RAG.
3. **`3_pareto_frontier.*`**: Multi-objective Pareto-optimal prompting configurations.
4. **`4_marginal_energy_gain.*`**: Marginal Energy Gain ($\text{MEG}$) quantifying Joules required per 1% accuracy improvement.

#### Publication Figures (PNG @ 300 DPI, Vector PDF, SVG):
1. **`01_energy_vs_accuracy_pareto.*`**: Multi-objective Energy vs. Accuracy Pareto Frontier with collision-free callouts.
2. **`02_strategy_energy_breakdown.*`**: Stacked Prefill vs. Decode energy distribution.
3. **`03_latency_ttft_breakdown.*`**: TTFT and decode latency phase breakdown.
4. **`04_unit_energy_per_token.*`**: Input token prefill cost vs. output token generation cost ($m\text{J/token}$).

---

## 6. Running Unit Tests

Execute the complete automated test suite (77 unit tests covering prompt formatting, metrics, telemetry, evaluators, and Pareto analysis):

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest discover -s test -p "test_*.py"
```

---

## 7. Methodological & Rigor Standards

* **Zero-Coercion Policy**: Failed or unparseable queries preserve strict `null` values (rendered as `"N/A"` in tables/plots) and are never coerced to `0.0` or synthetic defaults.
* **Denominator Integrity**: Explicitly reports $N_{\text{requested}}$, $N_{\text{completed}}$, $N_{\text{valid}}$, and $N_{\text{correct}}$ to eliminate reporting bias.
* **Idle Power Calibration**: All hardware runs calibrate a 10-second idle baseline ($P_{\text{idle}}$) to compute net dynamic energy: $E_{\text{net}} = E_{\text{total}} - (P_{\text{idle}} \times \Delta t)$.
* **Warmup & JIT Normalization**: Execution of unmeasured warmup cycles ($W \ge 1$) prior to benchmark logging ensures memory paging and cache initialization do not contaminate energy measurements.