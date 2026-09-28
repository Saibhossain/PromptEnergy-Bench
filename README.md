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

| Platform / Environment | Telemetry Adapter | Measurement Level | Telemetry Resolution | Measurement Interface |
| :--- | :--- | :--- | :--- | :--- |
| **macOS (Apple Silicon M1/M2/M3)** | `apple_powermetrics` | Hardware Reported (Full SoC) | 100 ms | `/usr/bin/powermetrics` register counters |
| **Windows with NVIDIA GPU** | `nvidia_nvml` / `codecarbon` | Hardware Reported (GPU Package) | 10 ms | NVML (`nvidia-ml-py`) power integration |
| **Windows (CPU-Only / Intel / AMD)** | `rapl` / `codecarbon_estimated` | Software/RAPL Estimated (CPU Package) | 100 ms | Intel/AMD MSR power estimation + `psutil` |
| **Linux (x86 Server + NVIDIA GPU)** | `rapl` + `nvidia_nvml` | Hardware Reported (CPU + GPU) | Microsecond / 10 ms | `/sys/class/powercap` + NVML |

---

## 4. How to Run Experiments

The primary execution entry point is `scripts/run_all_experiments.py`, which orchestrates all 3 benchmark experiment modules:
1. **Experiment 1: Prompting Strategies** (`zero_shot_direct`, `few_shot_3`, `zero_shot_cot`, `short_cot`, `long_cot`)
2. **Experiment 2: Context Scaling** (`ctx_0`, `ctx_512`, `ctx_1024`, `ctx_2048`, `ctx_4096`, `ctx_8192`)
3. **Experiment 3: Retrieval-Augmented Generation** (`rag_top_1`, `rag_top_3`, `rag_top_5`)

---

### Step 1: Environment Setup

```bash
# 1. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull target models via Ollama
ollama pull qwen3.5:0.8b
ollama pull qwen3.5:2b
```

#### macOS Passwordless Telemetry Setup
On macOS (Apple Silicon), enable non-interactive energy sampling via `powermetrics`:
```bash
sudo visudo
# Add this line at the bottom of the file (replace <your_username> with your macOS username):
<your_username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

---

### Step 2: Quick Smoke Test (Verify Pipeline)

Run a fast validation on **10 samples per dataset** across all 4 benchmark datasets:

```bash
./.venv/bin/python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:0.8b-mlx" \
    --dataset all \
    --eval-size 10
```

---

### Step 3: Run Full Benchmark Suite

#### A. Full Run on macOS (Apple Silicon M1/M2/M3)
```bash
./.venv/bin/python scripts/run_all_experiments.py \
    --hardware macbook_air_m1 \
    --models "qwen3.5:0.8b-mlx,qwen3.5:2b-mlx" \
    --dataset all \
    --eval-size full \
    --runs-per-condition 1 \
    --skip-existing
```

> **Note on `--skip-existing`**: If your run is interrupted, re-running with `--skip-existing` will automatically resume where it left off, skipping completed conditions.

#### B. Full Run on Windows with NVIDIA GPU (PowerShell)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_cuda_pc `
    --models "qwen3.5:0.8b,qwen3.5:2b" `
    --dataset all `
    --eval-size full `
    --runs-per-condition 1 `
    --skip-existing
```

#### C. Full Run on Windows CPU-Only (PowerShell)
```powershell
python scripts/run_all_experiments.py `
    --hardware windows_10core_pc `
    --models "qwen3.5:0.8b" `
    --dataset all `
    --eval-size full `
    --runs-per-condition 1 `
    --skip-existing
```

#### D. Full Run on Linux Server (Ubuntu/Debian + NVIDIA GPU)
```bash
python scripts/run_all_experiments.py \
    --hardware generic_cuda_server \
    --models "qwen3.5:0.8b,qwen3.5:2b" \
    --dataset all \
    --eval-size full \
    --runs-per-condition 1 \
    --skip-existing
```

---

### Step 4: Run Targeted Experiments or Datasets Individually

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