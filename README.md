# PromptEnergy-Bench: Energy–Accuracy–Latency Benchmark for LLM Prompting

A research-grade, device-portable benchmarking framework designed to empirically measure the Energy–Accuracy–Latency trade-offs induced by prompting strategies, controlled context scaling, and retrieval-augmented generation (RAG) across heterogeneous hardware platforms.

---

## 1. Project Purpose & Research Objectives

While prompt engineering techniques such as Few-Shot prompting and Chain-of-Thought (CoT) reasoning often yield improvements in task accuracy, they systematically expand token generation budgets, inflight KV-cache occupancy, prefill compute, and decode latency. 

**PromptEnergy-Bench** quantifies the marginal energy cost of reasoning:
* **Marginal Energy Gain (MEG)**: $\text{MEG}(B_1, B_2) = \frac{\text{Accuracy}(B_2) - \text{Accuracy}(B_1)}{\text{Energy}(B_2) - \text{Energy}(B_1)}$
* **Accuracy per Joule**: Normalized task throughput per unit of hardware energy consumed.
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
* `datasets/gsm8k/train.jsonl` provides the 3 fixed few-shot examples, controlled context scaling blocks, and BM25 retrieval documents.
* A test question is never evaluated against its own answer or used as retrieved context.
* Selecting `--eval-size full` evaluates the entire GSM8K **TEST** split (1,319 problems). It never switches evaluation to the train split.

Each record is normalized to:
```json
{
  "id": "gsm8k_test_0000",
  "question": "Janet’s ducks lay 16 eggs per day...",
  "answer": "18",
  "solution": "Janet sells 16 - 3 - 4 = 9 duck eggs a day...",
  "raw_answer": "... #### 18"
}
```

---

## 3. Supported Backends & Model Formats

The architecture strictly decouples experiment logic from execution providers:

| Backend Provider | Model Formats Supported | Platforms | Token Usage Source | TTFT Support |
| :--- | :--- | :--- | :--- | :--- |
| **Ollama** (`src/backends/ollama_backend.py`) | MLX, GGUF, F16, BF16, FP8, INT8, INT4 | macOS, Linux, Windows | Native backend usage (`eval_count`, `prompt_eval_count`) | Yes (streaming) |
| **OpenAI / Compatible** (`src/backends/openai_backend.py`) | Cloud API | Cross-platform | API response usage metadata | Yes (streaming) |
| **Hugging Face Transformers** (`src/backends/transformers_backend.py`) | PyTorch / SafeTensors (F16, BF16, 4-bit) | Linux, Windows, macOS | Model tokenizer | Yes |
| **MLX-LM** (`src/backends/mlx_backend.py`) | Native MLX Apple Silicon format | macOS (Apple Silicon) | Tokenizer | Yes |
| **llama.cpp** (`src/backends/llama_cpp_backend.py`) | GGUF | Linux, macOS, Windows | Backend reported usage | Yes |

---

## 4. Hardware Profiles & Energy Measurement Methodology

### Measurement Adapters
1. **Physical Power Meter (`physical_meter`)**: External hardware meter (Shelly, Tapo, WattsUp) recording direct mains electrical draw. Quality: `direct_measurement`.
2. **NVIDIA NVML (`nvidia_nvml`)**: High-frequency GPU power polling via NVML (`nvidia-ml-py`) integrated over time ($E = \int P(t) dt$). Quality: `hardware_reported`, Level: `gpu_only`.
3. **Apple Silicon Estimator (`apple_estimated`)**: Tracks CPU/RAM power via CodeCarbon constant TDP models (5.0W CPU + 3.0W RAM base on Apple M1). Quality: `software_estimate`, Level: `estimated_system`.
4. **CodeCarbon Software Estimator (`codecarbon_estimated`)**: Cross-platform software emission and energy tracking. Quality: `software_estimate`.
5. **Null Monitor (`unavailable`)**: Fallback when no grounded meter exists. Emits `energy_total_j = null` and `energy_status = "unavailable"`. **Energy values are never faked.**

---

## 5. Result Directory Architecture

Every run produces a completely isolated, immutable, timestamped directory:

```text
results/
├── {experiment_name}/
│   └── {normalized_device_name}/
│       └── {experiment_timestamp}/
│           ├── metadata.json       # System, git, hardware, and package provenance
│           ├── config.json         # Exact experiment configuration
│           ├── results.jsonl       # Per-sample atomic inference records
│           ├── summary.json        # Aggregate metrics across conditions
│           ├── plots/
│           │   ├── validation/     # Validation-specific plots
│           │   └── final/          # Final benchmark plots
│           └── logs/
│               └── experiment.log  # Full execution transcript
```

### Canonical Device Slug Normalization
User input is automatically sanitized into filesystem-safe slugs:
* `"MacBook Air M1"` $\rightarrow$ `macbook_air_m1`
* `"Windows RTX 4070 PC"` $\rightarrow$ `windows_rtx_4070_pc`
* `"Ubuntu NVIDIA A100 Server!"` $\rightarrow$ `ubuntu_nvidia_a100_server`

---

## 6. Experiments Implemented

### Experiment 1: Standard Prompting vs Few-Shot vs Chain-of-Thought
Compares five prompt-induced reasoning strategies on the GSM8K test split:
1. `zero_shot_direct`: Direct answer prompt with internal solving instruction.
2. `few_shot_3`: Deterministic 3-shot training examples preceding the target problem.
3. `zero_shot_cot`: Step-by-step reasoning prompt ("Let's think step by step").
4. `short_cot`: Concise reasoning constrained to at most 2 deductive steps.
5. `long_cot`: Detailed mathematical reasoning prompt.

### Experiment 2: Controlled Context Scaling
Evaluates the impact of input context expansion on prefill latency, TTFT, generation latency, and energy:
* Context lengths: 512, 1024, 2048, 4096 tokens (optionally 8192).
* Context types: `relevant` (conceptually aligned train problems) vs `distractor` (unrelated train problems).
* Target problem remains fixed while context scales.
* Models exceeding context limits automatically record `status = "skipped", reason = "context_limit"`.

### Experiment 3: BM25 Retrieval-Augmented Generation (Optional Extension)
Evaluates retrieval overhead vs inference:
* Corpus: GSM8K TRAIN split.
* Queries: GSM8K TEST split.
* Top-$k$: 1, 3, 5 documents.
* Isolates `retrieval_latency_ms` and `retrieved_context_tokens` from LLM decode metrics.

---

## 7. Results JSONL Schema & Condition Keys

Each inference row in `results.jsonl` contains:

```json
{
    "run_id": "primary_exp_gsm8k__macbook_air_m1__2026-09-11_12-55-18",
    "experiment_name": "primary_exp_gsm8k",
    "sample_id": "gsm8k_test_0000",
    "model": "qwen3.5:0.8b-mlx",
    "backend": "ollama",
    "strategy": "zero_shot_direct",
    "repetition": 1,
    "condition_key": "fac606a57e06a566beffbc8f",
    "status": "success",
    
    "input_tokens": 134,
    "thinking_tokens": 128,
    "visible_output_tokens": 0,
    "output_tokens": 128,
    "total_tokens": 262,
    "token_count_method": "backend_usage",
    "reasoning_measurement_method": "backend_reported",
    
    "ttft_ms": 59.67,
    "generation_latency_ms": 3429.48,
    "total_latency_ms": 3489.15,
    
    "energy_total_j": 27.8829,
    "energy_prefill_j": null,
    "energy_decode_j": null,
    "energy_overhead_j": null,
    "energy_net_j": null,
    
    "energy_measurement_method": "apple_estimated",
    "energy_quality": "software_estimate",
    "energy_measurement_level": "estimated_system",
    "energy_status": "estimated",
    
    "gold_answer": "18",
    "raw_thinking": "The user wants me to solve...",
    "raw_response": "",
    "raw_output": "<think>\nThe user wants me to solve...\n</think>\n",
    "extracted_answer": "3",
    "answer_parse_success": true,
    "answer_correct": false,
    "error_type": null,
    "error_message": null
}
```

### Reasoning Token Accounting
For native reasoning models (e.g. Qwen 3.5), thinking tokens (`<think>`) and visible output tokens are tracked independently:
$$\text{output\_tokens} = \text{thinking\_tokens} + \text{visible\_output\_tokens}$$

---

## 8. Checkpoint & Resume System

Experiments are resilient to interruptions. If a run halts, restart it using `--resume`:

```bash
python experiments/primary_exp_gsm8k.py \
    --resume results/primary_exp_gsm8k/macbook_air_m1/2026-09-11_12-55-18/
```

The resume manager:
1. Validates that `config.json` and `metadata.json` match the experiment.
2. Identifies completed conditions via SHA-256 `condition_key`.
3. Skips existing conditions in $\mathcal{O}(1)$ time and executes only remaining samples.

---

## 9. Verification & Execution Instructions

### Running Unit Tests (No LLM Required)
```bash
python -m unittest discover -s test -p "test_*.py" -v
```

### Running the 50-Example Validation (Development Mode)
> [!NOTE]
> Running the coding-agent validation does not constitute the full research experiment.

```bash
python experiments/primary_exp_gsm8k.py \
    --device-name "MacBook Air M1" \
    --os "macOS" \
    --cpu "Apple M1" \
    --gpu yes \
    --gpu-count 1 \
    --gpu-name "Apple M1 GPU" \
    --ram-gb 8 \
    --operator ollama \
    --model-format mlx \
    --model "qwen3.5:0.8b-mlx" \
    --eval-size 50 \
    --warmups 1 \
    --repetitions 1 \
    --validation
```

### Manually Running the Full Research Benchmark
To run the full research experiment across all 1,319 GSM8K test samples with 3 repetitions and 3 warm-ups:

```bash
# Experiment 1: Primary Prompting Strategies (Full Benchmark)
python experiments/primary_exp_gsm8k.py \
    --device-name "MacBook Air M1" \
    --operator ollama \
    --model "qwen3.5:0.8b-mlx" \
    --eval-size full \
    --warmups 3 \
    --repetitions 3

# Experiment 2: Controlled Context Scaling (Full Benchmark)
python experiments/context_scaling_gsm8k.py \
    --device-name "MacBook Air M1" \
    --operator ollama \
    --model "qwen3.5:0.8b-mlx" \
    --eval-size full \
    --warmups 3 \
    --repetitions 3

# Experiment 3: BM25 RAG Extension (Full Benchmark)
python experiments/rag_gsm8k.py \
    --device-name "MacBook Air M1" \
    --operator ollama \
    --model "qwen3.5:0.8b-mlx" \
    --eval-size full \
    --warmups 3 \
    --repetitions 3
```

---

## 10. Generating Publication Figures

Run the automated plotting suite on any run directory:

```bash
python visual/plot_results.py --run-dir results/primary_exp_gsm8k/macbook_air_m1/2026-09-11_12-55-18/
```

Generated publication plots:
1. `01_accuracy_by_strategy.png`
2. `02_energy_by_strategy.png`
3. `03_latency_by_strategy.png`
4. `04_output_tokens_by_strategy.png`
5. `05_accuracy_vs_energy.png`
6. `06_accuracy_vs_latency.png`
7. `07_energy_vs_output_tokens.png`
8. `08_context_vs_energy.png` (Context Scaling)
9. `09_context_vs_ttft.png` (Context Scaling)
10. `10_context_vs_accuracy.png` (Context Scaling)
11. `11_pareto_frontier.png`

---

## 11. Extending the Framework

### Adding a New Model
Add the entry into [configs/models.yaml](file:///Users/mdsaibhossain/code/Research_paper/PromptEnergy-Bench/configs/models.yaml):
```yaml
models:
  - name: "mistral-7b-instruct"
    backend: "ollama"
    format: "gguf"
    context_limit: 8192
    supports_thinking: false
    default_parameters:
      temperature: 0.0
      seed: 42
      max_tokens: 1024
```

### Adding a New Backend
1. Subclass `ModelBackend` in `src/backends/base.py`.
2. Implement `load_model`, `generate`, and `get_model_metadata`.
3. Register the backend in `get_backend()` inside `src/backends/__init__.py`.
# CNN-or-ViT
