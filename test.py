import time
import json
import pandas as pd
from datasets import load_dataset
from codecarbon import OfflineEmissionsTracker
import ollama

# ==========================================
# 1. Configuration & Dataset Setup
# ==========================================
MODEL_NAME = "qwen3:1.7b"
# Using OfflineEmissionsTracker for local offline CPU tracking (specify your ISO code, e.g., 'USA' or 'DEU')
COUNTRY_ISO = "USA" 

print("Loading GSM8K dataset from Hugging Face...")
# Load a single sample from GSM8K (Section 5 of paper)
dataset = load_dataset("openai/gsm8k", "main", split="test")
sample = dataset[0]  # Single sample experiment

question = sample["question"]
ground_truth = sample["answer"]

print(f"\nTarget Question: {question}\n")

# ==========================================
# 2. Prompting Strategy Definitions (Section 4)
# ==========================================
# Static RAG snippet simulation as defined in paper methodology
rag_context = (
    "Context: To solve word math problems, break the problem into steps. "
    "Identify known variables, formulate basic equations, perform elementary arithmetic, "
    "and calculate the cumulative totals carefully."
)

prompts = {
    "Standard": question,
    "Chain_of_Thought": f"{question}\n\nLet's think step by step to find the correct answer.",
    "RAG_Simulated": f"{rag_context}\n\nQuestion: {question}\n\nProvide the final calculated answer."
}

# ==========================================
# 3. Experiment Execution Loop
# ==========================================
results = []

for strategy_name, prompt_text in prompts.items():
    print(f"--- Running Strategy: {strategy_name} ---")
    
    # Initialize CodeCarbon local tracker (Section 4 formula integration)
    tracker = OfflineEmissionsTracker(
        country_iso_code=COUNTRY_ISO,
        project_name=f"exp_{strategy_name}",
        measure_power_secs=0.1,
        save_to_file=False
    )
    
    # Start energy tracking & latency measurement
    tracker.start()
    start_time = time.time()
    
    # Local inference via Ollama (Section 6)
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt_text
    )
    
    end_time = time.time()
    emissions = tracker.stop()  # Returns emissions in kg CO2eq
    
    # Calculate Latency
    latency_seconds = end_time - start_time
    
    # Extract hardware energy consumption metrics captured by CodeCarbon
    data = tracker.final_emissions_data
    cpu_energy_kwh = getattr(data, 'cpu_energy', 0.0)
    ram_energy_kwh = getattr(data, 'ram_energy', 0.0)
    total_energy_kwh = cpu_energy_kwh + ram_energy_kwh

    # Response token generation details
    generated_text = response.get("response", "")
    eval_count = response.get("eval_count", 0)  # Number of output tokens
    
    results.append({
        "Strategy": strategy_name,
        "Latency (s)": round(latency_seconds, 3),
        "Tokens Generated": eval_count,
        "Tokens/Sec": round(eval_count / latency_seconds, 2) if latency_seconds > 0 else 0,
        "Total Energy (kWh)": f"{total_energy_kwh:.8f}",
        "Carbon Emissions (kgCO2eq)": f"{emissions:.8f}",
        "Output": generated_text[:120].replace("\n", " ") + "..."
    })

# ==========================================
# 4. Results Formatting & Export
# ==========================================
df_results = pd.DataFrame(results)

# Save the DataFrame to a CSV file in your current directory
csv_filename = "gsm8k_energy_experiment_results.csv"
df_results.to_csv(csv_filename, index=False)

print("\n" + "="*80)
print("EXPERIMENT RESULTS SUMMARY")
print("="*80)
print(df_results[["Strategy", "Latency (s)", "Tokens Generated", "Tokens/Sec", "Total Energy (kWh)", "Carbon Emissions (kgCO2eq)"]].to_string(index=False))

print(f"\n[+] Results successfully saved to: {csv_filename}")

print("\nSample Output Texts:")
for res in results:
    print(f"\n[{res['Strategy']}]\n{res['Output']}")