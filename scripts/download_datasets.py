import os
from datasets import load_dataset


output_dir = "datasets/gsm8k"


os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset("openai/gsm8k", "main")

dataset.save_to_disk(output_dir)
for split in dataset.keys():
    output_path = os.path.join(output_dir, f"{split}.jsonl")
    dataset[split].to_json(output_path, orient="records", lines=True)
    
print(f"Dataset successfully saved to: ./{output_dir}/")