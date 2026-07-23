import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
csv_filename = "gsm8k_energy_experiment_results.csv"
print(f"Loading data from {csv_filename}...")
df = pd.read_csv(csv_filename)

# 2. Set up the visual style
sns.set_theme(style="whitegrid")
# Define a custom color palette for the three strategies
palette = {"Standard": "#4C72B0", "Chain_of_Thought": "#DD8452", "RAG_Simulated": "#55A868"}

# 3. Create a 2x2 grid of subplots for the dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LLM Prompting Strategy: Performance & Energy Analysis', fontsize=18, fontweight='bold', y=0.98)

# --- Chart 1: Latency (Top Left) ---
sns.barplot(x='Strategy', y='Latency (s)', data=df, ax=axes[0, 0], palette=palette)
axes[0, 0].set_title('Inference Latency (Lower is Better)', fontsize=14)
axes[0, 0].set_ylabel('Seconds')

# --- Chart 2: Total Energy (Top Right) ---
sns.barplot(x='Strategy', y='Total Energy (kWh)', data=df, ax=axes[0, 1], palette=palette)
axes[0, 1].set_title('Total Energy Consumption (Lower is Better)', fontsize=14)
axes[0, 1].set_ylabel('Energy (kWh)')

# --- Chart 3: Tokens Generated (Bottom Left) ---
sns.barplot(x='Strategy', y='Tokens Generated', data=df, ax=axes[1, 0], palette=palette)
axes[1, 0].set_title('Total Tokens Generated', fontsize=14)
axes[1, 0].set_ylabel('Token Count')

# --- Chart 4: Generation Efficiency (Bottom Right) ---
sns.barplot(x='Strategy', y='Tokens/Sec', data=df, ax=axes[1, 1], palette=palette)
axes[1, 1].set_title('Generation Speed (Higher is Better)', fontsize=14)
axes[1, 1].set_ylabel('Tokens per Second')

# 4. Clean up layout and rotate x-axis labels for better readability
for ax in axes.flat:
    ax.set_xlabel('') # Remove redundant 'Strategy' x-label
    ax.tick_params(axis='x', labelrotation=15)
    
    # Add data labels on top of the bars for clarity
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4g', padding=3)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to fit the main title

# 5. Save the figure as an image and display it
output_image = "experiment_dashboard.png"
plt.savefig(output_image, dpi=300)
print(f"Diagram successfully saved as {output_image}")

# Show the interactive plot window
plt.show()