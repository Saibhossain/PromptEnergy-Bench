import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("gsm8k_energy_experiment_results.csv")

# Set theme and palette
sns.set_theme(style="white")
colors = {"Standard": "#e74c3c", "Chain_of_Thought": "#2ecc71", "RAG_Simulated": "#3498db"}

# Create a master figure with a custom grid (1 Radar Plot + 2 Subplots)
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Advanced LLM Sustainability & Efficiency Analysis', fontsize=18, fontweight='bold', y=0.98)

# ----------------------------------------------------
# PLOT 1: RADAR / SPIDER CHART (Normalized Comparison)
# ----------------------------------------------------
ax_radar = fig.add_subplot(121, polar=True)

# Categories for Radar
categories = ['Latency\n(Lower Best)', 'Energy\n(Lower Best)', 'Carbon\n(Lower Best)', 'Tokens/Sec\n(Higher Best)', 'Output Size\n(Tokens)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Close the circle loop

# Normalize metrics [0, 1] for relative comparison
def min_max_norm(series, invert=False):
    norm = (series - series.min()) / (series.max() - series.min() + 1e-9)
    return 1 - norm if invert else norm

df_norm = pd.DataFrame()
# Invert Latency, Energy, Carbon so that 1.0 is always the "Best"
df_norm['Latency'] = min_max_norm(df['Latency (s)'], invert=True)
df_norm['Energy'] = min_max_norm(df['Total Energy (kWh)'], invert=True)
df_norm['Carbon'] = min_max_norm(df['Carbon Emissions (kgCO2eq)'], invert=True)
df_norm['Tokens/Sec'] = min_max_norm(df['Tokens/Sec'], invert=False)
df_norm['Output Size'] = min_max_norm(df['Tokens Generated'], invert=False)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax_radar.set_ylim(0, 1)

for idx, row in df.iterrows():
    strat = row['Strategy']
    values = df_norm.iloc[idx].values.flatten().tolist()
    values += values[:1] # Close the polygon
    
    ax_radar.plot(angles, values, linewidth=2.5, linestyle='solid', label=strat, color=colors[strat])
    ax_radar.fill(angles, values, color=colors[strat], alpha=0.15)

ax_radar.set_title("Multi-Metric Profile Normalized [0 = Worst, 1 = Best]", fontsize=12, pad=20, fontweight='bold')
ax_radar.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.1))

# ----------------------------------------------------
# PLOT 2: ENERGY VS LATENCY TRADE-OFF (Bubble Plot)
# ----------------------------------------------------
ax_bubble = fig.add_subplot(222)

# Bubble size proportional to tokens generated
bubble_sizes = (df['Tokens Generated'] / df['Tokens Generated'].max()) * 1000

scatter = ax_bubble.scatter(
    df['Latency (s)'], 
    df['Total Energy (kWh)'], 
    s=bubble_sizes, 
    c=[colors[s] for s in df['Strategy']], 
    alpha=0.7, 
    edgecolors="black", 
    linewidth=1.5
)

# Annotate points with strategy names
for idx, row in df.iterrows():
    ax_bubble.annotate(
        f"{row['Strategy']}\n({row['Tokens Generated']} tokens)",
        (row['Latency (s)'], row['Total Energy (kWh)']),
        xytext=(10, -10), textcoords='offset points',
        fontsize=9, fontweight='bold'
    )

ax_bubble.set_title("Energy vs. Latency Pareto Trade-off", fontsize=12, fontweight='bold')
ax_bubble.set_xlabel("Latency (Seconds)", fontweight='bold')
ax_bubble.set_ylabel("Total Energy (kWh)", fontweight='bold')
ax_bubble.grid(True, linestyle="--", alpha=0.5)

# ----------------------------------------------------
# PLOT 3: SAVINGS RELATIVE TO STANDARD BASELINE (%)
# ----------------------------------------------------
ax_savings = fig.add_subplot(224)

# Calculate % savings compared to 'Standard'
std_row = df[df['Strategy'] == 'Standard'].iloc[0]
df_savings = df[df['Strategy'] != 'Standard'].copy()

df_savings['Energy Saved (%)'] = ((std_row['Total Energy (kWh)'] - df_savings['Total Energy (kWh)']) / std_row['Total Energy (kWh)']) * 100
df_savings['Time Saved (%)'] = ((std_row['Latency (s)'] - df_savings['Latency (s)']) / std_row['Latency (s)']) * 100

df_melted = df_savings.melt(id_vars=['Strategy'], value_vars=['Energy Saved (%)', 'Time Saved (%)'], 
                            var_name='Metric', value_name='Percentage')

sns.barplot(x='Strategy', y='Percentage', hue='Metric', data=df_melted, ax=ax_savings, palette='Set2')

ax_savings.set_title("% Efficiency Gain relative to Standard Baseline", fontsize=12, fontweight='bold')
ax_savings.set_ylabel("% Savings (Higher is Better)", fontweight='bold')
ax_savings.set_xlabel("")
ax_savings.set_ylim(0, 100)

for container in ax_savings.containers:
    ax_savings.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("advanced_energy_analysis.png", dpi=300)
print("Saved advanced static diagram to 'advanced_energy_analysis.png'")
plt.show()