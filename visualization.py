import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load results from saved file
data_path = Path("subset_simulation_results.npy")
results_by_strategy = np.load(data_path, allow_pickle=True).item()

# Prepare DataFrame for plotting
df_list = []
for strategy, values in results_by_strategy.items():
    for k_val, mean_p, std_p in values:
        df_list.append({
            "Strategy": strategy,
            "k": k_val,
            "Mean Probability": mean_p,
            "Std Dev": std_p
        })

df = pd.DataFrame(df_list)

# Plot
plt.figure(figsize=(12, 6))
for strategy in df["Strategy"].unique():
    subset = df[df["Strategy"] == strategy]
    plt.errorbar(subset["k"], subset["Mean Probability"], yerr=subset["Std Dev"],
                 label=strategy, marker='o', capsize=4, linewidth=2)

plt.xlabel("Wind Intensity Parameter k")
plt.ylabel("Estimated Failure Probability")
plt.yscale("log")
plt.title("Failure Probability vs. Wind Intensity (Subset Simulation)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("subset_simulation_plot.png", dpi=300)
print("Plot saved as subset_simulation_plot.png")

# Rebuild dataframe
df_list = []
for strategy, values in results_by_strategy.items():
    for k_val, mean_p, std_p in values:
        df_list.append({
            "Strategy": strategy,
            "k": k_val,
            "Mean Probability": mean_p,
            "Std Dev": std_p
        })

df = pd.DataFrame(df_list)
df["CV"] = df["Std Dev"] / df["Mean Probability"]

# Pivot for clearer tabular display
df_table = df.pivot_table(index=["k"], columns="Strategy", values=["Mean Probability", "Std Dev", "CV"])
df_table_rounded = df_table.round(5)

print("\nSummary Table of Results:")
print(df_table_rounded)
df_table_rounded.to_csv("subset_simulation_summary.csv")