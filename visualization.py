import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load results from saved files
# data_path_1 = Path("subset_simulation_results_1.npy")
# data_path_3 = Path("subset_simulation_results_3.npy")
data_path_15 = Path("subset_simulation_results_15.npy")

# results_1 = np.load(data_path_1, allow_pickle=True).item()
# results_3 = np.load(data_path_3, allow_pickle=True).item()
results_15 = np.load(data_path_15, allow_pickle=True).item()

k_values_15 = set()
for strategy, values in results_15.items():
    for k_val, mean_p, std_p in values:
        k_values_15.add(k_val)
print(f"K values in results_15: {sorted(k_values_15)}")
# Prepare DataFrame for plotting
df_list_15 = []
for strategy, values in results_15.items():
    for k_val, mean_p, std_p in values:
        df_list_15.append({
            "Strategy": strategy,
            "k": k_val,
            "Mean Probability": mean_p,
            "Std Dev": std_p
        })
df_15 = pd.DataFrame(df_list_15)
# Get available strategies from the actual data
available_strategies_15 = df_15["Strategy"].unique().tolist()
print(f"Strategies available for plotting (15): {available_strategies_15}")
strategy_configs_15 = [
    "No adaptation", "Adaptation u_3", "Double measure", "Double measure Bayesian"
]
# Filter to only include strategies that exist in the data
strategies_to_plot_15 = [s for s in strategy_configs_15 if s in available_strategies_15]
print(f"Strategies to be plotted (15): {strategies_to_plot_15}")
# Plot for 15 test time
plt.figure(figsize=(12, 5))
for strategy in strategies_to_plot_15:
    subset = df_15[df_15["Strategy"] == strategy]
    plt.errorbar(subset["k"], subset["Mean Probability"], yerr=subset["Std Dev"],
                 label=strategy, marker='o', capsize=4, linewidth=2)
plt.xlabel("Wind Intensity Parameter k")
plt.ylabel("Estimated Failure Probability")
plt.yscale("log")
plt.title("Failure Probability vs. Wind Intensity (15 Test Time)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# Set better x-axis limits and ticks
k_min_15 = df_15["k"].min()
k_max_15 = df_15["k"].max()
plt.xlim(k_min_15 - 0.01, k_max_15 + 0.01)  # Add small padding
plt.xticks(sorted(df_15["k"].unique()))  # Show all k values as ticks
plt.legend()
plt.tight_layout()
plt.savefig("subset_simulation_plot_15.png", dpi=300)
print("Plot saved as subset_simulation_plot_15.png")

# # Merge results: use results_3 to replace corresponding k values in results_1
# results_by_strategy = results_1.copy()

# # Get k values from results_3 for matching
# k_values_3 = set()
# for strategy, values in results_3.items():
#     for k_val, mean_p, std_p in values:
#         k_values_3.add(k_val)

# print(f"K values in results_3: {sorted(k_values_3)}")

# # Replace values from results_3 where both strategy and k value match
# for strategy, values_3 in results_3.items():
#     if strategy in results_by_strategy:
#         print(f"Processing strategy: {strategy}")
        
#         # Create a dictionary for quick lookup of results_3 data by k value
#         results_3_dict = {k_val: (mean_p, std_p) for k_val, mean_p, std_p in values_3}
        
#         # Replace matching k values in results_1
#         new_values = []
#         for k_val, mean_p, std_p in results_by_strategy[strategy]:
#             if k_val in results_3_dict:
#                 # Replace with results_3 data
#                 new_mean_p, new_std_p = results_3_dict[k_val]
#                 new_values.append((k_val, new_mean_p, new_std_p))
#                 print(f"  Replaced k={k_val}: {mean_p:.6f}±{std_p:.6f} -> {new_mean_p:.6f}±{new_std_p:.6f}")
#             else:
#                 # Keep original results_1 data
#                 new_values.append((k_val, mean_p, std_p))
#                 print(f"  Kept k={k_val}: {mean_p:.6f}±{std_p:.6f}")
        
#         results_by_strategy[strategy] = new_values
#     else:
#         print(f"Adding new strategy {strategy} from subset_simulation_results_3.npy")
#         results_by_strategy[strategy] = values_3

# print(f"Available strategies: {list(results_by_strategy.keys())}")

# # Prepare DataFrame for plotting
# df_list = []
# for strategy, values in results_by_strategy.items():
#     for k_val, mean_p, std_p in values:
#         df_list.append({
#             "Strategy": strategy,
#             "k": k_val,
#             "Mean Probability": mean_p,
#             "Std Dev": std_p
#         })

# df = pd.DataFrame(df_list)

# # Get available strategies from the actual data
# available_strategies = df["Strategy"].unique().tolist()
# print(f"Strategies available for plotting: {available_strategies}")

# strategy_configs = [
#     "No adaptation", "Adaptation u_3", "Double measure", "Double measure Bayesian"
# ]

# # Filter to only include strategies that exist in the data
# strategies_to_plot = [s for s in strategy_configs if s in available_strategies]
# print(f"Strategies to be plotted: {strategies_to_plot}")

# # Plot
# plt.figure(figsize=(12, 5))
# for strategy in strategies_to_plot:
#     subset = df[df["Strategy"] == strategy]
#     plt.errorbar(subset["k"], subset["Mean Probability"], yerr=subset["Std Dev"],
#                  label=strategy, marker='o', capsize=4, linewidth=2)

# plt.xlabel("Wind Intensity Parameter k")
# plt.ylabel("Estimated Failure Probability")
# plt.yscale("log")
# plt.title("Failure Probability vs. Wind Intensity (Merged Results)")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# # Set better x-axis limits and ticks
# k_min = df["k"].min()
# k_max = df["k"].max()
# plt.xlim(k_min - 0.01, k_max + 0.01)  # Add small padding
# plt.xticks(sorted(df["k"].unique()))  # Show all k values as ticks

# plt.legend()
# plt.tight_layout()
# plt.savefig("merged_subset_simulation_plot.png", dpi=300)
# print("Plot saved as merged_subset_simulation_plot.png")

# # Rebuild dataframe
# df_list = []
# for strategy, values in results_by_strategy.items():
#     for k_val, mean_p, std_p in values:
#         df_list.append({
#             "Strategy": strategy,
#             "k": k_val,
#             "Mean Probability": mean_p,
#             "Std Dev": std_p
#         })

# df = pd.DataFrame(df_list)
# df["CV"] = df["Std Dev"] / df["Mean Probability"]

# # Pivot for clearer tabular display
# df_table = df.pivot_table(index=["k"], columns="Strategy", values=["Mean Probability", "Std Dev", "CV"])
# df_table_rounded = df_table.round(5)

# print("\nSummary Table of Results:")
# print(df_table_rounded)
# df_table_rounded.to_csv("merged_subset_simulation_summary.csv")
# print("Summary table saved as merged_subset_simulation_summary.csv")