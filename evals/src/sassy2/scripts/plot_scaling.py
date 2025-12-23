#!/usr/bin/env python3
# Plot scaling benchmark results: throughput vs target length
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FormatStrFormatter
import os

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.5,
        "lines.markersize": 2,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
    }
)

# Read the scaling benchmark results
csv_files = ["scaling_benchmark_results.csv", "../scaling_benchmark_results.csv"]
csv_file = None
for candidate in csv_files:
    if os.path.exists(candidate):
        csv_file = candidate
        break

if csv_file is None:
    print("Error: scaling_benchmark_results.csv not found in current or parent directory!")
    exit(1)

df = pd.read_csv(csv_file)
print(f"Loaded {csv_file} with {len(df)} rows")

# Filter out invalid data (where k >= query_len)
df = df[df["k"] < df["query_len"]]
print(f"After filtering: {len(df)} rows")

sassy_search_color = "#006400"  # dark green for sassy1 (search)
sassy_tiling_color = "#9B30FF"  # purple for sassy2 (tiling)
edlib_color = "#1F78B4"         # blue for edlib

# Line styles based on k values (similar to original plot_throughput.py)
line_style_map = {
    0: ("-", 2),
    3: ("--", 2),
    4: ("-.", 2),
    10: (":", 2)
}

# === Start plot ===
fig, ax = plt.subplots(figsize=(6, 4))

# Get unique values
k_values = sorted(df["k"].unique())
query_lengths = sorted(df["query_len"].unique())

print(f"K values: {k_values}")
print(f"Query lengths: {query_lengths}")

for k in k_values:
    for query_len in query_lengths:
        # Filter data for this combination
        mask = (df["k"] == k) & (df["query_len"] == query_len)
        sub_df = df[mask]

        if sub_df.empty:
            continue

        # Plot each algorithm (only sassy_search, sassy_tiling, edlib)
        for algorithm in ["sassy_search", "sassy_tiling", "edlib"]:
            if algorithm not in sub_df["algorithm"].values:
                continue

            algo_data = sub_df[sub_df["algorithm"] == algorithm].copy()
            algo_data = algo_data.sort_values("target_len")

            if algo_data.empty:
                continue

            if algorithm == "sassy_search":
                color = sassy_search_color
            elif algorithm == "sassy_tiling":
                color = sassy_tiling_color
            elif algorithm == "edlib":
                color = edlib_color
            else:
                color = "gray"  # fallback
            linestyle, linewidth = line_style_map.get(k, ("-", 2))

            # Create a more readable label
            if algorithm == "sassy_search":
                algo_name = "Sassy1"
            elif algorithm == "sassy_tiling":
                algo_name = "Sassy2"
            elif algorithm == "edlib":
                algo_name = "Edlib"
            else:
                algo_name = algorithm.title()

            label = f"{algo_name} (q={query_len}, k={k})"

            ax.plot(
                algo_data["target_len"],
                algo_data["throughput_gbps"],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                marker='o' if algorithm == "edlib" else None,
                markersize=3,
                label=label
            )

# Set log scales
ax.set_xscale("log")
ax.set_yscale("log")

# Grid, labels
ax.grid(True, which="major", linewidth=0.3, alpha=0.7)

ax.set_xlabel("Target length (bp)")
ax.set_ylabel("Search throughput (GB/s)")

# Formatters
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Group by algorithm type and k value
legend_handles = []
legend_labels = []

# Algorithm types
legend_handles.extend([
    Line2D([0], [0], color=sassy_search_color, lw=2, label="Sassy1 (Search)"),
    Line2D([0], [0], color=sassy_tiling_color, lw=2, label="Sassy2 (Tiling)"),
    Line2D([0], [0], color=edlib_color, lw=2, marker='o', markersize=4, label="Edlib"),
])
legend_labels.extend(["Sassy1 (Search)", "Sassy2 (Tiling)", "Edlib"])

# K values with different line styles
for k, (linestyle, linewidth) in line_style_map.items():
    if k in k_values:
        legend_handles.append(Line2D([0], [0], color="gray", linestyle=linestyle, linewidth=linewidth, label=f"k={k}"))
        legend_labels.append(f"k={k}")

ax.legend(
    handles=legend_handles,
    labels=legend_labels,
    frameon=True,
    fancybox=True,
    shadow=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    handlelength=2.0,
    handletextpad=0.4,
    labelspacing=0.2,
    ncol=1,
)

# Create output directory

plt.tight_layout()
plt.show()


print("Scaling throughput plots saved successfully to figs/ directory")
