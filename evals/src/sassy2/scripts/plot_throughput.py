import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Set publication-quality matplotlib parameters
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
})

# Read CSV
csv_path = "pattern_throughput_results.csv"  
df = pd.read_csv(csv_path)

# Tools and their colors/markers
tools = ['search', 'tiling', 'edlib']
tool_colors = {'search': '#006400', 'tiling': '#9B30FF', 'edlib': '#1F78B4'}
tool_markers = {'search': 'D', 'tiling': 's', 'edlib': 'o'}
tool_labels = {'search': 'sassy1', 'tiling': 'sassy2', 'edlib': 'edlib'}

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(9, 5))

# Plot throughput on left y-axis
for tool in tools:
    throughput_col = f"{tool}_throughput_gbps"
    ax1.plot(
        df['num_queries'], df[throughput_col],
        color=tool_colors[tool],
        marker=tool_markers[tool],
        linestyle='-',
        linewidth=2,
        markersize=6,
        label=tool_labels[tool]
    )

ax1.set_xlabel("Number of patterns")
ax1.set_ylabel("Throughput (GB/s)")
ax1.set_xscale('log')

# Show all num_queries values as x-ticks
num_queries = df['num_queries'].tolist()
print(num_queries)
ax1.set_xticks(num_queries)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Show as plain integers
ax1.set_xticklabels([str(n) for n in num_queries])

# Set throughput ticks nicely
max_throughput = df[[f"{tool}_throughput_gbps" for tool in tools]].max().max()
tick_step = 0.5
ax1.set_yticks(np.arange(0, max_throughput + tick_step, tick_step))
ax1.set_ylim(0, max_throughput * 1.05)

ax1.grid(True, which='major', linestyle='--', alpha=0.3)
ax1.grid(True, which='minor', linestyle=':', alpha=0.15)

# Secondary y-axis for IPC
ax2 = ax1.twinx()
for tool in tools:
    ipc_col = f"{tool}_ipc"
    ax2.plot(
        df['num_queries'], df[ipc_col],
        color=tool_colors[tool],
        linestyle=':',
        linewidth=2,
        markersize=0,
        alpha=0.3,  # reduce opacity
        label=f"{tool_labels[tool]} IPC"
    )
ax2.set_ylabel("IPC")
ax2.set_ylim(0, df[[f"{tool}_ipc" for tool in tools]].max().max() * 1.1)

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

# plt.title("Throughput and IPC vs Number of Patterns")
plt.tight_layout()
plt.show()