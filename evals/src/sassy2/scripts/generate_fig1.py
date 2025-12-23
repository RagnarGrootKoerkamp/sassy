#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FormatStrFormatter
import numpy as np
import os

plt.rcParams.update({
    'font.size': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 9,
    'axes.titlesize': 11,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.25,
    'lines.linewidth': 1.8,
    'lines.markersize': 3.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
})


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.27, 4.09))

csv_files = ["evals/src/sassy2/output/scaling_benchmark_results.csv"]
scaling_csv = None
for candidate in csv_files:
    if os.path.exists(candidate):
        scaling_csv = candidate
        break

if scaling_csv is None:
    print("Warning: scaling_benchmark_results.csv not found, skipping Panel A")
else:
    df_scaling = pd.read_csv(scaling_csv)
    print(f"Loaded scaling data from {scaling_csv} with {len(df_scaling)} rows")

    df_scaling = df_scaling[df_scaling["k"] < df_scaling["query_len"]]
    print(f"After filtering: {len(df_scaling)} rows")

    sassy_search_color = "#0A9396"  # teal for sassy1 (search)
    sassy_tiling_color = "#9B5DE5"  # purple for sassy2 (tiling)
    edlib_color = "#E76F51"         # coral for edlib

    line_style_map = {
        0: ("-", 1.8),
        3: ("--", 1.8),
        4: ("-.", 1.8),
        10: (":", 2.0)
    }

    k_values = sorted(df_scaling["k"].unique())
    query_lengths = sorted(df_scaling["query_len"].unique())

    print(f"Scaling - K values: {k_values}")
    print(f"Scaling - Query lengths: {query_lengths}")

    for k in k_values:
        for query_len in query_lengths:
            mask = (df_scaling["k"] == k) & (df_scaling["query_len"] == query_len)
            sub_df = df_scaling[mask]

            if sub_df.empty:
                continue

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
                    color = "gray"

                linestyle, linewidth = line_style_map.get(k, ("-", 1.2))

                # Create label
                if algorithm == "sassy_search":
                    algo_name = "Sassy1"
                elif algorithm == "sassy_tiling":
                    algo_name = "Sassy2"
                elif algorithm == "edlib":
                    algo_name = "Edlib"
                else:
                    algo_name = algorithm.title()

                label = f"{algo_name} (q={query_len}, k={k})"

                ax1.plot(
                    algo_data["target_len"],
                    algo_data["throughput_gbps"],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker='o' if algorithm == "edlib" else None,
                    markersize=3.5,
                    markeredgewidth=0.6,
                    label=label,
                    alpha=0.9
                )

    
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)

    ax1.grid(True, which="major", linewidth=0.4, alpha=0.5, color='#CCCCCC')
    ax1.grid(True, which="minor", linewidth=0.2, alpha=0.3, color='#E0E0E0')
    ax1.set_axisbelow(True)
    
    ax1.set_xlabel("Target length (bp)", fontsize=9)
    ax1.set_ylabel("Throughput (GB/s)", fontsize=9)
    ax1.set_title("A", fontweight='bold', loc='left', fontsize=11, pad=10)

    ax1.xaxis.set_major_locator(LogLocator(base=2.0, subs=[1.0], numticks=15))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax1.yaxis.set_major_locator(LogLocator(base=2.0, subs=[1.0], numticks=15))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


    for spine in ax1.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#888888')

    # Legend for Panel A
    legend_handles_a = [
        Line2D([0], [0], color=sassy_search_color, lw=1.8, label="Sassy1"),
        Line2D([0], [0], color=sassy_tiling_color, lw=1.8, label="Sassy2"),
        Line2D([0], [0], color=edlib_color, lw=1.8, marker='o', markersize=3.5, markeredgewidth=0.6, label="Edlib"),
    ]
    
    # Add k values
    for k, (linestyle, linewidth) in line_style_map.items():
        if k in k_values:
            legend_handles_a.append(
                Line2D([0], [0], color="#555555", linestyle=linestyle, 
                       linewidth=linewidth, label=f"k={k}")
            )
    
    ax1.legend(
        handles=legend_handles_a,
        loc='lower right',
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=6
    )


throughput_csv = "evals/src/sassy2/output/pattern_throughput_results.csv"
if not os.path.exists(throughput_csv):
    print("Warning: pattern_throughput_results.csv not found, skipping Panel B")
else:
    df_throughput = pd.read_csv(throughput_csv)

    tools = ['search', 'tiling', 'edlib']
    tool_colors = {'search': '#0A9396', 'tiling': '#9B5DE5', 'edlib': '#E76F51'}
    tool_markers = {'search': 'D', 'tiling': 's', 'edlib': 'o'}
    tool_labels = {'search': 'Sassy1', 'tiling': 'Sassy2', 'edlib': 'Edlib'}

    lines1 = []
    labels1 = []
    for tool in tools:
        throughput_col = f"{tool}_throughput_gbps"
        if throughput_col in df_throughput.columns:
            line, = ax2.plot(
                df_throughput['num_queries'], df_throughput[throughput_col],
                color=tool_colors[tool],
                marker=tool_markers[tool],
                linestyle='--',  # Dashed for k=3 consistency
                linewidth=1.8,
                markersize=4.5,
                markeredgewidth=0.6,
                label=f"{tool_labels[tool]} (k=3)",
                alpha=0.9
            )
            lines1.append(line)
            labels1.append(f"{tool_labels[tool]} (k=3)")

    ax2.set_xlabel("Number of patterns", fontsize=9)
    ax2.set_ylabel("Throughput (GB/s)", fontsize=9)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_title("B", fontweight='bold', loc='left', fontsize=11, pad=10)

    # Show all num_queries values as x-ticks
    if not df_throughput.empty:
        num_queries = df_throughput['num_queries'].tolist()
        ax2.set_xticks(num_queries)
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.set_xticklabels([str(n) for n in num_queries])


        throughput_cols = [f"{tool}_throughput_gbps" for tool in tools if f"{tool}_throughput_gbps" in df_throughput.columns]
        if throughput_cols:
            min_throughput = df_throughput[throughput_cols].min().min()
            max_throughput = df_throughput[throughput_cols].max().max()
            # Set y-ticks as powers of 2
            ax2.yaxis.set_major_locator(LogLocator(base=2.0, subs=[1.0], numticks=15))
            ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax2.set_ylim(min_throughput * 0.8, max_throughput * 1.1)

    ax2.grid(True, which='major', linewidth=0.4, alpha=0.5, color='#CCCCCC')
    ax2.grid(True, which='minor', linewidth=0.2, alpha=0.3, color='#E0E0E0')
    ax2.set_axisbelow(True)

    ax2_twin = ax2.twinx()
    lines2 = []
    labels2 = []
    for tool in tools:
        ipc_col = f"{tool}_ipc"
        if ipc_col in df_throughput.columns:
            line, = ax2_twin.plot(
                df_throughput['num_queries'], df_throughput[ipc_col],
                color=tool_colors[tool],
                linestyle=':',
                linewidth=1.5,
                markersize=0,
                alpha=0.35,
                label=f"{tool_labels[tool]} IPC"
            )
            lines2.append(line)
            labels2.append(f"{tool_labels[tool]} IPC")

    ax2_twin.set_ylabel("IPC", fontsize=9, color='#666666')
    ax2_twin.tick_params(axis='y', labelcolor='#666666', labelsize=7, width=0.6)

    if not df_throughput.empty:
        ipc_cols = [f"{tool}_ipc" for tool in tools if f"{tool}_ipc" in df_throughput.columns]
        if ipc_cols:
            max_ipc = df_throughput[ipc_cols].max().max()
            ax2_twin.set_ylim(0, max_ipc * 1.1)


    for spine in ax2.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#888888')
    for spine in ax2_twin.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#888888')

    # Legend panel B
    legend_handles_b = []
    
    # Add throughput lines
    for tool in tools:
        legend_handles_b.append(
            Line2D([0], [0], color=tool_colors[tool], linestyle='--', 
                   marker=tool_markers[tool], markersize=3.5, markeredgewidth=0.6,
                   lw=1.8, label=f"{tool_labels[tool]} (k=3)")
        )
    
    # Add IPC lines
    legend_handles_b.append(
        Line2D([0], [0], color='#666666', linestyle=':', lw=1.5, label="IPC (dotted)")
    )
    
    ax2.legend(
        handles=legend_handles_b,
        loc='upper left',
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        handlelength=1.5,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=6
    )

plt.tight_layout()

plt.savefig('evals/src/sassy2/figs/figure1_scaling_throughput.pdf', dpi=300, bbox_inches='tight')
plt.savefig('evals/src/sassy2/figs/figure1_scaling_throughput.png', dpi=300, bbox_inches='tight')
plt.savefig('evals/src/sassy2/figs/figure1_scaling_throughput.svg', dpi=300, bbox_inches='tight')

print("Saved evals/src/sassy2/figs/figure1_scaling_throughput.pdf and .png")

plt.show()