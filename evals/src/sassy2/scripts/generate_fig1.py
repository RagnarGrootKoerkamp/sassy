#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FormatStrFormatter
import numpy as np
import os


plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.25,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.27, 3.27), sharey=True)


scaling_csv = "evals/src/sassy2/output/pattern_scaling_results.csv"
sassy_search_color = "#0A9396"  # teal
sassy_tiling_color = "#7F5DE5"  # slightly softer purple
edlib_color = "#E76F51"  # coral

if os.path.exists(scaling_csv):
    df_scaling = pd.read_csv(scaling_csv)
    # CSV columns: num_queries, target_len, query_len, k, search_*, tiling_*, edlib_*
    df_scaling = df_scaling[df_scaling["k"] < df_scaling["query_len"]]

    # median aggregation over runs (if any)
    df_scaling = df_scaling.groupby(
        ["num_queries", "target_len", "query_len", "k"], as_index=False
    ).median(numeric_only=True)

    line_style_map = {0: ("-", 1.2), 3: ("--", 1.2), 4: ("-.", 1.2), 10: (":", 1.4)}

    k_values = sorted(df_scaling["k"].unique())
    query_lengths = sorted(df_scaling["query_len"].unique())

    tool_config = [
        ("search", "search_throughput_gbps", sassy_search_color, "D"),
        ("tiling", "tiling_throughput_gbps", sassy_tiling_color, "s"),
        ("edlib", "edlib_throughput_gbps", edlib_color, "o"),
    ]

    for k in k_values:
        for query_len in query_lengths:
            sub_df = df_scaling[
                (df_scaling["k"] == k) & (df_scaling["query_len"] == query_len)
            ].sort_values("target_len")
            if sub_df.empty:
                continue

            linestyle, linewidth = line_style_map.get(k, ("-", 1.2))
            for _name, col, color, marker in tool_config:
                if col not in sub_df.columns:
                    continue
                ax1.plot(
                    sub_df["target_len"],
                    sub_df[col],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3,
                    markeredgewidth=0.5,
                    alpha=0.9,
                )

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)

    ax1.set_xlabel("Text length (bp)")
    ax1.set_ylabel("Total throughput (Gbp/s)")
    ax1.set_title("A", fontweight="bold", loc="left", pad=10)

    ax1.xaxis.set_major_locator(LogLocator(base=2.0))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    # ax1.xaxis.set_minor_locator(
    #     LogLocator(base=2.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    # )
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())

    ax1.yaxis.set_major_locator(LogLocator(base=2.0, numticks=100))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    # ax1.yaxis.set_minor_locator(
    #     LogLocator(base=2.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    # )
    # ax1.yaxis.set_minor_formatter(plt.NullFormatter())

    ax1.grid(True, which="major", linewidth=0.4, alpha=0.5, color="#DCDCDC")
    ax1.grid(True, which="minor", linewidth=0.3, alpha=0.45, color="#E8E8E8")
    ax1.set_axisbelow(True)

    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#333333")

    # Legend
    legend_handles_a = [
        Line2D(
            [0],
            [0],
            color=sassy_search_color,
            lw=1.2,
            marker="D",
            markersize=3,
            markeredgewidth=0.5,
            label="Sassy1",
        ),
        Line2D(
            [0],
            [0],
            color=sassy_tiling_color,
            lw=1.2,
            marker="s",
            markersize=3,
            markeredgewidth=0.5,
            label="Sassy2",
        ),
        Line2D(
            [0],
            [0],
            color=edlib_color,
            lw=1.2,
            marker="o",
            markersize=3,
            markeredgewidth=0.5,
            label="Edlib",
        ),
    ]
    for k, (linestyle, linewidth) in line_style_map.items():
        if k in k_values:
            legend_handles_a.append(
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=f"k={k}",
                )
            )
    ax1.legend(
        handles=legend_handles_a,
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handlelength=2.5,
        handletextpad=0.5,
        fontsize=6,
    )


throughput_csv = "evals/src/sassy2/output/text_scaling_results.csv"
if os.path.exists(throughput_csv):
    df_throughput = pd.read_csv(throughput_csv)
    # median aggregation
    df_throughput = df_throughput.groupby("num_queries", as_index=False).median(
        numeric_only=True
    )

    tools = ["search", "tiling", "edlib"]
    tool_colors = {
        "search": sassy_search_color,
        "tiling": sassy_tiling_color,
        "edlib": edlib_color,
    }
    tool_markers = {"search": "D", "tiling": "s", "edlib": "o"}
    tool_labels = {"search": "Sassy1", "tiling": "Sassy2", "edlib": "Edlib"}

    for tool in tools:
        col = f"{tool}_throughput_gbps"
        if col not in df_throughput.columns:
            continue
        ax2.plot(
            df_throughput["num_queries"],
            df_throughput[col],
            color=tool_colors[tool],
            marker=tool_markers[tool],
            linestyle="--",
            linewidth=1.2,
            markersize=3,
            markeredgewidth=0.5,
            alpha=0.9,
            label=f"{tool_labels[tool]} (k=3)",
        )

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)

    ax2.set_xlabel("Number of patterns")
    ax2.set_ylabel("Total throughput (Gbp/s)")
    ax2.set_title("B", fontweight="bold", loc="left", pad=10)

    # Use same y-axis formatter as ax1 (raw numbers instead of powers of 2)
    ax2.yaxis.set_major_locator(LogLocator(base=2.0, numticks=100))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    # ax2.yaxis.set_minor_locator(LogLocator(base=2.0, subs=[1], numticks=100))
    # ax2.yaxis.set_minor_formatter(plt.NullFormatter())
    ax2.tick_params(labelleft=True)  # Show y-axis labels on the right plot

    num_queries = df_throughput["num_queries"].tolist()
    ax2.set_xticks(num_queries)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xticklabels([str(n) for n in num_queries])
    # ax2.xaxis.set_minor_locator(LogLocator(base=2.0, subs=[0.5], numticks=100))
    # ax2.xaxis.set_minor_formatter(plt.NullFormatter())

    ax2.grid(True, which="major", linewidth=0.4, alpha=0.5, color="#DCDCDC")
    ax2.grid(True, which="minor", linewidth=0.3, alpha=0.45, color="#E8E8E8")
    ax2.set_axisbelow(True)

    # IPC twin axis
    ax2_twin = ax2.twinx()
    for tool in tools:
        ipc_col = f"{tool}_ipc"
        if ipc_col not in df_throughput.columns:
            continue
        ax2_twin.plot(
            df_throughput["num_queries"],
            df_throughput[ipc_col],
            color=tool_colors[tool],
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
        )
    ax2_twin.set_ylabel("IPC", fontsize=9, color="#666666")
    ax2_twin.tick_params(axis="y", labelcolor="#666666", labelsize=7, width=0.5)
    ipc_cols = [
        f"{tool}_ipc" for tool in tools if f"{tool}_ipc" in df_throughput.columns
    ]
    if ipc_cols:
        ax2_twin.set_ylim(0, df_throughput[ipc_cols].max().max() * 1.1)

    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#333333")
    for spine in ax2_twin.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#333333")

    # Legend
    legend_handles_b = [
        Line2D(
            [0],
            [0],
            color=tool_colors[tool],
            linestyle="--",
            marker=tool_markers[tool],
            markersize=3,
            markeredgewidth=0.5,
            lw=1.2,
            label=f"{tool_labels[tool]} (k=3)",
        )
        for tool in tools
    ]
    legend_handles_b.append(
        Line2D([0], [0], color="#666666", linestyle=":", lw=1.0, label="IPC (dotted)")
    )
    ax2.legend(
        handles=legend_handles_b,
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handlelength=1.5,
        handletextpad=0.3,
        fontsize=6,
    )

plt.tight_layout()
plt.savefig("evals/src/sassy2/figs/figure1_scaling_throughput.pdf", bbox_inches="tight")
plt.savefig("evals/src/sassy2/figs/figure1_scaling_throughput.png", bbox_inches="tight")
plt.savefig("evals/src/sassy2/figs/figure1_scaling_throughput.svg", bbox_inches="tight")
plt.show()
