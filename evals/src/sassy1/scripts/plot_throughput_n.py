#!/usr/bin/env python3
"""Plot throughput vs text length from throughput_n benchmark CSV (same format as throughput_m)."""
import argparse
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.linewidth": 0.5,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.5,
        "lines.markersize": 3,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "figure.dpi": 600,
    }
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(
    SCRIPT_DIR, "..", "output", "search_throughput_text_len.csv"
)

LINE_STYLE_MAP = {3: ("-", 3), 20: ("--", 3), 0.01: ("-.", 3), 0.05: (":", 3)}
LINESTYLES = ["-", "--", "-.", ":"]


def _k_legend_label(k):
    """Show k as integer when whole (e.g. 3.0 -> '3'), else as float (e.g. 0.05 -> '0.05')."""
    return str(int(k)) if k == int(k) else str(k)


def _x_fmt(x, _pos):
    """X-axis: whole numbers until >1000, then N K (e.g. 2000 -> '2K')."""
    if x >= 1000:
        return f"{int(round(x / 1000))}K"
    return str(int(round(x)))


def _y_fmt(x, _pos):
    """Y-axis: powers of 2 as fractions (1/2, 1/4, 1/8) or integers (1, 2, 4, ...)."""
    if x <= 0:
        return "0"
    n = round(math.log2(x))
    if abs(x - 2**n) < 1e-9 * max(x, 1):
        if n >= 0:
            return str(int(2**n))
        return f"1/{int(2 ** (-n))}"
    return f"{x:.2g}"


def _draw_plot(ax, df, line_style_map, include_parasail, legend_anchor_y=0.02):
    """Draw throughput vs target_len on ax. include_parasail: add Parasail series if column exists.
    legend_anchor_y: y position for legend (lower right anchor)."""
    sassy_color = "#fcc007"
    edlib_color = "black"
    parasail_color = "#666666"  # mid-dark gray

    unique_ks = sorted(df["k"].unique())
    for k in unique_ks:
        sub = df[df["k"] == k]
        if sub.empty:
            continue
        linestyle, linewidth = line_style_map[k]

        ax.plot(
            sub["target_len"],
            sub["search_throughput_gbps"],
            color=sassy_color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.plot(
            sub["target_len"],
            sub["edlib_throughput_gbps"],
            color=edlib_color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        if include_parasail and "parasail_throughput_gbps" in df.columns:
            ax.plot(
                sub["target_len"],
                sub["parasail_throughput_gbps"],
                color=parasail_color,
                linewidth=linewidth,
                linestyle=linestyle,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="major", linewidth=0.8, alpha=0.2)
    ax.set_xlabel("Text length (bp)")
    ax.set_ylabel("Search throughput (Gbp/s)")

    unique_target_lens = sorted(df["target_len"].unique())
    ax.set_xticks(unique_target_lens)
    ax.xaxis.set_major_formatter(FuncFormatter(_x_fmt))
    ax.yaxis.set_major_locator(LogLocator(base=2, numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tools_handles = [
        Line2D([0], [0], color=sassy_color, lw=3, label="Sassy"),
        Line2D([0], [0], color=edlib_color, lw=3, label="Edlib"),
    ]
    if include_parasail and "parasail_throughput_gbps" in df.columns:
        tools_handles.append(
            Line2D([0], [0], color=parasail_color, lw=3, label="Parasail")
        )
    k_handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle=line_style_map[k][0],
            linewidth=3,
            label=f"$k$={_k_legend_label(k)}",
        )
        for k in unique_ks
    ]
    legend_handles = tools_handles + k_handles
    legend_labels = (
        ["Sassy", "Edlib"]
        + (["Parasail"] if (include_parasail and "parasail_throughput_gbps" in df.columns) else [])
    ) + [f"$k$={_k_legend_label(k)}" for k in unique_ks]

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.97, legend_anchor_y),
        loc="lower right",
        handlelength=3.0,
        handletextpad=0.5,
        labelspacing=0.3,
        ncol=2,
        columnspacing=1.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot throughput vs text length (from throughput_n benchmark)"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Path to CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--query-len",
        type=int,
        default=None,
        help="Filter to this query_len (default: use first value in CSV)",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default=None,
        help="Comma-separated k values to plot (default: all in data)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)

    # k is written as string in CSV (keeps int vs float)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    required = [
        "query_len",
        "target_len",
        "k",
        "search_throughput_gbps",
        "edlib_throughput_gbps",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        return 1

    if args.query_len is not None:
        df = df[df["query_len"] == args.query_len]
    if df.empty:
        print("Error: No rows after filter.")
        return 1

    if args.ks is not None:
        ks_want = [int(x.strip()) for x in args.ks.split(",")]
        df = df[df["k"].isin(ks_want)]

    agg_dict = {
        "query_len": "first",
        "search_throughput_gbps": "median",
        "edlib_throughput_gbps": "median",
    }
    if "parasail_throughput_gbps" in df.columns:
        agg_dict["parasail_throughput_gbps"] = "median"
    df = (
        df.groupby(["target_len", "k"], as_index=False)
        .agg(agg_dict)
        .sort_values(["k", "target_len"])
    )

    unique_ks = sorted(df["k"].unique())
    line_style_map = {}
    for i, k in enumerate(unique_ks):
        key = int(k) if k == int(k) else k
        if key in LINE_STYLE_MAP:
            line_style_map[k] = LINE_STYLE_MAP[key]
        else:
            line_style_map[k] = (LINESTYLES[i % len(LINESTYLES)], 3)

    out_dir = os.path.join(SCRIPT_DIR, "..", "figs")
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Sassy + Edlib
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    _draw_plot(ax1, df, line_style_map, include_parasail=False)
    plt.sca(ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "throughput_n.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "throughput_n.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Sassy + Edlib + Parasail
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    _draw_plot(ax2, df, line_style_map, include_parasail=True, legend_anchor_y=0.12)
    plt.sca(ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "throughput_n_parasail.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "throughput_n_parasail.pdf"), bbox_inches="tight")
    plt.close(fig2)

    print(f"Plots saved to {out_dir}/ (throughput_n.* and throughput_n_parasail.*)")
    return 0


if __name__ == "__main__":
    exit(main())
