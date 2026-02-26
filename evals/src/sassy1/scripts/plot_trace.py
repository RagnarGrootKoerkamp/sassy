#!/usr/bin/env python3
"""Plot trace benchmark: extra time per match as derived trace throughput (10³ matches/s).

Reads CSV with columns query_len, target_len, k, tool, extra_time_ms.
Same layout as plot_throughput_m: x = pattern length, y = trace throughput (10³ matches/s).
Trace throughput = 1 / extra_time_ms (extrapolated from one extra match).
"""
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
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "..", "output", "trace_extra_time.csv")

LINE_STYLE_MAP = {3: ("-", 3), 20: ("--", 3), 0.01: ("-.", 3), 0.05: (":", 3)}
LINESTYLES = ["-", "--", "-.", ":"]

TOOL_COLORS = {
    "search": "#fcc007",   # Sassy
    "tiling": "#7F5DE5",  # Sassy2 purple
    "edlib": "black",
    "parasail": "#666666",
}
TOOL_LABELS = {"search": "Sassy", "tiling": "Sassy2", "edlib": "Edlib", "parasail": "Parasail"}


def _k_legend_label(k):
    return str(int(k)) if k == int(k) else str(k)


def _x_fmt(x, _pos):
    if x >= 1000:
        return f"{int(round(x / 1000))}K"
    return str(int(round(x)))


def _y_fmt(x, _pos):
    if x <= 0:
        return "0"
    n = round(math.log2(x))
    if abs(x - 2**n) < 1e-9 * max(x, 1):
        if n >= 0:
            return str(int(2**n))
        return f"1/{int(2 ** (-n))}"
    return f"{x:.2g}"


def main():
    parser = argparse.ArgumentParser(
        description="Plot trace extra time as throughput (10³ matches/s)"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Path to trace CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=None,
        help="Filter to this target_len (default: use first value in CSV)",
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
    required = ["query_len", "target_len", "k", "tool", "extra_time_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        return 1

    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    if args.target_len is not None:
        df = df[df["target_len"] == args.target_len]
    if df.empty:
        print("Error: No rows after filter.")
        return 1

    if args.ks is not None:
        ks_want = [float(x.strip()) for x in args.ks.split(",")]
        df = df[df["k"].isin(ks_want)]

    df["throughput_1e3_per_s"] = df["extra_time_ms"].apply(
        lambda x: 1.0 / x if x > 0 else float("nan")
    )

    wide = df.pivot_table(
        index=["query_len", "target_len", "k"],
        columns="tool",
        values="throughput_1e3_per_s",
        aggfunc="mean",
    ).reset_index()

    unique_ks = sorted(wide["k"].dropna().unique())
    line_style_map = {}
    for i, k in enumerate(unique_ks):
        key = int(k) if k == int(k) else k
        if key in LINE_STYLE_MAP:
            line_style_map[k] = LINE_STYLE_MAP[key]
        else:
            line_style_map[k] = (LINESTYLES[i % len(LINESTYLES)], 3)

    fig, ax = plt.subplots(figsize=(8, 6))

    tools_to_plot = [t for t in ["search", "tiling", "edlib", "parasail"] if t in wide.columns]
    for k in unique_ks:
        sub = wide[wide["k"] == k]
        if sub.empty:
            continue
        linestyle, linewidth = line_style_map[k]
        for tool in tools_to_plot:
            col = tool
            if col not in sub.columns:
                continue
            ax.plot(
                sub["query_len"],
                sub[col],
                color=TOOL_COLORS[tool],
                linewidth=linewidth,
                linestyle=linestyle,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="major", linewidth=0.8, alpha=0.2)
    ax.set_xlabel("Pattern length (bp)")
    ax.set_ylabel("Trace throughput (10³ matches/s)")

    unique_query_lens = sorted(wide["query_len"].unique())
    ax.set_xticks(unique_query_lens)
    ax.xaxis.set_major_formatter(FuncFormatter(_x_fmt))
    ax.yaxis.set_major_locator(LogLocator(base=2, numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tools_handles = [
        Line2D([0], [0], color=TOOL_COLORS[t], lw=3, label=TOOL_LABELS[t])
        for t in tools_to_plot
    ]
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
    legend_labels = [TOOL_LABELS[t] for t in tools_to_plot] + [
        f"$k$={_k_legend_label(k)}" for k in unique_ks
    ]

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.97, 0.97),
        loc="upper right",
        handlelength=3.0,
        handletextpad=0.5,
        labelspacing=0.3,
        ncol=min(3, len(legend_labels)),
        columnspacing=1.0,
    )

    out_dir = os.path.join(SCRIPT_DIR, "..", "figs")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trace.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "trace.pdf"), bbox_inches="tight")
    print(f"Plots saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
