#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FormatStrFormatter, FuncFormatter
import os
from glob import glob

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.linewidth": 0.5,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.5,
        "lines.markersize": 3,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "figure.dpi": 600,
    }
)

QUERY_LENGTH_FIXED = 100
KS = [3, 20] # dropping percetanges

files = glob("data/*.csv")

all_query_lengths = set()
for file in files:
    try:
        df_temp = pd.read_csv(file)
        all_query_lengths.update(df_temp["query_length"].unique())
    except Exception:
        continue
if QUERY_LENGTH_FIXED in all_query_lengths:
    query_length_used = QUERY_LENGTH_FIXED
else:
    query_length_used = min(all_query_lengths) if all_query_lengths else 20
    print(f"Note: query_length={QUERY_LENGTH_FIXED} not in data; using query_length={query_length_used}")

dfs = []
k_group_files = []
for file in files:
    try:
        df_temp = pd.read_csv(file)
        k = df_temp["k"].iloc[0]
        group_k = float(file.rstrip(".csv").split("_")[-1])
        group_k = float(group_k) if "0.0" in str(group_k) else int(group_k)
        if group_k not in KS:
            continue
        k_group_files.append((group_k, k, file))
        df_temp = df_temp[df_temp["query_length"] > 3 * k]
        df_temp = df_temp[df_temp["query_length"] == query_length_used]
        df_temp = (
            df_temp.groupby("text_length")
            .agg({"edlib_ns": "mean", "sassy_ns": "mean"})
            .reset_index()
        )
        df_temp["k_group"] = f"k={group_k}"
        dfs.append(df_temp)
    except Exception as e:
        print(f"Warning: {file}: {e}. Skipping.")

if not dfs:
    print("Error: No data files found!")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
# We only use 100K for search/trace plots as in original submision
# but for text len throughput fig we use orders of 2 (so exclude 100K here)
df = df[df["text_length"] != 100_000].copy()
df["sassy_gbps"] = df["text_length"] / df["sassy_ns"]
df["edlib_gbps"] = df["text_length"] / df["edlib_ns"]

sassy_color = "#fcc007"
edlib_color = "black"
line_style_map = {3: ("-", 3), 20: ("--", 3)}

fig, ax = plt.subplots(figsize=(8, 6))

for group_k, single_k, file in k_group_files:
    sub = df[df["k_group"] == f"k={group_k}"]
    if sub.empty:
        continue
    linestyle, linewidth = line_style_map.get(group_k, ("-", 3))
    ax.plot(sub["text_length"], sub["sassy_gbps"],
            color=sassy_color, linewidth=linewidth, linestyle=linestyle)
    ax.plot(sub["text_length"], sub["edlib_gbps"],
            color=edlib_color, linewidth=linewidth, linestyle=linestyle)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)

ax.set_xlabel("Text length (bp)")
ax.set_ylabel("Total throughput (Gbp/s)")

unique_text_lengths = sorted(df["text_length"].unique())
ax.set_xticks(unique_text_lengths)


def x_text_length_formatter(x, _pos):
    if x >= 1000:
        k = int(round(x / 1000))
        return f"{k}k"
    return str(int(x))


ax.xaxis.set_major_formatter(FuncFormatter(x_text_length_formatter))
ax.xaxis.set_minor_formatter(plt.NullFormatter())

ax.yaxis.set_major_locator(LogLocator(base=2.0, numticks=100))


def y_fraction_formatter(x, _pos):
    """Format y-axis: fractions like 1/2, 1/4 for v < 1; integers otherwise."""
    if x <= 0:
        return "0"
    if x >= 1:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.2g}"
    # v in (0, 1): try to show as 1/n or k/n
    for n in (2, 3, 4, 5, 8, 10, 16, 32):
        if abs(x - 1 / n) < 0.01 / n:
            return f"$\\frac{{1}}{{{n}}}$"
    for n in (2, 3, 4):
        for k in range(1, n):
            if abs(x - k / n) < 0.01 / n:
                return f"$\\frac{{{k}}}{{{n}}}$"
    return f"{x:.2g}"


ax.yaxis.set_major_formatter(FuncFormatter(y_fraction_formatter))

ax.grid(True, which="major", linewidth=0.4, alpha=0.5, color="#DCDCDC")
ax.grid(True, which="minor", linewidth=0.3, alpha=0.45, color="#E8E8E8")
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#333333")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

tools_handles = [
    Line2D([0], [0], color=sassy_color, lw=3, label="Sassy"),
    Line2D([0], [0], color=edlib_color,  lw=3, label="Edlib"),
]
fixedk_handles = [
    Line2D([0], [0], color="gray", linestyle=line_style_map[3][0],  lw=3, label="$k$=3"),
    Line2D([0], [0], color="gray", linestyle=line_style_map[20][0], lw=3, label="$k$=20"),
]

ax.legend(
    handles=tools_handles + fixedk_handles,
    labels=["Sassy", "Edlib", "$k$=3", "$k$=20"],
    loc="lower right",
    handlelength=3.0,
    handletextpad=0.5,
    labelspacing=0.3,
    ncol=2,
    columnspacing=1.0,
    frameon=True,
    fancybox=False,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    shadow=False,
)

os.makedirs("figs", exist_ok=True)
plt.tight_layout()
plt.savefig("figs/throughput_text.svg", bbox_inches="tight")
plt.savefig("figs/throughput_text.pdf", bbox_inches="tight")
print(f"Saved figs/throughput_text.svg and .pdf  (query_length={query_length_used})")