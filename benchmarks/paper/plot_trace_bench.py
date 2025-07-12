import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FormatStrFormatter
import os
from glob import glob


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.linewidth': 0.5,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 3,
    'legend.fontsize': 14,
    'legend.frameon': False,
    'figure.dpi': 600,
})

# Get all files with "results_" in the name
files = glob("results_*.csv")

# Read and process all files
dfs = []
k_group_files = []
for file in files:
    try:
        df_temp = pd.read_csv(file)
        k = df_temp['k'].iloc[0]
        group_k = float(file.rstrip(".csv").split("_")[-1])
        # group_k is absolute k, make integer if it is a float
        print("Group k :", group_k)
        if "0.0" in str(group_k): # hacky
            group_k = float(group_k)
        else:
            group_k = int(group_k)
        k_group_files.append((group_k, k, file))
        df_temp = df_temp[df_temp['query_length'] > 3 * k]
        df_temp = df_temp.groupby('query_length').agg({
            'sassy_ns_plus_one': 'mean',
            'edlib_ns_plus_one': 'mean',
            'sassy_matches': 'mean',
            'edlib_matches': 'mean'
        }).reset_index()
        df_temp['k_group'] = f'k={group_k}'
        dfs.append(df_temp)
    except FileNotFoundError:
        print(f"Warning: File {file} not found. Skipping k={group_k}")
        continue

if not dfs:
    print("Error: No data files found!")
    exit(1)

df = pd.concat(dfs, ignore_index=True)

# Convert time differences to microseconds 
df['sassy_time_diff'] = df['sassy_ns_plus_one'] * -1 / 1000
df['edlib_time_diff'] = df['edlib_ns_plus_one'] * -1 / 1000

# === Plotting parameters ===
sassy_color = '#fcc007'  # yellow
edlib_color = 'black'

line_style_map = {
    3: ('-', 3),      
    20: ('--', 3),    
    0.01: ('-.', 3), 
    0.05: (':', 3)    
}

# === Start plot ===
fig, ax = plt.subplots(figsize=(8, 6))

for group_k, single_k, file in k_group_files:
    sub = df[df['k_group'] == f'k={group_k}']
    if sub.empty:
        continue
        
    linestyle, linewidth = line_style_map[group_k]
    
    # Plot Sassy
    ax.plot(sub['query_length'], sub['sassy_time_diff'],
            color=sassy_color, linewidth=linewidth,
            linestyle=linestyle)
    
    # Plot Edlib
    ax.plot(sub['query_length'], sub['edlib_time_diff'],
            color=edlib_color, linewidth=linewidth,
            linestyle=linestyle)

# Set log scales
ax.set_xscale('log')
ax.set_yscale('log')

# Grid, labels
ax.grid(True, which='major', linewidth=0.5, alpha=0.7)

ax.set_xlabel("Query Length")
ax.set_ylabel("Trace time (μs)")

# Y-axis formatter
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=10))

# Custom formatter to show integers without .0 for values >= 1
def custom_formatter(x, pos):
    if x >= 1:
        return f'{int(x)}'
    else:
        return f'{x:.1f}'

ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

# Show exact query lengths on x-axis
unique_query_lengths = sorted(df['query_length'].unique())
ax.set_xticks(unique_query_lengths)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === Custom Legend ===
legend_handles = []
legend_labels = []

# Column 1: Tools
tools_handles = [
    Line2D([0], [0], color=sassy_color, lw=3, label='Sassy'),
    Line2D([0], [0], color=edlib_color, lw=3, label='Edlib')
]
tools_labels = ['Sassy', 'Edlib']

# Column 2: Fixed k
fixedk_handles = [
    Line2D([0], [0], color='gray', linestyle=line_style_map[3][0], linewidth=3, label='$k$=3'),
    Line2D([0], [0], color='gray', linestyle=line_style_map[20][0], linewidth=3, label='$k$=20')
]
fixedk_labels = ['$k$=3', '$k$=20']

# Column 3: Relative k %
relk_handles = [
    Line2D([0], [0], color='gray', linestyle=line_style_map[0.01][0], linewidth=3, label='$k$=1%'),
    Line2D([0], [0], color='gray', linestyle=line_style_map[0.05][0], linewidth=3, label='$k$=5%')
]
relk_labels = ['$k$=1%', '$k$=5%']

# Combine all for legend
legend_handles = tools_handles + fixedk_handles + relk_handles
legend_labels = tools_labels + fixedk_labels + relk_labels

# Create the legend at bottom with 3 columns
ax.legend(
    handles=legend_handles,
    labels=legend_labels,
    frameon=True, fancybox=True, shadow=False,
  #  bbox_to_anchor=(0.95, 0.1),  # lower right but moved up a bit
    loc='lower right',
    handlelength=3.0,
    handletextpad=0.5,
    labelspacing=0.3,
    ncol=3,
    columnspacing=1.0
)

# Create output directory
os.makedirs("figs", exist_ok=True)

plt.tight_layout()

plt.savefig("figs/trace_comparison_high_res.svg", dpi=600, bbox_inches='tight')
# plt.savefig("figs/trace_comparison_high_res.pdf", dpi=600, bbox_inches='tight')
# plt.show()

print("Plots saved successfully to figs/ directory")