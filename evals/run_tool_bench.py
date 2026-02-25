#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import numpy as np
import os
import toml

bench_cmd = os.path.join(os.path.dirname(__file__), "..", "target", "release", "evals")

# Test if we are all set
status = os.system(f"{bench_cmd} sassy1 edlib --help")  # just to check if all is good
if status != 0:
    print("Error: evals executable not found. Build with: cargo build --release -p evals")
    exit(1)

# Create benchmarks folder if not exists already
out_dir = "benchmarks/tool_comp_configs/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Create tool comparison template file from Python
tool_comp_template = f"{out_dir}/template.toml"
template_data = {
    "query_lengths": [20, 30, 50, 100, 200, 300, 500, 1000],
    # 100k in text lengths for original figures
    "text_lengths": [50, 100, 150, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 128000],
    "k": [0],
    "matches": [0],
    "bench_iter": [1000],
    "alphabet": ["dna"],
    "profile": ["dna"],
    "rc": ["withoutrc"],
    "verbose": False,
    "edlib": True,
}

# Write the template file
with open(tool_comp_template, "w") as f:
    toml.dump(template_data, f)
print("Created tool comparison template:")
print(template_data)

# Load and edit the template
tool_data = toml.load(tool_comp_template)

# Create config files for different k values
k_list = [3, 20, 0.01, 0.05]
config_files = []

for k in k_list:
    # Create a copy of the template data
    config_data = tool_data.copy()
    config_data["k"] = [k]

    # Determine if k is absolute or relative
    if isinstance(k, float) and k < 1.0:
        filename = f"k_{int(k*100)}_percent.toml"
    else:
        filename = f"k{k}.toml"

    # Save the config file
    config_path = f"{out_dir}{filename}"
    with open(config_path, "w") as f:
        toml.dump(config_data, f)
    print(f"Created config file: {config_path}")
    config_files.append(config_path)

print("All tool comparison config files created successfully!")

print("Running benchmarks...")
for config_file in config_files:
    cmd = f"{bench_cmd} sassy1 edlib --config {config_file}"
    print(cmd)
    status = os.system(cmd)
    if status != 0:
        print(f"Error: benchmark failed for: {config_file}")
        exit(1)
