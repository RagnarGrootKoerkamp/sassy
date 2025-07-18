#!/usr/bin/env python3
"""
Command-line utility to extract throughput statistics from Sassy benchmark
CSV result files.

For every selected input file the script prints a table with the following
columns:

    query_len, text_len, k, edlib_GBps, sassy_GBps, speedup_sassy

Throughput is computed exactly like in *plot_throughput.py*:

    throughput [GB/s] = text_length / elapsed_ns

(assuming one byte per character, the ‑ns suffix denotes nanoseconds).

Usage
-----
$ python throughput_stats.py results_match_frac_0_k_0.01.csv

Multiple files can be provided.  If no files are given the script looks for
CSV files in the current working directory whose name contains the substring
"results_".
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd
try:
    from tabulate import tabulate  # type: ignore
except ImportError:  # pragma: no cover
    tabulate = None  # type: ignore

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def load_result_file(path: os.PathLike | str) -> pd.DataFrame:
    """Load *path* into a DataFrame and compute throughput columns.

    Returns the resulting DataFrame **without** any grouping or aggregation.  Each row
    in the original CSV is preserved with two additional columns:

    - ``edlib_gbps``  – throughput of Edlib in GB/s
    - ``sassy_gbps``  – throughput of Sassy in GB/s
    """

    df = pd.read_csv(path)

    if not {"text_length", "edlib_ns", "sassy_ns"}.issubset(df.columns):
        missing = {"text_length", "edlib_ns", "sassy_ns"} - set(df.columns)
        raise ValueError(
            f"Required columns {missing} missing in input file '{path}'."
        )
    
    # Filter query should be > 3 * k 
  #  df = df[df["query_length"] > 3 * df["k"]]

    # Compute throughput in GB/s (text_length bytes processed per ns)
    df["edlib_gbps"] = df["text_length"] / df["edlib_ns"]
    df["sassy_gbps"] = df["text_length"] / df["sassy_ns"]

    # Speed-up of Sassy over Edlib
    df["speedup_sassy"] = df["sassy_gbps"] / df["edlib_gbps"]

    return df



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: D401
    """Parse command-line arguments."""

    p = argparse.ArgumentParser(
        description=(
            "Compute throughput statistics (GB/s) and speed-up for Sassy benchmark "
            "result files.  If no input files are supplied, all CSV files whose "
            "name contains 'results_' in the current directory are processed."
        )
    )

    p.add_argument(
        "files",
        metavar="CSV",
        nargs="*",
        help="Path(s) to result CSV file(s).  Globs are supported.",
    )
    p.add_argument(
        "-g",
        "--group",
        action="store_true",
        help=(
            "Group by 'query_length' and aggregate across repetitions using the "
            "mean of 'text_length', 'edlib_ns' and 'sassy_ns'.  This mirrors the "
            "behaviour in plot_throughput.py."
        ),
    )

    return p.parse_args(argv)



def expand_globs(paths: list[str]) -> list[Path]:
    """Expand any glob patterns in *paths* and return a list of Paths."""

    expanded: list[Path] = []
    for p in paths:
        matches = glob(p)
        if not matches:
            expanded.append(Path(p))
        else:
            expanded.extend(Path(m) for m in matches)
    return expanded



def short_name(path: Path) -> str:
    """Return just the file name without extension for display purposes."""

    return path.name.rsplit(".", 1)[0]



def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """Program entry-point."""

    args = parse_args(argv)

    # Determine which files to process
    if args.files:
        files = expand_globs(args.files)
    else:
        files = [Path(p) for p in glob("results_*.csv")] + [
            Path(p) for p in glob("data/results_*.csv")
        ]

    if not files:
        sys.exit("No input CSV files found.")

    for path in files:
        if not path.exists():
            print(f"Warning: input file '{path}' does not exist — skipping.")
            continue

        df = load_result_file(path)

        # Optional grouping to match plotting script behaviour
        if args.group:
            df = (
                df.groupby("query_length")
                .agg({
                    "text_length": "mean",
                    "k": "first",
                    "edlib_gbps": "mean",
                    "sassy_gbps": "mean",
                    "speedup_sassy": "mean",
                })
                .reset_index()
            )

        # Derived boolean column: whether k <= query_len / 20
        df["k_le_q_div20"] = df["k"] <= (df["query_length"] / 20.0)

        # Select and rename columns for output
        out = df[
            [
                "query_length",
                "text_length",
                "k",
                "k_le_q_div20",
                "edlib_gbps",
                "sassy_gbps",
                "speedup_sassy",
            ]
        ].copy()
        out.columns = [
            "query_len",
            "text_len",
            "k",
            "k<=q/20",
            "edlib GB/s",
            "Sassy GB/s",
            "speedup Sassy",
        ]

        print("\n" + "=" * 80)
        print(short_name(path))
        print("-" * 80)
        if tabulate:
            print(
                tabulate(
                    out,
                    headers="keys",
                    tablefmt="github",
                    floatfmt=".3f",
                    showindex=False,
                )
            )
        else:
            print(out.to_string(index=False))


if __name__ == "__main__":
    main()
