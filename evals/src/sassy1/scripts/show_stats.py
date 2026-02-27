#!/usr/bin/env python3
"""
Show throughput statistics from Sassy1 benchmark CSV result files.

Reads the throughput CSV format produced by throughput_m / throughput_n:
  num_queries, target_len, query_len, k, search_throughput_gbps, edlib_throughput_gbps, ...

Prints a table: query_len, target_len, k, Sassy GB/s, Edlib GB/s, [Parasail GB/s], speedup vs Edlib, [speedup vs Parasail].

Usage
-----
  python show_stats.py
  python show_stats.py path/to/search_throughput_pat_len.csv
  python show_stats.py evals/src/sassy1/output/*.csv

If no files are given, looks for CSVs in ../output/ relative to this script.
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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / ".." / "output"


def load_throughput_csv(path: os.PathLike | str) -> pd.DataFrame:
    """Load a sassy1 throughput CSV (throughput_m / throughput_n format).

    Expects columns: query_len, target_len, k, search_throughput_gbps, edlib_throughput_gbps.
    Optional: parasail_throughput_gbps.
    Adds speedup_sassy_vs_edlib and (when Parasail present) speedup_sassy_vs_parasail.
    """
    df = pd.read_csv(path)

    required = {"query_len", "target_len", "k", "search_throughput_gbps", "edlib_throughput_gbps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns {missing} missing in '{path}'. "
            "Expected sassy1 throughput CSV (search_throughput_pat_len.csv or search_throughput_text_len.csv)."
        )

    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    # Assert no duplicate (query_len, target_len, k) combinations
    key_cols = ["query_len", "target_len", "k"]
    dupes = df[df.duplicated(subset=key_cols, keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate rows found in '{path}' for key (query_len, target_len, k):\n"
            f"{dupes[key_cols].drop_duplicates().to_string(index=False)}"
        )

    # Speed-up of Sassy over Edlib
    df["speedup_sassy_vs_edlib"] = df.apply(
        lambda r: r["search_throughput_gbps"] / r["edlib_throughput_gbps"]
        if r["edlib_throughput_gbps"] and r["edlib_throughput_gbps"] > 0
        else float("nan"),
        axis=1,
    )

    # Speed-up of Sassy over Parasail (when column present and > 0)
    if "parasail_throughput_gbps" in df.columns:
        df["speedup_sassy_vs_parasail"] = df.apply(
            lambda r: r["search_throughput_gbps"] / r["parasail_throughput_gbps"]
            if r["parasail_throughput_gbps"] and r["parasail_throughput_gbps"] > 0
            else float("nan"),
            axis=1,
        )

    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Show throughput statistics (GB/s) and Sassy speedup from sassy1 throughput "
            "CSV files. If no files are given, looks in ../output/ for search_throughput_*.csv."
        ),
    )
    p.add_argument(
        "files",
        metavar="CSV",
        nargs="*",
        help="Path(s) to throughput CSV file(s). Globs supported.",
    )
    return p.parse_args(argv)


def expand_globs(paths: list[str]) -> list[Path]:
    """Expand any glob patterns in paths and return list of Paths."""
    expanded: list[Path] = []
    for p in paths:
        matches = glob(p)
        if not matches:
            expanded.append(Path(p))
        else:
            expanded.extend(Path(m) for m in matches)
    return expanded


def short_name(path: Path) -> str:
    return path.name.rsplit(".", 1)[0]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.files:
        files = expand_globs(args.files)
    else:
        out = DEFAULT_OUTPUT_DIR.resolve()
        files = list(out.glob("search_throughput_*.csv"))
        if not files:
            files = [Path(p) for p in glob("search_throughput_*.csv")]

    if not files:
        sys.exit(
            "No input CSV files found. Run with path(s) to throughput CSV(s), e.g.\n"
            "  python show_stats.py ../output/search_throughput_pat_len.csv"
        )

    for path in files:
        if not path.exists():
            print(f"Warning: '{path}' does not exist — skipping.")
            continue

        try:
            df = load_throughput_csv(path)
        except ValueError as e:
            print(f"Error: {e} — skipping.")
            continue

        # Build output columns
        out_cols = ["query_len", "target_len", "k", "search_throughput_gbps", "edlib_throughput_gbps"]
        out_names = ["query_len", "target_len", "k", "Sassy GB/s", "Edlib GB/s"]

        if "parasail_throughput_gbps" in df.columns:
            out_cols.append("parasail_throughput_gbps")
            out_names.append("Parasail GB/s")

        out_cols.append("speedup_sassy_vs_edlib")
        out_names.append("speedup vs Edlib")

        if "speedup_sassy_vs_parasail" in df.columns:
            out_cols.append("speedup_sassy_vs_parasail")
            out_names.append("speedup vs Parasail")

        out = df[[c for c in out_cols if c in df.columns]].copy()
        out.columns = out_names

        # Sort for readable output
        out = out.sort_values(["query_len", "target_len", "k"]).reset_index(drop=True)

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