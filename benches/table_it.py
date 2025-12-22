import json
import os
from pathlib import Path

# Path to the group results
RESULTS_DIR = Path("target/criterion/search_performance")

def format_throughput(bytes_per_sec):
    """Matches Criterion's GiB/s and MiB/s output using binary units."""
    if bytes_per_sec >= 1024**3:
        return f"{bytes_per_sec / (1024**3):.2f} GiB/s"
    if bytes_per_sec >= 1024**2:
        return f"{bytes_per_sec / (1024**2):.2f} MiB/s"
    return f"{bytes_per_sec / 1024:.2f} KiB/s"

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} not found.")
        return

    results = []

    for bench_dir in RESULTS_DIR.iterdir():
        if not bench_dir.is_dir() or bench_dir.name == "report":
            continue

        # New benches are in new folder 
        estimates_file = bench_dir / "new" / "estimates.json"
        bench_file = bench_dir / "new" / "benchmark.json"

        if not (estimates_file.exists() and bench_file.exists()):
            continue

        # Throughput is in benchmark folder
        with open(bench_file) as f:
            bench_data = json.load(f)
            throughput_bytes = bench_data.get("throughput", {}).get("Bytes", 0)

        # Times are in estimates folder
        with open(estimates_file) as f:
            estimates_data = json.load(f)
            mean_ns = estimates_data["mean"]["point_estimate"]

        # Each folder has the number of patterns as q = X, pares that as well
        folder_name = bench_dir.name
        if "_q=" in folder_name:
            algo, q_val = folder_name.rsplit("_q=", 1)
            algo = algo.replace("_", " ")
        else:
            algo, q_val = folder_name, "0"

        # Calculate Throughput: (Bytes / Seconds)
        bytes_per_sec = (throughput_bytes / mean_ns) * 1_000_000_000

        results.append({
            "algo": algo,
            "q": int(q_val),
            "throughput": bytes_per_sec,
            "time_ms": mean_ns / 1_000_000
        })

    # Sort by Q, then by Throughput
    results.sort(key=lambda x: (x['q'], -x['throughput']))

    # Print Output
    print(f"\n{'Algorithm':<25} | {'Q':<8} | {'Avg Throughput':<15} | {'Mean Time':<10}")
    print("-" * 68)
    for r in results:
        print(f"{r['algo']:<25} | {r['q']:<8} | {format_throughput(r['throughput']):<15} | {r['time_ms']:>8.2f} ms")

if __name__ == "__main__":
    main()