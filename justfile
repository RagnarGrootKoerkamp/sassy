
api:
    cargo modules structure --package sassy --lib --sort-by visibility

doc:
    cargo doc --no-deps --open

cbindgen:
    cbindgen --config cbindgen.toml --output c/sassy.h

bench_patterns:
    cargo bench --bench perf
    python3 benches/table_it.py

perm:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'