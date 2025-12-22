
api:
    cargo modules structure --package sassy --lib --sort-by visibility

doc:
    cargo doc --no-deps --open

cbindgen:
    cbindgen --config cbindgen.toml --output c/sassy.h

bench_patterns:
    cargo bench --bench perf
    python3 benches/table_it.py