
api:
    cargo modules structure --package sassy --lib --sort-by visibility

doc:
    cargo doc --no-deps --open

cbindgen:
    cbindgen --config cbindgen.toml --output c/sassy.h

sassy2_fig1:
    cargo run -r -p evals -- sassy2 scaling-benchmark --config evals/src/sassy2/scaling_config.toml 
    cargo run -r -p evals -- sassy2 pattern-throughput --config evals/src/sassy2/pattern_throughput_config.toml 
    python3  evals/src/sassy2/scripts/generate_fig1.py


perm:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'