
api:
    cargo modules structure --package sassy --lib --sort-by visibility

doc:
    cargo doc --no-deps --open

cbindgen:
    cbindgen --config cbindgen.toml --output c/sassy.h

sassy2_fig1:
    mkdir -p evals/src/sassy2/output/
    mkdir -p evals/src/sassy2/figs/
    cargo run -r -p evals -- sassy2 pattern-scaling --config evals/src/sassy2/pattern_scaling_config.toml
    cargo run -r -p evals -- sassy2 text-scaling --config evals/src/sassy2/text_scaling_config.toml
    python3  evals/src/sassy2/scripts/generate_fig1.py

perm:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'


unperm:
	sudo sysctl -w kernel.perf_event_paranoid=2
	sudo sysctl -w kernel.kptr_restrict=1