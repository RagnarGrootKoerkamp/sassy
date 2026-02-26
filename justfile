
api:
    cargo modules structure --package sassy --lib --sort-by visibility

doc:
    cargo doc --no-deps --open

cbindgen:
    cbindgen --config cbindgen.toml --output c/sassy.h

perm:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

unperm:
	sudo sysctl -w kernel.perf_event_paranoid=2
	sudo sysctl -w kernel.kptr_restrict=1

perf: 
    sudo sudo cpupower frequency-set -u 3.7GHz -g performance



### Sassy 2 evals

# Off-targets
sassy2_prep_off_targets:
    mkdir -p evals/src/sassy2/data
    mkdir -p evals/src/sassy2/data/downloaded
    wget -O evals/src/sassy2/data/downloaded/chm13v2.0.fa.gz https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz
    gunzip evals/src/sassy2/data/downloaded/chm13v2.0.fa.gz

sassy2_run_off_targets:
    just perf
    mkdir -p evals/src/sassy2/output
    cargo run -r -p evals -- sassy2 off-targets --config evals/src/sassy2/configs/crispr_off_target_config.toml

# Nanopore 
sassy2_prep_nanopore:
    mkdir -p evals/src/sassy2/data
    mkdir -p evals/src/sassy2/data/downloaded
    wget -O evals/src/sassy2/data/downloaded/SRR26425221_1.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR264/021/SRR26425221/SRR26425221_1.fastq.gz
    gunzip evals/src/sassy2/data/downloaded/SRR26425221_1.fastq.gz

sassy2_run_nanopore:
    just perf
    mkdir -p evals/src/sassy2/output
    cargo run -r -p evals -- sassy2 nanopore --config evals/src/sassy2/configs/nanopore_config.toml

# Text and pattern scaling fig
sassy2_fig1:
    just perm
    just perf   
    mkdir -p evals/src/sassy2/output/
    mkdir -p evals/src/sassy2/figs/
    cargo run -r -p evals -- sassy2 pattern-scaling --config evals/src/sassy2/configs/pattern_scaling_config.toml
    cargo run -r -p evals -- sassy2 text-scaling --config evals/src/sassy2/configs/text_scaling_config.toml
    python3  evals/src/sassy2/scripts/generate_fig1.py
    just unperm

# Flamegraph: run lib unit test (workaround when package has lib + bin with same name)
flamegraph test_name:
    cargo test --no-run --lib
    flamegraph -- $(cargo test --release --no-run --lib 2>&1 | sed -n '/Executable/s/.*(\(.*\))/\1/p') {{test_name}}


# Sassy 1 run all evals
sassy1_figs:
    mkdir -p evals/src/sassy1/output
    mkdir -p evals/src/sassy1/figs
    cargo run -r -p evals -- sassy1 throughput_m --config evals/src/sassy1/configs/throughput_m.toml
    cargo run -r -p evals -- sassy1 throughput_n --config evals/src/sassy1/configs/throughput_n.toml
    cargo run -r -p evals -- sassy1 trace --config evals/src/sassy1/configs/trace.toml
    python3 evals/src/sassy1/scripts/plot_throughput_m.py
    python3 evals/src/sassy1/scripts/plot_throughput_n.py
    python3 evals/src/sassy1/scripts/plot_trace.py