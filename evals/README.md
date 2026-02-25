# Evals

There are 3 separate evals related to the paper:
- Section 3: synthetic results of Sassy vs Edlib
- Section 3: searching a 23bp pattern in the human genome with Sassy vs Parasail
  vs Ish.
- Section 4: comparison with crispr off-target tools SWOffinder and CHOPOFF.


## Section 3: Sassy vs Edlib on synthetic data
First make sure to build the `evals` executable:

`cargo build --release -p evals`

This creates the `benchmarks` executable in `/target/release/`.
This is what the python scripts use to run the benchmarks. 


**Running the benchmark.**
Run the command below inside **this directory**.
```bash
python3 run_tool_bench.py
```
This creates `benchmarks/` that contains some configuration, and `data/*.csv`
with benchmark results.

**Throughput plot.**
```bash
python3 plot_tool_bench.py
```
Outputs `figs/throughput.{svg,pdf}`.

**Throughput statistics.**
Prints numeric throughput (GB/s) statistics corresponding to the plot above.
```bash
python3 throughput_stats.py results_*.csv
```

**Trace plot.**
```bash
python3 plot_trace_bench.py
```
Outputs `figs/trace.{svg,pdf}`.

## Section 3: Comparison with Parasail and Ish

To compare with [parasail](https://github.com/jeffdaily/parasail),
clone the repo and follow the cmake build instructions.
`make parasail_aligner` is sufficient, but will still take a while.
Then, create [`patterns.fa`](./patterns.fa) and download a human genome ([chm13v2.0.fa](https://github.com/marbl/CHM13?tab=readme-ov-file#t2t-chm13v20-t2t-chm13y), [direct download
link](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz)):

```text
>pattern
ACTCGACTTCAGCTACGCACATA
```

and to run with cost parameters matching Ish:

```bash
# Default cost params: 53-69s
time ./parasail_aligner -a sg_dx_striped_sse41_128_8  -x -d -t 1 -q patterns.fa -f chm13v2.0.fa -v -V
time ./parasail_aligner -a sg_dx_striped_sse41_128_16 -x -d -t 1 -q patterns.fa -f chm13v2.0.fa -v -V
time ./parasail_aligner -a sg_dx_striped_avx2_256_8   -x -d -t 1 -q patterns.fa -f chm13v2.0.fa -v -V
time ./parasail_aligner -a sg_dx_striped_avx2_256_16  -x -d -t 1 -q patterns.fa -f chm13v2.0.fa -v -V
# Ish' cost params: 81s-198s
time ./parasail_aligner -a sg_dx_striped_sse41_128_8  -x -d -t 1 -X 2 -M 2 -o 3 -e 1 -q patterns.fa -f chm13v2.0.fa -v -V
time ./parasail_aligner -a sg_dx_striped_sse41_128_16 -x -d -t 1 -X 2 -M 2 -o 3 -e 1 -q patterns.fa -f chm13v2.0.fa -v -V
time ./parasail_aligner -a sg_dx_striped_avx2_256_8   -x -d -t 1 -X 2 -M 2 -o 3 -e 1 -q patterns.fa -f chm13v2.0.fa -v -V
time ./parasail_aligner -a sg_dx_striped_avx2_256_16  -x -d -t 1 -X 2 -M 2 -o 3 -e 1 -q patterns.fa -f chm13v2.0.fa -v -V
```

For [Ish](https://github.com/BioRadOpenSource/ish) ([commit](https://github.com/BioRadOpenSource/ish/commit/92f4fad04142f2f7d08770feec66f928ee1f7079)), follow the build
instructions. To build with `sse` as the target instead of default `avx`
(because it's faster in our benchmark with quite short queries):
``` bash
pixi run build -D ISH_SIMD_TARGET=sse
```
Then run:

``` bash
time ./ish --scoring-matrix actgn --record-type fastx --threads 1 ACTCGACTTCAGCTACGCACATA chm13v2.0.fa
```

For me, this takes 69s for SSE, and 110s for AVX2.


## Section 4: CIRSPR off-traget
We reused the benchmark of the
[ChopOff paper](https://www.biorxiv.org/content/10.1101/2025.01.06.603201v1.full.pdf)
from
[their gitlab](https://git.app.uib.no/valenlab/chopoff-benchmark/-/tree/master?ref_type=heads)
with
[our fork here](https://github.com/rickbeeloo/sassy-crispr-bench):

This workflows uses conda/mamba so you need to install that if you don't have it already.
Run the following in the root of the fork:

```bash
mamba env create --name sassy-benchmark --file environment.yaml
mamba activate sassy-benchmark
snakemake --cores 16
```
Then the output of the tools is in `out_dir` and the timings in `summary.txt`.

Modifications we made to the Chopoff benchmark code:
- **Force serial execution** as the code did not use `threads:` in each rule it would execute multiple tools simultaneously that all use the maximum 
CPU's thereby competing with each other. 
- **Time indexes**, the Chopoff index construction was not timed, we added timings for the construction time for all edit cut-offs.
- **Added Sassy**
- **Remove older tools** that were shown to perform worse (CRISPRITz, Cas-OFFinder)
- Used a genome without N characters as tools might deal with that different ([chm13](https://github.com/marbl/CHM13))

