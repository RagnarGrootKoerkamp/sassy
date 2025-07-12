

## Running the bench 
Frist make sure to create the "benchmarks" executable:

`cargo build --release -p benchmarks`

This creates the `benchmarks` executable in `/target/release/`.
This is what the python scripts use to run the benchmarks. 



### Tool comparison (0 & 1 match)

#### Generating the data
Should be used from within **this folder** otherwise change `bench_cmd = ""` in `run_tool_bench.py` to the correct folder
```bash
python3 run_tool_bench.py
```
This creates a folder called `benchmarks` which has the tool configurations and several `results_*.csv` files which have the bench results.

#### Creating throughput (0-match) plot
```bash
python3 plot_tool_bench.py
```
Creates `figs` folder if it does not exist yet and writes the plot to `tool_comparison_high_res.svg` 

#### Creating trace cost plot 
```bash
python3 plot_trace_bench.py
```

### CIRSPR off-traget
todo!