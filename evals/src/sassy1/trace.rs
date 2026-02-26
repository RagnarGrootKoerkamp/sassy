use crate::benchsuite::bench;
use crate::benchsuite::sim_data::Alphabet;
use crate::benchsuite::sim_data::{generate_random_sequence, text_with_query_at_end};
use serde::Deserialize;
use std::fs::File;
use std::io::Write;

#[derive(Deserialize)]
struct Config {
    target_len: usize,
    query_lens: Vec<usize>,
    /// K values: integers (e.g. 3, 20) or fraction of pattern length (e.g. 0.05 = 5%, rounded up).
    ks: Vec<f64>,
    /// Number of (query, text) pairs per (query_len, k). We drop IQR outliers then report median(extra_time).
    #[serde(default)]
    n_pairs: Option<usize>,
    min_benchtime: f64,
    warmup_iterations: usize,
    output_file: String,
    run_edlib: bool,
    edlib_alphabet: String,
    #[serde(default)]
    tools: Option<Vec<bench::BenchTool>>,
}

/// Write one row per (query_len, target_len, k, tool): extra_time_ms = robust median (IQR-outliers dropped).
fn write_extra_times(
    file: &mut File,
    query_len: usize,
    target_len: usize,
    k: &str,
    extra_per_tool: &[(/* name */ &str, /* extra_ms */ f64)],
) -> std::io::Result<()> {
    for (tool, extra_ms) in extra_per_tool {
        writeln!(
            file,
            "{},{},{},{},{}",
            query_len, target_len, k, tool, extra_ms
        )?;
    }
    Ok(())
}

/// Standard median
fn median_f64(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn linear_interpol_index(v: &[f64], idx: f64) -> f64 {
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * idx.fract()
}

fn quartiles_sorted(v: &[f64]) -> (f64, f64) {
    let n = v.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let q1 = linear_interpol_index(v, (n - 1) as f64 * 0.25);
    let q3 = linear_interpol_index(v, (n - 1) as f64 * 0.75);
    (q1, q3)
}

/// Median after filtering out [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
fn median_without_outliers(values: &[f64]) -> f64 {
    // No use to filter advanced if we dont have more than 4 values anyway
    if values.len() < 4 {
        return median_f64(values);
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (q1, q3) = quartiles_sorted(&v);
    let iqr = q3 - q1;
    let lo = q1 - 1.5 * iqr;
    let hi = q3 + 1.5 * iqr;
    let filtered: Vec<f64> = v.into_iter().filter(|&x| x >= lo && x <= hi).collect();
    if filtered.is_empty() {
        // If empty after filtering we also use regular median
        median_f64(values)
    } else {
        median_f64(&filtered)
    }
}

pub fn run(config_path: &str) {
    let toml_str = std::fs::read_to_string(config_path).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    println!("Running trace benchmark (extra time per match)");
    println!("Config: {:?}", config_path);
    println!("Output: {}", config.output_file);
    println!("Target length: {}", config.target_len);
    println!("Query lengths: {:?}", config.query_lens);
    println!("K values: {:?}", config.ks);
    let n_pairs = config.n_pairs.unwrap_or(1);
    println!("n_pairs: {}", n_pairs);
    println!(
        "Warmup: {}, min_benchtime: {} s",
        config.warmup_iterations, config.min_benchtime
    );
    println!();

    let tools = config.tools.as_deref().unwrap_or(&bench::DEFAULT_TOOLS);

    std::fs::create_dir_all(
        std::path::Path::new(&config.output_file)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )
    .ok();
    let mut file = File::create(&config.output_file).unwrap();
    writeln!(file, "query_len,target_len,k,tool,extra_time_ms").unwrap();

    let target_len = config.target_len;

    for &query_len in &config.query_lens {
        for &k_config in &config.ks {
            let k = bench::resolve_k(k_config, query_len);
            if k >= query_len {
                continue;
            }
            if query_len <= 3 * k || query_len > target_len {
                continue;
            }

            // println!(
            //     "Testing q_len={}, t_len={}, k={} (config k={})",
            //     query_len, target_len, k, k_config
            // );

            // Run each pair separately so we get extra_i = (match_i - random_i) per pair, then median(extra_i)
            let mut extra_search: Vec<f64> = Vec::with_capacity(n_pairs);
            let mut extra_tiling: Vec<f64> = Vec::with_capacity(n_pairs);
            let mut extra_edlib: Vec<f64> = Vec::with_capacity(n_pairs);
            let mut extra_parasail: Vec<f64> = Vec::with_capacity(n_pairs);

            // For tracing we just generate random (q,t=p,t) pairs and run each
            // for at least the min benchtime, then we combine all the runtimes,
            // strip the outliers, and then get the median (worked better than median directly)
            for _ in 0..n_pairs {
                let query = generate_random_sequence(query_len, &Alphabet::Dna, None);
                let text_random: Vec<u8> =
                    generate_random_sequence(target_len, &Alphabet::Dna, None);

                // Random pair
                let queries_1 = vec![query.clone()];
                let suite_random = bench::benchmark_tools(
                    &queries_1,
                    &[text_random.as_slice()],
                    k,
                    config.warmup_iterations,
                    config.min_benchtime,
                    1,
                    &Alphabet::Iupac,
                    false,
                    tools,
                    false,
                );

                // Random pair with foreced match at end of text
                let text_match_at_end = text_with_query_at_end(&text_random, &query);
                let queries_2 = vec![query];
                let suite_match = bench::benchmark_tools(
                    &queries_2,
                    &[text_match_at_end.as_slice()],
                    k,
                    config.warmup_iterations,
                    config.min_benchtime,
                    1,
                    &Alphabet::Iupac,
                    false,
                    tools,
                    false,
                );

                extra_search.push(suite_match.search.median - suite_random.search.median);
                extra_tiling.push(suite_match.tiling.median - suite_random.tiling.median);
                extra_edlib.push(suite_match.edlib.median - suite_random.edlib.median);
                extra_parasail.push(suite_match.parasail.median - suite_random.parasail.median);
            }

            let med_search = median_without_outliers(&extra_search);
            let med_tiling = median_without_outliers(&extra_tiling);
            let med_edlib = median_without_outliers(&extra_edlib);
            let med_parasail = median_without_outliers(&extra_parasail);

            // println!(
            //     "  Extra per match (median (-outliers) over {} pairs): search={:.2}ms tiling={:.2}ms edlib={:.2}ms parasail={:.2}ms",
            //     n_pairs, med_search, med_tiling, med_edlib, med_parasail,
            // );

            let extra_per_tool = [
                ("search", med_search),
                ("tiling", med_tiling),
                ("edlib", med_edlib),
                ("parasail", med_parasail),
            ];
            write_extra_times(
                &mut file,
                query_len,
                target_len,
                &format!("{}", k_config),
                &extra_per_tool,
            )
            .unwrap();
        }
    }

    println!(
        "\nTrace benchmark complete. Extra times (ms) written to {}",
        config.output_file
    );
}
