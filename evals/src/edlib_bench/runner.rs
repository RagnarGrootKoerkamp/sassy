use crate::edlib_bench::edlib::*;
use crate::edlib_bench::grid::*;
use crate::edlib_bench::sim_data::*;
use sassy::{Match, Searcher};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

fn remove_10_percent_outliers(times: &mut Vec<f64>) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let trim_size = (times.len() as f64 * 0.1) as usize;
    let trimmed_times = &times[trim_size..times.len() - trim_size];
    *times = trimmed_times.to_vec();
}

macro_rules! time_it {
    ($label:expr, $expr:expr, $iters:expr, $pairs:expr) => {{
        let _label = $label;
        const WARMUP_RUNS: usize = 5;
        const SAMPLES_PER_PAIR: usize = 3;
        let mut final_result = None;

        let mut base_times = Vec::with_capacity($iters);
        let mut plus_one_times = Vec::with_capacity($iters);

        // Warmup phase
        for _ in 0..WARMUP_RUNS {
            for (q, t, _, _) in $pairs.iter() {
                std::hint::black_box($expr(q, t));
            }
        }

        for (q, t, t_plus_one_q, _) in $pairs.iter() {
            for _ in 0..SAMPLES_PER_PAIR {
                // Run base case
                let start = std::time::Instant::now();
                let r = std::hint::black_box($expr(q, t));
                let elapsed = start.elapsed();
                base_times.push(elapsed.as_nanos() as f64);
                final_result = Some(r);

                // Run base case + 1 extra query
                let start = std::time::Instant::now();
                 std::hint::black_box($expr(q, t_plus_one_q));
                let elapsed = start.elapsed();
                plus_one_times.push(elapsed.as_nanos() as f64);
            }
        }

        // Then we also subtract the plus one from base times to get the extra time for one more match
        let mut base_minus_plus_one: Vec<f64> = base_times.iter().zip(plus_one_times.iter()).map(|(b, p)| p - b).collect();

        // For the base times we remove the top and bottom 10% outliers
        remove_10_percent_outliers(&mut base_times);
        remove_10_percent_outliers(&mut base_minus_plus_one);

        let base_mean = base_times.iter().sum::<f64>() / base_times.len() as f64;
        let base_minus_plus_one_mean = base_minus_plus_one.iter().sum::<f64>() / base_minus_plus_one.len() as f64;

        (final_result.unwrap(), base_mean, base_minus_plus_one_mean)
    }};
}

pub fn run(grid_config: &str) {
    // Read grid config file for benching
    let grid = read_grid(grid_config).expect("Invalid grid config");

    // Open output file and write header
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(grid.output_file())
        .unwrap_or_else(|_| panic!("Unable to open {}", grid.output_file()));

    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(
        writer,
        "query_length,text_length,k,edlib_matches,sassy_matches,max_edits,bench_iter,alphabet,profile,rc,edlib_ns,edlib_ns_plus_one,sassy_ns,sassy_ns_plus_one"
    ).unwrap();

    // Get combinations
    for param_set in grid.all_combinations() {
        println!("Param set: {param_set:?}");

        let num_matches = param_set.matches;
        let bench_iter = param_set.bench_iter;

        // Generate all query-text pairs upfront
        let pairs: Vec<_> = (0..bench_iter)
            .map(|_| {
                generate_query_and_text_with_matches(
                    param_set.query_length,
                    param_set.text_length,
                    num_matches,
                    param_set.max_edits,
                    param_set.max_edits,
                    &param_set.alphabet,
                )
            })
            .collect();

        // Running Edlib
        let (edlib_matches, edlib_mean_ms, edlib_plus_one_ms) = if param_set.edlib {
            let edlib_config = get_edlib_config(param_set.k as i32, &param_set.alphabet);
            let (r, ms, ms_plus_one) = time_it!(
                "edlib",
                |q, t| { run_edlib(q, t, &edlib_config) },
                bench_iter,
                &pairs
            );
            let edlib_matches = r.startLocations.unwrap_or(vec![]);
            (edlib_matches, ms, ms_plus_one)
        } else {
            (vec![], 0.0, 0.0)
        };

        // Get the correct search function (not timed)
        let mut search_fn = get_search_fn(&param_set);

        // Now time the search
        let (sassy_matches, sassy_mean_ms, sassy_plus_one_ms) = time_it!(
            "sassy",
            |q, t| { search_fn(q, t, param_set.k) },
            bench_iter,
            &pairs
        );

        if param_set.edlib {
            println!("Edlib matches: {:?}", edlib_matches.len());
        }
        println!("Sassy matches: {:?}", sassy_matches.len());

        if param_set.verbose {
            println!("Edlib matches");
            for loc in edlib_matches.iter() {
                println!("{loc}");
            }
            println!("Sassy matches");
            for loc in sassy_matches.iter() {
                println!("{loc:?}");
            }
        }

        // Write row to CSV
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{:?},{},{},{:.0},{:.0},{:.0},{:.0}", // Added one more .0 for plus_one_ms
            param_set.query_length,
            param_set.text_length,
            param_set.k,
            edlib_matches.len(),
            sassy_matches.len(),
            param_set.max_edits,
            param_set.bench_iter,
            param_set.alphabet,
            param_set.profile,
            param_set.rc,
            edlib_mean_ms,
            edlib_plus_one_ms,
            sassy_mean_ms,
            sassy_plus_one_ms
        )
        .unwrap();
    }
    // Ensure all data is written
    writer.flush().unwrap();
}

type SearchFn = Box<dyn FnMut(&[u8], &[u8], usize) -> Vec<Match>>;

fn get_search_fn(param_set: &ParamSet) -> SearchFn {
    let rc = match param_set.rc {
        "withrc" => true,
        "withoutrc" => false,
        x => panic!("Unsupported rc config: {x}"),
    };
    match param_set.profile {
        // IUPAC profile
        "iupac" => {
            let mut searcher = Searcher::<sassy::profiles::Iupac>::new(rc, None);
            Box::new(move |q, t, k| searcher.search(q, &t, k))
        }

        // DNA profile
        "dna" => {
            let mut searcher = Searcher::<sassy::profiles::Dna>::new(rc, None);
            Box::new(move |q, t, k| searcher.search(q, &t, k))
        }

        // ASCII profile
        "ascii" => {
            let mut searcher = Searcher::<sassy::profiles::Ascii>::new(rc, None);
            Box::new(move |q, t, k| searcher.search(q, &t, k))
        }

        _ => panic!(
            "Unsupported combination: {:?} {:?} {:?}",
            param_set.profile, param_set.rc, param_set.alphabet
        ),
    }
}
