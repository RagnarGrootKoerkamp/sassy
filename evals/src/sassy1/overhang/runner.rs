use rand::Rng;
use sassy::Searcher;
use serde::Deserialize;
use std::fs;
use std::fs::File;
use std::io::Write;

#[derive(Deserialize)]
struct Config {
    alphas: Vec<f32>,
}

// Generate random dna sequence of length "length" with alphabet
fn generate_random_dna_sequence(length: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    (0..length)
        .map(|_| b"ACGT"[rng.random_range(0..4)])
        .collect()
}

pub fn run(config: &str) {
    let toml_str = fs::read_to_string(config).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    let query = generate_random_dna_sequence(20);
    let mut text = generate_random_dna_sequence(100);
    let text_len = text.len();

    println!(
        "Query (len: {}): {:?}",
        query.len(),
        String::from_utf8(query.clone()).unwrap()
    );
    println!(
        "Text (len: {}): {:?}",
        text_len,
        String::from_utf8(text.clone()).unwrap()
    );

    // Insert query segments at specific positions:
    // 1. At position 10: insert last 10 chars of query (query[10..])
    // 2. At position 50: insert middle 10 chars of query (query[5..15])
    // 3. At position 90: insert first 10 chars of query (query[..10])

    // Helper function to insert a segment at a position
    fn insert_segment_ending_at(text: &mut [u8], end_pos: usize, segment: &[u8]) {
        let start_pos = end_pos.saturating_sub(segment.len());
        if start_pos + segment.len() <= text.len() {
            text[start_pos..end_pos].copy_from_slice(segment);
        }
    }

    // Insert the three segments
    insert_segment_ending_at(&mut text, 10, &query[10..]); // Last 10 chars at pos 10
    insert_segment_ending_at(&mut text, 50, &query[..]); // All chars
    insert_segment_ending_at(&mut text, 100, &query[..10]); // First 10 chars at pos 90

    println!(
        "Text with insert: {:?}",
        String::from_utf8(text.clone()).unwrap()
    );

    // Create output file
    let mut file = File::create("overhang_results.csv").unwrap();
    writeln!(file, "alpha,end_pos,cost").unwrap();

    for alpha in config.alphas {
        let mut searcher = Searcher::<sassy::profiles::Iupac>::new_fwd_with_overhang(alpha);
        let matches = searcher.search_all(&query, &text, query.len() * 2);

        // Track how many matches we've seen at each end position
        let mut end_pos_counts = std::collections::HashMap::new();

        for m in matches {
            let end_pos = m.text_end;
            let count = end_pos_counts.entry(end_pos).or_insert(0);
            *count += 1;

            // For positions after text_len, increment by the number of times we've seen this position
            let adjusted_end_pos = if end_pos >= text_len {
                text_len + (*count - 1)
            } else {
                end_pos
            };

            // write alpha, adjusted_end_pos, cost
            writeln!(file, "{},{},{}", alpha, adjusted_end_pos, m.cost).unwrap();
        }
    }
}
