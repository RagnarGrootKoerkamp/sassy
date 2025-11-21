mod crispr;
mod grep;
mod input_iterator;

use clap::Parser;
use crispr::{CrisprArgs, crispr};

#[derive(clap::Parser)]
#[command(author, version, about)]
enum Args {
    /// Search and print matches of a pattern.
    ///
    /// Fasta/Fastq record based for DNA/IUPAC, line-based for ASCII.
    Grep(grep::GrepArgs),
    /// Like Grep, but output a .tsv with locations of matches.
    Search(grep::SearchArgs),
    /// Like Grep, but records containing a match.
    Filter(grep::FilterArgs),
    /// CRISPR-specific search with PAM and edit-free region
    Crispr(CrisprArgs),
    /// Test CPU features and search throughput
    Test,
}

fn main() {
    ensure_simd::ensure_simd();

    let args = Args::parse();
    env_logger::init();

    match args {
        Args::Grep(args) => args.run(),
        Args::Search(args) => args.run(),
        Args::Filter(args) => args.run(),
        Args::Crispr(crispr_args) => crispr(&crispr_args),
        Args::Test => {
            sassy::test_cpu_features();
            sassy::test_throughput();
        }
    }
}
