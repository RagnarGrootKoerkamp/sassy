mod crispr;
mod grep;
mod input_iterator;
mod search;

use clap::Parser;
use grep::GrepArgs;
use {
    crispr::{CrisprArgs, crispr},
    search::{SearchArgs, search},
};

#[derive(clap::Parser)]
#[command(author, version, about)]
enum Args {
    /// Search a single sequence or multi-fasta in a multi-fasta text
    Search(SearchArgs),
    /// CRISPR-specific search with PAM and edit-free region
    Crispr(CrisprArgs),
    /// Search and print matches of a pattern.
    ///
    /// Fasta/Fastq record based for DNA/IUPAC, line-based for ASCII.
    Grep(GrepArgs),
    /// Filter input to only (non)-matching records.
    Filter(GrepArgs),
    /// Test CPU features and search throughput
    Test,
}

fn main() {
    let args = Args::parse();
    env_logger::init();

    match args {
        Args::Search(search_args) => search(&search_args),
        Args::Crispr(crispr_args) => crispr(&crispr_args),
        Args::Grep(grep_args) => grep_args.grep(),
        Args::Filter(grep_args) => grep_args.filter(),
        Args::Test => {
            sassy::test_cpu_features();
            sassy::test_throughput();
        }
    }
}
