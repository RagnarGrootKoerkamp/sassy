mod crispr;
mod grep;
mod input_iterator;

use clap::Parser;
use crispr::{CrisprArgs, crispr};
use grep::GrepArgs;

#[derive(clap::Parser)]
#[command(author, version, about)]
enum Args {
    /// Search and print matches of a pattern.
    ///
    /// Fasta/Fastq record based for DNA/IUPAC, line-based for ASCII.
    Grep(GrepArgs),
    /// Like Grep, but only output tsv of matches.
    Search(GrepArgs),
    /// Like Grep, but only output matching records.
    Filter(GrepArgs),
    /// CRISPR-specific search with PAM and edit-free region
    Crispr(CrisprArgs),
    /// Test CPU features and search throughput
    Test,
}

fn main() {
    let args = Args::parse();
    env_logger::init();

    match args {
        Args::Grep(grep_args) => grep_args.grep(),
        Args::Search(grep_args) => grep_args.search(),
        Args::Filter(grep_args) => grep_args.filter(),
        Args::Crispr(crispr_args) => crispr(&crispr_args),
        Args::Test => {
            sassy::test_cpu_features();
            sassy::test_throughput();
        }
    }
}
