use clap::{Parser, Subcommand};

mod sassy1;
mod sassy2;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run sassy1 benchmarks
    Sassy1 {
        #[command(subcommand)]
        command: Sassy1Commands,
    },

    /// Run sassy2 benchmarks
    Sassy2 {
        #[command(subcommand)]
        command: Sassy2Commands,
    },
}

#[derive(Subcommand)]
enum Sassy1Commands {
    /// Run the edlib grid benchmark
    Edlib {
        /// Path to the grid config TOML file
        #[arg(long)]
        config: String,
    },

    /// Run the overhang benchmark
    Overhang {
        /// Path to the overhang config TOML file
        #[arg(long)]
        config: String,
    },

    /// Run the profiles benchmark
    Profiles {
        /// Path to the profiles config TOML file
        #[arg(long)]
        config: String,
    },

    /// Run the agrep comparison benchmark
    Agrep {
        /// Path to the agrep config TOML file
        #[arg(long)]
        config: String,
    },
}

#[derive(Subcommand)]
enum Sassy2Commands {
    /// Run the pattern throughput benchmark
    PatternThroughput {
        /// Path to the pattern throughput config TOML file
        #[arg(long)]
        config: String,
    },

    /// Run the scaling throughput benchmark
    ScalingBenchmark {
        /// Path to the scaling benchmark config TOML file
        #[arg(long)]
        config: String,
    },
}

fn main() {
    let args = Args::parse();
    match args.command {
        Commands::Sassy1 { command } => match command {
            Sassy1Commands::Edlib { config } => {
                println!("Running sassy1 edlib grid");
                sassy1::edlib_bench::runner::run(&config);
            }
            Sassy1Commands::Overhang { config } => {
                println!("Running sassy1 overhang benchmark");
                sassy1::overhang::runner::run(&config);
            }
            Sassy1Commands::Profiles { config } => {
                println!("Running sassy1 profiles benchmark");
                sassy1::profiles::runner::run(&config);
            }
            Sassy1Commands::Agrep { config } => {
                println!("Running sassy1 agrep comparison benchmark");
                sassy1::agrep_comparison::runner::run(&config);
            }
        },

        Commands::Sassy2 { command } => match command {
            Sassy2Commands::PatternThroughput { config } => {
                println!("Running sassy2 pattern throughput benchmark");
                sassy2::pattern_throughput::run(&config);
            }
            Sassy2Commands::ScalingBenchmark { config } => {
                println!("Running sassy2 scaling throughput benchmark");
                sassy2::scaling_benchmark::run(&config);
            }
        },
    }
}
