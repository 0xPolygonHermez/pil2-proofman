use clap::{Parser, Subcommand};
use pil2_stark_setup::commands::setup::{self, SetupOptions};
use pil2_stark_setup::commands::stats::{self as stats_cmd, StatsOptions};
use pil2_stark_setup::commands::setup_snark::{self as snark_cmd, SetupSnarkOptions};
use pil2_stark_setup::commands::setup_recursive_test::{self as recursive_test_cmd, SetupRecursiveTestOptions};

// Uses the default system allocator (glibc malloc). After each AIR,
// setup_cmd calls malloc_trim(0) to return freed pages to the OS and
// keep peak RSS bounded across the AIR sequence.

#[derive(Parser)]
#[command(name = "proofman-setup", about = "Proving key setup")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run non-recursive (and optionally recursive) setup for all AIRs.
    Setup(SetupArgs),
    /// Compute per-AIR statistics (constraints, intermediate polynomials, etc.).
    Stats(StatsArgs),
    /// Generate final SNARK setup (recursivef + fflonk/plonk final).
    SetupSnark(SetupSnarkArgs),
    /// Set up a test recursive circuit from a user-provided circom file.
    SetupRecursiveTest(SetupRecursiveTestArgs),
}

#[derive(Parser)]
struct SetupArgs {
    /// Path to compiled .pilout file
    #[arg(short = 'a', long)]
    airout: String,

    /// Build output directory
    #[arg(short = 'b', long)]
    build_dir: String,

    /// Standard PIL library path
    #[arg(short = 't', long)]
    std_path: Option<String>,

    /// Directory containing fixed column files
    #[arg(short = 'u', long)]
    fixed_dir: Option<String>,

    /// Enable recursive/aggregation setup
    #[arg(short = 'r', long)]
    recursive: bool,

    /// Path to starkstructs.json settings
    #[arg(short = 's', long)]
    stark_structs: Option<String>,

    /// Max concurrent recursive1 air pipelines (default 1 = serial).
    /// Each slot runs one circom compile + pil2com. Size by available RAM:
    /// set to floor(available_GB / per_air_peak_GB).
    #[arg(long, default_value_t = 1, env = "RECURSIVE_JOBS")]
    recursive_jobs: usize,

    /// Max concurrent AIRs during non-recursive setup (default 1 = serial).
    /// Each slot runs pil_info + file I/O. Size by available RAM.
    #[arg(long, default_value_t = 1, env = "SETUP_JOBS")]
    setup_jobs: usize,

    /// Output file for per-AIR stats (same format as `stats` subcommand).
    /// If omitted, no stats file is written.
    #[arg(short = 'o', long)]
    output: Option<String>,
}

#[derive(Parser)]
struct StatsArgs {
    /// Path to compiled .pilout file
    #[arg(short = 'a', long)]
    airout: String,

    /// Output file for detailed stats (default: tmp/stats.txt)
    #[arg(short = 'o', long)]
    output: Option<String>,

    /// Path to starkstructs.json settings
    #[arg(short = 's', long)]
    starkstructs: Option<String>,

    /// Filter by airgroup names (repeat for multiple)
    #[arg(short = 'g', long = "airgroups", num_args = 1..)]
    airgroups: Vec<String>,

    /// Filter by air names (repeat for multiple)
    #[arg(short = 'i', long = "airs", num_args = 1..)]
    airs: Vec<String>,

    /// Show intermediate polynomial details per stage
    #[arg(short = 'm', long)]
    impols: bool,
}

#[derive(Parser)]
struct SetupSnarkArgs {
    /// Build directory (must already contain provingKey/ from a previous setup run)
    #[arg(short = 'b', long)]
    build_dir: String,

    /// Standard PIL library path
    #[arg(short = 't', long)]
    std_path: Option<String>,

    /// Powers-of-tau (.ptau) file for snarkjs setup
    #[arg(long)]
    powers_of_tau: Option<String>,

    /// Final SNARK type: fflonk (default) or plonk
    #[arg(long, default_value = "fflonk")]
    final_snark: String,

    /// Path to publics hash info JSON (optional)
    #[arg(long)]
    publics_info: Option<String>,

    /// Only generate the recursivef step; skip the final SNARK
    #[arg(long)]
    only_recursive_final: bool,
}

#[derive(Parser)]
struct SetupRecursiveTestArgs {
    /// Build output directory
    #[arg(short = 'b', long)]
    build_dir: String,

    /// Path to the circom source file
    #[arg(short = 'c', long = "circom")]
    circom_path: String,

    /// Circuit name (e.g. "test")
    #[arg(short = 'n', long = "name")]
    circom_name: String,

    /// Standard PIL library path
    #[arg(short = 'p', long, default_value = "pil2-components/lib/std/pil")]
    std_path: String,

    /// Setup type: compressor, aggregation, final_vadcop, or light
    #[arg(short = 't', long, default_value = "aggregation")]
    r#type: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let builder = rayon::ThreadPoolBuilder::new().stack_size(64 * 1024 * 1024); // 64 MB per thread
    builder.build_global().ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Setup(args) => {
            tracing::info!("proofman-setup setup: starting");
            tracing::info!("  airout: {}", args.airout);
            tracing::info!("  build_dir: {}", args.build_dir);
            tracing::info!("  recursive: {}", args.recursive);

            let opts = SetupOptions {
                airout_path: args.airout,
                build_dir: args.build_dir,
                fixed_dir: args.fixed_dir,
                stark_structs_path: args.stark_structs,
                recursive: args.recursive,
                std_pil_path: args.std_path,
                recursive_jobs: args.recursive_jobs,
                setup_jobs: args.setup_jobs,
                stats_output_path: args.output,
            };

            let result = setup::run_setup(&opts);

            // Log peak memory at exit for measurement validation
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmHWM:") || line.starts_with("VmPeak:") {
                        tracing::info!("{}", line.trim());
                    }
                }
            }

            result
        }

        Commands::Stats(args) => {
            tracing::info!("proofman-setup stats: starting");
            let opts = StatsOptions {
                airout_path: args.airout,
                output_path: args.output,
                stark_structs_path: args.starkstructs,
                airgroups: args.airgroups,
                airs: args.airs,
                im_pols_stages: args.impols,
            };
            // Expression trees in large AIRs (e.g. ZisK) can be thousands of levels deep,
            // which overflows the default 8 MB main-thread stack. Run on a thread with the
            // same 64 MB stack used by the rayon pool above.
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || stats_cmd::run_stats(&opts))
                .expect("failed to spawn stats thread")
                .join()
                .expect("stats thread panicked")
        }

        Commands::SetupSnark(args) => {
            tracing::info!("proofman-setup setup-snark: starting");
            let opts = SetupSnarkOptions {
                build_dir: args.build_dir,
                std_pil_path: args.std_path,
                powers_of_tau: args.powers_of_tau,
                final_snark: args.final_snark,
                publics_info: args.publics_info,
                only_recursive_final: args.only_recursive_final,
            };
            snark_cmd::run_setup_snark(&opts)
        }

        Commands::SetupRecursiveTest(args) => {
            tracing::info!("proofman-setup setup-recursive-test: starting");
            tracing::info!("  build_dir: {}", args.build_dir);
            tracing::info!("  circom: {}", args.circom_path);
            tracing::info!("  name: {}", args.circom_name);
            tracing::info!("  type: {}", args.r#type);
            let opts = SetupRecursiveTestOptions {
                build_dir: args.build_dir,
                circom_path: args.circom_path,
                circom_name: args.circom_name,
                std_pil_path: args.std_path,
                setup_type: args.r#type,
            };
            recursive_test_cmd::run_setup_recursive_test(&opts)
        }
    }
}
