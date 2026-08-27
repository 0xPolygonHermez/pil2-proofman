use clap::{Parser, Subcommand};
use pil2_stark_setup::commands::compile_pil::{self as compile_pil_cmd, CompilePilOptions};
use pil2_stark_setup::commands::gen_exps::{run_gen_exps, GenExpsOptions};
use pil2_stark_setup::commands::rebuild_witness::{self as rebuild_witness_cmd, RebuildWitnessOptions};
use pil2_stark_setup::commands::setup::{self, SetupOptions};
use pil2_stark_setup::commands::stats::{self as stats_cmd, StatsOptions};
use pil2_stark_setup::commands::setup_compressed_final::{self as compressed_final_cmd, SetupCompressedFinalOptions};
use pil2_stark_setup::commands::setup_recursive_test::{self as recursive_test_cmd, SetupRecursiveTestOptions};
use pil2_stark_setup::commands::setup_snark::{self as snark_cmd, SetupSnarkOptions};

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
    /// Run only the `vadcop_final_compressed` stage on top of an existing
    /// provingKey/<name>/vadcop_final/. Useful for iterating on this stage.
    SetupCompressedFinal(SetupCompressedFinalArgs),
    /// Set up a test recursive circuit from a user-provided circom file.
    SetupRecursiveTest(SetupRecursiveTestArgs),
    /// Rebuild every witness library (.so/.dylib) in an existing provingKey
    /// without re-running the full setup pipeline.
    RebuildWitnessLibs(RebuildWitnessArgs),
    /// Compile a `.pil` source into a `.pilout` via the JS pil2-compiler.
    CompilePil(CompilePilArgs),
    /// Generate + compile per-AIR Q-expression CUDA kernels (.exps.so) for an
    /// existing provingKey, without re-running setup. No-op if nvcc is absent.
    GenExps(GenExpsArgs),
}

#[derive(Parser)]
struct SetupArgs {
    /// Path to compiled .pilout file
    #[arg(short = 'a', long)]
    airout: String,

    /// Build output directory
    #[arg(short = 'b', long)]
    build_dir: String,

    /// Directory containing fixed column files
    #[arg(short = 'u', long)]
    fixed_dir: Option<String>,

    /// Enable recursive/aggregation setup
    #[arg(short = 'r', long)]
    recursive: bool,

    /// Path to starkstructs.json settings
    #[arg(short = 's', long)]
    stark_structs: Option<String>,

    /// Max concurrent recursive1 air pipelines (default: min(2, num_cpus)).
    /// Each slot runs one circom compile + pil2com and uses ~1-4 GB peak RAM.
    /// Size by available RAM: set to floor(available_GB / per_air_peak_GB).
    #[arg(long, env = "RECURSIVE_JOBS")]
    recursive_jobs: Option<usize>,

    /// Max concurrent AIRs during non-recursive setup (default: min(4, num_cpus)).
    /// Each slot runs pil_info + file I/O and uses ~64 MB stack + working set.
    /// Size by available RAM.
    #[arg(long, env = "SETUP_JOBS")]
    setup_jobs: Option<usize>,

    /// Output file for per-AIR stats (same format as `stats` subcommand).
    /// If omitted, no stats file is written.
    #[arg(short = 'o', long)]
    output: Option<String>,

    #[arg(long, default_value = proofman_common::hash_family::DEFAULT_HASH_ID)]
    hash: String,

    /// Proofs each recursive2 circuit aggregates. Must be 2 or 3. Defaults per hash family --
    /// 2 for blake3, 3 for poseidon -- see hash_family::default_aggregation_arity.
    #[arg(long)]
    agg_arity: Option<usize>,

    /// Pin every recursive air to 2^N rows instead of letting each size itself to its own gate
    /// count. The recursion fixpoint needs recursive1 and recursive2 to be the same size anyway,
    /// so the derived sizes are only ever a lower bound on what the pipeline can use. Setup fails,
    /// naming the air and both sizes, if any air needs more than N -- a pinned size that does not
    /// fit is a wrong answer, not something to silently round up.
    #[arg(long, env = "RECURSIVE_N_BITS")]
    recursive_n_bits: Option<usize>,

    /// Build the `vadcop_final_compressed` stage. Defaults per hash family: on for poseidon, off
    /// for blake3, where it measured a 2% smaller proof for a whole extra recursion layer. A key
    /// built without it can gain it later with `setup-compressed-final`.
    #[arg(long)]
    compressed_final: Option<bool>,

    /// Generate + compile per-AIR Q-expression CUDA kernels (.exps.so) at the end
    /// of setup. No-op if nvcc is not on PATH.
    #[arg(long, default_value_t = false)]
    gen_exps: bool,

    /// CUDA arch spec for --gen-exps: auto | major | "89,120" | sm_120.
    #[arg(long, default_value = "auto")]
    exps_arch: String,

    /// Skip an AIR whose Q has more than N ops (stays on the interpreter).
    #[arg(long, default_value_t = 60000)]
    exps_cap: usize,

    /// Fixed ops/chunk for every AIR; omit to auto-tune the largest no-spill size.
    #[arg(long)]
    exps_chunk: Option<usize>,

    /// pil2-stark source root for the nvcc includes (default: resolved automatically).
    #[arg(long)]
    exps_stark_src: Option<String>,
}

#[derive(Parser)]
struct StatsArgs {
    /// Path to compiled .pilout file
    #[arg(short = 'a', long)]
    airout: String,

    /// Hash family the setup will use; determines tree/transcript geometry.
    #[arg(long, default_value = proofman_common::hash_family::DEFAULT_HASH_ID)]
    hash: String,

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
struct SetupCompressedFinalArgs {
    /// Build directory containing `provingKey/<name>/vadcop_final/`.
    #[arg(short = 'b', long)]
    build_dir: String,
}

#[derive(Parser)]
struct RebuildWitnessArgs {
    /// Path to the `provingKey/` directory.
    #[arg(short = 'p', long = "proving-key")]
    proving_key: String,

    /// Optional path to the `provingKeySnark/` directory. When set, the snark
    /// witness libraries (`recursivef`, `final`) are rebuilt too.
    #[arg(short = 's', long = "proving-key-snark")]
    proving_key_snark: Option<String>,

    /// Max number of witness libraries to compile concurrently. Each `make`
    /// build is g++-bound and uses ~1–2 GB RAM; defaults to the number of
    /// available CPUs. Lower it on memory-constrained machines.
    #[arg(short = 'j', long = "jobs", env = "REBUILD_JOBS")]
    jobs: Option<usize>,
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

    /// Setup type: compressor, aggregation
    #[arg(short = 't', long, default_value = "aggregation")]
    r#type: String,

    #[arg(long, default_value = proofman_common::hash_family::DEFAULT_HASH_ID)]
    hash: String,

    #[arg(long)]
    blake3_lanes: Option<usize>,

    /// Generate + compile per-AIR Q-expression CUDA kernels (.exps.so) at the end.
    /// No-op if nvcc is not on PATH.
    #[arg(long, default_value_t = false)]
    gen_exps: bool,

    /// CUDA arch spec for --gen-exps: auto | major | "89,120" | sm_120.
    #[arg(long, default_value = "auto")]
    exps_arch: String,

    /// Skip an AIR whose Q has more than N ops (stays on the interpreter).
    #[arg(long, default_value_t = 60000)]
    exps_cap: usize,

    /// Fixed ops/chunk for every AIR; omit to auto-tune the largest no-spill size.
    #[arg(long)]
    exps_chunk: Option<usize>,

    /// pil2-stark source root for the nvcc includes (default: resolved automatically).
    #[arg(long)]
    exps_stark_src: Option<String>,
}

#[derive(Parser)]
struct CompilePilArgs {
    /// Path to the entry `.pil` file
    #[arg(short = 'p', long = "pil")]
    pil_path: String,

    /// Output `.pilout` path
    #[arg(short = 'o', long = "output")]
    output_path: String,

    /// `-I` include search paths (repeat for multiple, or pass a comma-separated value)
    #[arg(short = 'I', long = "include", num_args = 1.., value_delimiter = ',')]
    include_paths: Vec<String>,

    /// `-u` directory for fixed columns
    #[arg(short = 'u', long = "fixed-dir")]
    fixed_dir: Option<String>,

    /// Pass `-O fixed-to-file` to write fixed columns to disk
    #[arg(long = "fixed-to-file")]
    fixed_to_file: bool,

    /// Pass `-O no-proto-fixed-data` to omit fixed-column values from the pilout
    /// protobuf. Use with `--fixed-dir` + `--fixed-to-file` to avoid the V8 heap
    /// blowup on huge PILs (e.g. zisk.pil at ~9 M Keccakf constraints).
    #[arg(long = "no-proto-fixed-data")]
    no_proto_fixed_data: bool,
}

#[derive(Parser)]
struct GenExpsArgs {
    /// Path to an existing `provingKey/` directory (globbed for each AIR's
    /// `*.starkinfo.json` + `*.expressionsinfo.json`).
    #[arg(short = 'p', long = "proving-key")]
    proving_key: String,

    /// CUDA arch spec: auto | major | "89,120" | sm_120.
    #[arg(long, default_value = "auto")]
    exps_arch: String,

    /// Skip an AIR whose Q has more than N ops (stays on the interpreter).
    #[arg(long, default_value_t = 60000)]
    exps_cap: usize,

    /// Fixed ops/chunk for every AIR; omit to auto-tune the largest no-spill size.
    #[arg(long)]
    exps_chunk: Option<usize>,

    /// pil2-stark source root for the nvcc includes (default: resolved automatically).
    #[arg(long)]
    exps_stark_src: Option<String>,
}

fn main() -> anyhow::Result<()> {
    // The shared formatter, not a plain `fmt()`: it is the one the proofman entry points install, it
    // renders the span scope, and `initialize_logger` no-ops if a dispatcher is already set -- so
    // installing a different one here silently gave this binary a different log format from every
    // other, which is what hid the per-air spans.
    // RUST_LOG still selects the level, as it did under `fmt::init()`: this binary has no verbosity
    // flag, and the shared initializer takes a mode rather than reading the environment.
    let verbose = match std::env::var("RUST_LOG").unwrap_or_default().to_ascii_lowercase() {
        v if v.contains("trace") => proofman_common::VerboseMode::Trace,
        v if v.contains("debug") => proofman_common::VerboseMode::Debug,
        _ => proofman_common::VerboseMode::Info,
    };
    proofman_common::initialize_logger(verbose, None);

    let builder = rayon::ThreadPoolBuilder::new().stack_size(64 * 1024 * 1024); // 64 MB per thread
    builder.build_global().ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Setup(args) => {
            let available = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            // Conservative defaults: setup_jobs caps at 4 because pil_info uses
            // 64 MB stack × N slots and large airs hold working sets in memory;
            // recursive_jobs caps at 2 because each slot can use multiple GB
            // during plonk2pil. Both are overridable via env or --flag.
            let setup_jobs = args.setup_jobs.unwrap_or(available.min(4));
            let recursive_jobs = args.recursive_jobs.unwrap_or(available.min(2));

            tracing::info!("proofman-setup setup: starting");
            tracing::info!("  airout: {}", args.airout);
            tracing::info!("  build_dir: {}", args.build_dir);
            tracing::info!("  recursive: {}", args.recursive);
            tracing::info!("  setup_jobs: {} (override with --setup-jobs or SETUP_JOBS env)", setup_jobs);
            tracing::info!(
                "  recursive_jobs: {} (override with --recursive-jobs or RECURSIVE_JOBS env)",
                recursive_jobs
            );

            if !proofman_common::hash_family::is_known_family(&args.hash) {
                anyhow::bail!("unknown --hash {:?}; known: {:?}", args.hash, proofman_common::hash_family::FAMILIES);
            }

            let agg_arity =
                args.agg_arity.unwrap_or_else(|| proofman_common::hash_family::default_aggregation_arity(&args.hash));
            let compressed_final = args
                .compressed_final
                .unwrap_or_else(|| proofman_common::hash_family::compressed_final_by_default(&args.hash));
            if !proofman_common::global_info::is_valid_aggregation_arity(agg_arity) {
                anyhow::bail!(
                    "unsupported --agg-arity {}; valid values: {:?}",
                    agg_arity,
                    proofman_common::global_info::VALID_AGGREGATION_ARITIES
                );
            }

            let opts = SetupOptions {
                airout_path: args.airout,
                build_dir: args.build_dir,
                fixed_dir: args.fixed_dir,
                stark_structs_path: args.stark_structs,
                recursive: args.recursive,
                recursive_jobs,
                setup_jobs,
                stats_output_path: args.output,
                hash: args.hash,
                agg_arity,
                recursive_n_bits: args.recursive_n_bits,
                compressed_final,
                gen_exps: args.gen_exps,
                exps_arch: args.exps_arch,
                exps_cap: args.exps_cap,
                exps_chunk: args.exps_chunk,
                exps_stark_src: args.exps_stark_src,
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
            if !proofman_common::hash_family::is_known_family(&args.hash) {
                anyhow::bail!("unknown --hash {:?}; known: {:?}", args.hash, proofman_common::hash_family::FAMILIES);
            }
            let opts = StatsOptions {
                airout_path: args.airout,
                hash: args.hash,
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
                powers_of_tau: args.powers_of_tau,
                final_snark: args.final_snark,
                publics_info: args.publics_info,
                only_recursive_final: args.only_recursive_final,
            };
            snark_cmd::run_setup_snark(&opts)
        }

        Commands::SetupCompressedFinal(args) => {
            tracing::info!("proofman-setup setup-compressed-final: starting");
            tracing::info!("  build_dir: {}", args.build_dir);
            let opts = SetupCompressedFinalOptions { build_dir: args.build_dir };
            compressed_final_cmd::run_setup_compressed_final(&opts)
        }

        Commands::RebuildWitnessLibs(args) => {
            tracing::info!("proofman-setup rebuild-witness-libs: starting");
            tracing::info!("  proving_key: {}", args.proving_key);
            if let Some(ref pks) = args.proving_key_snark {
                tracing::info!("  proving_key_snark: {}", pks);
            }
            let jobs = args.jobs.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            tracing::info!("  jobs: {} (override with --jobs or REBUILD_JOBS env)", jobs);
            let opts = RebuildWitnessOptions {
                proving_key: args.proving_key,
                proving_key_snark: args.proving_key_snark,
                jobs,
            };
            rebuild_witness_cmd::run_rebuild_witness(&opts)
        }

        Commands::SetupRecursiveTest(args) => {
            tracing::info!("proofman-setup setup-recursive-test: starting");
            tracing::info!("  build_dir: {}", args.build_dir);
            tracing::info!("  circom: {}", args.circom_path);
            tracing::info!("  name: {}", args.circom_name);
            tracing::info!("  type: {}", args.r#type);
            if !proofman_common::hash_family::is_known_family(&args.hash) {
                anyhow::bail!("unknown --hash {:?}; known: {:?}", args.hash, proofman_common::hash_family::FAMILIES);
            }
            let build_dir = args.build_dir.clone();
            if let Some(l) = args.blake3_lanes {
                if !(1..=8).contains(&l) {
                    anyhow::bail!("--blake3-lanes must be in 1..8 (the air's boundary depth caps it), got {l}");
                }
                tracing::info!("  blake3_lanes: {l}");
            }
            let opts = SetupRecursiveTestOptions {
                build_dir: args.build_dir,
                circom_path: args.circom_path,
                circom_name: args.circom_name,
                setup_type: args.r#type,
                hash: args.hash,
                blake3_lanes: args.blake3_lanes,
            };
            recursive_test_cmd::run_setup_recursive_test(&opts)?;

            // Optionally generate the per-AIR Q-expression CUDA kernels for the
            // provingKey just produced. No-op when nvcc is absent.
            if args.gen_exps {
                let gen_opts = GenExpsOptions {
                    proving_key: std::path::PathBuf::from(&build_dir).join("provingKey"),
                    arch: args.exps_arch,
                    cap: args.exps_cap,
                    chunk: args.exps_chunk,
                    stark_src: args.exps_stark_src.map(std::path::PathBuf::from),
                };
                run_gen_exps(&gen_opts)?;
            }
            Ok(())
        }

        Commands::CompilePil(args) => {
            tracing::info!("proofman-setup compile-pil: starting");
            tracing::info!("  pil: {}", args.pil_path);
            tracing::info!("  output: {}", args.output_path);
            let opts = CompilePilOptions {
                pil_path: args.pil_path,
                output_path: args.output_path,
                include_paths: args.include_paths,
                fixed_dir: args.fixed_dir,
                fixed_to_file: args.fixed_to_file,
                no_proto_fixed_data: args.no_proto_fixed_data,
            };
            compile_pil_cmd::run_compile_pil(&opts)
        }

        Commands::GenExps(args) => {
            tracing::info!("proofman-setup gen-exps: starting");
            tracing::info!("  proving-key: {}", args.proving_key);
            let gen_opts = GenExpsOptions {
                proving_key: std::path::PathBuf::from(&args.proving_key),
                arch: args.exps_arch,
                cap: args.exps_cap,
                chunk: args.exps_chunk,
                stark_src: args.exps_stark_src.map(std::path::PathBuf::from),
            };
            run_gen_exps(&gen_opts)
        }
    }
}
