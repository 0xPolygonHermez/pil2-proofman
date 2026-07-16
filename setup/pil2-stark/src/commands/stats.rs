//! Stats command: compute per-AIR statistics and report constraint/polynomial counts.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use pilout::pilout as pb;
use prost::Message;
use proofman_multilinear::{build_kernels, referenced_leaves, AirIr, Boundary, KernelSpec, MERKLE_ARITY};

use crate::pil::prepare::PrepareOptions;
use crate::types::stark_struct::{generate_stark_struct, StarkStructsConfig};

/// Options for the stats subcommand.
pub struct StatsOptions {
    /// Path to compiled .pilout file.
    pub airout_path: String,
    /// Output file for detailed per-AIR stats (default: `tmp/stats.txt`).
    pub output_path: Option<String>,
    /// Optional path to starkstructs.json.
    pub stark_structs_path: Option<String>,
    /// Airgroup name filter (empty = all airgroups).
    pub airgroups: Vec<String>,
    /// Air name filter (empty = all airs).
    pub airs: Vec<String>,
    /// Whether to show intermediate polynomial details per stage.
    pub im_pols_stages: bool,
    /// Report the multilinear prover's view (committed columns, zerocheck,
    /// LogUp-GKR bus, estimated proof size and prover memory) instead of the
    /// univariate layout.
    pub multilinear: bool,
}

/// Run the stats pipeline and write the output file.
pub fn run_stats(opts: &StatsOptions) -> Result<()> {
    let pilout_data = fs::read(&opts.airout_path)?;
    let pilout = pb::PilOut::decode(pilout_data.as_slice())?;

    let settings_map: StarkStructsConfig = if let Some(ref settings_path) = opts.stark_structs_path {
        let data = fs::read_to_string(settings_path)?;
        StarkStructsConfig::from_json_str(&data)?
    } else {
        StarkStructsConfig::default()
    };

    let output_path = opts.output_path.clone().unwrap_or_else(|| "tmp/stats.txt".to_string());

    if let Some(parent) = PathBuf::from(&output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut stats_lines: Vec<String> = Vec::new();
    let mut summary_lines: Vec<String> = Vec::new();
    // Multilinear totals: (estimated proof bytes, estimated prover memory
    // bytes, committed trace bytes, number of GKR-bus AIRs).
    let mut ml_totals = (0u64, 0u64, 0u64, 0usize);

    for (ag_idx, airgroup) in pilout.air_groups.iter().enumerate() {
        let airgroup_name = airgroup.name.clone().unwrap_or_else(|| format!("airgroup_{}", ag_idx));

        if !opts.airgroups.is_empty() && !opts.airgroups.contains(&airgroup_name) {
            tracing::info!("Skipping airgroup '{}'", airgroup_name);
            continue;
        }

        for (air_idx, air) in airgroup.airs.iter().enumerate() {
            let air_name = air.name.clone().unwrap_or_else(|| format!("air_{}", air_idx));

            if !opts.airs.is_empty() && !opts.airs.contains(&air_name) {
                tracing::info!("Skipping air '{}'", air_name);
                continue;
            }

            let num_rows = air.num_rows.unwrap_or(0) as usize;
            if num_rows == 0 {
                tracing::warn!("Skipping air '{}' with numRows=0", air_name);
                continue;
            }

            let n_bits = log2_usize(num_rows);

            let air_settings = settings_map.resolve(&airgroup_name, &air_name);

            let stark_struct = generate_stark_struct(&air_settings, n_bits);

            let prepare_opts = PrepareOptions { debug: false, im_pols_stages: opts.im_pols_stages };

            tracing::info!("Computing stats for air '{}'", air_name);
            let pil_result = crate::pil::info::pil_info(&pilout, ag_idx, air_idx, &stark_struct, &prepare_opts);

            if opts.multilinear {
                match crate::output::mlinfo::build_air_ir(
                    &pil_result.setup,
                    n_bits as u32,
                    proofman_multilinear::MlParams::default(),
                ) {
                    Ok(mut ir) => {
                        ir.params = crate::commands::setup::ml_params(&stark_struct, n_bits, ir.total_cols());
                        let ml = ml_air_stats(&ir);
                        summary_lines.push(format!("{} | {} | {}", airgroup_name, air_name, ml.summary));
                        stats_lines.push(format!("Airgroup: {} Air: {}", airgroup_name, air_name));
                        stats_lines.extend(ml.details);
                        stats_lines.push(String::new());
                        ml_totals.0 += ml.proof_bytes;
                        ml_totals.1 += ml.mem_bytes;
                        ml_totals.2 += ml.trace_bytes;
                        ml_totals.3 += ir.bus.is_some() as usize;
                    }
                    Err(e) => {
                        summary_lines
                            .push(format!("{} | {} | not multilinear-provable ({e})", airgroup_name, air_name));
                    }
                }
                continue;
            }

            let summary = format!("{} | {} | {}", airgroup_name, air_name, pil_result.summary);
            summary_lines.push(summary.clone());

            stats_lines.push(format!("Airgroup: {} Air: {}", airgroup_name, air_name));
            stats_lines.push(format!("Summary: {}", pil_result.summary));

            let (base_field, extended_field) = &pil_result.im_pols_info;
            if !base_field.is_empty() {
                stats_lines.push("Intermediate polynomials baseField:".to_string());
                for pol in base_field {
                    stats_lines.push(format!("    {}", pol));
                }
            }
            if !extended_field.is_empty() {
                stats_lines.push("Intermediate polynomials extendedField:".to_string());
                for pol in extended_field {
                    stats_lines.push(format!("    {}", pol));
                }
            }
            stats_lines.push(String::new());
        }
    }

    println!("-------------------------- SUMMARY -------------------------");
    for line in &summary_lines {
        println!("{}", line);
    }
    if opts.multilinear {
        println!("------------------------------------------------------------");
        println!(
            "TOTAL (one instance per AIR) | committed trace: {} | est. proof set: {} | est. prover memory: {} | GKR-bus AIRs: {}",
            fmt_bytes(ml_totals.2),
            fmt_bytes(ml_totals.0),
            fmt_bytes(ml_totals.1),
            ml_totals.3,
        );
    }
    println!("------------------------------------------------------------");

    fs::write(&output_path, stats_lines.join("\n"))?;
    println!("Stats written to {}", output_path);

    Ok(())
}

/// floor(log2(n)) for a nonzero usize.
fn log2_usize(n: usize) -> usize {
    assert!(n > 0, "log2_usize: n must be positive");
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

// ---------------------------------------------------------------------------
// Multilinear view
// ---------------------------------------------------------------------------

const GL: u64 = 8; // Goldilocks element
const EXT: u64 = 24; // cubic extension element
const DIGEST: u64 = 32; // Poseidon2 digest ([Goldilocks; 4])

struct MlAirStats {
    summary: String,
    details: Vec<String>,
    proof_bytes: u64,
    mem_bytes: u64,
    trace_bytes: u64,
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.2} MB", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{:.1} KB", b as f64 / (1u64 << 10) as f64)
    }
}

/// Per-AIR statistics of the multilinear prover: what gets committed, what
/// the zerocheck and LogUp-GKR phases cost, and the estimated proof size and
/// prover memory. All sizes are analytic estimates from the IR and
/// `MlParams` — no proving key needed.
fn ml_air_stats(ir: &AirIr) -> MlAirStats {
    let n_bits = ir.n_bits as u64;
    let n = 1u64 << n_bits;
    let p = &ir.params;
    let lb = p.log_blowup as u64;

    // --- Committed data ---
    let committed = ir.total_witness_cols() as u64;
    let fixed = ir.n_const_cols as u64;
    let custom: u64 = ir.custom_commits.iter().map(|c| c.n_cols as u64).sum();
    let total_cols = ir.total_cols() as u64;
    let stages = ir.cols_per_stage.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("+");
    let trace_bytes = total_cols * n * GL;

    // --- Zerocheck ---
    let every_row = ir.constraints.iter().filter(|c| c.boundary == Boundary::EveryRow).count() as u64;
    let d = ir.max_constraint_degree as u64;
    let evals = referenced_leaves(ir).len() as u64;
    let l = p.univariate_skip_bits.min(ir.n_bits as usize) as u64;
    let zc_bytes = if l > 0 { (d * ((1 << l) - 1) + 1) * EXT } else { 0 } + (n_bits - l) * d * EXT;

    // --- Kernels ---
    let kernels = build_kernels(ir);
    let n_kernels = kernels.len() as u64;
    let (mut k_rot, mut k_bus) = (0u64, 0u64);
    for k in &kernels {
        match k {
            KernelSpec::Rot(_) => k_rot += 1,
            KernelSpec::BusRot(_) => k_bus += 1,
            KernelSpec::Point(_) => {}
        }
    }

    // --- LogUp-GKR bus ---
    let (bus_summary, bus_detail, bus_bytes, tree_mem) = match &ir.bus {
        Some(bus) => {
            let m = bus.terms.len() as u64;
            let s = bus.scalar_terms.len() as u64;
            let cap = bus.terms.len().next_power_of_two() as u64;
            let nu = cap.trailing_zeros() as u64;
            let layers = n_bits + nu;
            let walk_rounds = layers * (layers - 1) / 2;
            let d_bus = bus.max_term_degree.max(1) as u64;
            // Input layer cap·n fractions; p base + q ext + internal p/q
            // layers (geometric, ≈ one extra input-size each) ≈ 80 B/fraction.
            let tree_mem = 80 * cap * n;
            // p_out/q_out + walk rounds (2 Exts) + 4 split values per layer +
            // the input-reduction rounds.
            let bytes = 2 * EXT + walk_rounds * 2 * EXT + layers * 4 * EXT + n_bits * d_bus * EXT;
            (
                format!("Bus: {m}+{s} terms (max deg {d_bus}) GKR tree 2^{layers}", ),
                format!(
                    "Bus: {m} row terms + {s} direct terms | max term degree: {d_bus} | input layer: 2^{} fractions | {layers} GKR layers, {walk_rounds} walk rounds | tree memory: {}",
                    n_bits + nu,
                    fmt_bytes(tree_mem),
                ),
                bytes,
                tree_mem,
            )
        }
        None => ("Bus: -".to_string(), "Bus: none".to_string(), 0, 0),
    };

    // --- Proof-size estimate ---
    let n_matrices = ir.n_stages() as u64 + 1 + ir.custom_commits.len() as u64;
    let claims_bytes = total_cols * n_kernels * EXT;
    let red_bytes = n_bits * 2 * EXT;
    let folds = p.num_folds(ir.n_bits as usize) as u64;
    let n0 = n_bits + lb;
    let arity_log = MERKLE_ARITY.trailing_zeros() as u64;
    let path_bytes = |leaf_bits: u64| leaf_bits.div_ceil(arity_log) * (MERKLE_ARITY - 1) * DIGEST;
    // Per query: pair-packed stage leaves + one path per matrix, then one
    // (value, path) per committed fold oracle.
    let per_query = 2 * total_cols * GL
        + n_matrices * path_bytes(n0.saturating_sub(1))
        + (1..folds).map(|f| 2 * EXT + path_bytes(n0.saturating_sub(f + 1))).sum::<u64>();
    let opening_bytes = n_bits * EXT
        + folds.saturating_sub(1) * DIGEST
        + (1u64 << (n_bits - folds)) * EXT
        + p.n_queries as u64 * per_query;
    let proof_bytes = n_matrices * DIGEST + zc_bytes + bus_bytes + claims_bytes + red_bytes + opening_bytes;

    // --- Prover-memory estimate ---
    // Trace + committed codewords/leaves (~2× the encoded trace) + zerocheck
    // leaf tables (round 0, base field) + the fraction tree.
    let commit_mem = 2 * total_cols * (n << lb) * GL;
    let zc_mem = evals * n * GL;
    let mem_bytes = trace_bytes + commit_mem + zc_mem + tree_mem;

    let summary = format!(
        "Bits: {n_bits} | BF: {lb} | Committed: {committed} ({stages}) | Fixed: {fixed} | Custom: {custom} | Constraints: {every_row} (max deg {d}) | Evals: {evals} | Kernels: {k_rot}R+{k_bus}B | {bus_summary} | Est. proof: {} | Est. memory: {}",
        fmt_bytes(proof_bytes),
        fmt_bytes(mem_bytes),
    );

    let details = vec![
        format!(
            "Committed: {committed} witness ({stages} per stage) + {fixed} fixed + {custom} custom = {total_cols} columns | trace: {} | encoded (2^{lb}x + leaves): {}",
            fmt_bytes(trace_bytes),
            fmt_bytes(commit_mem),
        ),
        format!(
            "Zerocheck: {every_row} Constraints | Max Degree: {d} | Evaluations: {evals} | instrs: {} (temps: {}) | skip bits: {l}",
            ir.instrs.len(),
            ir.n_temps,
        ),
        bus_detail,
        format!("Kernels: {k_rot} rotation + {k_bus} bus"),
        format!(
            "Est. proof: {} = roots {} + zerocheck {} + bus {} + claims {} ({total_cols} cols x {n_kernels} kernels) + reduction {} + opening {} ({} queries, {folds} folds)",
            fmt_bytes(proof_bytes),
            fmt_bytes(n_matrices * DIGEST),
            fmt_bytes(zc_bytes),
            fmt_bytes(bus_bytes),
            fmt_bytes(claims_bytes),
            fmt_bytes(red_bytes),
            fmt_bytes(opening_bytes),
            p.n_queries,
        ),
        format!(
            "Est. prover memory: {} = trace {} + commitment {} + zerocheck tables {} + fraction tree {}",
            fmt_bytes(mem_bytes),
            fmt_bytes(trace_bytes),
            fmt_bytes(commit_mem),
            fmt_bytes(zc_mem),
            fmt_bytes(tree_mem),
        ),
    ];

    MlAirStats { summary, details, proof_bytes, mem_bytes, trace_bytes }
}
