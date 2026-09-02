//! Stats command: compute per-AIR statistics and report constraint/polynomial counts.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use pil2_pilout::pilout as pb;
use prost::Message;

use crate::pil::prepare::PrepareOptions;
use crate::types::stark_struct::{generate_stark_struct, StarkStructsConfig};

/// Options for the stats subcommand.
pub struct StatsOptions {
    /// A built proving key to read as-built circuits from (basic and, with `aggregation`,
    /// the recursion layers). Mutually exclusive with `airout_path`.
    pub proving_key_path: Option<String>,
    /// Include the recursion circuits when reading a proving key.
    pub aggregation: bool,
    /// Path to compiled .pilout file.
    pub airout_path: String,
    /// Hash family (tree/transcript geometry), as in the setup command.
    pub hash: String,
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
}

/// Run the stats pipeline and write the output file.
pub fn run_stats(opts: &StatsOptions) -> Result<()> {
    if opts.proving_key_path.is_some() {
        return run_stats_proving_key(opts);
    }
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

            let stark_struct = generate_stark_struct(&air_settings, n_bits, &opts.hash);

            let prepare_opts = PrepareOptions { debug: false, im_pols_stages: opts.im_pols_stages };

            tracing::info!("Computing stats for air '{}'", air_name);
            let pil_result = crate::pil::info::pil_info(&pilout, ag_idx, air_idx, &stark_struct, &prepare_opts);

            // Both the native and the in-circuit verifier run the same algorithm, so one count
            // serves both; only the price of a single hash differs.
            let family = opts.hash.as_str();
            let geom = crate::verifier_hashes::geometry_for_family(
                &stark_struct,
                &pil_result.setup,
                pil_result.pil_code.ev_map.len(),
            );
            let counts = crate::verifier_hashes::verifier_hashes(&geom, family);
            let verifier_hashes = counts.total().to_string();

            summary_lines.push(format!(
                "{} | {} | {} | verifierHashes: {}",
                airgroup_name, air_name, pil_result.summary, verifier_hashes
            ));

            stats_lines.push(format!("Airgroup: {} Air: {}", airgroup_name, air_name));
            stats_lines.push(format!("Summary: {}", pil_result.summary));
            stats_lines.push(format!("Verifier hashes ({family}):"));
            stats_lines.push(format!(
                "    total: {:>9} | leaf: {:>8} merkle: {:>8} ldt: {:>8} transcript: {:>5} grinding: {} \
                 (arity {}, {} queries, llv {}, {} grinding bits, standalone transcript)",
                counts.total(),
                counts.leaf,
                counts.merkle,
                counts.fri,
                counts.transcript,
                counts.grinding,
                geom.arity,
                geom.n_queries,
                geom.last_level_verification,
                geom.pow_bits,
            ));

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
    println!("------------------------------------------------------------");

    fs::write(&output_path, stats_lines.join("\n"))?;
    println!("Stats written to {}", output_path);

    Ok(())
}

/// Stats from a built proving key: every circuit's as-built `starkinfo.json` (basic airs and,
/// with `--aggregation`, compressor / recursive2 / vadcop_final / vadcop_final_compressed).
/// Unlike the pilout path nothing is re-solved, so post-solve adjustments (A2's query
/// equalization) are reflected; the hash family comes from the key itself.
fn run_stats_proving_key(opts: &StatsOptions) -> Result<()> {
    use proofman_common::{GlobalInfoAir, MpiCtx, ProofCtx, ProofType, Setup, SetupCtx, VerboseMode};
    use proofman_fields::Goldilocks;
    use std::sync::Arc;

    let pk_path = PathBuf::from(opts.proving_key_path.as_ref().unwrap());
    // aggregation = false here: the recursion setups are loaded one by one below, guarded by
    // presence, so a key whose compressed-final layer is missing still reports the rest.
    let pctx = ProofCtx::<Goldilocks>::create_ctx(pk_path, false, VerboseMode::Info, Arc::new(MpiCtx::new()), false)
        .map_err(|e| anyhow::anyhow!("Failed to load proving key: {e:?}"))?;
    let family = pctx.global_info.hash.clone();

    // (name, group, geometry)
    let mut circuits: Vec<(String, String, crate::verifier_hashes::VerifierGeometry)> = Vec::new();

    let sctx = SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Basic, false, &[], &[], false)
        .map_err(|e| anyhow::anyhow!("Failed to load basic setups: {e:?}"))?;
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, air) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((air.name.clone(), "basic".to_string(), geom));
        }
    }

    if opts.aggregation {
        let sctx_compressor =
            SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Compressor, false, &[], &[], false)
                .map_err(|e| anyhow::anyhow!("Failed to load compressor setups: {e:?}"))?;
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, air) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                    let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
                    circuits.push((format!("{}_compressor", air.name), "compression".to_string(), geom));
                }
            }
        }

        // recursive1 shares recursive2's circuit, so one entry covers both.
        let sctx_recursive2 =
            SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Recursive2, false, &[], &[], false)
                .map_err(|e| anyhow::anyhow!("Failed to load recursive2 setups: {e:?}"))?;
        for airgroup_id in 0..pctx.global_info.air_groups.len() {
            let setup = sctx_recursive2.get_setup(airgroup_id, 0).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((format!("Recursive1/2 (airgroup {airgroup_id})"), "aggregation".to_string(), geom));
        }

        for (template, name) in [("vadcop_final", "VadcopFinal"), ("vadcop_final_compressed", "VadcopFinalCompressed")]
        {
            let path = pctx.global_info.get_setup_path(template);
            if !path.with_extension("starkinfo.json").exists() {
                tracing::warn!("Skipping {template}: no starkinfo at {}", path.display());
                continue;
            }
            let proof_type =
                if template == "vadcop_final" { ProofType::VadcopFinal } else { ProofType::VadcopFinalCompressed };
            let setup = Setup::<Goldilocks>::new(
                &path,
                0,
                0,
                &GlobalInfoAir::new(name.to_string()),
                &proof_type,
                false,
                false,
                false,
                false,
                None,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((name.to_string(), "final".to_string(), geom));
        }
    }

    let output_path = opts.output_path.clone().unwrap_or_else(|| "tmp/stats.txt".to_string());
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut stats_lines: Vec<String> = Vec::new();
    let mut summary_lines: Vec<String> = Vec::new();
    for (name, group, geom) in &circuits {
        let counts = crate::verifier_hashes::verifier_hashes(geom, &family);
        let ldt = if geom.stir_rounds.is_empty() { "FRI" } else { "STIR" };
        summary_lines.push(format!(
            "{} | {} | {} | nBitsExt: {} | verifierHashes: {}",
            group,
            name,
            ldt,
            geom.n_bits_ext,
            counts.total()
        ));
        stats_lines.push(format!("Circuit: {name} ({group}, {ldt}, hash {family})"));
        stats_lines.push(format!("Verifier hashes ({family}):"));
        stats_lines.push(format!(
            "    total: {:>9} | leaf: {:>8} merkle: {:>8} ldt: {:>8} transcript: {:>5} grinding: {} \
             (arity {}, {} queries, llv {}, {} grinding bits, standalone transcript)",
            counts.total(),
            counts.leaf,
            counts.merkle,
            counts.fri,
            counts.transcript,
            counts.grinding,
            geom.arity,
            geom.n_queries,
            geom.last_level_verification,
            geom.pow_bits,
        ));
        stats_lines.push(String::new());
    }

    println!("-------------------------- SUMMARY -------------------------");
    for line in &summary_lines {
        println!("{}", line);
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
