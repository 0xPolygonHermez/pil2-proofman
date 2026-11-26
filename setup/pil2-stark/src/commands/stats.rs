//! Stats command: compute per-AIR statistics and report constraint/polynomial counts.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use pil2_pilout::pilout as pb;
use prost::Message;

use crate::commands::common::push_im_pols_stats;
use pil_info::PrepareOptions;
use crate::types::stark_struct::{generate_basic_stark_struct, StarkStructsConfig};

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
    pub hash: proofman_common::hash_family::HashFamily,
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
    /// Lanes the blake3 recursion is built at: compressions per block.
    pub blake3_lanes: usize,
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

            let n_bits = crate::proving_key::util::exact_log2(num_rows as u64)
                .with_context(|| format!("air '{air_name}' row count"))?;

            let air_settings = settings_map.resolve(&airgroup_name, &air_name);

            let stark_struct = generate_basic_stark_struct(&air_settings, n_bits, opts.hash);

            let prepare_opts = PrepareOptions { im_pols_stages: opts.im_pols_stages, inplace_stage_commit: true };

            tracing::info!("Computing stats for air '{}'", air_name);
            let pil_result = crate::pil::stark_setup::stark_setup(
                &pilout,
                ag_idx,
                air_idx,
                &stark_struct,
                &prepare_opts,
                &Default::default(),
            )?;

            // Both the native and the in-circuit verifier run the same algorithm, so one count
            // serves both; only the price of a single hash differs.
            let geom = crate::verifier_hashes::geometry_from_starkinfo(&serde_json::to_value(&pil_result.starkinfo)?)?;
            let counts = crate::verifier_hashes::verifier_hashes(&geom, opts.hash);
            let verifier_hashes = counts.total().to_string();

            // Only blake3 sizes its recursion by hashing blocks, so only there can hashes answer this.
            let fit = opts.hash.is_blake3().then(|| {
                let f = crate::verifier_hashes::blake3_recursion_fit(&counts, opts.blake3_lanes);
                (f, if f.needs_compressor { "yes" } else { "no" })
            });

            let ldt = if stark_struct.low_degree_test.kind().is_fri() { "FRI" } else { "STIR" };
            summary_lines.push(format!(
                "{} | {} | {} | {} | verifierHashes: {}{}",
                airgroup_name,
                air_name,
                ldt,
                pil_result.summary,
                verifier_hashes,
                fit.map(|(_, n)| format!(" | needsCompressor: {n}")).unwrap_or_default(),
            ));

            stats_lines.push(format!("Airgroup: {} Air: {} ({ldt})", airgroup_name, air_name));
            stats_lines.push(format!("Summary: {}", pil_result.summary));
            stats_lines.push(format!("Verifier hashes ({}):", opts.hash));
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
            if let Some((f, n)) = fit {
                stats_lines.push(format!(
                    "    needsCompressor: {n:>3} ({} of {} hashing blocks at LANES={}; hashing only, a \
                     plonk-dominated band can still size the air)",
                    f.blocks, f.capacity, opts.blake3_lanes,
                ));
            }

            let (base_field, extended_field) = &pil_result.im_pols_info;
            push_im_pols_stats(&mut stats_lines, base_field, extended_field);
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
    let family = pctx.global_info.hash;

    // (name, group, geometry)
    let mut circuits: Vec<(String, String, crate::verifier_hashes::VerifierGeometry, String)> = Vec::new();

    let sctx = SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Basic, false, false)
        .map_err(|e| anyhow::anyhow!("Failed to load basic setups: {e:?}"))?;
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, air) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((air.name.clone(), "basic".to_string(), geom, security_line(&setup.stark_info)));
        }
    }

    if opts.aggregation {
        let sctx_compressor = SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Compressor, false, false)
            .map_err(|e| anyhow::anyhow!("Failed to load compressor setups: {e:?}"))?;
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, air) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                    let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
                    circuits.push((
                        format!("{}_compressor", air.name),
                        "compression".to_string(),
                        geom,
                        security_line(&setup.stark_info),
                    ));
                }
            }
        }

        // recursive1 shares recursive2's circuit, so one entry covers both.
        let sctx_recursive2 = SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Recursive2, false, false)
            .map_err(|e| anyhow::anyhow!("Failed to load recursive2 setups: {e:?}"))?;
        for airgroup_id in 0..pctx.global_info.air_groups.len() {
            let setup = sctx_recursive2.get_setup(airgroup_id, 0).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((
                format!("Recursive1/2 (airgroup {airgroup_id})"),
                "aggregation".to_string(),
                geom,
                security_line(&setup.stark_info),
            ));
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
                None,
                &std::collections::HashMap::new(),
                false,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((name.to_string(), "final".to_string(), geom, security_line(&setup.stark_info)));
        }
    }

    let output_path = opts.output_path.clone().unwrap_or_else(|| "tmp/stats.txt".to_string());
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut stats_lines: Vec<String> = Vec::new();
    let mut summary_lines: Vec<String> = Vec::new();
    for (name, group, geom, security) in &circuits {
        let counts = crate::verifier_hashes::verifier_hashes(geom, family);
        let ldt = if geom.stir_rounds.is_empty() { "FRI" } else { "STIR" };
        summary_lines.push(format!(
            "{} | {} | {} | nBitsExt: {} | verifierHashes: {} | {}",
            group,
            name,
            ldt,
            geom.n_bits_ext,
            counts.total(),
            security
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

/// The exact security report of a built circuit: `security: BCHKS25 X bits`.
fn security_line(si: &proofman_common::StarkInfo) -> String {
    if si.fri_batch_size == 0 || si.security.proximity_gap <= 0.0 {
        return "security: n/a (key lacks friBatchSize or proximityGap, regenerate)".to_string();
    }
    let report = crate::types::security::check::report_built(
        &si.stark_struct,
        si.fri_batch_size,
        &si.security.regime,
        si.security.proximity_gap,
    )
    .map_err(|e| e.to_string());
    match report {
        Ok(r) => format!("{} [{}, weakest {}]", r.summary(), r.config, r.weakest_phase),
        Err(e) => format!("security: ERROR {e}"),
    }
}
