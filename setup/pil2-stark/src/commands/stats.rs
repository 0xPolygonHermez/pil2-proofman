//! Stats command: compute per-AIR statistics and report constraint/polynomial counts.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use pil2_pilout::pilout as pb;
use prost::Message;
use proofman_multilinear::{build_kernels, referenced_leaves, AirIr, Boundary, KernelSpec, MERKLE_ARITY};

use crate::pil::prepare::PrepareOptions;
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
    /// Lanes the blake3 recursion is built at: compressions per block.
    pub blake3_lanes: usize,
    /// With a proving key: also price each circuit under STIR, plain and with columns stacked
    /// by opening set (one stacked polynomial per group of columns sharing their opening points).
    pub stack_stir: bool,
    /// Report the multilinear prover's view (committed columns, zerocheck,
    /// LogUp-GKR bus, estimated proof size and prover memory) instead of the
    /// univariate layout.
    pub multilinear: bool,
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

            let stark_struct = generate_basic_stark_struct(&air_settings, n_bits, &opts.hash);

            let prepare_opts =
                PrepareOptions { debug: false, im_pols_stages: opts.im_pols_stages, inplace_stage_commit: true };

            tracing::info!("Computing stats for air '{}'", air_name);
            let pil_result = crate::pil::info::pil_info(&pilout, ag_idx, air_idx, &stark_struct, &prepare_opts)?;

            if opts.multilinear {
                match crate::output::mlinfo::build_air_ir(
                    &pil_result.setup,
                    n_bits as u32,
                    proofman_multilinear::MlParams::default(),
                ) {
                    Ok(mut ir) => {
                        ir.params = crate::commands::setup::ml_params(
                            &stark_struct,
                            n_bits,
                            ir.total_cols(),
                            proofman_multilinear::MlHashFamily::Poseidon2,
                        );
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

            // Both the native and the in-circuit verifier run the same algorithm, so one count
            // serves both; only the price of a single hash differs.
            let family = opts.hash.as_str();
            let geom = crate::verifier_hashes::geometry_for_family(
                &stark_struct,
                &pil_result.setup,
                pil_result.pil_code.ev_map.len(),
                pil_result.pil_code.fri_batch_size,
            );
            let counts = crate::verifier_hashes::verifier_hashes(&geom, family);
            let verifier_hashes = counts.total().to_string();

            // Only blake3 sizes its recursion by hashing blocks, so only there can hashes answer this.
            let fit = (family == "blake3").then(|| {
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
            if let Some((f, n)) = fit {
                stats_lines.push(format!(
                    "    needsCompressor: {n:>3} ({} of {} hashing blocks at LANES={}; hashing only, a \
                     plonk-dominated band can still size the air)",
                    f.blocks, f.capacity, opts.blake3_lanes,
                ));
            }

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

/// Stats from a built proving key: every circuit's as-built `starkinfo.json` (basic airs and,
/// with `--aggregation`, compressor / recursive2 / vadcop_final / vadcop_final_compressed).
/// Unlike the pilout path nothing is re-solved, so post-solve adjustments (A2's query
/// equalization) are reflected; the hash family comes from the key itself.
fn run_stats_proving_key(opts: &StatsOptions) -> Result<()> {
    use proofman_common::{GlobalInfoAir, MpiCtx, ProofCtx, ProofType, Setup, SetupCtx, VerboseMode};
    use proofman_fields::Goldilocks;
    use std::sync::Arc;

    // Investigation hook: STACK_WHIR="n_vars,log_blowup,grinding,batch" prints the solved WHIR schedule.
    if let Ok(spec) = std::env::var("STACK_WHIR") {
        let v: Vec<usize> = spec.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let q = crate::commands::setup::whir_query_schedule(v[0], v[1], &[1, 4, 4, 4], v[3], v[2]);
        println!("WHIR nv {} lb {} grind {} batch {} -> {:?}", v[0], v[1], v[2], v[3], q);
        return Ok(());
    }

    // Investigation hook: STACK_STIR_SCHED="n_bits,blowup,llv,grinding" prints the solved STIR schedule.
    if let Ok(spec) = std::env::var("STACK_STIR_SCHED") {
        use crate::types::stark_struct::{generate_stark_struct, LowDegreeTestKind, StarkSettings};
        let v: Vec<usize> = spec.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let ss = generate_stark_struct(
            &StarkSettings {
                low_degree_test: Some(LowDegreeTestKind::Stir),
                initial_blowup_factor: Some(v[1]),
                last_level_verification: Some(v[2]),
                grinding_bits: Some(v[3]),
                ..Default::default()
            },
            v[0],
            "blake3",
        );
        let batch: usize = std::env::var("STACK_BATCH").ok().and_then(|b| b.parse().ok()).unwrap_or(1000);
        let (_, _, _, rounds, final_pol) = crate::verifier_hashes::ldt_geometry(&ss, batch);
        println!(
            "SCHED folds {:?} domains {:?} queries {:?} final {}",
            rounds.iter().map(|r| r.log_folding_factor).collect::<Vec<_>>(),
            rounds.iter().map(|r| r.log_domain_size).collect::<Vec<_>>(),
            rounds.iter().map(|r| r.n_queries).collect::<Vec<_>>(),
            final_pol
        );
        return Ok(());
    }

    // Investigation hook: price one recursion starkinfo (STACK_SI) as recursive2, FRI as built vs
    // STIR vs STIR + packing, at STACK_LLV.
    if let Ok(si_path) = std::env::var("STACK_SI") {
        let si: serde_json::Value = serde_json::from_str(&fs::read_to_string(&si_path)?)?;
        let geom = crate::verifier_hashes::geometry_from_starkinfo(&si)?;
        let llv = std::env::var("STACK_LLV").ok().and_then(|v| v.parse().ok()).unwrap_or(geom.last_level_verification);
        let grinding = std::env::var("STACK_GRIND").ok().and_then(|v| v.parse().ok());
        let built = crate::verifier_hashes::verifier_hashes(&geom, "blake3");
        let (stir, stacked, groups) = stack_stir_counts(
            &geom,
            std::path::Path::new(&si_path),
            Some(crate::proving_key::recursive::RecursiveTemplate::Recursive2),
            "blake3",
            llv,
            grinding,
        )?;
        println!(
            "SELF | as built {} ({}) | STIR {} | STIR+pack {} | llv {} | groups/tree {:?}",
            built.total(),
            if geom.stir_rounds.is_empty() { "FRI" } else { "STIR" },
            stir,
            stacked,
            llv,
            groups.iter().map(|g| g.len()).collect::<Vec<_>>()
        );
        return Ok(());
    }

    let pk_path = PathBuf::from(opts.proving_key_path.as_ref().unwrap());
    // aggregation = false here: the recursion setups are loaded one by one below, guarded by
    // presence, so a key whose compressed-final layer is missing still reports the rest.
    let pctx = ProofCtx::<Goldilocks>::create_ctx(pk_path, false, VerboseMode::Info, Arc::new(MpiCtx::new()), false)
        .map_err(|e| anyhow::anyhow!("Failed to load proving key: {e:?}"))?;
    let family = pctx.global_info.hash.clone();

    // (name, group, geometry)
    let mut circuits: Vec<(String, String, crate::verifier_hashes::VerifierGeometry)> = Vec::new();
    // (starkinfo path, recursion template) per circuit, for --stack-stir.
    let mut sources: Vec<(PathBuf, Option<crate::proving_key::recursive::RecursiveTemplate>)> = Vec::new();

    let sctx = SetupCtx::<Goldilocks>::new(&pctx.global_info, &ProofType::Basic, false, false)
        .map_err(|e| anyhow::anyhow!("Failed to load basic setups: {e:?}"))?;
    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, air) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let geom = crate::verifier_hashes::geometry_from_stark_info(&setup.stark_info);
            circuits.push((air.name.clone(), "basic".to_string(), geom));
            sources.push((
                pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic).with_extension("starkinfo.json"),
                None,
            ));
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
                    circuits.push((format!("{}_compressor", air.name), "compression".to_string(), geom));
                    sources.push((
                        pctx.global_info
                            .get_air_setup_path(airgroup_id, air_id, &ProofType::Compressor)
                            .with_extension("starkinfo.json"),
                        Some(crate::proving_key::recursive::RecursiveTemplate::Compressor),
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
            circuits.push((format!("Recursive1/2 (airgroup {airgroup_id})"), "aggregation".to_string(), geom));
            sources.push((
                pctx.global_info.get_air_setup_path(airgroup_id, 0, &ProofType::Recursive2).with_extension("starkinfo.json"),
                Some(crate::proving_key::recursive::RecursiveTemplate::Recursive2),
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
            circuits.push((name.to_string(), "final".to_string(), geom));
            sources.push((path.with_extension("starkinfo.json"), None));
        }
    }

    let output_path = opts.output_path.clone().unwrap_or_else(|| "tmp/stats.txt".to_string());
    if let Some(parent) = PathBuf::from(&output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut stats_lines: Vec<String> = Vec::new();
    let mut summary_lines: Vec<String> = Vec::new();
    if opts.stack_stir {
        println!("STACK | circuit | n | FRI as-built | STIR | STIR+stack | groups/tree | extra bits/tree | lanes@2^19 (x1 / x2 proofs) STIR+stack");
        for ((name, group, geom), (path, template)) in circuits.iter().zip(sources.iter()) {
            if group == "final" {
                continue;
            }
            let fri = crate::verifier_hashes::verifier_hashes(geom, &family).total();
            let grinding: Option<usize> = std::env::var("STACK_GRIND").ok().and_then(|v| v.parse().ok());
            let llvs: Vec<u64> = match std::env::var("STACK_LLV").ok().and_then(|v| v.parse().ok()) {
                Some(l) => vec![l],
                None => vec![geom.last_level_verification],
            };
            let (mut stir, mut stacked, mut groups) = (u64::MAX, u64::MAX, vec![]);
            let (mut llv_stir, mut llv_stack) = (0, 0);
            for &llv in &llvs {
                let (a, b, g) = stack_stir_counts(geom, path, *template, &family, llv, grinding)?;
                if a < stir {
                    (stir, llv_stir) = (a, llv);
                }
                if b < stacked {
                    (stacked, llv_stack, groups) = (b, llv, g);
                }
            }
            let _ = (llv_stir, llv_stack);
            let per_lane = pil2_stark_recurser::plonk2pil::setups::blake3::blake3_max_blocks(1 << 19) as u64;
            println!(
                "STACK | {name} | {} | {fri} | {stir} ({:.2}) | {stacked} ({:.2}) | {:?} | {:?} | {} / {}",
                geom.n_bits_ext,
                stir as f64 / fri as f64,
                stacked as f64 / fri as f64,
                groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
                groups
                    .iter()
                    .map(|g| g.iter().map(|&m| m.next_power_of_two().trailing_zeros()).max().unwrap_or(0))
                    .collect::<Vec<_>>(),
                stacked.div_ceil(per_lane),
                (2 * stacked).div_ceil(per_lane),
            );
        }
    }

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

/// (STIR, STIR + stacking by opening set) verifier hashes of an as-built circuit, re-solving the
/// low-degree test the way the setup would: basic airs through `generate_basic_stark_struct`,
/// recursion circuits through their template's settings. Returns the groups per tree too.
fn stack_stir_counts(
    geom: &crate::verifier_hashes::VerifierGeometry,
    starkinfo: &std::path::Path,
    template: Option<crate::proving_key::recursive::RecursiveTemplate>,
    family: &str,
    llv: u64,
    grinding: Option<usize>,
) -> Result<(u64, u64, Vec<Vec<u64>>)> {
    let mut geom = geom.clone();
    geom.last_level_verification = llv;
    let geom = &geom;
    use crate::types::stark_struct::{generate_stark_struct, LowDegreeTestKind, StarkSettings};
    use std::collections::{BTreeMap, BTreeSet};

    let si: serde_json::Value = serde_json::from_str(&fs::read_to_string(starkinfo)?)?;
    let n_bits = si["starkStruct"]["nBits"].as_u64().unwrap() as usize;
    let blowup = si["starkStruct"]["nBitsExt"].as_u64().unwrap() as usize - n_bits;
    let n_stages = geom.stage_widths.len();
    let dim = |p: &serde_json::Value| p["dim"].as_u64().unwrap_or(1);

    // Columns (tree, column id) -> (base width, opening set).
    let mut cols: BTreeMap<(usize, u64), (u64, BTreeSet<u64>)> = BTreeMap::new();
    for e in si["evMap"].as_array().unwrap() {
        let id = e["id"].as_u64().unwrap();
        let pos = e["openingPos"].as_u64().unwrap();
        let (tree, width) = match e["type"].as_str().unwrap() {
            "cm" => {
                let p = &si["cmPolsMap"][id as usize];
                (p["stage"].as_u64().unwrap() as usize - 1, dim(p))
            }
            "const" => (n_stages, dim(&si["constPolsMap"][id as usize])),
            "custom" => {
                let c = e["commitId"].as_u64().unwrap() as usize;
                (n_stages + 1 + c, dim(&si["customCommitsMap"][c][id as usize]))
            }
            t => anyhow::bail!("unknown evMap type {t}"),
        };
        cols.entry((tree, id)).or_insert((width, BTreeSet::new())).1.insert(pos);
    }
    let n_trees = n_stages + 1 + geom.custom_commit_widths.len();
    let mut by_set: Vec<BTreeMap<BTreeSet<u64>, u64>> = vec![BTreeMap::new(); n_trees];
    let mut opened = vec![0u64; n_trees];
    for ((tree, _), (w, set)) in &cols {
        *by_set[*tree].entry(set.clone()).or_insert(0) += w;
        opened[*tree] += w;
    }
    let widths: Vec<u64> = geom
        .stage_widths
        .iter()
        .copied()
        .chain(std::iter::once(geom.n_constants))
        .chain(geom.custom_commit_widths.iter().copied())
        .collect();
    let mut groups: Vec<Vec<u64>> = by_set.iter().map(|m| m.values().copied().collect()).collect();
    for t in 0..n_trees {
        // Committed but never opened columns still have to live somewhere: one group of their own.
        if widths[t] > opened[t] {
            groups[t].push(widths[t] - opened[t]);
        }
    }
    let batch: usize = by_set.iter().flat_map(|m| m.keys().map(|k| k.len())).sum();
    let lc = groups.iter().flatten().map(|&m| m.next_power_of_two().trailing_zeros() as usize).max().unwrap_or(0);

    let stark_struct = |bits: usize| match template {
        None => generate_basic_stark_struct(
            &StarkSettings {
                low_degree_test: Some(LowDegreeTestKind::Stir),
                initial_blowup_factor: Some(blowup),
                grinding_bits: grinding,
                ..Default::default()
            },
            bits,
            family,
        ),
        Some(t) => generate_stark_struct(
            &crate::proving_key::recursive::recursive_stark_settings(
                t,
                family,
                &StarkSettings {
                    low_degree_test: Some(LowDegreeTestKind::Stir),
                    grinding_bits: grinding,
                    initial_folding_factor: std::env::var("STACK_FOLD").ok().and_then(|v| v.parse().ok()),
                    ..Default::default()
                },
            ),
            bits,
            family,
        ),
    };
    let fri_batch = si["friBatchSize"].as_u64().unwrap_or(geom.n_evals) as usize;
    let ldt_geom = |ss: &crate::types::stark_struct::StarkStruct, b: usize| {
        let (n_queries, pow_bits, step_n_bits, stir_rounds, final_pol_size) = crate::verifier_hashes::ldt_geometry(ss, b);
        crate::verifier_hashes::VerifierGeometry {
            n_queries,
            pow_bits,
            step_n_bits,
            stir_rounds,
            final_pol_size,
            ..Default::default()
        }
    };

    let plain_ss = stark_struct(n_bits);
    let mut plain = geom.clone();
    let l = ldt_geom(&plain_ss, fri_batch);
    plain.n_queries = l.n_queries;
    plain.pow_bits = l.pow_bits;
    plain.step_n_bits = l.step_n_bits;
    plain.stir_rounds = l.stir_rounds;
    plain.final_pol_size = l.final_pol_size;

    // Partial stacking: each tree stacks its groups in chunks of 2^b columns (b per tree, at most
    // the global LC that sets the low-degree test's degree). b = 0 everywhere is plain STIR.
    let chunked = |bits: &[usize]| -> Vec<Vec<u64>> {
        groups
            .iter()
            .zip(bits)
            .map(|(g, &b)| {
                let m = 1u64 << b;
                g.iter().flat_map(|&s| std::iter::repeat_n(s.min(m), s.div_ceil(m) as usize)).collect()
            })
            .collect()
    };
    let mut best: Option<(u64, crate::verifier_hashes::VerifierGeometry, Vec<usize>)> = None;
    for big_lc in 0..=lc {
        let ldt = ldt_geom(&stark_struct(n_bits + big_lc), batch);
        // Trees are independent given the low-degree test: pick each tree's b alone.
        let mut bits = vec![0usize; n_trees];
        for t in 0..n_trees {
            let mut best_t = (u64::MAX, 0usize);
            for b in 0..=big_lc {
                let mut trial = bits.clone();
                trial[t] = b;
                let g = crate::verifier_hashes::stacked_by_opening(geom, &chunked(&trial), &ldt);
                let c = crate::verifier_hashes::verifier_hashes(&g, family);
                if c.leaf + c.merkle < best_t.0 {
                    best_t = (c.leaf + c.merkle, b);
                }
            }
            bits[t] = best_t.1;
        }
        let g = crate::verifier_hashes::stacked_by_opening(geom, &chunked(&bits), &ldt);
        let total = crate::verifier_hashes::verifier_hashes(&g, family).total();
        if best.as_ref().is_none_or(|(b, _, _)| total < *b) {
            best = Some((total, g, bits));
        }
    }
    let (_, stacked, best_bits) = best.unwrap();
    let full = crate::verifier_hashes::stacked_by_opening(geom, &groups, &ldt_geom(&stark_struct(n_bits + lc), batch));
    let (cp, cs) = (crate::verifier_hashes::verifier_hashes(&plain, family), crate::verifier_hashes::verifier_hashes(&stacked, family));
    let cf = crate::verifier_hashes::verifier_hashes(&full, family);
    if std::env::var("STACK_DETAIL").is_ok() {
        println!(
            "DETAIL {} | STIR {} (leaf {} merkle {} ldt {}) | full stack {} | best {} (leaf {} merkle {} ldt {}) chunk bits {:?}",
            starkinfo.display(), cp.total(), cp.leaf, cp.merkle, cp.fri, cf.total(),
            cs.total(), cs.leaf, cs.merkle, cs.fri, best_bits
        );
    }
    Ok((cp.total(), cs.total(), groups))
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

    let uni_folds = p.whir_fold_schedule.clone();
    let mut k1 = vec![1usize];
    k1.extend(uni_folds.iter().copied());
    let va = ml_verifier_cost(ir, &uni_folds, false, std::env::var("ML_PHI").is_err());
    let vb = ml_verifier_cost(ir, &k1, false, false);
    let fmt_v = |v: &MlVerifierCost| {
        format!(
            "{} hashes (leaf {} merkle {} transcript {}) | query muls {} | queries {:?}",
            v.hashes, v.leaf, v.merkle, v.transcript, v.query_muls, v.queries
        )
    };
    let mut details = vec![
        format!("MLV | folds {:?} lb {} grind {}", uni_folds, p.log_blowup, p.grinding_bits),
        format!("MLV | A as-is {}", fmt_v(&va)),
        format!("MLV | B k1=1 {}", fmt_v(&vb)),
        format!(
            "MLVROW|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            n_bits, total_cols, evals, n_kernels, va.hashes, va.query_muls, vb.hashes, vb.query_muls, 0, 0,
            0, vb.ir_muls, vb.once_muls
        ),
    ];
    details.extend(vec![
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
    ]);

    MlAirStats { summary, details, proof_bytes, mem_bytes, trace_bytes }
}

/// One PCS variant of the multilinear verifier, priced as a recursion circuit would pay it.
struct MlVerifierCost {
    hashes: u64,
    leaf: u64,
    merkle: u64,
    transcript: u64,
    /// Base-field multiplications executed once per query, summed over all queries.
    query_muls: u64,
    /// One-off verifier work (base muls), excluding the constraint evaluation both systems pay.
    once_muls: u64,
    /// The constraint/IR evaluation at the point (paid by the univariate verifier too).
    ir_muls: u64,
    queries: Vec<usize>,
}

/// Recursion-relevant cost of the multilinear verifier (blake3, arity-2 trees, llv 4, as ZisK's
/// basic key). Variants: `stacked` packs each matrix into one polynomial so a query opens
/// `2^k₁` values regardless of the column count; otherwise a query opens the `2^k₁` fiber of
/// every column (as `Whir::verify` does today).
fn ml_verifier_cost(ir: &AirIr, fold_bits: &[usize], stacked: bool, dense_claims: bool) -> MlVerifierCost {
    use crate::verifier_hashes::{leaf_hashes, merkle_path_permutations, root_reduction_hashes, TranscriptSim};
    const FAM: &str = "blake3";
    const ARITY: u64 = 2;
    let llv_env: u64 = std::env::var("ML_LLV").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    #[allow(non_snake_case)]
    let LLV = llv_env;
    let p = &ir.params;
    let n = ir.n_bits as usize;
    let lb: usize = std::env::var("ML_BLOWUP").ok().and_then(|v| v.parse().ok()).unwrap_or(p.log_blowup);
    // ML_DROP_STAGE2: stage >= 2 columns go to GKR (e.g. connections), so they are not committed.
    let drop_s2 = std::env::var("ML_DROP_STAGE2").is_ok();
    // ML_PHI: the δ-combined Φ is committed (as FRI/STIR commit the DEEP poly): stage trees are
    // opened at one row, and Φ's tree serves the first fold.
    let phi = std::env::var("ML_PHI").is_ok() && !stacked;
    let mut widths: Vec<u64> = ir
        .cols_per_stage
        .iter()
        .enumerate()
        .filter(|&(s, &c)| c > 0 && (s == 0 || !drop_s2))
        .map(|(_, &c)| c as u64)
        .collect();
    if ir.n_const_cols > 0 {
        widths.push(ir.n_const_cols as u64);
    }
    widths.extend(ir.custom_commits.iter().map(|c| c.n_cols as u64));
    let total_cols: u64 = widths.iter().sum();
    // Stacking: the opened polynomial grows by log2(widest matrix) variables.
    let lc = if stacked { widths.iter().map(|w| w.next_power_of_two().trailing_zeros() as usize).max().unwrap_or(0) } else { 0 };
    let nv = n + lc;
    let fold_bits: Vec<usize> = {
        let mut f = fold_bits.to_vec();
        // Stacking's extra variables get folds of their own: same final-poly length as unstacked.
        let mut extra = lc;
        while extra > 0 {
            let k = extra.min(4);
            f.push(k);
            extra -= k;
        }
        while f.iter().sum::<usize>() > nv {
            let last = f.len() - 1;
            if f[last] > 1 { f[last] -= 1 } else { f.pop(); }
        }
        f
    };
    let batch = if stacked { widths.len() } else { total_cols as usize };
    let grind: usize = std::env::var("ML_GRIND").ok().and_then(|v| v.parse().ok()).unwrap_or(p.grinding_bits);
    let queries = crate::commands::setup::whir_query_schedule(nv, lb, &fold_bits, batch, grind);

    let (mut leaf, mut merkle) = (0u64, 0u64);
    let mut query_muls = 0u64;
    let mut dom = nv + lb;
    let d = ir.max_constraint_degree as u64;
    for (i, (&k, &t)) in fold_bits.iter().zip(queries.iter()).enumerate() {
        let t = t as u64;
        let h = (dom - k) as u64;
        if i == 0 {
            if phi {
                // One row per stage tree (leaves are rows of the 2^(n+lb) domain), then Φ's fiber.
                let hr = (dom) as u64;
                for &w in &widths {
                    leaf += t * leaf_hashes(FAM, ARITY, w);
                    merkle += t * merkle_path_permutations(hr, ARITY, LLV) + root_reduction_hashes(ARITY, LLV, hr);
                }
                leaf += t * leaf_hashes(FAM, ARITY, (1u64 << k) * 3);
                merkle += t * merkle_path_permutations(h, ARITY, LLV) + root_reduction_hashes(ARITY, LLV, h);
                query_muls += t * total_cols * 3;
            } else {
                for &w in &widths {
                    let lw = if stacked { 1u64 << k } else { (1u64 << k) * w };
                    leaf += t * leaf_hashes(FAM, ARITY, lw);
                    merkle += t * merkle_path_permutations(h, ARITY, LLV) + root_reduction_hashes(ARITY, LLV, h);
                }
                // δ-combination of the opened base values (base×ext = 3 muls) unless stacked.
                if !stacked {
                    query_muls += t * (1u64 << k) * total_cols * 3;
                }
            }
        } else {
            leaf += t * leaf_hashes(FAM, ARITY, (1u64 << k) * 3);
            merkle += t * merkle_path_permutations(h, ARITY, LLV) + root_reduction_hashes(ARITY, LLV, h);
        }
        // Fold the 2^k fiber (≈2^k ext muls) + the Pow(z) weight term over the remaining vars.
        let rem = (nv - fold_bits[..=i].iter().sum::<usize>()) as u64;
        query_muls += t * ((1u64 << k) + rem) * 9;
        dom -= 1;
    }

    // Transcript replay (blake3 streams 8 words per compression).
    let mut tr = TranscriptSim::new(FAM, ARITY);
    tr.put(4 + ir.n_publics as u64);
    for _ in &widths {
        tr.put(4);
    }
    let l = p.univariate_skip_bits.min(n) as u64;
    for _ in 0..(n as u64 - l) {
        tr.put(d * 3);
        tr.get_field();
    }
    if let Some(bus) = &ir.bus {
        let layers = n as u64 + bus.terms.len().next_power_of_two().trailing_zeros() as u64;
        for layer in 0..layers {
            for _ in 0..layer {
                tr.put(6);
                tr.get_field();
            }
            tr.put(12);
            tr.get_field();
        }
    }
    let kernels = build_kernels(ir).len() as u64;
    let claims = if dense_claims { total_cols * kernels } else { referenced_leaves(ir).len() as u64 + kernels };
    tr.put(claims * 3);
    tr.get_field();
    tr.get_field();
    for _ in 0..(n + lc) {
        tr.put(6);
        tr.get_field();
    }
    let mut q_hashes = 0;
    for (i, &k) in fold_bits.iter().enumerate() {
        for _ in 0..k {
            tr.put(6);
            tr.get_field();
        }
        tr.put(if i + 1 < fold_bits.len() { 4 + 3 } else { (1u64 << (nv - fold_bits.iter().sum::<usize>())) * 3 });
        tr.get_field();
        let mut qt = TranscriptSim::new(FAM, ARITY);
        qt.put(4);
        qt.get_permutations(queries[i] as u64, (nv + lb - i - k) as u64);
        q_hashes += qt.hashes + 1; // + grinding check
    }
    let transcript = tr.hashes + q_hashes;

    // One-off ext muls (×9 base muls). Kernel MLEs: the per-bit factor products are shared, and
    // the bits above a small offset collapse in closed form → ~4 ext muls per significant bit.
    let bits = |s: i64| 64 - s.unsigned_abs().leading_zeros() as u64 + 1;
    let mut once = 4 * n as u64;
    for k in build_kernels(ir) {
        once += match k {
            KernelSpec::Rot(s) => 2 * 4 * bits(s as i64) + 3 * (1u64 << l),
            KernelSpec::BusRot(s) => 4 * bits(s as i64),
            KernelSpec::Point(_) => n as u64,
        };
    }
    once += claims * 2;
    once += (n as u64 - l) * (d + 1) * 2 + if l > 0 { (d * ((1 << l) - 1) + 1) * ((1 << l) + 1) } else { 0 };
    if let Some(bus) = &ir.bus {
        let layers = n as u64 + bus.terms.len().next_power_of_two().trailing_zeros() as u64;
        once += layers * (layers - 1) / 2 * 4 + layers * 8 + bus.terms.len() as u64 * 2;
    }
    once += (n + lc) as u64 * 6;
    let ir_muls = ir.instrs.len() as u64 * 9 * if ir.bus.is_some() { 2 } else { 1 };
    MlVerifierCost {
        hashes: leaf + merkle + transcript,
        leaf,
        merkle,
        transcript,
        query_muls,
        once_muls: once * 9,
        ir_muls,
        queries,
    }
}
