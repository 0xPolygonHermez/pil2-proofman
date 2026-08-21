//! Non-recursive setup command: the main orchestrator for the setup pipeline.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rayon::prelude::*;

use pil2_pilout::pilout::{self as pb};
use prost::Message;
use crate::output::global_info::{build_global_info_json, write_global_constraints, write_global_info_json};
use crate::pil::prepare::PrepareOptions;
use crate::commands::recursive_setup::run_recursive_setup;
use crate::types::security::{
    self,
    pcs::{FriConfig, Fri, Batching},
    regimes::DecodingRegime,
};
use crate::types::stark_struct::{generate_stark_struct, StarkStructsConfig};
use crate::output::stark_info::{build_starkinfo_output, collect_opening_points, compute_log_folding_factors};

/// Setup options parsed from CLI args.
pub struct SetupOptions {
    pub airout_path: String,
    pub build_dir: String,
    pub fixed_dir: Option<String>,
    pub stark_structs_path: Option<String>,
    pub recursive: bool,
    /// Maximum number of recursive1 air pipelines to run in parallel (default 1 = serial).
    /// Each concurrent pipeline runs one circom compile + pil2com at a time.
    /// Rule of thumb: set to available_RAM_GB / per_air_peak_RAM_GB.
    pub recursive_jobs: usize,
    /// Maximum number of AIRs to process in parallel during non-recursive setup (default 1 = serial).
    /// Each slot runs pil_info + file I/O for one AIR.
    /// Rule of thumb: set to available_RAM_GB / per_air_peak_RAM_GB.
    pub setup_jobs: usize,
    /// Optional path to write per-AIR stats (same format as `proofman-setup stats`).
    /// If None, no stats file is written.
    pub stats_output_path: Option<String>,
    pub hash: String,
    /// Generate + compile per-AIR Q-expression CUDA kernels (`.exps.so`) at the
    /// end of setup. No-op (logged) if `nvcc` is not on PATH.
    pub gen_exps: bool,
    /// CUDA arch spec for `--gen-exps`: `auto` | `major` | e.g. `89,120` / `sm_120`.
    pub exps_arch: String,
    /// Skip an AIR whose Q has more than this many ops (stays on the interpreter).
    pub exps_cap: usize,
    /// Fixed ops/chunk for every AIR; `None` => the no-spill autotuner.
    pub exps_chunk: Option<usize>,
    /// pil2-stark source root for the nvcc includes; `None` resolves relative to
    /// the exps-codegen crate.
    pub exps_stark_src: Option<String>,
}

/// True if the CUDA `nvcc` compiler is resolvable on PATH. Used to gate
/// `--gen-exps` so a setup run on a machine without the CUDA toolchain skips
/// expression-kernel codegen cleanly instead of erroring mid-compile.
pub(crate) fn nvcc_present() -> bool {
    which::which("nvcc").is_ok()
}

/// Run the non-recursive setup pipeline.
pub fn run_setup(opts: &SetupOptions) -> Result<()> {
    proofman_starks_lib_c::set_hash_family_c(&opts.hash);
    let pilout_data = fs::read(&opts.airout_path)?;
    let pilout = pb::PilOut::decode(pilout_data.as_slice())?;
    let pilout_name = pilout.name.clone().unwrap_or_else(|| "pilout".to_string());

    let settings_map: StarkStructsConfig = if let Some(ref settings_path) = opts.stark_structs_path {
        let data = fs::read_to_string(settings_path)?;
        StarkStructsConfig::from_json_str(&data)?
    } else {
        StarkStructsConfig::default()
    };

    struct AirWorkItem {
        ag_idx: usize,
        air_idx: usize,
        airgroup_name: String,
        air_name: String,
        num_rows: usize,
    }

    let mut work_items = Vec::new();
    for (ag_idx, airgroup) in pilout.air_groups.iter().enumerate() {
        let airgroup_name = airgroup.name.clone().unwrap_or_else(|| format!("airgroup_{}", ag_idx));
        for (air_idx, air) in airgroup.airs.iter().enumerate() {
            let air_name = air.name.clone().unwrap_or_else(|| format!("air_{}", air_idx));
            let num_rows = air.num_rows.unwrap_or(0) as usize;
            if num_rows == 0 {
                tracing::warn!("Skipping air '{}' with numRows=0", air_name);
                continue;
            }
            work_items.push(AirWorkItem { ag_idx, air_idx, airgroup_name: airgroup_name.clone(), air_name, num_rows });
        }
    }

    tracing::info!("Processing {} AIRs", work_items.len());

    let pilout = Arc::new(pilout);
    let settings_map = Arc::new(settings_map);
    let build_dir = opts.build_dir.clone();
    let fixed_dir = opts.fixed_dir.clone();
    let pilout_name_shared = pilout_name.clone();

    // Always write globalConstraints first — run_recursive_setup reads them from disk.
    // globalInfo.json is written now only for the non-recursive path; for recursive it is
    // written once at the end after hasCompressor flags are known.
    write_global_constraints(&pilout, &pilout_name, &opts.build_dir, &settings_map)?;
    if !opts.recursive {
        write_global_info_json(&pilout, &pilout_name, &opts.build_dir, &settings_map, &opts.hash)?;
    }

    // Thread pool for per-AIR processing.  setup_jobs > 1 enables parallel AIR
    // pipelines; each gets the 64 MB stack that pil_info requires.
    let air_pool =
        rayon::ThreadPoolBuilder::new().num_threads(opts.setup_jobs.max(1)).stack_size(64 * 1024 * 1024).build()?;

    // Each entry: (ag_idx, air_idx, airgroup_name, air_name, summary, im_pols_info)
    type StatsEntry = (usize, usize, String, String, String, (Vec<String>, Vec<String>));

    let results: Vec<Result<StatsEntry>> = air_pool.install(|| {
        work_items
            .par_iter()
            .map(|item| {
                let n_bits = log2_usize(item.num_rows);
                tracing::info!("Computing setup for air '{}'", item.air_name);

                let air_settings = settings_map.resolve(&item.airgroup_name, &item.air_name);

                let stark_struct = generate_stark_struct(&air_settings, n_bits, &opts.hash);

                let files_dir = PathBuf::from(&build_dir)
                    .join("provingKey")
                    .join(&pilout_name_shared)
                    .join(&item.airgroup_name)
                    .join("airs")
                    .join(&item.air_name)
                    .join("air");
                fs::create_dir_all(&files_dir)?;

                let const_path = files_dir.join(format!("{}.const", item.air_name));

                // A path that cannot produce `.const` must drop any file an earlier run
                // left, or the `const_path.exists()` check below builds the const tree
                // and verkey from stale fixed columns and still exits 0. The verkey pair
                // goes too: it is only rewritten when `.const` exists.
                let drop_stale_const = |why: &str| -> Result<()> {
                    let mut dropped = false;
                    for stale in [
                        const_path.clone(),
                        files_dir.join(format!("{}.verkey.json", item.air_name)),
                        files_dir.join(format!("{}.verkey.bin", item.air_name)),
                    ] {
                        if stale.exists() {
                            fs::remove_file(&stale)
                                .with_context(|| format!("removing stale {} ({})", stale.display(), why))?;
                            dropped = true;
                        }
                    }
                    if dropped {
                        tracing::warn!(
                            "Air '{}': dropped stale .const/.verkey in {} — {}",
                            item.air_name,
                            files_dir.display(),
                            why
                        );
                    }
                    Ok(())
                };

                if let Some(ref fd) = fixed_dir {
                    let src = Path::new(fd).join(format!("{}.fixed", item.air_name));
                    if src.exists() {
                        fs::copy(&src, &const_path)?;
                    } else {
                        tracing::warn!("Fixed file not found: {}, skipping copy", src.display());
                        drop_stale_const("its .fixed source is missing")?;
                    }
                } else {
                    // No --fixed-dir: try to generate .const from inline fixed_cols in the pilout.
                    // The pil2-compiler embeds selector/constant polynomial values directly in the
                    // pilout protobuf when they are statically known.  If every fixed_col entry has
                    // non-empty inline values we can write the .const file without an external file.
                    // If any entry has empty values (large external polynomials), we warn and skip —
                    // the downstream bctree / prover will fail to find the file.
                    let air = &pilout.air_groups[item.ag_idx].airs[item.air_idx];
                    if air.fixed_cols.is_empty() {
                        tracing::debug!("Air '{}': no fixed columns — skipping .const generation", item.air_name);
                        drop_stale_const("the air has no fixed columns")?;
                    } else {
                        let has_external = air.fixed_cols.iter().any(|fc| fc.values.is_empty());
                        if has_external {
                            tracing::warn!(
                                "Air '{}': .const file cannot be generated from pilout — \
                             some fixed columns have no inline values. \
                             Provide --fixed-dir (-u) to supply the pre-computed .const file.",
                                item.air_name
                            );
                            drop_stale_const("no --fixed-dir and its values are not inline")?;
                        } else {
                            // All fixed columns are inline — write .const directly.
                            crate::io::fixed_cols::write_const_file(
                                const_path.to_str().unwrap_or(""),
                                air,
                                &[], // no external plonk polynomial values needed
                            )?;
                            tracing::info!(
                                "Generated .const from pilout inline fixed_cols for air '{}' \
                             ({} columns, {} rows)",
                                item.air_name,
                                air.fixed_cols.len(),
                                air.num_rows.unwrap_or(0)
                            );
                        }
                    }
                }

                let starkinfo_path = files_dir.join(format!("{}.starkinfo.json", item.air_name));

                let prepare_opts = PrepareOptions { debug: false, im_pols_stages: false };
                let pil_result =
                    crate::pil::info::pil_info(&pilout, item.ag_idx, item.air_idx, &stark_struct, &prepare_opts);
                let setup_result = &pil_result.setup;
                let pil_code = &pil_result.pil_code;

                let ev_map_len = pil_code.ev_map.len();
                let log_folding_factors = compute_log_folding_factors(&stark_struct);
                let opening_points = collect_opening_points(setup_result);
                let field_size = security::goldilocks_safe_extension_field_size();
                let regime = DecodingRegime::Jbr;
                let fri_config = FriConfig {
                    field_size,
                    trace_length: 1u32 << stark_struct.n_bits,
                    rate: 1.0 / (1u64 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64,
                    batch_size: ev_map_len.max(1) as u64,
                    batching: Batching::Powers,
                    log_folding_factors,
                    max_grinding_bits_query: stark_struct.pow_bits as u64,
                    use_max_grinding_bits_query: true,
                    tree_arity: stark_struct.merkle_tree_arity as u64,
                    hash_size_bits: 256,
                    target_security_bits: 128,
                    regime,
                };
                let fri = Fri::new(fri_config);

                let starkinfo_output = build_starkinfo_output(
                    setup_result,
                    &stark_struct,
                    pil_code,
                    &opening_points,
                    &fri,
                    item.ag_idx,
                    item.air_idx,
                    &item.air_name,
                    pil_result.c_exp_id,
                    pil_result.fri_exp_id,
                    pil_result.q_deg,
                );

                let starkinfo_json = crate::output::json::to_json_string(&starkinfo_output)?;
                fs::write(&starkinfo_path, &starkinfo_json)?;

                fs::write(
                    files_dir.join(format!("{}.expressionsinfo.json", item.air_name)),
                    &crate::output::json::to_json_string(&pil_code.expressions_info)?,
                )?;
                fs::write(
                    files_dir.join(format!("{}.verifierinfo.json", item.air_name)),
                    &crate::output::json::to_json_string(&pil_code.verifier_info)?,
                )?;

                let verkey_json_path = files_dir.join(format!("{}.verkey.json", item.air_name));
                if const_path.exists() {
                    let const_root = crate::proving_key::bctree::compute_const_tree(
                        const_path.to_str().unwrap_or(""),
                        starkinfo_path.to_str().unwrap_or(""),
                        verkey_json_path.to_str().unwrap_or(""),
                    );
                    let verkey_bin: Vec<u8> = const_root.iter().flat_map(|v| v.to_le_bytes()).collect();
                    fs::write(files_dir.join(format!("{}.verkey.bin", item.air_name)), &verkey_bin)?;
                }

                write_bin_files_from_pil_code(
                    &starkinfo_json,
                    &pil_code.expressions_info,
                    &pil_code.verifier_info,
                    &files_dir.join(format!("{}.bin", item.air_name)),
                    &files_dir.join(format!("{}.verifier.bin", item.air_name)),
                )?;

                tracing::info!("Setup for air '{}' complete", item.air_name);
                Ok((
                    item.ag_idx,
                    item.air_idx,
                    item.airgroup_name.clone(),
                    item.air_name.clone(),
                    pil_result.summary.clone(),
                    pil_result.im_pols_info.clone(),
                ))
            })
            .collect()
    });

    let mut stats_entries: Vec<StatsEntry> = Vec::new();
    for result in results {
        stats_entries.push(result?);
    }

    if let Some(ref stats_path) = opts.stats_output_path {
        // Sort by (ag_idx, air_idx) so the file is deterministic regardless of parallel order.
        stats_entries.sort_by_key(|(ag, air, ..)| (*ag, *air));

        if let Some(parent) = std::path::PathBuf::from(stats_path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut stats_lines: Vec<String> = Vec::new();
        for (_, _, airgroup_name, air_name, summary, (base_field, extended_field)) in &stats_entries {
            stats_lines.push(format!("Airgroup: {} Air: {}", airgroup_name, air_name));
            stats_lines.push(format!("Summary: {}", summary));
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

        fs::write(stats_path, stats_lines.join("\n"))?;
        tracing::info!("Stats written to {}", stats_path);
    }

    if opts.recursive {
        tracing::info!("Starting recursive setup...");
        let global_info_base = build_global_info_json(&pilout, &pilout_name, &settings_map, &opts.hash);
        let airs_with_compressor = run_recursive_setup(&pilout, &pilout_name, opts, &settings_map, global_info_base)?;

        // Build final settings map: start from user-supplied settings and overlay any
        // hasCompressor flags auto-detected at runtime via NeedsCompressorError.
        // Then write globalInfo.json exactly once with the complete information.
        let mut final_settings: StarkStructsConfig = (*settings_map).clone();
        for air_name in &airs_with_compressor {
            final_settings.set_has_compressor(air_name);
        }
        write_global_info_json(&pilout, &pilout_name, &opts.build_dir, &final_settings, &opts.hash)?;
        tracing::info!("Wrote globalInfo.json with hasCompressor flags");
    }

    if opts.gen_exps {
        let gen_opts = crate::commands::gen_exps::GenExpsOptions {
            proving_key: std::path::PathBuf::from(&opts.build_dir).join("provingKey"),
            arch: opts.exps_arch.clone(),
            cap: opts.exps_cap,
            chunk: opts.exps_chunk,
            stark_src: opts.exps_stark_src.clone().map(std::path::PathBuf::from),
        };
        // Non-fatal: setup itself succeeded and the provingKey is valid; the
        // prover falls back to the interpreter for any AIR without a .so.
        if let Err(e) = crate::commands::gen_exps::run_gen_exps(&gen_opts) {
            tracing::error!("Expression kernel codegen failed (continuing): {:#}", e);
        }
    }

    tracing::info!("Setup complete");
    Ok(())
}

/// Write binary files using in-memory generate_pil_code structs (no disk round-trip).
fn write_bin_files_from_pil_code(
    starkinfo_json: &str,
    expressions_info: &crate::pil::gen_code::ExpressionsInfo,
    verifier_info: &crate::pil::gen_code::VerifierInfo,
    bin_output: &Path,
    verifier_bin_output: &Path,
) -> Result<()> {
    use crate::types::stark_info::StarkInfo;

    let si_json: serde_json::Value = serde_json::from_str(starkinfo_json)?;
    let stark_info = StarkInfo::from_json(&si_json)?;

    let ei = crate::types::stark_info::ExpressionsInfo::from(expressions_info);
    crate::io::bin_file::write_expressions_bin_file(bin_output.to_str().unwrap_or(""), &stark_info, &ei)?;

    let vi = crate::types::stark_info::VerifierInfo::from(verifier_info);
    crate::io::bin_file::write_verifier_expressions_bin_file(
        verifier_bin_output.to_str().unwrap_or(""),
        &stark_info,
        &vi,
    )?;

    Ok(())
}

/// Compute floor(log2(n)) for a nonzero usize.
fn log2_usize(n: usize) -> usize {
    assert!(n > 0, "log2_usize: n must be positive");
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use pil2_pilout::pilout as pb;
    use prost::Message;

    #[test]
    fn setup_options_has_gen_exps_fields() {
        // Compile-level guard: the new gen-exps fields exist with the expected types.
        let o = SetupOptions {
            airout_path: String::new(),
            build_dir: String::new(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: false,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: "Poseidon2".to_string(),
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 60000,
            exps_chunk: None,
            exps_stark_src: None,
        };
        assert!(!o.gen_exps);
        assert_eq!(o.exps_arch, "auto");
        assert_eq!(o.exps_cap, 60000);
        assert!(o.exps_chunk.is_none());
    }

    #[test]
    fn nvcc_present_returns_bool_without_panicking() {
        // Can't assert true/false (host-dependent), but it must not panic and
        // must agree with whether `nvcc` is actually resolvable on PATH.
        let got = nvcc_present();
        let actual = which::which("nvcc").is_ok();
        assert_eq!(got, actual);
    }

    #[test]
    fn test_run_setup_writes_global_files_before_airs() {
        let pilout_proto = pb::PilOut {
            name: Some("globaltest".to_string()),
            air_groups: vec![pb::AirGroup {
                name: Some("TestGroup".to_string()),
                airs: vec![pb::Air { name: Some("TestAir".to_string()), num_rows: Some(0), ..Default::default() }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join(format!("pil2_run_setup_global_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        std::fs::create_dir_all(&build_dir).unwrap();
        let pilout_path = tmp.join("test.pilout");
        let mut buf = Vec::new();
        pilout_proto.encode(&mut buf).unwrap();
        std::fs::write(&pilout_path, &buf).unwrap();

        let opts = SetupOptions {
            airout_path: pilout_path.to_str().unwrap().to_string(),
            build_dir: build_dir.to_str().unwrap().to_string(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: false,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: "Poseidon2".to_string(),
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 60000,
            exps_chunk: None,
            exps_stark_src: None,
        };
        let result = run_setup(&opts);
        assert!(result.is_ok(), "run_setup should succeed: {:#}", result.unwrap_err());
        let pk = build_dir.join("provingKey");
        assert!(pk.join("pilout.globalInfo.json").exists());
        assert!(pk.join("pilout.globalConstraints.json").exists());
        assert!(pk.join("pilout.globalConstraints.bin").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_run_setup_err_with_global_files_surviving() {
        let pilout_proto = pb::PilOut {
            name: Some("pilout.globalInfo.json".to_string()),
            air_groups: vec![pb::AirGroup {
                name: Some("G".to_string()),
                airs: vec![pb::Air { name: Some("A".to_string()), num_rows: Some(4), ..Default::default() }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join(format!("pil2_err_global_survive_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        std::fs::create_dir_all(&build_dir).unwrap();
        let pilout_path = tmp.join("collision.pilout");
        let mut buf = Vec::new();
        pilout_proto.encode(&mut buf).unwrap();
        std::fs::write(&pilout_path, &buf).unwrap();

        let opts = SetupOptions {
            airout_path: pilout_path.to_str().unwrap().to_string(),
            build_dir: build_dir.to_str().unwrap().to_string(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: false,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: "Poseidon2".to_string(),
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 60000,
            exps_chunk: None,
            exps_stark_src: None,
        };
        assert!(run_setup(&opts).is_err());
        let pk = build_dir.join("provingKey");
        assert!(pk.join("pilout.globalInfo.json").exists());
        assert!(pk.join("pilout.globalConstraints.json").exists());
        assert!(pk.join("pilout.globalConstraints.bin").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
