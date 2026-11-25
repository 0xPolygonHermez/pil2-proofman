//! Recursive proving-key setup: compressor → recursive1 → recursive2 → final → compressed-final.
//!
//! Called from `setup_cmd::run_setup` when `--recursive` is set.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pil2_pilout::pilout as pb;

use crate::commands::common::{parse_root_strings, witness_tracker_from_env};
use crate::commands::setup::SetupOptions;
use crate::output::witness_gen::WitnessTracker;
use crate::proving_key::pipeline::{ReusedSetup, Tools};
use crate::proving_key::recursive::{RecursiveSetupConfig, RecursiveTemplate};
use crate::types::stark_struct::StarkStructsConfig;

/// What the recursive pipeline learned that globalInfo.json has to record.
#[derive(Debug, Default)]
pub(crate) struct RecursiveSetupSummary {
    /// Airs that needed a compressor, whether from settings or auto-detected.
    pub airs_with_compressor: std::collections::HashSet<String>,
    /// (airgroup, air) -> proofs its recursive1 verifies; absent means 1.
    pub r1_batch_sizes: std::collections::HashMap<(usize, usize), usize>,
}

/// Run the recursive setup pipeline after non-recursive AIR setup.
///
/// Per airgroup: each air's compressor (if needed) and recursive1 (air[0] serially, the rest in
/// parallel), then the airgroup's recursive2. Then the final and compressed-final setups.
pub(crate) fn run_recursive_setup(
    pilout: &pb::PilOut,
    pilout_name: &str,
    opts: &SetupOptions,
    settings_map: &StarkStructsConfig,
    global_info: serde_json::Value,
) -> Result<RecursiveSetupSummary> {
    use crate::proving_key::compressed_final;
    use crate::proving_key::final_setup;

    let build_dir = &opts.build_dir;

    // The tree's own low-degree test: FRI unless the starkstructs "recursion" entry says
    // otherwise. Every recursion circuit below is built from it.
    let recursion_settings = settings_map.recursion_settings()?;
    tracing::info!(
        "Recursion tree low-degree test: {:?}{}",
        recursion_settings.low_degree_test.unwrap_or(crate::types::stark_struct::LowDegreeTestKind::Fri),
        if recursion_settings.low_degree_test.is_some() { " (from starkstructs \"recursion\")" } else { " (default)" }
    );

    let tools = Tools::from_env();
    let witness_tracker = witness_tracker_from_env();

    let global_constraints_path = Path::new(build_dir).join("provingKey").join("pilout.globalConstraints.json");
    if !global_constraints_path.exists() {
        anyhow::bail!("globalConstraints.json not found at {:?}, cannot run recursive setup", global_constraints_path);
    }
    let global_constraints: serde_json::Value = serde_json::from_str(&fs::read_to_string(&global_constraints_path)?)?;
    let hash = opts.hash;
    let run = Run {
        // What every recursion circuit of the run shares.
        base: RecursiveSetupConfig {
            build_dir,
            template: RecursiveTemplate::Recursive1,
            airgroup_name: "",
            airgroup_id: 0,
            air_id: 0,
            air_name: "",
            global_info: &global_info,
            const_root: &[String::new(), String::new(), String::new(), String::new()],
            verification_keys: &[],
            stark_info: &serde_json::Value::Null,
            verifier_info: &serde_json::Value::Null,
            has_compressor: false,
            hash,
            recursion_settings: &recursion_settings,
            agg_arity: opts.agg_arity,
            recursive_n_bits: opts.recursive_n_bits,
            reuse: None,
            tools: &tools,
        },
        witness_tracker: &witness_tracker,
        stark_structs_path: opts.stark_structs_path.as_deref(),
        persist_mutex: std::sync::Mutex::new(()),
    };

    // 64 MB per worker, as every pool running this work: the pil_info and expression recursion
    // follows the air's size, and rayon's default stack overflows on it.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.recursive_jobs.max(1))
        .stack_size(64 * 1024 * 1024)
        .build()
        .context("Failed to build recursive-jobs rayon pool")?;

    let mut summary = RecursiveSetupSummary::default();
    for (ag_idx, airgroup) in pilout.air_groups.iter().enumerate() {
        let airgroup_name = airgroup.name.clone().unwrap_or_else(|| format!("airgroup_{}", ag_idx));
        let items = air_items(build_dir, pilout_name, &airgroup_name, airgroup, settings_map)?;
        let Some((first, rest)) = items.split_first() else { continue };

        // Logged, not just propagated: tracing goes to stdout and the error to stderr, so a run
        // redirected with `> log` would show no line at all for the air that failed.
        let one = |item: &AirItem, existing: Option<&ReusedSetup>| {
            run_one_air(&run, &airgroup_name, ag_idx, item, existing)
                .inspect_err(|e| tracing::error!("Recursive setup FAILED for air '{}': {e:#}", item.air_name))
        };
        // air[0] serially: its recursive1 setup is the one the rest of the airgroup reuse.
        let first_out = one(first, None)?;
        let r1_setup = first_out.setup.clone().expect("a fresh run returns its setup");
        let rest_out: Vec<AirOutcome> = pool.install(|| {
            use rayon::prelude::*;
            rest.par_iter().map(|item| one(item, Some(&r1_setup))).collect::<Result<_>>()
        })?;

        let mut vkeys = Vec::with_capacity(items.len());
        for (item, out) in items.iter().zip(std::iter::once(&first_out).chain(&rest_out)) {
            vkeys.push(out.vk.clone());
            if out.has_compressor {
                summary.airs_with_compressor.insert(item.air_name.clone());
            }
            if out.batch > 1 {
                summary.r1_batch_sizes.insert((ag_idx, item.air_idx), out.batch);
            }
        }
        let first_air_name = airgroup.airs.first().and_then(|a| a.name.clone()).unwrap_or_else(|| "air_0".into());
        run_recursive2(&run, &airgroup_name, ag_idx, &first_air_name, &r1_setup, vkeys, first_out.n_bits)?;
    }

    tracing::info!("Running final setup...");
    let final_config = final_setup::FinalSetupConfig {
        build_dir,
        hash,
        global_info: &global_info,
        global_constraints: &global_constraints,
        tools: &tools,
    };
    let final_result = final_setup::gen_final_setup(&final_config, &witness_tracker).context("Final setup failed")?;
    tracing::info!("Final setup complete");

    // Skipping leaves the rest of the key complete: `witness_rebuild` treats the stage as optional.
    if !opts.compressed_final {
        tracing::info!(
            "Skipping the compressed final stage: it is off for the {} family, which measured a 2% \
             smaller proof for a whole extra recursion layer. `proofman-setup setup-compressed-final` \
             adds it to this key later if it is wanted.",
            opts.hash
        );
    } else {
        let fr = &final_result;
        tracing::info!("Running compressed final setup...");
        let compressed_tools = Tools::compressed_final_from_env();
        let compressed_config = compressed_final::CompressedFinalConfig {
            build_dir,
            hash,
            name: pilout_name,
            const_root: &fr.const_root.map(|v| v.to_string()),
            stark_info: &fr.stark_info,
            verifier_info: &fr.verifier_info,
            tools: &compressed_tools,
            low_degree_test: None,
        };
        compressed_final::gen_compressed_final_setup(&compressed_config, &witness_tracker)
            .context("Compressed final setup failed")?;
        tracing::info!("Compressed final setup complete");
    }

    witness_tracker.await_all()?;
    tracing::info!("Recursive setup complete");
    Ok(summary)
}

/// What every air's and airgroup's recursion setup of a run shares.
struct Run<'a> {
    base: RecursiveSetupConfig<'a>,
    witness_tracker: &'a WitnessTracker,
    stark_structs_path: Option<&'a str>,
    /// Serialises starkstructs.json rewrites (the rare auto-compressor retry).
    persist_mutex: std::sync::Mutex<()>,
}

/// One air's basic artefacts, which its compressor or recursive1 verifies.
struct AirItem {
    air_idx: usize,
    air_name: String,
    stark_info: serde_json::Value,
    verifier_info: serde_json::Value,
    const_root_strings: [String; 4],
    has_compressor: bool,
}

/// One air's recursive1 (after its compressor, if any).
struct AirOutcome {
    vk: Vec<String>,
    /// This run's setup when it reused none (the airgroup's first air).
    setup: Option<ReusedSetup>,
    /// From the settings, or turned on by a `NeedsCompressorError`.
    has_compressor: bool,
    /// recursive1's rows, which recursive2's must match.
    n_bits: usize,
    batch: usize,
}

/// The airgroup's airs with rows, read from their basic setup's files.
fn air_items(
    build_dir: &str,
    pilout_name: &str,
    airgroup_name: &str,
    airgroup: &pb::AirGroup,
    settings_map: &StarkStructsConfig,
) -> Result<Vec<AirItem>> {
    let mut items = Vec::new();
    for (air_idx, air) in airgroup.airs.iter().enumerate() {
        let air_name = air.name.clone().unwrap_or_else(|| format!("air_{}", air_idx));
        if air.num_rows.unwrap_or(0) == 0 {
            continue;
        }
        let files_dir = PathBuf::from(build_dir)
            .join("provingKey")
            .join(pilout_name)
            .join(airgroup_name)
            .join("airs")
            .join(&air_name)
            .join("air");
        let si_path = files_dir.join(format!("{}.starkinfo.json", air_name));
        let vi_path = files_dir.join(format!("{}.verifierinfo.json", air_name));
        let vk_path = files_dir.join(format!("{}.verkey.json", air_name));
        if !si_path.exists() || !vi_path.exists() {
            anyhow::bail!(
                "Recursive setup failed for air '{}': starkinfo/verifierinfo not found at {:?}",
                air_name,
                files_dir
            );
        }
        let stark_info = serde_json::from_str(&fs::read_to_string(&si_path)?)?;
        let verifier_info = serde_json::from_str(&fs::read_to_string(&vi_path)?)?;
        let const_root_strings =
            parse_verkey_json(&vk_path).with_context(|| format!("Failed to load verkey for air '{}'", air_name))?;
        let has_compressor = settings_map.has_compressor(airgroup_name, &air_name);
        if has_compressor {
            tracing::info!("Air '{}': hasCompressor=true from settings", air_name);
        }
        items.push(AirItem { air_idx, air_name, stark_info, verifier_info, const_root_strings, has_compressor });
    }
    Ok(items)
}

/// The compressor (if needed) then recursive1 of one air; `existing` is the airgroup's first
/// recursive1 setup, reused by the others.
fn run_one_air(
    run: &Run,
    airgroup_name: &str,
    ag_idx: usize,
    item: &AirItem,
    existing: Option<&ReusedSetup>,
) -> Result<AirOutcome> {
    use crate::proving_key::recursive::{gen_recursive_setup, NeedsCompressorError};
    // Every line plonk2pil, circom and pil2com emit from here names no air of its own, and with
    // --recursive-jobs above 1 those interleave.
    let _span = tracing::info_span!("air", name = %item.air_name).entered();
    let mut has_compressor = item.has_compressor;
    // One retry trigger: NeedsCompressorError turns the compressor on. Two passes at most.
    for _ in 0..2 {
        let compressor = if has_compressor {
            tracing::info!("Running compressor setup for air '{}'", item.air_name);
            let cfg = RecursiveSetupConfig {
                template: RecursiveTemplate::Compressor,
                airgroup_name,
                airgroup_id: ag_idx,
                air_id: item.air_idx,
                air_name: &item.air_name,
                const_root: &item.const_root_strings,
                stark_info: &item.stark_info,
                verifier_info: &item.verifier_info,
                ..run.base
            };
            let r = gen_recursive_setup(&cfg, run.witness_tracker)
                .with_context(|| format!("Compressor setup failed for air '{}'", item.air_name))?;
            tracing::info!("Compressor setup complete for air '{}'", item.air_name);
            Some(r)
        } else {
            None
        };
        let (const_root, stark_info, verifier_info) = match &compressor {
            Some(c) => (c.const_root.map(|v| v.to_string()), &c.setup.pil_info.0, &c.setup.pil_info.1),
            None => (item.const_root_strings.clone(), &item.stark_info, &item.verifier_info),
        };
        let cfg = RecursiveSetupConfig {
            template: RecursiveTemplate::Recursive1,
            airgroup_name,
            airgroup_id: ag_idx,
            air_id: item.air_idx,
            air_name: &item.air_name,
            const_root: &const_root,
            stark_info,
            verifier_info,
            has_compressor,
            reuse: existing,
            ..run.base
        };
        tracing::info!("Running recursive1 for air '{}'", item.air_name);
        match gen_recursive_setup(&cfg, run.witness_tracker) {
            Ok(r) => {
                tracing::info!("Recursive1 setup complete for air '{}'", item.air_name);
                return Ok(AirOutcome {
                    vk: r.const_root.iter().map(|v| v.to_string()).collect(),
                    setup: existing.is_none().then_some(r.setup),
                    has_compressor,
                    n_bits: r.n_bits,
                    batch: r.batch_size,
                });
            }
            Err(e) if e.is::<NeedsCompressorError>() => {
                let n_bits = e.downcast_ref::<NeedsCompressorError>().expect("checked").n_bits;
                tracing::warn!(
                    "Air '{}' needs compressor (n_bits={}); retrying with compressor",
                    item.air_name,
                    n_bits
                );
                if let Some(sp) = run.stark_structs_path {
                    let _lock = run.persist_mutex.lock().unwrap();
                    match persist_has_compressor(sp, &item.air_name) {
                        Err(we) => tracing::warn!("Could not update starkstructs.json: {}", we),
                        Ok(()) => {
                            tracing::info!("Updated starkstructs.json: hasCompressor=true for air '{}'", item.air_name)
                        }
                    }
                }
                has_compressor = true;
            }
            Err(e) => return Err(e.context(format!("Recursive1 setup failed for air '{}'", item.air_name))),
        }
    }
    anyhow::bail!("Air '{}' recursive1 still asks for a compressor after 2 attempts", item.air_name)
}

/// The airgroup's recursive2 over its airs' recursive1 keys, from the first air's recursive1 setup
/// (in memory: recursive1's starkinfo is never written).
fn run_recursive2(
    run: &Run,
    airgroup_name: &str,
    ag_idx: usize,
    first_air_name: &str,
    r1: &ReusedSetup,
    vkeys: Vec<Vec<String>>,
    r1_n_bits: usize,
) -> Result<()> {
    // Outside run_one_air's `air` span: without its own these lines have no prefix.
    let _span = tracing::info_span!("airgroup", name = %airgroup_name).entered();
    tracing::info!("Running recursive2 for airgroup '{}'", airgroup_name);
    // Its verifier takes the root as an input: the constant root is never read.
    let cfg = RecursiveSetupConfig {
        template: RecursiveTemplate::Recursive2,
        airgroup_name,
        airgroup_id: ag_idx,
        air_name: first_air_name,
        verification_keys: &[vkeys],
        stark_info: &r1.pil_info.0,
        verifier_info: &r1.pil_info.1,
        ..run.base
    };
    let r2 = crate::proving_key::recursive::gen_recursive_setup(&cfg, run.witness_tracker)
        .with_context(|| format!("Recursive2 setup failed for airgroup '{}'", airgroup_name))?;
    if r2.n_bits != r1_n_bits {
        anyhow::bail!(
            "Recursive2 n_bits ({}) does not match recursive1 n_bits ({}) for airgroup '{}' \
             (first air '{}'). The recursive2 circuit must be sized identically to recursive1; \
             a mismatch usually means the recursive2 starkStruct is inconsistent with recursive1's, \
             or recursive2's circom expands to a different row count than expected.",
            r2.n_bits,
            r1_n_bits,
            airgroup_name,
            first_air_name,
        );
    }
    tracing::info!("Recursive2 setup complete for airgroup '{}' (n_bits={})", airgroup_name, r2.n_bits);
    Ok(())
}

/// Parse a verkey.json file and return exactly 4 u64 limb strings.
pub(crate) fn parse_verkey_json(path: &Path) -> Result<[String; 4]> {
    let content = fs::read_to_string(path).with_context(|| format!("Failed to read verkey file: {:?}", path))?;
    let json = serde_json::from_str(&content).with_context(|| format!("Failed to parse verkey JSON: {:?}", path))?;
    parse_root_strings(&json)
}

/// Update (or insert) `"hasCompressor": true` for `air_name` in a starkstructs JSON file.
///
/// The file is rewritten atomically (write to temp then rename).
pub(crate) fn persist_has_compressor(path: &str, air_name: &str) -> anyhow::Result<()> {
    use std::io::Write;

    let content = fs::read_to_string(path).with_context(|| format!("Cannot read starkstructs file: {}", path))?;
    let mut map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&content).with_context(|| format!("Cannot parse starkstructs JSON: {}", path))?;

    let entry = map.entry(air_name.to_string()).or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if let serde_json::Value::Object(ref mut obj) = entry {
        obj.insert("hasCompressor".to_string(), serde_json::Value::Bool(true));
    }

    let updated = serde_json::to_string_pretty(&serde_json::Value::Object(map))?;

    let tmp_path = format!("{}.tmp", path);
    {
        let mut f =
            std::fs::File::create(&tmp_path).with_context(|| format!("Cannot create temp file: {}", tmp_path))?;
        f.write_all(updated.as_bytes())?;
    }
    fs::rename(&tmp_path, path).with_context(|| format!("Cannot rename {} -> {}", tmp_path, path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::setup::SetupOptions;
    use prost::Message;

    fn options(pilout_path: &std::path::Path, build_dir: &std::path::Path) -> SetupOptions {
        SetupOptions {
            airout_path: pilout_path.to_str().unwrap().to_string(),
            build_dir: build_dir.to_str().unwrap().to_string(),
            fixed_dir: None,
            stark_structs_path: None,
            recursive: true,
            recursive_jobs: 1,
            setup_jobs: 1,
            stats_output_path: None,
            hash: proofman_common::hash_family::HashFamily::Poseidon2,
            agg_arity: 3,
            recursive_n_bits: None,
            compressed_final: true,
            setup_version: None,
            gen_exps: false,
            exps_arch: "auto".to_string(),
            exps_cap: 60000,
            exps_chunk: None,
            exps_stark_src: None,
        }
    }

    #[test]
    fn test_recursive_fails_on_missing_global_constraints() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let pilout_path = base.join("pil/zisk.pilout");
        if !pilout_path.exists() {
            eprintln!("Skipping: zisk.pilout not found");
            return;
        }

        let tmp = std::env::temp_dir().join(format!("pil2_recursive_failfast_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        let pk_dir = build_dir.join("provingKey");
        std::fs::create_dir_all(&pk_dir).unwrap();
        std::fs::write(pk_dir.join("pilout.globalInfo.json"), r#"{"name":"zisk","nPublics":68,"hash":"Poseidon2"}"#)
            .unwrap();

        let pilout = pb::PilOut::decode(std::fs::read(&pilout_path).unwrap().as_slice()).unwrap();
        let opts = options(&pilout_path, &build_dir);
        let result = run_recursive_setup(&pilout, "zisk", &opts, &StarkStructsConfig::default(), serde_json::json!({}));
        assert!(result.is_err());
        assert!(format!("{:#}", result.unwrap_err()).contains("globalConstraints.json not found"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_recursive_fails_on_missing_verkey() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let pilout_path = base.join("pil/zisk.pilout");
        if !pilout_path.exists() {
            eprintln!("Skipping: zisk.pilout not found");
            return;
        }

        let tmp = std::env::temp_dir().join(format!("pil2_verkey_failfast_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        let pk_dir = build_dir.join("provingKey").join("zisk").join("Zisk").join("airs").join("Dma").join("air");
        std::fs::create_dir_all(&pk_dir).unwrap();
        let ginfo = build_dir.join("provingKey");
        std::fs::write(ginfo.join("pilout.globalInfo.json"),
            r#"{"name":"zisk","nPublics":68,"numProofValues":[8],"proofValuesMap":[],"publicsMap":[],"airGroupsInfo":[{"airGroupId":0,"nAirs":35}],"aggTypes":[[]],"hash":"Poseidon2"}"#
        ).unwrap();
        std::fs::write(ginfo.join("pilout.globalConstraints.json"), r#"{"constraints":[],"hints":[]}"#).unwrap();
        std::fs::write(pk_dir.join("Dma.starkinfo.json"), r#"{"nStages":2}"#).unwrap();
        std::fs::write(pk_dir.join("Dma.verifierinfo.json"), r#"{}"#).unwrap();

        let pilout = pb::PilOut::decode(std::fs::read(&pilout_path).unwrap().as_slice()).unwrap();
        let opts = options(&pilout_path, &build_dir);
        let result = run_recursive_setup(&pilout, "zisk", &opts, &StarkStructsConfig::default(), serde_json::json!({}));
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("verkey") || msg.contains("not found"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_recursive2_fails_without_recursive1_artifacts() {
        let pilout = pb::PilOut {
            name: Some("test".to_string()),
            air_groups: vec![pb::AirGroup {
                name: Some("TestGroup".to_string()),
                airs: vec![pb::Air { name: Some("TestAir".to_string()), num_rows: Some(4), ..Default::default() }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join(format!("pil2_r2_prereq_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let build_dir = tmp.join("build");
        let pk_dir = build_dir.join("provingKey");
        std::fs::create_dir_all(&pk_dir).unwrap();
        std::fs::write(pk_dir.join("pilout.globalInfo.json"), r#"{"name":"test","nPublics":0,"hash":"Poseidon2"}"#)
            .unwrap();
        std::fs::write(pk_dir.join("pilout.globalConstraints.json"), r#"{"constraints":[],"hints":[]}"#).unwrap();

        let pilout_path = tmp.join("test.pilout");
        let mut buf = Vec::new();
        pilout.encode(&mut buf).unwrap();
        std::fs::write(&pilout_path, &buf).unwrap();

        let opts = options(&pilout_path, &build_dir);
        let result = run_recursive_setup(&pilout, "test", &opts, &StarkStructsConfig::default(), serde_json::json!({}));
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("starkinfo/verifierinfo not found"), "unexpected error: {}", msg);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parse_verkey_json_rejects_a_missing_file() {
        assert!(parse_verkey_json(std::path::Path::new("/nonexistent/verkey.json")).is_err());
    }
}
