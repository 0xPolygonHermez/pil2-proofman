use colored::Colorize;
use pilout::pilout_proxy::PilOutProxy;
use proofman_common::{StarkStruct, StepStruct};
use serde_json::json;
use std::path::PathBuf;

use anyhow::Result;

use crate::node_bridge::call_node;

const WRAPPER: &str = "src/cmd/setup_cmd_wrapper.js";

/// Configuration for setup generation.
#[derive(Debug, Clone)]
pub struct SetupConfig {
    /// Path to the pilout .ptb file
    pub pilout: PathBuf,
    /// Build output directory
    pub builddir: PathBuf,
    /// Binary files
    pub binfiles: Vec<PathBuf>,
    /// Stark structs JSON file (optional)
    pub starkstructs: Option<PathBuf>,
    /// Standard path (optional)
    pub std_path: Option<PathBuf>,
    /// Fixed polynomials path (optional)
    pub fixed: Option<PathBuf>,
    /// Generate aggregation/recursive setup
    pub recursive: bool,
    /// Optimize intermediate polynomials
    pub impols: bool,
    /// Path to the pil2-proofman-js root directory
    pub js_root: PathBuf,
    /// Node.js max old space size in MB
    pub max_old_space_size: Option<u64>,
}

/// Generate a setup from the given configuration.
pub fn generate_setup(config: &SetupConfig) -> Result<()> {
    // Load and resolve starkStruct settings from JSON file (if provided)
    let starkstructs: serde_json::Value = match &config.starkstructs {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
        None => json!({}),
    };

    let stark_structs = run_part1_rust(config, &starkstructs)?;

    run_part2_js(config, starkstructs, stark_structs)
}

/// Port of JS Part 1: generate per-air starkStruct configs and write them to disk.
fn run_part1_rust(config: &SetupConfig, settings_json: &serde_json::Value) -> Result<Vec<Vec<StarkStruct>>> {
    if !config.binfiles.is_empty() && config.fixed.is_none() {
        return Err(anyhow::anyhow!("binFiles case not yet implemented in Rust Part 1. Use --fixed/-u instead."));
    }

    let proxy = PilOutProxy::new(config.pilout.to_str().unwrap_or_default()).map_err(|e| anyhow::anyhow!("{}", e))?;
    let pilout_name = proxy.pilout.name.as_deref().unwrap_or("unknown");

    let mut result: Vec<Vec<StarkStruct>> = Vec::with_capacity(proxy.pilout.air_groups.len());

    for (airgroup_id, airgroup) in proxy.pilout.air_groups.iter().enumerate() {
        let airgroup_name = airgroup.name.as_deref().unwrap_or("unknown");
        let mut air_row: Vec<StarkStruct> = Vec::with_capacity(airgroup.airs.len());

        for (air_id, air) in airgroup.airs.iter().enumerate() {
            let air_name = air.name.as_deref().unwrap_or("unknown");
            let num_rows = air.num_rows.unwrap_or(0) as u64;

            if num_rows == 0 {
                return Err(anyhow::anyhow!("Air '{}' has num_rows=0", air_name));
            }

            // log2 of num_rows (must be a power of 2)
            let n_bits = num_rows.trailing_zeros() as u64;

            // Resolve settings for this air (air-specific → default → empty)
            let mut settings = resolve_air_settings(settings_json, air_name);

            // Replicate JS Part 1: if powBits is absent/zero, default to 16
            if settings.get("powBits").and_then(|v| v.as_u64()).unwrap_or(0) == 0 {
                settings.insert("powBits".to_string(), json!(16u64));
            }

            // Use preset starkStruct if provided, otherwise generate it
            let stark_struct: StarkStruct = if let Some(preset) = settings.get("starkStruct") {
                serde_json::from_value(preset.clone())?
            } else {
                generate_stark_struct(&settings, n_bits)
            };

            // Create output directory: {builddir}/provingKey/{piloutName}/{airgroupName}/airs/{airName}/air
            let files_dir = config
                .builddir
                .join("provingKey")
                .join(pilout_name)
                .join(airgroup_name)
                .join("airs")
                .join(air_name)
                .join("air");
            std::fs::create_dir_all(&files_dir)
                .map_err(|e| anyhow::anyhow!("Failed to create directory {}: {}", files_dir.display(), e))?;

            // Handle fixed columns
            if let Some(fp) = config.fixed.as_deref() {
                let src = fp.join(format!("{}.fixed", air_name));
                let dst = files_dir.join(format!("{}.const", air_name));
                let base = &config.builddir;
                let src_display = src.strip_prefix(base).unwrap_or(&src);
                let dst_display = dst.strip_prefix(base).unwrap_or(&dst);
                tracing::info!("Copying {} → {}", src_display.display(), dst_display.display());

                std::fs::copy(&src, &dst).map_err(|e| anyhow::anyhow!("Failed to copy {}: {}", src.display(), e))?;
            }

            tracing::info!(
                "Air {airgroup_id}:{air_id} {}: nBits={n_bits}, numRows={} {}",
                air_name.bright_white().bold(),
                num_rows,
                format!("(2^{})", num_rows.ilog2()).dimmed()
            );

            air_row.push(stark_struct);
        }

        result.push(air_row);
    }

    Ok(result)
}

/// Resolve settings for a specific air from the top-level settings JSON.
/// Matches JS: proofManagerConfig.setup.settings[air.name] || .default || {}
fn resolve_air_settings(
    settings_json: &serde_json::Value,
    air_name: &str,
) -> serde_json::Map<String, serde_json::Value> {
    if let serde_json::Value::Object(map) = settings_json {
        if let Some(serde_json::Value::Object(m)) = map.get(air_name) {
            return m.clone();
        }
        if let Some(serde_json::Value::Object(m)) = map.get("default") {
            return m.clone();
        }
    }
    serde_json::Map::new()
}

/// Port of utils.js `generateStarkStruct(settings, nBits)`.
fn generate_stark_struct(settings: &serde_json::Map<String, serde_json::Value>, n_bits: u64) -> StarkStruct {
    const MERKLE_TREE_ARITY: u64 = 4;

    let verification_hash_type =
        settings.get("verificationHashType").and_then(|v| v.as_str()).unwrap_or("GL").to_string();

    let blowup_factor = settings.get("blowupFactor").and_then(|v| v.as_u64()).unwrap_or(1);
    let folding_factor = settings.get("foldingFactor").and_then(|v| v.as_u64()).unwrap_or(3);
    let final_degree = settings.get("finalDegree").and_then(|v| v.as_u64()).unwrap_or(5);

    let merkle_tree_arity: u64;
    let transcript_arity: u64;
    let merkle_tree_custom: bool;
    let last_level_verification: Option<u64>;
    let pow_bits: u64;
    let hash_commits: bool;

    if verification_hash_type == "BN128" {
        merkle_tree_arity = settings.get("merkleTreeArity").and_then(|v| v.as_u64()).unwrap_or(16);
        transcript_arity = merkle_tree_arity;
        merkle_tree_custom = settings.get("merkleTreeCustom").and_then(|v| v.as_bool()).unwrap_or(false);
        last_level_verification = Some(0);
        pow_bits = settings.get("powBits").and_then(|v| v.as_u64()).unwrap_or(0);
        hash_commits = false;
    } else {
        merkle_tree_arity = settings.get("merkleTreeArity").and_then(|v| v.as_u64()).unwrap_or(MERKLE_TREE_ARITY);
        transcript_arity = MERKLE_TREE_ARITY;
        merkle_tree_custom = true;
        last_level_verification = Some(settings.get("lastLevelVerification").and_then(|v| v.as_u64()).unwrap_or(2));
        pow_bits = settings.get("powBits").and_then(|v| v.as_u64()).unwrap_or(20);
        // JS: `settings.hashCommits || true` — due to JS || semantics, even false || true = true,
        // so for GL this is effectively always true.
        hash_commits = true;
    }

    let n_bits_ext = n_bits + blowup_factor;

    // FRI steps: mirrors JS while loop
    let mut steps = vec![StepStruct { n_bits: n_bits_ext }];
    let mut fri_step_bits = n_bits_ext;
    while fri_step_bits > final_degree + 1 {
        fri_step_bits = fri_step_bits.saturating_sub(folding_factor).max(final_degree);
        steps.push(StepStruct { n_bits: fri_step_bits });
    }

    StarkStruct {
        n_bits,
        n_bits_ext,
        verification_hash_type,
        merkle_tree_arity,
        transcript_arity,
        merkle_tree_custom,
        last_level_verification,
        pow_bits,
        hash_commits,
        steps,
        ..Default::default()
    }
}

/// Delegate Part 2 to the Node.js wrapper, calling the three sub-steps in sequence.
fn run_part2_js(
    config: &SetupConfig,
    starkstructs: serde_json::Value,
    stark_structs: Vec<Vec<StarkStruct>>,
) -> Result<()> {
    let node_config = create_node_config(config, starkstructs);
    let builddir = config.builddir.to_string_lossy().to_string();

    // Stark setup generation
    let stark_structs_json = serde_json::to_value(&stark_structs)?;
    let setup = stark_setup(config, &node_config, &builddir, stark_structs_json)?;

    // Circuit generation
    let global_data = generate_circuits(config, &node_config, &builddir, setup)?;

    // Write global data to disk
    write_global_data(config, &builddir, global_data)?;

    Ok(())
}

fn create_node_config(config: &SetupConfig, starkstructs: serde_json::Value) -> serde_json::Value {
    let node_config = json!({
        "airout": {
            "airoutFilename": config.pilout.to_string_lossy()
        },
        "setup": {
            "settings":            &starkstructs,
            "genAggregationSetup": config.recursive,
            "optImPols":           config.impols,
            "binFiles":            config.binfiles.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "stdPath":             config.std_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            "fixedPath":           config.fixed.as_ref().map(|p| p.to_string_lossy().to_string()),
        }
    });
    node_config
}

fn stark_setup(
    config: &SetupConfig,
    node_config: &serde_json::Value,
    builddir_str: &str,
    stark_structs_json: serde_json::Value,
) -> Result<serde_json::Value, anyhow::Error> {
    call_node(
        &config.js_root,
        WRAPPER,
        config.max_old_space_size,
        "stark_setup",
        json!({ "config": &node_config, "buildDir": &builddir_str, "starkStructs": stark_structs_json }),
    )?
    .ok_or_else(|| anyhow::anyhow!("stark_setup returned no value"))
}

fn generate_circuits(
    config: &SetupConfig,
    node_config: &serde_json::Value,
    builddir_str: &str,
    setup: serde_json::Value,
) -> Result<serde_json::Value, anyhow::Error> {
    let global_data = call_node(
        &config.js_root,
        WRAPPER,
        config.max_old_space_size,
        "generate_circuits",
        json!({ "config": &node_config, "buildDir": &builddir_str, "setup": setup }),
    )?
    .ok_or_else(|| anyhow::anyhow!("generate_circuits returned no value"))?;
    Ok(global_data)
}

fn write_global_data(
    config: &SetupConfig,
    builddir_str: &str,
    global_data: serde_json::Value,
) -> Result<(), anyhow::Error> {
    call_node(
        &config.js_root,
        WRAPPER,
        config.max_old_space_size,
        "write_global_data",
        json!({
            "buildDir":          &builddir_str,
            "globalInfo":        &global_data["globalInfo"],
            "globalConstraints": &global_data["globalConstraints"],
        }),
    )?;
    Ok(())
}
