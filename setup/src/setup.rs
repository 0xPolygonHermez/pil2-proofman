use num_traits::ToPrimitive;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tokio::fs as async_fs;
use tracing::info;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use num_bigint::BigUint;

use crate::{
    airout::AirOut,
    cli::Config,
    f3g::F3g,
    get_pilout_info::get_fixed_pols_pil2,
    pil_info::pil_info,
    utils::{log2, set_airout_info},
    witness_calculator::{generate_fixed_cols, Symbol},
};

pub async fn setup_cmd(config: &Config, build_dir: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("build_dir: {:?}", build_dir.as_ref());
    tracing::info!("Attempting to load airout file from '{}'", config.airout.airout_filename.display());
    let airout = AirOut::from_file(&config.airout.airout_filename)?;
    tracing::info!("Successfully loaded airout file");
    let setup_options = SetupOptions {
        opt_im_pols: config.setup.opt_im_pols,
        const_tree: config.setup.const_tree.clone(),
        bin_file: config.setup.bin_file.clone(),
        stdlib: config.setup.stdlib.clone(),
        settings: config.setup.settings.clone(),
    };

    let mut setup: Vec<Vec<StarkStruct>> = vec![];
    let mut stark_structs = vec![];
    let mut min_final_degree = 5;

    // Determine minimum final degree across all air groups
    tracing::info!("Determining minimum final degree across all air groups");
    for airgroup in &airout.pilout().air_groups {
        for air in &airgroup.airs {
            let name = match { air.name.as_ref().map(|s| s.as_str()) } {
                Some(name) => name,
                None => {
                    info!("Air name not found");
                    continue;
                }
            };
            let settings = config
                .setup
                .settings
                .get(name)
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or(AirSettings { stark_struct: None, final_degree: min_final_degree });

            let air_num_rows: u32 = air.num_rows.unwrap_or(0);

            min_final_degree = if let Some(stark_struct) = &settings.stark_struct {
                stark_struct.steps.last().map_or(min_final_degree, |step| step.n_bits)
            } else {
                min_final_degree.min((log2(air_num_rows) + 1).try_into().unwrap())
            };
        }
    }
    tracing::info!("Minimum final degree: {}", min_final_degree);

    tracing::info!("Generating setup for each air group");
    for (airgroup_id, airgroup) in airout.pilout().air_groups.iter().enumerate() {
        setup.push(vec![]);
        for (air_id, air) in airgroup.airs.iter().enumerate() {
            tracing::info!("Computing setup for air '{}'", air.name.as_ref().unwrap());

            let settings = config
                .setup
                .settings
                .get(air.name.as_ref().unwrap())
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or(AirSettings { stark_struct: None, final_degree: min_final_degree });
            tracing::info!("Settings: {:?}", settings);

            let files_dir = build_dir
                .as_ref()
                .join(&airgroup.name.clone().unwrap())
                .join(air_id.to_string())
                .join("airs")
                .join(air.name.as_ref().unwrap())
                .join("air");
            tracing::info!("Creating directory '{}'", files_dir.display());

            async_fs::create_dir_all(&files_dir).await?;

            tracing::info!("Generating setup for air '{}'", air.name.as_ref().unwrap());
            let air_num_rows: u32 = air.num_rows.unwrap().try_into().unwrap();

            tracing::info!("Generating STARK struct for air '{}'", air.name.as_ref().unwrap());
            let stark_struct = settings
                .stark_struct
                .as_ref()
                .cloned()
                .unwrap_or_else(|| generate_stark_struct(&settings, log2(air_num_rows).try_into().unwrap()));
            stark_structs.push(stark_struct.clone());

            // Generate Fixed Columns
            let field_modulus = BigUint::from(1u32) << 256; // Placeholder modulus
            tracing::info!("Generating fixed columns for air '{}'", air.name.as_ref().unwrap());
            let fixed_pols = generate_fixed_cols(airout.pilout().symbols.clone(), air_num_rows, field_modulus);

            let air_json = serde_json::to_value(air)?;
            let mut fixed_pols_map = fixed_pols.to_hashmap();

            let air_json_map: HashMap<String, Value> = serde_json::from_value(air_json.clone())?;
            tracing::info!("Getting fixed polynomials for air '{}'", air.name.as_ref().unwrap());
            get_fixed_pols_pil2(files_dir.to_str().unwrap(), &air_json_map, &mut fixed_pols_map)?;

            // STARK Setup
            tracing::info!("Running STARK setup for air '{}'", air.name.as_ref().unwrap());
            let stark_setup_result = stark_setup(air_json, &stark_struct, &setup_options).await?;
            tracing::info!("STARK setup completed for air '{}'", air.name.as_ref().unwrap());
            let json_output = serde_json::to_string_pretty(&stark_setup_result)?;

            tracing::info!(
                "Writing STARK setup output to '{}'",
                files_dir.join(format!("{}.starkinfo.json", air.name.as_ref().unwrap())).display()
            );
            async_fs::write(files_dir.join(format!("{}.starkinfo.json", air.name.as_ref().unwrap())), json_output)
                .await?;

            // Compute Constant Tree
            tracing::info!("Computing constant tree for air '{}'", air.name.as_ref().unwrap());
            let const_tree_cmd = format!(
                "{} -c {} -s {} -v {}",
                setup_options.const_tree.display(),
                files_dir.join(format!("{}.const", air.name.as_ref().unwrap())).display(),
                files_dir.join(format!("{}.starkinfo.json", air.name.as_ref().unwrap())).display(),
                files_dir.join(format!("{}.verkey.json", air.name.as_ref().unwrap())).display()
            );

            tracing::info!("Running command: '{}'", const_tree_cmd);
            let output = tokio::process::Command::new("sh").arg("-c").arg(&const_tree_cmd).output().await?;

            tracing::info!("Constant tree output: {}", String::from_utf8_lossy(&output.stdout));

            tracing::info!(
                "Reading constant root from '{}'",
                files_dir.join(format!("{}.verkey.json", air.name.as_ref().unwrap())).display()
            );
            let const_root = serde_json::from_str::<Value>(
                &tokio::fs::read_to_string(files_dir.join(format!("{}.verkey.json", air.name.as_ref().unwrap())))
                    .await?,
            )?;

            setup[airgroup_id][air_id].const_root = Some(const_root.clone());

            // Compute Bin File
            let bin_cmd = format!(
                "{} -s {} -e {} -b {}",
                setup_options.bin_file.display(),
                files_dir.join(format!("{}.starkinfo.json", air.name.as_ref().unwrap())).display(),
                files_dir.join(format!("{}.expressionsinfo.json", air.name.as_ref().unwrap())).display(),
                files_dir.join(format!("{}.bin", air.name.as_ref().unwrap())).display()
            );

            tracing::info!("Running command: '{}'", bin_cmd);
            let output = tokio::process::Command::new("sh").arg("-c").arg(&bin_cmd).output().await?;

            tracing::info!("Bin file output: {}", String::from_utf8_lossy(&output.stdout));

            setup[airgroup_id].push(stark_struct);
        }
    }

    // Generate Final Recursive Setup
    if config.setup.gen_aggregation_setup {
        tracing::info!("Generating final recursive setup");
        let (global_info, global_constraints) = set_airout_info(&airout, &stark_structs);

        tracing::info!("Writing global info and constraints to '{}'", build_dir.as_ref().join("provingKey").display());
        async_fs::write(
            build_dir.as_ref().join("provingKey/pilout.globalInfo.json"),
            serde_json::to_string_pretty(&global_info)?,
        )
        .await?;

        tracing::info!("Writing global constraints to '{}'", build_dir.as_ref().join("provingKey").display());
        async_fs::write(
            build_dir.as_ref().join("provingKey/pilout.globalConstraints.json"),
            serde_json::to_string_pretty(&global_constraints)?,
        )
        .await?;

        // Compute Global Bin File
        let global_bin_cmd = format!(
            "{} -g -e {} -b {}",
            setup_options.bin_file.display(),
            build_dir.as_ref().join("provingKey/pilout.globalConstraints.json").display(),
            build_dir.as_ref().join("provingKey/pilout.globalConstraints.bin").display()
        );

        tracing::info!("Running command: '{}'", global_bin_cmd);
        let output = tokio::process::Command::new("sh").arg("-c").arg(&global_bin_cmd).output().await?;

        info!("Global bin file output: {}", String::from_utf8_lossy(&output.stdout));
    }

    tracing::info!("Setup completed successfully");

    Ok(())
}

pub fn generate_stark_struct(settings: &AirSettings, n_bits: usize) -> StarkStruct {
    // Extract or calculate parameters with defaults
    let hash_commits = true; // Default to true as in JavaScript
    let blowup_factor = 1; // Default value for blowupFactor
    let mut n_queries = (128.0 / blowup_factor as f64).ceil() as usize;

    if let Some(stark_struct) = &settings.stark_struct {
        n_queries = stark_struct.steps.last().map_or(n_queries, |step| step.n_bits.max(n_queries));
    }

    // Default values for foldingFactor and finalDegree
    let folding_factor = 4;
    let final_degree = settings.final_degree.max(5);

    // Determine `VerificationType`
    let verification_hash_type = VerificationType::from_final_degree(final_degree);

    // Initialize nBitsExt
    let n_bits_ext = n_bits + blowup_factor;

    // Initialize the StarkStruct
    let mut stark_struct = StarkStruct {
        n_bits,
        n_bits_ext,
        n_queries,
        verification_hash_type,
        hash_commits,
        const_root: None,
        steps: vec![Step { n_bits: n_bits_ext }],
    };

    // Compute FRI steps
    let mut fri_step_bits = n_bits_ext;
    while fri_step_bits > final_degree {
        fri_step_bits = (fri_step_bits - folding_factor).max(final_degree);
        stark_struct.steps.push(Step { n_bits: fri_step_bits });
    }

    stark_struct
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkSetupResult {
    pub stark_info: Value,
    pub expressions_info: Value,
    pub verifier_info: Value,
    pub stats: Value,
}

/// Helper function for field multiplication
fn multiply_f64(a: f64, b: f64) -> f64 {
    let f3g = F3g::new(); // Instantiate inside the function to avoid captures
    let result = f3g.mul(&BigUint::from(a as u64), &BigUint::from(b as u64));
    result.to_f64().unwrap_or(0.0) // Convert BigUint to f64 safely
}

pub async fn stark_setup(
    air_json: serde_json::Value,
    stark_struct: &StarkStruct,
    setup_options: &SetupOptions,
) -> Result<StarkSetupResult, Box<dyn std::error::Error>> {
    // Check if pil2 mode is enabled
    let pil2 = setup_options.settings.get("pil2").and_then(|v| v.as_bool()).unwrap_or(true);

    // Convert setup_options to a HashMap<String, Value> for compatibility with pil_info
    let options_map: HashMap<String, Value> = serde_json::from_value(serde_json::to_value(setup_options)?)?;

    // Call `pil_info`, using the function pointer `multiply_f64`
    let pil_result = pil_info(
        multiply_f64, // Pass the function pointer
        &air_json,
        pil2,
        &serde_json::to_value(stark_struct)?,
        options_map,
    )
    .await;

    Ok(StarkSetupResult {
        stark_info: pil_result["pilInfo"].clone(),
        expressions_info: pil_result["expressionsInfo"].clone(),
        verifier_info: pil_result["verifierInfo"].clone(),
        stats: pil_result["stats"].clone(),
    })
}

#[derive(Debug, Clone)]
pub struct AirGroup {
    pub airgroup_id: usize,
    pub airs: Vec<Air>,
}

#[derive(Debug, Clone)]
pub struct Air {
    pub name: String,
    pub air_id: usize,
    pub num_rows: usize,
    pub symbols: Vec<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetupOptions {
    pub opt_im_pols: bool,
    pub const_tree: PathBuf,
    pub bin_file: PathBuf,
    pub stdlib: Option<PathBuf>,
    pub settings: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AirSettings {
    pub stark_struct: Option<StarkStruct>,
    pub final_degree: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationType {
    GL,
    BN128,
}

impl VerificationType {
    pub fn from_final_degree(final_degree: usize) -> Self {
        if final_degree > 10 {
            VerificationType::BN128
        } else {
            VerificationType::GL
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkStruct {
    pub n_bits: usize,
    pub n_bits_ext: usize,
    pub n_queries: usize,
    pub verification_hash_type: VerificationType,
    pub hash_commits: bool,
    pub steps: Vec<Step>,
    pub const_root: Option<Value>, // Add this field to store const_root
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub n_bits: usize,
}
