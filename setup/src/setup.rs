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
    cli::Config,
    get_pilout_info::get_fixed_pols_pil2,
    utils::{log2, set_airout_info},
    witness_calculator::{generate_fixed_cols, Symbol},
};

pub async fn setup_cmd(config: &Config, build_dir: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
    let airout = AirOut::from_file(&config.airout.airout_filename).await?;
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
    for airgroup in &airout.air_groups {
        for air in &airgroup.airs {
            let settings = config
                .setup
                .settings
                .get(&air.name)
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or(AirSettings { stark_struct: None, final_degree: min_final_degree });

            let air_num_rows: u32 = air.num_rows.try_into().unwrap();

            min_final_degree = if let Some(stark_struct) = &settings.stark_struct {
                stark_struct.steps.last().map_or(min_final_degree, |step| step.n_bits)
            } else {
                min_final_degree.min((log2(air_num_rows) + 1).try_into().unwrap())
            };
        }
    }

    for airgroup in &airout.air_groups {
        setup.push(vec![]);
        for air in &airgroup.airs {
            info!("Computing setup for air '{}'", air.name);

            let settings = config
                .setup
                .settings
                .get(&air.name)
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or(AirSettings { stark_struct: None, final_degree: min_final_degree });

            let files_dir = build_dir
                .as_ref()
                .join(airgroup.airgroup_id.to_string())
                .join(air.air_id.to_string())
                .join("airs")
                .join(&air.name)
                .join("air");

            async_fs::create_dir_all(&files_dir).await?;

            let air_num_rows: u32 = air.num_rows.try_into().unwrap();
            let stark_struct = settings
                .stark_struct
                .as_ref()
                .cloned()
                .unwrap_or_else(|| generate_stark_struct(&settings, log2(air_num_rows).try_into().unwrap()));

            stark_structs.push(stark_struct.clone());

            // Generate Fixed Columns
            let field_modulus = BigUint::from(1u32) << 256; // Placeholder modulus
            let mut fixed_pols = generate_fixed_cols(air.symbols.clone(), air.num_rows, field_modulus);

            // Get Fixed Polynomials for PIL2
            let pil = serde_json::from_str::<HashMap<String, Value>>("{}").unwrap(); // Placeholder, replace with actual PIL data.
            let mut const_pols: HashMap<String, Vec<Value>> = HashMap::new();

            get_fixed_pols_pil2(files_dir.to_str().unwrap(), &pil, &mut const_pols)?;

            // STARK Setup
            let stark_setup_result = stark_setup(air, &stark_struct, &setup_options).await?;
            let json_output = serde_json::to_string_pretty(&stark_setup_result)?;

            async_fs::write(files_dir.join(format!("{}.starkinfo.json", air.name)), json_output).await?;

            // Compute Constant Tree
            let const_tree_cmd = format!(
                "{} -c {} -s {} -v {}",
                setup_options.const_tree.display(),
                files_dir.join(format!("{}.const", air.name)).display(),
                files_dir.join(format!("{}.starkinfo.json", air.name)).display(),
                files_dir.join(format!("{}.verkey.json", air.name)).display()
            );

            let output = tokio::process::Command::new("sh").arg("-c").arg(&const_tree_cmd).output().await?;

            info!("Constant tree output: {}", String::from_utf8_lossy(&output.stdout));

            let const_root = serde_json::from_str::<Value>(
                &tokio::fs::read_to_string(files_dir.join(format!("{}.verkey.json", air.name))).await?,
            )?;

            // Compute Bin File
            let bin_cmd = format!(
                "{} -s {} -e {} -b {}",
                setup_options.bin_file.display(),
                files_dir.join(format!("{}.starkinfo.json", air.name)).display(),
                files_dir.join(format!("{}.expressionsinfo.json", air.name)).display(),
                files_dir.join(format!("{}.bin", air.name)).display()
            );

            let output = tokio::process::Command::new("sh").arg("-c").arg(&bin_cmd).output().await?;

            info!("Bin file output: {}", String::from_utf8_lossy(&output.stdout));

            setup[airgroup.airgroup_id].push(stark_struct);
        }
    }

    // Generate Final Recursive Setup
    if config.setup.gen_aggregation_setup {
        let (global_info, global_constraints) = set_airout_info(&airout, &stark_structs);

        async_fs::write(
            build_dir.as_ref().join("provingKey/pilout.globalInfo.json"),
            serde_json::to_string_pretty(&global_info)?,
        )
        .await?;

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

        let output = tokio::process::Command::new("sh").arg("-c").arg(&global_bin_cmd).output().await?;

        info!("Global bin file output: {}", String::from_utf8_lossy(&output.stdout));
    }

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

pub async fn stark_setup(
    _air: &Air,
    _stark_struct: &StarkStruct,
    _setup_options: &SetupOptions,
) -> Result<StarkStruct, Box<dyn std::error::Error>> {
    todo!()
}

impl AirOut {
    pub async fn from_file(filename: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = async_fs::read_to_string(filename).await?;
        let airout: Self = serde_json::from_str(&content)?;
        Ok(airout)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AirOut {
    pub air_groups: Vec<AirGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AirGroup {
    pub airgroup_id: usize,
    pub airs: Vec<Air>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub n_bits: usize,
}
