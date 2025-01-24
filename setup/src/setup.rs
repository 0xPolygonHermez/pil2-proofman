use num_bigint::BigUint;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tokio::fs as async_fs;
use tracing::info;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use crate::cli::Config;

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

    for airgroup in &airout.air_groups {
        for air in &airgroup.airs {
            let settings = config
                .setup
                .settings
                .get(&air.name)
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or(AirSettings { stark_struct: None, final_degree: min_final_degree });

            min_final_degree = if let Some(stark_struct) = &settings.stark_struct {
                stark_struct.steps.last().map_or(min_final_degree, |step| step.n_bits)
            } else {
                min_final_degree.min(log2(air.num_rows) + 1)
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

            let stark_struct = settings
                .stark_struct
                .as_ref()
                .cloned()
                .unwrap_or_else(|| generate_stark_struct(&settings, log2(air.num_rows)));

            stark_structs.push(stark_struct.clone());

            let mut fixed_pols = generate_fixed_cols(&air.symbols, air.num_rows);
            get_fixed_pols_pil2(&files_dir, air, &mut fixed_pols)?;

            let stark_setup_result = stark_setup(air, &stark_struct, &setup_options).await?;
            let json_output = serde_json::to_string_pretty(&stark_setup_result)?;

            async_fs::write(files_dir.join(format!("{}.starkinfo.json", air.name)), json_output).await?;
        }
    }

    Ok(())
}

// Public Helper Functions
pub fn log2(num: usize) -> usize {
    (num as f64).log2().ceil() as usize
}

pub fn get_fixed_pols_pil2(
    files_dir: &Path,
    air: &Air,
    fixed_pols: &mut [Symbol],
) -> Result<(), Box<dyn std::error::Error>> {
    for i in 0..fixed_pols.len() {
        let def = &mut fixed_pols[i]; // Mutable borrow for this specific element
        let id = def.id; // Equivalent to `def.id`
        let deg = def.pol_deg; // Equivalent to `def.polDeg`

        // Ensure `id` is within bounds
        if id >= fixed_pols.len() {
            return Err(format!("Invalid ID: {} exceeds fixed_pols length", id).into());
        }

        let fixed_cols = &air.symbols[i]; // Maps to `pil.fixedCols[i]`
        let const_pol = &mut fixed_pols[id]; // Access the specific polynomial by `id`

        // Ensure `const_pol.values` has enough space for degrees
        if const_pol.values.len() < deg {
            const_pol.values.resize(deg, 0);
        }

        // Process each degree
        for j in 0..deg {
            const_pol.values[j] = fixed_cols.values[j]; // Equivalent to `constPol[j] = buf2bint(fixedCols.values[j])`
        }
    }

    // Save the fixed polynomials to a file
    let output_file = files_dir.join(format!("{}.const", air.name));
    let mut file = fs::File::create(output_file)?;

    for pol in fixed_pols {
        for value in &pol.values {
            file.write_all(&value.to_le_bytes())?; // Serialize as bytes
        }
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

pub fn generate_fixed_cols(symbols: &[Symbol], _num_rows: usize) -> Vec<Symbol> {
    symbols.to_vec()
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
pub struct Symbol {
    pub air_group_id: usize, // Existing field
    pub id: usize,           // Unique identifier for the symbol (equivalent to `def.id`)
    pub pol_deg: usize,      // Polynomial degree (equivalent to `def.polDeg`)
    pub values: Vec<u128>,   // Container for polynomial values (equivalent to `constPol` in JavaScript)
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
