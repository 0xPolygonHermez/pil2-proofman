use serde::{Serialize, Deserialize};
use serde_json::Value;
use tokio::fs as async_fs;
use tracing::info;
use std::path::{Path, PathBuf};
use crate::cli::Config;

// Library function
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

            let fixed_pols = generate_fixed_cols(&air.symbols, air.num_rows);
            get_fixed_pols_pil2(&files_dir, air, &fixed_pols).await?;

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

pub async fn get_fixed_pols_pil2(
    _files_dir: &Path,
    _air: &Air,
    _fixed_pols: &[Symbol],
) -> Result<(), Box<dyn std::error::Error>> {
    // Implement logic
    Ok(())
}

pub fn generate_stark_struct(_settings: &AirSettings, _log2_rows: usize) -> StarkStruct {
    StarkStruct { steps: vec![] }
}

pub async fn stark_setup(
    _air: &Air,
    _stark_struct: &StarkStruct,
    _setup_options: &SetupOptions,
) -> Result<StarkStruct, Box<dyn std::error::Error>> {
    // Implement logic
    Ok(StarkStruct { steps: vec![] })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub air_group_id: usize,
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
pub struct StarkStruct {
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub n_bits: usize,
}
