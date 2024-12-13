use std::path::PathBuf;
use serde_json::Value;

#[derive(Debug)]
pub struct Config {
    pub airout: AiroutConfig,
    pub setup: SetupConfig,
}

#[derive(Debug)]
pub struct AiroutConfig {
    pub airout_filename: PathBuf,
}

#[derive(Debug)]
pub struct SetupConfig {
    pub settings: Value,
    pub gen_aggregation_setup: bool,
    pub opt_im_pols: bool,
    pub const_tree: PathBuf,
    pub bin_file: PathBuf,
    pub stdlib: Option<PathBuf>,
}
