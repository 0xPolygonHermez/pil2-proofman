use std::process::exit;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use crate::cli::parse_cli;
use crate::setup::setup_cmd;

pub mod cli;
pub mod setup;
pub mod f3g;
pub mod fft;
pub mod witness_calculator;
pub mod add_intermediate_pols;
pub mod helpers;
pub mod gen_constraint_pol;
pub mod utils;
pub mod mapping;
pub mod gen_code;
pub mod fri_poly;
pub mod gen_pil_code;
pub mod pil_info;
pub mod prepare_pil;
pub mod get_pilout_info;
pub mod gen_pil1_pols;
pub mod grand_productp_lookup;
pub mod gen_libs_pols;
pub mod grand_product_utils;
pub mod grand_product_permutation_utils;
pub mod airout;

use tracing::*;

#[tokio::main]
async fn main() {
    // Set up the tracing subscriber for logging
    let subscriber = FmtSubscriber::builder().with_max_level(Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    // Parse CLI arguments into configuration
    let config = parse_cli().await;

    // Create build directory if it doesn't exist
    if let Err(e) = async_fs::create_dir_all(&config.build_dir).await {
        error!("Failed to create build directory '{}': {}", config.build_dir.display(), e);
        exit(1);
    }

    info!("Build directory set to '{}'", config.build_dir.display());

    // Ensure required files are provided
    if config.setup.const_tree.to_str().is_none() {
        error!("Bctree path must be provided");
        exit(1);
    }

    if config.setup.bin_file.to_str().is_none() {
        error!("BinFile path must be provided");
        exit(1);
    }

    // Run the setup command
    match setup_cmd(&config, &config.build_dir).await {
        Ok(_) => {
            println!("Files generated correctly");
            exit(0);
        }
        Err(e) => {
            error!("Error: {}", e);
            exit(1);
        }
    }
}
