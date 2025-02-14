use std::process::exit;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use proofman_setup::cli::parse_cli;
use proofman_setup::setup::setup_cmd;

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
