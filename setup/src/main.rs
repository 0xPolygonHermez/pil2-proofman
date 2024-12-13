use std::process::exit;
use crate::cli::parse_cli;
use crate::setup::setup_cmd;

mod cli;
mod setup;

#[tokio::main]
async fn main() {
    let config = parse_cli().await;

    if let Err(e) = setup_cmd(&config, &config.setup.const_tree).await {
        eprintln!("Error: {}", e);
        eprintln!("{:?}", e);
        exit(1);
    }

    println!("Files generated correctly");
    exit(0);
}
