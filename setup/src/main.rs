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

#[tokio::main]
async fn main() {
    // a builder for `FmtSubscriber`.
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(Level::TRACE)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    let config = parse_cli().await;

    if let Err(e) = setup_cmd(&config, &config.setup.const_tree).await {
        eprintln!("Error: {}", e);
        eprintln!("{:?}", e);
        exit(1);
    }

    println!("Files generated correctly");
    exit(0);
}
