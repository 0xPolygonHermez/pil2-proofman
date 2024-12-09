use std::path::PathBuf;
use clap::{Arg, Command};
use serde_json::Value;
use tokio::fs::{create_dir_all, read_to_string};
use std::process::exit;

async fn setup_cmd(config: Config, build_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Implement the actual setup logic here
    Ok(())
}

#[derive(Debug)]
struct Config {
    airout: AiroutConfig,
    setup: SetupConfig,
}

#[derive(Debug)]
struct AiroutConfig {
    airout_filename: PathBuf,
}

#[derive(Debug)]
struct SetupConfig {
    settings: Value,
    gen_aggregation_setup: bool,
    opt_im_pols: bool,
    const_tree: PathBuf,
    bin_file: PathBuf,
    stdlib: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    let matches = Command::new("main_gensetup")
        .version("1.0.0") // Replace with actual version
        .about("Setup script for generating configuration files")
        .arg(
            Arg::new("airout")
                .short('a')
                .long("airout")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Path to airout.ptb"),
        )
        .arg(
            Arg::new("builddir")
                .short('b')
                .long("builddir")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Build directory"),
        )
        .arg(
            Arg::new("starkstructs")
                .short('s')
                .long("starkstructs")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Path to starkstructs.json"),
        )
        .arg(
            Arg::new("consttree")
                .short('t')
                .long("consttree")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("Path to consttree"),
        )
        .arg(
            Arg::new("binfile")
                .short('f')
                .long("binfile")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("Path to binfile"),
        )
        .arg(
            Arg::new("recursive")
                .short('r')
                .long("recursive")
                .action(clap::ArgAction::SetTrue)
                .help("Recursive option"),
        )
        .arg(
            Arg::new("impols")
                .short('m')
                .long("impols")
                .action(clap::ArgAction::SetTrue)
                .help("Optimize ImPols option"),
        )
        .arg(
            Arg::new("stdlib")
                .short('l')
                .long("stdlib")
                .value_parser(clap::value_parser!(PathBuf))
                .help("Path to stdlib"),
        )
        .get_matches();

    let build_dir = matches.get_one::<PathBuf>("builddir").cloned().unwrap_or_else(|| PathBuf::from("tmp"));
    create_dir_all(&build_dir).await.expect("Failed to create build directory");

    let const_tree = matches.get_one::<PathBuf>("consttree").expect("Bctree path must be provided").clone();
    let bin_file = matches.get_one::<PathBuf>("binfile").expect("BinFile path must be provided").clone();
    let pilout_path = matches.get_one::<PathBuf>("airout").expect("Airout path must be provided").clone();

    let stark_structs_info = if let Some(starkstructs) = matches.get_one::<PathBuf>("starkstructs") {
        let data = read_to_string(starkstructs).await.expect("Failed to read starkstructs file");
        serde_json::from_str(&data).expect("Failed to parse starkstructs JSON")
    } else {
        serde_json::json!({})
    };

    let config = Config {
        airout: AiroutConfig { airout_filename: pilout_path },
        setup: SetupConfig {
            settings: stark_structs_info,
            gen_aggregation_setup: matches.get_flag("recursive"),
            opt_im_pols: matches.get_flag("impols"),
            const_tree,
            bin_file,
            stdlib: matches.get_one::<PathBuf>("stdlib").cloned(),
        },
    };

    if let Err(e) = setup_cmd(config, &build_dir).await {
        eprintln!("Error: {}", e);
        eprintln!("{:?}", e);
        exit(1);
    }

    println!("Files generated correctly");
    exit(0);
}
