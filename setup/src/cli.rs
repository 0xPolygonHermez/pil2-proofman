#![allow(unused)]

use std::path::PathBuf;
use serde_json::Value;
use clap::{Arg, Command};
use tokio::fs::read_to_string;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Config {
    pub airout: AiroutConfig,
    pub setup: SetupConfig,
    pub build_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AiroutConfig {
    pub airout_filename: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetupConfig {
    pub settings: Value,
    pub gen_aggregation_setup: bool,
    pub opt_im_pols: bool,
    pub const_tree: PathBuf,
    pub bin_file: PathBuf,
    pub stdlib: Option<PathBuf>,
}

pub async fn parse_cli() -> Config {
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
    let const_tree = matches.get_one::<PathBuf>("consttree").expect("Bctree path must be provided").clone();
    let bin_file = matches.get_one::<PathBuf>("binfile").expect("BinFile path must be provided").clone();
    let pilout_path = matches.get_one::<PathBuf>("airout").expect("Airout path must be provided").clone();

    let stark_structs_info = if let Some(starkstructs) = matches.get_one::<PathBuf>("starkstructs") {
        let data = read_to_string(starkstructs).await.expect("Failed to read starkstructs file");
        serde_json::from_str(&data).expect("Failed to parse starkstructs JSON")
    } else {
        serde_json::json!({})
    };

    Config {
        airout: AiroutConfig { airout_filename: pilout_path },
        setup: SetupConfig {
            settings: stark_structs_info,
            gen_aggregation_setup: matches.get_flag("recursive"),
            opt_im_pols: matches.get_flag("impols"),
            const_tree,
            bin_file,
            stdlib: matches.get_one::<PathBuf>("stdlib").cloned(),
        },
        build_dir,
    }
}
