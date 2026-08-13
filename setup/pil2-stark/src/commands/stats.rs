//! Stats command: compute per-AIR statistics and report constraint/polynomial counts.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use pil2_pilout::pilout as pb;
use prost::Message;

use crate::pil::prepare::PrepareOptions;
use crate::types::stark_struct::{generate_stark_struct, StarkStructsConfig};

/// Options for the stats subcommand.
pub struct StatsOptions {
    /// Path to compiled .pilout file.
    pub airout_path: String,
    /// Output file for detailed per-AIR stats (default: `tmp/stats.txt`).
    pub output_path: Option<String>,
    /// Optional path to starkstructs.json.
    pub stark_structs_path: Option<String>,
    /// Airgroup name filter (empty = all airgroups).
    pub airgroups: Vec<String>,
    /// Air name filter (empty = all airs).
    pub airs: Vec<String>,
    /// Whether to show intermediate polynomial details per stage.
    pub im_pols_stages: bool,
}

/// Run the stats pipeline and write the output file.
pub fn run_stats(opts: &StatsOptions) -> Result<()> {
    let pilout_data = fs::read(&opts.airout_path)?;
    let pilout = pb::PilOut::decode(pilout_data.as_slice())?;

    let settings_map: StarkStructsConfig = if let Some(ref settings_path) = opts.stark_structs_path {
        let data = fs::read_to_string(settings_path)?;
        StarkStructsConfig::from_json_str(&data)?
    } else {
        StarkStructsConfig::default()
    };

    let output_path = opts.output_path.clone().unwrap_or_else(|| "tmp/stats.txt".to_string());

    if let Some(parent) = PathBuf::from(&output_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut stats_lines: Vec<String> = Vec::new();
    let mut summary_lines: Vec<String> = Vec::new();

    for (ag_idx, airgroup) in pilout.air_groups.iter().enumerate() {
        let airgroup_name = airgroup.name.clone().unwrap_or_else(|| format!("airgroup_{}", ag_idx));

        if !opts.airgroups.is_empty() && !opts.airgroups.contains(&airgroup_name) {
            tracing::info!("Skipping airgroup '{}'", airgroup_name);
            continue;
        }

        for (air_idx, air) in airgroup.airs.iter().enumerate() {
            let air_name = air.name.clone().unwrap_or_else(|| format!("air_{}", air_idx));

            if !opts.airs.is_empty() && !opts.airs.contains(&air_name) {
                tracing::info!("Skipping air '{}'", air_name);
                continue;
            }

            let num_rows = air.num_rows.unwrap_or(0) as usize;
            if num_rows == 0 {
                tracing::warn!("Skipping air '{}' with numRows=0", air_name);
                continue;
            }

            let n_bits = log2_usize(num_rows);

            let air_settings = settings_map.resolve(&airgroup_name, &air_name);

            let stark_struct = generate_stark_struct(&air_settings, n_bits);

            let prepare_opts = PrepareOptions { debug: false, im_pols_stages: opts.im_pols_stages };

            tracing::info!("Computing stats for air '{}'", air_name);
            let pil_result = crate::pil::info::pil_info(&pilout, ag_idx, air_idx, &stark_struct, &prepare_opts);

            let summary = format!("{} | {} | {}", airgroup_name, air_name, pil_result.summary);
            summary_lines.push(summary.clone());

            stats_lines.push(format!("Airgroup: {} Air: {}", airgroup_name, air_name));
            stats_lines.push(format!("Summary: {}", pil_result.summary));

            let (base_field, extended_field) = &pil_result.im_pols_info;
            if !base_field.is_empty() {
                stats_lines.push("Intermediate polynomials baseField:".to_string());
                for pol in base_field {
                    stats_lines.push(format!("    {}", pol));
                }
            }
            if !extended_field.is_empty() {
                stats_lines.push("Intermediate polynomials extendedField:".to_string());
                for pol in extended_field {
                    stats_lines.push(format!("    {}", pol));
                }
            }
            stats_lines.push(String::new());
        }
    }

    println!("-------------------------- SUMMARY -------------------------");
    for line in &summary_lines {
        println!("{}", line);
    }
    println!("------------------------------------------------------------");

    fs::write(&output_path, stats_lines.join("\n"))?;
    println!("Stats written to {}", output_path);

    Ok(())
}

/// floor(log2(n)) for a nonzero usize.
fn log2_usize(n: usize) -> usize {
    assert!(n > 0, "log2_usize: n must be positive");
    (usize::BITS - 1 - n.leading_zeros()) as usize
}
