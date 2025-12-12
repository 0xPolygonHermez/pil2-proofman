use crate::Setup;
use serde::Serialize;
use tabled::{Tabled, Table};
use std::collections::HashMap;
use crate::ProofType;
use crate::ParamsGPU;
use crate::VerboseMode;
use crate::ProofmanResult;
use crate::ProofmanError;
use crate::MpiCtx;
use crate::ProofCtx;
use crate::SetupCtx;
use crate::SetupsVadcop;
use std::path::PathBuf;
use std::sync::Arc;
use fields::PrimeField64;

#[derive(Tabled)]
pub struct AirTableRow {
    pub name: String,
    pub trace_length: u64,
    pub rho: f64,
    pub air_max_degree: u64,
    pub num_columns: u64,
    pub opening_points: u64,
    pub batch_size: u64,
    pub power_batching: bool,
    pub num_queries: u64,
    pub fri_folding_factors: String,
    pub fri_early_stop_degree: u64,
    pub grinding_query_phase: u64,
    pub prover_size: String,
}

#[derive(Serialize)]
pub struct SoundnessToml {
    pub zkevm: ZkevmConfig,
    pub circuits: Vec<TomlCircuit>,
}

#[derive(Serialize)]
pub struct ZkevmConfig {
    pub name: String,
    pub protocol_family: String,
    pub version: String,
    pub field: String,
    pub hash_size_bits: u32,
}

#[derive(Serialize)]
pub struct TomlCircuit {
    pub name: String,
    pub group: String,
    #[serde(flatten)]
    pub air: AirInfo,
}

#[derive(Serialize, Clone)]
pub struct AirInfo {
    pub trace_length: u64,
    pub rho: f64,
    pub air_max_degree: u64,
    pub num_columns: u64,
    pub opening_points: u64,
    pub batch_size: u64,
    pub power_batching: bool,
    pub num_queries: u64,
    pub fri_folding_factors: Vec<u64>,
    pub fri_early_stop_degree: u64,
    pub grinding_query_phase: u64,
    pub prover_size: String,
}

impl AirTableRow {
    fn from_air_info(name: &str, air: &AirInfo) -> Self {
        AirTableRow {
            name: name.to_string(),
            trace_length: air.trace_length,
            rho: air.rho,
            air_max_degree: air.air_max_degree,
            num_columns: air.num_columns,
            opening_points: air.opening_points,
            batch_size: air.batch_size,
            power_batching: air.power_batching,
            num_queries: air.num_queries,
            fri_folding_factors: format!("{:?}", air.fri_folding_factors),
            fri_early_stop_degree: air.fri_early_stop_degree,
            grinding_query_phase: air.grinding_query_phase,
            prover_size: air.prover_size.clone(),
        }
    }
}

pub fn print_soundness_table(soundness: &SoundnessToml) {
    println!("=== Basics ===");
    let basics_rows: Vec<AirTableRow> = soundness
        .circuits
        .iter()
        .filter(|circuit| circuit.group == "basic")
        .map(|circuit| AirTableRow::from_air_info(&circuit.name, &circuit.air))
        .collect();
    let basics_table = Table::new(basics_rows);
    println!("{}", basics_table);
    
    let compressor_rows: Vec<AirTableRow> = soundness
        .circuits
        .iter()
        .filter(|circuit| circuit.group == "compression")
        .map(|circuit| AirTableRow::from_air_info(&circuit.name, &circuit.air))
        .collect();
    if !compressor_rows.is_empty() {
        println!("=== Compressor ===");
        println!("{}", Table::new(compressor_rows));
    }

    let aggregation_rows: Vec<AirTableRow> = soundness
        .circuits
        .iter()
        .filter(|circuit| circuit.group == "aggregation")
        .map(|circuit| AirTableRow::from_air_info(&circuit.name, &circuit.air))
        .collect();
    if !aggregation_rows.is_empty() {
        println!("=== Aggregation ===");
        println!("{}", Table::new(aggregation_rows));
    }

    let final_rows: Vec<AirTableRow> = soundness
        .circuits
        .iter()
        .filter(|circuit| circuit.group == "final")
        .map(|circuit| AirTableRow::from_air_info(&circuit.name, &circuit.air))
        .collect();
    if !final_rows.is_empty() {
        println!("=== Final Circuit ===");
        println!("{}", Table::new(final_rows));
    }
}

pub fn get_soundness_air_info<F: PrimeField64>(setup: &Setup<F>) -> (String, AirInfo) {
    (
        setup.air_name.clone(),
        AirInfo {
            trace_length: 1 << setup.stark_info.stark_struct.n_bits,
            rho: 1.0 / (1 << (setup.stark_info.stark_struct.n_bits_ext - setup.stark_info.stark_struct.n_bits)) as f64,
            air_max_degree: setup.stark_info.q_deg + 1,
            num_columns: setup.stark_info.n_constants + setup.stark_info.cm_pols_map.as_ref().unwrap().len() as u64,
            opening_points: setup.stark_info.opening_points.len() as u64,
            batch_size: setup.stark_info.ev_map.len() as u64,
            power_batching: true,
            num_queries: setup.stark_info.stark_struct.n_queries,
            fri_folding_factors: setup
                .stark_info
                .stark_struct
                .steps
                .windows(2)
                .map(|pair| 1 << (pair[0].n_bits - pair[1].n_bits))
                .collect(),
            fri_early_stop_degree: 1 << setup.stark_info.stark_struct.steps.last().unwrap().n_bits,
            grinding_query_phase: 0,
            prover_size: crate::format_bytes((setup.prover_buffer_size as f64 + setup.const_tree_size as f64 + setup.const_pols_size as f64) * 8.0),
        },
    )
}

pub fn soundness_info<F: PrimeField64>(
    proving_key_path: PathBuf,
    aggregation: bool,
    verbose_mode: VerboseMode,
) -> ProofmanResult<SoundnessToml> {
    // Check proving_key_path exists
    if !proving_key_path.exists() {
        return Err(ProofmanError::InvalidParameters(format!(
            "Proving key folder not found at path: {proving_key_path:?}"
        )));
    }

    let mpi_ctx = Arc::new(MpiCtx::new());

    let pctx = ProofCtx::<F>::create_ctx(proving_key_path, HashMap::new(), aggregation, false, verbose_mode, mpi_ctx)?;

    let setups_aggregation =
        Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, false, &ParamsGPU::new(false)));

    let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, &ParamsGPU::new(false));

    let mut circuits = Vec::new();

    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let (air_name, air_info) = get_soundness_air_info(sctx.get_setup(airgroup_id, air_id)?);
            circuits.push(TomlCircuit { name: air_name, group: "basic".to_string(), air: air_info });
        }
    }

    if aggregation {
        let sctx_compressor = setups_aggregation.sctx_compressor.as_ref().unwrap();
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let (air_name, air_info) = get_soundness_air_info(sctx_compressor.get_setup(airgroup_id, air_id)?);
                    circuits.push(TomlCircuit { name: format!("{}-compressor", air_name), group: "compression".to_string(), air: air_info });
                }
            }
        }

        let sctx_recursive2 = setups_aggregation.sctx_recursive2.as_ref().unwrap();
        let n_airgroups = pctx.global_info.air_groups.len();
        if n_airgroups > 1 {
            for airgroup in 0..n_airgroups {
                let (_, air_info) = get_soundness_air_info(sctx_recursive2.get_setup(airgroup, 0)?);
                circuits.push(TomlCircuit { name: format!("Recursive2 - Airgroup_{}", airgroup), group: "aggregation".to_string(), air: air_info });
            }
        } else {
            let (_, air_info) = get_soundness_air_info(sctx_recursive2.get_setup(0, 0)?);
            circuits.push(TomlCircuit { name: "Recursive2".to_string(), group: "aggregation".to_string(), air: air_info });
        }

        let setup_final_circuit = setups_aggregation.setup_vadcop_final.as_ref().unwrap();
        let (_, final_air_info) = get_soundness_air_info(setup_final_circuit);
        circuits.push(TomlCircuit { name: "Final".to_string(), group: "final".to_string(), air: final_air_info });
    }

    Ok(SoundnessToml {
        zkevm: ZkevmConfig {
            name: "ZisK".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_family: "FRI_STARK".to_string(),
            field: "Goldilocks^3".to_string(),
            hash_size_bits: 256,
        },
        circuits,
    })
}
