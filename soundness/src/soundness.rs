use serde::Serialize;
use tabled::{Tabled, Table};
use proofman_common::{
    Setup, SetupsVadcop, ProofType, ParamsGPU, ProofmanError, ProofmanResult, MpiCtx, ProofCtx, SetupCtx, VerboseMode,
    format_bytes,
};
use proofman_hints::{get_hint_ids_by_name, get_hint_field_constant_a, HintFieldOptions};
use pil_std_lib::{get_hint_field_constant_as_string, get_hint_field_constant_as_field, get_hint_field_constant_as};
use std::path::PathBuf;
use std::sync::Arc;
use fields::PrimeField64;
use std::collections::BTreeMap;

#[derive(Tabled)]
pub struct AirTableRow {
    pub name: String,
    pub trace_length: u64,
    pub rho: f64,
    pub air_max_degree: u64,
    pub num_columns_fixed: u64,
    pub num_columns_witness: u64,
    pub num_columns: u64,
    pub num_constraints: u64,
    pub opening_points: u64,
    pub batch_size: u64,
    pub power_batching: bool,
    pub num_queries: u64,
    pub fri_folding_factors: String,
    pub fri_early_stop_degree: u64,
    pub grinding_query_phase: u64,
    pub gap_to_radius: f64,
    pub proof_size: String,
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
pub struct Lookup {
    pub name: String,
    pub logup_type: String,

    #[serde(rename = "rows_L")]
    pub rows_l: u32,

    #[serde(rename = "rows_T")]
    pub rows_t: u32,

    #[serde(rename = "num_columns_S")]
    pub num_columns_s: u32,

    #[serde(rename = "num_columns_M")]
    pub num_columns_m: u32,

    pub grinding_bits_lookup: u32,
}

#[derive(Debug)]
pub struct BusInfo {
    pub rows: u32,
    pub num_expressions: u32,
    pub num_assumes: u32,
    pub num_proves: u32,
}

#[derive(Serialize)]
pub struct TomlCircuit {
    pub name: String,
    pub group: String,
    #[serde(flatten)]
    pub air: AirInfoSoundness,

    pub lookups: Vec<Lookup>,
}

#[derive(Serialize, Clone)]
pub struct AirInfoSoundness {
    pub trace_length: u64,
    pub rho: f64,
    pub air_max_degree: u64,
    pub num_columns_fixed: u64,
    pub num_columns_witness: u64,
    pub num_columns: u64,
    pub num_constraints: u64,
    pub opening_points: u64,
    pub batch_size: u64,
    pub power_batching: bool,
    pub num_queries: u64,
    pub fri_folding_factors: Vec<u64>,
    pub fri_early_stop_degree: u64,
    pub grinding_query_phase: u64,
    pub gap_to_radius: f64,
    pub proof_size: String,
}

impl AirTableRow {
    fn from_air_info(name: &str, air: &AirInfoSoundness) -> Self {
        AirTableRow {
            name: name.to_string(),
            trace_length: air.trace_length,
            rho: air.rho,
            air_max_degree: air.air_max_degree,
            num_columns: air.num_columns,
            num_columns_fixed: air.num_columns_fixed,
            num_columns_witness: air.num_columns_witness,
            num_constraints: air.num_constraints,
            opening_points: air.opening_points,
            batch_size: air.batch_size,
            power_batching: air.power_batching,
            num_queries: air.num_queries,
            fri_folding_factors: format!("{:?}", air.fri_folding_factors),
            fri_early_stop_degree: air.fri_early_stop_degree,
            grinding_query_phase: air.grinding_query_phase,
            gap_to_radius: air.gap_to_radius,
            proof_size: air.proof_size.clone(),
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

pub fn get_soundness_air_info<F: PrimeField64>(setup: &Setup<F>) -> (String, AirInfoSoundness) {
    let witness_cols = setup
        .stark_info
        .map_sections_n
        .iter()
        .filter(|(k, _)| k.as_str() != "const" && k.as_str() != "cm3")
        .map(|(_, n)| n)
        .sum::<u64>();
    (
        setup.air_name.clone(),
        AirInfoSoundness {
            trace_length: 1 << setup.stark_info.stark_struct.n_bits,
            rho: 1.0 / (1 << (setup.stark_info.stark_struct.n_bits_ext - setup.stark_info.stark_struct.n_bits)) as f64,
            air_max_degree: setup.stark_info.q_deg + 1,
            num_columns: setup.stark_info.n_constants + witness_cols,
            num_columns_fixed: setup.stark_info.n_constants,
            num_columns_witness: witness_cols,
            num_constraints: setup.stark_info.n_constraints,
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
            grinding_query_phase: setup.stark_info.stark_struct.pow_bits,
            gap_to_radius: setup.stark_info.security.proximity_gap,
            proof_size: format_bytes(setup.proof_size as f64 * 8.0),
        },
    )
}

pub fn get_bus_air_info<F: PrimeField64>(pctx: &ProofCtx<F>, setup: &Setup<F>) -> ProofmanResult<Vec<Lookup>> {
    let p_expressions_bin = setup.p_setup.p_expressions_bin;

    let mut lookups = vec![];

    for piop_type in ["gprod", "gsum"] {
        let debug_data_name = format!("{}_debug_data", piop_type);

        let debug_data_hints = get_hint_ids_by_name(p_expressions_bin, &debug_data_name);

        let num_rows = 1 << setup.stark_info.stark_struct.n_bits;

        let mut bus_info: BTreeMap<String, BusInfo> = BTreeMap::new();

        for hint in debug_data_hints {
            let opids = get_hint_field_constant_a(
                pctx,
                setup,
                setup.airgroup_id,
                setup.air_id,
                hint as usize,
                "opids",
                HintFieldOptions::default(),
            )?;

            let name_piop = get_hint_field_constant_as_string(
                pctx,
                setup,
                setup.airgroup_id,
                setup.air_id,
                hint as usize,
                "name_piop",
                HintFieldOptions::default(),
            )?;

            let len_expressions = get_hint_field_constant_as_field(
                pctx,
                setup,
                setup.airgroup_id,
                setup.air_id,
                hint as usize,
                "len_expressions",
                HintFieldOptions::default(),
            )?;

            let type_piop = get_hint_field_constant_as::<u64, F>(
                pctx,
                setup,
                setup.airgroup_id,
                setup.air_id,
                hint as usize,
                "type_piop",
                HintFieldOptions::default(),
            )?;

            let is_assume = match type_piop {
                0 | 2 => true,
                1 => false,
                _ => unreachable!(),
            };

            let name = format!("{}_{}_{}", name_piop, piop_type, opids);

            let entry = bus_info.entry(name.clone()).or_insert(BusInfo {
                rows: num_rows,
                num_expressions: len_expressions.as_canonical_u64() as u32,
                num_assumes: 0,
                num_proves: 0,
            });

            if is_assume {
                entry.num_assumes += 1;
            } else {
                entry.num_proves += 1;
            }
        }

        println!("BUS INFO for {}:{} {:?} {:?}", setup.airgroup_id, setup.air_id, setup.setup_type, bus_info);
        let lookups_air_info: Vec<Lookup> = bus_info
            .into_iter()
            .map(|(name, info)| {
                let num_columns_m = if info.num_assumes > 0 { info.num_assumes } else { (info.num_proves > 0) as u32 };
                Lookup {
                    name,
                    logup_type: "univariate".to_string(),
                    rows_l: if info.num_assumes > 0 { info.rows } else { 0 },
                    rows_t: if info.num_proves > 0 { info.rows } else { 0 },
                    num_columns_s: info.num_expressions,
                    num_columns_m,
                    grinding_bits_lookup: 0,
                }
            })
            .collect();
        lookups.extend(lookups_air_info);
    }
    Ok(lookups)
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

    let pctx = ProofCtx::<F>::create_ctx(proving_key_path, aggregation, verbose_mode, mpi_ctx)?;

    let setups_aggregation =
        Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, &ParamsGPU::new(false), &[]));

    let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, &ParamsGPU::new(false), &[]);

    let mut circuits = Vec::new();

    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let (air_name, air_info) = get_soundness_air_info(setup);
            let lookup_info = get_bus_air_info(&pctx, setup)?;
            circuits.push(TomlCircuit {
                name: air_name,
                group: "basic".to_string(),
                air: air_info,
                lookups: lookup_info,
            });
        }
    }

    if aggregation {
        let sctx_compressor = setups_aggregation.sctx_compressor.as_ref().unwrap();
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id)?;
                    let (air_name, air_info) = get_soundness_air_info(setup);
                    let lookup_info = get_bus_air_info(&pctx, setup)?;
                    circuits.push(TomlCircuit {
                        name: format!("{}-compressor", air_name),
                        group: "compression".to_string(),
                        air: air_info,
                        lookups: lookup_info,
                    });
                }
            }
        }

        let sctx_recursive2 = setups_aggregation.sctx_recursive2.as_ref().unwrap();
        let n_airgroups = pctx.global_info.air_groups.len();
        if n_airgroups > 1 {
            for airgroup in 0..n_airgroups {
                let setup = sctx_recursive2.get_setup(airgroup, 0)?;
                let (_, air_info) = get_soundness_air_info(setup);
                let lookup_info = get_bus_air_info(&pctx, setup)?;
                circuits.push(TomlCircuit {
                    name: format!("Recursive2 - Airgroup_{}", airgroup),
                    group: "aggregation".to_string(),
                    air: air_info,
                    lookups: lookup_info,
                });
            }
        } else {
            let setup = sctx_recursive2.get_setup(0, 0)?;
            let (_, air_info) = get_soundness_air_info(setup);
            let lookup_info = get_bus_air_info(&pctx, setup)?;
            circuits.push(TomlCircuit {
                name: "Recursive2".to_string(),
                group: "aggregation".to_string(),
                air: air_info,
                lookups: lookup_info,
            });
        }

        let setup_final_circuit = setups_aggregation.setup_vadcop_final.as_ref().unwrap();
        let (_, final_air_info) = get_soundness_air_info(setup_final_circuit);
        let lookup_info = get_bus_air_info(&pctx, setup_final_circuit)?;
        circuits.push(TomlCircuit {
            name: "Final".to_string(),
            group: "final".to_string(),
            air: final_air_info,
            lookups: lookup_info,
        });

        let setup_final_compressed_circuit = setups_aggregation.setup_vadcop_final_compressed.as_ref().unwrap();
        let (_, final_compressed_air_info) = get_soundness_air_info(setup_final_compressed_circuit);
        let lookup_info_c = get_bus_air_info(&pctx, setup_final_compressed_circuit)?;
        circuits.push(TomlCircuit {
            name: "Final_Compressed".to_string(),
            group: "final_compressed".to_string(),
            air: final_compressed_air_info,
            lookups: lookup_info_c,
        });
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
