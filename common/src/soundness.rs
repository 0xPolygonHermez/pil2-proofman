use crate::Setup;
use serde::Serialize;
use proofman_starks_lib_c::get_n_constraints_c;
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
use crate::format_bytes;
use fields::PrimeField64;

#[derive(Tabled)]
pub struct AirTableRow {
    pub name: String,
    pub n_bits: u64,
    pub blowup_factor: u64,
    pub d: u64,
    pub fixed: u64,
    pub witness: u64,
    pub constraints: u64,
    pub opening_points: u64,
    pub pols: u64,
    pub queries: u64,
    pub fri_fa: String,
    pub prover_mem: String,
    pub proof_size: String,
    pub grinding: String, // flattened as string
}

#[derive(Serialize)]
pub struct SoundnessYaml {
    pub basics: HashMap<String, AirYaml>,
    pub compressor: Option<HashMap<String, AirYaml>>,
    pub recursive2: Option<HashMap<String, AirYaml>>,
    pub vadcop_final: Option<AirYaml>,
}

#[derive(Serialize, Clone)]
pub struct GrindingYaml {
    pub aai: u64,
    pub ali: u64,
    pub deep: u64,
    pub batching: u64,
    pub commit_phase: Vec<u64>,
    pub query_phase: u64,
}

#[derive(Serialize, Clone)]
pub struct AirYaml {
    pub n_bits: u64,
    pub blowup_factor: u64,
    pub d: u64,
    pub fixed: u64,
    pub witness: u64,
    pub constraints: u64,
    pub opening_points: u64,
    pub pols: u64,
    pub queries: u64,
    pub fri_fa: Vec<u64>,
    pub proof_size: String,
    pub grinding: GrindingYaml,
    pub prover_mem: String,
}

impl AirTableRow {
    fn from_air_yaml(name: &str, air: &AirYaml) -> Self {
        AirTableRow {
            name: name.to_string(),
            n_bits: air.n_bits,
            blowup_factor: air.blowup_factor,
            d: air.d,
            fixed: air.fixed,
            witness: air.witness,
            constraints: air.constraints,
            opening_points: air.opening_points,
            pols: air.pols,
            queries: air.queries,
            fri_fa: format!("{:?}", air.fri_fa),
            prover_mem: air.prover_mem.clone(),
            proof_size: air.proof_size.clone(),
            grinding: format!(
                "aai:{} ali:{} deep:{} batching:{} commit_phase:{:?} query_phase:{}",
                air.grinding.aai,
                air.grinding.ali,
                air.grinding.deep,
                air.grinding.batching,
                air.grinding.commit_phase,
                air.grinding.query_phase
            ),
        }
    }
}

pub fn print_soundness_table(soundness: &SoundnessYaml) {
    println!("=== Basics ===");
    let basics_rows: Vec<AirTableRow> =
        soundness.basics.iter().map(|(name, air)| AirTableRow::from_air_yaml(name, air)).collect();
    let basics_table = Table::new(basics_rows);
    println!("{}", basics_table);

    if let Some(compressor) = &soundness.compressor {
        println!("=== Compressor ===");
        let compressor_rows: Vec<AirTableRow> =
            compressor.iter().map(|(name, air)| AirTableRow::from_air_yaml(name, air)).collect();
        println!("{}", Table::new(compressor_rows));
    }

    if let Some(rec2) = &soundness.recursive2 {
        println!("=== Aggregation ===");
        let rec2_rows: Vec<AirTableRow> =
            rec2.iter().map(|(name, air)| AirTableRow::from_air_yaml(name, air)).collect();
        println!("{}", Table::new(rec2_rows));
    }

    if let Some(vadcop) = &soundness.vadcop_final {
        println!("=== Vadcop Final ===");
        let row = AirTableRow::from_air_yaml("vadcop_final", vadcop);
        println!("{}", Table::new(vec![row]));
    }
}

pub fn get_soundness_air_info<F: PrimeField64>(setup: &Setup<F>) -> (String, AirYaml) {
    let p_setup = (&setup.p_setup).into();

    (
        setup.air_name.clone(),
        AirYaml {
            n_bits: setup.stark_info.stark_struct.n_bits,
            blowup_factor: setup.stark_info.stark_struct.n_bits_ext - setup.stark_info.stark_struct.n_bits,
            d: setup.stark_info.q_deg + 1,
            fixed: setup.stark_info.n_constants,
            witness: setup.stark_info.map_sections_n["cm1"]
                + setup.stark_info.map_sections_n["cm2"]
                + setup.stark_info.map_sections_n["cm3"],
            constraints: get_n_constraints_c(p_setup),
            opening_points: setup.stark_info.opening_points.len() as u64,
            pols: setup.stark_info.ev_map.len() as u64,
            queries: setup.stark_info.stark_struct.n_queries,
            fri_fa: setup.stark_info.stark_struct.steps.iter().map(|s| s.n_bits).collect(),
            prover_mem: format_bytes(8.0 * setup.prover_buffer_size as f64),
            proof_size: format_bytes(8.0 * setup.proof_size as f64),
            grinding: GrindingYaml {
                aai: 0,
                ali: 0,
                deep: 0,
                batching: 0,
                commit_phase: vec![0; setup.stark_info.stark_struct.steps.len()],
                query_phase: 0,
            },
        },
    )
}

pub fn soundness_info<F: PrimeField64>(
    proving_key_path: PathBuf,
    aggregation: bool,
    verbose_mode: VerboseMode,
) -> ProofmanResult<SoundnessYaml> {
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

    let mut basics = HashMap::new();
    let mut compressor = HashMap::new();
    let mut recursive2 = HashMap::new();
    let mut vadcop_final = None;

    for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
        for (air_id, _) in air_group.iter().enumerate() {
            let (air_name, air_info) = get_soundness_air_info(sctx.get_setup(airgroup_id, air_id)?);
            basics.insert(air_name, air_info);
        }
    }

    if aggregation {
        let sctx_compressor = setups_aggregation.sctx_compressor.as_ref().unwrap();
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let (air_name, air_info) = get_soundness_air_info(sctx_compressor.get_setup(airgroup_id, air_id)?);
                    compressor.insert(air_name, air_info);
                }
            }
        }

        let sctx_recursive2 = setups_aggregation.sctx_recursive2.as_ref().unwrap();
        let n_airgroups = pctx.global_info.air_groups.len();
        if n_airgroups > 1 {
            for airgroup in 0..n_airgroups {
                let (_, air_info) = get_soundness_air_info(sctx_recursive2.get_setup(airgroup, 0)?);
                recursive2.insert(format!("Recursive1 / Recursive2 - Airgroup_{}", airgroup), air_info);
            }
        } else {
            let (_, air_info) = get_soundness_air_info(sctx_recursive2.get_setup(0, 0)?);
            recursive2.insert("Recursive1 / Recursive2".to_string(), air_info);
        }

        let setup_vadcop_final = setups_aggregation.setup_vadcop_final.as_ref().unwrap();
        let (_, final_air_info) = get_soundness_air_info(setup_vadcop_final);
        vadcop_final = Some(final_air_info);
    }

    let soundness_info = SoundnessYaml {
        basics,
        compressor: if aggregation { Some(compressor) } else { None },
        recursive2: if aggregation { Some(recursive2) } else { None },
        vadcop_final,
    };

    Ok(soundness_info)
}
