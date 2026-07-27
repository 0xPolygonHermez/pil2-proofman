use serde::Serialize;
use tabled::{Alignment, Modify, Table, Tabled, object::Segment};
use pil2_stark_setup::types::security::{
    self,
    pcs::{Batching, Fri, FriConfig, Whir, WhirConfig, WhirSecurityParams},
    regimes::DecodingRegime,
};
use proofman_multilinear::{AirIr, WhirParams as ProverWhirParams, num_fold_rounds};
use proofman_common::{
    Setup, SetupsVadcop, ProofType, ProofmanError, ProofmanResult, MpiCtx, ProofCtx, SetupCtx, VerboseMode,
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
    pub trace_bits: u64,
    pub rate: f64,
    #[tabled(rename = "max_degree")]
    pub constraint_max_degree: u64,
    #[tabled(rename = "fixed_cols")]
    pub num_columns_fixed: u64,
    #[tabled(rename = "witness_cols")]
    pub num_columns_witness: u64,
    #[tabled(rename = "custom_cols")]
    pub num_columns_custom: u64,
    #[tabled(rename = "total_cols")]
    pub num_columns: u64,
    #[tabled(rename = "constraints")]
    pub num_constraints: u64,
    #[tabled(rename = "openings")]
    pub opening_points: u64,
    pub batch_size: u64,
    #[tabled(rename = "batching")]
    pub batching_mode: String,
    #[tabled(rename = "queries")]
    pub num_queries: u64,
    #[tabled(rename = "fri_folds")]
    pub fri_folding_factors: String,
    #[tabled(rename = "fri_early_stop")]
    pub fri_early_stop_degree: u64,
    #[tabled(rename = "grinding")]
    pub grinding_query_phase: u64,
    #[tabled(rename = "security")]
    pub security_bits: u32,
    pub proof_size: String,
}

#[derive(Serialize)]
pub struct SoundnessToml {
    pub zkevm: ZkevmConfig,
    pub circuits: Vec<TomlCircuit>,
    /// AIRs provable with the multilinear (WHIR) prover. Only shown in the
    /// printed summary for now, not serialized to the TOML.
    #[serde(skip)]
    pub multilinear: Vec<MlTableRow>,
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

    #[serde(rename = "num_lookups_M")]
    pub num_lookups_m: u32,

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
    /// log2 of the trace length.
    pub trace_bits: u64,
    pub rate: f64,
    pub constraint_max_degree: u64,
    pub num_columns_fixed: u64,
    pub num_columns_witness: u64,
    pub num_columns_custom: u64,
    pub num_columns: u64,
    pub num_constraints: u64,
    pub opening_points: u64,
    pub batch_size: u64,
    pub batching_mode: String,
    pub num_queries: u64,
    pub fri_folding_factors: Vec<u64>,
    pub fri_early_stop_degree: u64,
    pub grinding_query_phase: u64,
    pub regime: String,
    /// Total PCS security: the minimum over all phase levels.
    pub security_bits: u32,
    /// Per-phase PCS security levels (batching, commit rounds, query phase).
    pub security_levels: Vec<PhaseSecurity>,
    pub proof_size: String,
}

#[derive(Serialize, Clone)]
pub struct PhaseSecurity {
    pub phase: String,
    pub bits: u32,
}

/// Summary row for the multilinear (WHIR) PCS of one AIR, audited from the
/// schedule pinned in its `.mlinfo.bin` artifact.
#[derive(Tabled)]
pub struct MlTableRow {
    pub name: String,
    pub trace_bits: u64,
    pub rate: f64,
    #[tabled(rename = "total_cols")]
    pub num_columns: u64,
    #[tabled(rename = "folding")]
    pub folding_factor: u64,
    #[tabled(rename = "rounds")]
    pub fold_rounds: u64,
    #[tabled(rename = "final_bits")]
    pub final_poly_bits: u64,
    #[tabled(rename = "queries")]
    pub num_queries: u64,
    #[tabled(rename = "grinding")]
    pub grinding_bits: u64,
    pub hash: String,
    #[tabled(rename = "security")]
    pub security_bits: u32,
}

impl AirTableRow {
    fn from_air_info(name: &str, air: &AirInfoSoundness) -> Self {
        AirTableRow {
            name: name.to_string(),
            trace_bits: air.trace_bits,
            rate: air.rate,
            constraint_max_degree: air.constraint_max_degree,
            num_columns: air.num_columns,
            num_columns_fixed: air.num_columns_fixed,
            num_columns_witness: air.num_columns_witness,
            num_columns_custom: air.num_columns_custom,
            num_constraints: air.num_constraints,
            opening_points: air.opening_points,
            batch_size: air.batch_size,
            batching_mode: air.batching_mode.clone(),
            num_queries: air.num_queries,
            fri_folding_factors: format!("{:?}", air.fri_folding_factors),
            fri_early_stop_degree: air.fri_early_stop_degree,
            grinding_query_phase: air.grinding_query_phase,
            security_bits: air.security_bits,
            proof_size: air.proof_size.clone(),
        }
    }
}

pub fn print_soundness_table(soundness: &SoundnessToml) {
    let group_rows = |group: &str| -> Vec<AirTableRow> {
        soundness
            .circuits
            .iter()
            .filter(|circuit| circuit.group == group)
            .map(|circuit| AirTableRow::from_air_info(&circuit.name, &circuit.air))
            .collect()
    };

    fn render<I: IntoIterator>(rows: I) -> Table
    where
        I::Item: Tabled,
    {
        Table::new(rows).with(Modify::new(Segment::all()).with(Alignment::center()))
    }

    println!("=== Basics ===");
    println!("{}", render(group_rows("basic")));

    if !soundness.multilinear.is_empty() {
        println!("=== Multilinear (WHIR) ===");
        println!("{}", render(&soundness.multilinear));
    }

    let compressor_rows = group_rows("compression");
    if !compressor_rows.is_empty() {
        println!("=== Compressor ===");
        println!("{}", render(compressor_rows));
    }

    let aggregation_rows = group_rows("aggregation");
    if !aggregation_rows.is_empty() {
        println!("=== Aggregation ===");
        println!("{}", render(aggregation_rows));
    }

    let final_rows = group_rows("final");
    if !final_rows.is_empty() {
        println!("=== Final Circuit ===");
        println!("{}", render(final_rows));
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

    let custom_cols = setup
        .stark_info
        .custom_commits
        .iter()
        .map(|c| c.stage_widths.iter().map(|&w| w as u64).sum::<u64>())
        .sum::<u64>();

    let stark_struct = &setup.stark_info.stark_struct;
    let rate = 1.0 / (1 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64;
    let batch_size = (setup.stark_info.ev_map.len() as u64).max(1);
    let batching = Batching::Powers;
    let fri_folding_bits: Vec<u32> =
        stark_struct.steps.windows(2).map(|pair| (pair[0].n_bits - pair[1].n_bits) as u32).collect();
    let fri_folding_factors: Vec<u64> = fri_folding_bits.iter().map(|&b| 1u64 << b).collect();
    let fri_early_stop_degree = 1u64 << stark_struct.steps.last().unwrap().n_bits;

    // Rebuild the solved FRI PCS from the free parameters stored in the
    // proving key; the query count and grinding split are re-deduced and
    // cross-checked against the stored values.
    let fri = Fri::new(FriConfig {
        field_size: security::goldilocks_safe_extension_field_size(),
        trace_length: 1u32 << stark_struct.n_bits,
        rate,
        batch_size,
        batching,
        log_folding_factors: fri_folding_bits.clone(),
        max_grinding_bits_query: stark_struct.pow_bits,
        use_max_grinding_bits_query: true,
        tree_arity: stark_struct.merkle_tree_arity,
        hash_size_bits: 256,
        target_security_bits: 128,
        regime: DecodingRegime::Jbr,
    });
    let solved = fri.security_params();
    if solved.n_queries != stark_struct.n_queries || solved.grinding_bits_query as u64 != stark_struct.pow_bits {
        tracing::warn!(
            "{}: proving key pins {} queries / {} pow bits, but the soundness formulas deduce {} / {}; \
             the setup may predate the current formulas",
            setup.air_name,
            stark_struct.n_queries,
            stark_struct.pow_bits,
            solved.n_queries,
            solved.grinding_bits_query,
        );
    }

    let security_levels: Vec<PhaseSecurity> =
        fri.security_levels().into_iter().map(|(phase, bits)| PhaseSecurity { phase, bits }).collect();
    let security_bits = security_levels.iter().map(|l| l.bits).min().unwrap_or(0);

    (
        setup.air_name.clone(),
        AirInfoSoundness {
            trace_bits: stark_struct.n_bits,
            rate,
            constraint_max_degree: setup.stark_info.q_deg + 1,
            num_columns: setup.stark_info.n_constants + witness_cols + custom_cols,
            num_columns_fixed: setup.stark_info.n_constants,
            num_columns_witness: witness_cols,
            num_columns_custom: custom_cols,
            num_constraints: setup.stark_info.n_constraints,
            opening_points: setup.stark_info.opening_points.len() as u64,
            batch_size,
            batching_mode: batching.to_string(),
            num_queries: stark_struct.n_queries,
            fri_folding_factors,
            fri_early_stop_degree,
            grinding_query_phase: stark_struct.pow_bits,
            regime: fri.regime().identifier().to_string(),
            security_bits,
            security_levels,
            proof_size: format_bytes(setup.proof_size as f64 * 8.0),
        },
    )
}

/// Audit the multilinear (WHIR) PCS of one AIR from the schedule pinned in
/// its `.mlinfo.bin` artifact. Returns `None` when the AIR has no multilinear
/// setup (or a stale one that fails to load).
pub fn get_multilinear_air_info<F: PrimeField64>(setup: &Setup<F>) -> Option<MlTableRow> {
    let mlinfo_path = setup.setup_path.with_extension("mlinfo.bin");
    if !mlinfo_path.exists() {
        return None;
    }
    let air_ir = match AirIr::load(&mlinfo_path) {
        Ok(air_ir) => air_ir,
        Err(e) => {
            tracing::warn!("{}: could not load {}: {e}", setup.air_name, mlinfo_path.display());
            return None;
        }
    };

    let params = &air_ir.params;
    let n_bits = air_ir.n_bits as usize;
    // Mirror the prover's fold schedule exactly.
    let k = ProverWhirParams::for_params(params).folding_factor.min(n_bits.max(1));
    let n_rounds = num_fold_rounds(n_bits, k, params.log_final_poly_len);

    // Same audit as the setup's sanity check (`ml_params`): the schedule is
    // pinned by the artifact, so no solving.
    let whir = Whir::with_security_params(
        WhirConfig {
            field_size: security::goldilocks_safe_extension_field_size(),
            trace_length: 1u32 << n_bits,
            rate: f64::exp2(-(params.log_blowup as f64)),
            log_folding_factors: vec![k as u32; n_rounds],
            batch_size: air_ir.total_cols().max(1) as u64, // columns batched by δ into Φ
            batching: Batching::Powers,
            constraint_degree: 3, // ŵ(Z,X) = Z·(deg-1 in X) ⇒ d* = 1+1+1 = 3
            max_grinding_bits_query: params.grinding_bits as u64,
            use_max_grinding_bits_query: true,
            tree_arity: 4, // the multilinear prover's MERKLE_ARITY
            hash_size_bits: 256,
            base_field_bits: 64,
            target_security_bits: 128,
            regime: DecodingRegime::Jbr,
        },
        WhirSecurityParams {
            num_queries: vec![params.n_queries as u64; n_rounds],
            num_ood_samples: vec![1; n_rounds.saturating_sub(1)],
            grinding_bits_batching: 0,
            grinding_bits_folding: vec![vec![0u32; k]; n_rounds],
            grinding_bits_queries: vec![params.grinding_bits as u32; n_rounds],
            grinding_bits_ood: vec![0u32; n_rounds.saturating_sub(1)],
        },
    );
    let security_bits = whir.security_levels().into_iter().map(|(_, bits)| bits).min().unwrap_or(0);

    Some(MlTableRow {
        name: air_ir.name.clone(),
        trace_bits: n_bits as u64,
        rate: f64::exp2(-(params.log_blowup as f64)),
        num_columns: air_ir.total_cols() as u64,
        folding_factor: k as u64,
        fold_rounds: n_rounds as u64,
        final_poly_bits: params.log_final_poly_len as u64,
        num_queries: params.n_queries as u64,
        grinding_bits: params.grinding_bits as u64,
        hash: format!("{:?}", params.hash),
        security_bits,
    })
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

        let lookups_air_info: Vec<Lookup> = bus_info
            .into_iter()
            .map(|(name, info)| {
                let num_lookups_m = if info.num_assumes > 0 { info.num_assumes } else { (info.num_proves > 0) as u32 };
                Lookup {
                    name,
                    logup_type: "univariate".to_string(),
                    rows_l: if info.num_assumes > 0 { info.rows } else { 0 },
                    rows_t: if info.num_proves > 0 { info.rows } else { 0 },
                    num_columns_s: info.num_expressions,
                    num_lookups_m,
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

    let pctx = ProofCtx::<F>::create_ctx(proving_key_path, aggregation, verbose_mode, mpi_ctx, false)?;

    let setups_aggregation = Arc::new(SetupsVadcop::<F>::new(&pctx.global_info, false, aggregation, &[], false)?);

    let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false, &[], false)?;

    let mut circuits = Vec::new();
    let mut multilinear = Vec::new();

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
            if let Some(ml_row) = get_multilinear_air_info(setup) {
                multilinear.push(ml_row);
            }
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
        multilinear,
    })
}
