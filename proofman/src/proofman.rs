use libloading::{Library, Symbol};
use log::info;
use proofman_common::skip_prover_instance;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{save_challenges_c, save_proof_values_c, save_publics_c};
use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write;
use std::io::Read;
use std::error::Error;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use p3_field::AbstractField;
use p3_field::PrimeField64;
use proofman_starks_lib_c::{
    gen_proof_c, commit_witness_c, calculate_hash_c, load_custom_commit_c, calculate_impols_expressions_c,
};

use std::{collections::HashSet, path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessManager};

use crate::verify_basic_proof;
use crate::verify_global_constraints_proof;
use crate::{
    verify_constraints_proof, check_paths, print_summary_info, aggregate_proofs, get_buff_sizes,
    generate_vadcop_recursive1_proof,
};

use proofman_common::{ProofCtx, ProofType, ProofOptions, SetupCtx, SetupsVadcop};

use std::ffi::c_void;

use proofman_util::{
    create_buffer_fast, timer_start_info, timer_stop_and_log_info, timer_stop_and_log_trace, timer_start_trace,
};

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + 'static> ProofMan<F> {
    const MY_NAME: &'static str = "ProofMan";

    #[allow(clippy::too_many_arguments)]
    pub fn verify_proof_constraints(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (pctx, setups) = Self::initialize_proofman_1(
            witness_lib_path.clone(),
            rom_path.clone(),
            public_inputs_path.clone(),
            input_data_path.clone(),
            proving_key_path,
            output_dir_path,
            custom_commits_fixed,
            options,
        )?;
        let sctx = setups.sctx.clone();

        let wcm = Arc::new(WitnessManager::new(
            pctx.clone(),
            sctx.clone(),
            rom_path.clone(),
            public_inputs_path.clone(),
            input_data_path,
        ));

        Self::execute(witness_lib_path, wcm.clone())?;

        pctx.dctx_assign_instances();

        print_summary_info(Self::MY_NAME, pctx.clone(), setups.clone());

        Self::initialize_fixed_pols(setups.clone(), pctx.clone())?;

        pctx.dctx_close();

        let transcript: FFITranscript = FFITranscript::new(2, true);
        let dummy_element = [F::zero(), F::one(), F::two(), F::neg_one()];
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let global_challenge = [F::zero(); 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);
        pctx.set_global_challenge(2, global_challenge.to_vec());
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let valid_constraints = Arc::new(AtomicBool::new(true));

        let instances = pctx.dctx_get_instances();
        let airgroup_values_air_instances = Arc::new(Mutex::new(Vec::new()));

        let mut merkelize_handle: Option<std::thread::JoinHandle<()>> = None;

        for (instance_id, (airgroup_id, air_id, all)) in instances.iter().enumerate() {
            let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
            let (skip, _) = skip_prover_instance(&pctx, instance_id);

            timer_start_info!(GENERATING_WITNESS);

            if skip || (!all && !pctx.dctx_is_my_instance(instance_id)) {
                continue;
            };

            wcm.calculate_witness(1, &[instance_id]);

            timer_stop_and_log_info!(GENERATING_WITNESS);

            // Ensure the previous Merkelization is done before continuing
            if let Some(handle) = merkelize_handle.take() {
                handle.join().unwrap();
            }

            if !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            Self::initialize_air_instance(pctx.clone(), sctx.clone(), instance_id, true);

            #[cfg(feature = "diagnostic")]
            {
                let invalid_initialization = Self::diagnostic_instance(pctx.clone(), sctx.clone(), instance_id);
                if invalid_initialization {
                    return Err("Invalid initialization".into());
                }
            }

            let wcm_cloned = wcm.clone();
            let pctx_cloned = pctx.clone();
            let sctx_cloned = sctx.clone();
            let valid_constraints_cloned = valid_constraints.clone();
            let airgroup_values_air_instances_cloned = airgroup_values_air_instances.clone();

            let verify_constraints = pctx.options.verify_constraints;
            
            let airgroup_id2 = *airgroup_id;
            let air_id2 = *air_id;
            let air_instance_id2 = air_instance_id;
            
            merkelize_handle = Some(std::thread::spawn(move || {
                wcm_cloned.calculate_witness(2, &[instance_id]);
                Self::calculate_im_pols(2, sctx_cloned.clone(), pctx_cloned.clone(), instance_id);

                wcm_cloned.debug(&[instance_id]);

                if verify_constraints {
                    let valid = verify_constraints_proof(pctx_cloned.clone(), sctx_cloned, instance_id);
                    valid_constraints_cloned.fetch_and(valid, Ordering::Relaxed);
                }

                airgroup_values_air_instances_cloned
                    .lock()
                    .unwrap()
                    .push(pctx_cloned.get_air_instance_airgroup_values(airgroup_id2, air_id2, air_instance_id2));
                pctx_cloned.free_instance(instance_id);
            }));
        }

        if let Some(handle) = merkelize_handle {
            handle.join().unwrap();
        }

        wcm.end();

        let check_global_constraints = pctx.options.debug_info.debug_instances.is_empty()
            || !pctx.options.debug_info.debug_global_instances.is_empty();

        if check_global_constraints {
            let airgroup_values_air_instances =
                Arc::try_unwrap(airgroup_values_air_instances).unwrap().into_inner().unwrap();
            let airgroupvalues_u64 = aggregate_airgroupvals(pctx.clone(), airgroup_values_air_instances);
            let airgroupvalues = pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

            if pctx.dctx_get_rank() == 0 {
                let valid_global_constraints =
                    verify_global_constraints_proof(pctx.clone(), sctx.clone(), airgroupvalues);

                if valid_constraints.load(Ordering::Relaxed) && valid_global_constraints.is_ok() {
                    return Ok(());
                } else {
                    return Err("Constraints were not verified".into());
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_proof(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (pctx, setups) = Self::initialize_proofman_1(
            witness_lib_path.clone(),
            rom_path.clone(),
            public_inputs_path.clone(),
            input_data_path.clone(),
            proving_key_path,
            output_dir_path.clone(),
            custom_commits_fixed,
            options,
        )?;
        let sctx = setups.sctx.clone();

        let wcm = Arc::new(WitnessManager::new(
            pctx.clone(),
            sctx.clone(),
            rom_path.clone(),
            public_inputs_path.clone(),
            input_data_path.clone(),
        ));

        Self::execute(witness_lib_path, wcm.clone())?;

        pctx.dctx_assign_instances();

        print_summary_info(Self::MY_NAME, pctx.clone(), setups.clone());

        Self::initialize_fixed_pols(setups.clone(), pctx.clone())?;

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        Self::initialize_fixed_tree(setups.clone(), pctx.clone());
        pctx.dctx_barrier();

        Self::write_fixed_pols_tree(setups.clone(), pctx.clone());
        pctx.dctx_barrier();

        pctx.dctx_close();

        timer_start_info!(GENERATING_VADCOP_PROOF);

        timer_start_info!(GENERATING_PROOFS);

        let mut values = vec![0; my_instances.len() * 4];

        let aux_trace_contribution: Vec<F> = create_buffer_fast(pctx.max_contribution_air_buffer_size as usize);
        let aux_trace_contribution_ptr = aux_trace_contribution.as_ptr() as *mut u8;
        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if !all && !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            wcm.calculate_witness(1, &[instance_id]);

            if !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            let value = Self::get_contribution_air(pctx.clone(), sctx.clone(), instance_id, aux_trace_contribution_ptr);
            if !all {
                pctx.free_instance(instance_id);
            }
            for id in 0..4 {
                values[pctx.dctx_get_instance_idx(instance_id) * 4 + id] = value[id];
            }
        }

        Self::calculate_global_challenge(pctx.clone(), values);

        let (mut circom_witness, publics, trace, prover_buffer) = if pctx.options.aggregation {
            let (circom_witness_size, publics_size, trace_size, prover_buffer_size) =
                get_buff_sizes(pctx.clone(), setups.clone())?;
            let circom_witness: Vec<F> = create_buffer_fast(circom_witness_size);
            let publics: Vec<F> = create_buffer_fast(publics_size);
            let trace: Vec<F> = create_buffer_fast(trace_size);
            let prover_buffer: Vec<F> = create_buffer_fast(prover_buffer_size);
            (circom_witness, publics, trace, prover_buffer)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        let aux_trace: Vec<F> = create_buffer_fast(pctx.max_prover_buffer_size as usize);

        let mut valid_proofs = false;
        let mut proofs = Vec::new();
        let mut airgroup_values_air_instances = Vec::new();
        for (instance_id, (airgroup_id, air_id, all)) in instances.iter().enumerate() {
            if !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            if !all {
                timer_start_info!(GENERATING_WITNESS);
                wcm.calculate_witness(1, &[instance_id]);
                timer_stop_and_log_info!(GENERATING_WITNESS);
            }
            Self::initialize_air_instance(pctx.clone(), sctx.clone(), instance_id, false);

            let setup = sctx.get_setup(*airgroup_id, *air_id);
            let p_setup = (&setup.p_setup).into();
            let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
            let air_instance_name = &pctx.global_info.airs[*airgroup_id][*air_id].name;
            timer_start_info!(GENERATING_PROOF);

            let mut steps_params = pctx.get_air_instance_params(&sctx, instance_id, true);
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;

            let p_steps_params = (&steps_params).into();

            let output_file_path = output_dir_path.join(format!("proofs/{}_{}.json", air_instance_name, instance_id));

            let proof_file = match pctx.options.debug_info.save_proofs_to_file {
                true => output_file_path.to_string_lossy().into_owned(),
                false => String::from(""),
            };

            let proof = gen_proof_c(
                p_setup,
                p_steps_params,
                pctx.get_buff_helper_ptr(),
                pctx.get_global_challenge_ptr(),
                &proof_file,
                *airgroup_id as u64,
                *air_id as u64,
                air_instance_id as u64,
            );

            airgroup_values_air_instances.push(pctx.get_air_instance_airgroup_values(
                *airgroup_id,
                *air_id,
                air_instance_id,
            ));

            timer_start_info!(FREE_INSTANCE);
            pctx.free_instance(instance_id);
            timer_stop_and_log_info!(FREE_INSTANCE);

            if pctx.options.aggregation {
                timer_start_info!(GENERATING_COMPRESSOR_AND_RECURSIVE1_PROOF);
                let proof_recursive = generate_vadcop_recursive1_proof(
                    &pctx,
                    &setups,
                    instance_id,
                    proof,
                    &mut circom_witness,
                    &publics,
                    &trace,
                    &prover_buffer,
                    output_dir_path.clone(),
                )?;
                proofs.push(proof_recursive);
                timer_stop_and_log_info!(GENERATING_COMPRESSOR_AND_RECURSIVE1_PROOF);
                timer_stop_and_log_info!(GENERATING_PROOF);
            } else {
                proofs.push(proof);
                timer_stop_and_log_info!(GENERATING_PROOF);
                valid_proofs = verify_basic_proof(pctx.clone(), instance_id, proof);
            }
        }

        timer_stop_and_log_info!(GENERATING_PROOFS);

        let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
        let global_info_file: &str = global_info_path.to_str().unwrap();

        save_challenges_c(pctx.get_challenges_ptr(), global_info_file, output_dir_path.to_string_lossy().as_ref());
        save_proof_values_c(pctx.get_proof_values_ptr(), global_info_file, output_dir_path.to_string_lossy().as_ref());
        save_publics_c(
            pctx.global_info.n_publics as u64,
            pctx.get_publics_ptr(),
            output_dir_path.to_string_lossy().as_ref(),
        );

        if !pctx.options.aggregation {
            let check_global_constraints = pctx.options.debug_info.debug_instances.is_empty()
                || !pctx.options.debug_info.debug_global_instances.is_empty();

            if check_global_constraints {
                let airgroupvalues_u64 = aggregate_airgroupvals(pctx.clone(), airgroup_values_air_instances);
                let airgroupvalues = pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

                if pctx.dctx_get_rank() == 0 {
                    let valid_global_constraints =
                        verify_global_constraints_proof(pctx.clone(), sctx.clone(), airgroupvalues);
                    if valid_global_constraints.is_err() {
                        valid_proofs = false;
                    }
                }
            }

            if valid_proofs {
                return Ok(());
            } else {
                return Err("Basic proofs were not verified".into());
            }
        }

        let agg_proof = aggregate_proofs(Self::MY_NAME, pctx.clone(), setups.clone(), proofs, output_dir_path);
        timer_stop_and_log_info!(GENERATING_VADCOP_PROOF);

        agg_proof
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn initialize_proofman_1(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(Arc<ProofCtx<F>>, Arc<SetupsVadcop<F>>), Box<dyn std::error::Error>> {
        timer_start_info!(INITIALIZING_PROOFMAN_1);

        check_paths(
            &witness_lib_path,
            &rom_path,
            &public_inputs_path,
            &input_data_path,
            &proving_key_path,
            &output_dir_path,
            options.verify_constraints,
        )?;

        let mut pctx: ProofCtx<F> = ProofCtx::create_ctx(proving_key_path.clone(), custom_commits_fixed, options);

        let setups = Arc::new(SetupsVadcop::new(
            &pctx.global_info,
            pctx.options.verify_constraints,
            pctx.options.aggregation,
            pctx.options.final_snark,
        ));

        pctx.set_weights(&setups.sctx.clone());
        pctx.set_buff_helper(&setups.sctx.clone());

        let pctx = Arc::new(pctx);

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN_1);

        Ok((pctx, setups))
    }

    fn execute(witness_lib_path: PathBuf, wcm: Arc<WitnessManager<F>>) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing witness");
        timer_start_info!(EXECUTE);

        // Load the witness computation dynamic library
        let library = unsafe { Library::new(&witness_lib_path)? };

        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(wcm.get_pctx().options.verbose_mode)?;
        witness_lib.register_witness(wcm.clone());

        wcm.execute();

        timer_stop_and_log_info!(EXECUTE);
        Ok(())
    }

    fn calculate_global_challenge(pctx: Arc<ProofCtx<F>>, values: Vec<u64>) {
        let transcript = FFITranscript::new(2, true);

        transcript.add_elements(pctx.get_publics_ptr(), pctx.global_info.n_publics);

        let proof_values_stage = pctx.get_proof_values_by_stage(1);
        if !proof_values_stage.is_empty() {
            transcript.add_elements(proof_values_stage.as_ptr() as *mut u8, proof_values_stage.len());
        }

        let all_roots = pctx.dctx_distribute_roots(values);

        // add challenges to transcript in order
        for group_idxs in pctx.dctx_get_my_groups() {
            let mut values = Vec::new();
            for idx in group_idxs.iter() {
                let value = vec![
                    F::from_wrapped_u64(all_roots[*idx]),
                    F::from_wrapped_u64(all_roots[*idx + 1]),
                    F::from_wrapped_u64(all_roots[*idx + 2]),
                    F::from_wrapped_u64(all_roots[*idx + 3]),
                ];
                values.push(value);
            }
            if !values.is_empty() {
                let value = Self::hash_b_tree(values);
                transcript.add_elements(value.as_ptr() as *mut u8, value.len());
            }
        }

        let global_challenge = [F::zero(); 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);

        pctx.set_global_challenge(2, global_challenge.to_vec());
    }

    #[allow(dead_code)]
    fn diagnostic_instance(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>, instance_id: usize) -> bool {
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let air_name = pctx.global_info.airs[airgroup_id][air_id].clone().name;
        let setup = sctx.get_setup(airgroup_id, air_id);
        let cm_pols_map = setup.stark_info.cm_pols_map.as_ref().unwrap();
        let n_cols = *setup.stark_info.map_sections_n.get("cm1").unwrap() as usize;
        let n_rows = 1 << setup.stark_info.stark_struct.n_bits;

        let vals = unsafe {
            std::slice::from_raw_parts(pctx.get_air_instance_trace_ptr(instance_id) as *mut u64, n_cols * n_rows)
        };

        let mut invalid_initialization = false;

        for (pos, val) in vals.iter().enumerate() {
            if *val == u64::MAX - 1 {
                let row = pos / n_cols;
                let col_id = pos % n_cols;
                let col = cm_pols_map.get(col_id).unwrap();
                let col_name = if !col.lengths.is_empty() {
                    let lengths = col.lengths.iter().fold(String::new(), |mut acc, l| {
                        write!(acc, "[{}]", l).unwrap();
                        acc
                    });
                    format!("{}{}", col.name, lengths)
                } else {
                    col.name.clone()
                };
                log::warn!(
                    "{}: Missing initialization {} at row {} of {} in instance {}",
                    Self::MY_NAME,
                    col_name,
                    row,
                    air_name,
                    air_instance_id,
                );
                invalid_initialization = true;
            }
        }

        invalid_initialization
    }

    fn initialize_air_instance(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_id: usize,
        init_aux_trace: bool,
    ) {
        let instances = pctx.dctx_get_instances();

        let mut air_instances_w = pctx.air_instances.write().unwrap();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let setup = sctx.get_setup(airgroup_id, air_id);

        let air_instance = air_instances_w.get_mut(&instance_id).unwrap();
        if init_aux_trace {
            air_instance.init_aux_trace(setup.prover_buffer_size as usize);
        }
        air_instance.init_evals(setup.stark_info.ev_map.len() * 3);
        air_instance.init_challenges(
            (setup.stark_info.challenges_map.as_ref().unwrap().len() + setup.stark_info.stark_struct.steps.len() + 1)
                * 3,
        );

        air_instance.init_custom_commit_fixed_trace(setup.custom_commits_fixed_buffer_size as usize);

        let n_custom_commits = setup.stark_info.custom_commits.len();

        for commit_id in 0..n_custom_commits {
            if setup.stark_info.custom_commits[commit_id].stage_widths[0] > 0 {
                let custom_commit_file_path =
                    pctx.get_custom_commits_fixed_buffer(&setup.stark_info.custom_commits[commit_id].name).unwrap();

                load_custom_commit_c(
                    (&setup.p_setup).into(),
                    commit_id as u64,
                    air_instance.get_custom_commits_fixed_ptr(),
                    custom_commit_file_path,
                );
            }
        }

        let n_airgroup_values = setup.stark_info.airgroupvalues_map.as_ref().unwrap().len();
        let n_air_values = setup.stark_info.airvalues_map.as_ref().unwrap().len();

        if n_air_values > 0 && air_instance.airvalues.is_empty() {
            air_instance.init_airvalues(n_air_values * 3);
        }

        if n_airgroup_values > 0 && air_instance.airgroup_values.is_empty() {
            air_instance.init_airgroup_values(n_airgroup_values * 3);
        }
    }

    fn initialize_fixed_pols(setups: Arc<SetupsVadcop<F>>, pctx: Arc<ProofCtx<F>>) -> Result<(), Box<dyn Error>> {
        info!("{}: Initializing setup fixed pols", Self::MY_NAME);
        timer_start_info!(INITIALIZE_CONST_POLS);

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        let mut airs = Vec::new();
        let mut seen = HashSet::new();

        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id, _) = instances[*instance_id];
            if seen.insert((airgroup_id, air_id)) {
                airs.push((airgroup_id, air_id));
            }
        }

        airs.iter().for_each(|&(airgroup_id, air_id)| {
            let setup = setups.sctx.get_setup(airgroup_id, air_id);
            setup.load_const_pols();
        });

        for (airgroup_id, airs) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = setups.sctx.get_setup(airgroup_id, air_id);
                for custom_commit in &setup.stark_info.custom_commits {
                    if custom_commit.stage_widths[0] > 0 {
                        let custom_commit_name = &custom_commit.name;

                        // Handle the possibility that this returns None
                        let custom_file_path = pctx.get_custom_commits_fixed_buffer(custom_commit_name)?;

                        let mut file = File::open(custom_file_path)?;
                        let mut root_bytes = [0u8; 32];
                        file.read_exact(&mut root_bytes)?;

                        for (idx, p) in custom_commit.public_values.iter().enumerate() {
                            let public_id = p.idx as usize;
                            let byte_range = idx * 8..(idx + 1) * 8;
                            let value = u64::from_le_bytes(root_bytes[byte_range].try_into()?);
                            pctx.set_public_value(value, public_id);
                        }
                    }
                }
            }
        }

        timer_stop_and_log_info!(INITIALIZE_CONST_POLS);

        if pctx.options.aggregation {
            timer_start_info!(INITIALIZE_CONST_POLS_AGGREGATION);

            info!("{}: Initializing setup fixed pols aggregation", Self::MY_NAME);

            let global_info = pctx.global_info.clone();

            let sctx_compressor = setups.sctx_compressor.as_ref().unwrap().clone();
            info!("{}: ··· Initializing setup fixed pols compressor", Self::MY_NAME);
            timer_start_trace!(INITIALIZE_CONST_POLS_COMPRESSOR);

            airs.iter().for_each(|&(airgroup_id, air_id)| {
                if global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id);
                    setup.load_const_pols();
                }
            });
            timer_stop_and_log_trace!(INITIALIZE_CONST_POLS_COMPRESSOR);

            let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap().clone();
            timer_start_trace!(INITIALIZE_CONST_POLS_RECURSIVE1);
            info!("{}: ··· Initializing setup fixed pols recursive1", Self::MY_NAME);
            airs.iter().for_each(|&(airgroup_id, air_id)| {
                let setup = sctx_recursive1.get_setup(airgroup_id, air_id);
                setup.load_const_pols();
            });
            timer_stop_and_log_trace!(INITIALIZE_CONST_POLS_RECURSIVE1);

            let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap().clone();
            timer_start_trace!(INITIALIZE_CONST_POLS_RECURSIVE2);
            info!("{}: ··· Initializing setup fixed pols recursive2", Self::MY_NAME);
            let n_airgroups = global_info.air_groups.len();
            for airgroup in 0..n_airgroups {
                let setup = sctx_recursive2.get_setup(airgroup, 0);
                setup.load_const_pols();
            }
            timer_stop_and_log_trace!(INITIALIZE_CONST_POLS_RECURSIVE2);

            if pctx.dctx_get_rank() == 0 {
                let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap().clone();
                timer_start_trace!(INITIALIZE_CONST_POLS_VADCOP_FINAL);
                info!("{}: ··· Initializing setup fixed pols vadcop final", Self::MY_NAME);
                setup_vadcop_final.load_const_pols();
                timer_stop_and_log_trace!(INITIALIZE_CONST_POLS_VADCOP_FINAL);

                if pctx.options.final_snark {
                    let setup_recursivef = setups.setup_recursivef.as_ref().unwrap().clone();
                    timer_start_trace!(INITIALIZE_CONST_POLS_RECURSIVE_FINAL);
                    info!("{}: ··· Initializing setup fixed pols recursive final", Self::MY_NAME);
                    setup_recursivef.load_const_pols();
                    timer_stop_and_log_trace!(INITIALIZE_CONST_POLS_RECURSIVE_FINAL);
                }
            }
            timer_stop_and_log_info!(INITIALIZE_CONST_POLS_AGGREGATION);
        }

        Ok(())
    }

    fn initialize_fixed_tree(setups: Arc<SetupsVadcop<F>>, pctx: Arc<ProofCtx<F>>) {
        info!("{}: Initializing setup fixed tree", Self::MY_NAME);
        timer_start_info!(INITIALIZE_CONST_TREE);

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        let mut airs = Vec::new();
        let mut seen = HashSet::new();

        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id, _) = instances[*instance_id];
            if seen.insert((airgroup_id, air_id)) {
                airs.push((airgroup_id, air_id));
            }
        }

        airs.iter().for_each(|&(airgroup_id, air_id)| {
            let setup = setups.sctx.get_setup(airgroup_id, air_id);
            setup.load_const_pols_tree();
        });

        timer_stop_and_log_info!(INITIALIZE_CONST_TREE);

        if pctx.options.aggregation {
            timer_start_info!(INITIALIZE_CONST_TREE_AGGREGATION);

            info!("{}: Initializing setup fixed tree aggregation", Self::MY_NAME);

            let global_info = pctx.global_info.clone();

            let sctx_compressor = setups.sctx_compressor.as_ref().unwrap().clone();
            info!("{}: ··· Initializing setup fixed tree compressor", Self::MY_NAME);
            timer_start_trace!(INITIALIZE_CONST_TREE_COMPRESSOR);

            airs.iter().for_each(|&(airgroup_id, air_id)| {
                if global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id);
                    setup.load_const_pols_tree();
                }
            });
            timer_stop_and_log_trace!(INITIALIZE_CONST_TREE_COMPRESSOR);

            let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap().clone();
            timer_start_trace!(INITIALIZE_CONST_TREE_RECURSIVE1);
            info!("{}: ··· Initializing setup fixed tree recursive1", Self::MY_NAME);
            airs.iter().for_each(|&(airgroup_id, air_id)| {
                let setup = sctx_recursive1.get_setup(airgroup_id, air_id);
                setup.load_const_pols_tree();
            });
            timer_stop_and_log_trace!(INITIALIZE_CONST_TREE_RECURSIVE1);

            let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap().clone();
            timer_start_trace!(INITIALIZE_CONST_TREE_RECURSIVE2);
            info!("{}: ··· Initializing setup fixed tree recursive2", Self::MY_NAME);
            let n_airgroups = global_info.air_groups.len();
            for airgroup in 0..n_airgroups {
                let setup = sctx_recursive2.get_setup(airgroup, 0);
                setup.load_const_pols_tree();
            }
            timer_stop_and_log_trace!(INITIALIZE_CONST_TREE_RECURSIVE2);

            if pctx.dctx_get_rank() == 0 {
                let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap().clone();
                timer_start_trace!(INITIALIZE_CONST_TREE_VADCOP_FINAL);
                info!("{}: ··· Initializing setup fixed tree vadcop final", Self::MY_NAME);
                setup_vadcop_final.load_const_pols_tree();
                timer_stop_and_log_trace!(INITIALIZE_CONST_TREE_VADCOP_FINAL);

                if pctx.options.final_snark {
                    let setup_recursivef = setups.setup_recursivef.as_ref().unwrap().clone();
                    timer_start_trace!(INITIALIZE_CONST_TREE_RECURSIVE_FINAL);
                    info!("{}: ··· Initializing setup fixed tree recursive final", Self::MY_NAME);
                    setup_recursivef.load_const_pols_tree();
                    timer_stop_and_log_trace!(INITIALIZE_CONST_TREE_RECURSIVE_FINAL);
                }
            }
            timer_stop_and_log_info!(INITIALIZE_CONST_TREE_AGGREGATION);
        }
    }

    fn write_fixed_pols_tree(setups: Arc<SetupsVadcop<F>>, pctx: Arc<ProofCtx<F>>) {
        timer_start_info!(WRITE_CONST_TREE);
        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        let mut airs = Vec::new();
        let mut seen = HashSet::new();

        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id, _) = instances[*instance_id];
            if seen.insert((airgroup_id, air_id)) {
                airs.push((airgroup_id, air_id));
            }
        }

        airs.iter().for_each(|&(airgroup_id, air_id)| {
            let setup = setups.sctx.get_setup(airgroup_id, air_id);
            if setup.to_write_tree() {
                setup.write_const_tree();
            }
        });

        if pctx.options.aggregation {
            let global_info = pctx.global_info.clone();
            let sctx_compressor = setups.sctx_compressor.as_ref().unwrap().clone();
            airs.iter().for_each(|&(airgroup_id, air_id)| {
                if global_info.get_air_has_compressor(airgroup_id, air_id) {
                    let setup = sctx_compressor.get_setup(airgroup_id, air_id);
                    if pctx.dctx_is_min_rank_owner(airgroup_id, air_id) && setup.to_write_tree() {
                        setup.write_const_tree();
                    }
                }
            });
            let sctx_recursive1 = setups.sctx_recursive1.as_ref().unwrap().clone();
            airs.iter().for_each(|&(airgroup_id, air_id)| {
                let setup = sctx_recursive1.get_setup(airgroup_id, air_id);
                if pctx.dctx_is_min_rank_owner(airgroup_id, air_id) && setup.to_write_tree() {
                    setup.write_const_tree();
                }
            });

            if pctx.dctx_get_rank() == 0 {
                let sctx_recursive2 = setups.sctx_recursive2.as_ref().unwrap().clone();
                let n_airgroups = global_info.air_groups.len();
                for airgroup in 0..n_airgroups {
                    let setup = sctx_recursive2.get_setup(airgroup, 0);
                    if pctx.dctx_is_min_rank_owner(airgroup, 0) && setup.to_write_tree() {
                        setup.write_const_tree();
                    }
                }

                let setup_vadcop_final = setups.setup_vadcop_final.as_ref().unwrap().clone();
                if setup_vadcop_final.to_write_tree() {
                    setup_vadcop_final.write_const_tree();
                }

                if pctx.options.final_snark {
                    let setup_recursivef = setups.setup_recursivef.as_ref().unwrap().clone();
                    if setup_recursivef.to_write_tree() {
                        setup_recursivef.write_const_tree();
                    }
                }
            }
        }

        timer_stop_and_log_info!(WRITE_CONST_TREE);
    }

    pub fn calculate_im_pols(stage: u32, sctx: Arc<SetupCtx<F>>, pctx: Arc<ProofCtx<F>>, instance_id: usize) {
        let instances = pctx.dctx_get_instances();
        let (airgroup_id, air_id, _) = instances[instance_id];
        let setup = sctx.get_setup(airgroup_id, air_id);

        let steps_params = pctx.get_air_instance_params(&sctx, instance_id, false);

        calculate_impols_expressions_c((&setup.p_setup).into(), stage as u64, (&steps_params).into());
    }

    pub fn get_contribution_air(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
    ) -> Vec<u64> {
        let n_field_elements = 4;

        timer_start_info!(GET_CONTRIBUTION_AIR);
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let root = vec![F::zero(); n_field_elements];
        commit_witness_c(
            setup.stark_info.stark_struct.n_bits,
            setup.stark_info.stark_struct.n_bits_ext,
            *setup.stark_info.map_sections_n.get("cm1").unwrap(),
            root.as_ptr() as *mut u8,
            pctx.get_air_instance_trace_ptr(instance_id),
            aux_trace_contribution_ptr,
        );

        let mut value = vec![Goldilocks::zero(); n_field_elements];

        let n_airvalues: usize = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);

        let size = 2 * n_field_elements + n_airvalues;

        let mut values_hash = vec![F::zero(); size];

        let verkey = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic).display().to_string()
            + ".verkey.json";

        let mut file = File::open(&verkey).expect("Unable to open file");
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).expect("Unable to read file");
        let vk: Vec<u64> = serde_json::from_str(&json_str).expect("REASON");
        for j in 0..n_field_elements {
            values_hash[j] = F::from_canonical_u64(vk[j]);
            values_hash[j + n_field_elements] = root[j];
        }

        let air_values = pctx.get_air_instance_air_values(airgroup_id, air_id, air_instance_id);

        let airvalues_map = setup.stark_info.airvalues_map.as_ref().unwrap();
        let mut p = 0;
        let mut count = 0;
        for air_value in airvalues_map {
            if air_value.stage == 1 {
                values_hash[2 * n_field_elements + count] = air_values[p];
                count += 1;
                p += 1;
            }
        }

        calculate_hash_c(value.as_mut_ptr() as *mut u8, values_hash.as_mut_ptr() as *mut u8, size as u64);

        timer_stop_and_log_info!(GET_CONTRIBUTION_AIR);

        value.iter().map(|x| x.as_canonical_u64()).collect::<Vec<u64>>()
    }

    fn hash_b_tree(values: Vec<Vec<F>>) -> Vec<F> {
        if values.len() == 1 {
            return values[0].clone();
        }

        let mut result = Vec::new();

        for i in (0..values.len() - 1).step_by(2) {
            let mut buffer = values[i].clone();
            buffer.extend(values[i + 1].clone());

            let is_value1_zero = values[i].iter().all(|x| *x == F::zero());
            let is_value2_zero = values[i + 1].iter().all(|x| *x == F::zero());

            let mut value;
            if is_value1_zero && is_value2_zero {
                value = vec![F::zero(); 4];
            } else if is_value1_zero {
                value = values[i + 1].clone();
            } else if is_value2_zero {
                value = values[i].clone();
            } else {
                value = vec![F::zero(); 4];
                calculate_hash_c(value.as_mut_ptr() as *mut u8, buffer.as_mut_ptr() as *mut u8, buffer.len() as u64);
            }

            result.push(value);
        }

        if values.len() % 2 != 0 {
            result.push(values[values.len() - 1].clone());
        }

        Self::hash_b_tree(result)
    }
}
