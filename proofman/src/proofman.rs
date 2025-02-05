use libloading::{Library, Symbol};
use log::info;
use proofman_starks_lib_c::calculate_impols_expressions_c;
use std::fs::File;
use std::io::Read;
use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use p3_field::AbstractField;
use p3_field::PrimeField64;
use proofman_starks_lib_c::{gen_proof_c, commit_witness_c, calculate_hash_c};

use std::{collections::HashSet, path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessManager};

use crate::{verify_basic_proofs, verify_constraints_proof, check_paths, print_summary_info, aggregate_proofs, get_buff_sizes, generate_vadcop_recursive1_proof};

use proofman_common::{
    StepsParams, ProofCtx, ProofType, ProofOptions, SetupCtx, SetupsVadcop,
};

use std::ffi::c_void;

use proofman_util::{
    create_buffer_fast, timer_start_info, timer_stop_and_log_info,
    timer_stop_and_log_trace, timer_start_trace,
};

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + 'static> ProofMan<F> {
    const MY_NAME: &'static str = "ProofMan";

    pub fn generate_proof_constraints(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (pctx, setups) = Self::initialize_proofman_1(
            witness_lib_path.clone(),
            rom_path.clone(),
            public_inputs_path.clone(),
            input_data_path.clone(),
            proving_key_path,
            output_dir_path,
            options,
        )?;
        let sctx = setups.sctx.clone();

        let wcm =
            Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), rom_path.clone(), public_inputs_path.clone(), input_data_path));

        Self::execute(witness_lib_path, wcm.clone())?;

        Self::initialize_proofman_2(pctx.clone(), setups.clone());

        wcm.calculate_witness(1);

        #[cfg(feature = "diagnostic")]
        {
            let air_instances = pctx.air_instance_repo.air_instances.read().unwrap();
            let instances = pctx.dctx_get_instances();
            let my_instances = pctx.dctx_get_my_instances();
            let mut missing_initialization = false;
            for instance_id in my_instances.iter() {
                let (airgroup_id, air_id) = instances[*instance_id];
                let air_instance = air_instances.get(instance_id).unwrap();
                let air_instance_id = pctx.dctx_find_air_instance_id(*instance_id);
                let air_name = pctx.global_info.airs[airgroup_id][air_id].clone().name;
                let setup = setups.sctx.get_setup(airgroup_id, air_id);
                let cm_pols_map = setup.stark_info.cm_pols_map.as_ref().unwrap();
                let n_cols = *setup.stark_info.map_sections_n.get("cm1").unwrap() as usize;

                let len = air_instance.trace.len();
                let vals = unsafe { std::slice::from_raw_parts(air_instance.get_trace_ptr() as *mut u64, len) };

                for (pos, val) in vals.iter().enumerate() {
                    if *val == u64::MAX - 1 {
                        let row = pos / n_cols;
                        let col_id = pos % n_cols;
                        let col = cm_pols_map.get(col_id).unwrap();
                        let col_name = if !col.lengths.is_empty() {
                            let lengths = col.lengths.iter().map(|l| format!("[{}]", l)).collect::<String>();
                            &format!("{}{}", col.name, lengths)
                        } else {
                            &col.name
                        };
                        log::warn!(
                            "{}: Missing initialization {} at row {} of {} in instance {}",
                            Self::MY_NAME,
                            col_name,
                            row,
                            air_name,
                            air_instance_id,
                        );
                        missing_initialization = true;
                    }
                }
            }
            if missing_initialization {
                return Err("Missing initialization".into());
            } else {
                log::info!("{}: Witness Initialization is done properly", Self::MY_NAME);
                return Ok(());
            }
        }
        
        pctx.dctx_close();

        Self::initialize_air_instances(pctx.clone(), sctx.clone());

        let transcript: FFITranscript = FFITranscript::new(2, true);

        let dummy_element = [F::zero(), F::one(), F::two(), F::neg_one()];
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let num_commit_stages = pctx.global_info.n_challenges.len() as u32;
        for stage in 2..=num_commit_stages {
            let initial_pos = pctx.global_info.n_challenges.iter().take(stage as usize - 1).sum::<usize>();
            let num_challenges = pctx.global_info.n_challenges[stage as usize - 1];            
            for i in 0..num_challenges {
                transcript.get_challenge(
                    &pctx.challenges.values.write().unwrap()[(initial_pos + i) * 3] as *const F as *mut c_void,
                );
            }
        

            wcm.calculate_witness(stage);

            Self::calculate_im_pols(stage, sctx.clone(), pctx.clone());

            transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);
        }

        wcm.debug();

        if pctx.options.verify_constraints {
            return verify_constraints_proof(pctx.clone(), sctx.clone());
        }

        Ok(())
    }

    pub fn generate_proof(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (pctx, setups) = Self::initialize_proofman_1(
            witness_lib_path.clone(),
            rom_path.clone(),
            public_inputs_path.clone(),
            input_data_path.clone(),
            proving_key_path,
            output_dir_path.clone(),
            options,
        )?;
        let sctx = setups.sctx.clone();

        let wcm =
            Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), rom_path.clone(), public_inputs_path.clone(), input_data_path.clone()));

        Self::execute(witness_lib_path, wcm.clone())?;

        Self::initialize_proofman_2(pctx.clone(), setups.clone());

        wcm.calculate_witness(1);

        Self::initialize_fixed_tree(setups.clone(), pctx.clone());
        pctx.dctx_barrier();

        pctx.dctx_close();

        timer_start_info!(GENERATING_VADCOP_PROOF);

        timer_start_info!(GENERATING_PROOFS);

        Self::initialize_air_instances(pctx.clone(), sctx.clone());

        let values = Self::get_challenge_air(pctx.clone(), sctx.clone());

        let transcript = FFITranscript::new(2, true);

        // pctx.set_public_value(170320164326670721, 4);
        // pctx.set_public_value(17316773654152950811, 5);
        // pctx.set_public_value(11953482335611793423, 6);
        // pctx.set_public_value(12850755888860414039, 7);

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


        let global_challenge = vec![F::zero(); 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);

        pctx.set_global_challenge(global_challenge);

        let air_instances = pctx.air_instance_repo.air_instances.read().unwrap();

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

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();
        
        let mut proofs = Vec::new();
        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id) = instances[*instance_id];
            let setup = sctx.get_setup(airgroup_id, air_id);
            let p_setup = (&setup.p_setup).into();
            let air_instance = air_instances.get(instance_id).unwrap();
            let air_instance_id = pctx.dctx_find_air_instance_id(*instance_id);
            let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;

            timer_start_info!(GENERATING_RECURSIVE_PROOF);
            timer_start_info!(GENERATING_PROOF);

            let steps_params = StepsParams {
                trace: air_instance.get_trace_ptr(),
                aux_trace: air_instance.get_aux_trace_ptr(),
                public_inputs: pctx.get_publics_ptr(),
                proof_values: pctx.get_proof_values_ptr(),
                challenges: air_instance.get_challenges_ptr() as *mut u8,
                airgroup_values: air_instance.get_airgroup_values_ptr(),
                airvalues: air_instance.get_airvalues_ptr(),
                evals: air_instance.get_evals_ptr(),
                xdivxsub: std::ptr::null_mut(),
                p_const_pols: setup.get_const_ptr(),
                p_const_tree: setup.get_const_tree_ptr(),
                custom_commits: air_instance.get_custom_commits_ptr(),
                custom_commits_extended: air_instance.get_custom_commits_extended_ptr(),
            };

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
                airgroup_id as u64,
                air_id as u64,
                air_instance_id as u64,
            );   

            timer_stop_and_log_info!(GENERATING_PROOF);

            if pctx.options.aggregation {
                timer_start_info!(GENERATING_COMPRESSOR_AND_RECURSIVE1_PROOF);
                let proof_recursive = generate_vadcop_recursive1_proof(&pctx, &setups, *instance_id, proof, &mut circom_witness, &publics, &trace, &prover_buffer, output_dir_path.clone())?;
                proofs.push(proof_recursive);
                timer_stop_and_log_info!(GENERATING_COMPRESSOR_AND_RECURSIVE1_PROOF);
            } else {
                proofs.push(proof);
            }

            timer_stop_and_log_info!(GENERATING_RECURSIVE_PROOF);
        }

        timer_stop_and_log_info!(GENERATING_PROOFS);


        let mut valid_proofs = false;
        if !pctx.options.aggregation {
            valid_proofs = verify_basic_proofs(proofs.clone(), pctx.clone(), sctx.clone());
        }

        if !pctx.options.aggregation {
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
    fn initialize_proofman_1(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
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

        let mut pctx: ProofCtx<F> = ProofCtx::create_ctx(proving_key_path.clone(), options);

        let setups = Arc::new(SetupsVadcop::new(&pctx.global_info, pctx.options.verify_constraints, pctx.options.aggregation, pctx.options.final_snark));

        pctx.set_weights(&setups.sctx.clone());

        let pctx = Arc::new(pctx);

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN_1);

        Ok((pctx, setups))
    }

    fn initialize_proofman_2(pctx: Arc<ProofCtx<F>>, setups: Arc<SetupsVadcop<F>>) {
        timer_start_info!(INITIALIZING_PROOFMAN_2);

        pctx.dctx_assign_instances();

        print_summary_info(Self::MY_NAME, pctx.clone(), setups.clone());

        Self::initialize_fixed_pols(setups.clone(), pctx.clone());
        pctx.dctx_barrier();
        Self::write_fixed_pols_tree(setups.clone(), pctx.clone());

        pctx.dctx_barrier();
        timer_stop_and_log_info!(INITIALIZING_PROOFMAN_2);
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

    fn initialize_air_instances(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) {
        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        let mut buff_helper_size = 0_usize;

        for instance_id in my_instances.iter() {
            let mut air_instances_w = pctx.air_instance_repo.air_instances.write().unwrap();
            let (airgroup_id, air_id) = instances[*instance_id];
            let setup = sctx.get_setup(airgroup_id, air_id);
            let buff_helper_prover_size = setup.stark_info.get_buff_helper_size();
            if buff_helper_prover_size > buff_helper_size {
                buff_helper_size = buff_helper_prover_size;
            }

            let air_instance = air_instances_w.get_mut(&*instance_id).unwrap();
            air_instance.init_aux_trace(setup.prover_buffer_size as usize);
            air_instance.init_evals(setup.stark_info.ev_map.len() * 3);
            air_instance.init_challenges((setup.stark_info.challenges_map.as_ref().unwrap().len() + setup.stark_info.stark_struct.steps.len() + 1) * 3);
           
            let n_custom_commits = setup.stark_info.custom_commits.len();

            for commit_id in 0..n_custom_commits {
                let n_cols = *setup
                    .stark_info
                    .map_sections_n
                    .get(&(setup.stark_info.custom_commits[commit_id].name.clone() + "0"))
                    .unwrap() as usize;

                if air_instance.custom_commits[commit_id].is_empty() {
                    air_instance.init_custom_commit(commit_id, (1 << setup.stark_info.stark_struct.n_bits) * n_cols);
                }

                let extended_size = (1 << setup.stark_info.stark_struct.n_bits_ext) * n_cols;
                let mt_nodes = (2 * (1 << setup.stark_info.stark_struct.n_bits_ext) - 1) * 4;
                air_instance.init_custom_commit_extended(commit_id, extended_size + mt_nodes);
            }

            let n_airgroup_values = setup.stark_info.airgroupvalues_map.as_ref().unwrap().len();
            let n_air_values = setup.stark_info.airvalues_map.as_ref().unwrap().len();

            if n_air_values > 0 && air_instance.airvalues.is_empty() {
                air_instance.init_airvalues(n_air_values * 3);
            }

            if n_airgroup_values > 0 && air_instance.airgroup_values.is_empty() {
                air_instance.init_airgroup_values(n_airgroup_values * 3);
            }

            air_instance.set_prover_initialized();
        }

        let buff_helper = create_buffer_fast(buff_helper_size);

        *pctx.buff_helper.values.write().unwrap() = buff_helper;
    }

    fn initialize_fixed_pols(setups: Arc<SetupsVadcop<F>>, pctx: Arc<ProofCtx<F>>) {
        info!("{}: Initializing setup fixed pols", Self::MY_NAME);
        timer_start_info!(INITIALIZE_CONST_POLS);

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        let mut airs = Vec::new();
        let mut seen = HashSet::new();

        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id) = instances[*instance_id];
            if seen.insert((airgroup_id, air_id)) {
                airs.push((airgroup_id, air_id));
            }
        }

        airs.iter().for_each(|&(airgroup_id, air_id)| {
            let setup = setups.sctx.get_setup(airgroup_id, air_id);
            setup.load_const_pols();
        });

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
    }

    fn initialize_fixed_tree(setups: Arc<SetupsVadcop<F>>, pctx: Arc<ProofCtx<F>>) {
        info!("{}: Initializing setup fixed tree", Self::MY_NAME);
        timer_start_info!(INITIALIZE_CONST_TREE);

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();

        let mut airs = Vec::new();
        let mut seen = HashSet::new();

        for instance_id in my_instances.iter() {
            let (airgroup_id, air_id) = instances[*instance_id];
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
            let (airgroup_id, air_id) = instances[*instance_id];
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

    pub fn calculate_im_pols(
        stage: u32,
        sctx: Arc<SetupCtx<F>>,
        pctx: Arc<ProofCtx<F>>,
    ) {
        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();
        
        info!("{}: Calculating im pols {}", Self::MY_NAME, stage);
        timer_start_info!(CALCULATING_IM_POLS);
        for instance_id in my_instances.iter() {
            let air_instances = pctx.air_instance_repo.air_instances.read().unwrap();
            let air_instance = air_instances.get(&instance_id).unwrap();
            
            if !air_instance.prover_initialized {
                continue;
            }

            let (airgroup_id, air_id) = instances[*instance_id];
            let setup = sctx.get_setup(airgroup_id, air_id);

            let steps_params = StepsParams {
                trace: air_instance.get_trace_ptr(),
                aux_trace: air_instance.get_aux_trace_ptr(),
                public_inputs: pctx.get_publics_ptr(),
                proof_values: pctx.get_proof_values_ptr(),
                challenges: air_instance.get_challenges_ptr() as *mut u8,
                airgroup_values: air_instance.get_airgroup_values_ptr(),
                airvalues: air_instance.get_airvalues_ptr(),
                evals: air_instance.get_evals_ptr(),
                xdivxsub: std::ptr::null_mut(),
                p_const_pols: setup.get_const_ptr(),
                p_const_tree: setup.get_const_tree_ptr(),
                custom_commits: air_instance.get_custom_commits_ptr(),
                custom_commits_extended: air_instance.get_custom_commits_extended_ptr(),
            };

            calculate_impols_expressions_c((&setup.p_setup).into(),stage as u64, (&steps_params).into());
        }
        timer_stop_and_log_info!(CALCULATING_IM_POLS);
    }

    pub fn get_challenge_air(pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) -> Vec<u64> {
        info!("{}: Committing stage 1", Self::MY_NAME);

        let n_field_elements = 4;

        timer_start_info!(COMMITING_STAGE_1);
        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();
        let air_instances = pctx.air_instance_repo.air_instances.read().unwrap();

        let mut values = vec![0; my_instances.len() * n_field_elements];

        for (idx, instance_id) in my_instances.iter().enumerate() {
            let (airgroup_id, air_id) = instances[*instance_id];
            let setup = sctx.get_setup(airgroup_id, air_id);
            let air_instance = air_instances.get(instance_id).unwrap();

            let root = vec![F::zero(); n_field_elements];
            commit_witness_c(
                setup.stark_info.stark_struct.n_bits,
                setup.stark_info.stark_struct.n_bits_ext,
                *setup.stark_info.map_sections_n.get("cm1").unwrap(),
                root.as_ptr() as *mut u8,
                air_instance.get_trace_ptr(),
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

            let verkey =
                pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic).display().to_string()
                    + ".verkey.json";

            let mut file = File::open(&verkey).expect("Unable to open file");
            let mut json_str = String::new();
            file.read_to_string(&mut json_str).expect("Unable to read file");
            let vk: Vec<u64> = serde_json::from_str(&json_str).expect("REASON");
            for j in 0..n_field_elements {
                values_hash[j] = F::from_canonical_u64(vk[j]);
                values_hash[j + n_field_elements] = root[j];
            }

            let airvalues_map = setup.stark_info.airvalues_map.as_ref().unwrap();
            let mut p = 0;
            let mut count = 0;
            for air_value in airvalues_map {
                if air_value.stage == 1 {
                    values_hash[2 * n_field_elements + count] = air_instance.airvalues[p];
                    count += 1;
                    p += 1;
                }
            }

            calculate_hash_c(value.as_mut_ptr() as *mut u8, values_hash.as_mut_ptr() as *mut u8, size as u64);

            for id in 0..n_field_elements {
                values[idx * n_field_elements + id] = value[id].as_canonical_u64();
            }
        }

        timer_stop_and_log_info!(COMMITING_STAGE_1);
        values
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
