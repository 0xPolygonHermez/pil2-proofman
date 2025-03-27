use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve, goldilocks_quintic_extension::GoldilocksQuinticExtension};
use libloading::{Library, Symbol};
use log::info;
use p3_field::extension::BinomialExtensionField;
use p3_field::BasedVectorSpace;
use std::ops::Add;
use proofman_common::{load_const_pols, load_const_pols_tree, CurveType};
use proofman_common::{
    calculate_fixed_tree, skip_prover_instance, ProofCtx, ProofType, ProofOptions, SetupCtx, SetupsVadcop,
};

use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{gen_device_commit_buffers_c, gen_device_commit_buffers_free_c};
use proofman_starks_lib_c::{save_challenges_c, save_proof_values_c, save_publics_c};
use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write;
use std::io::Read;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use p3_goldilocks::Goldilocks;

use p3_field::PrimeField64;
use proofman_starks_lib_c::{
    gen_proof_c, commit_witness_c, calculate_hash_c, load_custom_commit_c, calculate_impols_expressions_c,
};

use std::{path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::{discover_max_sizes, discover_max_sizes_aggregation};
use crate::{check_paths2, check_tree_paths, check_tree_paths_vadcop};
use crate::verify_basic_proof;
use crate::verify_global_constraints_proof;
use crate::MaxSizes;
use crate::{
    verify_constraints_proof, check_paths, print_summary_info, aggregate_proofs, get_buff_sizes,
    generate_vadcop_recursive1_proof,
};

use std::ffi::c_void;

use proofman_util::{
    create_buffer_fast, timer_start_debug, timer_stop_and_log_debug, timer_start_info, timer_stop_and_log_info,
    DeviceBuffer,
};

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField64> ProofMan<F>
where
    BinomialExtensionField<Goldilocks, 5>: BasedVectorSpace<F>,
{
    const MY_NAME: &'static str = "ProofMan";

    pub fn check_setup(proving_key_path: PathBuf, options: ProofOptions) -> Result<(), Box<dyn std::error::Error>> {
        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        let pctx = ProofCtx::<F>::create_ctx(proving_key_path.clone(), HashMap::new(), options);

        let setups_aggregation = Arc::new(SetupsVadcop::<F>::new(
            &pctx.global_info,
            false,
            pctx.options.aggregation,
            pctx.options.final_snark,
        ));

        let sctx: SetupCtx<F> = SetupCtx::new(&pctx.global_info, &ProofType::Basic, false);

        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                calculate_fixed_tree(sctx.get_setup(airgroup_id, air_id));
            }
        }

        if pctx.options.aggregation {
            let sctx_compressor = setups_aggregation.sctx_compressor.as_ref().unwrap();
            for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                        calculate_fixed_tree(sctx_compressor.get_setup(airgroup_id, air_id));
                    }
                }
            }

            let sctx_recursive1 = setups_aggregation.sctx_recursive1.as_ref().unwrap();
            for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
                for (air_id, _) in air_group.iter().enumerate() {
                    calculate_fixed_tree(sctx_recursive1.get_setup(airgroup_id, air_id));
                }
            }

            let sctx_recursive2 = setups_aggregation.sctx_recursive2.as_ref().unwrap();
            let n_airgroups = pctx.global_info.air_groups.len();
            for airgroup in 0..n_airgroups {
                calculate_fixed_tree(sctx_recursive2.get_setup(airgroup, 0));
            }

            let setup_vadcop_final = setups_aggregation.setup_vadcop_final.as_ref().unwrap();
            calculate_fixed_tree(setup_vadcop_final);

            if pctx.options.final_snark {
                let setup_recursivef = setups_aggregation.setup_recursivef.as_ref().unwrap();
                calculate_fixed_tree(setup_recursivef);
            }
        }

        Ok(())
    }

    pub fn verify_proof_constraints(
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        check_paths(
            &witness_lib_path,
            &public_inputs_path,
            &input_data_path,
            &proving_key_path,
            &output_dir_path,
            options.verify_constraints,
        )?;

        let (pctx, sctx) = Self::initialize_proofman(proving_key_path, custom_commits_fixed, options)?;

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), public_inputs_path, input_data_path));

        Self::init_witness_lib(witness_lib_path, &pctx, wcm.clone())?;

        Self::_verify_proof_constraints(pctx, sctx, wcm)
    }

    pub fn verify_proof_constraints_from_lib(
        mut witness_lib: Box<dyn WitnessLibrary<F>>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        check_paths2(&proving_key_path, &output_dir_path, options.verify_constraints)?;

        let (pctx, sctx) = Self::initialize_proofman(proving_key_path, custom_commits_fixed, options)?;

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), None, None));

        witness_lib.register_witness(wcm.clone());

        Self::_verify_proof_constraints(pctx, sctx, wcm)
    }

    fn _verify_proof_constraints(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        wcm: Arc<WitnessManager<F>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(EXECUTE);
        wcm.execute();
        timer_stop_and_log_info!(EXECUTE);

        pctx.dctx_assign_instances();
        pctx.dctx_close();

        print_summary_info(Self::MY_NAME, &pctx, &sctx);

        let transcript = FFITranscript::new(2, true);
        let dummy_element = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let global_challenge = [F::ZERO; 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);
        pctx.set_global_challenge(2, &global_challenge);
        transcript.add_elements(dummy_element.as_ptr() as *mut u8, 4);

        let instances = pctx.dctx_get_instances();
        let airgroup_values_air_instances = Arc::new(Mutex::new(Vec::new()));
        let valid_constraints = Arc::new(AtomicBool::new(true));
        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;

        for (instance_id, (airgroup_id, air_id, all)) in instances.iter().enumerate() {
            let is_my_instance = pctx.dctx_is_my_instance(instance_id);
            let (skip, _) = skip_prover_instance(&pctx, instance_id);

            if skip || (!all && !is_my_instance) {
                continue;
            };

            wcm.calculate_witness(1, &[instance_id]);

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            thread_handle = is_my_instance.then(|| {
                Self::verify_proof_constraints_stage(
                    pctx.clone(),
                    sctx.clone(),
                    wcm.clone(),
                    valid_constraints.clone(),
                    airgroup_values_air_instances.clone(),
                    instance_id,
                    *airgroup_id,
                    *air_id,
                    pctx.dctx_find_air_instance_id(instance_id),
                )
            });
        }

        if let Some(handle) = thread_handle {
            handle.join().unwrap()
        }

        wcm.end();

        let check_global_constraints = pctx.options.debug_info.debug_instances.is_empty()
            || !pctx.options.debug_info.debug_global_instances.is_empty();

        if check_global_constraints {
            let airgroup_values_air_instances =
                Arc::try_unwrap(airgroup_values_air_instances).unwrap().into_inner().unwrap();
            let airgroupvalues_u64 = aggregate_airgroupvals(&pctx, &airgroup_values_air_instances);
            let airgroupvalues = pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

            if pctx.dctx_get_rank() == 0 {
                let valid_global_constraints = verify_global_constraints_proof(&pctx, &sctx, airgroupvalues);

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
    fn verify_proof_constraints_stage(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        wcm: Arc<WitnessManager<F>>,
        valid_constraints: Arc<AtomicBool>,
        airgroup_values_air_instances: Arc<Mutex<Vec<Vec<F>>>>,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            Self::initialize_air_instance(&pctx, &sctx, instance_id, true);

            #[cfg(feature = "diagnostic")]
            {
                let invalid_initialization = Self::diagnostic_instance(&pctx, &sctx, instance_id);
                if invalid_initialization {
                    panic!("Invalid initialization");
                    // return Some(Err("Invalid initialization".into()));
                }
            }

            wcm.calculate_witness(2, &[instance_id]);
            Self::calculate_im_pols(2, &sctx, &pctx, instance_id);

            wcm.debug(&[instance_id]);

            if pctx.options.verify_constraints {
                let valid = verify_constraints_proof(&pctx, &sctx, instance_id);
                valid_constraints.fetch_and(valid, Ordering::Relaxed);
            }

            airgroup_values_air_instances.lock().unwrap().push(pctx.get_air_instance_airgroup_values(
                airgroup_id,
                air_id,
                air_instance_id,
            ));
            pctx.free_instance(instance_id);
        })
    }

    pub fn generate_proof(
        witness_lib_path: PathBuf,
        public_inputs_path: Option<PathBuf>,
        input_data_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        check_paths(
            &witness_lib_path,
            &public_inputs_path,
            &input_data_path,
            &proving_key_path,
            &output_dir_path,
            options.verify_constraints,
        )?;

        let (pctx, sctx) = Self::initialize_proofman(proving_key_path, custom_commits_fixed, options)?;

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), public_inputs_path, input_data_path));

        Self::init_witness_lib(witness_lib_path, &pctx, wcm.clone())?;

        Self::_generate_proof(output_dir_path, pctx, sctx, wcm)
    }

    pub fn generate_proof_from_lib(
        mut witness_lib: Box<dyn WitnessLibrary<F>>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        check_paths2(&proving_key_path, &output_dir_path, options.verify_constraints)?;

        let (pctx, sctx) = Self::initialize_proofman(proving_key_path, custom_commits_fixed, options)?;

        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), None, None));

        witness_lib.register_witness(wcm.clone());

        Self::_generate_proof(output_dir_path, pctx, sctx, wcm)
    }

    fn _generate_proof(
        output_dir_path: PathBuf,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        wcm: Arc<WitnessManager<F>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        timer_start_info!(GENERATING_VADCOP_PROOF);

        timer_start_info!(GENERATING_PROOFS);

        let setups_handle = {
            let pctx_clone = Arc::clone(&pctx);

            std::thread::spawn(move || {
                Arc::new(SetupsVadcop::new(
                    &pctx_clone.global_info,
                    pctx_clone.options.verify_constraints,
                    pctx_clone.options.aggregation,
                    pctx_clone.options.final_snark,
                ))
            })
        };

        timer_start_info!(EXECUTE);
        wcm.execute();
        timer_stop_and_log_info!(EXECUTE);

        pctx.dctx_assign_instances();
        pctx.dctx_close();

        print_summary_info(Self::MY_NAME, &pctx, &sctx);

        timer_start_info!(CALCULATING_CONTRIBUTIONS);

        let instances = pctx.dctx_get_instances();
        let my_instances = pctx.dctx_get_my_instances();
        let my_air_groups = pctx.dctx_get_my_air_groups();

        let values = Arc::new(Mutex::new(vec![0; my_instances.len() * 10]));

        let mut prover_buffer_size = 0;
        for (airgroup_id, air_group) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                if setup.prover_buffer_size > prover_buffer_size {
                    prover_buffer_size = setup.prover_buffer_size;
                }
            }
        }

        let aux_trace = Arc::new(create_buffer_fast(prover_buffer_size as usize));

        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;

        let setups = setups_handle.join().expect("Setups thread panicked");
        if pctx.options.aggregation {
            check_tree_paths_vadcop(&pctx, &setups)?;
        }

        let max_sizes = discover_max_sizes(&pctx, &sctx);
        let max_sizes_ptr = &max_sizes as *const MaxSizes as *mut c_void;
        let d_buffers = Arc::new(Mutex::new(DeviceBuffer(gen_device_commit_buffers_c(max_sizes_ptr))));

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if !all && !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            wcm.calculate_witness(1, &[instance_id]);

            if !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            // Join the previous thread (if any) before starting a new one
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }

            thread_handle = Some(Self::get_contribution(
                instance_id,
                pctx.clone(),
                sctx.clone(),
                aux_trace.clone(),
                *all,
                values.clone(),
                d_buffers.clone(),
            ));
        }

        if let Some(handle) = thread_handle {
            handle.join().unwrap()
        }

        let values_challenge = Arc::try_unwrap(values).unwrap().into_inner().unwrap();

        timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

        Self::calculate_global_challenge(&pctx, values_challenge);

        timer_start_info!(GENERATING_BASIC_PROOFS);

        let const_tree = Arc::new(create_buffer_fast(sctx.max_const_tree_size));
        let const_pols = Arc::new(create_buffer_fast(sctx.max_const_size));

        let proofs = Arc::new(Mutex::new(vec![Vec::new(); my_instances.len()]));
        let airgroup_values_air_instances = vec![Vec::new(); my_instances.len()];
        let airgroup_values_air_instances = Arc::new(Mutex::new(airgroup_values_air_instances));

        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;

        for air_groups in my_air_groups.iter() {
            let mut gen_const_tree = true;
            for my_instance_id in air_groups.iter() {
                let instance_id = my_instances[*my_instance_id];
                let (airgroup_id, air_id, all) = instances[instance_id];

                if !all {
                    wcm.calculate_witness(1, &[instance_id]);
                }

                // Join the previous thread (if any) before starting a new one
                if let Some(handle) = thread_handle.take() {
                    handle.join().unwrap();
                }

                thread_handle = Some(Self::generate_proof_thread(
                    proofs.clone(),
                    pctx.clone(),
                    sctx.clone(),
                    instance_id,
                    airgroup_id,
                    air_id,
                    output_dir_path.clone(),
                    aux_trace.clone(),
                    const_pols.clone(),
                    const_tree.clone(),
                    airgroup_values_air_instances.clone(),
                    gen_const_tree,
                    d_buffers.clone(),
                ));
                gen_const_tree = false;
            }
        }

        if let Some(handle) = thread_handle {
            handle.join().unwrap();
        }

        timer_stop_and_log_info!(GENERATING_BASIC_PROOFS);

        timer_stop_and_log_info!(GENERATING_PROOFS);

        let global_info_path = pctx.global_info.get_proving_key_path().join("pilout.globalInfo.json");
        let global_info_file = global_info_path.to_str().unwrap();

        save_challenges_c(pctx.get_challenges_ptr(), global_info_file, output_dir_path.to_string_lossy().as_ref());
        save_proof_values_c(pctx.get_proof_values_ptr(), global_info_file, output_dir_path.to_string_lossy().as_ref());
        save_publics_c(
            pctx.global_info.n_publics as u64,
            pctx.get_publics_ptr(),
            output_dir_path.to_string_lossy().as_ref(),
        );

        let airgroup_values_air_instances =
            Arc::try_unwrap(airgroup_values_air_instances).unwrap().into_inner().unwrap();

        if !pctx.options.aggregation {
            let mut valid_proofs = true;

            if pctx.options.verify_proofs {
                timer_start_info!(VERIFYING_PROOFS);
                let proofs_ = Arc::try_unwrap(proofs).unwrap().into_inner().unwrap();
                for instance_id in my_instances.iter() {
                    valid_proofs =
                        verify_basic_proof(&pctx, *instance_id, &proofs_[pctx.dctx_get_instance_idx(*instance_id)]);
                }
                timer_stop_and_log_info!(VERIFYING_PROOFS);
            }

            let check_global_constraints = pctx.options.debug_info.debug_instances.is_empty()
                || !pctx.options.debug_info.debug_global_instances.is_empty();

            if check_global_constraints {
                let airgroupvalues_u64 = aggregate_airgroupvals(&pctx, &airgroup_values_air_instances);
                let airgroupvalues = pctx.dctx_distribute_airgroupvalues(airgroupvalues_u64);

                if pctx.dctx_get_rank() == 0 {
                    let valid_global_constraints = verify_global_constraints_proof(&pctx, &sctx, airgroupvalues);
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

        // std::thread::spawn({
        //     move || {
        //         drop(aux_trace);
        //         drop(wcm);
        //         drop(sctx);
        //     }
        // });

        gen_device_commit_buffers_free_c(d_buffers.lock().unwrap().get_ptr());

        let max_sizes_aggregation = discover_max_sizes_aggregation(&pctx, &setups);
        let max_sizes_aggregation_ptr = &max_sizes_aggregation as *const MaxSizes as *mut c_void;
        let d_buffers_aggregation =
            Arc::new(Mutex::new(DeviceBuffer(gen_device_commit_buffers_c(max_sizes_aggregation_ptr))));

        let (circom_witness, publics, trace, prover_buffer) = if pctx.options.aggregation {
            let (circom_witness_size, publics_size, trace_size, prover_buffer_size) = get_buff_sizes(&pctx, &setups)?;
            let circom_witness = Arc::new(create_buffer_fast(circom_witness_size));
            let publics = Arc::new(create_buffer_fast(publics_size));
            let trace = Arc::new(create_buffer_fast(trace_size));
            let prover_buffer = Arc::new(create_buffer_fast(prover_buffer_size));
            (circom_witness, publics, trace, prover_buffer)
        } else {
            (Arc::new(Vec::new()), Arc::new(Vec::new()), Arc::new(Vec::new()), Arc::new(Vec::new()))
        };

        let proofs = Arc::try_unwrap(proofs).unwrap().into_inner().unwrap();
        timer_start_info!(GENERATING_COMPRESSOR_AND_RECURSIVE1_PROOF);
        let const_tree_aggregation: Vec<F> =
            create_buffer_fast(setups.sctx_recursive2.as_ref().unwrap().max_const_tree_size);
        let const_pols_aggregation: Vec<F> =
            create_buffer_fast(setups.sctx_recursive2.as_ref().unwrap().max_const_size);

        let const_tree_recursive2: Vec<F> =
            create_buffer_fast(setups.sctx_recursive2.as_ref().unwrap().max_const_tree_size);
        let const_pols_recursive2: Vec<F> = create_buffer_fast(setups.sctx_recursive2.as_ref().unwrap().max_const_size);

        let mut current_airgroup_id = instances[my_instances[my_air_groups[0][0]]].0;
        let setup_recursive2 = setups.sctx_recursive2.as_ref().unwrap().get_setup(current_airgroup_id, 0);
        load_const_pols(&setup_recursive2.setup_path, setup_recursive2.const_pols_size, &const_pols_recursive2);
        load_const_pols_tree(setup_recursive2, &const_tree_recursive2);

        let mut recursive2_proofs = vec![Vec::new(); pctx.global_info.air_groups.len()];
        #[allow(clippy::needless_range_loop)]
        for airgroup_id in 0..pctx.global_info.air_groups.len() {
            let setup = setups.sctx_recursive2.as_ref().unwrap().get_setup(airgroup_id, 0);
            let publics_aggregation = 1 + 4 * pctx.global_info.agg_types[airgroup_id].len() + 10;
            recursive2_proofs[airgroup_id] = create_buffer_fast(setup.proof_size as usize + publics_aggregation);
        }
        let mut recursive2_initialized = vec![false; pctx.global_info.air_groups.len()];
        for air_groups in my_air_groups.iter() {
            let (airgroup_id, air_id, _) = instances[my_instances[air_groups[0]]];
            if airgroup_id != current_airgroup_id {
                current_airgroup_id = airgroup_id;
                let setup_recursive2 = setups.sctx_recursive2.as_ref().unwrap().get_setup(current_airgroup_id, air_id);
                load_const_pols(&setup_recursive2.setup_path, setup_recursive2.const_pols_size, &const_pols_recursive2);
                load_const_pols_tree(setup_recursive2, &const_tree_recursive2);
            }
            let const_pols_compressor;
            let const_tree_compressor;
            let const_pols_recursive1;
            let const_tree_recursive1;
            let has_compressor = pctx.global_info.get_air_has_compressor(airgroup_id, air_id);
            if has_compressor {
                let setup = setups.sctx_compressor.as_ref().unwrap().get_setup(airgroup_id, air_id);
                load_const_pols(&setup.setup_path, setup.const_pols_size, &const_pols);
                load_const_pols_tree(setup, &const_tree);
                const_tree_compressor = &const_tree;
                const_pols_compressor = &const_pols;
                let setup_recursive1 = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);

                load_const_pols(
                    &setup_recursive1.setup_path,
                    setup_recursive1.const_pols_size,
                    &const_pols_aggregation,
                );
                load_const_pols_tree(setup_recursive1, &const_tree_aggregation);
                const_tree_recursive1 = &const_tree_aggregation;
                const_pols_recursive1 = &const_pols_aggregation;
            } else {
                let setup = setups.sctx_recursive1.as_ref().unwrap().get_setup(airgroup_id, air_id);
                load_const_pols(&setup.setup_path, setup.const_pols_size, &const_pols);
                load_const_pols_tree(setup, &const_tree);
                const_tree_compressor = &const_tree;
                const_tree_recursive1 = &const_tree;
                const_pols_compressor = &const_pols;
                const_pols_recursive1 = &const_pols;
            }
            for my_instance_id in air_groups.iter() {
                let instance_id = my_instances[*my_instance_id];

                generate_vadcop_recursive1_proof(
                    &pctx,
                    &setups,
                    instance_id,
                    &proofs[*my_instance_id],
                    &circom_witness,
                    &publics,
                    &trace,
                    &prover_buffer,
                    const_pols_compressor,
                    const_pols_recursive1,
                    &const_pols_recursive2,
                    const_tree_compressor,
                    const_tree_recursive1,
                    &const_tree_recursive2,
                    &mut recursive2_proofs[airgroup_id],
                    recursive2_initialized[airgroup_id],
                    output_dir_path.clone(),
                    d_buffers_aggregation.lock().unwrap().get_ptr(),
                )
                .expect("Failed to generate recursive proof");

                recursive2_initialized[airgroup_id] = true;
            }
        }
        timer_stop_and_log_info!(GENERATING_COMPRESSOR_AND_RECURSIVE1_PROOF);
        let agg_proof = aggregate_proofs(
            Self::MY_NAME,
            &pctx,
            &setups,
            &recursive2_proofs,
            &circom_witness,
            &publics,
            &trace,
            &prover_buffer,
            &const_pols,
            &const_tree,
            output_dir_path.clone(),
            d_buffers_aggregation.lock().unwrap().get_ptr(),
        );
        timer_stop_and_log_info!(GENERATING_VADCOP_PROOF);

        agg_proof
    }

    fn get_contribution(
        instance_id: usize,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        aux_trace_contribution_ptr: Arc<Vec<F>>,
        all: bool,
        values: Arc<Mutex<Vec<u64>>>,
        d_buffers: Arc<Mutex<DeviceBuffer>>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let ptr = aux_trace_contribution_ptr.as_ptr() as *mut u8;
            let value = Self::get_contribution_air(&pctx, &sctx, instance_id, ptr, d_buffers.clone());

            if !all {
                pctx.free_instance(instance_id);
            }

            for (id, value) in value.iter().enumerate().take(10) {
                values.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id) * 10 + id] = *value;
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_proof_thread(
        proofs: Arc<Mutex<Vec<Vec<u64>>>>,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        output_dir_path: PathBuf,
        aux_trace: Arc<Vec<F>>,
        const_pols: Arc<Vec<F>>,
        const_tree: Arc<Vec<F>>,
        airgroup_values_air_instances: Arc<Mutex<Vec<Vec<F>>>>,
        gen_const_tree: bool,
        d_buffers: Arc<Mutex<DeviceBuffer>>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            Self::initialize_air_instance(&pctx, &sctx, instance_id, false);

            let setup = sctx.get_setup(airgroup_id, air_id);
            let p_setup: *mut c_void = (&setup.p_setup).into();
            let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
            let air_instance_name = &pctx.global_info.airs[airgroup_id][air_id].name;
            timer_start_info!(GEN_PROOF);

            if gen_const_tree {
                timer_start_debug!(GENERATING_CONST_TREE);
                load_const_pols(&setup.setup_path, setup.const_pols_size, &const_pols);
                load_const_pols_tree(setup, &const_tree);
                timer_stop_and_log_debug!(GENERATING_CONST_TREE);
            }

            let mut steps_params = pctx.get_air_instance_params(&sctx, instance_id, true);
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;
            steps_params.p_const_pols = const_pols.as_ptr() as *mut u8;
            steps_params.p_const_tree = const_tree.as_ptr() as *mut u8;

            let p_steps_params: *mut u8 = (&steps_params).into();

            let output_file_path = output_dir_path.join(format!("proofs/{}_{}.json", air_instance_name, instance_id));

            let proof_file = match pctx.options.debug_info.save_proofs_to_file {
                true => output_file_path.to_string_lossy().into_owned(),
                false => String::from(""),
            };

            let mut proof: Vec<u64> = create_buffer_fast(setup.proof_size as usize);

            gen_proof_c(
                p_setup,
                p_steps_params,
                pctx.get_global_challenge_ptr(),
                proof.as_mut_ptr(),
                &proof_file,
                airgroup_id as u64,
                air_id as u64,
                air_instance_id as u64,
                d_buffers.lock().unwrap().get_ptr(),
            );

            airgroup_values_air_instances.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id)] =
                pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id);

            pctx.free_instance(instance_id);

            timer_stop_and_log_info!(GEN_PROOF);
            proofs.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id)] = proof;
        })
    }

    #[allow(clippy::type_complexity)]
    fn initialize_proofman(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<(Arc<ProofCtx<F>>, Arc<SetupCtx<F>>), Box<dyn std::error::Error>> {
        timer_start_info!(INITIALIZING_PROOFMAN);

        let mut pctx = ProofCtx::create_ctx(proving_key_path.clone(), custom_commits_fixed, options.clone());
        let sctx: Arc<SetupCtx<F>> =
            Arc::new(SetupCtx::new(&pctx.global_info, &ProofType::Basic, options.verify_constraints));
        pctx.set_weights(&sctx);

        let pctx = Arc::new(pctx);
        if !pctx.options.verify_constraints {
            check_tree_paths(&pctx, &sctx)?;
        }

        Self::initialize_publics(&sctx, &pctx)?;

        timer_stop_and_log_info!(INITIALIZING_PROOFMAN);

        Ok((pctx, sctx))
    }

    fn init_witness_lib(
        witness_lib_path: PathBuf,
        pctx: &ProofCtx<F>,
        wcm: Arc<WitnessManager<F>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing witness");

        // Load the witness computation dynamic library
        timer_start_info!(REGISTER_WITNESS);
        let library = unsafe { Library::new(&witness_lib_path)? };
        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };
        let mut witness_lib = witness_lib(pctx.options.verbose_mode)?;

        witness_lib.register_witness(wcm);

        timer_stop_and_log_info!(REGISTER_WITNESS);

        Ok(())
    }

    fn calculate_global_challenge(pctx: &ProofCtx<F>, values: Vec<u64>) {
        timer_start_info!(CALCULATE_GLOBAL_CHALLENGE);
        let transcript = FFITranscript::new(2, true);

        transcript.add_elements(pctx.get_publics_ptr(), pctx.global_info.n_publics);

        let proof_values_stage = pctx.get_proof_values_by_stage(1);
        if !proof_values_stage.is_empty() {
            transcript.add_elements(proof_values_stage.as_ptr() as *mut u8, proof_values_stage.len());
        }

        let all_roots = pctx.dctx_distribute_roots(values);

        // Add challenges to transcript in order
        for group_idxs in pctx.dctx_get_my_groups() {
            let mut values = Vec::new();
            for idx in group_idxs.iter() {
                let value = vec![
                    F::from_u64(all_roots[*idx]),
                    F::from_u64(all_roots[*idx + 1]),
                    F::from_u64(all_roots[*idx + 2]),
                    F::from_u64(all_roots[*idx + 3]),
                    F::from_u64(all_roots[*idx + 4]),
                    F::from_u64(all_roots[*idx + 5]),
                    F::from_u64(all_roots[*idx + 6]),
                    F::from_u64(all_roots[*idx + 7]),
                    F::from_u64(all_roots[*idx + 8]),
                    F::from_u64(all_roots[*idx + 9]),
                ];
                values.push(value);
            }
            if !values.is_empty() {
                let value = Self::add_contributions(&pctx.global_info.curve, &values);
                transcript.add_elements(value.as_ptr() as *mut u8, value.len());
            }
        }

        let global_challenge = [F::ZERO; 3];
        transcript.get_challenge(&global_challenge[0] as *const F as *mut c_void);

        pctx.set_global_challenge(2, &global_challenge);

        timer_stop_and_log_info!(CALCULATE_GLOBAL_CHALLENGE);
    }

    #[allow(dead_code)]
    fn diagnostic_instance(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, instance_id: usize) -> bool {
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

    fn initialize_air_instance(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, instance_id: usize, init_aux_trace: bool) {
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

    fn initialize_publics(sctx: &SetupCtx<F>, pctx: &ProofCtx<F>) -> Result<(), Box<dyn std::error::Error>> {
        info!("{}: Initializing publics custom_commits", Self::MY_NAME);
        for (airgroup_id, airs) in pctx.global_info.airs.iter().enumerate() {
            for (air_id, _) in airs.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                for custom_commit in &setup.stark_info.custom_commits {
                    if custom_commit.stage_widths[0] > 0 {
                        // Handle the possibility that this returns None
                        let custom_file_path = pctx.get_custom_commits_fixed_buffer(&custom_commit.name)?;

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

        Ok(())
    }

    pub fn calculate_im_pols(stage: u32, sctx: &SetupCtx<F>, pctx: &ProofCtx<F>, instance_id: usize) {
        let instances = pctx.dctx_get_instances();
        let (airgroup_id, air_id, _) = instances[instance_id];
        let setup = sctx.get_setup(airgroup_id, air_id);

        let steps_params = pctx.get_air_instance_params(sctx, instance_id, false);

        calculate_impols_expressions_c((&setup.p_setup).into(), stage as u64, (&steps_params).into());
    }

    pub fn get_contribution_air(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
        d_buffers: Arc<Mutex<DeviceBuffer>>,
    ) -> Vec<u64> {
        let n_field_elements = 4;

        timer_start_info!(GET_CONTRIBUTION_AIR);
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, _) = instances[instance_id];
        let air_instance_id = pctx.dctx_find_air_instance_id(instance_id);
        let setup = sctx.get_setup(airgroup_id, air_id);

        let root = vec![F::ZERO; n_field_elements];
        commit_witness_c(
            3,
            setup.stark_info.stark_struct.n_bits,
            setup.stark_info.stark_struct.n_bits_ext,
            *setup.stark_info.map_sections_n.get("cm1").unwrap(),
            root.as_ptr() as *mut u8,
            pctx.get_air_instance_trace_ptr(instance_id),
            aux_trace_contribution_ptr,
            d_buffers.lock().unwrap().get_ptr(),
        );

        let mut value = vec![F::ZERO; 10];

        let n_airvalues = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().filter(|entry| entry.stage == 1).count())
            .unwrap_or(0);

        let size = 2 * n_field_elements + n_airvalues;

        let mut values_hash = vec![F::ZERO; size];

        let verkey = pctx.global_info.get_air_setup_path(airgroup_id, air_id, &ProofType::Basic).display().to_string()
            + ".verkey.json";

        let mut file = File::open(&verkey).expect("Unable to open file");
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).expect("Unable to read file");
        let vk: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");
        for j in 0..n_field_elements {
            values_hash[j] = F::from_u64(vk[j]);
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

        calculate_hash_c(value.as_mut_ptr() as *mut u8, values_hash.as_mut_ptr() as *mut u8, size as u64, 10);

        timer_stop_and_log_info!(GET_CONTRIBUTION_AIR);

        value.iter().map(|x| x.as_canonical_u64()).collect::<Vec<u64>>()
    }

    fn add_contributions(curve_type: &CurveType, values: &[Vec<F>]) -> Vec<F> {
        if *curve_type == CurveType::EcGFp5 {
            let mut result = EcGFp5::hash_to_curve(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcGFp5::hash_to_curve(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );

                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        } else {
            let mut result = EcMasFp5::hash_to_curve(
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][0..5]),
                GoldilocksQuinticExtension::from_basis_coefficients_slice(&values[0][5..10]),
            );

            for value in values.iter().skip(1) {
                let curve_point = EcMasFp5::hash_to_curve(
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[0..5]),
                    GoldilocksQuinticExtension::from_basis_coefficients_slice(&value[5..10]),
                );
                result = result.add(&curve_point);
            }

            let mut curve_point_values = vec![F::ZERO; 10];
            curve_point_values[0..5].copy_from_slice(result.x().as_basis_coefficients_slice());
            curve_point_values[5..10].copy_from_slice(result.y().as_basis_coefficients_slice());
            curve_point_values
        }
    }
}
