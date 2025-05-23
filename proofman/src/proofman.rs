use curves::{EcGFp5, EcMasFp5, curve::EllipticCurve, goldilocks_quintic_extension::GoldilocksQuinticExtension};
use libloading::{Library, Symbol};
use log::info;
use p3_field::extension::BinomialExtensionField;
use p3_field::BasedVectorSpace;
use std::ops::Add;
use proofman_common::{load_const_pols, load_const_pols_tree, CurveType};
use proofman_common::{
    calculate_fixed_tree, skip_prover_instance, Proof, ProofCtx, ProofType, ProofOptions, SetupCtx, SetupsVadcop,
};
use colored::Colorize;
use proofman_hints::aggregate_airgroupvals;
use proofman_starks_lib_c::{gen_device_commit_buffers_c, gen_device_commit_buffers_free_c};
use proofman_starks_lib_c::{save_challenges_c, save_proof_values_c, save_publics_c, get_const_offset_c};
use std::collections::HashMap;
use std::fs::File;
use std::fmt::Write;
use std::io::Read;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, RwLock};
use std::sync::mpsc::channel;
use std::sync::atomic::AtomicU64;

use p3_goldilocks::Goldilocks;

use p3_field::PrimeField64;
use proofman_starks_lib_c::{
    gen_proof_c, commit_witness_c, calculate_hash_c, load_custom_commit_c, calculate_impols_expressions_c,
};

use std::{path::PathBuf, sync::Arc};

use transcript::FFITranscript;

use witness::{WitnessLibInitFn, WitnessLibrary, WitnessManager};
use crate::{discover_max_sizes, discover_max_sizes_aggregation};
use crate::{check_paths2, check_tree_paths, check_tree_paths_vadcop, initialize_fixed_pols_tree};
use crate::{verify_basic_proof, verify_proof, verify_global_constraints_proof};
use crate::MaxSizes;
use crate::{verify_constraints_proof, check_paths, print_summary_info, get_recursive_buffer_sizes};
use crate::{
    gen_witness_recursive, gen_witness_aggregation, generate_recursive_proof, generate_vadcop_final_proof,
    generate_fflonk_snark_proof, generate_recursivef_proof, total_recursive_proofs, initialize_size_witness,
};
use crate::aggregate_recursive2_proofs;

use std::ffi::c_void;

use proofman_util::{create_buffer_fast, timer_start_info, timer_stop_and_log_info, DeviceBuffer};

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
        witness_lib: &mut dyn WitnessLibrary<F>,
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
                if !valid {
                    valid_constraints.fetch_and(valid, Ordering::Relaxed);
                }
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
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
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
        witness_lib: &mut dyn WitnessLibrary<F>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        check_paths2(&proving_key_path, &output_dir_path, options.verify_constraints)?;

        let (pctx, sctx) = Self::initialize_proofman(proving_key_path, custom_commits_fixed, options)?;

        timer_start_info!(REGISTERING_WITNESS);
        let wcm = Arc::new(WitnessManager::new(pctx.clone(), sctx.clone(), None, None));
        witness_lib.register_witness(wcm.clone());
        timer_stop_and_log_info!(REGISTERING_WITNESS);

        Self::_generate_proof(output_dir_path, pctx, sctx, wcm)
    }

    fn _generate_proof(
        output_dir_path: PathBuf,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        wcm: Arc<WitnessManager<F>>,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        timer_start_info!(GENERATING_VADCOP_PROOF);

        timer_start_info!(GENERATING_PROOFS);

        let pctx_clone = pctx.clone();
        let setup_aggregation_handle = std::thread::spawn(move || {
            SetupsVadcop::new(
                &pctx_clone.global_info,
                pctx_clone.options.verify_constraints,
                pctx_clone.options.aggregation,
                pctx_clone.options.final_snark,
            )
        });

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

        let aux_trace_size = match cfg!(feature = "gpu") {
            true => sctx.max_const_tree_size + sctx.max_const_size,
            false => prover_buffer_size as usize,
        };
        let aux_trace = Arc::new(create_buffer_fast(aux_trace_size));

        let max_sizes = discover_max_sizes(&pctx, &sctx);
        let max_sizes_ptr = &max_sizes as *const MaxSizes as *mut c_void;
        let d_buffers = Arc::new(Mutex::new(DeviceBuffer(gen_device_commit_buffers_c(max_sizes_ptr))));

        let (tx, rx) = channel::<usize>();
        let max_pending_proofs = match cfg!(feature = "gpu") {
            true => 3,
            false => 1,
        };

        let proof_count = Arc::new(AtomicU64::new(0)); // shared between threads

        let proof_count_clone = Arc::clone(&proof_count);
        let pctx_clone = pctx.clone();
        let sctx_clone = sctx.clone();
        let aux_trace_clone = aux_trace.clone();
        let values_clone = values.clone();
        let d_buffers_clone = d_buffers.clone();

        let proof_thread = std::thread::spawn(move || {
            while let Ok(instance_id) = rx.recv() {
                let handle = Self::get_contribution(
                    instance_id,
                    pctx_clone.clone(),
                    sctx_clone.clone(),
                    aux_trace_clone.clone(),
                    values_clone.clone(),
                    d_buffers_clone.clone(),
                );

                handle.join().unwrap();
                proof_count_clone.fetch_sub(1, Ordering::SeqCst); // mark one as done
            }
        });

        for (instance_id, (_, _, all)) in instances.iter().enumerate() {
            if !all && !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            while proof_count.load(Ordering::SeqCst) >= max_pending_proofs {
                std::hint::spin_loop();
            }

            wcm.calculate_witness(1, &[instance_id]);

            if !pctx.dctx_is_my_instance(instance_id) {
                continue;
            }

            tx.send(instance_id).unwrap();
            proof_count.fetch_add(1, Ordering::SeqCst);
        }

        drop(tx);
        proof_thread.join().unwrap();

        let values_challenge = Arc::try_unwrap(values).unwrap().into_inner().unwrap();

        timer_stop_and_log_info!(CALCULATING_CONTRIBUTIONS);

        Self::calculate_global_challenge(&pctx, values_challenge);

        let setups = Arc::new(setup_aggregation_handle.join().unwrap());

        let mut init_const_tree_handle = if pctx.options.aggregation {
            check_tree_paths_vadcop(&pctx, &setups)?;

            let setups_clone = setups.clone();
            let pctx_clone = pctx.clone();
            Some(std::thread::spawn(move || {
                initialize_fixed_pols_tree(&pctx_clone.clone(), &setups_clone);
            }))
        } else {
            None
        };

        if pctx.options.aggregation {
            initialize_size_witness(&pctx, &setups)?;
        }

        timer_start_info!(GENERATING_BASIC_PROOFS);

        let proofs = Arc::new(RwLock::new(vec![Proof::default(); my_instances.len()]));
        let airgroup_values_air_instances = Arc::new(Mutex::new(vec![Vec::new(); my_instances.len()]));

        let mut thread_handle: Option<std::thread::JoinHandle<()>> = None;

        let mut recursive_witness = vec![None; my_instances.len()];
        let mut previous_instance_id = None;
        for idx in 0..my_air_groups.len() {
            let idx = (pctx.dctx_get_rank() + idx) % my_air_groups.len();
            let air_groups = &my_air_groups[idx];
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
                    airgroup_values_air_instances.clone(),
                    gen_const_tree,
                    d_buffers.clone(),
                ));

                if previous_instance_id.is_some() {
                    if pctx.options.aggregation {
                        let proof =
                            proofs.read().unwrap()[pctx.dctx_get_instance_idx(previous_instance_id.unwrap())].clone();
                        let witness = gen_witness_recursive(&pctx, &setups, &proof)?;
                        recursive_witness[pctx.dctx_get_instance_idx(previous_instance_id.unwrap())] = Some(witness);
                    }
                    let pctx_clone = pctx.clone();
                    let instance_id = previous_instance_id.unwrap();
                    std::thread::spawn(move || {
                        pctx_clone.free_instance(instance_id);
                    });
                }

                previous_instance_id = Some(instance_id);
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
                    let valid_proof = verify_basic_proof(
                        &pctx,
                        *instance_id,
                        &proofs_[pctx.dctx_get_instance_idx(*instance_id)].proof,
                    );
                    if !valid_proof {
                        valid_proofs = false;
                    }
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
                log::info!(
                    "{}: ··· {}",
                    Self::MY_NAME,
                    "\u{2713} All proofs were successfully verified".bright_green().bold()
                );
                return Ok(None);
            } else {
                return Err("Basic proofs were not verified".into());
            }
        }

        std::thread::spawn({
            move || {
                drop(aux_trace);
                drop(wcm);
                drop(sctx);
            }
        });

        gen_device_commit_buffers_free_c(d_buffers.lock().unwrap().get_ptr());

        let max_sizes_aggregation = discover_max_sizes_aggregation(&pctx, &setups);
        let max_sizes_aggregation_ptr = &max_sizes_aggregation as *const MaxSizes as *mut c_void;
        let d_buffers_aggregation =
            Arc::new(Mutex::new(DeviceBuffer(gen_device_commit_buffers_c(max_sizes_aggregation_ptr))));

        let (trace, prover_buffer) = if pctx.options.aggregation {
            let (trace_size, prover_buffer_size) = get_recursive_buffer_sizes(&pctx, &setups)?;
            let trace = Arc::new(create_buffer_fast::<F>(trace_size));
            let prover_buffer = Arc::new(create_buffer_fast::<F>(prover_buffer_size));
            (trace, prover_buffer)
        } else {
            (Arc::new(Vec::new()), Arc::new(Vec::new()))
        };

        let proofs = Arc::try_unwrap(proofs).unwrap().into_inner().unwrap();

        if let Some(handle) = init_const_tree_handle.take() {
            handle.join().unwrap();
        }

        timer_start_info!(GENERATING_COMPRESSED_PROOFS);
        let mut recursive2_proofs = vec![Vec::new(); pctx.global_info.air_groups.len()];
        #[allow(unused_assignments)]
        let mut load_const_tree = true;
        for air_groups in my_air_groups.iter() {
            load_const_tree = true;
            for my_instance_id in air_groups.iter() {
                let instance_id = my_instances[*my_instance_id];
                let (airgroup_id, air_id, _) = instances[instance_id];

                let mut proof = proofs[*my_instance_id].clone();

                if pctx.global_info.get_air_has_compressor(airgroup_id, air_id) {
                    if recursive_witness[*my_instance_id].is_none() {
                        let witness = gen_witness_recursive(&pctx, &setups, &proof)?;
                        recursive_witness[*my_instance_id] = Some(witness);
                    }
                    let witness = recursive_witness[*my_instance_id].as_ref().unwrap();
                    proof = generate_recursive_proof(
                        &pctx,
                        &setups,
                        witness,
                        &trace,
                        &prover_buffer,
                        &output_dir_path,
                        d_buffers_aggregation.lock().unwrap().get_ptr(),
                        load_const_tree,
                    );
                    recursive_witness[*my_instance_id] = None;
                }

                if recursive_witness[*my_instance_id].is_none() {
                    let witness = gen_witness_recursive(&pctx, &setups, &proof)?;
                    recursive_witness[*my_instance_id] = Some(witness);
                }

                let witness = recursive_witness[*my_instance_id].as_ref().unwrap();
                let proof_recursive1 = generate_recursive_proof(
                    &pctx,
                    &setups,
                    witness,
                    &trace,
                    &prover_buffer,
                    &output_dir_path,
                    d_buffers_aggregation.lock().unwrap().get_ptr(),
                    load_const_tree,
                );
                recursive2_proofs[airgroup_id].push(proof_recursive1);
                load_const_tree = false;
            }
        }

        let n_airgroups = pctx.global_info.air_groups.len();

        let mut thread_handle_recursion: Option<std::thread::JoinHandle<()>> = None;
        let recursive2_proofs = Arc::new(RwLock::new(recursive2_proofs));

        let mut n_initial_proofs = vec![0; n_airgroups];
        let mut n_rec2_proofs = vec![0; n_airgroups];
        for airgroup in 0..n_airgroups {
            n_initial_proofs[airgroup] = recursive2_proofs.read().unwrap()[airgroup].len();
            n_rec2_proofs[airgroup] = total_recursive_proofs(n_initial_proofs[airgroup]);
            load_const_tree = true;
            for i in 0..n_rec2_proofs[airgroup] {
                if i == n_rec2_proofs[airgroup] - 1 {
                    if let Some(handle) = thread_handle_recursion.take() {
                        handle.join().unwrap();
                    }
                }

                let witness_recursive2 = gen_witness_aggregation(
                    &pctx,
                    &setups,
                    &recursive2_proofs.read().unwrap()[airgroup][3 * i],
                    &recursive2_proofs.read().unwrap()[airgroup][3 * i + 1],
                    &recursive2_proofs.read().unwrap()[airgroup][3 * i + 2],
                )?;

                // Join the previous thread (if any) before starting a new one
                if let Some(handle) = thread_handle_recursion.take() {
                    handle.join().unwrap();
                }

                thread_handle_recursion = Some(Self::generate_recursive_proof_thread(
                    recursive2_proofs.clone(),
                    pctx.clone(),
                    setups.clone(),
                    witness_recursive2,
                    airgroup,
                    trace.clone(),
                    prover_buffer.clone(),
                    output_dir_path.clone(),
                    d_buffers_aggregation.clone(),
                    load_const_tree,
                ));

                load_const_tree = false;
            }
        }

        if let Some(handle) = thread_handle_recursion {
            handle.join().unwrap();
        }

        let mut recursive2_proofs = Arc::try_unwrap(recursive2_proofs).unwrap().into_inner().unwrap();
        for airgroup in 0..n_airgroups {
            recursive2_proofs[airgroup] = recursive2_proofs[airgroup][3 * n_rec2_proofs[airgroup]..].to_vec();
        }

        let agg_recursive2_proof = aggregate_recursive2_proofs(
            &pctx,
            &setups,
            &recursive2_proofs,
            &trace,
            &prover_buffer,
            output_dir_path.clone(),
            d_buffers_aggregation.lock().unwrap().get_ptr(),
        )?;
        pctx.dctx.read().unwrap().barrier();

        let mut proof_id = None;
        if pctx.dctx_get_rank() == 0 {
            let mut vadcop_final_proof = generate_vadcop_final_proof(
                &pctx,
                &setups,
                &agg_recursive2_proof,
                &trace,
                &prover_buffer,
                output_dir_path.clone(),
                d_buffers_aggregation.lock().unwrap().get_ptr(),
            )?;

            proof_id = Some(
                blake3::hash(unsafe {
                    std::slice::from_raw_parts(
                        vadcop_final_proof.proof.as_ptr() as *const u8,
                        vadcop_final_proof.proof.len() * 8,
                    )
                })
                .to_hex()
                .to_string(),
            );

            if pctx.options.final_snark {
                timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
                let recursivef_proof = generate_recursivef_proof(
                    &pctx,
                    &setups,
                    &vadcop_final_proof.proof,
                    &trace,
                    &prover_buffer,
                    output_dir_path.clone(),
                )?;
                timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

                timer_start_info!(GENERATING_FFLONK_SNARK_PROOF);
                let _ = generate_fflonk_snark_proof(&pctx, recursivef_proof, output_dir_path.clone());
                timer_stop_and_log_info!(GENERATING_FFLONK_SNARK_PROOF);
            } else if pctx.options.verify_proofs {
                let setup_path = pctx.global_info.get_setup_path("vadcop_final");
                let stark_info_path = setup_path.display().to_string() + ".starkinfo.json";
                let expressions_bin_path = setup_path.display().to_string() + ".verifier.bin";
                let verkey_path = setup_path.display().to_string() + ".verkey.json";

                timer_start_info!(VERIFYING_VADCOP_FINAL_PROOF);
                let valid_proofs = verify_proof(
                    vadcop_final_proof.proof.as_mut_ptr(),
                    stark_info_path,
                    expressions_bin_path,
                    verkey_path,
                    Some(pctx.get_publics().clone()),
                    None,
                    None,
                );
                timer_stop_and_log_info!(VERIFYING_VADCOP_FINAL_PROOF);
                if !valid_proofs {
                    log::info!(
                        "{}: ··· {}",
                        Self::MY_NAME,
                        "\u{2717} Vadcop Final proof was not verified".bright_red().bold()
                    );
                    return Err("Vadcop Final proof was not verified".into());
                } else {
                    log::info!(
                        "{}:     {}",
                        Self::MY_NAME,
                        "\u{2713} Vadcop Final proof was verified".bright_green().bold()
                    );
                }
            }
        }
        timer_stop_and_log_info!(GENERATING_COMPRESSED_PROOFS);
        timer_stop_and_log_info!(GENERATING_VADCOP_PROOF);
        Ok(proof_id)
    }

    fn get_contribution(
        instance_id: usize,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        aux_trace_contribution_ptr: Arc<Vec<F>>,
        values: Arc<Mutex<Vec<u64>>>,
        d_buffers: Arc<Mutex<DeviceBuffer>>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let ptr = aux_trace_contribution_ptr.as_ptr() as *mut u8;
            let value = Self::get_contribution_air(pctx.clone(), &sctx, instance_id, ptr, d_buffers.clone());

            for (id, value) in value.iter().enumerate().take(10) {
                values.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id) * 10 + id] = *value;
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_proof_thread(
        proofs: Arc<RwLock<Vec<Proof<F>>>>,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        instance_id: usize,
        airgroup_id: usize,
        air_id: usize,
        output_dir_path: PathBuf,
        aux_trace: Arc<Vec<F>>,
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

            let offset_const = get_const_offset_c(setup.p_setup.p_stark_info) as usize;

            if gen_const_tree {
                load_const_pols(
                    &setup.setup_path,
                    setup.const_pols_size,
                    &aux_trace[offset_const..offset_const + setup.const_pols_size],
                );
                load_const_pols_tree(setup, &aux_trace[0..setup.const_tree_size]);
            }

            let mut steps_params = pctx.get_air_instance_params(&sctx, instance_id, true);
            steps_params.aux_trace = aux_trace.as_ptr() as *mut u8;

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
                gen_const_tree,
            );

            airgroup_values_air_instances.lock().unwrap()[pctx.dctx_get_instance_idx(instance_id)] =
                pctx.get_air_instance_airgroup_values(airgroup_id, air_id, air_instance_id);

            timer_stop_and_log_info!(GEN_PROOF);
            proofs.write().unwrap()[pctx.dctx_get_instance_idx(instance_id)] =
                Proof::new(ProofType::Basic, airgroup_id, air_id, Some(instance_id), proof);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_recursive_proof_thread(
        recursive2_proofs: Arc<RwLock<Vec<Vec<Proof<F>>>>>,
        pctx: Arc<ProofCtx<F>>,
        setups: Arc<SetupsVadcop<F>>,
        witness_recursive2: Proof<F>,
        airgroup_id: usize,
        trace: Arc<Vec<F>>,
        prover_buffer: Arc<Vec<F>>,
        output_dir_path: PathBuf,
        d_buffers: Arc<Mutex<DeviceBuffer>>,
        load_constants: bool,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let proof_recursive2 = generate_recursive_proof(
                &pctx,
                &setups,
                &witness_recursive2,
                &trace,
                &prover_buffer,
                &output_dir_path,
                d_buffers.clone().lock().unwrap().get_ptr(),
                load_constants,
            );
            recursive2_proofs.write().unwrap()[airgroup_id].push(proof_recursive2);
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
                    custom_commit_file_path.to_str().expect("Invalid path"),
                );
            }
        }

        let n_airgroup_values = setup
            .stark_info
            .airgroupvalues_map
            .as_ref()
            .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
            .unwrap_or(0);

        let n_air_values = setup
            .stark_info
            .airvalues_map
            .as_ref()
            .map(|map| map.iter().map(|entry| if entry.stage == 1 { 1 } else { 3 }).sum::<usize>())
            .unwrap_or(0);

        if n_air_values > 0 && air_instance.airvalues.is_empty() {
            air_instance.init_airvalues(n_air_values);
        }

        if n_airgroup_values > 0 && air_instance.airgroup_values.is_empty() {
            air_instance.init_airgroup_values(n_airgroup_values);
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
        pctx: Arc<ProofCtx<F>>,
        sctx: &SetupCtx<F>,
        instance_id: usize,
        aux_trace_contribution_ptr: *mut u8,
        d_buffers: Arc<Mutex<DeviceBuffer>>,
    ) -> Vec<u64> {
        let n_field_elements = 4;

        timer_start_info!(GET_CONTRIBUTION_AIR);
        let instances = pctx.dctx_get_instances();

        let (airgroup_id, air_id, all) = instances[instance_id];
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

        if !all {
            let pctx_clone = pctx.clone();
            std::thread::spawn(move || {
                pctx_clone.free_instance(instance_id);
            });
        }

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
