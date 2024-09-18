use libloading::{Library, Symbol};
use log::{debug, info, trace};
use p3_field::Field;
use stark::{StarkBufferAllocator, StarkProver};
use proofman_starks_lib_c::{save_challenges_c, save_publics_c, verify_global_constraints_c};
use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use std::{cmp, fs};
use proofman_starks_lib_c::*;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use transcript::FFITranscript;

use crate::{WitnessLibrary, WitnessLibInitFn};

use proofman_common::{
    AirInstancesRepository, ConstraintInfo, ExecutionCtx, ProofCtx, Prover, SetupCtx, GlobalInfo, WitnessPilout,
    ProofType,
};

use colored::*;

use std::os::raw::c_void;

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

type GetCommitedPolsFunc = unsafe extern "C" fn(
    p_address: *mut c_void,
    zkin: *mut c_void,
    n: u64,
    offset_cm1: u64,
    dat_file: *const c_char,
    exec_file: *const c_char,
);

impl<F: Field + 'static> ProofMan<F> {
    const MY_NAME: &'static str = "ProofMan";

    pub fn generate_proof(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: Option<PathBuf>,
        proving_key_path: PathBuf,
        output_dir_path: PathBuf,
        debug_mode: u64,
    ) -> Result<(ProofCtx<F>, WitnessPilout, GlobalInfo, Vec<*mut c_void>), Box<dyn std::error::Error>> {
        // Check witness_lib path exists
        if !witness_lib_path.exists() {
            return Err(format!("Witness computation dynamic library not found at path: {:?}", witness_lib_path).into());
        }

        // Check rom_path path exists
        if let Some(rom_path) = rom_path.as_ref() {
            if !rom_path.exists() {
                return Err(format!("ROM file not found at path: {:?}", rom_path).into());
            }
        }

        // Check public_inputs_path is a folder
        if let Some(publics_path) = public_inputs_path.as_ref() {
            if !publics_path.exists() {
                return Err(format!("Public inputs file not found at path: {:?}", publics_path).into());
            }
        }

        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        // Check proving_key_path is a folder
        if !proving_key_path.is_dir() {
            return Err(format!("Proving key parameter must be a folder: {:?}", proving_key_path).into());
        }

        if debug_mode == 0 && !output_dir_path.exists() {
            fs::create_dir_all(&output_dir_path)
                .map_err(|err| format!("Failed to create output directory: {:?}", err))?;
        }

        // Load the witness computation dynamic library
        let library = unsafe { Library::new(&witness_lib_path)? };

        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };

        let mut witness_lib = witness_lib(rom_path.clone(), public_inputs_path.clone())?;

        let air_instances_repo = AirInstancesRepository::new();
        let air_instances_repo = Arc::new(air_instances_repo);

        let mut pctx = ProofCtx::create_ctx(witness_lib.pilout(), air_instances_repo.clone());

        let mut provers: Vec<Box<dyn Prover<F>>> = Vec::new();

        let global_info = GlobalInfo::from_file(&proving_key_path.display().to_string());

        let sctx = SetupCtx::new(&witness_lib.pilout(), &global_info, &ProofType::Basic);

        let buffer_allocator: Arc<StarkBufferAllocator> = Arc::new(StarkBufferAllocator::new(proving_key_path.clone()));

        let mut ectx = ExecutionCtx::builder().with_buffer_allocator(buffer_allocator).build();

        Self::initialize_witness(&mut witness_lib, &mut pctx, &mut ectx, &sctx);

        witness_lib.calculate_witness(1, &mut pctx, &ectx, &sctx);

        Self::initialize_provers(&sctx, &proving_key_path, &mut provers, &mut pctx);

        if provers.is_empty() {
            return Err("No instances found".into());
        }
        let mut transcript = provers[0].new_transcript();

        Self::calculate_challenges(0, &mut provers, &mut pctx, &mut transcript, 0);
        provers[0].add_publics_to_transcript(&mut pctx, &transcript);

        // Commit stages
        let num_commit_stages = pctx.pilout.num_stages();
        for stage in 1..=num_commit_stages {
            Self::get_challenges(stage, &mut provers, &mut pctx, &transcript);

            if stage != 1 {
                witness_lib.calculate_witness(stage, &mut pctx, &ectx, &sctx);
            }

            Self::calculate_stage(stage, &mut provers, &mut pctx);

            if debug_mode == 0 {
                Self::commit_stage(stage, &mut provers, &mut pctx);
            }

            if debug_mode == 0 || stage < num_commit_stages {
                Self::calculate_challenges(stage, &mut provers, &mut pctx, &mut transcript, debug_mode);
            }
        }

        witness_lib.end_proof();

        if debug_mode != 0 {
            let mut proofs: Vec<*mut c_void> = Vec::new();

            for prover in provers.iter_mut() {
                let proof = prover.get_proof();
                proofs.push(proof);
            }

            log::info!("{}: <-- Verifying constraints", Self::MY_NAME);

            witness_lib.debug(&pctx, &ectx, &sctx);

            let constraints = Self::verify_constraints(&mut provers, &mut pctx);

            let mut valid_constraints = true;
            for (idx, prover) in provers.iter_mut().enumerate() {
                let prover_info = prover.get_prover_info();
                let air_instances =
                    pctx.air_instance_repo.find_air_instances(prover_info.airgroup_id, prover_info.air_id);
                let air_instance_index = air_instances.iter().position(|&x| x == prover_info.prover_idx).unwrap();
                let air = pctx.pilout.get_air(prover_info.airgroup_id, prover_info.air_id);
                let mut valid_constraints_prover = true;
                log::debug!("{}: ··· Air {} Instance {}:", Self::MY_NAME, air.name().unwrap(), air_instance_index);
                for constraint in &constraints[idx] {
                    if (debug_mode == 1 && constraint.n_rows == 0) || (debug_mode != 3 && constraint.im_pol) {
                        continue;
                    }
                    let line_str = unsafe { CStr::from_ptr(constraint.line) };
                    let valid = if constraint.n_rows > 0 {
                        format!("has {} invalid rows", constraint.n_rows).bright_red()
                    } else {
                        "is valid".bright_green()
                    };
                    if constraint.im_pol {
                        log::debug!(
                            "{}: ···    Intermediate polynomial (stage {}) {} -> {:?}",
                            Self::MY_NAME,
                            constraint.stage,
                            valid,
                            line_str.to_str().unwrap()
                        );
                    } else {
                        log::debug!(
                            "{}: ···    Constraint {} (stage {}) {} -> {:?}",
                            Self::MY_NAME,
                            constraint.id,
                            constraint.stage,
                            valid,
                            line_str.to_str().unwrap()
                        );
                    }
                    if constraint.n_rows > 0 {
                        valid_constraints_prover = false;
                    }
                    let n_rows = cmp::min(constraint.n_rows, 10);
                    for i in 0..n_rows {
                        let row = constraint.rows[i as usize];
                        if row.dim == 1 {
                            log::debug!(
                                "{}: ···        Failed at row {} with value: {}",
                                Self::MY_NAME,
                                row.row,
                                row.value[0]
                            );
                        } else {
                            log::debug!(
                                "{}: ···        Failed at row {} with value: [{}, {}, {}]",
                                Self::MY_NAME,
                                row.row,
                                row.value[0],
                                row.value[1],
                                row.value[2]
                            );
                        }
                    }
                    log::debug!("{}: ···   ", Self::MY_NAME);
                }

                if !valid_constraints_prover {
                    log::debug!(
                        "{}: ··· {}",
                        Self::MY_NAME,
                        format!(
                            "Not all constraints for instance {} of air {} were verified!",
                            air_instance_index,
                            air.name().unwrap()
                        )
                        .bright_yellow()
                        .bold()
                    );
                } else {
                    log::debug!(
                        "{}: ··· {}",
                        Self::MY_NAME,
                        format!(
                            "All constraints for instance {} of air {} were verified!",
                            air_instance_index,
                            air.name().unwrap()
                        )
                        .bright_cyan()
                        .bold()
                    );
                }
                log::debug!("{}: ···   ", Self::MY_NAME);
                if !valid_constraints_prover {
                    valid_constraints = false;
                }
            }

            log::info!("{}: <-- Checking global constraints", Self::MY_NAME);

            let global_constraints_verified = verify_global_constraints_c(
                proving_key_path.join("pilout.globalInfo.json").to_str().unwrap(),
                proving_key_path.join("pilout.globalConstraints.bin").to_str().unwrap(),
                pctx.public_inputs.clone().as_ptr() as *mut c_void,
                proofs.as_mut_ptr() as *mut c_void,
                provers.len() as u64,
            );

            if !global_constraints_verified {
                log::debug!(
                    "{}: ··· {}",
                    Self::MY_NAME,
                    "Not all global constraints were verified.".bright_yellow().bold()
                );
            } else {
                log::debug!(
                    "{}: ··· {}",
                    Self::MY_NAME,
                    "All global constraints were successfully verified.".bright_cyan().bold()
                );
            }

            if valid_constraints && global_constraints_verified {
                log::debug!("{}: ··· {}", Self::MY_NAME, "All constraints were verified!".bright_green().bold());
            } else {
                log::debug!("{}: ··· {}", Self::MY_NAME, "Not all constraints were verified.".bright_red().bold());
            }

            return Ok((pctx, witness_lib.pilout(), global_info, Vec::new()));
        }

        // Compute Quotient polynomial
        Self::get_challenges(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx, &transcript);
        Self::calculate_stage(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx);
        Self::commit_stage(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx);
        Self::calculate_challenges(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx, &mut transcript, 0);

        // Compute openings
        Self::opening_stages(&mut provers, &mut pctx, &mut transcript);

        //Generate prooves_out
        let mut proves_out = Vec::new();
        let _proof = Self::finalize_proof(
            &proving_key_path,
            &mut provers,
            &mut pctx,
            output_dir_path.to_string_lossy().as_ref(),
            &mut proves_out,
        );

        Ok((pctx, witness_lib.pilout(), global_info, proves_out))
    }

    fn initialize_witness(
        witness_lib: &mut Box<dyn WitnessLibrary<F>>,
        pctx: &mut ProofCtx<F>,
        ectx: &mut ExecutionCtx,
        sctx: &SetupCtx,
    ) {
        witness_lib.start_proof(pctx, ectx, sctx);

        witness_lib.execute(pctx, ectx, sctx);

        // After the execution print the planned instances
        trace!("{}: --> Air instances: ", Self::MY_NAME);

        let mut group_ids = HashMap::new();

        for air_instance in pctx.air_instance_repo.air_instances.read().unwrap().iter() {
            let group_map = group_ids.entry(air_instance.airgroup_id).or_insert_with(HashMap::new);
            *group_map.entry(air_instance.air_id).or_insert(0) += 1;
        }

        let mut sorted_group_ids: Vec<_> = group_ids.keys().collect();
        sorted_group_ids.sort();

        for &airgroup_id in &sorted_group_ids {
            if let Some(air_map) = group_ids.get(airgroup_id) {
                let mut sorted_air_ids: Vec<_> = air_map.keys().collect();
                sorted_air_ids.sort();

                let air_group = pctx.pilout.get_air_group(*airgroup_id);
                let name = air_group.name().unwrap_or("Unnamed");
                trace!("{}:     + AirGroup [{}] {}", Self::MY_NAME, *airgroup_id, name);

                for &air_id in &sorted_air_ids {
                    if let Some(&count) = air_map.get(air_id) {
                        let air = pctx.pilout.get_air(*airgroup_id, *air_id);
                        let name = air.name().unwrap_or("Unnamed");
                        trace!("{}:       · {} x Air[{}] {}", Self::MY_NAME, count, air.air_id, name);
                    }
                }
            }
        }
    }

    fn initialize_provers(
        sctx: &SetupCtx,
        proving_key_path: &Path,
        provers: &mut Vec<Box<dyn Prover<F>>>,
        pctx: &mut ProofCtx<F>,
    ) {
        info!("{}: Initializing prover and creating buffers", Self::MY_NAME);

        for (prover_idx, air_instance) in pctx.air_instance_repo.air_instances.read().unwrap().iter().enumerate() {
            debug!(
                "{}: Initializing prover for air instance ({}, {})",
                Self::MY_NAME,
                air_instance.airgroup_id,
                air_instance.air_id
            );

            let prover = Box::new(StarkProver::new(
                sctx,
                proving_key_path,
                air_instance.airgroup_id,
                air_instance.air_id,
                prover_idx,
            ));

            provers.push(prover);
        }

        for prover in provers.iter_mut() {
            prover.build(pctx);
        }
    }

    pub fn verify_constraints(provers: &mut [Box<dyn Prover<F>>], pctx: &mut ProofCtx<F>) -> Vec<Vec<ConstraintInfo>> {
        let mut invalid_constraints = Vec::new();
        for prover in provers.iter_mut() {
            let invalid_constraints_prover = prover.verify_constraints(pctx);
            invalid_constraints.push(invalid_constraints_prover);
        }
        invalid_constraints
    }

    pub fn calculate_stage(stage: u32, provers: &mut [Box<dyn Prover<F>>], pctx: &mut ProofCtx<F>) {
        info!("{}: Calculating stage {}", Self::MY_NAME, stage);
        for (idx, prover) in provers.iter_mut().enumerate() {
            info!("{}: Calculating stage {}, for prover {}", Self::MY_NAME, stage, idx);
            prover.calculate_stage(stage, pctx);
        }
    }

    pub fn commit_stage(stage: u32, provers: &mut [Box<dyn Prover<F>>], pctx: &mut ProofCtx<F>) {
        info!("{}: Committing stage {}", Self::MY_NAME, stage);

        for (idx, prover) in provers.iter_mut().enumerate() {
            info!("{}: Committing stage {}, for prover {}", Self::MY_NAME, stage, idx);
            prover.commit_stage(stage, pctx);
        }
    }

    fn calculate_challenges(
        stage: u32,
        provers: &mut [Box<dyn Prover<F>>],
        pctx: &mut ProofCtx<F>,
        transcript: &mut FFITranscript,
        debug_mode: u64,
    ) {
        info!("{}: Calculating challenges for stage {}", Self::MY_NAME, stage);
        for prover in provers.iter_mut() {
            if debug_mode != 0 {
                let dummy_elements = [F::zero(), F::one(), F::two(), F::neg_one()];
                transcript.add_elements(dummy_elements.as_ptr() as *mut c_void, 4);
            } else {
                prover.add_challenges_to_transcript(stage as u64, pctx, transcript);
            }
        }
    }

    fn get_challenges(
        stage: u32,
        provers: &mut [Box<dyn Prover<F>>],
        pctx: &mut ProofCtx<F>,
        transcript: &FFITranscript,
    ) {
        info!("{}: Getting challenges for stage {}", Self::MY_NAME, stage);
        provers[0].get_challenges(stage, pctx, transcript); // Any prover can get the challenges which are common among them
    }

    pub fn opening_stages(provers: &mut [Box<dyn Prover<F>>], pctx: &mut ProofCtx<F>, transcript: &mut FFITranscript) {
        for opening_id in 1..=provers[0].num_opening_stages() {
            Self::get_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx, transcript);
            for (idx, prover) in provers.iter_mut().enumerate() {
                info!("{}: Opening stage {}, for prover {}", Self::MY_NAME, opening_id, idx);
                prover.opening_stage(opening_id, pctx, transcript);
            }
            if opening_id < provers[0].num_opening_stages() {
                Self::calculate_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx, transcript, 0);
            }
        }
    }

    fn finalize_proof(
        proving_key_path: &Path,
        provers: &mut [Box<dyn Prover<F>>],
        pctx: &mut ProofCtx<F>,
        output_dir: &str,
        proves_out: &mut Vec<*mut c_void>,
    ) -> Vec<F> {
        let n_publics = (pctx.public_inputs.borrow().len() / 8) as u64;
        let public_inputs = (*pctx.public_inputs.borrow()).as_ptr() as *mut c_void;
        let challenges = (*pctx.challenges.borrow()).as_ptr() as *mut c_void;

        let global_info_path = proving_key_path.join("pilout.globalInfo.json");
        let global_info_file: &str = global_info_path.to_str().unwrap();

        for (idx, prover) in provers.iter_mut().enumerate() {
            proves_out.push(fri_proof_get_zkinproof_c(
                idx as u64,
                prover.get_proof(),
                public_inputs,
                challenges,
                prover.get_prover_params(),
                global_info_file,
                output_dir,
            ));
        }

        for (idx, prover) in provers.iter_mut().enumerate() {
            prover.save_proof(idx as u64, output_dir);
        }

        save_publics_c(n_publics, public_inputs, output_dir);
        save_challenges_c(challenges, global_info_file, output_dir);

        vec![]
    }

    //
    // Recursion prove
    //

    pub fn generate_recursion_proof(
        pctx: &mut ProofCtx<F>,
        pilout: &WitnessPilout,
        global_setup_info: &GlobalInfo,
        proves: &Vec<*mut c_void>,
        proof_type: &ProofType,
    ) -> Result<Vec<*mut c_void>, Box<dyn std::error::Error>> {
        //
        let sctx = SetupCtx::new(pilout, global_setup_info, proof_type); /*problem*/
        let mut proves_out: Vec<*mut c_void> = Vec::new();

        // Run proves
        for (prover_idx, air_instance) in pctx.air_instance_repo.air_instances.write().unwrap().iter_mut().enumerate() {
            // get buffer address
            let p_address = air_instance.get_buffer_ptr() as *mut c_void;

            let air_setup_folder =
                global_setup_info.get_air_setup_path(air_instance.airgroup_id, air_instance.air_id, proof_type);
            trace!("{}   : ··· Setup AIR folder: {:?}", Self::MY_NAME, air_setup_folder);

            // Check path exists and is a folder
            if !air_setup_folder.exists() {
                panic!("Setup AIR folder not found at path: {:?}", air_setup_folder);
            }
            if !air_setup_folder.is_dir() {
                panic!("Setup AIR path is not a folder: {:?}", air_setup_folder);
            }
            let base_filename_path = match proof_type {
                ProofType::Basic => air_setup_folder
                    .join(global_setup_info.get_air_name(air_instance.airgroup_id, air_instance.air_id))
                    .display()
                    .to_string(),
                ProofType::Compressor => air_setup_folder.join("compressor").display().to_string(),
                ProofType::Recursive1 => air_setup_folder.join("recursive1").display().to_string(),
                ProofType::Recursive2 => air_setup_folder.join("recursive2").display().to_string(),
            };

            // witness computation
            let rust_lib_filename = base_filename_path.clone() + ".so";
            let rust_lib_path = Path::new(rust_lib_filename.as_str());

            if !rust_lib_path.exists() {
                return Err(format!("Rust lib dynamic library not found at path: {:?}", rust_lib_path).into());
            }

            // Load the dynamic library at runtime
            let library = unsafe { Library::new(&rust_lib_path)? };

            // get setup
            let setup: &proofman_common::Setup =
                sctx.setups.get_setup(air_instance.airgroup_id, air_instance.air_id).expect("Setup not found");
                
            let p_setup: *mut c_void = setup.p_setup;
            let p_stark_info: *mut c_void = setup.p_stark_info;

            let n = get_stark_info_n_c(p_stark_info);
            let offset_cm1 = get_map_offsets_c(p_stark_info, "cm1", false);

            // Load the symbol (function) from the library
            unsafe {
                let get_commited_pols: Symbol<GetCommitedPolsFunc> = library.get(b"getCommitedPols\0")?;

                // Call the function
                let dat_filename = base_filename_path.clone() + ".dat";
                let dat_filename_str = CString::new(dat_filename.as_str()).unwrap();
                let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

                let exec_filename = base_filename_path.clone() + ".exec";
                let exec_filename_str = CString::new(exec_filename.as_str()).unwrap();
                let exec_filename_ptr = exec_filename_str.as_ptr() as *mut std::os::raw::c_char;

                let zkin = proves[prover_idx];
                get_commited_pols(p_address, zkin, n, offset_cm1, dat_filename_ptr, exec_filename_ptr);
            }

            // TODO: THIS IS WRONG
            let publics = vec![F::zero(); 1];

            // prove
            let p_prove = gen_recursive_proof_c(p_setup, p_address, publics.as_ptr() as *mut c_void);
            proves_out.push(p_prove);
        }
        Ok(proves_out)
    }
}
