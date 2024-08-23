use libloading::{Library, Symbol};
use log::{debug, info, trace};
use p3_field::Field;
use stark::{GlobalInfo, StarkBufferAllocator, StarkProver};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use transcript::FFITranscript;

use crate::{WitnessLibrary, WitnessLibInitFn};

use proofman_common::{Prover, ExecutionCtx, ProofCtx};

use colored::*;

use std::os::raw::c_void;

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field + 'static> ProofMan<F> {
    const MY_NAME: &'static str = "ProofMan";

    pub fn generate_proof(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: PathBuf,
        proving_key_path: PathBuf,
        debug_mode: bool,
    ) -> Result<Vec<F>, Box<dyn std::error::Error>> {
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
        if !public_inputs_path.exists() {
            return Err(format!("Public inputs file not found at path: {:?}", public_inputs_path).into());
        }

        // Check proving_key_path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key folder not found at path: {:?}", proving_key_path).into());
        }

        // Check proving_key_path is a folder
        if !proving_key_path.is_dir() {
            return Err(format!("Proving key parameter must be a folder: {:?}", proving_key_path).into());
        }

        // Load the witness computation dynamic library
        let library = unsafe { Library::new(&witness_lib_path)? };

        let witness_lib: Symbol<WitnessLibInitFn<F>> = unsafe { library.get(b"init_library")? };

        let mut witness_lib = witness_lib(rom_path.clone(), public_inputs_path.clone(), proving_key_path.clone())?;

        let pilout = witness_lib.pilout();

        let mut pctx = ProofCtx::create_ctx(pilout);

        let mut provers: Vec<Box<dyn Prover<F>>> = Vec::new();

        let buffer_allocator = Arc::new(StarkBufferAllocator::new(proving_key_path.clone()));
        let mut ectx = ExecutionCtx::builder().with_buffer_allocator(buffer_allocator).build();

        Self::initialize_witness(&mut witness_lib, &mut pctx, &mut ectx);
        witness_lib.calculate_witness(1, &mut pctx, &ectx);

        Self::initialize_provers(&proving_key_path, &mut provers, &mut pctx);

        if provers.is_empty() {
            return Err("No instances found".into());
        }
        let mut transcript = provers[0].new_transcript();

        Self::calculate_challenges(0, &mut provers, &mut pctx, &mut transcript, debug_mode);
        provers[0].add_publics_to_transcript(&mut pctx, &transcript);

        let mut valid_constraints = true;

        // Commit stages
        let num_commit_stages = pctx.pilout.num_stages();
        for stage in 1..=num_commit_stages {
            if stage != 1 {
                witness_lib.calculate_witness(stage, &mut pctx, &ectx);
            }

            Self::get_challenges(stage, &mut provers, &mut pctx, &transcript);

            if debug_mode {
                valid_constraints = Self::verify_constraints(stage, &mut provers);
            } else {
                Self::commit_stage(stage, &mut provers, &mut pctx, debug_mode);
            }
            
            Self::calculate_challenges(stage, &mut provers, &mut pctx, &mut transcript, debug_mode);
        }

        witness_lib.end_proof();

        if debug_mode {

            // TODO: Verify global constraints! 

            if !valid_constraints {
                println!("{}", format!("Not all constraints were verified.").bright_red().bold());
            } else {
                println!("{}", format!("All constraints were successfully verified.").bright_green().bold());
            }

            return Ok(vec![]);
        }

        // Compute Quotient polynomial
        Self::get_challenges(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx, &transcript);
        Self::commit_stage(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx, debug_mode);
        Self::calculate_challenges(pctx.pilout.num_stages() + 1, &mut provers, &mut pctx, &mut transcript, false);

        // Compute openings
        Self::opening_stages(&mut provers, &mut pctx, &mut transcript);

        let proof = Self::finalize_proof(&pctx);

        Ok(proof)
    }

    fn initialize_witness(
        witness_lib: &mut Box<dyn WitnessLibrary<F>>,
        pctx: &mut ProofCtx<F>,
        ectx: &mut ExecutionCtx,
    ) {
        witness_lib.start_proof(pctx, ectx);

        witness_lib.execute(pctx, ectx);

        trace!("{}: Air instances: ", Self::MY_NAME);

        for air_instance in pctx.air_instances.read().unwrap().iter() {
            let air = pctx.pilout.get_air(air_instance.air_group_id, air_instance.air_id);

            let name = if air.name().is_some() { air.name().unwrap() } else { "Unnamed" };
            trace!("{}:     + Air[{}][{}] {}", Self::MY_NAME, air.air_group_id, air.air_id, name);
        }
    }

    fn initialize_provers(proving_key_path: &Path, provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        info!("{}: Initializing prover and creating buffers", Self::MY_NAME);

        let global_info = GlobalInfo::from_file(&proving_key_path.join("pilout.globalInfo.json"));

        for air_instance in pctx.air_instances.write().unwrap().iter_mut() {
            debug!(
                "{}: Initializing prover for air instance ({}, {})",
                Self::MY_NAME,
                air_instance.air_group_id,
                air_instance.air_id
            );

            let prover = Box::new(StarkProver::new(
                proving_key_path,
                &global_info,
                air_instance.air_group_id,
                air_instance.air_id,
            ));

            provers.push(prover);
        }
        for (idx, prover) in provers.iter_mut().enumerate() {
            prover.build(pctx, idx);
        }
    }

    pub fn verify_constraints(stage: u32, provers: &mut [Box<dyn Prover<F>>]) -> bool {
        info!("{}: Verifying constraints stage {}", Self::MY_NAME, stage);

        let mut valid_constraints = true;
        for (idx, prover) in provers.iter_mut().enumerate() {
            info!("{}: Verifying constraints stage {}, for prover {}", Self::MY_NAME, stage, idx);
            valid_constraints = prover.verify_constraints(stage);
        }
        valid_constraints
    }

    pub fn commit_stage(stage: u32, provers: &mut [Box<dyn Prover<F>>], pctx: &mut ProofCtx<F>, debug_mode: bool) {
        if debug_mode { return; }

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
        debug_mode: bool,
    ) {
        info!("{}: Calculating challenges for stage {}", Self::MY_NAME, stage);
        for prover in provers.iter_mut() {
            if debug_mode {
                let dummy_elements = vec![F::zero(), F::one(), F::two(), F::neg_one()];
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
            Self::calculate_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx, transcript, false);
        }
    }

    fn finalize_proof(_proof_ctx: &ProofCtx<F>) -> Vec<F> {
        // This is a mock implementation
        vec![]
    }
}
