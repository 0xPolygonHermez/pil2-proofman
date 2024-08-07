use libloading::{Library, Symbol};
use log::{debug, info, trace};
use p3_field::AbstractField;
use stark::{GlobalInfo, StarkBufferAllocator, StarkProver};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use transcript::FFITranscript;

use witness_helpers::{WitnessLibrary, WitnessLibInitFn};

use proofman_common::{Prover, ExecutionCtx, ProofCtx};

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: AbstractField + 'static> ProofMan<F> {
    const MY_NAME: &'static str = "ProofMan";

    pub fn generate_proof(
        witness_lib_path: PathBuf,
        rom_path: Option<PathBuf>,
        public_inputs_path: PathBuf,
        proving_key_path: PathBuf,
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

        // TODO! Check hash

        let mut pctx = ProofCtx::create_ctx(pilout);

        let buffer_allocator = Arc::new(StarkBufferAllocator);
        let mut ectx = ExecutionCtx::builder().is_discovery_execution().with_buffer_allocator(buffer_allocator).build();

        Self::init_proof(&mut witness_lib, &mut pctx, &mut ectx);

        // Initialize prover and buffers to fit the proof
        let mut provers: Vec<Box<dyn Prover<F>>> = Vec::new();
        Self::initialize_provers(&proving_key_path, &mut provers, &mut pctx);

        if provers.is_empty() {
            return Err("No instances found".into());
        }
        let mut transcript = provers[0].new_transcript();

        ectx.discovering = false;
        for stage in 1..=(pctx.pilout.num_stages() + 1) {
            witness_lib.calculate_witness(stage, &mut pctx, &ectx);
            Self::get_challenges(stage, &mut provers, &mut pctx, &mut transcript);
            Self::commit_stage(stage, &mut provers, &mut pctx);
            Self::calculate_challenges(stage, &mut provers, &mut pctx, &mut transcript);
        }

        witness_lib.end_proof();

        Self::opening_stages(&mut provers, &mut pctx, &mut transcript);

        let proof = Self::finalize_proof(&pctx);

        Ok(proof)
    }

    fn init_proof(wc_lib: &mut Box<dyn WitnessLibrary<F>>, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
        wc_lib.start_proof(pctx, ectx);
        wc_lib.execute(pctx, ectx);

        wc_lib.calculate_plan(ectx);

        trace!("{}: Plan: ", Self::MY_NAME);
        for air_instance in ectx.instances.iter() {
            let air = pctx.pilout.get_air(air_instance.air_group_id, air_instance.air_id);
            let name = if air.name().is_some() { air.name().unwrap() } else { "Unnamed" };
            trace!("{}:     + Air[{}][{}] {}", Self::MY_NAME, air_instance.air_group_id, air_instance.air_id, name);
        }

        // Initialize air instances
        for id in ectx.owned_instances.iter() {
            pctx.air_instances.push((&ectx.instances[*id]).into());
        }
    }

    fn initialize_provers(proving_key_path: &Path, provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        info!("{}: Initializing prover and creating buffers", Self::MY_NAME);

        let global_info = GlobalInfo::from_file(&proving_key_path.join("pilout.globalInfo.json"));

        for air_instance in pctx.air_instances.iter_mut() {
            debug!(
                "{}: Initializing prover for air instance ({}, {})",
                Self::MY_NAME,
                air_instance.air_group_id,
                air_instance.air_id
            );

            let mut prover = Box::new(StarkProver::new(
                proving_key_path,
                &global_info,
                air_instance.air_group_id,
                air_instance.air_id,
            ));

            let buffer_size = prover.get_total_bytes();
            trace!("{}: ··· Preallocating a buffer of {} bytes", Self::MY_NAME, buffer_size);
            air_instance.buffer = vec![0u8; buffer_size];

            prover.as_mut().build(air_instance);
            provers.push(prover);
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
    ) {
        info!("{}: Calculating challenges for stage {}", Self::MY_NAME, stage);
        if stage == 1 {
            for prover in provers.iter_mut() {
                prover.add_challenges_to_transcript(0, pctx, transcript);
            }
            // provers[0].add_publics_to_transcript(pctx);
        }
        for prover in provers.iter_mut() {
            prover.add_challenges_to_transcript(stage as u64, pctx, transcript);
        }
    }

    fn get_challenges(
        stage: u32,
        provers: &mut [Box<dyn Prover<F>>],
        pctx: &mut ProofCtx<F>,
        transcript: &mut FFITranscript,
    ) {
        info!("{}: Getting challenges for stage {}", Self::MY_NAME, stage);
        provers[0].get_challenges(stage, pctx, transcript); // Any prover can get the challenges which are common among them
    }

    pub fn opening_stages(provers: &mut [Box<dyn Prover<F>>], pctx: &mut ProofCtx<F>, transcript: &mut FFITranscript) {
        for opening_id in 1..=3 {
            Self::get_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx, transcript);
            for (idx, prover) in provers.iter_mut().enumerate() {
                info!("{}: Opening stage {}, for prover {}", Self::MY_NAME, opening_id, idx);
                prover.opening_stage(opening_id, pctx);
            }
            Self::calculate_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx, transcript);
        }
    }

    fn finalize_proof(_proof_ctx: &ProofCtx<F>) -> Vec<F> {
        // This is a mock implementation
        vec![]
    }
}
