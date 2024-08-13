use libloading::{Library, Symbol};
use log::{debug, info, trace};
use p3_field::AbstractField;
use stark::{GlobalInfo, StarkProver};
use std::path::PathBuf;
use common::Prover;

use wchelpers::WCLibrary;

use common::{ExecutionCtx, ProofCtx};

pub struct ProofMan<F> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: AbstractField + 'static> ProofMan<F> {
    const MY_NAME: &'static str = "ProofMan";

    pub fn generate_proof(
        wc_lib_path: PathBuf,
        proving_key_path: PathBuf,
        public_inputs: Vec<u8>,
    ) -> Result<Vec<F>, Box<dyn std::error::Error>> {
        // Check wc_lib path exists
        if !wc_lib_path.exists() {
            return Err(format!("Witness computation dynamic library not found at path: {:?}", wc_lib_path).into());
        }

        // Check proving_key path exists
        if !proving_key_path.exists() {
            return Err(format!("Proving key not found at path: {:?}", proving_key_path).into());
        }
        // Check provingKey is a folder
        if !proving_key_path.is_dir() {
            return Err(format!("Proving key path is not a folder: {:?}", proving_key_path).into());
        }

        // Load the witness computation dynamic library
        let library = unsafe { Library::new(wc_lib_path.clone())? };
        let wc_lib: Symbol<fn() -> Box<dyn WCLibrary<F>>> = unsafe { library.get(b"init_library")? };
        let mut wc_lib: Box<dyn WCLibrary<F>> = wc_lib();

        let mut pctx = ProofCtx::create_ctx(wc_lib.pilout(), public_inputs);
        let mut ectx = ExecutionCtx::builder().is_discovery_execution().build();
        let mut provers: Vec<Box<dyn Prover<F>>> = Vec::new();

        Self::initialize_witness(&mut wc_lib, &mut pctx, &mut ectx);
        Self::initialize_provers(&proving_key_path, &mut provers, &mut pctx);

        // Commit stages
        let num_commit_stages = pctx.pilout.num_stages() + 1;
        for stage in 1..=num_commit_stages {
            Self::get_challenges(stage, &mut provers, &mut pctx);
            wc_lib.calculate_witness(stage, &mut pctx, &ectx, &provers);
            Self::commit_stage(stage, &mut provers, &mut pctx);
            Self::calculate_challenges(stage, &mut provers, &mut pctx);
        }

        wc_lib.end_proof();

        Self::opening_stages(&mut provers, &mut pctx);

        let proof = Self::finalize_proof(&pctx);

        Ok(proof)
    }

    fn initialize_witness(wc_lib: &mut Box<dyn WCLibrary<F>>, pctx: &mut ProofCtx<F>, ectx: &mut ExecutionCtx) {
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
        ectx.discovering = false;
    }

    fn initialize_provers(proving_key_path: &PathBuf, provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        info!("{}: Initializing prover and creating buffers", Self::MY_NAME);

        let global_info = GlobalInfo::from_file(&proving_key_path.join("pilout.globalInfo.json"));

        for air_instance in pctx.air_instances.iter_mut() {
            debug!(
                "{}: Initializing prover for air instance ({}, {})",
                Self::MY_NAME,
                air_instance.air_group_id,
                air_instance.air_id
            );

            let prover = Box::new(StarkProver::new(
                &proving_key_path,
                &global_info,
                air_instance.air_group_id,
                air_instance.air_id,
            ));

            let buffer_size = prover.get_total_bytes();
            trace!("{}: ··· Preallocating a buffer of {} bytes", Self::MY_NAME, buffer_size);
            air_instance.buffer = vec![0u8; buffer_size];

            provers.push(prover);
        }
        for (idx, prover) in provers.iter_mut().enumerate() {
            prover.build(pctx, idx);
        }
        Self::calculate_challenges(0, provers, pctx);
        provers[0].add_publics_to_transcript(pctx);
    }

    pub fn commit_stage(stage: u32, provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        info!("{}: Committing stage {}", Self::MY_NAME, stage);

        for (idx, prover) in provers.iter_mut().enumerate() {
            info!("{}: Committing stage {}, for prover {}", Self::MY_NAME, stage, idx);
            prover.commit_stage(stage, pctx);
        }
    }

    fn calculate_challenges(stage: u32, provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        info!("{}: Calculating challenges for stage {}", Self::MY_NAME, stage);
        for prover in provers.iter_mut() {
            prover.add_challenges_to_transcript(stage as u64, pctx);
        }
    }

    fn get_challenges(stage: u32, provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        info!("{}: Getting challenges for stage {}", Self::MY_NAME, stage);
        provers[0].get_challenges(stage, pctx); // Any prover can get the challenges which are common among them
    }

    pub fn opening_stages(provers: &mut Vec<Box<dyn Prover<F>>>, pctx: &mut ProofCtx<F>) {
        for opening_id in 1..=provers[0].num_opening_stages() {
            Self::get_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx);
            for (idx, prover) in provers.iter_mut().enumerate() {
                info!("{}: Opening stage {}, for prover {}", Self::MY_NAME, opening_id, idx);
                prover.opening_stage(opening_id, pctx);
            }
            Self::calculate_challenges(pctx.pilout.num_stages() + 1 + opening_id, provers, pctx);
        }
    }

    fn finalize_proof(_proof_ctx: &ProofCtx<F>) -> Vec<F> {
        // This is a mock implementation
        vec![]
    }
}
