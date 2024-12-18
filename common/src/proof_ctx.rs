use std::sync::RwLock;
use std::path::PathBuf;

use p3_field::Field;

use crate::{distribution_ctx::DistributionCtx, AirInstancesRepository, GlobalInfo, StdMode, VerboseMode};

pub struct Values<F> {
    pub values: RwLock<Vec<F>>,
}

impl<F: Field> Values<F> {
    pub fn new(n_values: usize) -> Self {
        Self { values: RwLock::new(vec![F::zero(); n_values]) }
    }
}

impl<F> Default for Values<F> {
    fn default() -> Self {
        Self { values: RwLock::new(Vec::new()) }
    }
}

#[derive(Clone)]
pub struct ProofOptions {
    pub verify_constraints: bool,
    pub verbose_mode: VerboseMode,
    pub std_mode: StdMode,
    pub aggregation: bool,
    pub final_snark: bool,
}

impl ProofOptions {
    pub fn new(
        verify_constraints: bool,
        verbose_mode: VerboseMode,
        std_mode: StdMode,
        aggregation: bool,
        final_snark: bool,
    ) -> Self {
        Self { verify_constraints, verbose_mode, std_mode, aggregation, final_snark }
    }
}

#[allow(dead_code)]
pub struct ProofCtx<F> {
    pub public_inputs: Values<F>,
    pub proof_values: Values<F>,
    pub challenges: Values<F>,
    pub buff_helper: Values<F>,
    pub global_info: GlobalInfo,
    pub air_instance_repo: AirInstancesRepository<F>,
    pub options: ProofOptions,
    pub dctx: RwLock<DistributionCtx>,
}

impl<F: Field> ProofCtx<F> {
    const MY_NAME: &'static str = "ProofCtx";

    pub fn create_ctx(proving_key_path: PathBuf, options: ProofOptions) -> Self {
        log::info!("{}: Creating proof context", Self::MY_NAME);

        let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path);
        let n_publics = global_info.n_publics;
        let n_proof_values = global_info.proof_values_map.as_ref().unwrap().len();
        let n_challenges = 4 + global_info.n_challenges.iter().sum::<usize>();

        Self {
            global_info,
            public_inputs: Values::new(n_publics),
            proof_values: Values::new(n_proof_values * 3),
            challenges: Values::new(n_challenges * 3),
            buff_helper: Values::default(),
            air_instance_repo: AirInstancesRepository::new(),
            dctx: RwLock::new(DistributionCtx::new()),
            options,
        }
    }

    pub fn dctx_add_instance(&self, airgroup_id: usize, air_id: usize, weight: usize) -> (bool, usize) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.add_instance(airgroup_id, air_id, weight)
    }

    pub fn dctx_distribute_multiplicity(&self, multiplicity: &mut [u64], instance_idx: usize) {
        let dctx = self.dctx.read().unwrap();
        let owner = dctx.owner(instance_idx);
        dctx.distribute_multiplicity(multiplicity, owner);
    }

    pub fn get_proof_values_ptr(&self) -> *mut u8 {
        let guard = &self.proof_values.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn set_public_value(&self, value: F, public_id: usize) {
        self.public_inputs.values.write().unwrap()[public_id] = value;
    }

    pub fn get_publics(&self) -> std::sync::RwLockWriteGuard<Vec<F>> {
        self.public_inputs.values.write().unwrap()
    }

    pub fn get_proof_values(&self) -> std::sync::RwLockWriteGuard<Vec<F>> {
        self.proof_values.values.write().unwrap()
    }

    pub fn get_proof_values_by_stage(&self, stage: u32) -> Vec<F> {
        let proof_vals = self.proof_values.values.read().unwrap();

        let mut values = Vec::new();
        let mut p = 0;
        for proof_value in self.global_info.proof_values_map.as_ref().unwrap() {
            if proof_value.stage > stage as u64 {
                break;
            }
            if proof_value.stage == 1 {
                if stage == 1 {
                    values.push(proof_vals[p]);
                }
                p += 1;
            } else {
                if proof_value.stage == stage as u64 {
                    values.push(proof_vals[p]);
                    values.push(proof_vals[p + 1]);
                    values.push(proof_vals[p + 2]);
                }
                p += 3;
            }
        }

        values
    }

    pub fn get_publics_ptr(&self) -> *mut u8 {
        let guard = &self.public_inputs.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_challenges_ptr(&self) -> *mut u8 {
        let guard = &self.challenges.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_buff_helper_ptr(&self) -> *mut u8 {
        let guard = &self.buff_helper.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }
}
