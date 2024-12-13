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
        let n_proof_values = global_info.n_proof_values;
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

    pub fn set_proof_value(&self, name: &str, value: F) {
        let id = (0..self.global_info.n_proof_values)
            .find(|&i| {
                if let Some(proof_value) = self
                    .global_info
                    .proof_values_map
                    .as_ref()
                    .expect("global_info.proof_values_map is not initialized")
                    .get(i)
                {
                    proof_value.name == name
                } else {
                    false
                }
            })
            .unwrap_or_else(|| panic!("No proof value found with name {}", name));

        self.proof_values.values.write().unwrap()[3 * id] = value;
        self.proof_values.values.write().unwrap()[3 * id + 1] = F::zero();
        self.proof_values.values.write().unwrap()[3 * id + 2] = F::zero();
    }

    pub fn set_proof_value_ext(&self, name: &str, value: Vec<F>) {
        let id = (0..self.global_info.n_proof_values)
            .find(|&i| {
                if let Some(proof_value) = self
                    .global_info
                    .proof_values_map
                    .as_ref()
                    .expect("global_info.proof_values_map is not initialized")
                    .get(i)
                {
                    proof_value.name == name
                } else {
                    false
                }
            })
            .unwrap_or_else(|| panic!("No proof value found with name {}", name));

        self.proof_values.values.write().unwrap()[3 * id] = value[0];
        self.proof_values.values.write().unwrap()[3 * id + 1] = value[1];
        self.proof_values.values.write().unwrap()[3 * id + 2] = value[2];
    }

    pub fn get_proof_values_ptr(&self) -> *mut u8 {
        let guard = &self.proof_values.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn set_public_value(&self, value: F, public_id: usize) {
        self.public_inputs.values.write().unwrap()[public_id] = value;
    }

    pub fn set_public_value_by_name(&self, value: F, public_name: &str, lengths: Option<Vec<u64>>) {
        let n_publics: usize = self.global_info.publics_map.as_ref().expect("REASON").len();
        let public_id = (0..n_publics)
            .find(|&i| {
                let public = self.global_info.publics_map.as_ref().expect("REASON").get(i).unwrap();

                // Check if name matches
                let name_matches = public.name == public_name;

                // If lengths is provided, check that it matches public.lengths
                let lengths_match = if let Some(ref provided_lengths) = lengths {
                    Some(&public.lengths) == Some(provided_lengths)
                } else {
                    true // If lengths is None, skip the lengths check
                };

                name_matches && lengths_match
            })
            .unwrap_or_else(|| {
                panic!("Name {} with specified lengths {:?} not found in publics_map", public_name, lengths)
            });

        self.set_public_value(value, public_id);
    }

    pub fn get_public_value(&self, public_name: &str) -> F {
        let n_publics: usize = self.global_info.publics_map.as_ref().expect("REASON").len();
        let public_id = (0..n_publics)
            .find(|&i| {
                let public = self.global_info.publics_map.as_ref().expect("REASON").get(i).unwrap();
                public.name == public_name
            })
            .unwrap_or_else(|| panic!("Name {} not found in publics_map", public_name));

        self.public_inputs.values.read().unwrap()[public_id]
    }

    pub fn get_publics(&self) -> std::sync::RwLockWriteGuard<Vec<F>> {
        self.public_inputs.values.write().unwrap()
    }

    pub fn get_proof_values(&self) -> std::sync::RwLockWriteGuard<Vec<F>> {
        self.proof_values.values.write().unwrap()
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
