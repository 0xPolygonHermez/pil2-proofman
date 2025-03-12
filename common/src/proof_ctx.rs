use std::os::raw::c_void;
use std::sync::atomic::AtomicU64;
use std::{collections::HashMap, sync::RwLock};
use std::path::PathBuf;

use p3_field::Field;
use transcript::FFITranscript;

use crate::{AirInstance, DistributionCtx, GlobalInfo, SetupCtx, StdMode, StepsParams, VerboseMode};

pub struct Values<F> {
    pub values: RwLock<Vec<F>>,
}

impl<F: Field> Values<F> {
    pub fn new(n_values: usize) -> Self {
        Self { values: RwLock::new(vec![F::ZERO; n_values]) }
    }
}

impl<F> Default for Values<F> {
    fn default() -> Self {
        Self { values: RwLock::new(Vec::new()) }
    }
}

pub type AirGroupMap = HashMap<usize, AirIdMap>;
pub type AirIdMap = HashMap<usize, InstanceMap>;
pub type InstanceMap = HashMap<usize, Vec<usize>>;

#[derive(Clone)]
pub struct ProofOptions {
    pub verify_constraints: bool,
    pub verbose_mode: VerboseMode,
    pub aggregation: bool,
    pub final_snark: bool,
    pub debug_info: DebugInfo,
    pub verify_proofs: bool,
}

#[derive(Default, Clone)]
pub struct DebugInfo {
    pub debug_instances: AirGroupMap,
    pub debug_global_instances: Vec<usize>,
    pub std_mode: StdMode,
    pub save_proofs_to_file: bool,
}

impl DebugInfo {
    pub fn new_debug() -> Self {
        Self {
            debug_instances: HashMap::new(),
            debug_global_instances: Vec::new(),
            std_mode: StdMode::new_debug(),
            save_proofs_to_file: true,
        }
    }
}
impl ProofOptions {
    pub fn new(
        verify_constraints: bool,
        verbose_mode: VerboseMode,
        aggregation: bool,
        final_snark: bool,
        verify_proofs: bool,
        debug_info: DebugInfo,
    ) -> Self {
        Self { verify_constraints, verbose_mode, aggregation, final_snark, debug_info, verify_proofs }
    }
}

#[allow(dead_code)]
pub struct ProofCtx<F: Field> {
    pub public_inputs: Values<F>,
    pub proof_values: Values<F>,
    pub global_challenge: Values<F>,
    pub challenges: Values<F>,
    pub global_info: GlobalInfo,
    pub air_instances: RwLock<HashMap<usize, AirInstance<F>>>,
    pub options: ProofOptions,
    pub weights: HashMap<(usize, usize), u64>,
    pub custom_commits_fixed: HashMap<String, PathBuf>,
    pub dctx: RwLock<DistributionCtx>,
}

impl<F: Field> ProofCtx<F> {
    const MY_NAME: &'static str = "ProofCtx";

    pub fn create_ctx(
        proving_key_path: PathBuf,
        custom_commits_fixed: HashMap<String, PathBuf>,
        options: ProofOptions,
    ) -> Self {
        log::info!("{}: Creating proof context", Self::MY_NAME);

        let global_info: GlobalInfo = GlobalInfo::new(&proving_key_path);
        let n_publics = global_info.n_publics;
        let n_proof_values = global_info.proof_values_map.as_ref().unwrap().len();
        let n_challenges = global_info.n_challenges.iter().sum::<usize>();

        let weights = HashMap::new();

        Self {
            global_info,
            public_inputs: Values::new(n_publics),
            proof_values: Values::new(n_proof_values * 3),
            challenges: Values::new(n_challenges * 3),
            global_challenge: Values::new(3),
            air_instances: RwLock::new(HashMap::new()),
            dctx: RwLock::new(DistributionCtx::new()),
            custom_commits_fixed,
            weights,
            options,
        }
    }

    pub fn set_weights(&mut self, sctx: &SetupCtx<F>) {
        for (airgroup_id, air_group) in self.global_info.airs.iter().enumerate() {
            for (air_id, _) in air_group.iter().enumerate() {
                let setup = sctx.get_setup(airgroup_id, air_id);
                let weight = (setup
                    .stark_info
                    .map_sections_n
                    .iter()
                    .filter(|(key, _)| *key != "const")
                    .map(|(_, value)| *value)
                    .sum::<u64>())
                    * (1 << (setup.stark_info.stark_struct.n_bits_ext));
                self.weights.insert((airgroup_id, air_id), weight);
            }
        }
    }

    pub fn get_weight(&self, airgroup_id: usize, air_id: usize) -> u64 {
        *self.weights.get(&(airgroup_id, air_id)).unwrap()
    }

    pub fn get_custom_commits_fixed_buffer(&self, name: &str) -> Result<&str, Box<std::io::Error>> {
        let file_name = self.custom_commits_fixed.get(name);
        match file_name {
            Some(path) => Ok(path.to_str().expect("Invalid UTF-8 in path")),
            None => {
                // Return error
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Custom Commit Fixed {:?} not found", file_name),
                )))
            }
        }
    }

    pub fn add_air_instance(&self, air_instance: AirInstance<F>, global_idx: usize) {
        let mut air_instances = self.air_instances.write().unwrap();
        air_instances.insert(global_idx, air_instance);
    }

    pub fn dctx_barrier(&self) {
        let dctx = self.dctx.read().unwrap();
        dctx.barrier();
    }

    pub fn dctx_is_min_rank_owner(&self, airgroup_id: usize, air_id: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.is_min_rank_owner(airgroup_id, air_id)
    }

    pub fn dctx_get_rank(&self) -> usize {
        let dctx = self.dctx.read().unwrap();
        dctx.rank as usize
    }

    pub fn dctx_get_n_processes(&self) -> usize {
        let dctx = self.dctx.read().unwrap();
        dctx.n_processes as usize
    }

    pub fn dctx_get_instances(&self) -> Vec<(usize, usize, bool)> {
        let dctx = self.dctx.read().unwrap();
        dctx.instances.clone()
    }

    pub fn dctx_get_my_instances(&self) -> Vec<usize> {
        let dctx = self.dctx.read().unwrap();
        dctx.my_instances.clone()
    }

    pub fn dctx_get_my_groups(&self) -> Vec<Vec<usize>> {
        let dctx = self.dctx.read().unwrap();
        dctx.my_groups.clone()
    }

    pub fn dctx_get_my_air_groups(&self) -> Vec<Vec<usize>> {
        let dctx = self.dctx.read().unwrap();
        dctx.my_air_groups.clone()
    }

    pub fn dctx_get_instance_info(&self, global_idx: usize) -> (usize, usize) {
        let dctx = self.dctx.read().unwrap();
        dctx.get_instance_info(global_idx)
    }

    pub fn dctx_get_instance_idx(&self, global_idx: usize) -> usize {
        let dctx = self.dctx.read().unwrap();
        dctx.get_instance_idx(global_idx)
    }

    pub fn dctx_is_my_instance(&self, global_idx: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.is_my_instance(global_idx)
    }

    pub fn dctx_is_instance_all(&self, global_idx: usize) -> bool {
        let dctx = self.dctx.read().unwrap();
        dctx.instances[global_idx].2
    }

    pub fn dctx_find_air_instance_id(&self, global_idx: usize) -> usize {
        let dctx = self.dctx.read().unwrap();
        dctx.find_air_instance_id(global_idx)
    }

    pub fn dctx_find_instance(&self, airgroup_id: usize, air_id: usize) -> (bool, usize) {
        let dctx = self.dctx.read().unwrap();
        dctx.find_instance(airgroup_id, air_id)
    }

    pub fn add_instance(&self, airgroup_id: usize, air_id: usize) -> usize {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_instance_no_assign(airgroup_id, air_id, weight)
    }

    pub fn add_instance_all(&self, airgroup_id: usize, air_id: usize) -> usize {
        let mut dctx = self.dctx.write().unwrap();
        let weight = self.get_weight(airgroup_id, air_id);
        dctx.add_instance_no_assign_all(airgroup_id, air_id, weight)
    }

    pub fn dctx_distribute_roots(&self, roots: Vec<u64>) -> Vec<u64> {
        let dctx = self.dctx.read().unwrap();
        dctx.distribute_roots(roots)
    }

    pub fn dctx_add_instance_no_assign(&self, airgroup_id: usize, air_id: usize, weight: u64) -> usize {
        let mut dctx = self.dctx.write().unwrap();
        dctx.add_instance_no_assign(airgroup_id, air_id, weight)
    }

    pub fn dctx_assign_instances(&self) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.assign_instances();
    }

    pub fn dctx_load_balance_info(&self) -> (f64, u64, u64, f64) {
        let dctx = self.dctx.read().unwrap();
        dctx.load_balance_info()
    }

    pub fn dctx_set_balance_distribution(&self, balance: bool) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.set_balance_distribution(balance);
    }

    pub fn dctx_distribute_multiplicity(&self, multiplicity: &[AtomicU64], global_idx: usize) {
        let dctx = self.dctx.read().unwrap();
        let owner = dctx.owner(global_idx);
        dctx.distribute_multiplicity(multiplicity, owner);
    }

    pub fn dctx_distribute_publics(&self, publics: Vec<u64>) {
        let dctx = self.dctx.read().unwrap();
        let publics_to_set = dctx.distribute_publics(publics);
        for idx in (0..publics_to_set.len()).step_by(2) {
            self.set_public_value(publics_to_set[idx + 1], publics_to_set[idx] as usize);
        }
    }

    pub fn dctx_distribute_multiplicities(&self, multiplicities: &[Vec<AtomicU64>], global_idx: usize) {
        let dctx = self.dctx.read().unwrap();
        let owner = dctx.owner(global_idx);
        dctx.distribute_multiplicities(multiplicities, owner);
    }

    pub fn dctx_distribute_airgroupvalues(&self, airgroup_values: Vec<Vec<u64>>) -> Vec<Vec<F>> {
        let dctx = self.dctx.read().unwrap();
        dctx.distribute_airgroupvalues(airgroup_values, &self.global_info)
    }

    pub fn dctx_close(&self) {
        let mut dctx = self.dctx.write().unwrap();
        dctx.close(self.global_info.air_groups.len());
    }

    pub fn get_proof_values_ptr(&self) -> *mut u8 {
        let guard = &self.proof_values.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn set_public_value(&self, value: u64, public_id: usize) {
        self.public_inputs.values.write().unwrap()[public_id] = F::from_u64(value);
    }

    pub fn set_global_challenge(&self, stage: usize, global_challenge: &[F]) {
        let mut global_challenge_guard = self.global_challenge.values.write().unwrap();
        global_challenge_guard[0] = global_challenge[0];
        global_challenge_guard[1] = global_challenge[1];
        global_challenge_guard[2] = global_challenge[2];

        let transcript = FFITranscript::new(2, true);

        transcript.add_elements(global_challenge.as_ptr() as *mut u8, 3);
        let challenges_guard = self.challenges.values.read().unwrap();

        let initial_pos = self.global_info.n_challenges.iter().take(stage - 1).sum::<usize>();
        let num_challenges = self.global_info.n_challenges[stage - 1];
        for i in 0..num_challenges {
            transcript.get_challenge(&challenges_guard[(initial_pos + i) * 3] as *const F as *mut c_void);
        }
    }

    pub fn set_challenge(&self, index: usize, challenge: &[F]) {
        let mut challenges_guard = self.challenges.values.write().unwrap();
        challenges_guard[index] = challenge[0];
        challenges_guard[index + 1] = challenge[1];
        challenges_guard[index + 2] = challenge[2];
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

    pub fn get_challenges(&self) -> std::sync::RwLockWriteGuard<Vec<F>> {
        self.challenges.values.write().unwrap()
    }

    pub fn get_challenges_ptr(&self) -> *mut u8 {
        let guard = &self.challenges.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_global_challenge(&self) -> std::sync::RwLockWriteGuard<Vec<F>> {
        self.global_challenge.values.write().unwrap()
    }

    pub fn get_global_challenge_ptr(&self) -> *mut u8 {
        let guard = &self.global_challenge.values.read().unwrap();
        guard.as_ptr() as *mut u8
    }

    pub fn get_air_instance_params(&self, sctx: &SetupCtx<F>, instance_id: usize, gen_proof: bool) -> StepsParams {
        let air_instances = self.air_instances.read().unwrap();
        let air_instance = air_instances.get(&instance_id).unwrap();

        let instances = self.dctx_get_instances();
        let (airgroup_id, air_id, _) = instances[instance_id];
        let setup = sctx.get_setup(airgroup_id, air_id);

        let challenges = if gen_proof { air_instance.get_challenges_ptr() } else { self.get_challenges_ptr() };
        let aux_trace: *mut u8 = if gen_proof { std::ptr::null_mut() } else { air_instance.get_aux_trace_ptr() };

        StepsParams {
            trace: air_instance.get_trace_ptr(),
            aux_trace,
            public_inputs: self.get_publics_ptr(),
            proof_values: self.get_proof_values_ptr(),
            challenges,
            airgroup_values: air_instance.get_airgroup_values_ptr(),
            airvalues: air_instance.get_airvalues_ptr(),
            evals: air_instance.get_evals_ptr(),
            xdivxsub: std::ptr::null_mut(),
            p_const_pols: setup.get_const_ptr(),
            p_const_tree: std::ptr::null_mut(),
            custom_commits_fixed: air_instance.get_custom_commits_fixed_ptr(),
        }
    }

    pub fn get_air_instance_trace_ptr(&self, instance_id: usize) -> *mut u8 {
        self.air_instances.read().unwrap().get(&instance_id).unwrap().get_trace_ptr()
    }

    pub fn get_air_instance_trace(&self, airgroup_id: usize, air_id: usize, air_instance_id: usize) -> Vec<F> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_instance_id(airgroup_id, air_id, air_instance_id);
        if let Some(index) = index {
            return self.air_instances.read().unwrap().get(&index).unwrap().get_trace();
        } else {
            panic!(
                "Air Instance with id {} for airgroup {} and air {} not found",
                air_instance_id, airgroup_id, air_id
            );
        }
    }

    pub fn get_air_instance_air_values(&self, airgroup_id: usize, air_id: usize, air_instance_id: usize) -> Vec<F> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_instance_id(airgroup_id, air_id, air_instance_id);
        if let Some(index) = index {
            return self.air_instances.read().unwrap().get(&index).unwrap().get_air_values();
        } else {
            panic!(
                "Air Instance with id {} for airgroup {} and air {} not found",
                air_instance_id, airgroup_id, air_id
            );
        }
    }

    pub fn get_air_instance_airgroup_values(
        &self,
        airgroup_id: usize,
        air_id: usize,
        air_instance_id: usize,
    ) -> Vec<F> {
        let dctx = self.dctx.read().unwrap();
        let index = dctx.find_instance_id(airgroup_id, air_id, air_instance_id);
        if let Some(index) = index {
            return self.air_instances.read().unwrap().get(&index).unwrap().get_airgroup_values();
        } else {
            panic!(
                "Air Instance with id {} for airgroup {} and air {} not found",
                air_instance_id, airgroup_id, air_id
            );
        }
    }

    pub fn free_instance(&self, instance_id: usize) {
        let mut air_instances = self.air_instances.write().unwrap();
        air_instances.remove(&instance_id);
    }
}
