use std::{collections::HashMap, os::raw::c_void, sync::Arc};

use p3_field::Field;
use proofman_starks_lib_c::{
    get_airval_id_by_name_c, get_n_airgroupvals_c, get_n_airvals_c, get_n_evals_c, get_airgroupval_id_by_name_c,
    get_n_custom_commits_c,
};

use crate::SetupCtx;

#[repr(C)]
pub struct StepsParams {
    pub buffer: *mut c_void,
    pub public_inputs: *mut c_void,
    pub challenges: *mut c_void,
    pub airgroup_values: *mut c_void,
    pub airvalues: *mut c_void,
    pub evals: *mut c_void,
    pub xdivxsub: *mut c_void,
    pub p_const_pols: *mut c_void,
    pub p_const_tree: *mut c_void,
    pub custom_commits: *mut *mut c_void,
}

impl From<&StepsParams> for *mut c_void {
    fn from(params: &StepsParams) -> *mut c_void {
        params as *const StepsParams as *mut c_void
    }
}

/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
#[repr(C)]
pub struct AirInstance<F> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_segment_id: Option<usize>,
    pub air_instance_id: Option<usize>,
    pub idx: Option<usize>,
    pub global_idx: Option<usize>,
    pub buffer: Vec<F>,
    pub custom_commits: Vec<Vec<F>>,
    pub airgroup_values: Vec<F>,
    pub airvalues: Vec<F>,
    pub evals: Vec<F>,
    pub commits_calculated: HashMap<usize, bool>,
    pub airgroupvalue_calculated: HashMap<usize, bool>,
    pub airvalue_calculated: HashMap<usize, bool>,
    pub custom_commits_calculated: Vec<HashMap<usize, bool>>,
}

impl<F: Field> AirInstance<F> {
    pub fn new(
        setup_ctx: Arc<SetupCtx<F>>,
        airgroup_id: usize,
        air_id: usize,
        air_segment_id: Option<usize>,
        buffer: Vec<F>,
    ) -> Self {
        let ps = setup_ctx.get_setup(airgroup_id, air_id);

        let custom_commits_calculated = vec![HashMap::new(); get_n_custom_commits_c(ps.p_setup.p_stark_info) as usize];

        let mut custom_commits = Vec::new();

        let n_custom_commits = get_n_custom_commits_c(ps.p_setup.p_stark_info);
        for _ in 0..n_custom_commits {
            custom_commits.push(Vec::new());
        }

        AirInstance {
            airgroup_id,
            air_id,
            air_segment_id,
            air_instance_id: None,
            idx: None,
            global_idx: None,
            buffer,
            custom_commits,
            airgroup_values: vec![F::zero(); get_n_airgroupvals_c(ps.p_setup.p_stark_info) as usize * 3],
            airvalues: vec![F::zero(); get_n_airvals_c(ps.p_setup.p_stark_info) as usize * 3],
            evals: vec![F::zero(); get_n_evals_c(ps.p_setup.p_stark_info) as usize * 3],
            commits_calculated: HashMap::new(),
            airgroupvalue_calculated: HashMap::new(),
            airvalue_calculated: HashMap::new(),
            custom_commits_calculated,
        }
    }

    pub fn get_buffer_ptr(&self) -> *mut u8 {
        self.buffer.as_ptr() as *mut u8
    }

    pub fn get_custom_commits_ptr(&self) -> *mut *mut c_void {
        let mut custom_commits = Vec::new();
        for c in &self.custom_commits {
            let ptr = c.as_ptr() as *mut c_void;
            custom_commits.push(ptr);
        }
        
        custom_commits.as_mut_ptr()
    }

    pub fn set_custom_commit_id_buffer(&mut self, buffer: Vec<F>, commit_id: u64) {
        self.custom_commits[commit_id as usize] = buffer;
    }

    pub fn set_airvalue(&mut self, setup_ctx: &SetupCtx<F>, name: &str, value: F) {
        let ps = setup_ctx.get_setup(self.airgroup_id, self.air_id);

        let id = get_airval_id_by_name_c(ps.p_setup.p_stark_info, name);
        if id == -1 {
            panic!("{} is not an airval!", name);
        }

        self.airvalues[id as usize * 3] = value;
        self.set_airvalue_calculated(id as usize);
    }

    pub fn set_airvalue_ext(&mut self, setup_ctx: &SetupCtx<F>, name: &str, value: Vec<F>) {
        let ps = setup_ctx.get_setup(self.airgroup_id, self.air_id);

        let id = get_airval_id_by_name_c(ps.p_setup.p_stark_info, name);
        if id == -1 {
            panic!("{} is not an airval!", name);
        }

        assert!(value.len() == 3, "Value vector must have exactly 3 elements");

        let mut value_iter = value.into_iter();

        self.airvalues[id as usize * 3] = value_iter.next().unwrap();
        self.airvalues[id as usize * 3 + 1] = value_iter.next().unwrap();
        self.airvalues[id as usize * 3 + 2] = value_iter.next().unwrap();

        self.set_airvalue_calculated(id as usize);
    }

    pub fn set_airgroupvalue(&mut self, setup_ctx: &SetupCtx<F>, name: &str, value: F) {
        let ps = setup_ctx.get_setup(self.airgroup_id, self.air_id);

        let id = get_airgroupval_id_by_name_c(ps.p_setup.p_stark_info, name);
        if id == -1 {
            panic!("{} is not an airval!", name);
        }

        self.airgroup_values[id as usize * 3] = value;
        self.set_airgroupvalue_calculated(id as usize);
    }

    pub fn set_airgroupvalue_ext(&mut self, setup_ctx: &SetupCtx<F>, name: &str, value: Vec<F>) {
        let ps = setup_ctx.get_setup(self.airgroup_id, self.air_id);

        let id = get_airgroupval_id_by_name_c(ps.p_setup.p_stark_info, name);
        if id == -1 {
            panic!("{} is not an airval!", name);
        }

        assert!(value.len() == 3, "Value vector must have exactly 3 elements");

        let mut value_iter = value.into_iter();

        self.airgroup_values[id as usize * 3] = value_iter.next().unwrap();
        self.airgroup_values[id as usize * 3 + 1] = value_iter.next().unwrap();
        self.airgroup_values[id as usize * 3 + 2] = value_iter.next().unwrap();

        self.set_airgroupvalue_calculated(id as usize);
    }

    pub fn set_commit_calculated(&mut self, id: usize) {
        self.commits_calculated.insert(id, true);
    }

    pub fn set_custom_commit_calculated(&mut self, commit_id: usize, id: usize) {
        self.custom_commits_calculated[commit_id].insert(id, true);
    }

    pub fn set_air_instance_id(&mut self, air_instance_id: usize, idx: usize) {
        self.air_instance_id = Some(air_instance_id);
        self.idx = Some(idx);
    }

    pub fn set_airgroupvalue_calculated(&mut self, id: usize) {
        self.airgroupvalue_calculated.insert(id, true);
    }

    pub fn set_airvalue_calculated(&mut self, id: usize) {
        self.airvalue_calculated.insert(id, true);
    }
}
