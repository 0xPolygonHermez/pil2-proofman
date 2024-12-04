use std::{collections::HashMap, os::raw::c_void, sync::Arc};
use std::path::PathBuf;
use p3_field::Field;
use proofman_starks_lib_c::get_custom_commit_map_ids_c;
use proofman_util::create_buffer_fast;

use crate::{SetupCtx, StarkInfo};

#[repr(C)]
pub struct StepsParams {
    pub witness: *mut c_void,
    pub trace: *mut c_void,
    pub public_inputs: *mut c_void,
    pub challenges: *mut c_void,
    pub airgroup_values: *mut c_void,
    pub airvalues: *mut c_void,
    pub evals: *mut c_void,
    pub xdivxsub: *mut c_void,
    pub p_const_pols: *mut c_void,
    pub p_const_tree: *mut c_void,
    pub custom_commits: [*mut c_void; 10],
    pub custom_commits_extended: [*mut c_void; 10],
}

impl From<&StepsParams> for *mut c_void {
    fn from(params: &StepsParams) -> *mut c_void {
        params as *const StepsParams as *mut c_void
    }
}

impl Default for StepsParams {
    fn default() -> Self {
        StepsParams {
            trace: ptr::null_mut(),
            pols: ptr::null_mut(),
            public_inputs: ptr::null_mut(),
            challenges: ptr::null_mut(),
            airgroup_values: ptr::null_mut(),
            airvalues: ptr::null_mut(),
            evals: ptr::null_mut(),
            xdivxsub: ptr::null_mut(),
            p_const_pols: ptr::null_mut(),
            p_const_tree: ptr::null_mut(),
            custom_commits: [ptr::null_mut(); 10],
        }
    }
}

#[derive(Default)]
pub struct CustomCommitsInfo<F> {
    pub buffer: Vec<F>,
    pub cached_file: PathBuf,
}

impl<F> CustomCommitsInfo<F> {
    pub fn new(buffer: Vec<F>, cached_file: PathBuf) -> Self {
        Self { buffer, cached_file }
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
    pub witness: Vec<F>,
    pub trace: Option<Vec<F>>,
    pub custom_commits: Vec<CustomCommitsInfo<F>>,
    pub custom_commits_extended: Vec<CustomCommitsInfo<F>>,
    pub airgroup_values: Vec<F>,
    pub airvalues: Vec<F>,
    pub evals: Vec<F>,
    pub commits_calculated: HashMap<usize, bool>,
    pub airgroupvalue_calculated: HashMap<usize, bool>,
    pub airvalue_calculated: HashMap<usize, bool>,
    pub custom_commits_calculated: Vec<HashMap<usize, bool>>,
    pub stark_info: StarkInfo,
}

impl<F: Field> AirInstance<F> {
    pub fn new(
        setup_ctx: Arc<SetupCtx>,
        airgroup_id: usize,
        air_id: usize,
        air_segment_id: Option<usize>,
        witness: Vec<F>,
    ) -> Self {
        let ps = setup_ctx.get_setup(airgroup_id, air_id);

        let n_custom_commits = ps.stark_info.custom_commits.len();
        let custom_commits_calculated = vec![HashMap::new(); n_custom_commits];

        let mut custom_commits = Vec::new();
        let mut custom_commits_extended = Vec::new();

        for i in 0..n_custom_commits {
            let n_cols =
                *ps.stark_info.map_sections_n.get(&(ps.stark_info.custom_commits[i].name.clone() + "0")).unwrap()
                    as usize;
            custom_commits.push(CustomCommitsInfo::new(
                create_buffer_fast((1 << ps.stark_info.stark_struct.n_bits) * n_cols),
                PathBuf::new(),
            ));
            custom_commits_extended.push(CustomCommitsInfo::new(
                create_buffer_fast((1 << ps.stark_info.stark_struct.n_bits_ext) * n_cols),
                PathBuf::new(),
            ));
        }

        AirInstance {
            airgroup_id,
            air_id,
            air_segment_id,
            air_instance_id: None,
            idx: None,
            global_idx: None,
            witness,
            trace: None,
            custom_commits,
            custom_commits_extended,
            airgroup_values: vec![F::zero(); ps.stark_info.airgroupvalues_map.as_ref().unwrap().len() * 3],
            airvalues: vec![F::zero(); ps.stark_info.airvalues_map.as_ref().unwrap().len() * 3],
            evals: vec![F::zero(); ps.stark_info.ev_map.len() * 3],
            commits_calculated: HashMap::new(),
            airgroupvalue_calculated: HashMap::new(),
            airvalue_calculated: HashMap::new(),
            custom_commits_calculated,
            stark_info: ps.stark_info.clone(),
        }
    }

    pub fn get_witness_ptr(&self) -> *mut u8 {
        self.witness.as_ptr() as *mut u8
    }

    pub fn set_trace(&mut self, trace: Vec<F>) {
        self.trace = Some(trace);
    }

    pub fn get_trace_ptr(&self) -> *mut u8 {
        match &self.trace {
            Some(trace) => trace.as_ptr() as *mut u8,
            None => std::ptr::null_mut(), // Return null if `trace` is `None`
        }
    }

    pub fn get_custom_commits_ptr(&self) -> [*mut c_void; 10] {
        let mut ptrs = [std::ptr::null_mut(); 10];
        for (i, custom_commit) in self.custom_commits.iter().enumerate() {
            ptrs[i] = custom_commit.buffer.as_ptr() as *mut c_void;
        }
        ptrs
    }

    pub fn get_custom_commits_extended_ptr(&self) -> [*mut c_void; 10] {
        let mut ptrs = [std::ptr::null_mut(); 10];
        for (i, custom_commit) in self.custom_commits_extended.iter().enumerate() {
            ptrs[i] = custom_commit.buffer.as_ptr() as *mut c_void;
        }
        ptrs
    }

    pub fn set_custom_commit_cached_file(&mut self, setup_ctx: &SetupCtx, commit_id: u64, cached_file: PathBuf) {
        let ps = setup_ctx.get_setup(self.airgroup_id, self.air_id);

        self.custom_commits[commit_id as usize].cached_file = cached_file;

        let ids = get_custom_commit_map_ids_c(ps.p_setup.p_stark_info, commit_id, 0);
        for idx in ids {
            self.set_custom_commit_calculated(commit_id as usize, idx as usize);
        }
    }

    pub fn set_custom_commit_id_buffer(&mut self, setup_ctx: &SetupCtx, buffer: Vec<F>, commit_id: u64) {
        self.custom_commits[commit_id as usize].buffer = buffer;

        let ps = setup_ctx.get_setup(self.airgroup_id, self.air_id);

        let ids = get_custom_commit_map_ids_c(ps.p_setup.p_stark_info, commit_id, 0);
        for idx in ids {
            self.set_custom_commit_calculated(commit_id as usize, idx as usize);
        }
    }

    pub fn set_airvalue(&mut self, name: &str, lengths: Option<Vec<u64>>, value: F) {
        let airvalues_map = self.stark_info.airvalues_map.as_ref().unwrap();
        let airvalue_id = (0..airvalues_map.len())
            .find(|&i| {
                let airvalue = airvalues_map.get(i).unwrap();

                // Check if name matches
                let name_matches = airvalues_map[i].name == name;

                // If lengths is provided, check that it matches airvalue.lengths
                let lengths_match = if let Some(ref provided_lengths) = lengths {
                    Some(&airvalue.lengths) == Some(provided_lengths)
                } else {
                    true // If lengths is None, skip the lengths check
                };

                name_matches && lengths_match
            })
            .unwrap_or_else(|| panic!("Name {} with specified lengths {:?} not found in airvalues", name, lengths));

        self.airvalues[airvalue_id * 3] = value;
        self.set_airvalue_calculated(airvalue_id);
    }

    pub fn set_airvalue_ext(&mut self, name: &str, lengths: Option<Vec<u64>>, value: Vec<F>) {
        let airvalues_map = self.stark_info.airvalues_map.as_ref().unwrap();
        let airvalue_id = (0..airvalues_map.len())
            .find(|&i| {
                let airvalue = airvalues_map.get(i).unwrap();

                // Check if name matches
                let name_matches = airvalues_map[i].name == name;

                // If lengths is provided, check that it matches airvalue.lengths
                let lengths_match = if let Some(ref provided_lengths) = lengths {
                    Some(&airvalue.lengths) == Some(provided_lengths)
                } else {
                    true // If lengths is None, skip the lengths check
                };

                name_matches && lengths_match
            })
            .unwrap_or_else(|| panic!("Name {} with specified lengths {:?} not found in airvalues", name, lengths));

        assert!(value.len() == 3, "Value vector must have exactly 3 elements");

        let mut value_iter = value.into_iter();

        self.airvalues[airvalue_id * 3] = value_iter.next().unwrap();
        self.airvalues[airvalue_id * 3 + 1] = value_iter.next().unwrap();
        self.airvalues[airvalue_id * 3 + 2] = value_iter.next().unwrap();

        self.set_airvalue_calculated(airvalue_id);
    }

    pub fn set_airgroupvalue(&mut self, name: &str, lengths: Option<Vec<u64>>, value: F) {
        let airgroupvalues_map = self.stark_info.airgroupvalues_map.as_ref().unwrap();
        let airgroupvalue_id = (0..airgroupvalues_map.len())
            .find(|&i| {
                let airgroupvalue = airgroupvalues_map.get(i).unwrap();

                // Check if name matches
                let name_matches = airgroupvalues_map[i].name == name;

                // If lengths is provided, check that it matches airgroupvalues.lengths
                let lengths_match = if let Some(ref provided_lengths) = lengths {
                    Some(&airgroupvalue.lengths) == Some(provided_lengths)
                } else {
                    true // If lengths is None, skip the lengths check
                };

                name_matches && lengths_match
            })
            .unwrap_or_else(|| {
                panic!("Name {} with specified lengths {:?} not found in airgroupvalues", name, lengths)
            });

        self.airgroup_values[airgroupvalue_id * 3] = value;
        self.set_airgroupvalue_calculated(airgroupvalue_id);
    }

    pub fn set_airgroupvalue_ext(&mut self, name: &str, lengths: Option<Vec<u64>>, value: Vec<F>) {
        let airgroupvalues_map = self.stark_info.airgroupvalues_map.as_ref().unwrap();
        let airgroupvalue_id = (0..airgroupvalues_map.len())
            .find(|&i| {
                let airgroupvalue = airgroupvalues_map.get(i).unwrap();

                // Check if name matches
                let name_matches = airgroupvalues_map[i].name == name;

                // If lengths is provided, check that it matches airgroupvalues.lengths
                let lengths_match = if let Some(ref provided_lengths) = lengths {
                    Some(&airgroupvalue.lengths) == Some(provided_lengths)
                } else {
                    true // If lengths is None, skip the lengths check
                };

                name_matches && lengths_match
            })
            .unwrap_or_else(|| {
                panic!("Name {} with specified lengths {:?} not found in airgroupvalues", name, lengths)
            });

        assert!(value.len() == 3, "Value vector must have exactly 3 elements");

        let mut value_iter = value.into_iter();

        self.airgroup_values[airgroupvalue_id * 3] = value_iter.next().unwrap();
        self.airgroup_values[airgroupvalue_id * 3 + 1] = value_iter.next().unwrap();
        self.airgroup_values[airgroupvalue_id * 3 + 2] = value_iter.next().unwrap();

        self.set_airgroupvalue_calculated(airgroupvalue_id);
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
