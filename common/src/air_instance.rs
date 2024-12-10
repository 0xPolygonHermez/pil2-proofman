use std::sync::Arc;
use std::path::PathBuf;
use p3_field::Field;
use proofman_util::create_buffer_fast;

use crate::{trace::Trace, trace::Values, SetupCtx, Setup, StarkInfo};

#[repr(C)]
pub struct StepsParams {
    pub trace: *mut u8,
    pub aux_trace: *mut u8,
    pub public_inputs: *mut u8,
    pub challenges: *mut u8,
    pub airgroup_values: *mut u8,
    pub airvalues: *mut u8,
    pub evals: *mut u8,
    pub xdivxsub: *mut u8,
    pub p_const_pols: *mut u8,
    pub p_const_tree: *mut u8,
    pub custom_commits: [*mut u8; 10],
    pub custom_commits_extended: [*mut u8; 10],
}

impl From<&StepsParams> for *mut u8 {
    fn from(params: &StepsParams) -> *mut u8 {
        params as *const StepsParams as *mut u8
    }
}

impl Default for StepsParams {
    fn default() -> Self {
        StepsParams {
            trace: ptr::null_mut(),
            aux_trace: ptr::null_mut(),
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

pub struct TraceInfo<F> {
    airgroup_id: usize,
    air_id: usize,
    trace: Vec<F>,
    custom_traces: Option<Vec<Vec<F>>>,
    air_values: Option<Vec<F>>,
}

impl<F> TraceInfo<F> {
    pub fn new(airgroup_id: usize, air_id: usize, trace: Vec<F>) -> Self {
        Self { airgroup_id, air_id, trace, custom_traces: None, air_values: None }
    }

    pub fn with_custom_traces(mut self, custom_traces: Vec<Vec<F>>) -> Self {
        self.custom_traces = Some(custom_traces);
        self
    }

    pub fn with_air_values(mut self, air_values: Vec<F>) -> Self {
        self.air_values = Some(air_values);
        self
    }
}

pub struct FromTrace<'a, F> {
    pub trace: &'a mut dyn Trace<F>,
    pub custom_traces: Option<Vec<&'a mut dyn Trace<F>>>,
    pub air_values: Option<&'a mut dyn Values<F>>,
}

impl<'a, F> FromTrace<'a, F> {
    pub fn new(trace: &'a mut dyn Trace<F>) -> Self {
        Self { trace, custom_traces: None, air_values: None }
    }

    pub fn with_custom_traces(mut self, custom_traces: Vec<&'a mut dyn Trace<F>>) -> Self {
        self.custom_traces = Some(custom_traces);
        self
    }

    pub fn with_air_values(mut self, air_values: &'a mut dyn Values<F>) -> Self {
        self.air_values = Some(air_values);
        self
    }
}

/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone)]
pub struct AirInstance<F> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub air_instance_id: Option<usize>,
    pub idx: Option<usize>,
    pub global_idx: Option<usize>,
    pub trace: Vec<F>,
    pub aux_trace: Option<Vec<F>>,
    pub custom_commits: Vec<CustomCommitsInfo<F>>,
    pub custom_commits_extended: Vec<CustomCommitsInfo<F>>,
    pub airgroup_values: Vec<F>,
    pub airvalues: Vec<F>,
    pub evals: Vec<F>,
    pub stark_info: StarkInfo,
}

impl<F: Field> AirInstance<F> {
    pub fn new(setup_ctx: Arc<SetupCtx>, trace_info: TraceInfo<F>) -> Self {
        let airgroup_id = trace_info.airgroup_id;
        let air_id = trace_info.air_id;

        let ps = setup_ctx.get_setup(airgroup_id, air_id);

        let (custom_commits, custom_commits_extended) = Self::init_custom_commits(ps, trace_info.custom_traces);

        let airvalues = if let Some(air_values) = trace_info.air_values {
            air_values
        } else {
            vec![F::zero(); ps.stark_info.airvalues_map.as_ref().unwrap().len() * 3]
        };

        AirInstance {
            airgroup_id,
            air_id,
            air_instance_id: None,
            idx: None,
            global_idx: None,
            trace: trace_info.trace,
            aux_trace: None,
            custom_commits,
            custom_commits_extended,
            airgroup_values: vec![F::zero(); ps.stark_info.airgroupvalues_map.as_ref().unwrap().len() * 3],
            airvalues,
            evals: vec![F::zero(); ps.stark_info.ev_map.len() * 3],
            stark_info: ps.stark_info.clone(),
        }
    }

    pub fn new_from_trace(setup_ctx: Arc<SetupCtx>, mut traces: FromTrace<'_, F>) -> Self {
        let mut trace_info =
            TraceInfo::new(traces.trace.airgroup_id(), traces.trace.air_id(), traces.trace.get_buffer());

        if let Some(custom_traces) = traces.custom_traces.as_mut() {
            let mut traces = Vec::new();
            for custom_trace in custom_traces.iter_mut() {
                traces.push(custom_trace.get_buffer());
            }
            trace_info = trace_info.with_custom_traces(traces);
        }

        if let Some(air_values) = traces.air_values.as_mut() {
            trace_info = trace_info.with_air_values(air_values.get_buffer());
        }

        AirInstance::new(setup_ctx, trace_info)
    }

    pub fn init_custom_commits(
        setup: &Setup,
        traces_custom: Option<Vec<Vec<F>>>,
    ) -> (Vec<CustomCommitsInfo<F>>, Vec<CustomCommitsInfo<F>>) {
        let n_custom_commits = setup.stark_info.custom_commits.len();

        let mut custom_commits = Vec::new();
        let mut custom_commits_extended = Vec::new();

        for commit_id in 0..n_custom_commits {
            let n_cols = *setup
                .stark_info
                .map_sections_n
                .get(&(setup.stark_info.custom_commits[commit_id].name.clone() + "0"))
                .unwrap() as usize;
            if let Some(traces_custom_vals) = traces_custom.as_ref() {
                custom_commits.push(CustomCommitsInfo::new(traces_custom_vals[commit_id].clone(), PathBuf::new()));
            } else {
                println!("No custom trace data found.");
            }

            custom_commits_extended.push(CustomCommitsInfo::new(
                create_buffer_fast((1 << setup.stark_info.stark_struct.n_bits_ext) * n_cols),
                PathBuf::new(),
            ));
        }

        (custom_commits, custom_commits_extended)
    }

    pub fn get_trace_ptr(&self) -> *mut u8 {
        self.trace.as_ptr() as *mut u8
    }

    pub fn get_evals_ptr(&self) -> *mut u8 {
        self.evals.as_ptr() as *mut u8
    }

    pub fn get_airgroup_values_ptr(&self) -> *mut u8 {
        self.airgroup_values.as_ptr() as *mut u8
    }

    pub fn get_airvalues_ptr(&self) -> *mut u8 {
        self.airvalues.as_ptr() as *mut u8
    }

    pub fn set_aux_trace(&mut self, aux_trace: Vec<F>) {
        self.aux_trace = Some(aux_trace);
    }

    pub fn get_aux_trace_ptr(&self) -> *mut u8 {
        match &self.aux_trace {
            Some(aux_trace) => aux_trace.as_ptr() as *mut u8,
            None => std::ptr::null_mut(), // Return null if `trace` is `None`
        }
    }

    pub fn get_custom_commits_ptr(&self) -> [*mut u8; 10] {
        let mut ptrs = [std::ptr::null_mut(); 10];
        for (i, custom_commit) in self.custom_commits.iter().enumerate() {
            ptrs[i] = custom_commit.buffer.as_ptr() as *mut u8;
        }
        ptrs
    }

    pub fn get_custom_commits_extended_ptr(&self) -> [*mut u8; 10] {
        let mut ptrs = [std::ptr::null_mut(); 10];
        for (i, custom_commit) in self.custom_commits_extended.iter().enumerate() {
            ptrs[i] = custom_commit.buffer.as_ptr() as *mut u8;
        }
        ptrs
    }

    pub fn set_custom_commit_cached_file(&mut self, commit_id: u64, cached_file: PathBuf) {
        self.custom_commits[commit_id as usize].cached_file = cached_file;
    }

    pub fn set_custom_commit_id_buffer(&mut self, buffer: Vec<F>, commit_id: u64) {
        self.custom_commits[commit_id as usize].buffer = buffer;
    }

    pub fn set_airvalue(&mut self, name: &str, lengths: Option<Vec<u64>>, value: F) {
        let airvalues_map = self.stark_info.airvalues_map.as_ref().unwrap();
        let mut id = 0;
        let mut found = false;
        for air_value in airvalues_map {
            // Check if name matches
            let name_matches = air_value.name == name;

            
            // If lengths is provided, check that it matches airvalue.lengths
            let lengths_match = if let Some(ref provided_lengths) = lengths {
                Some(&air_value.lengths) == Some(provided_lengths)
            } else {
                true // If lengths is None, skip the lengths check
            };

            if !name_matches || !lengths_match {
                if air_value.stage == 1 {
                    id += 1;
                } else {
                    id += 3;
                }
            } else {
                found = true;
                break;
            }
        }

        if !found {
            panic!("Name {} with specified lengths {:?} not found in airvalues", name, lengths);
        }

        self.airvalues[id] = value;
    }

    pub fn set_airvalue_ext(&mut self, name: &str, lengths: Option<Vec<u64>>, value: Vec<F>) {
        let airvalues_map = self.stark_info.airvalues_map.as_ref().unwrap();
        let mut id = 0;
        let mut found = false;
        for air_value in airvalues_map {
            // Check if name matches
            let name_matches = air_value.name == name;

            
            // If lengths is provided, check that it matches airvalue.lengths
            let lengths_match = if let Some(ref provided_lengths) = lengths {
                Some(&air_value.lengths) == Some(provided_lengths)
            } else {
                true // If lengths is None, skip the lengths check
            };

            if !name_matches || !lengths_match {
                if air_value.stage == 1 {
                    id += 1;
                } else {
                    id += 3;
                }
            } else {
                found = true;
                break;
            }
        }

        if !found {
            panic!("Name {} with specified lengths {:?} not found in airvalues", name, lengths);
        }

        assert!(value.len() == 3, "Value vector must have exactly 3 elements");

        let mut value_iter = value.into_iter();

        self.airvalues[id] = value_iter.next().unwrap();
        self.airvalues[id + 1] = value_iter.next().unwrap();
        self.airvalues[id + 2] = value_iter.next().unwrap();
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
    }

    pub fn set_air_instance_id(&mut self, air_instance_id: usize, idx: usize) {
        self.air_instance_id = Some(air_instance_id);
        self.idx = Some(idx);
    }
}
