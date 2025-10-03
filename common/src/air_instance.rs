use std::ptr;
use fields::Field;
use proofman_util::create_buffer_fast;

use crate::{
    trace::{Trace, Values},
    PackedInfo,
};

#[repr(C)]
pub struct StepsParams {
    pub trace: *mut u8,
    pub aux_trace: *mut u8,
    pub public_inputs: *mut u8,
    pub proof_values: *mut u8,
    pub challenges: *mut u8,
    pub airgroup_values: *mut u8,
    pub airvalues: *mut u8,
    pub evals: *mut u8,
    pub xdivxsub: *mut u8,
    pub p_const_pols: *mut u8,
    pub p_const_tree: *mut u8,
    pub custom_commits_fixed: *mut u8,
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
            proof_values: ptr::null_mut(),
            challenges: ptr::null_mut(),
            airgroup_values: ptr::null_mut(),
            airvalues: ptr::null_mut(),
            evals: ptr::null_mut(),
            xdivxsub: ptr::null_mut(),
            p_const_pols: ptr::null_mut(),
            p_const_tree: ptr::null_mut(),
            custom_commits_fixed: ptr::null_mut(),
        }
    }
}

pub struct CustomCommitInfo<F> {
    pub trace: Vec<F>,
    pub commit_id: usize,
}

pub struct TraceInfo<F> {
    airgroup_id: usize,
    air_id: usize,
    num_rows: usize,
    trace: Vec<F>,
    custom_traces: Option<Vec<CustomCommitInfo<F>>>,
    air_values: Option<Vec<F>>,
    airgroup_values: Option<Vec<F>>,
    shared_buffer: bool,
    is_packed: bool,
    num_packed_words: u64,
    unpack_info: Vec<u64>,
}

impl<F> TraceInfo<F> {
    pub fn new(airgroup_id: usize, air_id: usize, num_rows: usize, trace: Vec<F>, shared_buffer: bool) -> Self {
        Self {
            airgroup_id,
            air_id,
            num_rows,
            trace,
            custom_traces: None,
            air_values: None,
            airgroup_values: None,
            shared_buffer,
            is_packed: false,
            num_packed_words: 0,
            unpack_info: vec![],
        }
    }

    pub fn with_packed_info(mut self, is_packed: bool, num_packed_words: u64, unpack_info: Vec<u64>) -> Self {
        self.is_packed = is_packed;
        self.num_packed_words = num_packed_words;
        self.unpack_info = unpack_info;
        self
    }

    pub fn with_custom_traces(mut self, custom_traces: Vec<CustomCommitInfo<F>>) -> Self {
        self.custom_traces = Some(custom_traces);
        self
    }

    pub fn with_air_values(mut self, air_values: Vec<F>) -> Self {
        self.air_values = Some(air_values);
        self
    }

    pub fn with_airgroup_values(mut self, airgroup_values: Vec<F>) -> Self {
        self.air_values = Some(airgroup_values);
        self
    }
}

pub struct FromTrace<'a, F> {
    pub trace: &'a mut dyn Trace<F>,
    pub custom_traces: Option<Vec<&'a mut dyn Trace<F>>>,
    pub air_values: Option<&'a mut dyn Values<F>>,
    pub airgroup_values: Option<&'a mut dyn Values<F>>,
}

impl<'a, F> FromTrace<'a, F> {
    pub fn new(trace: &'a mut dyn Trace<F>) -> Self {
        Self { trace, custom_traces: None, air_values: None, airgroup_values: None }
    }

    pub fn with_custom_traces(mut self, custom_traces: Vec<&'a mut dyn Trace<F>>) -> Self {
        self.custom_traces = Some(custom_traces);
        self
    }

    pub fn with_air_values(mut self, air_values: &'a mut dyn Values<F>) -> Self {
        self.air_values = Some(air_values);
        self
    }

    pub fn with_airgroup_values(mut self, airgroup_values: &'a mut dyn Values<F>) -> Self {
        self.airgroup_values = Some(airgroup_values);
        self
    }
}

/// Air instance context for managing air instances (traces)
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Default)]
pub struct AirInstance<F> {
    pub airgroup_id: usize,
    pub air_id: usize,
    pub num_rows: usize,
    pub trace: Vec<F>,
    pub aux_trace: Vec<F>,
    pub custom_commits_fixed: Vec<F>,
    pub airgroup_values: Vec<F>,
    pub airvalues: Vec<F>,
    pub challenges: Vec<F>,
    pub evals: Vec<F>,
    pub fixed: Vec<F>,
    pub shared_buffer: bool,
    pub packed_info: PackedInfo,
}

impl<F: Field> AirInstance<F> {
    pub fn new(trace_info: TraceInfo<F>) -> Self {
        let airgroup_id = trace_info.airgroup_id;
        let air_id = trace_info.air_id;
        let num_rows = trace_info.num_rows;

        let airvalues = trace_info.air_values.unwrap_or_default();

        let airgroup_values = trace_info.airgroup_values.unwrap_or_default();

        AirInstance {
            airgroup_id,
            air_id,
            num_rows,
            trace: trace_info.trace,
            aux_trace: Vec::new(),
            custom_commits_fixed: Vec::new(),
            airgroup_values,
            airvalues,
            evals: Vec::new(),
            challenges: Vec::new(),
            shared_buffer: trace_info.shared_buffer,
            fixed: Vec::new(),
            packed_info: PackedInfo::new(trace_info.is_packed, trace_info.num_packed_words, trace_info.unpack_info),
        }
    }

    pub fn new_from_trace(mut traces: FromTrace<'_, F>) -> Self {
        let mut trace_info = TraceInfo::new(
            traces.trace.airgroup_id(),
            traces.trace.air_id(),
            traces.trace.num_rows(),
            traces.trace.get_buffer(),
            traces.trace.is_shared_buffer(),
        );

        if let Some(custom_traces) = traces.custom_traces.as_mut() {
            let mut traces = Vec::new();
            for custom_trace in custom_traces.iter_mut() {
                traces.push(CustomCommitInfo {
                    trace: custom_trace.get_buffer(),
                    commit_id: custom_trace.commit_id().unwrap(),
                });
            }
            trace_info = trace_info.with_custom_traces(traces);
        }

        if let Some(air_values) = traces.air_values.as_mut() {
            trace_info = trace_info.with_air_values(air_values.get_buffer());
        }

        AirInstance::new(trace_info)
    }

    pub fn get_trace(&self) -> Vec<F> {
        self.trace.clone()
    }

    pub fn get_trace_stage(&self, stage: usize) -> Vec<F> {
        if stage < 2 {
            panic!("Stage must be 2 or higher");
        }

        Vec::new()
    }

    pub fn get_trace_ptr(&self) -> *mut u8 {
        self.trace.as_ptr() as *mut u8
    }

    pub fn get_evals_ptr(&self) -> *mut u8 {
        self.evals.as_ptr() as *mut u8
    }

    pub fn get_challenges_ptr(&self) -> *mut u8 {
        self.challenges.as_ptr() as *mut u8
    }

    pub fn set_challenge(&mut self, index: usize, challenge: Vec<F>) {
        self.challenges[index] = challenge[0];
        self.challenges[index + 1] = challenge[1];
        self.challenges[index + 2] = challenge[2];
    }

    pub fn get_airgroup_values_ptr(&self) -> *mut u8 {
        self.airgroup_values.as_ptr() as *mut u8
    }

    pub fn get_air_values(&self) -> Vec<F> {
        self.airvalues.clone()
    }

    pub fn get_airgroup_values(&self) -> Vec<F> {
        self.airgroup_values.clone()
    }

    pub fn get_airvalues_ptr(&self) -> *mut u8 {
        self.airvalues.as_ptr() as *mut u8
    }

    pub fn init_evals(&mut self, size: usize) {
        self.evals = vec![F::ZERO; size];
    }

    pub fn init_challenges(&mut self, size: usize) {
        self.challenges = vec![F::ZERO; size];
    }

    pub fn init_aux_trace(&mut self, size: usize) {
        self.aux_trace = create_buffer_fast(size);
    }

    pub fn init_airvalues(&mut self, size: usize) {
        self.airvalues = vec![F::ZERO; size];
    }

    pub fn init_fixed(&mut self, fixed: Vec<F>) {
        self.fixed = fixed;
    }

    pub fn init_airgroup_values(&mut self, size: usize) {
        self.airgroup_values = vec![F::ZERO; size];
    }

    pub fn init_custom_commit_fixed_trace(&mut self, size: usize) {
        self.custom_commits_fixed = create_buffer_fast(size);
    }

    pub fn get_aux_trace_ptr(&self) -> *mut u8 {
        match &self.aux_trace.is_empty() {
            false => self.aux_trace.as_ptr() as *mut u8,
            true => std::ptr::null_mut(), // Return null if `trace` is `None`
        }
    }

    pub fn get_fixed_ptr(&self) -> *mut u8 {
        self.fixed.as_ptr() as *mut u8
    }

    pub fn get_custom_commits_fixed_ptr(&self) -> *mut u8 {
        self.custom_commits_fixed.as_ptr() as *mut u8
    }

    pub fn clear_traces(&mut self) -> (bool, Vec<F>) {
        let trace = std::mem::take(&mut self.trace);
        let shared_buffer = self.shared_buffer && !trace.is_empty();
        self.custom_commits_fixed = Vec::new();
        self.aux_trace = Vec::new();
        self.fixed = Vec::new();
        (shared_buffer, trace)
    }

    pub fn clear_custom_commits_fixed_trace(&mut self) {
        self.custom_commits_fixed = Vec::new();
    }

    pub fn reset(&mut self) -> (bool, Vec<F>) {
        self.airgroup_id = 0;
        self.air_id = 0;
        let trace = std::mem::take(&mut self.trace);
        let shared_buffer = self.shared_buffer && !trace.is_empty();
        self.shared_buffer = false;
        self.aux_trace = Vec::new();
        self.custom_commits_fixed = Vec::new();
        self.airgroup_values = Vec::new();
        self.airvalues = Vec::new();
        self.evals = Vec::new();
        self.challenges = Vec::new();
        self.fixed = Vec::new();
        (shared_buffer, trace)
    }

    pub fn get_packed_info(&self) -> PackedInfo {
        self.packed_info.clone()
    }
}
